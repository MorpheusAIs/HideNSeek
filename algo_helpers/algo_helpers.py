import os
import re
from typing import List, Tuple
import json

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from together import Together

import pandas as pd
from scipy.stats import ttest_ind, ttest_ind, mannwhitneyu, t, sem
from dotenv import load_dotenv
import argparse

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.append(parent_dir)

from llm.llm_client import TogetherClient
from utils.logger_config import setup_logger
from language_metric_helper import evaluate_similarity, convert_to_json_format

logger = setup_logger(__name__)


def extract_json(text):  # FIXME, duplicate from llm/llm_provider_ranking_experiment.py
    json_pattern = re.compile(r'```(.*?)```', re.DOTALL)
    match = json_pattern.search(text)
    if match:
        try:
            matched = match.group(1).strip()

            if matched.startswith("json"):
                matched = matched[4:].strip()

            matched = matched.replace('\n', '').replace('\r', '')
            return json.loads(matched)
        except json.JSONDecodeError as e:
            logger.error(f"{e}")
            return None
    return None

class LLMModel:
    def __init__(self, model_handle, MMLU_score):
        self.model_handle = model_handle
        self.MMLU_score = MMLU_score


class EvaluationConfig:
    def __init__(self, config):

        essential_fields = ['num_trials', 'rewrite_prompt', 'save_response']
        self.num_trials = config.get(essential_fields[0], 0)
        self.rewrite_prompt = config.get(essential_fields[1], False)
        self.save_response = config.get(essential_fields[2], False)
        # Store additional attributes
        self.additional_attributes = {k: v for k, v in config.items() 
                                      if k not in essential_fields}

    def __getattr__(self, item):
        if item in self.additional_attributes:
            return self.additional_attributes[item]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")


class ResponseEvaluationTensor:
    def __init__(self):
        self.together_models = [
            # LLMModel(model_handle="mistralai/Mixtral-8x7B-v0.1", MMLU_score=0.6859),
            LLMModel(model_handle="Qwen/Qwen2-72B-Instruct", MMLU_score=0.842),
            # LLMModel(model_handle="mistralai/Mixtral-8x22B", MMLU_score=0.7781),
            LLMModel(model_handle="meta-llama/Llama-3-70b-chat-hf", MMLU_score=0.795),
            LLMModel(model_handle="meta-llama/Llama-3-8b-chat-hf", MMLU_score=0.684),
            LLMModel(model_handle="google/gemma-7b-it", MMLU_score=0.643),
            #LLMModel(model_handle="google/gemma-2b-it", MMLU_score=0.423),
            LLMModel(model_handle='mistralai/Mistral-7B-Instruct-v0.2', MMLU_score=0.73),
            LLMModel(model_handle='mistralai/Mistral-7B-Instruct-v0.3', MMLU_score=0.73)            
        ]
        self.together_models.sort(key=lambda x: x.MMLU_score, reverse=True)  # sort these by MMLU score

    @staticmethod
    def _extract_prompt(text):
        extracted_json = extract_json(text)

        if extracted_json is not None:
            prompt_value = extracted_json.get('prompt')
            if prompt_value:
                return prompt_value
            else:
                logger.warning("No 'prompt' key found in the extracted JSON.")
        else:
            logger.warning("No valid JSON found in the text.")

        return None

    def generate_prompt(self, model_handle: str, past_prompts=[], _num_attempts=0):

        if _num_attempts > 4:
            return None
        
        message_str = ""
        if past_prompts:
            message_str = f"here are your past prompts: {past_prompts}"

        response = TogetherClient(model=model_handle, api_key=os.environ["TOGETHER_API_KEY"]).get_completion(
            system="""
            Generate a new prompt about any topics that users are interested in. Place the prompt in JSON format
            ```json{"prompt": "_____"}```
            """,
            message=message_str
        )

        prompt = self._extract_prompt(response)

        if prompt:
            return prompt
        else:
            return self.generate_prompt(model_handle, _num_attempts=_num_attempts+1)

    def optimize_prompt(self, model_handle: str, unoptimized_prompt: str, _num_attempts=0):

        if _num_attempts > 4:
            return None

        response = TogetherClient(model=model_handle, api_key=os.environ["TOGETHER_API_KEY"]).get_completion(
            system='Given a prompt, optimize it so that you, when asked, would produce the best possible answer:',
            message='BEGINNING OF PROMPT\n'
                    f'{unoptimized_prompt}\n'
                    'END OF PROMPT\n'
                    'Place your optimized prompt in JSON format '
                    '```json{"prompt": "_____"}```'
        )

        prompt = self._extract_prompt(response)
        if prompt:
            return prompt
        else:
            return self.generate_prompt(model_handle, _num_attempts=_num_attempts+1)

    @staticmethod
    def _extract_rating(text):
        json_str = extract_json(text)
        if json_str:
            rating_value = json_str.get('rating')
            if rating_value:
                rating_value = int(rating_value)
                return int(rating_value)
            else:
                return None
        else:
            # attempt to extract `rating: #` out of the text
            pattern = re.compile(r'(?:"rating"|rating)\s*:\s*(?:"?(\d+)"?)(?:\s*,|\s*\n|\s*$)', re.IGNORECASE)

            match = pattern.search(text)
            if match:
                return int(match.group(1))
            else:
                logger.warning("NO JSON string found")
                return None

    def rate_response(self, model_handle: str, model_prompt: str, model_output: str, retry_count=5):

        _num_attempts = 0

        while _num_attempts < retry_count:
            user_message = """You are a content grader who will output a rating between 1 to 5 indicating how well the provided content follows the user query."""

            user_message += """Output the following JSON dictionary, make sure you include ```json {...}``` around the JSON string: 
            ```json
            {
                "rationale" : describe why you chose the rating,
                "rating" : Integer representing the rating between 1 to 5
            }
            ```
            """
            user_message += (f"given the following user query:\n{model_prompt}. "
                            f"provide a rating for this Context:\n{model_output}")

            response = TogetherClient(model=model_handle, api_key=os.environ["TOGETHER_API_KEY"]).get_completion(
                system=user_message, message="")

            print(f"Response - {response}")
            # extract rating
            rating_value = self._extract_rating(response)

            # ensure it's an int between 1 and 5 (incl.)
            if isinstance(rating_value, int) and 1 <= rating_value <= 5:
                return rating_value
            _num_attempts=_num_attempts + 1

        return None

    def compute_response_evaluation_tensor(self, config: EvaluationConfig):
        models = self.together_models

        ratings_array = np.zeros(shape=(len(models), len(models), config.num_trials), dtype=np.float64)

        if config.save_response:
            response_array = np.empty((len(models), len(models), config.num_trials), dtype=object)

        for row_idx, auditing_model in enumerate(models):
            for col_idx, model_under_test in enumerate(models):
                past_prompts = []
                for trial in range(config.num_trials):
                    p = self.generate_prompt(model_handle=auditing_model.model_handle, past_prompts=past_prompts)
                    if p is None:
                        logger.warning(f"Unable to generate prompt using model handle {auditing_model.model_handle}")
                        ratings_array[row_idx, col_idx] = np.nan
                        continue
                    past_prompts.append(p)
                    logger.info(f"generated prompt: {p}")

                    if config.rewrite_prompt:
                        p_optim = self.optimize_prompt(auditing_model.model_handle, p)
                        if p_optim is None:
                            logger.warning(f"Unable to optimize prompt using model handle {auditing_model.model_handle}")
                            ratings_array[row_idx, col_idx] = 0
                            continue
                        logger.info(f"optimized prompt: {p_optim}")
                    else:
                        p_optim = p

                    # DUT
                    response = TogetherClient(
                        api_key=os.environ["TOGETHER_API_KEY"], model=model_under_test.model_handle).get_completion(
                        system="",
                        message=p_optim)

                    rating = self.rate_response(model_handle=auditing_model.model_handle,
                                                model_prompt=p_optim,
                                                model_output=response)

                    ratings_array[row_idx, col_idx, trial] = rating
                    if config.save_response:
                        response_array[row_idx, col_idx, trial] = response

                    logger.info(
                        f"Auditor: {models[row_idx].model_handle}, "
                        f"Model Under Test: {models[col_idx].model_handle}, "
                        f"Trial: {trial}, "
                        f"Rating: {rating}"
                    )

        output_obj = {
            'ratings': ratings_array,
            'models': self.together_models,            
        }
        
        if config.save_response:
            output_obj['responses'] = response_array

        return output_obj

    def run_eval(self):
        pass

    def compare_model_scores(self,tensor, ref_idx, test_idx, alpha=0.05):
        def cohen_d(x, y):
                return (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x) ** 2 + np.std(y) ** 2) / 2)
        
        scores_given_ref = tensor[ref_idx, :, :].flatten()
        scores_given_test = tensor[test_idx, :, :].flatten()
        scores_received_ref = tensor[:, ref_idx, :].flatten()
        scores_received_test = tensor[:, test_idx, :].flatten()
        
        mean_given_ref, std_given_ref = np.mean(scores_given_ref), np.std(scores_given_ref)
        mean_given_test, std_given_test = np.mean(scores_given_test), np.std(scores_given_test)
        mean_received_ref, std_received_ref = np.mean(scores_received_ref), np.std(scores_received_ref)
        mean_received_test, std_received_test = np.mean(scores_received_test), np.std(scores_received_test)
        
        # If the variance is zero, skip statistical tests and manually set similarity results
        if std_given_ref == 0 and std_given_test == 0 and std_received_ref == 0 and std_received_test == 0:
            identical_given = np.array_equal(scores_given_ref, scores_given_test)
            identical_received = np.array_equal(scores_received_ref, scores_received_test)
            
            reject_null_given = not identical_given
            reject_null_received = not identical_received
            p_value_given = 1.0 if identical_given else 0.0
            p_value_received = 1.0 if identical_received else 0.0
            
            conf_int_given_ref = (mean_given_ref, mean_given_ref)
            conf_int_given_test = (mean_given_test, mean_given_test)
            conf_int_received_ref = (mean_received_ref, mean_received_ref)
            conf_int_received_test = (mean_received_test, mean_received_test)
            
            effect_size_given = 0.0
            effect_size_received = 0.0
        else:
            t_test_given = ttest_ind(scores_given_ref, scores_given_test)
            t_test_received = ttest_ind(scores_received_ref, scores_received_test)
            
            u_test_given = mannwhitneyu(scores_given_ref, scores_given_test)
            u_test_received = mannwhitneyu(scores_received_ref, scores_received_test)
            
            conf_int_given_ref = t.interval(0.95, len(scores_given_ref)-1, loc=mean_given_ref, scale=sem(scores_given_ref))
            conf_int_given_test = t.interval(0.95, len(scores_given_test)-1, loc=mean_given_test, scale=sem(scores_given_test))
            conf_int_received_ref = t.interval(0.95, len(scores_received_ref)-1, loc=mean_received_ref, scale=sem(scores_received_ref))
            conf_int_received_test = t.interval(0.95, len(scores_received_test)-1, loc=mean_received_test, scale=sem(scores_received_test))
            
            effect_size_given = cohen_d(scores_given_ref, scores_given_test)
            effect_size_received = cohen_d(scores_received_ref, scores_received_test)
            
            reject_null_given = t_test_given.pvalue <= alpha
            reject_null_received = t_test_received.pvalue <= alpha
            p_value_given = t_test_given.pvalue
            p_value_received = t_test_received.pvalue

        is_similar = not reject_null_given and not reject_null_received

        results = {
            "mean_given_ref": mean_given_ref,
            "std_given_ref": std_given_ref,
            "mean_given_test": mean_given_test,
            "std_given_test": std_given_test,
            "mean_received_ref": mean_received_ref,
            "std_received_ref": std_received_ref,
            "mean_received_test": mean_received_test,
            "std_received_test": std_received_test,
            "t_test_given": t_test_given if 't_test_given' in locals() else None,
            "t_test_received": t_test_received if 't_test_received' in locals() else None,
            "u_test_given": u_test_given if 'u_test_given' in locals() else None,
            "u_test_received": u_test_received if 'u_test_received' in locals() else None,
            "conf_int_given_ref": conf_int_given_ref,
            "conf_int_given_test": conf_int_given_test,
            "conf_int_received_ref": conf_int_received_ref,
            "conf_int_received_test": conf_int_received_test,
            "effect_size_given": effect_size_given,
            "effect_size_received": effect_size_received,
            "reject_null_given": reject_null_given,
            "reject_null_received": reject_null_received,
            "p_value_given": p_value_given,
            "p_value_received": p_value_received,
            "is_similar": is_similar
        }
        
        return results
    

    def group_models(self, confusion_matrix: np.ndarray, models: List[LLMModel]) -> List[List[Tuple[str, float]]]:
        if confusion_matrix.shape[0] != len(models):
            raise ValueError("Number of models doesn't match confusion matrix dimensions")

        n = len(models)
        visited = [False] * n
        groups = []

        for i in range(n):
            if not visited[i]:
                group = []
                self._dfs(i, confusion_matrix, visited, group)
                groups.append([(models[j].model_handle, models[j].MMLU_score) for j in group])

        return groups

    def _dfs(self, i: int, matrix: np.ndarray, visited: List[bool], group: List[int]):
        visited[i] = True
        group.append(i)
        for j in range(len(visited)):
            if matrix[i][j] == 1 and not visited[j]:
                self._dfs(j, matrix, visited, group)

    def visualize_groups(self, groups: List[List[Tuple[str, float]]], output_file: str = 'model_groups.png'):
        G = nx.Graph()
        colors = []
        labels = {}

        for i, group in enumerate(groups):
            group_color = plt.cm.Set3(i / len(groups))
            for model, score in group:
                G.add_node(model)
                colors.append(group_color)
                labels[model] = f"{model.split('/')[-1]}\n(MMLU: {score:.3f})"
            
            # Connect all models within the group
            for model1, _ in group:
                for model2, _ in group:
                    if model1 != model2:
                        G.add_edge(model1, model2)

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        nx.draw(G, pos, node_color=colors, node_size=3000, alpha=0.8, with_labels=False)
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold")
        
        plt.title("LLM Model Groupings", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved as {output_file}")    

def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--num_trials', type=int, required=False, default=5, help="Number of trials to run")
    parser.add_argument('--task', type=str, choices=['relevance', 'lang_trend'], default='relevance', help='Which task to run. Relevance runs the prompt relevance subtask whereas lang_trend focuses on LLM language trends')
    parser.add_argument('--config_path', type=str, required=False, help="Path for loading model api config")

    # Task arguments
    parser.add_argument('--rewrite_prompt', action='store_true', help="Prevent prompt rewrite")
    parser.add_argument('--save_response', action='store_true', help="Save LLM Response")
    parser.add_argument('--vectorizer', choices = ['tf_idf', 'ngram'], default='tf_idf', help='Vectorization approach taken for language stat identification')    
    parser.add_argument('--lang_metric_approach', type=str, default='question_wise', help="Evaluation strategy for calculating language stats")
    parser.add_argument('--lang_metric_cosine', type=float, default=0.53)
    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    load_dotenv(args.config_path)

    evaluator = ResponseEvaluationTensor()
    evaluation_config = EvaluationConfig({
        "num_trials": args.num_trials,
        "rewrite_prompt": args.rewrite_prompt,
        "save_response": args.save_response
    })

    eval_output = evaluator.compute_response_evaluation_tensor(evaluation_config)
    ratings_array, models = eval_output['ratings'], eval_output['models']
    model_response = None
    if args.save_response:
        model_response = eval_output['responses']
        np.save(f'response_trials_{args.num_trials}.npy', model_response)

    if args.task == 'relevance':
        similarity = np.eye(len(models))

        print(similarity)
        for idx, model in enumerate(models):
                for j in range(idx+1, len(evaluator.together_models)):
                    print(f"comparing {model.model_handle} to {models[j].model_handle}")

                    is_sim = int(evaluator.compare_model_scores(ratings_array, idx, j)["is_similar"])
                    similarity[idx, j] = is_sim
                    similarity[j, idx] = is_sim
        
        print(similarity)

        groups = evaluator.group_models(similarity, models)

        if args.output_path:
            evaluator.visualize_groups(groups, output_file=output_path)
        else:
            evaluator.visualize_groups(groups)

    elif args.task == 'lang_trend':
        assert model_response is not None, "Model responses must be recorded to do language analysis"
        assert args.vectorizer is not None, "Must select a vectorizer option for lang stat"

        model_names = [model.model_handle for model in evaluator.together_models]
        eval_results = evaluate_similarity(response_array=model_response,
                                           model_names=model_names,
                                           cosine_threshold=args.lang_metric_cosine,
                                           approach=args.lang_metric_approach,
                                           vectorization_approach=args.vectorizer,
                                           debug=True)
        
        if args.output_path:
            json.dump(convert_to_json_format(eval_results), open(args.output_path, 'w'))
    else:
        raise ValueError(f"Unsupported Task: {args.task}")
