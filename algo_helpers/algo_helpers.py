import os
import re
import json

import numpy as np
import matplotlib.pyplot as plt

from together import Together

import pandas as pd
from scipy.stats import gaussian_kde, ttest_1samp
from scipy.signal import argrelextrema

from llm.llm_client import TogetherClient
from utils.logger_config import setup_logger

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


class ResponseEvaluationTensor:
    def __init__(self):
        self.together_models = [
            # LLMModel(model_handle="mistralai/Mixtral-8x7B-v0.1", MMLU_score=0.6859),
            LLMModel(model_handle="Qwen/Qwen2-72B-Instruct", MMLU_score=0.842),
            LLMModel(model_handle="meta-llama/Llama-3-8b-chat-hf", MMLU_score=0.684),
            LLMModel(model_handle="google/gemma-2b-it", MMLU_score=0.423),
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

    def generate_prompt(self, model_handle: str, _num_attempts=0):

        if _num_attempts > 4:
            return None

        response = TogetherClient(model=model_handle, api_key=os.environ["TOGETHER_API_KEY"]).get_completion(
            system="""
            Generate a prompt as if a user would. Place the prompt in JSON format
            ```json{"prompt": "_____"}```
            """,
            message=""
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

    def rate_response(self, model_handle: str, model_prompt: str, model_output: str, _num_attempts=0):

        if _num_attempts > 4:
            return None

        user_message = """You are a content grader who will output a rating between 1 to 5 indicating how well the provided content follows the user query.
        Output a 5 if the content fully followed the user query, and output a 1 if it doesn't follow it at all."""

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
        else:
            return self.rate_response(
                model_handle,
                model_prompt,
                model_output,
                _num_attempts=_num_attempts + 1
            )

    def compute_response_evaluation_tensor(self, num_trials=5):
        models = self.together_models

        ratings_array = np.zeros(shape=(len(models), len(models), num_trials), dtype=np.float64)

        for row_idx, auditing_model in enumerate(models):
            for col_idx, model_under_test in enumerate(models):
                for trial in range(num_trials):
                    p = self.generate_prompt(model_handle=auditing_model.model_handle)
                    if p is None:
                        logger.warning(f"Unable to generate prompt using model handle {auditing_model.model_handle}")
                        ratings_array[row_idx, col_idx] = np.nan
                        continue
                    logger.info(f"generated prompt: {p}")

                    p_optim = self.optimize_prompt(auditing_model.model_handle, p)
                    if p_optim is None:
                        logger.warning(f"Unable to optimize prompt using model handle {auditing_model.model_handle}")
                        ratings_array[row_idx, col_idx] = np.nan
                        continue
                    logger.info(f"optimized prompt: {p_optim}")

                    # DUT
                    response = TogetherClient(
                        api_key=os.environ["TOGETHER_API_KEY"], model=model_under_test.model_handle).get_completion(
                        system="",
                        message=p)

                    rating = self.rate_response(model_handle=auditing_model.model_handle,
                                                model_prompt=p,
                                                model_output=response)

                    ratings_array[row_idx, col_idx, trial] = rating

                    logger.info(
                        f"Auditor: {models[row_idx].model_handle}, "
                        f"Model Under Test: {models[col_idx].model_handle}, "
                        f"Trial: {trial}, "
                        f"Rating: {rating}"
                    )

        return ratings_array, self.together_models

    def run_eval(self):
        pass

    @staticmethod
    def test_self_preference_bias(tensor):
        """
        Perform a statistical test for self-preference bias on a 3D tensor.
        
        Parameters:
        tensor (numpy.ndarray): A 3D numpy array with shape [model, model, trials]
        
        Returns:
        dict: A dictionary containing the test results
        """
        # Ensure the tensor is a numpy array
        tensor = np.array(tensor)
        
        # Get the dimensions
        num_models, _, num_trials = tensor.shape
        
        # Calculate self-preference scores
        self_preference_scores = []
        for i in range(num_models):
            self_score = np.mean(tensor[i, i, :])
            others_score = np.mean(tensor[i, :, :][tensor[i, :, :] != tensor[i, i, :]])
            self_preference_scores.append(self_score - others_score)
        
        # Perform one-sample t-test
        t_statistic, p_value = ttest_1samp(self_preference_scores, 0)
        
        # Calculate effect size (Cohen's d)
        effect_size = np.mean(self_preference_scores) / np.std(self_preference_scores, ddof=1)
        
        # Prepare results
        test_results = {
            'mean_self_preference_score': np.mean(self_preference_scores),
            't_statistic': t_statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'self_preference_scores': self_preference_scores
        }
        
        return test_results

    @staticmethod
    def permutation_test_model_preference(tensor, n_permutations=10000):
        """
        Perform a permutation test to check if models prefer higher-rated models than themselves.
        
        Parameters:
        tensor (numpy.ndarray): A 3D numpy array with shape [model, model, trials]
                                Models are sorted in descending order of overall rating.
        n_permutations (int): Number of permutations for the test.
        
        Returns:
        dict: A dictionary containing the test results
        """
        num_models, _, num_trials = tensor.shape
        
        def compute_preference_statistic(data):
            preference_sum = 0
            count = 0
            for i in range(num_models):
                for j in range(i+1, num_models):  # Only compare with lower-rated models
                    preference_sum += np.mean(data[i, j] - data[j, i])
                    count += 1
            return preference_sum / count if count > 0 else 0
        
        # Compute observed statistic
        observed_statistic = compute_preference_statistic(tensor)
        
        # Perform permutations
        permuted_statistics = []
        for _ in range(n_permutations):
            permuted_tensor = tensor.copy()
            for i in range(num_models):
                for j in range(i+1, num_models):
                    if np.random.rand() < 0.5:
                        permuted_tensor[i, j], permuted_tensor[j, i] = permuted_tensor[j, i], permuted_tensor[i, j]
            permuted_statistics.append(compute_preference_statistic(permuted_tensor))
        
        # Compute p-value
        p_value = np.mean([stat >= observed_statistic for stat in permuted_statistics])
        
        # Compute effect size (standardized mean difference)
        effect_size = observed_statistic / np.std(permuted_statistics)
        
        # Prepare results
        test_results = {
            'observed_statistic': observed_statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'n_permutations': n_permutations
        }
        
        return test_results
    


if __name__ == "__main__":
    evaluator = ResponseEvaluationTensor()
    ratings_array, models = evaluator.compute_response_evaluation_tensor(num_trials=5)

    print(ratings_array)
    print()