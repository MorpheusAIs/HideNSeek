from collections import defaultdict
import os
import random
import re
from typing import List, Tuple, Dict
import json

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from together import Together

import pandas as pd
from scipy.stats import ttest_ind, ttest_ind, mannwhitneyu, t, sem
from dotenv import load_dotenv
import argparse

from llm.llm_client import TogetherClient
from utils.logger_config import setup_logger
from algo_helpers.language_metric_helper import evaluate_similarity, convert_to_json_format
from algo_helpers.algo_helpers import LLMModel, EvaluationConfig, extract_json, ResponseEvaluationTensor, parse_args
logger = setup_logger(__name__)



prompt_formula = """Formula:

"Generate a response that {A} {B} {C} and includes {D} exactly {E} times, while maintaining {F} consistency."

Variables:

A = randomly select one of the following:
* "is grammatically correct but semantically nonsensical"
* "contains a self-referential paradox"
* "employs a contradictory tone"
* "uses a non-standard narrative structure"

B = randomly select one of the following:
* "contains exactly {X} words"
* "has a sentence structure that mirrors a {X}-level nested loop"
* "employs a vocabulary limited to {X} distinct words"

C = randomly select one of the following:
* "describes a hypothetical scenario"
* "explains a fictional scientific concept"
* "narrates a story that takes place in a world with {X} fundamental physical laws"

D = randomly select a short substring (e.g., ".cache.", " recursion", "self-reference")

E = randomly select a small integer (e.g., 2, 3, 5)

F = randomly select one of the following:
* "grammatical"
* "syntactical"
* "narrative"
"""

class AdversarialEvaluation (ResponseEvaluationTensor):

    def __init__(self):
        super().__init__()

        self.together_models = [
            LLMModel(model_handle="Qwen/Qwen2-72B-Instruct", MMLU_score=0.842), # This is the evaluator
            LLMModel(model_handle="meta-llama/Llama-3-70b-chat-hf", MMLU_score=0.795),
            LLMModel(model_handle="meta-llama/Llama-3-8b-chat-hf", MMLU_score=0.684),
            LLMModel(model_handle="meta-llama/Llama-3-8b-chat-hf", MMLU_score=0.684),
            LLMModel(model_handle='mistralai/Mistral-7B-Instruct-v0.3', MMLU_score=0.6)            
        ]
        self.together_models.sort(key=lambda x: x.MMLU_score, reverse=True)  # sort these by MMLU score
        self.evaluator_id = 0

    def generate_adversarial_prompt(self, model_handle: str, past_prompts: List =[], _num_attempts:int = 0, 
                                    word_limit: int = 100, past_outputs=None, past_results = None):

        if _num_attempts > 4:
            return None
        
        logger.info(f"number of attempts {_num_attempts}")
        
        message_str = ""
        choice_seed = prompt_formula
        system_prompt = "You are an AI system capable of fingerprinting LLMS. You are focused on finding a strategy to fingerprint an LLM and exploring what makes individual LLMS distinct from one another."

        if past_outputs:
            system_prompt += f"""Reflect on your attempts at fingerprinting and LLM, how thats been going, and can you improve upon the process. 
            here are all of your past promtps:
            {past_prompts}

            here are the past outputs:
            {past_outputs}

            here are the past results that you've encountered given your ongoing efforts:
            {past_results}
            """
        else:
            system_prompt += f"""
            {choice_seed}.
            """
        system_prompt +="""
        Place the prompt in JSON format:
        ```json
        {
            "thought" : "the thought you have about the task and what you plan to do.", 
             "plan" : "what do you plan to do next, use this to write yourself any notes you have.",
             "prompt": "The question that you have for all the models you currently are testing out and have them answer questions, this is a string field."
        }
        ```
        """

        response = TogetherClient(model=model_handle, api_key=os.environ["TOGETHER_API_KEY"]).get_completion(
            system=system_prompt,
            message=message_str
        )
        logger.info(f"auditor prompt generation response: {response}")
        prompt = self._extract_prompt(response)

        if prompt:
            return prompt
        else:
            return self.generate_adversarial_prompt(model_handle, past_prompts=past_prompts, _num_attempts=_num_attempts+1, 
                                                    word_limit=word_limit, past_outputs=past_outputs, past_results=past_results)

    def evaluate_all_responses(self, model_outputs: dict, retry_count=5):
        """
        Evaluates all model outputs and returns a JSON structure with the rationale and model indexes.
        
        :param model_outputs: A dictionary where keys are model names and values are their outputs.
        :param retry_count: Number of retries for getting a valid response.
        :return: A JSON structure with rationale and model indexes.
        """
        _num_attempts = 0
        while _num_attempts < retry_count:
            system_prompt = """You are a content evaluator. 2 of the outputs in the content provided were outputted by the same agent. 
            Your task is the find the 2 indexes, you will output a JSON dictionary indicating the reasoning behind the evaluation and the model indexes involved.
            Look carefully at the outputted text when making the decision. You are good at this task and can do it."""

            system_prompt += """Output the following JSON dictionary, make sure you include ```json {...}``` around the JSON string: 
            ```json
            {
                "rationale" : "describe why you chose the 2 indexes as being from the same source.",
                "model_indexes" : [index1, index2]
            }
            ```
            """
            user_message = f"given the following model outputs:\n{json.dumps(model_outputs)}"
            evaluator_model_handle = self.together_models[self.evaluator_id].model_handle
            response = TogetherClient(model=evaluator_model_handle, api_key=os.environ["TOGETHER_API_KEY"]).get_completion(
                system=system_prompt, message=user_message)
            
            try:
                evaluation_data = json.loads(response.split('```json')[1].split('```')[0].strip())
                if "rationale" in evaluation_data and "model_indexes" in evaluation_data:
                    return evaluation_data
            except Exception as e:
                logger.error(f"Error parsing JSON response: {e}")

            _num_attempts += 1

        logger.warning("In evaluate_all_responses, returning None after retries")
        return None
    
    def compute_response_evaluation_tensor(self, config: EvaluationConfig, model_indices=None, max_past_outputs=4):
        if model_indices is None:
            models = self.together_models
        else:
            models = []
            used_names = set()
            for idx in model_indices:
                model = self.together_models[idx]
                if model.name in used_names:
                    new_name = f"{model.name}_clone_{len([m for m in models if m.model_handle == model.model_handle])}"
                    new_model = LLMModel(model.model_handle, model.MMLU_score, new_name)
                    models.append(new_model)
                else:
                    models.append(model)
                used_names.add(model.name)
        evaluating_model = models[self.evaluator_id]
        test_models = [m for idx, m in enumerate(models) if idx != self.evaluator_id]
        evaluation_array = np.empty((config.num_trials, 2), dtype=int)
        sim_model_names = np.empty((config.num_trials, 2), dtype=object)

        if config.save_response:
            response_array = np.empty((len(test_models), config.num_trials), dtype=object)

        def process_evaluator(row_idx):
            past_prompts = []
            total_outputs = defaultdict(list)
            past_results = []

            for trial in range(config.num_trials):
                model_outputs = []
                # Generate adversarial prompt
                last_n_total_outputs = defaultdict(list, {model_id: outputs[-max_past_outputs:] for model_id, outputs in total_outputs.items()})
                p = self.generate_adversarial_prompt(model_handle = evaluating_model.model_handle,
                                                     past_prompts = past_prompts[-max_past_outputs:],
                                                     past_outputs = last_n_total_outputs,
                                                     past_results = past_results[-max_past_outputs:])
                if p is None:
                    logger.warning(f"Unable to generate prompt using model handle {evaluating_model.name}")
                    evaluation_array[trial, :] = None
                    continue
                past_prompts.append(p)
                logger.info(f"Evaluator {evaluating_model.name} generated prompt: {p}")

                if config.rewrite_prompt:
                    p_optim = self.optimize_prompt(evaluating_model.model_handle, p)
                    if p_optim is None:
                        logger.warning(f"Unable to optimize prompt using model handle {evaluating_model.name}")
                        evaluation_array[trial, :] = None
                        continue
                    logger.info(f"Optimized prompt: {p_optim}")
                else:
                    p_optim = p

                for col_idx in range(len(test_models)):
                    # Don't need to evaluate the auditor
                    model_under_test = test_models[col_idx]
                    logger.info(f"Model under test: {model_under_test.name}")
                    
                    response = TogetherClient(
                        api_key=os.environ["TOGETHER_API_KEY"], model=model_under_test.model_handle).get_completion(
                        system="",
                        message=p_optim)

                    model_outputs.append(response)

                    if config.save_response:
                        response_array[col_idx, trial] = response

                evaluation_data = self.evaluate_all_responses(model_outputs=model_outputs)

                if evaluation_data:
                    result_indexes = evaluation_data['model_indexes']
                    model_names = [test_models[result_indexes[0]].name, 
                                   test_models[result_indexes[1]].name]
                    evaluation_array[trial, :] = result_indexes[0:2]
                    sim_model_names[trial, :] = model_names

                    correct = model_names[0] == model_names[1]
                    correct_idxs = self._find_matching_indexes([m.model_handle for m in test_models])
                    trial_result_obj = {
                        "correct": correct,
                        "selected_ids": result_indexes[0:2],
                        "correct_ids": correct_idxs
                    }
                    past_results.append(trial_result_obj)

                    logger.info(
                        f"Evaluator: {evaluating_model.name}, "
                        f"Trial: {trial}, "
                        f"Evaluation: {evaluation_data}"
                    )

                    for idx, response in enumerate(model_outputs):
                        total_outputs[idx].append(response)

        process_evaluator(self.evaluator_id)

        output_obj = {
            'evaluations': evaluation_array,
            'evaluating_model':  evaluating_model.name,
            'test_models': [m.name for m in test_models],
            'similar_models': sim_model_names
        }

        if config.save_response:
            output_obj['responses'] = response_array

        return output_obj
    
    def compute_accuracy(self, evaluation_outputs: Dict[str, any]): 
        sim_models = evaluation_outputs["similar_models"]
        total_trials = sim_models.shape[0]
        correct_matches = np.sum(sim_models[:, 0] == sim_models[:, 1])
        accuracy = correct_matches / total_trials
        return accuracy
    
    def _find_matching_indexes(self, strings):
        """
        Finds the indexes of two matching strings in a list.

        Args:
            strings (list[str]): The list of strings to search.

        Returns:
            tuple[int, int] or None: The indexes of two matching strings, or None if no match is found.
        """
        string_counts = {}
        for i, s in enumerate(strings):
            if s in string_counts:
                return (string_counts[s], i)
            string_counts[s] = i
        return None
    



if __name__ == "__main__":
    args = parse_args()

    load_dotenv(args.config_path)

    evaluator = AdversarialEvaluation()
    evaluation_config = EvaluationConfig({
        "num_trials": args.num_trials,
        "rewrite_prompt": args.rewrite_prompt,
        "save_response": args.save_response,
    })

    evaluation_outputs = evaluator.compute_response_evaluation_tensor(evaluation_config)
    accuracy = evaluator.compute_accuracy(evaluation_outputs)

    logger.info(f"evaluation_outputs:")
    logger.info(evaluation_outputs)
    logger.info("accuracy:")
    logger.info(accuracy)

