from collections import defaultdict
import os
import random
import re
from typing import List, Tuple, Dict
import json
import yaml

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


def load_config(file_path='config.yaml'):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)   


class AdversarialEvaluation (ResponseEvaluationTensor):

    def __init__(self, models_config: str):
        super().__init__()

        config = load_config(models_config)
        self.auditor_model = LLMModel(config["auditor_model"]) 
        self.test_models = [LLMModel(model_handle=model) for model in config["test_models"]]
        self.test_indexes = config["test_indexes"]

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
            evaluator_model_handle = self.auditor_model.model_handle
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
    
    def compute_response_evaluation_tensor(self, config: EvaluationConfig, max_past_outputs=4):
        evaluation_array = np.empty((config.num_trials, 2), dtype=int)
        sim_model_names = np.empty((config.num_trials, 2), dtype=object)

        if config.save_response:
            response_array = np.empty((len(self.test_models), config.num_trials), dtype=object)

        def process_evaluator():
            past_prompts = []
            total_outputs = defaultdict(list)
            past_results = []

            for trial in range(config.num_trials):
                model_outputs = []
                # Generate adversarial prompt
                last_n_total_outputs = defaultdict(list, {model_id: outputs[-max_past_outputs:] for model_id, outputs in total_outputs.items()})
                p = self.generate_adversarial_prompt(model_handle = self.auditor_model.model_handle,
                                                     past_prompts = past_prompts[-max_past_outputs:],
                                                     past_outputs = last_n_total_outputs,
                                                     past_results = past_results[-max_past_outputs:])
                if p is None:
                    logger.warning(f"Unable to generate prompt using model handle {self.auditor_model.name}")
                    evaluation_array[trial, :] = None
                    continue
                past_prompts.append(p)
                logger.info(f"Evaluator {self.auditor_model.name} generated prompt: {p}")

                if config.rewrite_prompt:
                    p_optim = self.optimize_prompt(self.auditor_model.model_handle, p)
                    if p_optim is None:
                        logger.warning(f"Unable to optimize prompt using model handle {self.auditor_model.name}")
                        evaluation_array[trial, :] = None
                        continue
                    logger.info(f"Optimized prompt: {p_optim}")
                else:
                    p_optim = p

                for col_idx in range(len(self.test_models)):
                    # Don't need to evaluate the auditor
                    model_under_test = self.test_models[col_idx]
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
                    evaluation_array[trial, :] = result_indexes[0:2]
                    model_names = [self.test_models[result_indexes[0]].name, 
                                   self.test_models[result_indexes[1]].name]
                    sim_model_names[trial, :] = model_names
                    
                    correct = (result_indexes[0] ==  self.test_indexes[0])  & (result_indexes[1] ==  self.test_indexes[1])
                    correct_idxs = self.test_indexes
                    trial_result_obj = {
                        "correct": correct,
                        "selected_ids": result_indexes[0:2],
                        "correct_ids": correct_idxs
                    }
                    past_results.append(trial_result_obj)

                    logger.info(
                        f"Evaluator: {self.auditor_model.name}, "
                        f"Trial: {trial}, "
                        f"Trial Result: {trial_result_obj}"
                    )

                    for idx, response in enumerate(model_outputs):
                        total_outputs[idx].append(response)

        process_evaluator()

        output_obj = {
            'evaluations': evaluation_array,
            'evaluating_model':  self.auditor_model.name,
            'test_models': [m.name for m in self.test_models],
            'similar_models': sim_model_names
        }

        if config.save_response:
            output_obj['responses'] = response_array

        return output_obj
    
    def compute_accuracy(self, evaluation_outputs: Dict[str, any], warmup_steps: int): 

        evals = evaluation_outputs["evaluations"]
        total_trials = evals.shape[0]

        # Discount warmup steps
        effective_trials = total_trials - warmup_steps
        if effective_trials <= 0:
            raise ValueError("Warmup steps exceed or equal the total number of trials.")

        correct_matches = np.sum((evals[warmup_steps:, 0] == self.test_indexes[0]) & 
                                 (evals[warmup_steps:, 1] == self.test_indexes[1]))
        accuracy = correct_matches / effective_trials
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

def convert_ndarray_to_list(data):
    if isinstance(data, dict):
        return {key: convert_ndarray_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray_to_list(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data    

def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--models_file", type=str, required=True, help="Yaml file that contains info on the models")
    parser.add_argument('--num_trials', type=int, required=False, default=5, help="Number of trials to run")
    parser.add_argument('--config_path', type=str, required=False, help="Path for loading model api config")

    # Task arguments
    parser.add_argument('--rewrite_prompt', action='store_true', help="Prevent prompt rewrite")
    parser.add_argument('--save_response', action='store_true', help="Save LLM Response")
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--num_workers', type=int, default=5, help="Number of concurrent experiments to run")
    parser.add_argument('--warmup_steps', type=int, default=3, help="number of warmup steps")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    load_dotenv(args.config_path)
    model_yaml_file = args.models_file

    evaluator = AdversarialEvaluation(model_yaml_file)
    evaluation_config = EvaluationConfig({
        "num_trials": args.num_trials,
        "rewrite_prompt": args.rewrite_prompt,
        "save_response": args.save_response,
        "warmup_steps": args.warmup_steps
    })

    evaluation_outputs = evaluator.compute_response_evaluation_tensor(evaluation_config)
    accuracy = evaluator.compute_accuracy(evaluation_outputs, warmup_steps=evaluation_config.additional_attributes['warmup_steps'])

    logger.info(f"evaluation_outputs:")
    logger.info(evaluation_outputs)
    logger.info("accuracy:")
    logger.info(accuracy)
    
    metrics = {
        'test_models': [m.model_handle for m in evaluator.test_models],
        'evaluator_model': evaluator.auditor_model.model_handle,
        'accuracy': accuracy
    }
    
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)

        eval_output_path = os.path.join(args.output_path, 'eval_output.json')
        json.dump(convert_ndarray_to_list(evaluation_outputs), open(eval_output_path, 'w'))

        metric_output_path = os.path.join(args.output_path, 'metrics.json')
        json.dump(metrics, open(metric_output_path, 'w'))
        