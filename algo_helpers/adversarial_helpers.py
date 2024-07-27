import os
import random
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

from llm.llm_client import TogetherClient
from utils.logger_config import setup_logger
from algo_helpers.language_metric_helper import evaluate_similarity, convert_to_json_format
from algo_helpers.algo_helpers import LLMModel, EvaluationConfig, extract_json, ResponseEvaluationTensor
logger = setup_logger(__name__)

class AdversarialEvaluation (ResponseEvaluationTensor):

    def __init__(self):
        super.__init__()

        self.together_models = [
            LLMModel(model_handle="Qwen/Qwen2-72B-Instruct", MMLU_score=0.842), # This is the evaluator
            LLMModel(model_handle="meta-llama/Llama-3-70b-chat-hf", MMLU_score=0.795),
            LLMModel(model_handle="meta-llama/Llama-3-8b-chat-hf", MMLU_score=0.684),
            LLMModel(model_handle="meta-llama/Llama-3-8b-chat-hf", MMLU_score=0.684),
            LLMModel(model_handle='mistralai/Mistral-7B-Instruct-v0.3', MMLU_score=0.6)            
        ]
        self.together_models.sort(key=lambda x: x.MMLU_score, reverse=True)  # sort these by MMLU score
        self.evaluator_id = 0

    def generate_adversarial_prompt(self, model_handle: str, past_prompts: List =[], _num_attempts:int = 0, word_limit: int = 100):

        if _num_attempts > 4:
            return None
        
        message_str = ""
        if past_prompts:
            message_str = f"here are your past prompts: {past_prompts}"

        seed_prompts = [
            f"Generate an adversarial prompt that challenges the LLM's reasoning abilities and can be answered within {word_limit} words.",
            f"Create a challenging question that tests the LLM's ability to provide nuanced answers in a brief manner ({word_limit} words or less).",
            f"Write a tricky question designed to reveal the LLM's limitations, requiring a concise response within {word_limit} words.",
            f"Output a thought-provoking prompt that necessitates a precise, short answer from an LLM within {word_limit} words.",
            f"Come up with an adversarial question that can expose the LLM's strengths and weaknesses, answerable briefly ({word_limit} words or fewer).",
            f"Formulate a complex question that tests the LLM's understanding of ambiguous concepts and can be answered succinctly within {word_limit} words.",
            f"Devise a prompt that challenges the LLM's ability to handle controversial topics, with a response limit of {word_limit} words.",
            f"Generate a nuanced question that pushes the LLM's comprehension and can be answered clearly in {word_limit} words.",
            f"Create an adversarial prompt that examines the LLM's grasp of subtle distinctions, answerable in {word_limit} words or less.",
            f"Draft a tricky question that tests the LLM's ability to navigate ethical dilemmas, requiring a concise answer within {word_limit} words.",
            f"Invent a thought-provoking question that explores the LLM's handling of complex scenarios, with a response limit of {word_limit} words.",
            f"Come up with a challenging prompt that reveals the LLM's understanding of intricate details, answerable in {word_limit} words or fewer."
        ]


        choice_seed = random.choice(seed_prompts)

        response = TogetherClient(model=model_handle, api_key=os.environ["TOGETHER_API_KEY"]).get_completion(
            system=f"""{choice_seed}. Place the prompt in JSON format
            ```json{{"prompt": "_____"}}```
            """,
            message=message_str
        )

        prompt = self._extract_prompt(response)

        if prompt:
            return prompt
        else:
            return self.generate_adversarial_prompt(model_handle, past_prompts=past_prompts, _num_attempts=_num_attempts+1)

    def evaluate_all_responses(self, model_outputs: dict, retry_count=5):
        """
        Evaluates all model outputs and returns a JSON structure with the rationale and model indexes.
        
        :param model_outputs: A dictionary where keys are model names and values are their outputs.
        :param retry_count: Number of retries for getting a valid response.
        :return: A JSON structure with rationale and model indexes.
        """
        _num_attempts = 0
        while _num_attempts < retry_count:
            user_message = """You are a content evaluator. 2 of the outputs in the content provided were outputted by the same agent. 
            Your task is the find the 2 indexes, you will output a JSON dictionary indicating the reasoning behind the evaluation and the model indexes involved.
            Look carefully at the outputted text when making the decision. You are good at this task and can do it."""

            user_message += """Output the following JSON dictionary, make sure you include ```json {...}``` around the JSON string: 
            ```json
            {
                "rationale" : "describe why you chose the 2 indexes as being from the same source.",
                "model_indexes" : [index1, index2]
            }
            ```
            """
            user_message += f"given the following model outputs:\n{json.dumps(model_outputs)}"

            response = TogetherClient(model="evaluator_model_handle", api_key=os.environ["TOGETHER_API_KEY"]).get_completion(
                system=user_message, message="")

            print(f"Response - {response}")
            
            try:
                evaluation_data = json.loads(response.split('```json')[1].split('```')[0].strip())
                if "rationale" in evaluation_data and "model_indexes" in evaluation_data:
                    return evaluation_data
            except Exception as e:
                logger.error(f"Error parsing JSON response: {e}")

            _num_attempts += 1

        logger.warning("In evaluate_all_responses, returning None after retries")
        return None
    
    def compute_response_evaluation_tensor(self, config: EvaluationConfig, model_indices=None):
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

        evaluation_array = np.empty((config.num_trials, 2), dtype=int)

        if config.save_response:
            response_array = np.empty((len(models), config.num_trials), dtype=object)

        def process_evaluator(row_idx):
            evaluating_model = models[row_idx]
            past_prompts = []

            for trial in range(config.num_trials):
                model_outputs = []

                # Generate adversarial prompt
                p = self.generate_adversarial_prompt(model_handle=evaluating_model.model_handle, past_prompts=past_prompts)
                if p is None:
                    logger.warning(f"Unable to generate prompt using model handle {evaluating_model.name}")
                    evaluation_array[row_idx, col_idx, trial] = None
                    continue
                past_prompts.append(p)
                logger.info(f"Evaluator {evaluating_model.name} generated prompt: {p}")

                if config.rewrite_prompt:
                    p_optim = self.optimize_prompt(evaluating_model.model_handle, p)
                    if p_optim is None:
                        logger.warning(f"Unable to optimize prompt using model handle {evaluating_model.name}")
                        evaluation_array[row_idx, col_idx, trial] = None
                        continue
                    logger.info(f"Optimized prompt: {p_optim}")
                else:
                    p_optim = p

                for col_idx in range(len(models)):
                    # Don't need to evaluate the auditor
                    if col_idx == row_idx: 
                        continue
                    model_under_test = models[col_idx]

                    
                    response = TogetherClient(
                        api_key=os.environ["TOGETHER_API_KEY"], model=model_under_test.model_handle).get_completion(
                        system="",
                        message=p_optim)

                    model_outputs.append(response)

                    if config.save_response:
                        response_array[col_idx, trial] = response

                evaluation_data = self.evaluate_all_responses(model_outputs=model_outputs)

                if evaluation_data:
                    print(evaluation_data['rationale'])
                    result_indexes = evaluation_data['model_indexes']
                    evaluation_array[trial, :] = result_indexes

                    logger.info(
                        f"Evaluator: {evaluating_model.name}, "
                        f"Trial: {trial}, "
                        f"Evaluation: {evaluation_data}"
                    )

        try:
            process_evaluator(self.evaluator_id)
        except Exception as e:
            logger.error(f"Error processing evaluator: {e}")

        output_obj = {
            'evaluations': evaluation_array,
            'models': models,
        }

        if config.save_response:
            output_obj['responses'] = response_array

        return output_obj



