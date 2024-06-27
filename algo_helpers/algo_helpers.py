import re
import json

import matplotlib.pyplot as plt
import numpy as np
import ollama
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema

from llm.llm_provider_ranking_experiment import extract_json


class LLMModel:
    def __init__(self, ollama_handle, MMLU_score):
        self.ollama_handle = ollama_handle
        self.MMLU_score = MMLU_score


class ResponseEvaluationTensor:
    def __init__(self, models_to_compare=("phi3:mini", "llama3:8b", "gemma:7b", "gemma:2b")):
        self.ollama_models = [
            LLMModel("phi3:mini", MMLU_score=0.688),  # 3b params
            LLMModel("llama3:8b", MMLU_score=0.684),  # 8b params
            LLMModel("gemma:7b", MMLU_score=0.643),   # 7b params
            LLMModel("gemma:2b", MMLU_score=0.423),   # 2b params
            # LLMModel("phi3:medium", MMLU_score=0.782),          # 14b params
        ]

    @staticmethod
    def _extract_prompt(text):
        extracted_json = extract_json(text)
        
        if extracted_json is not None:
            prompt_value = extracted_json.get('prompt')
            if prompt_value:
                return prompt_value
            else:
                print("No 'prompt' key found in the extracted JSON.")
        else:
            print("No valid JSON found in the text.")
        
        return None

    def generate_prompt(self, model_handle: str):
        response = ollama.chat(model=model_handle, messages=[
            {
                'role': 'user',
                'content': 'Generate a prompt as if a user would. Place the prompt in JSON format '
                           '```json{"prompt": "_____"}```',
            },
        ])

        prompt = self._extract_prompt(response['message']['content'])
        if prompt:
            return prompt
        else:
            return self.generate_prompt(model_handle)

    def optimize_prompt(self, model_handle: str, unoptimized_prompt: str):
        response = ollama.chat(model=model_handle, messages=[
            {
                'role': 'user',
                'content': 'Given this prompt, optimize it so that you, when asked, '
                           'would produce the best possible answer: BEGINNING OF PROMPT\n'
                           f'{unoptimized_prompt}\n'
                           'END OF PROMPT\n'
                            ' Place your optimized prompt in JSON format '
                           '```json{"prompt": "_____"}```'
            }
        ])

        prompt = self._extract_prompt(response['message']['content'])
        if prompt:
            return prompt
        else:
            return self.generate_prompt(model_handle)


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
            print("NO JSON string found")
            return None

    def rate_response(self, model_handle: str, model_prompt: str, model_output: str):
        response = ollama.chat(model=model_handle, messages=[
            {
                'role': 'user',
                'content': f'Given this user query: {model_prompt}, '
                           f'how would you rate this output on a scale of 1 to 5\n'
                           f'model output: {model_output}\n'
                           ' Place your rating in JSON format '
                           '```json{"rating": <integer, either 1 2 3 4 5}```'
            }
        ])

        # extract rating
        rating_value = self._extract_rating(response['message']['content'])

        # ensure it's an int between 1 and 5 (incl.)
        if isinstance(rating_value, int) and 1 <= rating_value <= 5:
            return rating_value
        else:
            return self.rate_response(model_handle, model_prompt, model_output)

    def compute_response_evaluation_tensor(self):
        return

    def run_eval(self):
        pass


if __name__ == "__main__":

    model_to_test = "gemma:2b"
    evaluator = ResponseEvaluationTensor()
    p = evaluator.generate_prompt(model_to_test)
    print(p)

    p_optim = evaluator.optimize_prompt(model_to_test, p)
    print(p_optim)

    # DUT
    response = ollama.chat(model=model_to_test, messages=[
        {
            'role': 'user',
            'content': p_optim
        }
    ])['message']['content']

    print(response)

    rating = evaluator.rate_response(model_to_test, p_optim, response)

    print(rating)
