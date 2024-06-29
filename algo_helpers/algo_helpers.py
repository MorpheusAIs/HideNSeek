import re
import json

import matplotlib.pyplot as plt
import numpy as np
import ollama
import pandas as pd
from scipy.stats import gaussian_kde, ttest_1samp
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
            # attempt to extract `rating: #` out of the text
            
                pattern = re.compile(r'(?:"rating"|rating)\s*:\s*(?:"?(\d+)"?)(?:\s*,|\s*\n|\s*$)', re.IGNORECASE)
    
                match = pattern.search(text)
                if match:
                    return int(match.group(1))
                else:
                     print("NO JSON string found")
                     return None

    def rate_response(self, model_handle: str, model_prompt: str, model_output: str):
        
        user_message = """You are a content grader who will output a rating between 1 to 5 indicating how well the provided content follows the user query.
        Output a 5 if the content fully followed the user query, and output a 1 if it doesn't follow it at all. 
        You will respond in the following format: 
        ```json
        the json output
        ```

        Output the following JSON dictionary: 
        {
            "rational" : describe why you chose the rating,
            "rating" : Integer representing the rating between 1 to 5
        }\n
        """
        user_message += f"given the following user query:\n{model_prompt}. provide a rating for this Context:\n{model_output}"

        response = ollama.chat(model=model_handle, messages=[
            {
                "role" : "user",
                "content" : user_message
            }
        ])
        print(f"response - {response}")
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

    # model_to_test = "gemma:2b"
    # evaluator = ResponseEvaluationTensor()
    # p = evaluator.generate_prompt(model_to_test)
    # print(p)

    # p_optim = evaluator.optimize_prompt(model_to_test, p)
    # print(p_optim)

    # # DUT
    # response = ollama.chat(model=model_to_test, messages=[
    #     {
    #         'role': 'user',
    #         'content': p_optim
    #     }
    # ])['message']['content']

    # print(response)

    # rating = evaluator.rate_response(model_to_test, p_optim, response)

    # print(rating)
    # test tensor
    random_tensor = np.random.randint(low=1, high=6, size=(3, 3, 3))
    print(f"random_tensor - {random_tensor}")
    evaluator = ResponseEvaluationTensor()
    # test out the stest methods on tensors using the random tensor
    pref_bias_stats = evaluator.test_self_preference_bias(random_tensor)
    print(f"pref_bias_stats - {pref_bias_stats}")
    higher_bias_stats = evaluator.permutation_test_model_preference(random_tensor)
    print(f"higher_bias_stats - {higher_bias_stats}")


    
