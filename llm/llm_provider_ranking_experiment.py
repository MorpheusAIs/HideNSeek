import json
import re

from llm.llm_client import client_from_args

from dataclasses import dataclass


def extract_json(text):
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
            print(f"error - {e}")
            return None
    return None


def check_top_rank(rankings: list[int], idx: int) -> int:
    # Adjust rankings if they are 1-indexed
    if max(rankings) == len(rankings):
        rankings = [rank - 1 for rank in rankings]
    
    # Check if the top rank matches the specified idx
    if rankings[0] == idx:
        return 1
    else:
        return 0



@dataclass
class OutputData:
    response: str
    client_model: str


def generate_and_rank_outputs(user_prompt):
    clients = [
        client_from_args("anthropic", model="claude-3-opus-20240229"),
        client_from_args("openai", model="gpt-4"),
        client_from_args("together", model="meta-llama/Llama-3-70b-chat-hf"),
    ]

    outputs = []
    for client in clients:
        output = client.get_completion(system="", message=user_prompt, temperature=0.7)
        outputs.append(OutputData(response=output, client_model=client.model))

    rankings = []
    for i, client in enumerate(clients):
        ranking_prompt = f"based on the following prompt: ```{user_prompt}``` evaluate the following outputs\n\nOutputs:\n{[f'```idx: {idx}, output: {output.response}```' for idx, output in enumerate(outputs)]}"
        system_prompt = """You are a content grader who will grade all content based on its clarity, relevance, and coherence. Evaluate all options and out a ranking.
        You will respond in the following format: 
        ```json
        the json output
        ```

        you have access to the following tools:
            {
                "name": "evaluate_content",
                "description": "Evaluate and rank content based on specified criteria",
                "input_schema": {
                    "type": "object",
                    "properties": {
                    "criteria": {
                        "type": "string",
                        "description": "An explanation of how the content will be evaluated"
                    },
                    "ranking": {
                        "type": "array",
                        "items": {
                        "type": "integer"
                        },
                        "description": "A list representing the ranking of the outputs from best to worst, e.g. [1,2,3]"
                    },
                    "rationale": {
                        "type": "string", 
                        "description": "A rationale for the ranking"
                    }
                    },
                    "required": ["criteria", "ranking", "rationale"]
                }
            }
        """
        ranking = client.get_completion(system=system_prompt, message=ranking_prompt, temperature=0.3)
        rankings.append(OutputData(response=ranking, client_model=client.model))

    return outputs, rankings


def check_self_consistency(rankings: list[OutputData], target_idx: int) -> int:
    print(f"rankings - {rankings}")
    if check_top_rank(rankings, target_idx):
        return 1
    else:
        return 0


if __name__ == "__main__":
    user_prompt = "What are the key differences between Python and JavaScript?"
    outputs, rankings = generate_and_rank_outputs(user_prompt)
    
    # print("Outputs:")
    # for output in outputs:
        # print(f"{output.client_model}:")
        # print(output.response + "\n")

    # print("Rankings:")
    # for ranking in rankings:
        # print(f"{ranking.client_model}:")
        # print(ranking.response + "\n")
    
    # for checking self-consistency, if all llms will select them selves, than we will see that their own idx is first
    total_consistency = 0
    for idx, ranking in enumerate(rankings):
        response_json = extract_json(ranking.response)
        ranking_list = response_json["ranking"]
        self_selected = check_self_consistency(ranking_list, idx)
        self_selected += total_consistency
    print(f"self-consistency ratio: {self_selected / len(rankings)}")
    
