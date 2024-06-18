from llm_client import client_from_args



from dataclasses import dataclass

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
        output = client.get_completion(system="", message=user_prompt)
        outputs.append(OutputData(response=output, client_model=client.model))

    rankings = []
    for i, client in enumerate(clients):
        ranking_prompt = f"based on the following prompt: ```{user_prompt}``` evaluate the following outputs\n\nOutputs:\n{[f'```idx: {idx}, output: {output.response}```' for idx, output in enumerate(outputs)]}"
        system_prompt = """You are a content grader who will grade all content based on its clarity, relevance, and coherence. Evaluate all options and out a ranking.
        You will respond in the following format: 
        ```json
        {
            criteria: an explination of how you will evaluate the content.
            ranking: A list representing the ranking of the outputs from best to worst like `[1,2,3]`
            rational: A rational for your ranking.
        }
        ```
        """
        ranking = client.get_completion(system=system_prompt, message=ranking_prompt)
        rankings.append(OutputData(response=ranking, client_model=client.model))

    return outputs, rankings


if __name__ == "__main__":
    user_prompt = "What are the key differences between Python and JavaScript?"
    outputs, rankings = generate_and_rank_outputs(user_prompt)

    print("Outputs:")
    for output in outputs:
        print(f"{output.client_model}:")
        print(output.response + "\n")

    print("Rankings:")
    for ranking in rankings:
        print(f"{ranking.client_model}:")
        print(ranking.response + "\n")
    