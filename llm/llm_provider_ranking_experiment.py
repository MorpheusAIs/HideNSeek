from llm.llm_client import client_from_args

def generate_and_rank_outputs(user_prompt):
    clients = [
        client_from_args("anthropic", model="claude-3-opus-20240229"),
        client_from_args("openai", model="gpt-4"),
        client_from_args("together", model="meta-llama/Llama-3-70b-chat-hf"),
    ]

    outputs = []
    for client in clients:
        output = client.get_completion(system="", message=user_prompt)
        outputs.append(output)

    rankings = []
    for i, client in enumerate(clients):
        ranking_prompt = f"Rank the following outputs from best to worst:\n\n{outputs}\n\nRanking:"
        ranking = client.get_completion(system="", message=ranking_prompt)
        rankings.append(ranking)

    return outputs, rankings

if __name__ == "__main__":
    user_prompt = "What are the key differences between Python and JavaScript?"
    outputs, rankings = generate_and_rank_outputs(user_prompt)

    print("Outputs:")
    for output in outputs:
        print(output + "\n")

    print("Rankings:")
    for ranking in rankings:
        print(ranking + "\n")
    