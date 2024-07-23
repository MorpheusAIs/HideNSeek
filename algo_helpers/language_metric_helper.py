import itertools
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

def extract_responses(response_array: np.ndarray, model_names: List[str]) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    models = {}
    for idx in range(len(response_array)):
        model_name = model_names[idx]
        models[model_name] = list(itertools.chain.from_iterable(response_array[:, idx, :]))

    all_responses = []
    response_mapping = []
    for model, model_responses in models.items():
        for response in model_responses:
            if response is not None:
                all_responses.append(response)
                response_mapping.append(model)

    return models, response_mapping, all_responses

def generate_similarity_matrix(responses: List[str], approach: str = 'tf_idf', 
                               vectorizer: CountVectorizer = None) -> np.ndarray:
    if vectorizer:
        transformed_matrix = vectorizer.transform(responses)
        similarity_matrix = cosine_similarity(transformed_matrix)
        return similarity_matrix
    
    if approach == 'tf_idf':
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(responses)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix
    elif approach == 'ngram':
        vectorizer = CountVectorizer(ngram_range=(1, 3))
        ngram_matrix = vectorizer.fit_transform(responses)
        similarity_matrix = cosine_similarity(ngram_matrix)
        return similarity_matrix
    else:
        raise ValueError(f"Unsupported approach: {approach}")

def word_metric_global(models: Dict[str, List[str]], response_mapping: List[str], vectorization_approach: str = 'tf_idf', cosine_threshold: float = 0.25, debug: bool = False) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Generate similarity matrices based on aggregate responses and then compute similarity. 
    """
    similar_models = {}

    # Collect responses indices for each model
    model_indices = {model: [] for model in models}
    for idx, model in enumerate(response_mapping):
        model_indices[model].append(idx)

    # Remove models if any responses are nan
    model_indices = {k: v for k, v in model_indices.items() if v}
    delete_models = [k for k in models if k not in model_indices]
    for model_delete in delete_models:
        print(f"Removing model: {model_delete} from computation. This is due to bad response by model.")
        del model_indices[model_delete]

    # Generate similarity matrix
    all_responses = [response for model_responses in models.values() for response in model_responses if response is not None]
    similarity_matrix = generate_similarity_matrix(all_responses, vectorization_approach)

    # Compare models based on their average similarity score
    for model_1, indices_1 in model_indices.items():
        for model_2, indices_2 in model_indices.items():
            if model_1 != model_2 and (model_2, model_1) not in similar_models:
                similarity_scores = [similarity_matrix[i, j] for i in indices_1 for j in indices_2]
                if similarity_scores:
                    average_similarity = round(sum(similarity_scores) / len(similarity_scores), 3)
                    if debug:
                        print(model_1, model_2, average_similarity)
                    similar_models[(model_1, model_2)] = {
                        'is_similar': bool(average_similarity > cosine_threshold),
                        'match_score': average_similarity
                    }

    return similar_models

def word_metric_question_wise(models: Dict[str, List[str]], model_names: List[str], cosine_threshold: float = 0.25, vectorization_approach: str = 'tf_idf', debug: bool = False) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Generate similarity matrices per question and then average up the results.
    """
    num_models = len(model_names)
    num_questions = len(models[model_names[0]])
    similarity_scores = np.zeros((num_models, num_models))

    for q in range(num_questions):
        question_responses = [models[model][q] for model in model_names]
        question_similarity_matrix = generate_similarity_matrix(question_responses, vectorization_approach)
        similarity_scores += question_similarity_matrix

    average_similarity_scores = similarity_scores / num_questions

    similar_models = {}
    for i in range(num_models):
        for j in range(i + 1, num_models):
            if debug:
                print(model_names[i], model_names[j], average_similarity_scores[i, j])
            similar_models[(model_names[i], model_names[j])] = {
                'is_similar': bool(average_similarity_scores[i, j] > cosine_threshold),
                'match_score': average_similarity_scores[i, j]
            }

    return similar_models

def evaluate_similarity(response_array: np.ndarray, model_names: List[str], cosine_threshold: float = 0.53, approach: str = 'question_wise', vectorization_approach: str = 'tf_idf', debug: bool = False) -> Dict[Tuple[str, str], Dict[str, float]]:
    models, response_mapping, all_responses = extract_responses(response_array, model_names)
    if debug:
        print(f'All responses length: {len(all_responses)}')
        print(f"Number of models: {len(models)}")
    
    if approach == 'global':
        return word_metric_global(models, response_mapping, vectorization_approach, cosine_threshold, debug)
    elif approach == 'question_wise':
        return word_metric_question_wise(models, model_names, cosine_threshold, vectorization_approach, debug)
    else:
        raise ValueError(f"Unsupported approach: {approach}")

def convert_to_json_format(metrics: Dict[Tuple[str], Dict]) -> List[Dict[Tuple[str], Dict]]:
    output = []
    for models, vals in metrics.items():
        a = {'models': models}
        a.update(vals)
        output.append(a)
    return output
