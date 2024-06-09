import numpy as np
from scipy.stats import norm, wilcoxon, rankdata

from algo_helpers.algo_helpers import identify_bad_actors
from mocks.mock_ranks import generate_mock_rank_matrix


# Example usage
num_trials = 30
sample_size = 33

# Simulated rank data for demonstration purposes
# ranks_no_bad = generate_mock_rank_matrix(sample_size=sample_size, num_trials=num_trials, bad_actor_indices=[])
# ranks_one_bad = generate_mock_rank_matrix(sample_size=sample_size, num_trials=num_trials, bad_actor_indices=[0, ])
ranks_two_bad = generate_mock_rank_matrix(sample_size=sample_size, num_trials=num_trials, bad_actor_indices=[0, 1])

# ranks_on_bad_manual = np.array([[5, 5, 5, 5, 5],
#                                 [4, 1, 3, 2, 4],
#                                 [2, 4, 1, 3, 2],
#                                 [3, 2, 4, 1, 3],
#                                 [1, 3, 2, 4, 1]])


constructed_ranks = []
for t in range(num_trials):
    sample_ranks = np.arange(1, sample_size)
    np.random.shuffle(sample_ranks)
    temp_array_trial = np.array([sample_size, *sample_ranks])
    constructed_ranks.append(temp_array_trial)

constructed_ranks = np.stack(constructed_ranks, axis=0)
constructed_ranks = constructed_ranks.T

# FIXME, make mock data selection more apparent
selected_ranks = ranks_two_bad

print(selected_ranks)
print(selected_ranks.shape)

bad_actors = identify_bad_actors(selected_ranks)

print(f"Identified bad actors: {bad_actors}")
