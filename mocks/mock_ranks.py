import time

import numpy as np

np.random.seed(int(time.time()))


def generate_mock_rank_matrix(sample_size, num_trials, bad_actor_indices=()):
    ranks = np.empty((sample_size, num_trials), dtype=int)

    for trial in range(num_trials):
        # Generate random ranks for each trial
        trial_ranks = np.arange(1, sample_size + 1)
        np.random.shuffle(trial_ranks)

        # Assign higher ranks to bad actors
        num_bad_actors = len(bad_actor_indices)
        bad_actor_ranks = np.random.choice(trial_ranks[-num_bad_actors:], size=num_bad_actors, replace=False)
        trial_ranks[bad_actor_indices] = bad_actor_ranks

        ranks[:, trial] = trial_ranks

    return ranks


if __name__ == '__main__':
    ranks = generate_mock_rank_matrix(4, 10, [0,])
    print(ranks)
    print(np.sum(ranks, axis=1).reshape((-1, 1)))
    # print(np.sum(ranks, axis=0))
