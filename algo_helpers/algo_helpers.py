import numpy
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema

import matplotlib.pyplot as plt


def is_all_same(arr):
    first_element = arr[0]
    return np.all(arr == first_element)


def normalize_rank_matrix(rank_matrix):
    median_per_trial = np.median(rank_matrix, axis=0)
    # assert is_all_same(median_per_trial), \
    #     ("The rank matrix is incorrect. "
    #      "Ensure all rank integers from [1..NumActors] are represented exactly once per column.")

    median = numpy.median(median_per_trial)

    rank_matrix_normalized = rank_matrix - median

    return rank_matrix_normalized


def identify_bad_actors(rank_matrix):
    rank_matrix_norm = normalize_rank_matrix(rank_matrix)
    score_per_actor = np.sum(rank_matrix_norm, axis=1).astype(int)

    # Estimate the PDF using KDE
    kde = gaussian_kde(score_per_actor)
    x = np.linspace(score_per_actor.min(), score_per_actor.max(), 1000)
    y = kde(x)

    # Find the indices of the local maxima (peaks) in the PDF
    peak_indices = argrelextrema(y, np.greater)[0]

    # Find the index of the minimum value between the two peaks
    if len(peak_indices) >= 2:
        min_index = np.argmin(y[peak_indices[0]:peak_indices[1]]) + peak_indices[0]
        threshold = x[min_index]
    else:
        threshold = None

    # Plot the KDE and the threshold
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, linewidth=2, label='PDF')
    ax.fill_between(x, y, alpha=0.3)
    if threshold is not None:
        ax.axvline(threshold, color='red', linestyle='--', label='Threshold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Bimodal Distribution - PDF and Threshold')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Print the threshold value
    print(f"Threshold: {threshold}")

    return np.where(score_per_actor > 0)
