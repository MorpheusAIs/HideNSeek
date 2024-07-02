# HideNSeek
HideNSeek is a Model Fidelity Verification Algorithm for Probabilistically Fingerprinting LLMs

`algo_helpers/algo_helpers.py` currently has a working version of an early preview version of Hide-N-Sekk that utilizes a probablistic approach for idetnifying various LLMS

![LLM Groupings discovered](model_groups.png)

### Algorithm

The algorithm consists of the following steps:

1. **Data Extraction**:
    - Extract the scores given by the reference model (ref_idx) and the test model (test_idx) across all trials.
    - Extract the scores received by the reference model and the test model from all other models across all trials.
2. **Statistical Analysis**:
    - Compute the means and standard deviations of the extracted scores.
    - Perform two-sample t-tests to evaluate the null hypothesis that the means of the scores are equal.
    - Perform Mann-Whitney U tests as a non-parametric alternative to the t-tests.
    - Calculate confidence intervals for the means of the scores.
    - Compute Cohen's d to measure the effect size.
3. **Decision Rule**:
    - If the standard deviations of all scores are zero, directly compare the scores for equality.
    - Use the p-values from the t-tests to determine if the models are statistically similar. If both p-values are above the significance level (α = 0.05), the null hypothesis is not rejected, indicating similarity.
    - The algorithm flags the models as similar if the null hypothesis is not rejected for both given and received scores.

A Matrix can than be generated thats `M X M` that contains `1`'s where the models where confused for one another. A grouping algorithm can than visualize the groups and showcase what models are confused for which