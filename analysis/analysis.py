# %%
import pandas as pd
import os

# %% [markdown]
# ## Output analysis and evaluation
# 
# Metrics to calculate:
# 
# - **Label flip score:** the percentage of the time the counterfactual flips the output of the classifier.
#     - You can calculate this easily using the found flip column.
# - **Similarity score:** 1 - the normalized Levenshtein distance between the original and counterfactual text.
# - **Perplexity:** the perplexity score of the generated counterfactual (lower is better)
#     - Calculate the average perplexity for the original and counterfactual text.

# %%
# %cd "masters-thesis-code/analysis"
# %pwd

# %%
dataset = "qnli"
# dataset = "sst_2"

dataset_size = 100

# %% [markdown]
# Best possible:
# - counterfactual_perplexity: 259.4843315696716, original_perplexity: 259.4843315696716
# - perplexity_score: ~1.0
# 
# Polyjuice:
# - counterfactual_perplexity: 330.9339884185791, original_perplexity: 259.4843315696716
# - perplexity score: ~0.7
# 
# CLOSS:
# - counterfactual_perplexity: 588.1533823394775, original_perplexity: 259.4843315696716
# - perplexity score: ~0.4
# 
# Worst possible score:
# - counterfactual_perplexity: 1500, original_perplexity: 259.4843315696716
# - perplexity score: ~0.1
# 

# %%
import numpy as np

# def calculate_perplexity_score(counterfactual_perplexity, original_perplexity):
#     print(f"Counterfactual perplexity: {counterfactual_perplexity}, Original perplexity: {original_perplexity}")
#     ratio = counterfactual_perplexity / original_perplexity
#     score = 1 / ratio
#     return score

def calculate_perplexity_score(counterfactual_perplexity, original_perplexity):
    ratio = counterfactual_perplexity / original_perplexity
    # midpoint = 1.5
    midpoint = 2
    steepness = 3
    score = 1 / (1 + np.exp(steepness * (ratio - midpoint)))
    print(f"Counterfactual perplexity: {counterfactual_perplexity}, Original perplexity: {original_perplexity}, Ratio: {ratio}, score: {score}")
    return score

def calculate_metrics(df, precision, successful_counterfactuals_only, median):
    """Calculate and round metrics from the given DataFrame."""
    # for each data CSV, filter out the rows that are True for the "found_flip" column:
    if successful_counterfactuals_only:
        df = df[df["found_flip"] == True]

    if median:
        metrics = {
            "label_flip_score": df["found_flip"].median(),
            "sparsity_score": df["levenshtein_similarity_score"].median(),
            "perplexity_score": calculate_perplexity_score(df["counterfactual_perplexity"].median(), df["original_perplexity"].median())
        }
    else:
        metrics = {
            "label_flip_score": df["found_flip"].mean(),
            "sparsity_score": df["levenshtein_similarity_score"].mean(),
            "perplexity_score": calculate_perplexity_score(df["counterfactual_perplexity"].mean(), df["original_perplexity"].mean())
        }
    return {key: round(value, precision) for key, value in metrics.items()}

def min_max_normalize(arr):
    """Perform min-max normalization on the given array."""
    min_val, max_val = min(arr), max(arr)
    return [(x - min_val) / (max_val - min_val) for x in arr]

def process_csv_files(directory, dataset, successful_counterfactuals_only, median):
    """Process all CSV files in the given directory and return collected data."""
    data = {
        "method_name": [],
        "label_flip_score": [],
        "sparsity_score": [],
        "perplexity_score": []
    }

    methods = ["HOTFLIP", "CLOSS", "POLYJUICE", "FIZLENAIVE", "FIZLEGUIDED"]
    # methods = ["HOTFLIP"]

    for method in methods:
        filename = f"{method.lower()}-output-{dataset}.csv"
        if filename in os.listdir(directory):
            data["method_name"].append(method)
            print(f"Processing method: {method}")
            df = pd.read_csv(os.path.join(directory, filename))
            metrics = calculate_metrics(df, 3, successful_counterfactuals_only, median)
            for key in metrics:
                data[key].append(metrics[key])

    return data


# %%
output_dir = "../output"

# Collect data from the CSV files
# data = process_csv_files(output_dir, dataset, successful_counterfactuals_only=True, median=True)
data = process_csv_files(output_dir, dataset, successful_counterfactuals_only=False, median=False)

# Display the final dataframe
df = pd.DataFrame(data).set_index("method_name")
df.head()

# %% [markdown]
# ## Create bar chart

# %%
import matplotlib.pyplot as plt

# plt.style.use('ggplot')
plt.style.use('default')

ax = df.plot.bar(figsize=(16, 8), fontsize=12)
ax.set_title(f'Metrics for {dataset}', fontsize=16, pad=20)

ax.set_xlabel('Method', fontsize=14, labelpad=10)
ax.set_xticklabels(df.index.str.upper(), rotation=0, ha='center', fontsize=12)

ax.set_ylabel('Scores', fontsize=14, labelpad=10)
ax.set_ylim([0, 1])

ax.legend(labels=["Coverage score", "Sparsity score", "Plausibility score"], fontsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.savefig(f"results-{dataset}-{dataset_size}.png", dpi=300)

# %% [markdown]
# ### Thoughts
# 
# - FIZLE seems to prioritize plausibility over sparsity
# - CLOSS prioritizes sparsity over plausibility
# - CLOSS and HOTFLIP focus on maximizing coverage while maximizing sparsity
# 
# Question: which metric do users care most about: plausibility, sparsity, or coverage when explaining a model?
# 
# You could find out using an RLHF-like method where users compare the usefulness of explanations and then you can calculate the three metrics and see which one is most correlated with real-world usefulness.

# %% [markdown]
# ## QNLI results analysis

# %%



