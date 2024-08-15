import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def calculate_metrics(df, successful_counterfactuals_only, calculate_perplexity_score):
    if successful_counterfactuals_only:
        df = df[df["found_flip"] == True]

    # if calculate_perplexity_score:
    #     perplexity_score = df.apply(lambda row: get_perplexity_score(row["counterfactual_perplexity"], row["original_perplexity"]), axis=1)
    # else:
    #     perplexity_score = df["counterfactual_perplexity"]

    metrics = {
        "label_flip_score": df["found_flip"],
        "sparsity_score": df["levenshtein_similarity_score"],
        # "perplexity_score": df["counterfactual_perplexity"]
        "perplexity_score": df["original_perplexity"]
    }

    return {key: value for key, value in metrics.items()}

def calculate_confidence_interval(data, precision, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return np.round(h, precision)

def process_csv_files(directory, dataset, dataset_size, successful_counterfactuals_only, median, precision, calculate_perplexity_score):
    data = {
        "method_name": [],
        "label_flip_score": [],
        "sparsity_score": [],
        "perplexity_score": [],
        "label_flip_score_ci": [],
        "sparsity_score_ci": [],
        "perplexity_score_ci": []
    }

    # methods = ["HOTFLIP", "CLOSS", "POLYJUICE", "FIZLENAIVE", "FIZLEGUIDED"]
    methods = ["FIZLENAIVE"]

    for method in methods:
        filename = f"{method.lower()}-output-{dataset}-{dataset_size}.csv"
        if filename in os.listdir(directory):
            data["method_name"].append(method)
            df = pd.read_csv(os.path.join(directory, filename))
            metrics = calculate_metrics(df, successful_counterfactuals_only, calculate_perplexity_score)
            for key in ["label_flip_score", "sparsity_score", "perplexity_score"]:
                metric = np.mean(metrics[key]) if not median else np.median(metrics[key])
                metric = np.round(metric, precision)
                data[key].append(metric)
                data[f"{key}_ci"].append(calculate_confidence_interval(metrics[key], precision))

    return data

def main():

    dataset_size = 1000
    output_dir = "../output"

    successful_counterfactuals_only = False
    precision = 3
    median = False
    calculate_perplexity_score = False

    dataset = "sst_2"
    print(f"Dataset: {dataset}:")
    data = process_csv_files(output_dir, dataset, dataset_size, successful_counterfactuals_only, median, precision, calculate_perplexity_score)
    df_sst = pd.DataFrame(data).set_index("method_name")

    df = df_sst

    print(df.head().to_string())

    original_perplexity = df["original_perplexity"]
    counterfactual_perplexity = df["counterfactual_perplexity"]

    cutoff = 5000
    original_perplexity = original_perplexity[original_perplexity < cutoff]
    counterfactual_perplexity = counterfactual_perplexity[counterfactual_perplexity < cutoff]

    # specify a range of values for the x-axis:
    plt.hist(original_perplexity, bins=50, alpha=0.5, label="Original perplexity", color="red");
    plt.hist(counterfactual_perplexity, bins=50, alpha=0.5, label="Counterfactual perplexity", color="green");

    mean_original_perplexity = np.mean(original_perplexity)
    mean_counterfactual_perplexity = np.mean(counterfactual_perplexity)

    print(f"Original perplexity mean: {mean_original_perplexity}")
    print(f"Counterfactual perplexity mean: {mean_counterfactual_perplexity}")

if __name__ == "__main__":
    main()
