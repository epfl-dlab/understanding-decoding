import argparse
import pandas as pd
import numpy as np

import json
import os


def get_model_name(file_name):
    parts = file_name.split("_")
    for part in parts:
        if part.startswith("step"):
            return part.split(".")[0]

    if "unbiased-albert" in file_name:
        return "oracle_unbiased-albert"

    raise Exception("Unexpected file name")


def get_results(path):
    with open(path) as f:
        results = json.load(f)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert old GenIE checkpoint to new setting.")
    parser.add_argument(
        "--results_folder", type=str, required=True,
        help="The directory containing the jsonfiles from the evaluations."
    )
    args = parser.parse_args()

    model_name2results = {get_model_name(file_name): get_results(os.path.join(args.results_folder, file_name))
                          for file_name in os.listdir(args.results_folder)
                          if file_name.startswith('results_N') or file_name.startswith("results_unbiased")}

    column_names = model_name2results['oracle_unbiased-albert'].keys()
    data = {model_name: [results[key] for key in column_names] for model_name, results in model_name2results.items()}
    results = pd.DataFrame.from_dict(data, orient='index', columns=column_names)
    results = results.iloc[np.argsort(
        list(map(lambda x: 10e20 if "oracle" in x else int(x.split("=")[1]), list(results.index.to_numpy()))))]
    results.to_csv("detoxify_noisy_oracle_results.csv", index=False)
