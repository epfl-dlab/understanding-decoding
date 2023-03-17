import os
import numpy as np
import pandas as pd

import src.utils.general as utils
from src.utils.evaluation import select_indices_on_quantiles


log = utils.get_logger(__name__)


def _filter_data_points_on(data, key):
    filtered_data = [dp for dp in data if dp[key][0] != None]
    log.info(f"Samples filtered due to invalid `{key}`: {len(data)-len(filtered_data)}")
    return filtered_data


def select_data_to_resample(output_dataset, num_qs, testing_output_parent_dir=None):
    dataset = output_dataset.data

    dataset = _filter_data_points_on(dataset, key="prediction_log_likelihood")
    has_targets = "target_log_likelihood" in dataset[0].keys()
    if has_targets:
        dataset = _filter_data_points_on(dataset, key="target_log_likelihood")

    data = pd.DataFrame(dataset)
    if has_targets:
        data["tgt_top_pred_likelihood_diff"] = data.apply(
            lambda x: np.exp(x.prediction_log_likelihood[0]) - np.exp(x.target_log_likelihood[0]), axis=1
        )
    else:
        data["tgt_top_pred_likelihood_diff"] = data.apply(lambda x: np.exp(x.prediction_log_likelihood[0]), axis=1)

    indices_to_keep = select_indices_on_quantiles(
        data=data["tgt_top_pred_likelihood_diff"].to_numpy(), num_qs=num_qs, is_data_sorted=False
    )

    target_quantiles = np.arange(0, num_qs + 1) / num_qs
    data_to_keep = data.iloc[indices_to_keep].copy()
    data_to_keep.reset_index(inplace=True, drop=True)
    data_to_keep["quantile"] = target_quantiles
    data_to_keep["empirical_quantile"] = [
        sum([diff <= tgt_val for diff in data["tgt_top_pred_likelihood_diff"]])
        / data["tgt_top_pred_likelihood_diff"].shape[0]
        for tgt_val in data_to_keep["tgt_top_pred_likelihood_diff"]
    ]
    data_to_keep["empirical_quantile"][0] = 0
    data_to_keep = data_to_keep[["id", "tgt_top_pred_likelihood_diff", "quantile", "empirical_quantile"]]
    print("[sanity check] Target quantiles:", data_to_keep["quantile"].to_list())
    print("[sanity check] Observed quantiles:", data_to_keep["empirical_quantile"].to_list())

    if testing_output_parent_dir is not None:
        data_save_path = "resample_input_stats.jsonl.gz"
        data_save_path = os.path.join(testing_output_parent_dir, data_save_path)
        data_to_keep.to_json(data_save_path, orient="records", lines=True, compression="gzip")

    ids_to_keep = data_to_keep["id"].to_numpy()

    return ids_to_keep
