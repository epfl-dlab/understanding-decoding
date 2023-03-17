import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

import jsonlines
import numpy as np
import wandb
from pytorch_lightning.utilities import rank_zero_only


def group_sequences(sequences, n_items_per_group):
    """

    Parameters
    ----------
    sequences: sequences to be grouped, [s1, s2, s3, s3, s5, s6]
    n_items_per_group: integer if all groups are of equal size, or a list of varying group lengths

    Returns
    -------
    A list of grouped sequences.

    Example 1:
    sequences=[s1,s2,s3,s3,s5,s6]
    n_items_per_group=3
    returns: [[s1,s2,s3], [s4,s5,s6]]

    Example 2:
    sequences=[s1,s2,s3,s3,s5,s6]
    n_items_per_group=[2,4]
    returns: [[s1,s2], [s3,s4,s5,s6]]

    """
    if isinstance(n_items_per_group, int):
        assert len(sequences) % n_items_per_group == 0
        n_items_per_group = [n_items_per_group for _ in range(len(sequences) // n_items_per_group)]

    grouped_sequences = []
    start_idx = 0
    for n_items in n_items_per_group:
        grouped_sequences.append(sequences[start_idx : start_idx + n_items])
        start_idx += n_items

    return grouped_sequences


def ungroup_sequences(grouped_sequences):
    """

    Parameters
    ----------
    grouped_sequences: a list like [[s1,s2], [s3,s4,s5,s6]]

    Returns
    -------
    a list, for example [s1,s2,s3,s4,s5,s6]

    """
    return [seq for group in grouped_sequences for seq in group]


def first_sequence(grouped_sequences):
    """

    Parameters
    ----------
    grouped_sequences: a list like [[s1,s2], [s3,s4,s5,s6], [s7,s8,s9]]

    Returns
    -------
    a list, for example [[s1],[s3],[s7]]

    """
    return [group[0] for group in grouped_sequences]


def get_summary(outputs):
    """

    Parameters
    ----------
    outputs: A dict of lists, each element of the list corresponding to one item.
             For example: {'id': [1,2,3], 'val': [72, 42, 32]}

    Returns
    -------
    A list of dicts of individual items.
    For example: [{'id': 1, 'val': 72}, {'id': 2, 'val': 42}, {'id': 3, 'val': 32}]

    """
    keys = outputs.keys()
    values = [outputs[key] for key in keys]
    items = [dict(zip(keys, item_vals)) for item_vals in zip(*values)]
    return items


def write_outputs(exp_dir, summary):
    with gzip.open(exp_dir, "a+") as fp:
        json_writer = jsonlines.Writer(fp)
        json_writer.write_all(summary)


def read_outputs(exp_dir, resample_exp=False):
    if not resample_exp:
        return read_outputs_standard(exp_dir)
    else:
        return read_outputs_resample_exp(exp_dir)


def read_outputs_standard(exp_dir):
    dataset_folder = os.path.join(exp_dir, "testing_output")

    items_dict = defaultdict(dict)
    for filename in os.listdir(dataset_folder):
        if not filename.endswith(".jsonl.gz"):
            continue

        input_file_path = os.path.join(dataset_folder, filename)
        with gzip.open(input_file_path, "r+") as fp:
            reader = jsonlines.Reader(fp)
            for element in reader:
                assert "id" in element
                items_dict[element["id"]].update(element)

    items = [items_dict[idx] for idx in sorted(items_dict.keys())]
    return items


def read_outputs_resample_exp(exp_dir):
    dataset_folder = os.path.join(exp_dir, "testing_output")

    items_dict = defaultdict(lambda: defaultdict(dict))
    for filename in os.listdir(dataset_folder):
        if not (filename.startswith("testing_output") and filename.endswith(".jsonl.gz")):
            continue

        input_file_path = os.path.join(dataset_folder, filename)
        with gzip.open(input_file_path, "r+") as fp:
            reader = jsonlines.Reader(fp)
            for element in reader:
                assert "id" in element
                assert "seed" in element
                items_dict[element["id"]][element["seed"]].update(element)

    items = [items_dict[idx][seed] for idx in sorted(items_dict.keys()) for seed in sorted(items_dict[idx].keys())]
    return items


def read_results(exp_dir):
    results_path = os.path.join(exp_dir, "results.json")
    if os.path.isfile(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    return results


def write_results(exp_dir, results):
    results_path = os.path.join(exp_dir, "results.json")
    with open(results_path, "w") as outfile:
        json.dump(results, outfile)


def select_indices_on_quantiles(data, num_qs, is_data_sorted=False):
    """
    Returns the set of indices that are on the boundaries of the desired number of quantiles.
    E.g. for num_qs=4
    It will return the 0.0, 0.25, 0.5, 0.75, 1. quantile.

    Parameters
    ----------
    data : list or numpy array of numbers
    num_qs : number of quantiles to keep
    is_data_sorted : a flag for whether the data is sorted

    Returns
    -------
    The
    """
    if isinstance(data, list):
        data = np.array(data)

    if not is_data_sorted:
        sorted_indices = np.argsort(data)
        data = data[sorted_indices]

    assert data.ndim == 1, "The `data` should be a one dimensional array"

    qs = np.array(range(1, data.shape[0] + 1)) / data.shape[0]
    qs_to_keep = np.linspace(0, 1, num_qs + 1)
    indices_to_keep = np.searchsorted(qs, qs_to_keep, side="left")

    if is_data_sorted:
        return indices_to_keep

    return sorted_indices[indices_to_keep]


def get_temp_exp_dir(work_dir, wandb_run_path):
    return os.path.join(work_dir, "data/_temp", wandb_run_path)


@rank_zero_only
def upload_outputs_to_wandb(hparams_to_log):
    output_files = os.listdir("testing_output")
    output_files = [os.path.join("testing_output", f) for f in output_files]
    wandb.config["output_files"] = output_files
    wandb.config.update(hparams_to_log, allow_val_change=True)
    wandb.save("testing_output/*", base_path=".", policy="end")
    wandb.finish()

@rank_zero_only
def log_old_outputs(outputs, log_to_wandb=True):
    file_name = "testing_output/testing_output_old.jsonl.gz"
    Path("testing_output").mkdir(exist_ok=True)
    write_outputs(file_name, outputs)
    if log_to_wandb:
        wandb.save("testing_output/*", base_path=".", policy="end")


def restore_outputs_from_wandb(wandb_run_path, exp_dir, mode=None):
    wapi = wandb.Api()
    wrun = wapi.run(wandb_run_path)
    for file in wrun.config["output_files"]:
        if not os.path.isfile(os.path.join(exp_dir, file)):
            wandb.restore(file, run_path=wandb_run_path, root=exp_dir)

    if mode != "resample" and mode != "visualise":
        results_file = "results.json"
        if not os.path.isfile(os.path.join(exp_dir, results_file)):
            try:
                wandb.restore(results_file, run_path=wandb_run_path, root=exp_dir)
            except ValueError:
                pass

    if mode == "visualise":
        results_file = "results.json"
        if not os.path.isfile(os.path.join(exp_dir, results_file)):
            wandb.restore(results_file, run_path=wandb_run_path, root=exp_dir)

        input_stats_file = "testing_output/resample_input_stats.jsonl.gz"
        if not os.path.isfile(os.path.join(exp_dir, input_stats_file)):
            try:
                wandb.restore(input_stats_file, run_path=wandb_run_path, root=exp_dir)
            except ValueError:
                pass

# @rank_zero_only
# def upload_outputs_to_wandb(logger):
#     output_files = os.listdir("testing_output")
#     output_files = [os.path.join("testing_output", f) for f in output_files]
#
#     if isinstance(logger, LoggerCollection):
#         loggers = logger
#     else:
#         loggers = [logger]
#
#     for logger in loggers:
#         if isinstance(logger, WandbLogger):
#             logger.config["output_files"] = output_files
#             logger.save("testing_output/*", base_path=".", policy="end")
#             logger.finish()