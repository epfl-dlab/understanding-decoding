import re
import os

from pathlib import Path
from omegaconf import OmegaConf


def get_trial_name(parent_run_name, parent_run_idx, trial_idx, run_name_str, run_name_vars, config):
    # ToDo: Include some version of script name to accommodate for two scripts within a trial run
    prefix = f"{parent_run_name}_rid-{parent_run_idx}_tid-{trial_idx}"  # Included to facilitate filtering in W&B

    if run_name_str is None or run_name_vars is None:
        return prefix

    assert len(re.findall(r"{[^}]*}", run_name_str)) == len(
        run_name_vars
    ), "There are different number of placeholders and variables in the trial run name specification!"

    for key in run_name_vars:
        assert (
                key in config or f"+{key}" in config or f"++{key}" in config
        ), f"The variable `{key}` to be included in the trial run name is not present in the config"

    run_name = run_name_str.format(
        *[config[key] if key in config else config[f"+{key}"] if f"+{key}" in config else config[f"++{key}"] for key in
          run_name_vars])
    run_name = f"{prefix}___{run_name}"
    return run_name


def paths_to_all_trials_in_run(path_to_hps, run_idx):
    run_name = os.path.basename(Path(path_to_hps))
    path_to_run = os.path.join(path_to_hps, run_idx if isinstance(run_idx, str) else str(run_idx))
    trial_dirs = [
        os.path.join(path_to_run, item_name) for item_name in os.listdir(path_to_run) if item_name.startswith(run_name)
    ]
    return trial_dirs


def are_configs_consistent(expected_config, observed_config, keys=None):
    if keys is None:
        keys = expected_config.keys()

    for key in keys:
        if key not in expected_config:
            return False

        if key not in observed_config:
            return False

        if expected_config[key] != observed_config[key]:
            return False

    return True


# two script arg
def is_experiment_successful(experiment_path, tag):
    experiment_name = os.path.basename(experiment_path)
    if tag == "train":
        # should check that a metrics.json exists as well as a best_ckpt_path.txt
        return os.path.isfile(os.path.join(experiment_path, 'metrics.json')) and os.path.isfile(
            os.path.join(experiment_path, 'best_ckpt_path.txt'))
    elif tag == "eval":
        # should only check whether a metrics.json exists or not.
        return os.path.isfile(os.path.join(experiment_path, 'results.json'))
    elif tag == "run":
        return os.path.exists(os.path.join(experiment_path, 'testing_output'))


def read_run_configs(parent_run_name, run_path, successful, tag):  # gather all successful runs
    # configs = []
    configs = {"config": [], "run_path": []}
    for experiment_directory in os.listdir(run_path):
        # Skip directories which are not experiment directories (and are relevant for the tag)

        if not experiment_directory.startswith(parent_run_name):
            continue

        if not experiment_directory.endswith(tag) and tag != "eval":
            continue

        experiment_path = os.path.join(run_path, experiment_directory)
        # print(f"experiment-path: {experiment_path}")

        # Skip unsuccessful runs if required
        if successful and not is_experiment_successful(experiment_path, tag):
            continue

        conf = OmegaConf.load(os.path.join(experiment_path, '.hydra', 'config.yaml'))
        # configs.append(conf)
        configs["config"].append(conf)
        configs["run_path"].append(experiment_path)

    return configs


def read_configs(parent_run_name, parent_run_idx, successful=True, tag="train"):
    cwd = os.getcwd()  # absolute path to 0/1/2
    parent_directory = Path(cwd).parent  # abs path to parent run name

    # configs = []
    configs = {"config": [], "run_path": []}

    for run_directory in os.listdir(parent_directory):  # over 0,1,2,3
        # Skip invalid directories and the current run directory
        if not run_directory.isnumeric() or run_directory == parent_run_idx:
            continue

        run_path = os.path.join(parent_directory, run_directory)  # abs path to parent/run_idx/
        curr_configs = read_run_configs(parent_run_name, run_path, successful, tag)  # reads all successful runs
        # configs.extend(curr_configs)
        configs["config"].extend(curr_configs["config"])
        configs["run_path"].extend(curr_configs["run_path"])

    return configs