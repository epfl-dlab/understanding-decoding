import concurrent
import os
import inspect
import random
from typing import List, Dict, Union

import hydra
import numpy as np
import wandb
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from tqdm import tqdm

import src.utils.evaluation as evaluation_utils
import src.utils.general as utils

configs_folder = "../configs"
log = utils.get_logger(__name__)


def get_confidence_interval(trials, confidence_level=0.95):
    # Calculate percentile interval
    alpha = (1 - confidence_level)/2

    interval = alpha, 1-alpha

    def percentile_fun(a, q):
        return np.percentile(a=a, q=q, axis=-1)

    # Calculate confidence interval of statistic
    ci_l = percentile_fun(trials, interval[0]*100)
    ci_u = percentile_fun(trials, interval[1]*100)
    return ci_l, ci_u


def get_score_from_metric(config, metric_alias, seed, device=None):
    # Load metric
    if "device" in config.metric[metric_alias]:
        # update device
        pass
    else:
        metric = hydra.utils.instantiate(config.metric[metric_alias], _recursive_=True)

    # Load dataset
    dataset = hydra.utils.instantiate(config.datamodule, seed=seed, _recursive_=False)

    # Calculate score
    corpus_score = metric.compute_from_dataset(dataset, per_datapoint=False)
    return corpus_score


def get_score_from_scores_per_datapoint(results, seed):
    scores_per_datapoint = results["datapoint"]
    num_datapoints = len(scores_per_datapoint)

    random.seed(seed)
    resampled_data = random.choices(scores_per_datapoint, k=num_datapoints)
    return np.mean(resampled_data)


def get_bootstrap_run_scores(config, results, starting_seed, num_workers=1):
    bootstrap_run_scores = results.get('bootstrap_runs', {})

    run_scores_for_ci = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i in tqdm(range(config.bootstrap_n)):
            seed = starting_seed + i

            if seed in bootstrap_run_scores:
                log.info(f"Score for seed {seed} was already computed.")
                run_scores_for_ci.append(bootstrap_run_scores[seed])
                continue
            elif str(seed) in bootstrap_run_scores:
                log.info(f"Score for seed {seed} was already computed.")
                run_scores_for_ci.append(bootstrap_run_scores[str(seed)])
                continue

            log.info(f"Computing the score for seed {seed}.")
            assert config.bootstrap_run_method in set(["from_metric", "from_scores_per_datapoint"])

            if config.bootstrap_run_method == "from_metric":
                future = executor.submit(get_score_from_metric, config, results['alias'], seed,
                                         f"cuda:{(seed-starting_seed) % num_workers}")
            elif config.bootstrap_run_method == "from_scores_per_datapoint":
                future = executor.submit(get_score_from_scores_per_datapoint, results, seed)
            bootstrap_run_scores[seed] = future.result()
            log.info(f"Score for {seed}: {bootstrap_run_scores[seed]:.2f}.")
            run_scores_for_ci.append(bootstrap_run_scores[seed])

    if len(results.get('bootstrap_runs', {})) < len(bootstrap_run_scores):
        results['bootstrap_runs'] = bootstrap_run_scores

    return run_scores_for_ci


def evaluate_from_file(config: DictConfig) -> Dict[str, Dict[str, Union[str, float, List[float]]]]:
    """Contains the code for running evaluation on an output file.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Dict[str, Dict[str, Union[str, float, List[Float]]]]
        For example: { "BLEU-1__tok-13a_sm-_exp_sv_None": {"alias":"bleu-1","corpus": 12, "datapoint": [9,10,17]}, ...}
    """

    entity, project, run_id = config.wandb_run_path.split("/")
    wandb.init(entity=entity, project=project, resume="must", id=run_id)

    # Set seed for random number generators in PyTorch, Numpy and Python (random)
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    wapi = wandb.Api()
    exp_dir = wapi.run(config.wandb_run_path).config["exp_dir"]
    if os.path.isdir(os.path.join(config.work_dir, exp_dir)):
        config.exp_dir = os.path.join(config.work_dir, exp_dir)
    else:
        config.exp_dir = evaluation_utils.get_temp_exp_dir(config.work_dir, config.wandb_run_path)
        evaluation_utils.restore_outputs_from_wandb(config.wandb_run_path, config.exp_dir)

    log.info(f"Existing results in {config.wandb_run_path}:")
    results = evaluation_utils.read_results(config.exp_dir)
    log.info(results)

    log.info(f"Reading the dataset and the metric: {config.exp_dir}")
    with hydra.initialize(version_base="1.2", config_path=configs_folder):
        dataset = hydra.utils.instantiate(config.datamodule, _recursive_=False)
        metrics = hydra.utils.instantiate(config.metric, _recursive_=True)

    for metric_id, metric in metrics.items():
        has_per_beam = "per_beam" in inspect.signature(metric.compute_from_dataset).parameters.keys()

        if metric.name in results and len(results[metric.name]) > 0:
            if has_per_beam and "beam" not in results[metric.name].keys():
                log.info(f"Add beam scores to metric `{metric.name}`")
                score_per_beam = metric.compute_from_dataset(dataset, per_datapoint=False, per_beam=True)
                results[metric.name]["beam"] = score_per_beam

                log.info(f"Updating results:")
                log.info(results)
                evaluation_utils.write_results(config.exp_dir, results)
            else:
                log.info(f"Metric `{metric.name}` was skipped as it is already present in the results json.")
            continue

        results[metric.name] = {}

        # Note that we assume that all metrics can be computed both sample-wise and on corpus
        log.info(f"Computing the corpus score for metric `{metric.name}`")
        corpus_score = metric.compute_from_dataset(dataset, per_datapoint=False)
        results[metric.name]["corpus"] = corpus_score

        score_per_datapoint = metric.compute_from_dataset(dataset, per_datapoint=True)
        results[metric.name]["datapoint"] = score_per_datapoint

        # if config.datamodule.get("resample_exp", False) and hasattr(metric, "compute_from_dataset_resample_exp"):
        #     results[metric.name]["beam"] = metric.compute_from_dataset_resample_exp(dataset)
        if has_per_beam:
            score_per_beam = metric.compute_from_dataset(dataset, per_datapoint=False, per_beam=True)
            results[metric.name]["beam"] = score_per_beam

        results[metric.name]["alias"] = metric_id

        log.info(f"Writing results:")
        log.info(results)
        evaluation_utils.write_results(config.exp_dir, results)

    if config.get('bootstrap_n', None):
        confidence_level = config.confidence_level

        for metric.name in results:
            confidence_intervals = results[metric.name].get('confidence_intervals', {})
            ci_id = f"{confidence_level}_confidence_level_{config.bootstrap_n}_bootstrap_samples"
            print(results[metric.name]['corpus'])
            print(confidence_intervals)
            if ci_id in confidence_intervals:
                log.info(f"Confidence level {confidence_level} for {metric.name} is already computed.")
                continue

            # Get scores for resampled runs
            bootstrap_run_scores = get_bootstrap_run_scores(config,
                                                            results=results[metric.name],
                                                            starting_seed=123,
                                                            num_workers=config.num_workers)

            # Construct confidence interval
            ci_l, ci_u = get_confidence_interval(np.array(bootstrap_run_scores), confidence_level)
            confidence_intervals[ci_id] = {"low": ci_l, "mean": np.mean(bootstrap_run_scores), "high": ci_u}

            # Update results
            results[metric.name]["confidence_intervals"] = confidence_intervals
            log.info(f"Writing results (including the confidence_intervals)")
            evaluation_utils.write_results(config.exp_dir, results)

    wandb.save(os.path.join(config.exp_dir, "results.json"), base_path=config.exp_dir, policy="end")
    wandb.finish()

    return results
