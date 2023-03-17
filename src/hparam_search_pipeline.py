from typing import List, Optional

import hydra
import os
import json
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from src.hparam_search.launchers import BaseLauncher
# from src.hparam_search.search import BaseSearch

import src.utils.general as utils
from src.hparam_search.search import BaseSearch

log = utils.get_logger(__name__)


def hparam_search(config: DictConfig) -> Optional[float]:
    """Contains the hyperparameter search pipeline.
    Instantiates all PyTorch Lightning objects from configs.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score useful for hyperparameter optimization.
    """

    # Init loggers
    logger = []
    if config.logger:
        # if config.logger.logger_name == "wandb_standalone":
        import wandb

        run_name_str = config.search.run_name_str
        from sklearn.model_selection import ParameterGrid
        param_configs = ParameterGrid(
            config.search.search_space if isinstance(config.search.search_space, dict) else OmegaConf.to_object(
                config.search.search_space)
        )
        search_param_ranges = {}
        for key in config.search.run_name_vars:
            search_param_ranges[key] = []
            for p_config in param_configs:
                if key in p_config:
                    search_param_ranges[key].append(p_config[key])


        run_name = run_name_str.format(*[search_param_ranges[key] for key in config.search.run_name_vars])
        config.logger.wandb.name = "hp_search=" + run_name

        print(config.logger.wandb)

        # run = wandb.init(**config.logger.wandb)
        run = wandb.init(project=config.logger.wandb.project, entity=config.logger.wandb.entity, name=config.logger.wandb.name)
        logger.append("wandb_standalone")
    else:
        raise Exception(
            f"The specified logger `{config.logger}` is not supported!"
        )


    # launch the search
    cwd = os.getcwd()
    parent_run_idx = os.path.basename(Path(cwd))
    parent_run_name = config.parent_run_name
    # parent_run_name = os.path.basename(Path(cwd).parent)

    launcher: BaseLauncher = hydra.utils.instantiate(config=config.launcher)

    search: BaseSearch = hydra.utils.instantiate(
        config.search, parent_run_name=parent_run_name, parent_run_idx=parent_run_idx
    )

    hparam_search_results = search.run_search(launcher)

    # Take tag of the last script to look for metrics.json in that subfolder
    parent_directory = Path(cwd).parent.absolute()

    results = []

    for run_directory in os.listdir(parent_directory):
    #     # Read all run_idx directories (including the current one)
        run_path = os.path.join(parent_directory, run_directory)  # abs path to parent/run_idx/
        print(f"{run_path}, {os.listdir((run_path))}, {parent_run_name}")

        for experiment_directory in os.listdir(run_path):
            if not experiment_directory.startswith(parent_run_name):
                continue

            experiment_path = os.path.join(run_path, experiment_directory)
            metrics_path = os.path.join(experiment_path, 'results.json')

            if os.path.isfile(metrics_path):
                with open(metrics_path, 'r') as f:
                    current_metrics = json.load(f)
                    results.append({
                        'experiment_path': experiment_path,
                        'experiment_name': os.path.basename(Path(experiment_path)),
                        'metrics': current_metrics
                    })

    # # If metric name and order was provided, sort the results
    if 'metric' in config.search and 'name' in config.search.metric:
        metric_name = config.search.metric.name
        metric_order = 'max'
        if 'order' in config.search.metric:
            metric_order = config.search.metric.order
            assert metric_order == 'min' or metric_order == 'max', \
                'Metric order must be either min or max'

        if 'level' in config.search.metric:
            metric_level = config.search.metric.level
            results.sort(key=lambda r: r['metrics'][metric_name][metric_level], reverse=metric_order == 'max')
        else:
            results.sort(key=lambda r: r['metrics'][metric_name], reverse=metric_order == 'max')

    with open('results.json', 'w') as f:
        f.write(json.dumps(results, indent=4, sort_keys=True))
    #
    # Return the path to best experiment and the best parameter config
    log.info(
        f"Hyperparameter search completed! The best combination of parameters and the path to this run are:\ncombination:{results[0]['experiment_name']}\nexp_path:{results[0]['experiment_path']}")
    # # logging the results of hparam search to loggers (wandb, etc.)
    for lg in logger:
        if lg == "wandb_standalone":
            # log `results` as a table to wandb
            columns = list(results[0].keys())
            table = wandb.Table(columns=columns)
            for result in results:
                row_data = [result[col_name] for col_name in columns]
                table.add_data(*row_data)
            # for res in hparam_search_results:
            #     wandb.log({"metric", res.result()})
            run.log({"HP Search Results": table})

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if lg == "wandb_standalone":
            import wandb
            wandb.finish()