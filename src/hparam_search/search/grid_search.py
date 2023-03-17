from sklearn.model_selection import ParameterGrid
from copy import deepcopy
from omegaconf import OmegaConf

import concurrent.futures
import os
from pathlib import Path
from . import BaseSearch
import src.utils.hparam_search_utils as hparam_search_utils
import src.utils.hydra_ad_hoc as hydra_ad_hoc
import src.utils.hydra_custom_resolvers as hydra_custom_resolvers
import src.utils.general as utils

log = utils.get_logger(__name__)


class GridSearch(BaseSearch):
    def __init__(
            self,
            parent_run_name,
            parent_run_idx,
            run_name_str,
            run_name_vars,
            scripts,
            search_space,
            max_workers,
            **kwargs,
    ):
        """
        Grid Search
        Parameters
        ----------
        search_space : should be defined in this format https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html
        max_workers : the maximum number of runs supported in parallel
        """
        super().__init__()

        self.parent_run_name = parent_run_name
        self.parent_run_idx = parent_run_idx
        self.run_name_str = run_name_str
        self.run_name_vars = run_name_vars

        self.max_workers = max_workers

        self.param_configs = ParameterGrid(
            search_space if isinstance(search_space, dict) else OmegaConf.to_object(search_space)
        )

        self.scripts = scripts
        self.hydra = kwargs.get("hydra_dir", "hparam_search")
        self.hydra_log_subdir = kwargs.get("hydra_log_subdir", "hparam_search")

        assert len(self.scripts) == 2, "You should pass both an evalution.py and evaluation_from_file.py scripts"

        self.script_1 = self.scripts[0].script
        script_kwargs = self.scripts[0].script_kwargs
        if script_kwargs is not None:
            self.script_1_kwargs = script_kwargs if isinstance(script_kwargs, dict) else OmegaConf.to_object(
                script_kwargs)
        else:
            self.script_1_kwargs = None
        self.tag_1 = self.scripts[0].tag

        self.script_1_successful_runs_configs = hparam_search_utils.read_configs(self.parent_run_name,
                                                                                 self.parent_run_idx,
                                                                                 tag=self.tag_1)
        # exit()
        self.script_2 = self.scripts[1].script
        self.script_2_kwargs = self.scripts[1].script_kwargs
        self.tag_2 = self.scripts[1].tag

        self.script_2_successful_runs_configs = hparam_search_utils.read_configs(self.parent_run_name,
                                                                                 self.parent_run_idx,
                                                                                 tag=self.tag_2)

        self.both_scripts_unsuccessful_param_configs = self.generate_run_configs(self.param_configs,
                                                                                 self.script_1_successful_runs_configs,
                                                                                 tag=self.tag_1)

        script_2_unsuccessful_param_configs = self.generate_run_configs(self.param_configs,
                                                                        self.script_2_successful_runs_configs,
                                                                        tag=self.tag_1) #used tag_1 because the eval does not create a new folder

        # removing unsuccessful evaluation configs that are going to be run after training with self.both_scripts_unsuccessful_param_configs
        self.script_2_unsuccessful_param_configs = [x for x in script_2_unsuccessful_param_configs if
                                                    x not in self.both_scripts_unsuccessful_param_configs]

        self.results = []

    def successful_config_exists(self, prev_successful_run_configs, parameters_to_comapre_keys,
                                 parameters_to_comapre_values, tag):
        for idx, config in enumerate(prev_successful_run_configs["config"]):
            if hydra_ad_hoc.compare_configs(config, parameters_to_comapre_keys, parameters_to_comapre_values, tag):
                return True, prev_successful_run_configs["run_path"][idx]

        return False, None

    def generate_run_configs(self, param_configs, prev_successful_run_configs, tag):

        log = utils.get_logger(__name__)
        configs = []

        for idx, config in enumerate(param_configs):
            search_params_keys = list(config.keys())
            search_params_values = list(config.values())
            str_params = [f"{k}={v}" for k, v in config.items()]

            successful_config_exists, optional_path = self.successful_config_exists(
                prev_successful_run_configs=prev_successful_run_configs
                , parameters_to_comapre_keys=search_params_keys
                , parameters_to_comapre_values=search_params_values
                , tag=tag)
            if successful_config_exists:
                log.info(
                    f"A successful Run for the following search arguments was found, resulting in this combination getting skipped:\njob:{tag}, search_params:{str_params}\npath to the successful run: {optional_path}")
            else:
                # if a successful config is not found and the tag is not eval, then we can safely assume that the config should be submitted
                # to be rerun.
                log.info(
                    f"No successful training exists for the following search parameters. This search combination will be queued for re-execution.\nfailed search params:{str_params}")
                configs.append(config)

        return configs

    def run_search(self, launcher):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # submit jobs which require both training and evaluation in the following for loop
            # and jobs that only have unfinished evaluation in the next for loop.
            last_idx = 0
            for idx, config in enumerate(self.both_scripts_unsuccessful_param_configs):
                config_keys = list(config.keys())
                cfg = deepcopy(self.script_1_kwargs) if self.script_1_kwargs is not None else dict()
                config["hydra"] = self.hydra
                config["+hydra_log_subdir"] = self.hydra_log_subdir
                config["+parent_run_name"] = os.path.join(self.parent_run_name, str(self.parent_run_idx))
                config["run_name"] = hparam_search_utils.get_trial_name(
                    self.parent_run_name, self.parent_run_idx, idx, self.run_name_str, self.run_name_vars, config
                ) + '_' + self.tag_1

                cfg.update(config)

                eval_cfg = deepcopy(self.script_2_kwargs) if self.script_2_kwargs is not None else dict()
                eval_config = dict()
                for key in config_keys:
                    if key.startswith("model"):
                        eval_config[key.replace("model.", "++model.hparams_overrides.")] = config.pop(key)

                # eval_config["hydra"] = self.hydra
                # eval_config["+hydra_log_subdir"] = self.hydra_log_subdir
                eval_config["+parent_run_name"] = os.path.join(self.parent_run_name, str(self.parent_run_idx))

                # this field is used with hydra custom resolver to retrieve the path of best ckpt
                # run_name is used with wandb, and should not coincide with another run's name.
                # eval_config["run_name"] = config["run_name"].strip("_script_1") + "_script_2"

                # exp_dir_path = os.path.join(os.getcwd(), config["+parent_run_name"], config["run_name"]) # absolute path to 0/1/2
                log_folder = []
                for f in os.getcwd().split('/'):
                    log_folder.append(f)
                    if f == 'logs':
                        break

                exp_dir_path = os.path.join('/'.join(log_folder), config['hydra'], config['+parent_run_name'], config['run_name'])
                print(f"exp_dir_path {exp_dir_path}")
                eval_config["--exp_dir"] = exp_dir_path

                eval_cfg.update(eval_config)

                self.results.append(
                    executor.submit(launcher.run_trial, idx, self.script_1, cfg, self.script_2, eval_cfg))
                last_idx = idx

            for idx_, config in enumerate(self.script_2_unsuccessful_param_configs):
                # note the _ appended to idx, because the idx of runs should not coincide, so unsuccessful evals should be written
                # to new directories, so idx should properly be adjusted (i.e. the correct index is last_idx+idx_)
                eval_cfg = deepcopy(self.script_2_kwargs) if self.script_2_kwargs is not None else dict()
                eval_config = deepcopy(config)
                config_keys = list(config.keys())
                # eval_config["hydra"] = self.hydra
                # eval_config["+hydra_log_subdir"] = self.hydra_log_subdir
                eval_config["+parent_run_name"] = os.path.join(self.parent_run_name, str(self.parent_run_idx))

                # run_name is used with wandb, and should not coincide with another run's name.
                eval_config["--exp_dir"] = hparam_search_utils.get_trial_name(
                    self.parent_run_name, self.parent_run_idx, last_idx + idx_, self.run_name_str,
                    self.run_name_vars, config
                )

                # this should be after get_trial_name because the following will pop parameters required for getting
                # the name, i.e. model.nn_params.d_embedding
                # for key in config_keys:
                #     if "model" in key and "model.checkpoint" not in key:  # chekpoint should be passed as an override, o.w.
                #         # interpolation resolve errors will be raised.
                #         eval_config[key.replace("model.", "++model.hparams_overrides.")] = eval_config.pop(key)

                # exp_dir_path = os.path.join('/'.join(log_folder), config['hydra'], config['+parent_run_name'],
                #                             config['run_name'])
                eval_cfg.update(eval_config)

                # script_2 (evaluation) is passed as script_1 to the launcher, and script_2 is None by default.
                self.results.append(executor.submit(launcher.run_trial, last_idx + idx_, self.script_2, eval_cfg))

            return self.results