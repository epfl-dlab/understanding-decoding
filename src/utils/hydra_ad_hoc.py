import hydra
from omegaconf import OmegaConf


def get_config(overrides=[], config_name="train_root.yaml", work_dir="../../", data_dir="../../data/"):
    configs_folder = "../../configs"
    default_overrides = [f"work_dir={work_dir}", f"data_dir={data_dir}"]
    overrides += default_overrides

    with hydra.initialize(version_base="1.2", config_path=configs_folder):
        config = hydra.compose(config_name=config_name, overrides=overrides)

    return config

def compare_configs(config, parameters_to_comapre_keys, parameters_to_comapre_values, tag):
    comparison = True

    for i, parameter in enumerate(parameters_to_comapre_keys):
        if tag == "train" or tag == "run":
            par_config = OmegaConf.select(config, parameter)
            comparison = comparison and (par_config == parameters_to_comapre_values[i])
        else:
            if "model" in parameter:
                par_config = OmegaConf.select(config, parameter.replace("model.", "model.hparams_overrides."))
                comparison = comparison and (par_config == parameters_to_comapre_values[i])

    return comparison