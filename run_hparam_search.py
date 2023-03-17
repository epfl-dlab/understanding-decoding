from src.utils import hydra_custom_resolvers
import hydra
from omegaconf import OmegaConf, DictConfig


@hydra.main(version_base="1.2", config_path="configs", config_name="hparam_search_root")
def main(hydra_config: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934

    import src.utils.general as utils
    from src.hparam_search_pipeline import hparam_search

    # Applies optional utilities:
    # - disabling python warnings
    # - prints config
    utils.extras(hydra_config)

    # Run the hyperparameter search model
    hparam_search(hydra_config)


if __name__ == "__main__":
    main()