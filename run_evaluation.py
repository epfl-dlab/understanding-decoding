from src.utils import hydra_custom_resolvers
import hydra
from omegaconf import DictConfig
#import wandb
#wandb.init(project="my-test-project", entity="debjitpaul")

# python run_evaluation.py evaluation=<evaluation_config> ckpt_path=<path_to_ckpt_to_evaluate> run_name=<run_name>


@hydra.main(version_base="1.2", config_path="configs", config_name="evaluate_root")
def main(hydra_config: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934

    import src.utils.general as utils
    from src.evaluation_pipeline import evaluate

    # Applies optional utilities:
    # - disabling python warnings
    # - prints config
    utils.extras(hydra_config)

    # Evaluate model
    evaluate(hydra_config)


if __name__ == "__main__":
    main()
