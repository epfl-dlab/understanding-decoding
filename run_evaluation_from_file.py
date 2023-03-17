import argparse
import os
import re

import hydra


# python run_evaluation_from_file.py --exp_dir "/home/martin_vm/understanding_decoding/logs/debug/runs/mbart_translation_v1/2022-05-24_11-02-22" --overrides evaluation_from_file=translation


def main(overrides):
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    with hydra.initialize(version_base="1.2", config_path="configs"):
        hydra_config = hydra.compose(config_name="evaluate_from_file_root", overrides=overrides)

    hydra.core.utils.configure_log(hydra_config.logger_config, False)

    # print(OmegaConf.to_yaml(hydra_config, resolve=True))
    #     # Imports should be nested inside @hydra.main to optimize tab completion
    #     # Read more here: https://github.com/facebookresearch/hydra/issues/934
    #
    import src.utils.general as utils
    from src.evaluation_from_file_pipeline import evaluate_from_file

    # Applies optional utilities:
    # - disabling python warnings
    # - prints config
    utils.extras(hydra_config)

    # Run evaluation on an output file
    evaluate_from_file(hydra_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on the output from a file.")
    parser.add_argument("--wandb_run_path", type=str, required=False,
                        help="The path to the wandb run that produced the output_files. "
                             "Specify either the `wandb_run_path` or `exp_dir`.")

    parser.add_argument("--exp_dir", type=str, required=False,
                        help="The path to the experiment log files. "
                             "Specify either the `wandb_run_path` or `exp_dir`.")
    parser.add_argument("--overrides", type=str, nargs="+", help="Space separated overrides")
    args = parser.parse_args()

    if args.overrides is not None:
        overrides = args.overrides
    else:
        overrides = []

    if args.wandb_run_path is not None:
        wandb_run_path = args.wandb_run_path
    else:
        assert args.exp_dir is not None

        # Extract the wandb_run_path (i.e. the unique wandb run identifier) from the correct log file in exp_dir
        wandb_debug_file_path = os.path.join(args.exp_dir, "wandb", "latest-run", "logs", "debug.log")

        KEYWORD = "finishing run"
        with open(wandb_debug_file_path, "r") as file:
            relevant_lines = [line for line in file if re.search(KEYWORD, line)]

        assert len(relevant_lines) == 1
        relevant_line = relevant_lines[0]
        wandb_run_path = relevant_line[relevant_line.find(KEYWORD) + len(KEYWORD):].strip()

    print(wandb_run_path)
    overrides.append(f"wandb_run_path={wandb_run_path}")
    overrides.append(f"work_dir={os.path.abspath(os.getcwd())}")
    overrides.append(f"data_dir={os.path.abspath(os.getcwd())}/data/")
    print(f"overrides={overrides}")
    main(overrides)
