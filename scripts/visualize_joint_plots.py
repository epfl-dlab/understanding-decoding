import argparse
import io
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import scipy.stats as stats

import numpy as np
import pandas as pd
import wandb
from PIL import Image
from tqdm import tqdm

sys.path.append("..")

import src.utils.evaluation as evaluation_utils
from src.datamodules import RebelOutputDataset, RTPOutputDataset, WMT14OutputDataset, ProteinOutputDataset

from src.datamodules.helper_data_classes import SampleInformation
from scripts.joint_plot import get_plot, get_filter_summary_df


ylabels = {
    "rtp": "Utility ($1-$Toxicity)",
    "wmt14": "Utility (BLEU)",
    "rebel": "Utility (F1)",
    "swissprot": "Utility ($1-$Solubility)"
}


class SiDataGetter:
    def __init__(self, si_list):
        self.si_list = si_list

    def get_si_data(self, difference, ratio):
        return [si.get_summary(difference, ratio) for si in self.si_list]


def log_plot_to_wandb(wandb_logger, log_key, backend=plt, dpi=200):
    with io.BytesIO() as f:
        backend.savefig(f, dpi=dpi, format="png")
        im = Image.open(f)
        wandb_logger.log({log_key: wandb.Image(im)})


def process_utility(results, dataset):
    if dataset in ["rtp", "swissprot"]:
        return 1 - np.asarray(results)
    elif dataset == "wmt14":
        return np.asarray(results) / 100.0
    else:
        return results


def get_si_dataframes(df, metric_names_and_aliases, has_targets):
    si_dataframes = dict()

    for experiment_name, df_exp in df.groupby("Experiment"):
        for metric_name, _ in metric_names_and_aliases:
            if has_targets:
                si_list = [
                    SampleInformation(s, u, (max(ts) if isinstance(ts, list) else ts)) for s, u, ts in zip(
                        df_exp["Best Prediction Normalized LL"],
                        df_exp[metric_name],
                        df_exp["Target Normalized LL"]
                    )
                ]
            else:
                si_list = [
                    SampleInformation(s, u)
                    for s, u in zip(df_exp["Best Prediction Normalized LL"], df_exp[metric_name])
                ]

            if args.plot_correlation:
                for si, c in zip(si_list, df_exp[f"{metric_name}_LL_correlation"]):
                    si.set_correlation(c)

            dataset_obj = SiDataGetter(si_list)
            si_iterable = dataset_obj.get_si_data(difference=has_targets, ratio=False)
            si_df = get_filter_summary_df(si_iterable, only_normalized=has_targets)

        si_dataframes[experiment_name] = si_df

    return si_dataframes


def get_joint_extent(si_dfs, experiment_names, metric_names_and_aliases):
    xlim = list(zip(*[
        (si_dfs[exp]["score"].min(), si_dfs[exp]["score"].max())
        for exp in experiment_names
    ]))
    xmin, xmax = min(xlim[0]), max(xlim[1])

    ymin, ymax = 0, 1

    return xmin, xmax, ymin, ymax


def get_joint_count_standard_kwargs():
    return {"mincnt": 1, "cmap": "flare"}


def get_joint_count_vlim(si_dfs, experiment_names, extent, gridsize):
    joint_kwargs = get_joint_count_standard_kwargs()
    joint_kwargs.update({"extent": extent, "gridsize": gridsize})
    hexbin_values = []

    plt.ioff()
    for exp in experiment_names:
        fig = plt.hexbin(si_dfs[exp]["score"], si_dfs[exp]["utility"], **joint_kwargs)
        hexbin_values.extend(fig.get_array())
    plt.ion()

    return max(hexbin_values)


def get_utility_marginal_max(si_dfs, experiment_names):
    bin_values = []

    plt.ioff()
    plt.xlim(0, 1)
    for exp in experiment_names:
        fig = plt.hist(si_dfs[exp]["utility"], bins=20, range=(0, 1))
        bin_values.extend(fig[0])
    plt.ion()

    return max(bin_values)


def log_joint_plot(
    df,
    wandb_logger,
    log_key,
    has_targets,
    plot_correlation,
    plot_title=None,
    gridsize=10,
    extent=None,
    vlim=None,
    colorbar=True,
    xlabel=None,
    ylabel=None,
    dpi=100,
    show_plot=False,
    save_to_file=None,
    utility_max=None,
    single_line=True
):
    grid_kwargs = {"ratio": 3, "space": 0, "marginal_ticks": True}
    marginal_kwargs = {"edgecolor": "none", "bins": 20, "linewidth": 0.15, "alpha": 0.8, "kde": False}

    if plot_correlation:
        joint_kwargs = {
            "C": df["correlation"],
            "reduce_C_function": np.mean,
            "mincnt": 5,
            "cmap": "coolwarm",
        }
    else:
        joint_kwargs = get_joint_count_standard_kwargs()

    joint_kwargs["gridsize"] = gridsize
    if extent is not None:
        joint_kwargs["extent"] = extent
        marginal_kwargs["extent"] = extent
    if vlim is not None:
        joint_kwargs.update({"vmin": vlim[0], "vmax": vlim[1]})

    get_plot(
        df,
        title=plot_title,
        difference=has_targets,
        only_normalized=has_targets,
        grid_kwargs=grid_kwargs,
        colorbar=colorbar,
        plot_correlation=plot_correlation,
        joint_kwargs=joint_kwargs,
        marginal_kwargs=marginal_kwargs,
        xlabel=xlabel,
        ylabel=ylabel,
        show_plot=show_plot,
        save_to_file=save_to_file,
        close_plot=False,
        utility_max=utility_max,
        single_line=single_line
    )
    log_plot_to_wandb(wandb_logger=wandb_logger, log_key=log_key, dpi=dpi)


def main(args):
    if os.getcwd().endswith("scripts"):
        work_dir = ".."
    else:
        work_dir = "."

    wapi = wandb.Api()
    wruns = [wapi.run(wrp) for wrp in args.wandb_run_paths]
    experiment_paths = [wrun.config["exp_dir"] for wrun in wruns]
    experiment_paths = [
        x if os.path.isdir(x := os.path.join(work_dir, exp_dir)) else evaluation_utils.get_temp_exp_dir(work_dir, wrp)
        for exp_dir, wrp in zip(experiment_paths, args.wandb_run_paths)
    ]

    df_all = pd.DataFrame()
    df_corpus_metrics_all = pd.DataFrame()
    metric_names_and_aliases = set()
    experiment_names = []

    if args.no_target_normalization:
        has_targets = False
    else:
        has_targets = None

    for _, (experiment_path, wrun_path, wrun) in enumerate(tqdm(zip(experiment_paths, args.wandb_run_paths, wruns))):
        print("\n\n" + "-" * 90)
        print(f"Processing results from experiment directory:")
        print(f"  {experiment_path}")
        print("-" * 90 + "\n")

        if "data/_temp" in experiment_path:
            evaluation_utils.restore_outputs_from_wandb(wrun_path, experiment_path, "visualise")
            print("Outputs restored from WANDB instead of reading from disk.")
        if "model/decoding/name" in wrun.config:
            experiment_name = wrun.config["model/decoding/name"]
        elif "model" in wrun.config and "decoding" in wrun.config["model"]:
            experiment_name = wrun.config["model"]["decoding"]["name"]
        else:
            # Hack related to: https://github.com/Lightning-AI/lightning/issues/6106
            # For some reason, our pipeline will end up writing a string to config["model"]
            experiment_name = eval(wrun.config["model"])["decoding"]["name"]
        assert experiment_name not in experiment_names, "Duplicate experiment name"
        experiment_names.append(experiment_name)
        print(f"  {experiment_name}")

        dataset_args = {"exp_dir": experiment_path, "resample_exp": args.resample_exp}
        if args.dataset == "rebel":
            dataset = RebelOutputDataset(**dataset_args)
            task_name = "cIE"
        elif args.dataset == "wmt14":
            dataset = WMT14OutputDataset(**dataset_args)
            task_name = "Translation"
        elif args.dataset == "rtp":
            dataset = RTPOutputDataset(**dataset_args)
            task_name = "Toxicity"
        elif args.dataset == "swissprot":
            dataset = ProteinOutputDataset(**dataset_args)
            task_name = "Protein"
        else:
            raise NotImplementedError(f"`{args.dataset}` output dataset does not exist.")

        if has_targets is None:
            has_targets = "target_log_likelihood" in dataset[0].keys()
        else:
            if not args.no_target_normalization:
                assert has_targets == ("target_log_likelihood" in dataset[0].keys())

        hypotheses_log_likelihood = [
            dataset.get_predictions(item, key="prediction_log_likelihood", top_pred_only=False) for item in dataset
        ]
        best_hypothesis_log_likelihood = [
            dataset.get_predictions(item, key="prediction_log_likelihood", top_pred_only=True) for item in dataset
        ]

        df = pd.DataFrame(
            {
                "Prediction Normalized LL": hypotheses_log_likelihood,
                "Best Prediction Normalized LL": best_hypothesis_log_likelihood,
                "Experiment": experiment_name,
            }
        )
        df_corpus_metrics = pd.DataFrame(
            {
                "Experiment": [experiment_name],
            }
        )

        if args.resample_exp:
            input_ids = [item["id"] for item in dataset]

            df_resample = pd.DataFrame({"input_id": input_ids})

            input_stats_path = os.path.join(experiment_path, "testing_output", "resample_input_stats.jsonl.gz")
            resample_input_stats = pd.read_json(input_stats_path, orient="records", lines=True, compression="gzip")

        evaluation_results = evaluation_utils.read_results(experiment_path)
        for metric_name, metric_results in evaluation_results.items():
            if metric_name in ["triplet_set_precision", "triplet_set_recall"]:
                continue

            metric_alias = metric_results["alias"]
            metric_names_and_aliases.add((metric_name, metric_alias))

            df[metric_name] = process_utility(metric_results["datapoint"], args.dataset)
            df_corpus_metrics[metric_name] = process_utility(metric_results["corpus"], args.dataset)

            if args.plot_correlation and "beam" in metric_results:
                df[f"Beam {metric_name}"] = metric_results["beam"]
                if dataset in ["rtp", "swissprot"]:
                    df[f"Beam {metric_name}"].apply(lambda x: [1 - i for i in x], axis=1)
                elif dataset == "wmt14":
                    df[f"Beam {metric_name}"].apply(lambda x: [i / 100.0 for i in x], axis=1)

                # if args.plot_correlation and not args.resample_exp:
                def get_correlation(row, metric_name):
                    utility = row[f"Beam {metric_name}"][:5]
                    score = row["Prediction Normalized LL"][:5]

                    return stats.kendalltau(utility, score)[0]

                df[f"{metric_name}_LL_correlation"] = df.apply(
                    lambda x: get_correlation(x, metric_name), axis=1
                )

        def target_had_none(target_normalized_ll):
            if target_normalized_ll is None:
                return True
            if None in target_normalized_ll:
                return True
            return False

        if has_targets:
            references_log_likelihood = [
                dataset.get_targets(item, key="target_log_likelihood", wrap_in_list=False) for item in dataset
            ]
            assert len(best_hypothesis_log_likelihood) == len(references_log_likelihood)
            df["Target Normalized LL"] = references_log_likelihood

            target_has_none = df["Target Normalized LL"].apply(target_had_none)
            df = df[~target_has_none]

            df_corpus_metrics["Count of targets with None"] = target_has_none.sum()

        df_corpus_metrics["Count"] = len(df)

        print(f"  df.describe():")
        print(f"  {df.describe()}")
        print(f"  df_corpus_metrics.describe():")
        print(f"  {df_corpus_metrics.describe()}")

        df_all = pd.concat([df_all, df])
        df_corpus_metrics_all = pd.concat([df_corpus_metrics_all, df_corpus_metrics])
        if args.resample_exp:
            df_all = pd.concat([df_all, df_resample], axis=1)

    df_all: pd.DataFrame = df_all.reset_index().rename(columns={"index": "datapoint_idx"})
    df_all.sort_values(by=["Experiment"], inplace=True)
    df_corpus_metrics_all.sort_values(by=["Experiment"], inplace=True)

    wandb_logger = wandb.init(
        entity=args.wandb_entity_for_report,
        project=args.wandb_project_for_report,
        id=args.wandb_id_for_report,
        resume="allow"
    )

    # Log relevant dataframes
    wandb_logger.log({f"{task_name}/DataFrames/dataframe_all": df_all})
    wandb_logger.log(
        {
            f"{task_name}/DataFrames/dataframe_mean_per_experiment_all": df_all.groupby("Experiment")
            .mean()
            .add_prefix("Mean ")
            .reset_index()
        }
    )
    wandb_logger.log({f"{task_name}/DataFrames/dataframe_corpus_metrics_all": df_corpus_metrics_all})
    
    experiment_names = ["greedy_search", "beam_search", "stochastic_beams"]
    si_dfs = get_si_dataframes(df_all, metric_names_and_aliases, has_targets)
    extent = get_joint_extent(si_dfs, experiment_names, metric_names_and_aliases)
    utility_max = get_utility_marginal_max(si_dfs, experiment_names)

    if args.share_colorbar:
        if args.plot_correlation:
            vlim = (-1, 1)
        elif len(experiment_names) > 1 or len(metric_names_and_aliases) > 1:
            vlim = (0, get_joint_count_vlim(si_dfs, experiment_names, extent, args.gridsize))
        else:
            vlim = None
    else:
        vlim = None

    if args.plot_correlation:
        experiment_names = ["beam_search"]

    num_exps = len(experiment_names)

    # Plot joint plots, one per experiment
    for i, experiment_name in enumerate(experiment_names):
        # title = f"{experiment_name.replace('_', ' ')}: {task_name}".title()
        ylabel = ylabels[args.dataset]
        if i == num_exps - 1:
            xlabel = "$\log p(\hat{y}) - \log p(\hat{y}^*)$" if has_targets else "$\log p(\hat{y})$"
        else:
            xlabel = None

        folder = f"{work_dir}/images/fig3" if args.plot_correlation else f"{work_dir}/images/fig2"
        if args.no_target_normalization:
            folder += "_no_target_normalization"

        if not os.path.isdir(folder):
            os.makedirs(folder)

        log_key = f"{task_name}/{experiment_name}"
        log_joint_plot(
            df=si_dfs[experiment_name],
            wandb_logger=wandb_logger,
            log_key=log_key,
            plot_title=False,
            has_targets=has_targets,
            plot_correlation=args.plot_correlation,
            gridsize=args.gridsize,
            extent=extent,
            vlim=vlim,
            dpi=args.dpi,
            show_plot=args.show_plots,
            xlabel=xlabel,
            ylabel=ylabel,
            colorbar=i == 0,
            save_to_file=os.path.join(folder, f"{task_name}_{experiment_name}"),
            utility_max=utility_max,
            single_line=num_exps == 1
        )

    wandb_logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise a list of evaluation experiments.")
    parser.add_argument("--dataset", type=str, required=True, help="The dataset used for the experiments.")
    parser.add_argument(
        "--resample_exp", action="store_true", help="Whether the predictions were resampled during the evaluation."
    )
    parser.add_argument(
        "--wandb_run_paths",
        nargs="+",
        required=True,
        default=[],
        help="List of Wandb run path identifiers of all the experiments to be visualised. They must be"
        " publicly accessible with the wandb public api, but do not need to be yours to be"
        " fetched. If you do not have access to update the logs of these runs, use the flag"
        " --no_wandb_logging_for_individual_runs to not create logs for the runs in this list.",
    )
    parser.add_argument(
        "--plot_correlation", action="store_true", help="Use Kendall's tau correlation as hue of joint plot."
    )
    parser.add_argument(
        "--no_target_normalization", action="store_true", help="Do not apply normalization using the target likelihood."
    )

    parser.add_argument("--gridsize", type=int, default=10, help="The number of hexagons in the x-direction of joint plots.")
    parser.add_argument("--share_colorbar", action="store_true", help="Share colorbar limits among the joint plots.")
    parser.add_argument("--height", type=float, default=4.5, help="Height of one column in sns.FacetGrid.")
    parser.add_argument("--dpi", type=float, default=200, help="The DPI of the plots logged to wandb.")
    parser.add_argument("--show_plots", action="store_true", help="Whether to show the plots using plt.show().")

    parser.add_argument(
        "--wandb_entity_for_report",
        type=str,
        default="user72",
        help="The wandb username or team name where final logs will be reported.",
    )
    parser.add_argument(
        "--wandb_project_for_report",
        type=str,
        default="understanding-decoding",
        help="The name of the wandb project where final logs will be reported.",
    )
    parser.add_argument(
        "--wandb_id_for_report",
        type=str,
        default=f"report_{datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}",
        help="ID of the wandb run to be used (or created) for reporting final logs.",
    )

    args = parser.parse_args()

    main(args)
