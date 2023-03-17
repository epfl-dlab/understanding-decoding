import argparse
import io
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import wandb
from PIL import Image
from tqdm import tqdm
from itertools import cycle

sns.set_theme(style="whitegrid", font_scale=1.9)
sns.set_style("ticks")
sns.set_palette("deep")

sys.path.append("..")

import src.utils.evaluation as evaluation_utils
from src.datamodules import RTPOutputDataset, WMT14OutputDataset


vgbs_getter = {
    "gpt2_toxicity_vgbs_em_detoxify_oracle_cf_0.25": "Oracle\nRMSE:0.1868",
    "gpt2_toxicity_vgbs_em_detoxify_noisy_oracle_rmse_02242_cf_0.25": "200 steps\nRMSE:0.2242",
    "gpt2_toxicity_vgbs_em_detoxify_noisy_oracle_rmse_02165_cf_0.25": "400 steps\nRMSE:0.2165",
    "gpt2_toxicity_vgbs_em_detoxify_noisy_oracle_rmse_01976_cf_0.25": "1200 steps\nRMSE:0.1976"
}


mcts_getter = {
    "gpt2_toxicity_mcts_em_detoxify_oracle_pb_c_init_0.25": "Oracle\nRMSE:0.1868",
    "gpt2_toxicity_mcts_em_detoxify_noisy_oracle_rmse_01976_pb_c_init_1.25": "1200 steps\nRMSE:0.1976",
    "gpt2_toxicity_mcts_em_detoxify_noisy_oracle_rmse_02165_pb_c_init_1.25": "400 steps\nRMSE:0.2165",
    "gpt2_toxicity_mcts_em_detoxify_noisy_oracle_rmse_02242_pb_c_init_1.25": "200 steps\nRMSE:0.2242"
}


class SiDataGetter:
    def __init__(self, si_list):
        self.si_list = si_list

    def get_si_data(self, difference, ratio):
        return [si.get_summary(difference, ratio) for si in self.si_list]


def get_references(task_name, work_dir):
    if task_name == "cIE":
        wrun_path = "epfl-dlab/understanding-decoding/9lbtlj0l"
    elif task_name == "Translation":
        wrun_path = "epfl-dlab/understanding-decoding/1y6yme2k"
    elif task_name == "Toxicity":
        wrun_path = "epfl-dlab/understanding-decoding/6yf8p424"
    elif task_name == "Protein":
        wrun_path = "epfl-dlab/understanding-decoding/1k2x9th9"

    wapi = wandb.Api()
    wrun = wapi.run(wrun_path)
    experiment_path = wrun.config["exp_dir"]
    experiment_path = x if os.path.isdir(x := os.path.join(work_dir, experiment_path)) \
        else evaluation_utils.get_temp_exp_dir(work_dir, wrun_path)

    print("\n\n" + "-" * 90)
    print(f"Processing references from experiment directory:")
    print(f"  {experiment_path}")
    print("-" * 90 + "\n")

    if "data/_temp" in experiment_path:
        evaluation_utils.restore_outputs_from_wandb(wrun_path, experiment_path, "visualise")
        print("Outputs restored from WANDB instead of reading from disk.")

    references = dict()
    evaluation_results = evaluation_utils.read_results(experiment_path)
    for metric_name, metric_results in evaluation_results.items():
        corpus_metric = metric_results["corpus"]
        if task_name == "Translation":
            corpus_metric /= 100.0
        elif task_name == "Toxicity":
            corpus_metric = 1 - corpus_metric
        references[metric_name] = corpus_metric
        
    return references


def log_plot_to_wandb(wandb_logger, log_key, backend=plt, dpi=200):
    with io.BytesIO() as f:
        backend.savefig(f, dpi=dpi, format="png")
        im = Image.open(f)
        wandb_logger.log({log_key: wandb.Image(im)})


def get_plot(
    df,
    y_legend,
    yerrs,
    marker_style="o",
    marker_size=50,
    marker_linewidth=2.5,
    capsize=0,
    figsize=(7, 4),
    drop_top_frame=True,
    drop_right_frame=True,
    show_plot=False,
    ax_lim=(0, 1),
    reference_line=None,
    task_name=None,
    work_dir="."
):
    c = cycle(["r", "GoldenRod", "violet", "yellow"])
    c_list = ["r", "GoldenRod"]

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlabel("Noise")

    sns.lineplot(data=df, x="x", y="y", hue="Algorithm", ax=ax, palette=c)  # , linestyle=linestyle)

    i = 0
    for algo, df_algo in df.groupby("Algorithm"):
        color = c_list[1 - i]
        i += 1

        if yerrs[algo] is not None:
            ax.errorbar(x=df_algo.x, y=df_algo.y, yerr=yerrs[algo], fmt="none", color="black", capsize=capsize, zorder=10000)

        if marker_size != 0:
            sns.scatterplot(
                x=df_algo.x,
                y=df_algo.y,
                marker=marker_style,
                color=color,
                linewidth=marker_linewidth,
                s=marker_size,
                edgecolor="none",
            )

    if drop_top_frame:
        ax.spines["top"].set_visible(False)

    if drop_right_frame:
        ax.spines["right"].set_visible(False)

    ax.set_ylabel(y_legend)
    ax.set_ylim(ax_lim)

    if reference_line is not None:
        ax.axhline(y=reference_line, linestyle="dashed", color="b", label="BS")
    
    if task_name == "Toxicity":
        plt.legend(loc="lower left", frameon=False, fontsize="small")
        plt.xticks(fontsize="x-small")
    else:
        plt.legend([],[], frameon=False)

    folder = f"{work_dir}/images/fig4"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    path = os.path.join(folder, f"{task_name}")
    plt.savefig(f"{path}.pdf", bbox_inches="tight")

    plt.tight_layout(rect=(0, 0, 1, 0.98))
    if show_plot:
        plt.show()

    plt.close()

    return fig, ax


def main(args):
    noise_getters = {
        "translation_vgbs": lambda experiment_name: 1-float(experiment_name.split("_")[-3]),
        "translation_mcts": lambda experiment_name: 1-float(experiment_name.split("_")[-5]),
        "toxicity_vgbs": lambda experiment_name: vgbs_getter[experiment_name],
        "toxicity_mcts": lambda experiment_name: mcts_getter[experiment_name],
    }


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
    has_targets = None
    for _, (experiment_path, wrun_path, wrun) in enumerate(tqdm(zip(experiment_paths, args.wandb_run_paths, wruns))):
        print("\n\n" + "-" * 90)
        print(f"Processing results from experiment directory:")
        print(f"  {experiment_path}")
        print("-" * 90 + "\n")

        if "data/_temp" in experiment_path:
            evaluation_utils.restore_outputs_from_wandb(wrun_path, experiment_path, "visualise")
            print("Outputs restored from WANDB instead of reading from disk.")
        experiment_name = wrun.config["exp_dir"].split("/")[-2]
        experiment_names.append(experiment_name)
        print(f"  {experiment_name}")

        dataset_args = {"exp_dir": experiment_path}
        if "translation" in experiment_name:
            dataset = WMT14OutputDataset(**dataset_args)
            task_name = "Translation"
        elif "toxicity" in experiment_name:
            dataset = RTPOutputDataset(**dataset_args)
            task_name = "Toxicity"
        else:
            raise NotImplementedError(f"Value model for `{args.dataset}` not implemented.")

        if has_targets is None:
            has_targets = "target_log_likelihood" in dataset[0].keys()
        else:
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
        algo = experiment_name.split("_")[2]
        df_corpus_metrics = pd.DataFrame(
            {
                "Experiment": [experiment_name],
                "Task": task_name,
                "Algorithm": algo.upper()
            }
        )

        evaluation_results = evaluation_utils.read_results(experiment_path)
        for metric_name, metric_results in evaluation_results.items():
            metric_alias = metric_results["alias"]
            metric_names_and_aliases.add((metric_name, metric_alias))
            
            corpus_result = metric_results["corpus"]
            if task_name == "Translation":
                corpus_result /= 100.0
            elif task_name == "Toxicity":
                corpus_result = 1 - corpus_result

            df[metric_name] = metric_results["datapoint"]
            df_corpus_metrics[metric_name] = corpus_result
            for cf_id in metric_results.get('confidence_intervals', []):
                cf_results = np.array([
                    metric_results['confidence_intervals'][cf_id]['low'],
                    metric_results['confidence_intervals'][cf_id]['mean'],
                    metric_results['confidence_intervals'][cf_id]['high']
                ])
                if task_name == "Translation":
                    cf_results /= 100.0
                elif task_name == "Toxicity":
                    cf_results = 1 - cf_results

                df_corpus_metrics[f"{metric_name}_{cf_id}_low"] = cf_results[0]
                df_corpus_metrics[f"{metric_name}_{cf_id}_mean"] = cf_results[1]
                df_corpus_metrics[f"{metric_name}_{cf_id}_high"] = cf_results[2]

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

    df_all: pd.DataFrame = df_all.reset_index().rename(columns={"index": "datapoint_idx"})
    df_all.sort_values(by=["Experiment"], inplace=True)
    df_corpus_metrics_all.sort_values(by=["Experiment"], inplace=True)

    # ToDo: get sd via boostrap resampling
    wandb_logger = wandb.init(
        entity=args.wandb_entity_for_report,
        project=args.wandb_project_for_report,
        id=args.wandb_id_for_report,
        resume="allow"
    )

    # Plot line plot + SDs

    # Log relevant dataframes
    wandb_logger.log({f"Visualisations/DataFrames/dataframe_all": df_all})
    wandb_logger.log(
        {
            f"Visualisations/DataFrames/dataframe_mean_per_experiment_all": df_all.groupby("Experiment")
            .mean()
            .add_prefix("Mean ")
            .reset_index()
        }
    )
    wandb_logger.log({f"Visualisations/DataFrames/dataframe_corpus_metrics_all": df_corpus_metrics_all})

    for metric_name, metric_alias in metric_names_and_aliases:
        references = get_references(task_name, work_dir)
        plot_data = {}

        for row_dict in df_corpus_metrics_all.to_dict(orient="records"):
            experiment_name = row_dict["Experiment"]
            if args.confidence_interval_id is not None:
                metric_score = row_dict[f"{metric_name}_{args.confidence_interval_id}_mean"]
                metric_score_low = row_dict[f"{metric_name}_{args.confidence_interval_id}_low"]
                metric_score_high = row_dict[f"{metric_name}_{args.confidence_interval_id}_high"]
                plot_data[experiment_name] = {'experiment_name': experiment_name,
                                              'metric_score': metric_score,
                                              'metric_score_low': metric_score_low,
                                              'metric_score_high': metric_score_high,
                                              'Algorithm': row_dict["Algorithm"]}
            else:
                metric_score = row_dict[metric_name]
                plot_data[experiment_name] = {'experiment_name': experiment_name, 'metric_score': metric_score}

        plot_data = [plot_data[experiment_name] for experiment_name in experiment_names]
        df_plt = pd.DataFrame(plot_data)

        x = []
        for exp in plot_data:
            if "translation" in exp["experiment_name"]:
                if "vgbs" in exp["experiment_name"]:
                    noise_getter = noise_getters["translation_vgbs"]
                elif "mcts" in exp["experiment_name"]:
                    noise_getter = noise_getters["translation_mcts"]
            elif "toxicity" in exp["experiment_name"]:
                if "vgbs" in exp["experiment_name"]:
                    noise_getter = noise_getters["toxicity_vgbs"]
                elif "mcts" in exp["experiment_name"]:
                    noise_getter = noise_getters["toxicity_mcts"]
            x += [noise_getter(exp["experiment_name"])]

        y = [exp["metric_score"] for exp in plot_data]
        df_plt["x"] = x
        df_plt["y"] = y
        yerrs = dict()
        for algo, df_algo in df_plt.groupby("Algorithm"):
            if args.confidence_interval_id is not None:
                yerr = np.stack([df_algo.metric_score_low, df_algo.metric_score_high])
                yerr = np.abs(yerr - df_algo.y.to_numpy())
            else:
                yerr = None
            yerrs[algo] = yerr

        ax_lim = (0, 1)
        y_legend = "Utility (BLEU)" if task_name == "Translation" else "Utility ($1-$Toxicity)"
        fig, ax = get_plot(
            df_plt, yerrs=yerrs, y_legend=y_legend, show_plot=False, ax_lim=ax_lim,
            reference_line=references[metric_name], task_name=task_name, work_dir=work_dir
        )
        log_plot_to_wandb(wandb_logger, f"{task_name}/{experiment_name}", backend=fig, dpi=200)

    wandb_logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise a list of evaluation experiments.")
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
        "--confidence_interval_id", type=str, default=None,
        help="Needs to be precomputed with `evaluation_from_file.py`. E.g. 0.95_confidence_level_50_bootstrap_samples"
    )

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

