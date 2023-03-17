import argparse
import io
import os
from datetime import datetime
from itertools import cycle

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from PIL import Image

sns.set_theme(style="whitegrid", font_scale=1.4)
sns.set_style("ticks")
sns.set_palette("flare")


def log_plot_to_wandb(wandb_logger, log_key, backend=plt, dpi=200):
    with io.BytesIO() as f:
        backend.savefig(f, dpi=dpi, format="png")
        im = Image.open(f)
        wandb_logger.log({log_key: wandb.Image(im)})


def load_list_from_file(path, post_processing_fn=float):
    """File contains floats separated by a new line."""
    values = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            value = post_processing_fn(line)
            values.append(value)
    return values


def is_it_a_yes(string):
    if string.strip().lower() == "yes":
        return 1.0
    if string.strip().lower() == "no":
        return 0.0
    raise ValueError("Could not determine whether or not it was a yes.")


def parsing_sports(targets_path, outputs_path, ll_path, all_probs_path, predictions_path):
    outputs = []
    probs_counter = 0
    with open(outputs_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            results_dict = eval(line)
            assert isinstance(results_dict, dict)
            assert len(results_dict["all_probs"]) > probs_counter
            probs = results_dict["all_probs"][probs_counter:]
            probs_counter += len(probs)
            results_dict["all_probs"] = probs

            outputs.append(results_dict)

    targets = load_list_from_file(targets_path, post_processing_fn=is_it_a_yes)

    assert len(outputs) == len(targets)
    for i in range(len(outputs)):
        outputs[i]["id"] = i
        outputs[i]["target"] = targets[i]

    print("Outputs (and targets) files preprocessed:")
    print(f"   outputs[0]={outputs[0]}")
    print(f"   outputs[-1]={outputs[-1]}")
    print(f"   len(outputs)={len(outputs)}")
    print()

    with open(predictions_path, "w") as f:
        for o in outputs:
            completion = o["completion"].lower().strip()
            pred = None
            if completion.endswith("the answer is yes."):
                pred = 1.0
            elif completion.endswith("the answer is no."):
                pred = 0.0
            elif len(completion) == 0:
                pred = -1.0

            if pred == None:
                pred = completion
                print(o["prompt"])
                print(f'id+1={o["id"] + 1}')
                print(o["completion"])
                print(pred)
                print(o["target"])
                print(f"\n\n\n\n\n")
            f.write(str(pred))
            f.write("\n")

    with open(all_probs_path, "w") as f:
        for o in outputs:
            f.write(str(o["all_probs"]))
            f.write("\n")

    with open(ll_path, "w") as f:
        for o in outputs:
            normalized_ll = -torch.tensor(o["all_probs"]).log().mean().item()
            f.write(str(normalized_ll))
            f.write("\n")


def parsing_zeroshot_sports(targets_path, model_outputs_path, ll_path, all_probs_path, predictions_path, stream=False):
    import jsonlines
    with jsonlines.open(model_outputs_path, "r") as reader:
        outputs = [e for e in reader]
    targets = load_list_from_file(targets_path, post_processing_fn=is_it_a_yes)

    assert len(outputs) > 0
    assert len(outputs) == len(targets)

    for i in range(len(outputs)):
        outputs[i]["id"] = i
        outputs[i]["target"] = targets[i]
        outputs[i]["all_logprobs"] = outputs[i]["logprobs"]["token_logprobs"]
        outputs[i]["all_probs"] = np.exp(np.array(outputs[i]["all_logprobs"])).reshape(-1, 1, 1).tolist()

    print("Outputs (and targets) files preprocessed:")
    print(f"   outputs[0]={outputs[0]}")
    print(f"   outputs[-1]={outputs[-1]}")
    print(f"   len(outputs)={len(outputs)}")
    print()

    with open(predictions_path, "w") as f:
        for o in outputs:
            completion = o["text"].lower().strip()
            pred = None
            if completion.startswith("yes"):
                pred = 1
            elif completion.startswith("no"):
                pred = 0
            elif len(completion) == 0:
                pred = -1

            # Automatic labeling will not work, we annotate the outputs by hand
            if pred == None:
                pred = f"`{completion}`".encode('unicode_escape')
            else:
                pred = f"{pred} " + str(f"`{completion}`".encode('unicode_escape'))
            print(f'id+1={o["id"] + 1}')
            print(o["text"])
            print(pred)
            print(o["target"])
            print(f"\n\n\n\n\n")
            f.write(str(pred))
            f.write("\n")

    with open(all_probs_path, "w") as f:
        for o in outputs:
            f.write(str(o["all_probs"]))
            f.write("\n")

    with open(ll_path, "w") as f:
        for o in outputs:
            normalized_ll = -torch.tensor(o["all_logprobs"]).mean().item()
            f.write(str(normalized_ll))
            f.write("\n")


def load_sports_prompt_dataset_dicts(targets_path, ll_path, predictions_path):
    dicts = []
    targets = load_list_from_file(targets_path, post_processing_fn=is_it_a_yes)
    lls = load_list_from_file(ll_path)
    predictions = load_list_from_file(predictions_path)

    assert len(targets) == len(lls) == len(predictions)

    for id, (t, ll, pred) in enumerate(zip(targets, lls, predictions)):
        d = {
            "id": id,
            "prediction": pred,
            "target": t,
            "score": -ll,
        }
        dicts.append(d)

    matches = 0
    for d in dicts:
        d["match"] = d["target"] == d["prediction"]
        matches += d["match"]
        if d["match"]:
            print(d)
        d["utility"] = float(d["match"])
    print(f"Acc: {matches / len(dicts)}")
    print()
    print()

    return dicts


def load_sports_prompt_dataset(parse=False):
    TARGETS_PATH = "data/prompts_sports/sports_understanding_8shot_stream_s0_targets"

    # RAW
    OUTPUTS_PATH_1 = "data/prompts_sports/sports_understanding_8shot_direct_s0_inputs.preprocessed.out"
    # PROCESSED
    ALL_PROBS_PATH_1 = "data/prompts_sports/sports_understanding_8shot_direct_s0_all_probs"
    PREDICTIONS_PATH_1 = "data/prompts_sports/sports_understanding_8shot_direct_s0_predictions"
    LL_PATH_1 = "data/prompts_sports/sports_understanding_8shot_direct_s0_ll"

    # RAW
    OUTPUTS_PATH_2 = "data/prompts_sports/sports_understanding_8shot_stream_s0_inputs.preprocessed.out"
    # PROCESSED
    ALL_PROBS_PATH_2 = "data/prompts_sports/sports_understanding_8shot_stream_s0_all_probs"
    PREDICTIONS_PATH_2 = "data/prompts_sports/sports_understanding_8shot_stream_s0_predictions"
    LL_PATH_2 = "data/prompts_sports/sports_understanding_8shot_stream_s0_ll"

    # RAW
    OUTPUTS_PATH_3 = "data/prompts_sports/tnlgv2_zero-shot_sports.jsonl"
    # PROCESSED
    ALL_PROBS_PATH_3 = "data/prompts_sports/tnlgv2_zero-shot_sports_all_probs"
    PREDICTIONS_PATH_3 = "data/prompts_sports/tnlgv2_zero-shot_sports_predictions"
    LL_PATH_3 = "data/prompts_sports/tnlgv2_zero-shot_sports_ll"

    if parse:
        parsing_sports(TARGETS_PATH, OUTPUTS_PATH_1, LL_PATH_1, ALL_PROBS_PATH_1, PREDICTIONS_PATH_1)
        parsing_sports(TARGETS_PATH, OUTPUTS_PATH_2, LL_PATH_2, ALL_PROBS_PATH_2, PREDICTIONS_PATH_2)
        parsing_zeroshot_sports(TARGETS_PATH, OUTPUTS_PATH_3, LL_PATH_3, ALL_PROBS_PATH_3, PREDICTIONS_PATH_3)
        print(f"The sports dataset results have been parsed. The predictions need to be curated by hand:")
        print(f"   - {PREDICTIONS_PATH_1}")
        print(f"   - {PREDICTIONS_PATH_2}")
        print(f"   - {PREDICTIONS_PATH_3}")
        return

    std_dicts = load_sports_prompt_dataset_dicts(TARGETS_PATH, LL_PATH_1, PREDICTIONS_PATH_1)
    stream_dicts = load_sports_prompt_dataset_dicts(TARGETS_PATH, LL_PATH_2, PREDICTIONS_PATH_2)
    zeroshot_dicts = load_sports_prompt_dataset_dicts(TARGETS_PATH, LL_PATH_3, PREDICTIONS_PATH_3)

    std_df = pd.DataFrame(std_dicts)
    stream_df = pd.DataFrame(stream_dicts)
    zeroshot_df = pd.DataFrame(zeroshot_dicts)
    zeroshot_df = zeroshot_df.dropna()

    return std_df, stream_df, zeroshot_df


def _plot_join_q_plot(
        df_label_tuples,
        num_bins,
        folder,
        name,
        ax_lim=(0, 1),
        y_label="Utility (Solve Rate)",
        x_label="Percentile",
):
    c = cycle(["r", "GoldenRod", "forestgreen", "yellow"])

    fig = plt.figure(figsize=(5.2, 5))
    ax = fig.gca()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for df, label in df_label_tuples:
        df['bin_id'] = pd.qcut(df.score, num_bins, precision=3, labels=False)
        df['bin_labels'] = pd.qcut(df.score, num_bins, precision=3)

        sns.lineplot(data=df,
                     x='bin_id',
                     y='utility',
                     color=next(c),
                     linestyle="-",
                     linewidth=2,
                     err_style="bars", markers=True, dashes=False, ci=95,
                     err_kws={"fmt": 'o', "linewidth": 2, "capsize": 6},
                     ax=ax, alpha=1, label=label)

        print(f"{label} range per bin")
        print(df[['bin_id', 'bin_labels']].value_counts(ascending=True).reset_index(name='count'))

    positions = np.arange(num_bins)
    labels = ((positions + 1) * 100 / num_bins).astype(int)
    ax.xaxis.set_major_locator(mticker.FixedLocator(positions))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(labels))

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_ylim(ax_lim)
    plt.legend(frameon=False)
    plt.tight_layout(rect=(-0.03, -0.03, 1.05, 1.03))

    log_plot_to_wandb(wandb_logger, f"Prompting/{name}", backend=plt, dpi=args.dpi)

    if not os.path.isdir(folder):
        os.makedirs(folder)
    path = os.path.join(folder, f"{name}")
    plt.savefig(f"{path}.pdf", bbox_inches="tight")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualise a list of evaluation experiments.")
    parser.add_argument("--dpi", type=float, default=200, help="The DPI of the plots logged to wandb.")
    parser.add_argument("--parse_dataset", action="store_true",
                        help="Whether parse to parse the raw dataset outputs into the intermediary format "
                             "we use (i.e., extract `all_probs`, `ll` and `predictions` from the raw `.out` file. "
                             "Note that `predictions` need to be curated by hand before the results can be "
                             "computed/plotted.")
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

    if os.getcwd().endswith("scripts"):
        work_dir = ".."
    else:
        work_dir = "."

    # 1. GET THE DATA
    if args.parse_dataset:
        load_sports_prompt_dataset(parse=True)
        print(f"The raw dataset data has been parsed and needs to be post-processed by hand "
              f"to curate the predictions. Once this is done, re-run the script with the `parse_dataset` flag "
              f"removed. Exiting now.")
        exit()
    else:
        fs_df, cot_df, zeroshot_df = load_sports_prompt_dataset()
        df_label_tuples = [
            (zeroshot_df, "Zero-Shot"),
            (fs_df, "Few-Shot"),
            (cot_df, "Chain-of-Thought"),
        ]

    # 2. GET THA LOGGER
    wandb_logger = wandb.init(
        entity=args.wandb_entity_for_report,
        project=args.wandb_project_for_report,
        id=args.wandb_id_for_report,
        resume="allow"
    )

    # 3. THE PLOT
    _plot_join_q_plot(df_label_tuples, 5, folder=f"{work_dir}/images/fig5", name="Prompting")
    wandb_logger.finish()
    print(wandb_logger.get_url())
