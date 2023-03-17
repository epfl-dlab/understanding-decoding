import argparse
import sys
import torch

sys.path.append("..")

from src.models.collators import RebelCollator
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert old GenIE checkpoint to new setting.")
    parser.add_argument(
        "--input_ckpt_dir", type=str, required=True, help="The directory containing the old checkpoint."
    )
    parser.add_argument("--output_ckpt_dir", type=str, required=True, help="The directory for the new checkpoint.")
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="genie_genre_r.ckpt",
        help="The full checkpoint name (e.g. genie_genre_r.ckpt).",
    )
    args = parser.parse_args()

    model = torch.load(f"{args.input_ckpt_dir}/{args.ckpt_name}")
    hparams = model["hyper_parameters"]

    hparams["decoding"] = hparams.pop("inference")
    hparams["from_checkpoint"] = True
    hparams["random_initialization"] = False
    hparams.pop("model_name_or_path", None)
    hparams["pretrained_model_name_or_path"] = "martinjosifoski/genie-rw"
    hparams["hf_config"].update(
        {
            "min_length": 0,
            "max_length": 256,
            "early_stopping": False,
            "encoder_no_repeat_ngram_size": 0,
            "no_repeat_ngram_size": 0,
            "temperature": 1.0,
            "length_penalty": 1.0,
            "forced_bos_token_id": 0,
        }
    )
    hparams["tokenizer"].model_max_length = 256

    collator = RebelCollator(
        hparams["tokenizer"],
        max_input_length=hparams["max_input_length"],
        max_output_length=hparams["max_output_length"],
        padding=True,
        truncation=True,
    )
    hparams["collator"] = collator
    Path(args.output_ckpt_dir).mkdir(exist_ok=True, parents=True)
    torch.save(model, f"{args.output_ckpt_dir}/{args.ckpt_name}")
    print("Checkpoint saved at:", f"{args.output_ckpt_dir}/{args.ckpt_name}")
