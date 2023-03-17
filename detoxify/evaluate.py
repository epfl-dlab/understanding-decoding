import argparse
import json

from scipy.special import rel_entr
from scipy.spatial import distance
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance

from sklearn.metrics import mean_squared_error

import numpy as np
import detoxify.src.data_loaders as module_data
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from train import ToxicClassifier

torch.multiprocessing.set_sharing_strategy('file_system')

DOWNLOAD_URL = "https://github.com/unitaryai/detoxify/releases/download/"
MODEL_URLS = {
    "original": DOWNLOAD_URL + "v0.1-alpha/toxic_original-c1212f89.ckpt",
    "unbiased": DOWNLOAD_URL + "v0.3-alpha/toxic_debiased-c7548aa0.ckpt",
    "multilingual": DOWNLOAD_URL + "v0.4-alpha/multilingual_debiased-0b549669.ckpt",
    "original-small": DOWNLOAD_URL + "v0.1.2/original-albert-0e1d6498.ckpt",
    "unbiased-small": DOWNLOAD_URL + "v0.1.2/unbiased-albert-c8519128.ckpt",
}


def test_classifier(config, dataset, checkpoint_path, device="cuda:0"):

    model = ToxicClassifier(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # if 'hyper_parameters' in checkpoint:
    #     checkpoint['config'] = checkpoint['hyper_parameters']['config']
    #     # remove the prefix "model." from every key in the state_dict
    #     checkpoint['state_dict'] = {k[6:]: v for k, v in checkpoint['state_dict'].items()}

    if 'hyper_parameters' not in checkpoint:
        # add the prefix "model." from every key in the state_dict
        checkpoint['state_dict'] = {f"model.{k}": v for k, v in checkpoint['state_dict'].items()}

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)

    def get_instance(module, name, config, *args, **kwargs):
        if "val_fraction" in config[name]["args"]:
            del config[name]["args"]["val_fraction"]
            del config[name]["args"]["create_val_set"]

        return getattr(module, config[name]["type"])(*args, **config[name]["args"], **kwargs)

    config["dataset"]["args"]["test_csv_file"] = dataset

    test_dataset = get_instance(module_data, "dataset", config, train=False)

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        num_workers=1,
        shuffle=False,
    )

    scores = []
    targets = []
    ids = []
    for *items, meta in tqdm(test_data_loader):
        if "multi_target" in meta:
            targets += meta["multi_target"]
        else:
            targets += meta["target"]
        ids += meta["text_id"]
        with torch.no_grad():
            out = model.forward(*items)
            # TODO: save embeddings
            sm = torch.sigmoid(out).cpu().detach().numpy()
        scores.extend(sm)

    scores = np.stack(scores)
    toxicity_scores = scores[:, 0]
    targets = np.stack(targets)
    toxicity_targets = targets[:, 0]
    assert len(toxicity_scores) == len(toxicity_targets)
    metrics = {}
    metrics['mse'] = mean_squared_error(toxicity_targets, toxicity_scores, squared=False)
    metrics['wd'] = wasserstein_distance(toxicity_targets, toxicity_scores)
    metrics['kl'] = sum(rel_entr(toxicity_targets, toxicity_scores))
    metrics['jensen'] = distance.jensenshannon(toxicity_targets, toxicity_scores) ** 2
    metrics['ks'] = {"statistic:": ks_2samp(toxicity_targets, toxicity_scores).statistic, "pvalue": ks_2samp(toxicity_targets, toxicity_scores).pvalue}


    text = "You are an asshole!"
    with torch.no_grad():
        out = model.forward(text)
        # TODO: save embeddings
        sm = torch.sigmoid(out).cpu().detach().numpy()
    print("\n", "~" * 80, f"\nCkpt: {checkpoint_path}", f"\n=== Input (sanity check): {text} \n=== Toxicity score (sanity check): {sm[0][0]} \n=== Metrics: {metrics}\n", "~" * 80)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoint",
        type=str,
        help="path to a saved checkpoint",
    )
    parser.add_argument(
        "-d",
        "--device",
        default='cpu',
        type=str,
        help="device name e.g., 'cpu' or 'cuda' (default cuda:0)",
    )
    parser.add_argument(
        "-t",
        "--test_csv",
        default=None,
        type=str,
        help="path to test dataset",
    )
    parser.add_argument(
        "-l",
        "--local",
        default=True,
        type=str,
        help="if the checkpoint is stored locally",
    )

    args = parser.parse_args()
    if args.config is None:
        loaded = torch.load(args.checkpoint, map_location='cpu')
        if 'config' in loaded:
            config = loaded['config']
        else:
            config = loaded['hyper_parameters']['config']
    else:
        config = json.load(open(args.config))

    if args.device is not None:
        config["gpus"] = args.device

    results = test_classifier(config, args.test_csv, args.checkpoint, args.device)
    test_set_name = args.test_csv.split("/")[-1:][0]

    def np_encoder(object):
        if isinstance(object, np.generic):
            return object.item()

    with open("data/detoxify/results/" + f"results_{args.checkpoint.split('/')[-1]}{test_set_name}.json", "w") as f:
        f.write(json.dumps(results, default=np_encoder))
