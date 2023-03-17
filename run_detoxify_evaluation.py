import concurrent.futures
import os
import argparse
import sys


def evaluate_checkpoints(checkpoints, test_csv, num_gpus):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        for i, ckpt_path in enumerate(checkpoints):
            executor.submit(evaluate_checkpoint, ckpt_path, test_csv, f"cuda:{i % num_gpus}")


def evaluate_checkpoint(checkpoint_path, test_csv, device):
    cmd = f"{sys.executable} detoxify/evaluate.py --checkpoint {checkpoint_path} --test_csv {test_csv} --device {device}"
    result_code = os.system(cmd)
    print("*" * 80, "\n Finished running command:", cmd, "\nResult code:", result_code, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate .")
    parser.add_argument(
        "--checkpoints_folder", type=str, required=True,
        help="The directory containing the checkpoints to be evaluated."
    )
    parser.add_argument(
        "--num_gpus", type=int, required=True, help="The directory containing the old checkpoint."
    )
    parser.add_argument(
        "--test_csv", type=str, default="data/detoxify/test_public_expanded.csv", help="The path to the test dataset."
    )

    args = parser.parse_args()
    checkpoints_to_evaluate = [os.path.join(args.checkpoints_folder, path)
                               for path in os.listdir(args.checkpoints_folder)]

    evaluate_checkpoints(checkpoints_to_evaluate, args.test_csv, args.num_gpus)
