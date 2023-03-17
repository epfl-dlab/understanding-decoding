import os
import random

import jsonlines

from .abstract import AbstractPLDataModule, AbstractDataset
from .utils import select_data_to_resample
from ..utils.evaluation import read_outputs
import src.utils.general as general_utils

log = general_utils.get_logger(__name__)


class RTPDataset(AbstractDataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.params = kwargs

        # Load data
        raw_data = self._load_data()

        # Process data
        self.data = self._process_data(raw_data)

        if self.params.get("subsample", False):
            random.seed(123)
            self.data = random.sample(self.data, len(self.data) // 2)

        if self.params.get("old_exp_dir", False):
            # load the data from the old experiment
            log.info(f"Loading output data from an old experiment at: {self.params['old_exp_dir']}")
            self.old_exp_output_dataset = RTPOutputDataset(exp_dir=self.params["old_exp_dir"])
            log.info(f"The old experiment output data has {len(self.old_exp_output_dataset.data)} samples")
            orig_len = len(self.data)
            old_ids = {dp['id'] for dp in self.old_exp_output_dataset.data}
            self.data = [dp for dp in self.data if dp['id'] not in old_ids]
            log.info(f"The original dataset of {orig_len} was filtered to {len(self.data)} samples")

            log.info(f"Logging old output dataset")
            import src.utils.evaluation as evaluation_utils
            evaluation_utils.log_old_outputs(self.old_exp_output_dataset, log_to_wandb=True)
        else:
            self.old_exp_output_dataset = None

        if self.params.get("resample_exp", False):
            out_ds = RTPOutputDataset(exp_dir=self.params["exp_dir"])
            resample_ids = select_data_to_resample(out_ds, self.params["num_qs"], self.params["testing_output_parent_dir"])
            self.data = [item for item in self.data if item["id"] in resample_ids]

    def _get_input_file_path(self):
        # ToDo: Fix file names
        if self.params["split"] == "train":
            raise Exception("Not supported!")
        elif self.params["split"] == "validation":
            raise Exception("Not supported!")
        elif self.params["split"] == "test":
            return os.path.join(self.params["dataset_data_dir"], "test_10000_prompts.jsonl")
        else:
            raise Exception(f"Unsupported split {self.params['split']}")

    def _load_data(self):
        input_file_path = self._get_input_file_path()
        assert os.path.isfile(input_file_path) and input_file_path.endswith(".jsonl")
        log.info(
            f"Loading dataset: `Real Toxicity Prompts` from {input_file_path} (debug={self.params.get('debug', False)})"
        )

        with jsonlines.open(input_file_path, "r") as reader:
            if not self.params.get("debug", False):
                data = [e for e in reader]
            else:
                data = []
                with jsonlines.open(input_file_path, "r") as reader:
                    for i, e in enumerate(reader):
                        if i >= self.params["debug_k"]:
                            break
                        data.append(e)
        return data

    def _process_data(self, raw_data):
        data = []
        for i, sample in enumerate(raw_data):
            data.append({"id": i, "input": sample["prompt"]["text"]})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class RTPDataModule(AbstractPLDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# ToDo [not now]: Get an abstract output dataset
class RTPOutputDataset(AbstractDataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.params = kwargs
        self.data = read_outputs(self.params["exp_dir"], self.params.get("resample_exp", False))
        assert len(self.data) > 0

        if self.params.get("seed", None):
            self._resample(self.params["seed"])

        self.data.sort(key=lambda x: int(x["id"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _resample(self, seed):
        data = self.data
        num_datapoints = len(data)

        random.seed(seed)
        resampled_data = random.choices(data, k=num_datapoints)
        self.data = resampled_data

    #  Define helper function that will be used by the metric(s)
    @staticmethod
    def get_predictions(item, key="prediction", top_pred_only=True):
        """

        Parameters
        ----------
        item
        top_pred_only

        Returns
        -------
        list of strings if top_pred_only=False, one string otherwise

        """
        preds = item[key]

        if top_pred_only and not isinstance(preds, str):
            return preds[0]

        return preds

    @staticmethod
    def get_targets(item, key=None):
        raise NotImplementedError("The `Real Toxicity Prompts` does not have targets!")
