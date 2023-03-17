import random

from datasets import load_dataset

import src.utils.general as utils
from .abstract import AbstractPLDataModule, AbstractDataset
from .utils import select_data_to_resample
from ..utils.evaluation import read_outputs


log = utils.get_logger(__name__)


class WMT14Dataset(AbstractDataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.params = kwargs
        self.data = load_dataset(path="wmt14", **kwargs["load_dataset_params"])["translation"]

        # get sentences and permute them
        en = [datapoint["en"] for datapoint in self.data]
        fr = [datapoint["fr"] for datapoint in self.data]

        # shuffle sentences
        random.Random(kwargs['seed']).shuffle(en)
        random.Random(kwargs['seed']+1).shuffle(fr)

        assert len(en) == len(fr) and len(en) == len(self.data)

        for i in range(len(en)):
            self.data[i]["noisy_en"] = en[i]
            self.data[i]["noisy_fr"] = fr[i]

        if self.params.get("debug", False):
            self.data = self.data[: self.params["debug_k"]]

        for idx, sample in enumerate(self.data):
            sample["id"] = idx

        if self.params.get("resample_exp", False):
            out_ds = WMT14OutputDataset(exp_dir=self.params["exp_dir"])
            resample_ids = select_data_to_resample(out_ds, self.params["num_qs"], self.params["testing_output_parent_dir"])
            self.data = [item for item in self.data if item["id"] in resample_ids]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class WMT14DataModule(AbstractPLDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# ToDo [not now]: Get an abstract output dataset
class WMT14OutputDataset(AbstractDataset):
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
        key
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
    def get_targets(item, key="target", wrap_in_list=False):
        """

        Parameters
        ----------
        item
        key
        wrap_in_list

        Returns
        -------
        list of strings when wrap_in_list=True, otherwise returns the item['targets'] (a list or a string)

        """
        tgts = item[key]

        if wrap_in_list and not isinstance(tgts, list):
            return [tgts]

        return tgts
