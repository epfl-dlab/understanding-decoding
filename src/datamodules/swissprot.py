import random

from datasets import load_dataset

from .abstract import AbstractPLDataModule, AbstractDataset
from ..utils.evaluation import read_outputs


class ProteinDataset(AbstractDataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.params = kwargs
        raw_data = load_dataset(path="lightonai/SwissProt-EC-leaf")
        raw_data = raw_data[self.params['split']]
        length = self.params["prompt_length"]
        debug_k = self.params["debug_k"] if self.params.get("debug", False) else None
        self.data = self._process_data(raw_data, length, debug_k)
        
        if self.params["debug"]:
            self.data = self.data[: self.params["debug_k"]]

        for idx, sample in enumerate(self.data):
            sample["id"] = idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _process_data(self, raw_data, length, debug_k=None):
        num_of_instances = debug_k if debug_k is not None else len(raw_data["seq"])
        return [{"id": i, "input": sample[:length]} for i, sample in enumerate(raw_data["seq"][:num_of_instances])]

class ProteinDataModule(AbstractPLDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

# ToDo [not now]: Get an abstract output dataset
class ProteinOutputDataset(AbstractDataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.params = kwargs
        self.data = read_outputs(self.params['exp_dir'])
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
        bootstrap_ids = random.choices(range(num_datapoints), k=num_datapoints)

        resampled_data = [data[i] for i in bootstrap_ids]
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
