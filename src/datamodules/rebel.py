import os
import jsonlines
import random
from itertools import islice
from tqdm import tqdm
from typing import List, Set, Union

import src.utils.general as utils
from .abstract import AbstractDataset, AbstractPLDataModule
from .utils import TripletUtils, select_data_to_resample
from ..utils.evaluation import read_outputs


log = utils.get_logger(__name__)


class RebelDataset(AbstractDataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.params = kwargs
        self.data = RebelDataset.from_kilt_dataset(
            data_split=self.params["load_dataset_params"].pop("split"),
            data_dir=self.params["load_dataset_params"].pop("data_dir"),
            debug=self.params.get("debug", False),
            debug_k=self.params.get("debug_k"),
            return_raw_data=False,
            **self.params["load_dataset_params"],
        )

        if self.params.get("resample_exp", False):
            out_ds = RebelOutputDataset(exp_dir=self.params["exp_dir"])
            resample_ids = select_data_to_resample(out_ds, self.params["num_qs"], self.params["testing_output_parent_dir"])
            self.data = [item for item in self.data if item[0] in resample_ids]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "id": self.data[idx][0],
            "src": self.data[idx][1],
            "tgt": self.data[idx][2],
        }

    @staticmethod
    def _process_obj(obj_orig, relations_to_ignore=None):
        _id = obj_orig["id"]
        src = obj_orig["input"]

        if "non_formatted_wikidata_id_output" in obj_orig["meta_obj"]:
            non_formatted_wikidata_id_output = obj_orig["meta_obj"]["non_formatted_wikidata_id_output"]
        else:
            non_formatted_wikidata_id_output = obj_orig["output"][0]["non_formatted_wikidata_id_output"]

        if relations_to_ignore is None:
            tgt = obj_orig["output"][0]["answer"]
            wikidata_id_triples = non_formatted_wikidata_id_output
        else:
            non_formatted_surface_output = obj_orig["output"][0]["non_formatted_surface_output"]

            triples = []
            wikidata_id_triples = []

            for triple, triple_of_ids in zip(non_formatted_surface_output, non_formatted_wikidata_id_output):
                o, r, s = triple

                if r in relations_to_ignore:
                    continue

                triples.append(triple)
                wikidata_id_triples.append(triple_of_ids)

            # print(triples)
            # print(non_formatted_surface_output)
            tgt = TripletUtils.triples_to_output_format(triples)

        return _id, src, tgt, wikidata_id_triples

    @staticmethod
    def read_relation_set(input_file_path):
        with jsonlines.open(input_file_path, "r") as reader:
            relations = [e for e in reader]

        return set(relations)

    @staticmethod
    def _get_num_lines(input_file_path):
        with open(input_file_path) as f:
            lines = sum(1 for _ in f)
        return lines

    @staticmethod
    def _read_data(input_file_path, debug, debug_k):
        num_lines = RebelDataset._get_num_lines(input_file_path)
        if debug:
            num_lines = min(num_lines, 3 * debug_k)  # input lines can be filtered

        with jsonlines.open(input_file_path) as f:
            data = [
                e
                for e in tqdm(
                    islice(f, num_lines), total=num_lines, desc=f"Loading data from {input_file_path}", leave=True
                )
            ]

        return data

    @staticmethod
    def _filter_on_matching_status(data, allowed_matching_status):
        if allowed_matching_status == "title":
            allowed_statuses = set(["title"])
        elif allowed_matching_status == "label":
            allowed_statuses = set(["title", "label"])
        else:
            raise Exception(f"Unexpected matching status `{allowed_matching_status}`")

        if "instance_matching_status" in data[0]["output"][0]:
            return [e for e in data if e["output"][0]["instance_matching_status"] in allowed_statuses]

        return [e for e in data if e["instance_matching_status"] in allowed_statuses]

    @staticmethod
    def _filter_on_relations_drop(data, relations_to_drop):
        filtered_data = []
        for e in data:
            to_drop = False
            for triple in e["output"][0]["non_formatted_surface_output"]:
                if triple[1] in relations_to_drop:
                    to_drop = True

            if to_drop:
                continue

            filtered_data.append(e)

        return filtered_data

    @staticmethod
    def _filter_on_relations_keep(data, relations_to_keep):
        filtered_data = []
        for e in data:
            to_drop = False
            for triple in e["output"][0]["non_formatted_surface_output"]:
                if triple[1] not in relations_to_keep:
                    to_drop = True

            if to_drop:
                continue

            filtered_data.append(e)

        return filtered_data

    @classmethod
    def from_kilt_dataset(cls, data_split, data_dir, debug, debug_k=None, **kwargs):
        # if data_dir is None:
        #     data_dir = os.path.join(config.DATA_DIR, "rebel")

        input_file = f"en_{data_split}.jsonl"
        input_file_path = os.path.join(data_dir, input_file)

        raw_data = cls._read_data(input_file_path, debug, debug_k)

        if kwargs.get("matching_status", False):
            raw_data = cls._filter_on_matching_status(raw_data, kwargs["matching_status"])

        if kwargs.get("relations_to_drop", False):
            log.info(f"Relations in: `{kwargs['relations_to_drop']}` are dropped")
            relations_to_drop = RebelDataset.read_relation_set(kwargs["relations_to_drop"])
            raw_data = cls._filter_on_relations_drop(raw_data, relations_to_drop)

        if kwargs.get("relations_to_keep", False):
            relations_to_keep = RebelDataset.read_relation_set(kwargs["relations_to_keep"])
            raw_data = cls._filter_on_relations_keep(raw_data, relations_to_keep)

        relations_to_ignore = None

        if "relations_not_to_ignore" in kwargs:
            assert "relations_to_ignore" not in kwargs
            log.info(f"All except for the relations in `{kwargs['relations_not_to_ignore']}` are ignored")

            if isinstance(kwargs["relations_not_to_ignore"], set):
                relations_not_to_ignore = kwargs["relations_not_to_ignore"]
            else:
                relations_not_to_ignore = RebelDataset.read_relation_set(kwargs["relations_not_to_ignore"])

            all_relations = set(
                [
                    triple[1]
                    for raw_sample in raw_data
                    for triple in raw_sample["output"][0]["non_formatted_surface_output"]
                ]
            )

            relations_to_ignore = all_relations - relations_not_to_ignore

        if "relations_to_ignore" in kwargs:
            log.info(f"Relations in: `{kwargs['relations_to_ignore']}` are ignored")
            if isinstance(kwargs["relations_to_ignore"], set):
                relations_to_ignore = kwargs["relations_to_ignore"]
            else:
                relations_to_ignore = RebelDataset.read_relation_set(kwargs["relations_to_ignore"])

        data = [cls._process_obj(obj, relations_to_ignore) for obj in raw_data]

        # filter out any samples that are empty as a consequence of ignored triples
        if relations_to_ignore is not None:
            idx_to_keep = [i for i, s in enumerate(data) if s[2] != ""]
            data = [data[i] for i in idx_to_keep]
            raw_data = [raw_data[i] for i in idx_to_keep]

        if debug:
            data = data[:debug_k]
            raw_data = raw_data[:debug_k]

        # dataset = cls(data, **kwargs)
        # dataset.data_split = data_split

        if kwargs.get("return_raw_data"):
            return raw_data, data
        else:
            return data


class RebelDataModule(AbstractPLDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RebelOutputDataset(AbstractDataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.params = kwargs
        self.data = read_outputs(self.params["exp_dir"], self.params.get("resample_exp", False))
        assert len(self.data) > 0, "Empty output file"

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

    @staticmethod
    def get_predictions(item, key="prediction", top_pred_only=True):
        preds = item[key]

        if top_pred_only and not isinstance(preds, str):
            return preds[0]

        return preds

    @staticmethod
    def get_targets(item, key="target", wrap_in_list=False):
        tgts = item[key]

        if wrap_in_list and not isinstance(tgts, list):
            return [tgts]

        return tgts

    @staticmethod
    def get_text_triples(text, verbose=False, return_set=True) -> Union[Set[tuple], List[tuple]]:
        return TripletUtils.convert_text_sequence_to_text_triples(text, verbose, return_set)
