import torch
from torchmetrics import Metric
from typing import List, Union


class TSF1(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_correct", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("total_predicted", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("total_target", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.name = "triplet_set_f1"

    @staticmethod
    def _process_test_sample(target_triples: set, pred_triples: set):
        num_matched = len(target_triples.intersection(pred_triples))
        num_predicted = len(pred_triples)
        num_target = len(target_triples)

        return num_matched, num_predicted, num_target

    def update(self, preds: list, targets: list):
        """Preds and targets should be lists of sets of triples, where each sets corresponds to a single sample"""
        assert len(preds) == len(targets)

        num_correct = []
        num_predicted = []
        num_target = []

        for t, p in zip(targets, preds):
            n_matched, n_predicted, n_target = TSF1._process_test_sample(t, p)

            num_correct.append(n_matched)
            num_predicted.append(n_predicted)
            num_target.append(n_target)

        num_correct = torch.tensor(num_correct).long()
        num_predicted = torch.tensor(num_predicted).long()
        num_target = torch.tensor(num_target).long()

        self.total_correct += torch.sum(num_correct)
        self.total_predicted += torch.sum(num_predicted)
        self.total_target += torch.sum(num_target)

    @staticmethod
    def _compute(correct, predicted, target, use_tensor=False) -> Union[float, torch.Tensor]:
        if correct == 0 or predicted == 0 or target == 0:
            return torch.tensor(0).float() if use_tensor else 0.0

        correct = correct.float() if use_tensor else float(correct)
        precision = correct / predicted
        recall = correct / target
        f1 = 2 * precision * recall / (precision + recall)

        return f1

    def compute(self):
        if self.total_predicted == 0 or self.total_target == 0 or self.total_correct == 0:
            return torch.tensor(0).float()

        precision = self.total_correct.float() / self.total_predicted
        recall = self.total_correct.float() / self.total_target
        f1 = 2 * precision * recall / (precision + recall)

        return f1

    @staticmethod
    def compute_from_datapoint(target_triples: set, pred_triples: set) -> float:
        n_matched, n_predicted, n_target = TSF1._process_test_sample(target_triples, pred_triples)
        return TSF1._compute(n_matched, n_predicted, n_target)

    def compute_from_dataset(self, dataset, per_datapoint=False, per_beam=False) -> Union[float, list]:
        assert sum([per_datapoint, per_beam]) <= 1
        
        targets = [dataset.get_text_triples(dataset.get_targets(item, wrap_in_list=False)) for item in dataset]
        if per_beam:
            preds = [
                [dataset.get_text_triples(pred) for pred in dataset.get_predictions(item, top_pred_only=False)]
                for item in dataset
            ]
        else:
            preds = [dataset.get_text_triples(dataset.get_predictions(item, top_pred_only=True)) for item in dataset]

        if per_datapoint:
            f1 = [TSF1.compute_from_datapoint(t, p) for t, p in zip(targets, preds)]

        elif per_beam:
            f1 = [[TSF1.compute_from_datapoint(t, beam) for beam in p] for t, p in zip(targets, preds)]
            
        else:
            num_correct = 0
            num_predicted = 0
            num_target = 0

            for t, p in zip(targets, preds):
                n_matched, n_predicted, n_target = TSF1._process_test_sample(t, p)

                num_correct += n_matched
                num_predicted += n_predicted
                num_target += n_target

            f1 = TSF1._compute(num_correct, num_predicted, num_target)

        return f1

    # def compute_from_dataset_resample_exp(self, dataset) -> List[List[float]]:
    #     targets = [dataset.get_text_triples(dataset.get_targets(item)) for item in dataset]
    #     preds = [
    #         [dataset.get_text_triples(pred) for pred in dataset.get_predictions(item, top_pred_only=False)]
    #         for item in dataset
    #     ]
        
    #     f1 = [[TSF1.compute_from_datapoint(tgt, beam) for beam in pred] for tgt, pred in zip(targets, preds)]

    #     return f1
