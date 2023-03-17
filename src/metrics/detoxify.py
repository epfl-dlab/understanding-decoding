from typing import List, Union

import numpy as np
import tqdm
from detoxify import Detoxify


class DetoxifyScore:
    def __init__(self, model_type, device="cpu"):
        """
        Class for predicting whether a textual sequence (or a list of textual sequences) is toxic.
        Can initialize 5 different model types from model type:
        - original:
            model trained on data from the Jigsaw Toxic Comment
            Classification Challenge
        - unbiased:
            model trained on data from the Jigsaw Unintended Bias in
            Toxicity Classification Challenge
        - multilingual:
            model trained on data from the Jigsaw Multilingual
            Toxic Comment Classification Challenge
        - original-small:
            lightweight version of the original model
        - unbiased-small:
            lightweight version of the unbiased model
        Parameters
        ----------
        model_type : the name of the model (see above)
        device : a string tag (e.g. 'cpu', 'cuda', 'cuda:0', 'cuda:2, etc) or a torch.device object
        """
        self.device = device
        self.name = self.model_type = model_type
        self.model = Detoxify(model_type=model_type, device=device)

        # used as cache
        self.scores_per_datapoint = None

    def compute(self, text: Union[str, List[str]]) -> List[float]:
        """

        Parameters
        ----------
        text : a string or a list of strings to score

        Returns
        -------
        scores : a list of toxicity scores for each of the textual sequences
        """
        if isinstance(text, str):
            text = [text]

        # TODO batches?
        scores = []
        for t in tqdm.tqdm(text):
            scores.append(self.model.predict([t])["toxicity"][0])
        return scores

    def compute_from_dataset(self, dataset, use_if_cached=False, per_datapoint=False, per_beam=False):
        assert sum([per_datapoint, per_beam]) <= 1

        if per_beam:
            hypotheses = [dataset.get_predictions(item, top_pred_only=False) for item in dataset]
            return [self.compute(hyp) for hyp in hypotheses]

        if not (use_if_cached and self.scores_per_datapoint):
            hypotheses = [dataset.get_predictions(item, top_pred_only=True) for item in dataset]
            self.scores_per_datapoint = self.compute(hypotheses)

        if per_datapoint:
            return self.scores_per_datapoint

        return np.mean(self.scores_per_datapoint)

    # def compute_from_dataset_resample_exp(self, dataset):
    #     hypotheses = [dataset.get_predictions(item, top_pred_only=False) for item in dataset]
    #     return [self.compute(hyp) for hyp in hypotheses]
