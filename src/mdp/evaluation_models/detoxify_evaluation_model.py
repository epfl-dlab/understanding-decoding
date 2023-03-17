import numpy as np
import torch
from detoxify import Detoxify

from . import EvaluationModel
from ..trees.lm_as_tree import LanguageModelAsTree


class DetoxifyEvaluationModel(EvaluationModel):
    def __init__(self, tree: LanguageModelAsTree, model_type: str, device="cpu"):
        """
        Evaluation model for predicting whether a textual sequence (or a list of textual sequences) is toxic.
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
        tree: instance of `LanguageModelAsTree`
        model_type : the name of the model (see above)
        device : a string tag (e.g. 'cpu', 'cuda', 'cuda:0', 'cuda:2, etc) or a torch.device object
        """
        super().__init__(tree)
        self.tree: LanguageModelAsTree = tree
        self.device = device
        self.name = self.model_type = model_type
        self.model = Detoxify(model_type=model_type, device=device)

    def _get_estimated_utility(self, node_id, **kwargs):
        """Estimated utility -- goodness -- of an internal node"""
        return self._get_utility(node_id, **kwargs)

    def _get_utility(self, node_id, **kwargs):
        """Utility of a terminal node"""
        input_ids = node_id
        text = self.tree.tokenizer.decode(input_ids, skip_special_tokens=True)

        scores = self.model.predict(text)
        return 1 - torch.tensor(scores["toxicity"])

    def reset_device(self, device: torch.device):
        self.device = device
        self.model.device = device
        self.model.model.to(device)
