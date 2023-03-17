from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import numpy as np
import torch

from . import EvaluationModel
from ..trees.lm_as_tree import LanguageModelAsTree


class SolubilityEvaluationModel(EvaluationModel):
    def __init__(self, tree: LanguageModelAsTree, device="cpu"):
        """
        Class for scoring a protein sequence (or a list of textual sequences) with its solubility.
        Parameters
        ----------
        device : number of cuda device integer (e.g. '0', '1', '2' etc)
        
        Parameters
        ----------
        tree: instance of `LanguageModelAsTree`
        device : a string tag (e.g. 'cpu', 'cuda', 'cuda:0', 'cuda:2, etc) or a torch.device object
        """
        
        super().__init__(tree)
        self.tree: LanguageModelAsTree = tree
        self.device = device
        self.model = TextClassificationPipeline(
                model=AutoModelForSequenceClassification.from_pretrained("Rostlab/prot_bert_bfd_membrane"),
                tokenizer=AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd_membrane"), device=self.device)

    def _get_estimated_utility(self, node_id, **kwargs):
        """Estimated utility -- goodness -- of an internal node"""
        return self._get_utility(node_id, **kwargs)

    def _get_utility(self, node_id, **kwargs):
        """Utility of a terminal node"""
        input_ids = node_id
        text = self.tree.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        #text = ' '.join([re.sub(r"[UZOB]", "X", sequence) for sequence in text[i]])
        
        score = self.model(text)
        soluble_score = score[0]['score']
        return 1 - soluble_score # non_solubility

    def reset_device(self, device: torch.device):
        self.device = device
        self.model.device = device
        self.model.model.to(device)
