from typing import Dict, Union

import pytorch_lightning as pl

from transformers import PreTrainedModel, PreTrainedTokenizer
from ..trees import Tree


class LanguageModelAsTree(Tree):
    def __init__(self, lm_model: Union[PreTrainedModel, pl.LightningModule], tokenizer: PreTrainedTokenizer):
        super().__init__()

        if isinstance(lm_model, pl.LightningModule):
            lm_model = lm_model.model

        self.lm_model = lm_model
        self.tokenizer = tokenizer
        self.root_id = tuple()

    def get_children(self, node_id: tuple) -> Dict[tuple, float]:
        raise NotImplemented()  # Not necessary atm

    def get_transition_prob(self, src_id: str, trg_id: str) -> float:
        raise NotImplemented()  # Not necessary atm

    def get_parent(self, node_id: tuple) -> tuple:
        if node_id == self.root_id:
            return self.NO_PARENT

        return node_id[:-1]

    def is_terminal(self, node_id: tuple) -> bool:
        """ToDo: Verify that this assumption is correct for all decoding algorithms. Update it if necessary"""
        return node_id[-1] == self.tokenizer.eos_token_id
