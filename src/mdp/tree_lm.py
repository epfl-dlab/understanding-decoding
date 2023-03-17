from typing import Optional, Tuple, List, Union

import jax.experimental.host_callback as hcb
import jax.numpy as jnp
import numpy as np
import torch

import mctx
from mctx._src import base as mctx_base
from src.mdp.evaluation_models.abstract import EvaluationModel
from src.mdp.trees import Tree
from transformers import PretrainedConfig, PreTrainedModel
from transformers.file_utils import ModelOutput
from transformers.generation_utils import MCTSOutput


class TreeLanguageModelConfig(PretrainedConfig):
    model_type = "treelm"
    EOS_TOKEN_ID = 0
    PAD_TOKEN_ID = 1

    def __init__(self, tree: Tree, decoding=None, max_position_embeddings=None, **kwargs):
        if decoding is not None:
            kwargs.update(**decoding)

        self.n_states = tree.nb_nodes()
        self.vocab_size = tree.nb_nodes() + 2  # add eos and pad
        self.max_position_embeddings = max_position_embeddings

        super().__init__(pad_token_id=self.PAD_TOKEN_ID, eos_token_id=self.EOS_TOKEN_ID, **kwargs)


class TreeLanguageModelOutput(ModelOutput):
    logits: torch.FloatTensor = None


class TreeLanguageModel(PreTrainedModel):
    def __init__(self, config: TreeLanguageModelConfig, tree: Tree, evaluation_model=None, decoding=None):
        # TODO What to do with the decoding argument, should it not be given only to the config for this LM?
        super().__init__(config)

        self.config = config
        self.tree = tree
        self.tokenizer = None

        self.decoding = decoding
        self.evaluation_model = evaluation_model
        # Add a dummy parameter because HuggingFace🤗 determines the model's device by looking at parameters
        self._dummy_parameter = torch.nn.Parameter(torch.randn(()))

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape

        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TreeLanguageModelOutput]:
        probs = TreeLanguageModel.get_next_token_probs(
            tree=self.tree,
            input_ids=input_ids,
            attention_mask=attention_mask,
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
            vocab_size=self.config.vocab_size,
        )
        logits = torch.log(probs)

        if not return_dict:
            return (logits,)

        return TreeLanguageModelOutput(logits=logits)

    @staticmethod
    def token_to_state_id(token):
        return token - 2

    @staticmethod
    def state_id_to_token(state_id):
        return state_id + 2

    @staticmethod
    def get_next_token_probs(
        tree: Tree,
        input_ids: torch.tensor,
        attention_mask: torch.tensor,
        eos_token_id: int,
        pad_token_id: int,
        vocab_size: int,
    ):

        next_token_probs = input_ids.new_zeros((input_ids.shape[0], 1, vocab_size), dtype=torch.float32)
        for batch_idx, (tokens, mask) in enumerate(zip(input_ids, attention_mask)):
            tokens = tokens[mask.type(torch.bool)].tolist()  # remove padding

            last_token = tokens[-1]
            if last_token in {eos_token_id, pad_token_id}:
                continue

            last_state_id = TreeLanguageModel.token_to_state_id(last_token)
            if tree.is_terminal(last_state_id):
                next_token_probs[batch_idx, 0, eos_token_id] = 1.0
                continue

            for child_state_id in tree.get_children(last_state_id):
                child_token = TreeLanguageModel.state_id_to_token(child_state_id)
                next_token_probs[batch_idx, 0, child_token] = tree.get_transition_prob(last_state_id, child_state_id)

        return next_token_probs

    def mcts_root_fn(
        self, input_ids, attention_mask, value_model: EvaluationModel, mcts_debug_prints, **kwargs
    ) -> mctx_base.RootFnOutput:
        if mcts_debug_prints:
            print(f"mcts_root_fn got the following kwargs that it will not use: {kwargs}")

        self.value_model = value_model

        model_inputs = self.prepare_inputs_for_generation(input_ids)
        root_outputs = self(**model_inputs, return_dict=True)
        root_logits = root_outputs.logits[:, -1, :]
        root_likelihood = 1.0
        root_value = self.value_model.evaluate(
            node_id=TreeLanguageModel.token_to_state_id(input_ids[-1]).item(),
            likelihood=root_likelihood,
        )

        # A tree to save all the input_ids, attention_mask, parent token ids, is state terminal, ...
        if mcts_debug_prints:
            print(f"Creating lm cache tree")
        self.lmcache_tree = {
            "next_id": 1,
            "debug_prints": mcts_debug_prints,
            0: {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "parent": -1,
                "is_terminal": False,
                "value": root_value,
                "likelihood": root_likelihood,  # TODO perhaps we want to save the loglikelihood
                "next_token_probs": torch.softmax(root_logits, dim=-1),
            },
        }

        if mcts_debug_prints:
            print(f"Evaluating root and creating RootFnOutput")
        root = mctx.RootFnOutput(
            prior_logits=jnp.asarray(root_logits),
            value=jnp.array([root_value]),
            embedding=jnp.array([[0]]),
        )

        return root

    @staticmethod
    def trace_lmcache_tree_to_root(tree, leaf_node_id, key):
        list = []
        while leaf_node_id != -1:
            list.append(tree[leaf_node_id][key])
            leaf_node_id = tree[leaf_node_id]["parent"]
        list.reverse()
        return list

    def mcts_stopping_criteria_fn(self, embedding: mctx_base.RecurrentState) -> bool:
        return hcb.call(lambda args: self._mcts_stopping_criteria_fn(args), embedding, result_shape=bool)

    def _mcts_stopping_criteria_fn(self, embedding: mctx_base.RecurrentState):
        current_node_id = np.array(embedding)[0, 0]
        current_node = self.lmcache_tree[current_node_id]
        return bool(current_node["is_terminal"])

    def mcts_recurrent_fn(
            self, params, rng_key, action, embedding
    ) -> Tuple[mctx_base.RecurrentFnOutput, mctx_base.RecurrentState]:
        return hcb.call(
            lambda args: self._mcts_recurrent_fn(args),
            (params, rng_key, action, embedding),
            result_shape=(
                mctx.RecurrentFnOutput(
                    reward=jnp.array([0.0]),
                    discount=jnp.array([0.0]),
                    prior_logits=jnp.zeros((action.shape[0], self.config.vocab_size)),
                    value=jnp.array([0.0]),
                ),
                embedding,
            ),
        )

    def _mcts_recurrent_fn(self, arg):
        params, rng_key, action, embedding = arg

        parent_node_id = np.array(embedding)[0, 0]
        assert parent_node_id in self.lmcache_tree
        parent_node = self.lmcache_tree[parent_node_id]

        next_token_id = torch.from_numpy(np.array(action[0])).reshape((1, 1))
        next_attention_mask = torch.Tensor([[1.0]])

        node_id = self.lmcache_tree["next_id"]
        self.lmcache_tree["next_id"] += 1

        if parent_node["is_terminal"]:
            next_token_id = torch.Tensor([[self.config.eos_token_id]])

        input_ids = torch.cat(
            self.trace_lmcache_tree_to_root(self.lmcache_tree, parent_node_id, "input_ids") + [next_token_id], dim=1
        )
        attention_mask = torch.cat(
            self.trace_lmcache_tree_to_root(self.lmcache_tree, parent_node_id, "attention_mask")
            + [next_attention_mask],
            dim=1,
        )
        is_terminal = next_token_id == self.config.eos_token_id or parent_node["is_terminal"]

        if is_terminal:
            terminal_next_token_probs = torch.zeros((1, self.config.vocab_size))
            terminal_next_token_probs[0, self.config.eos_token_id] = 1
            terminal_next_token_logits = torch.log(terminal_next_token_probs)

            logits = terminal_next_token_logits
        else:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids,
                attention_mask,
                cache=True,
            )
            model_output = self(**model_inputs, return_dict=True)
            logits = model_output.logits[:, -1, :]

        likelihood = parent_node["likelihood"] * parent_node["next_token_probs"][0, action]
        if is_terminal:
            # Nota bene:
            #    TreeLanguageModel adds artificially an EOS node to the Tree,
            #    and this new node has no value assigned in the corresponding
            #    EvaluationModel configurations. Thus, we look at the value
            #    of the parent node when a terminal node is reached.
            # Examples:
            #    In [1, 2, 3], we would return the value that the EvaluationModel
            #    assigns to 3. In [1, 2, 3, EOS], we would also return the value
            #    assigned to node 3. In [1, 2, 3, EOS, EOS, EOS] we would return
            #    the value of the penultimate EOS.
            # Language Models:
            #    Language models have a meaningful EOS, so one should returns
            #    it's value. In [1, 2, 3, EOS], the value function of an LM should
            #    return the value assigned to EOS, not to 3. Thus the if would
            #    turn into `if parent_node["is_terminal"]:`. Thanks for sticking
            #    through and have a successful day.
            value = parent_node["value"]
        else:
            value = self.value_model.evaluate(
                node_id=TreeLanguageModel.token_to_state_id(next_token_id).item(),
                likelihood=likelihood,
            )

        reward = 0.0
        discount = 1.0

        self.lmcache_tree[node_id] = {
            "input_ids": next_token_id,
            "attention_mask": next_attention_mask,
            "parent": parent_node_id,
            "is_terminal": is_terminal,
            "value": value,
            "reward": reward,
            "likelihood": likelihood,
            "next_token_probs": torch.softmax(logits, dim=-1),
        }

        if self.lmcache_tree["debug_prints"]:
            print("Recurrent function step with:")
            print(f"  input_ids={input_ids}")
            print(f"  attention_mask={attention_mask}")
            print(f"  value={value}")
            print(f"  reward={reward}")
            print(f"  discount={discount}")
            print()

        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=jnp.array([reward]),
            discount=jnp.array([discount]),
            prior_logits=jnp.asarray(logits),
            value=jnp.array([value]),
        )
        embedding = jnp.asarray([[node_id]])
        return recurrent_fn_output, embedding

    def mcts_finalize(self, policy_output, **kwargs) -> MCTSOutput:
        final_embedding = policy_output.search_tree.embeddings[0, policy_output.search_tree.root_index]
        final_node_id = np.array(final_embedding)[0, 0]
        assert final_node_id in self.lmcache_tree

        sequences = self.trace_lmcache_tree_to_root(self.lmcache_tree, final_node_id, "input_ids")
        sequences = torch.cat(sequences, dim=1)
        return MCTSOutput(sequences, policy_output)
