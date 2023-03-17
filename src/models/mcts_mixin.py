from contextlib import ExitStack
from typing import Tuple, Iterable, List

import numpy as np
import torch
from jax import numpy as jnp
from jax.experimental import host_callback as hcb

import mctx
from mctx._src import base as mctx_base
from transformers.generation_utils import GenerationMixin, MCTSOutput


class GenerationMixinWithGenericMCTSSupport(GenerationMixin):
    def mcts_root_fn(
        self,
        input_ids,
        attention_mask,
        logits_processor,
        stopping_criteria,
        topk_actions,
        value_model,
        mcts_debug_prints,
        use_cache,
        fp16,
        past=None,
        max_length=None,
        **kwargs,
    ) -> mctx_base.RootFnOutput:
        self.value_model = value_model

        if self.config.is_encoder_decoder:
            assert "encoder_outputs" in kwargs

        model_inputs = self.prepare_inputs_for_generation(
            input_ids,
            attention_mask=attention_mask,
            past=past,
            use_cache=use_cache,
            **kwargs,
        )
        root_outputs = self(**model_inputs, return_dict=True)
        root_logits = root_outputs.logits[:, -1, :]
        root_logits = logits_processor(input_ids, root_logits).detach().cpu()
        root_likelihood = np.array(1.0)
        root_value = self.value_model.evaluate(node_id=input_ids.tolist(), likelihood=root_likelihood)
        if use_cache:
            root_past_key_values = [[kv.cpu() for kv in layer[:2]] for layer in root_outputs.past_key_values]
        else:
            root_past_key_values = None

        # To numpy
        if isinstance(root_value, torch.Tensor):
            root_value = root_value.cpu().numpy()
        # To 32 bit precision
        root_value = root_value.squeeze().astype(np.float32)

        if topk_actions:
            assert topk_actions > 0
            # Keep only the top k logits
            topk_indices = np.argpartition(root_logits, -topk_actions, axis=-1)[:, -topk_actions:]
            root_logits = root_logits[0, topk_indices]

        num_actions = topk_actions if topk_actions else self.config.vocab_size
        assert root_logits.shape[1] == num_actions

        # A tree to cache input_ids, attention_mask, past_key_values, parent token ids, is state terminal, ...
        self.lmcache_tree = {
            "next_id": 1,
            "debug_prints": mcts_debug_prints,
            "attention_mask": attention_mask.cpu(),
            "use_cache": use_cache,
            "topk_actions": topk_actions,
            "num_actions": num_actions,
            "logits_processor": logits_processor,
            "stopping_criteria": stopping_criteria,
            "max_length": max_length if max_length is not None else self.config.max_position_embeddings,
            "fp16": fp16,
            0: {
                "input_ids": input_ids.cpu(),
                "parent": -1,
                "past_key_values": root_past_key_values,
                "is_terminal": False,
                "value": root_value,
                "likelihood": root_likelihood,
                "topk_indices": topk_indices if topk_actions else None,
                "next_token_probs": torch.softmax(root_logits.float(), dim=-1).cpu().numpy(),
            },
        }

        if self.config.is_encoder_decoder:
            self.lmcache_tree["encoder_outputs"] = kwargs["encoder_outputs"]
            self.lmcache_tree["encoder_past_key_values"] = (
                [layer[2:] for layer in root_outputs.past_key_values] if use_cache else None
            )

        return mctx.RootFnOutput(
            prior_logits=jnp.asarray(root_logits),
            value=jnp.array([root_value]),
            embedding=jnp.array([[0]]),
        )

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

        # Stop if EOS was reached
        if current_node["is_terminal"]:
            return True

        # Otherwise call `stopping_criteria`
        stopping_criteria = self.lmcache_tree["stopping_criteria"]
        input_ids = torch.cat(self.trace_lmcache_tree_to_root(self.lmcache_tree, current_node_id, "input_ids"), dim=1)
        scores = None

        return stopping_criteria(input_ids, scores)

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
                    prior_logits=jnp.zeros((action.shape[0], self.lmcache_tree["num_actions"])),
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

        next_token_id: torch.Tensor = torch.tensor(np.array(action[0]), device="cpu").reshape((1, 1))
        topk_actions: np.ndarray = self.lmcache_tree["topk_actions"]
        if topk_actions:
            next_token_id[0, 0] = parent_node["topk_indices"][0, next_token_id[0, 0]]

        node_id: int = self.lmcache_tree["next_id"]
        self.lmcache_tree["next_id"] += 1

        if parent_node["is_terminal"]:
            next_token_id = next_token_id.new_tensor([[self.config.eos_token_id]])

        input_ids = torch.cat(
            self.trace_lmcache_tree_to_root(self.lmcache_tree, parent_node_id, "input_ids") + [next_token_id], dim=1
        )
        is_terminal = next_token_id == self.config.eos_token_id or parent_node["is_terminal"]

        use_cache = self.lmcache_tree["use_cache"]
        if use_cache and not is_terminal:
            past_key_values: Iterable[Iterable[torch.Tensor]] = [
                tuple(torch.cat(kv_layer_trace, dim=2).to(self.device) for kv_layer_trace in zip(*layer_trace))
                for layer_trace in zip(
                    *self.trace_lmcache_tree_to_root(self.lmcache_tree, parent_node_id, "past_key_values")
                )
            ]

            if self.config.is_encoder_decoder:
                encoder_past_key_values = self.lmcache_tree["encoder_past_key_values"]
                assert len(encoder_past_key_values) == len(past_key_values)
                past_key_values = [
                    d_layer + e_layer for d_layer, e_layer in zip(past_key_values, encoder_past_key_values)
                ]
        else:
            past_key_values = None

        if is_terminal:
            terminal_next_token_probs = torch.zeros((1, self.config.vocab_size), device=self.device)
            terminal_next_token_probs[0, self.config.eos_token_id] = 1
            terminal_next_token_logits = torch.log(terminal_next_token_probs)

            logits = terminal_next_token_logits
            next_past_key_values = None
        else:
            logits_processor = self.lmcache_tree["logits_processor"]

            gen_kwargs = {}
            if self.config.is_encoder_decoder:
                gen_kwargs["encoder_outputs"] = self.lmcache_tree["encoder_outputs"]

            input_ids = input_ids.to(self.device)
            model_inputs = self.prepare_inputs_for_generation(
                input_ids,
                past=past_key_values,
                use_cache=use_cache,
                **gen_kwargs,
            )
            assert input_ids.shape[-1] <= self.lmcache_tree["max_length"]

            with torch.cuda.amp.autocast() if self.lmcache_tree["fp16"] else ExitStack():
                model_output = self(**model_inputs, return_dict=True)

            logits: torch.Tensor = model_output.logits[:, -1, :]
            logits = logits_processor(input_ids, logits)
            next_past_key_values: List[List[torch.Tensor]] = (
                [[kv[:, :, -1:, :].cpu() for kv in layer[:2]] for layer in model_output.past_key_values]
                if use_cache
                else None
            )

        logits = logits.detach().cpu()
        if topk_actions:
            # Keep only the top k logits
            topk_indices = np.argpartition(logits, -topk_actions, axis=-1)[:, -topk_actions:]
            logits = logits[0, topk_indices]

        likelihood: np.ndarray = parent_node["likelihood"] * parent_node["next_token_probs"][0, action]

        if parent_node["is_terminal"]:
            value = parent_node["value"]
        else:
            with torch.cuda.amp.autocast() if self.lmcache_tree["fp16"] else ExitStack():
                value = self.value_model.evaluate(node_id=input_ids.tolist(), likelihood=likelihood)
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()

        # To 32 bit precision
        value: np.ndarray = value.squeeze().astype(np.float32)
        logits = logits.float()

        reward = 0.0
        discount = 1.0

        self.lmcache_tree[node_id] = {
            "input_ids": next_token_id,
            "parent": parent_node_id,
            "past_key_values": next_past_key_values,
            "is_terminal": bool(is_terminal),
            "value": value,
            "reward": reward,
            "likelihood": likelihood,
            "topk_indices": topk_indices if topk_actions else None,
            "next_token_probs": torch.softmax(logits, dim=-1).cpu().numpy(),
        }

        if self.lmcache_tree["debug_prints"]:
            print("Recurrent function step with:")
            print(f"  input_ids={input_ids}")
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

    def mcts_finalize(self, policy_output: mctx_base.PolicyOutput, **kwargs) -> MCTSOutput:
        final_embedding = policy_output.search_tree.embeddings[0, policy_output.search_tree.root_index]
        final_node_id = np.array(final_embedding)[0, 0]
        assert final_node_id in self.lmcache_tree

        sequences = self.trace_lmcache_tree_to_root(self.lmcache_tree, final_node_id, "input_ids")
        sequences = torch.cat(sequences, dim=1)
        sequences = sequences.to(self.device)
        return MCTSOutput(sequences, policy_output)
