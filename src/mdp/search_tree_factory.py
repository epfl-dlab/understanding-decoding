from collections import defaultdict

from tqdm import tqdm
import numpy as np

from src.mdp import TransitionModel, UtilityValueModel
from src.mdp.utils import get_list_with_tau

import scipy.stats as stats


class SearchTreeFactory:
    def __init__(self, tree_structure_params, transition_model_params, utility_value_params):

        self.ts_params = tree_structure_params
        self.lm_params = transition_model_params
        self.uv_params = utility_value_params

    def sample(self, override_ts_params=None, overide_lm_params=None, override_uv_params=None):
        if override_ts_params:
            struct = self.sample_structure(override_ts_params)
        else:
            struct = self.sample_structure()

        if overide_lm_params:
            likelihood_struct = self.sample_likelihood(struct, overide_lm_params)
        else:
            likelihood_struct = self.sample_likelihood(struct)

        lt = TransitionModel(likelihood_struct)
        if override_uv_params:
            utilities = self.sample_utilities_values(lt, override_uv_params)
        else:
            utilities = self.sample_utilities_values(lt)

        return lt, UtilityValueModel(lt, utilities)

    def expected_number_of_nodes(self):
        depth = self.ts_params["depth"]
        bfs = self.bfs
        nnodes_per_layer = []
        for i in range(depth):
            if i == 0:
                nnode_parenting = 1
            else:
                prev_nnodes = nnodes_per_layer[i - 1]
                prev_bfs = bfs[i - 1]
                nnode_parenting = prev_nnodes - prev_nnodes / prev_bfs
            nnodes_per_layer.append(int(nnode_parenting * bfs[i]))
        return sum([1] + nnodes_per_layer)

    def sample_structure(self, override_ts_params=None):
        if override_ts_params:
            ts_params = override_ts_params
        else:
            ts_params = self.ts_params

        root = 0
        structure = defaultdict(lambda: defaultdict(float))
        structure[root] = defaultdict(float)

        if "decay_branching_factor" in ts_params:
            bfs = [
                max(ts_params["branching_factor"] - i * ts_params["decay_branching_factor"], 3)
                for i in range(ts_params["depth"])
            ]
        else:
            bfs = [ts_params["branching_factor"]] * ts_params["depth"]  ## always same bf at all depths

        self.bfs = bfs

        idx = 1
        level = 0
        nodes_expand = [root]
        with tqdm(total=self.expected_number_of_nodes()) as pbar:
            while level < ts_params["depth"]:
                next_nodes_expand = []
                for n in nodes_expand:
                    children = [(idx + i) for i in range(bfs[level])]
                    idx += len(children)
                    for child in children:
                        structure[n][child] = 0.0
                    pbar.update(len(children))
                    next_nodes_expand.extend(children[:-1])  # the last token is not expanding, it is EOS
                nodes_expand = next_nodes_expand
                level += 1

        # Mark terminal states as leaves
        non_root_states = set([nbs_ids for node_id, nbs in structure.items() for nbs_ids in nbs.keys()])
        non_terminal_states = set(structure.keys())
        self.terminal_states = non_root_states - non_terminal_states

        return structure

    def sample_likelihood(self, structure: dict, override_lm_params=None):
        if override_lm_params:
            lm_params = override_lm_params
        else:
            lm_params = self.lm_params

        assert lm_params["alpha"] > 0, "The concentration parameter alpha must be greater than 0"
        if "alpha_decay" in lm_params:
            assert self.lm_params["alpha-decay"] > 0, ""

        for node, children in tqdm(structure.items()):
            sample = np.random.dirichlet([lm_params["alpha"]] * len(children))
            for child, p in zip(list(children.keys()), sample):
                structure[node][child] = p

        return structure

    def sample_utilities_values(self, lt: TransitionModel, override_uv_params=None):
        if override_uv_params:
            uv_params = override_uv_params
        else:
            uv_params = self.uv_params

        # Sample utilities on terminal nodes based on required alignment with likelihood
        terminal_prob = [(s, lt.marginal_probs[s]) for s in lt.terminal_states]
        sorted_terminal_probs = sorted(terminal_prob, key=lambda item: item[1], reverse=True)
        nodes = [t[0] for t in sorted_terminal_probs]
        probs = [t[1] for t in sorted_terminal_probs]
        utilities_score = get_list_with_tau(probs, uv_params["ul_alignment"])

        utilities = dict(zip(nodes, utilities_score))
        uv_model = UtilityValueModel(lt, utilities)

        # Sample values on inner nodes based on required alignment with downstream utilities
        for node, children in lt.structure.items():
            first_children = list(children.keys())[0]
            if first_children not in lt.structure:
                continue

            expected_u = [(s, uv_model.expected_utilities[s]) for s in children]
            sorted_us = sorted(expected_u, key=lambda item: item[1], reverse=True)
            nodes_u = [t[0] for t in sorted_us]
            probs_u = [t[1] for t in sorted_us]
            fixed_pos = np.argmax(nodes_u)
            values_score = get_list_with_tau(probs_u, uv_params["uu_alignment"], fixed_pos=fixed_pos)

            for k, v in zip(nodes_u, values_score):
                if k not in lt.terminal_states:
                    utilities[k] = v

        return utilities
