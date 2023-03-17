import copy
import random

import numpy as np
import scipy.stats as stats

from IPython.display import Image


def max_depth(struct, node, depth=0):
    if node not in struct:
        return depth
    left_first_child = list(struct[node].keys())[0]
    return max_depth(
        struct, left_first_child, depth + 1
    )  # we know that the left_first_child will be instantiated as long as we are not to the bottom


def swap_position(li, pos1, pos2):
    li[pos1], li[pos2] = li[pos2], li[pos1]
    return li


def get_list_with_tau(la, tau, fixed_pos=None, eps=0.1, max_iter=1000):
    if tau < 0:  # require less swapping to work on positive tau: use positive tau on reversed sequence
        la = la[::-1]
        tau = -tau

    lb = copy.deepcopy(la)
    nb_iter = 0
    while np.abs(stats.kendalltau(la, lb)[0] - tau) > eps:
        nb_iter += 1
        if nb_iter > max_iter:
            break

        decreased_range = 1 if fixed_pos else 0
        r = range(len(la) - decreased_range)
        id_a = random.choice(r)
        id_b = random.choice(r)
        if id_a == fixed_pos or id_b == fixed_pos:
            continue
        lb = swap_position(lb, id_a, id_b)

    return lb


# Tree Visualization

def draw_graph(graph):
    return Image(graph.draw(format='png', prog='dot'))


def save_graph(graph, output_file):
    return graph.draw(output_file, prog="dot")
