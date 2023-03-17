from typing import Union

import numpy as np
import torch
from scipy.stats import truncnorm, bernoulli

import src.utils.general as general_utils

log = general_utils.get_logger(__name__)


def get_np_random_state(rank, debug_info=True):
    seed = 123
    seed += rank

    if debug_info:
        msg = f"Random state: rank -> {rank}, seed -> {seed}"
        log.info(msg)
        print(msg)

    rng_state = np.random.RandomState(seed)
    return rng_state


def truncated_normal(min_v, max_v, mean, sigma, random_state: Union[np.random.RandomState, None]):
    return (
        truncnorm(a=(min_v - mean) / sigma, b=(max_v - mean) / sigma, loc=mean, scale=sigma)
        .rvs(mean.shape[-1], random_state=random_state)
        .squeeze()
    )


def add_noise_binary(value_scores, sigma: float, np_random_state: Union[np.random.RandomState, None]):
    assert value_scores.ndim == 2  # Input must be a batch

    if sigma == 0:
        return value_scores

    dtype = torch.float32
    device = "cpu"
    if isinstance(value_scores, torch.Tensor):
        device = value_scores.device
        dtype = value_scores.dtype
        value_scores = value_scores.cpu()

    min_v = 0.0
    max_v = 1.0
    new_vector = []
    for v in value_scores:
        new_vector.append(truncated_normal(min_v, max_v, v, sigma, random_state=np_random_state))
    new_vector = np.array(new_vector)
    new_vector = new_vector.reshape(value_scores.shape)
    return torch.tensor(new_vector, dtype=dtype, device=device)


def _biased_coin_flip(p: float, random_state: Union[np.random.RandomState, None]):
    return bernoulli(p).rvs(1, random_state=random_state)[0]


def add_noise_continuous(
    value_scores, probability_inversion: float, np_random_state: Union[np.random.RandomState, None]
):
    if probability_inversion == 0:
        return value_scores

    new_vector = np.zeros(value_scores.shape[0])
    indices = list(range(value_scores.shape[0]))
    swapped_indices = []
    for i in indices:
        if i in swapped_indices:
            continue
        if _biased_coin_flip(probability_inversion, random_state=np_random_state) and i != indices[-1]:
            available_indices = [idx for idx in indices[i + 1 :] if idx not in swapped_indices]
            if len(available_indices) == 0:
                continue
            if np_random_state is not None:
                idx_to_swap = np_random_state.choice(available_indices)
            else:
                idx_to_swap = np.random.choice(available_indices)
            new_vector[i] = value_scores[idx_to_swap]
            new_vector[idx_to_swap] = value_scores[i]
            swapped_indices.append(i)
            swapped_indices.append(idx_to_swap)
        else:
            new_vector[i] = value_scores[i]
    return torch.FloatTensor(new_vector)
