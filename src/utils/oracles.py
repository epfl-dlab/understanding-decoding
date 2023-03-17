import scipy.stats as stats
import scipy.integrate as integrate
import math
import torch


def bart_get_evaluation_model_target_ids(model, tokenizer, target_ids):
    # decoder_input_ids = model.prepare_decoder_input_ids_from_labels(
    #     target_ids
    # )
    eos_token = torch.ones(target_ids.shape[0], 1, dtype=target_ids.dtype,
                           device=target_ids.device) * tokenizer.eos_token_id
    # last_token = torch.where(target_ids[:, -1] == tokenizer.pad_token_id, tokenizer.pad_token_id, eos_token)

    full_target_ids = torch.cat([eos_token, target_ids], dim=-1)
    return full_target_ids


def compute_flip_probability(sigma, k):
    """
    Computes the probability that at least one utility value that was previously 0 will exceed the utility value that was previously 1 when adding noise from a truncated normal distribution.

    Variable names follow the notation from https://en.wikipedia.org/wiki/Truncated_normal_distribution

    Parameters:
    sigma (float): std deviation
    k (int): number of 0-entries

    Returns:
    prob (float): the probability that there will be at least one former 0-entry that is now larger than the former 1-entry
    error (float): the estimated error due to the numerical integration
    """
    normal_rv = stats.norm()

    def pdf_X_1(c):
        ksi = (c - 1) / sigma
        alpha = -1 / sigma
        beta = 0
        Z = normal_rv.cdf(beta) - normal_rv.cdf(alpha)
        return normal_rv.pdf(ksi) / (sigma * Z)

    def cdf_X_0(c):
        ksi = c / sigma
        alpha = 0
        beta = 1 / sigma
        Z = normal_rv.cdf(beta) - normal_rv.cdf(alpha)
        return (normal_rv.cdf(ksi) - normal_rv.cdf(alpha)) / Z

    integrand = lambda c: pdf_X_1(c) * math.pow(cdf_X_0(c), k)

    one_minus_flip_prob, error = integrate.quad(integrand, 0, 1)

    return 1 - one_minus_flip_prob, error


