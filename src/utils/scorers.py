import torch

from .score_helpers import get_logits_processor_from_generate, get_forced_token_logits_processors


class SampleScores:
    def __init__(
        self,
        ids,
        untampered_log_probs,
        force_corrected_untampered_log_probs,
        processed_log_probs,
        length_penalty,
        prompt_length=0,
    ):
        self.ids = ids.squeeze(0)
        self.untampered_log_probs = untampered_log_probs.squeeze(0)
        self.force_corrected_untampered_log_probs = force_corrected_untampered_log_probs.squeeze(0)
        self.processed_log_probs = processed_log_probs.squeeze(0)
        self.length_penalty = length_penalty
        self.prompt_length = prompt_length

        assert self.ids.ndim == 1, "Unexpected number of dimensions. Note that batches are not supported"
        assert self.untampered_log_probs.ndim == 2, "Unexpected number of dimensions"
        assert self.processed_log_probs.ndim == 2, "Unexpected number of dimensions"

    @staticmethod
    def _get_scores(ids, log_probs, cumulative, normalized, return_full_log_probs, length_penalty, prompt_length):
        if return_full_log_probs == True:
            return log_probs

        per_token_log_prob = log_probs.gather(dim=-1, index=ids[:, None]).squeeze()
        if not cumulative:
            return per_token_log_prob

        cum_per_token_log_prob = per_token_log_prob.cumsum(0)

        if not normalized:
            return cum_per_token_log_prob
        num_tokens = cum_per_token_log_prob.nelement()
        return cum_per_token_log_prob / (
                (torch.arange(num_tokens).to(cum_per_token_log_prob.device) + 1 + prompt_length) ** length_penalty
        )

    def get_processed_score(
            self, cumulative=True, normalized=True, return_full_log_probs=False, return_final_score_only=True
    ):
        per_token_scores = self._get_scores(
            self.ids,
            self.processed_log_probs,
            cumulative,
            normalized,
            return_full_log_probs,
            self.length_penalty,
            self.prompt_length,
        )
        if return_final_score_only:
            return per_token_scores[-1].item()

        return per_token_scores

    def get_untampered_score(
        self,
        force_corrected=True,
        cumulative=True,
        normalized=True,
        return_full_log_probs=False,
        return_final_score_only=True,
    ):
        if force_corrected:
            log_probs = self.force_corrected_untampered_log_probs
        else:
            log_probs = self.untampered_log_probs
        per_token_scores = self._get_scores(
            self.ids, log_probs, cumulative, normalized, return_full_log_probs, self.length_penalty, self.prompt_length
        )

        if return_final_score_only:
            return per_token_scores[-1].item()

        return per_token_scores


def get_encoder_decoder_scores(hf_model, hf_generation_params, model_kwargs):
    model_output = hf_model(return_dict=True, **model_kwargs)
    logits = model_output.logits
    log_probs = logits.log_softmax(dim=-1)

    hf_generation_params = hf_generation_params.copy()
    # Hack: Because we call the logits processors and warpers with one beam instead of num_beams, we need to make
    #       sure that the warpers get the correct number of beams so that can initialize the `TopKLogitsWarper` and
    #       `TopPLogitsWarper` warpers with `min_tokens_to_keep` correctly, but for the logits processors,
    #       we will pass `num_beams=1` to make sure that the `PrefixConstrainedLogitsProcessor`
    #       processor does not break.
    hf_generation_params["num_beams_warper"] = hf_generation_params.get("num_beams", hf_model.config.num_beams)
    hf_generation_params["num_beams"] = 1
    hf_generation_params["num_beam_groups"] = 1

    prompt_len = 1  # Assume that the first token is the prompt
    assert model_kwargs["decoder_input_ids"][:, 0] == hf_model.config.decoder_start_token_id
    pred_len = model_kwargs["labels"].shape[-1] - prompt_len

    logits_processors = get_logits_processor_from_generate(
        hf_model,
        input_ids=model_kwargs["input_ids"],
        attention_mask=model_kwargs["attention_mask"],
        **hf_generation_params
    )
    forced_token_logits_processors = get_forced_token_logits_processors(logits_processors)

    processed_log_probs = log_probs.clone()
    force_corrected_untampered_log_probs = log_probs.clone()
    for idx in range(prompt_len - 1, pred_len):
        processed_log_probs[:, idx] = logits_processors(
            input_ids=model_kwargs["decoder_input_ids"][:, : 1 + idx], scores=processed_log_probs[:, idx]
        )

        if len(forced_token_logits_processors) > 0:
            force_corrected_untampered_log_probs[:, idx] = forced_token_logits_processors(
                input_ids=model_kwargs["decoder_input_ids"][:, : 1 + idx],
                scores=force_corrected_untampered_log_probs[:, idx],
            )

    labels = model_kwargs["labels"][:, prompt_len:]
    log_probs = log_probs[:, prompt_len - 1: -1, :]
    processed_log_probs = processed_log_probs[:, prompt_len - 1: -1, :]
    force_corrected_untampered_log_probs = force_corrected_untampered_log_probs[:, prompt_len - 1: -1, :]

    scores = SampleScores(
        ids=labels,
        untampered_log_probs=log_probs,
        force_corrected_untampered_log_probs=force_corrected_untampered_log_probs,
        processed_log_probs=processed_log_probs,
        length_penalty=hf_generation_params.length_penalty if "length_penalty" in hf_generation_params else 1.0,
        prompt_length=prompt_len,
    )
    return scores


def get_decoder_only_scores(hf_model, hf_generation_params, prompt_kwargs, model_kwargs, labels):
    assert all([
        "attention_mask" not in model_kwargs
        or model_kwargs["attention_mask"] is None
        or model_kwargs["attention_mask"].sum() == 0
        ,
        "Attention mask not supported. To support it, fix labels."
    ])

    model_output = hf_model(return_dict=True, **model_kwargs)
    logits = model_output.logits
    log_probs = logits.log_softmax(dim=-1)

    hf_generation_params = hf_generation_params.copy()
    # Hack: Because we call the logits processors and warpers with one beam instead of num_beams, we need to make
    #       sure that the warpers get the correct number of beams so that can initialize the `TopKLogitsWarper` and
    #       `TopPLogitsWarper` warpers with `min_tokens_to_keep` correctly, but for the logits processors,
    #       we will pass `num_beams=1` to make sure that the `PrefixConstrainedLogitsProcessor`
    #       processor does not break.
    hf_generation_params["num_beams_warper"] = hf_generation_params.get("num_beams", hf_model.config.num_beams)
    hf_generation_params["num_beams"] = 1
    hf_generation_params["num_beam_groups"] = 1

    prompt_len = prompt_kwargs["input_ids"].shape[-1]
    pred_len = model_kwargs["input_ids"].shape[-1]

    logits_processors = get_logits_processor_from_generate(
        hf_model,
        input_ids=model_kwargs["input_ids"],
        attention_mask=model_kwargs["attention_mask"],
        **hf_generation_params
    )
    forced_token_logits_processors = get_forced_token_logits_processors(logits_processors)

    processed_log_probs = log_probs.clone()
    force_corrected_untampered_log_probs = log_probs.clone()
    for idx in range(prompt_len - 1, pred_len):
        processed_log_probs[:, idx] = logits_processors(
            input_ids=model_kwargs["input_ids"][:, : 1 + idx], scores=processed_log_probs[:, idx]
        )
        if len(forced_token_logits_processors) > 0:
            force_corrected_untampered_log_probs[:, idx] = forced_token_logits_processors(
                input_ids=model_kwargs["input_ids"][:, : 1 + idx],
                scores=force_corrected_untampered_log_probs[:, idx],
            )

    log_probs = log_probs[:, prompt_len - 1: -1, :]
    processed_log_probs = processed_log_probs[:, prompt_len - 1: -1, :]
    force_corrected_untampered_log_probs = force_corrected_untampered_log_probs[:, prompt_len - 1: -1, :]

    scores = SampleScores(
        ids=labels,
        untampered_log_probs=log_probs,
        force_corrected_untampered_log_probs=force_corrected_untampered_log_probs,
        processed_log_probs=processed_log_probs,
        length_penalty=hf_generation_params.length_penalty if "length_penalty" in hf_generation_params else 1.0,
        prompt_length=prompt_len,
    )
    return scores
