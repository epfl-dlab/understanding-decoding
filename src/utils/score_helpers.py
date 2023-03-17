import warnings
from copy import deepcopy
from typing import Optional, Iterable, Callable, List

import torch

import src.utils.general as general_utils
import transformers
from transformers.generation_utils import GenerationMixin

logger = general_utils.get_logger(__name__)


def get_hf_generation_params(pl_model, is_target=False):
    # PL: (P)yTorch (L)ightning
    # HF: (H)ugging(F)ace🤗

    hf_model = pl_model.model
    hf_generation_params = pl_model.hparams.decoding["hf_generation_params"].copy()

    if is_target:
        # Target token ids might have `-inf` if they do not oblige to some of the params below
        # For example, if `top_k=5` and the target token ids did not pick one of the `top_k`
        # tokens, then the sequence would have a score/likelihood of `-inf`.
        hf_generation_params.update(
            {
                "max_length": hf_model.config.max_position_embeddings,
                "min_length": 0,
                "top_k": 0,
                "top_p": 1,
                "no_repeat_ngram_size": 0,
                "encoder_no_repeat_ngram_size": 0,
                "max_new_tokens": hf_model.config.max_position_embeddings,
            }
        )

    return hf_generation_params


def get_logits_processor_from_generate(
    hf_model,
    input_ids: Optional[torch.LongTensor] = None,
    max_length: Optional[int] = None,
    min_length: Optional[int] = None,
    do_sample: Optional[bool] = None,
    do_stochastic: Optional[bool] = None,
    do_value_guided: Optional[bool] = None,
    early_stopping: Optional[bool] = None,
    num_beams: Optional[int] = None,
    num_beams_warper: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    bad_words_ids: Optional[Iterable[int]] = None,
    bos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    length_penalty: Optional[float] = None,
    no_repeat_ngram_size: Optional[int] = None,
    encoder_no_repeat_ngram_size: Optional[int] = None,
    num_return_sequences: Optional[int] = None,
    max_time: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    decoder_start_token_id: Optional[int] = None,
    use_cache: Optional[bool] = None,
    num_beam_groups: Optional[int] = None,
    diversity_penalty: Optional[float] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    forced_bos_token_id: Optional[int] = None,
    forced_eos_token_id: Optional[int] = None,
    remove_invalid_values: Optional[bool] = None,
    synced_gpus: Optional[bool] = None,
    use_mcts: bool = False,
    use_pplmcts: bool = False,
    **model_kwargs,
):
    r"""
    Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
    multinomial sampling, beam-search decoding, and beam-search multinomial sampling.

    Apart from :obj:`input_ids` and :obj:`attention_mask`, all the arguments below will default to the value of the
    attribute of the same name inside the :class:`~transformers.PretrainedConfig` of the model. The default values
    indicated are the default values of those config.

    Most of these parameters are explained in more detail in `this blog post
    <https://huggingface.co/blog/how-to-generate>`__.

    Parameters:

        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            The sequence used as a prompt for the generation. If :obj:`None` the method initializes it with
            :obj:`bos_token_id` and a batch size of 1.
        max_length (:obj:`int`, `optional`, defaults to :obj:`model.config.max_length`):
            The maximum length of the sequence to be generated.
        max_new_tokens (:obj:`int`, `optional`, defaults to None):
            The maximum numbers of tokens to generate, ignore the current number of tokens. Use either
            :obj:`max_new_tokens` or :obj:`max_length` but not both, they serve the same purpose.
        min_length (:obj:`int`, `optional`, defaults to 10):
            The minimum length of the sequence to be generated.
        do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use sampling ; use greedy decoding otherwise.
        early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
        num_beams (:obj:`int`, `optional`, defaults to 1):
            Number of beams for beam search. 1 means no beam search.
        temperature (:obj:`float`, `optional`, defaults to 1.0):
            The value used to module the next token probabilities.
        top_k (:obj:`int`, `optional`, defaults to 50):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (:obj:`float`, `optional`, defaults to 1.0):
            If set to float < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or
            higher are kept for generation.
        repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
            The parameter for repetition penalty. 1.0 means no penalty. See `this paper
            <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
        pad_token_id (:obj:`int`, `optional`):
            The id of the `padding` token.
        bos_token_id (:obj:`int`, `optional`):
            The id of the `beginning-of-sequence` token.
        eos_token_id (:obj:`int`, `optional`):
            The id of the `end-of-sequence` token.
        length_penalty (:obj:`float`, `optional`, defaults to 1.0):
            Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
            model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
            sequences.
        no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
            If set to int > 0, all ngrams of that size can only occur once.
        encoder_no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
            If set to int > 0, all ngrams of that size that occur in the ``encoder_input_ids`` cannot occur in the
            ``decoder_input_ids``.
        bad_words_ids(:obj:`List[List[int]]`, `optional`):
            List of token ids that are not allowed to be generated. In order to get the tokens of the words that
            should not appear in the generated text, use :obj:`tokenizer(bad_word,
            add_prefix_space=True).input_ids`.
        num_return_sequences(:obj:`int`, `optional`, defaults to 1):
            The number of independently computed returned sequences for each element in the batch.
        max_time(:obj:`float`, `optional`, defaults to None):
            The maximum amount of time you allow the computation to run for in seconds. generation will still
            finish the current pass after allocated time has been passed.
        attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
            tokens that are not masked, and 0 for masked tokens. If not provided, will default to a tensor the same
            shape as :obj:`input_ids` that masks the pad token. `What are attention masks?
            <../glossary.html#attention-mask>`__
        decoder_start_token_id (:obj:`int`, `optional`):
            If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
        use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should use the past last key/values attentions (if applicable to the model) to
            speed up decoding.
        num_beam_groups (:obj:`int`, `optional`, defaults to 1):
            Number of groups to divide :obj:`num_beams` into in order to ensure diversity among different groups of
            beams. `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.
        diversity_penalty (:obj:`float`, `optional`, defaults to 0.0):
            This value is subtracted from a beam's score if it generates a token same as any beam from other group
            at a particular time. Note that :obj:`diversity_penalty` is only effective if ``group beam search`` is
            enabled.
        prefix_allowed_tokens_fn: (:obj:`Callable[[int, torch.Tensor], List[int]]`, `optional`):
            If provided, this function constraints the beam search to allowed tokens only at each step. If not
            provided no constraint is applied. This function takes 2 arguments: the batch ID :obj:`batch_id` and
            :obj:`input_ids`. It has to return a list with the allowed tokens for the next generation step
            conditioned on the batch ID :obj:`batch_id` and the previously generated tokens :obj:`inputs_ids`. This
            argument is useful for constrained generation conditioned on the prefix, as described in
            `Autoregressive Entity Retrieval <https://arxiv.org/abs/2010.00904>`__.
        output_attentions (:obj:`bool`, `optional`, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
            returned tensors for more details.
        output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
            for more details.
        output_scores (:obj:`bool`, `optional`, defaults to `False`):
            Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
        return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        forced_bos_token_id (:obj:`int`, `optional`):
            The id of the token to force as the first generated token after the :obj:`decoder_start_token_id`.
            Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token
            needs to be the target language token.
        forced_eos_token_id (:obj:`int`, `optional`):
            The id of the token to force as the last generated token when :obj:`max_length` is reached.
        remove_invalid_values (:obj:`bool`, `optional`):
            Whether to remove possible `nan` and `inf` outputs of the model to prevent the generation method to
            crash. Note that using ``remove_invalid_values`` can slow down generation.
        synced_gpus (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)

        model_kwargs:
            Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If the
            model is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific
            kwargs should be prefixed with `decoder_`.

    Return:
        :class:`~transformers.file_utils.ModelOutput` or :obj:`torch.LongTensor`: A
        :class:`~transformers.file_utils.ModelOutput` (if ``return_dict_in_generate=True`` or when
        ``config.return_dict_in_generate=True``) or a :obj:`torch.FloatTensor`.

            If the model is `not` an encoder-decoder model (``model.config.is_encoder_decoder=False``), the
            possible :class:`~transformers.file_utils.ModelOutput` types are:

                - :class:`~transformers.generation_utils.GreedySearchDecoderOnlyOutput`,
                - :class:`~transformers.generation_utils.SampleDecoderOnlyOutput`,
                - :class:`~transformers.generation_utils.BeamSearchDecoderOnlyOutput`,
                - :class:`~transformers.generation_utils.BeamSampleDecoderOnlyOutput`

            If the model is an encoder-decoder model (``model.config.is_encoder_decoder=True``), the possible
            :class:`~transformers.file_utils.ModelOutput` types are:

                - :class:`~transformers.generation_utils.GreedySearchEncoderDecoderOutput`,
                - :class:`~transformers.generation_utils.SampleEncoderDecoderOutput`,
                - :class:`~transformers.generation_utils.BeamSearchEncoderDecoderOutput`,
                - :class:`~transformers.generation_utils.BeamSampleEncoderDecoderOutput`
    """
    # set init values
    if max_length is None and max_new_tokens is None:
        # Both are None, default
        max_length = hf_model.config.max_length
    elif max_length is not None and max_new_tokens is not None:
        # Both are set, this is odd, raise a warning
        warnings.warn(
            "Both `max_length` and `max_new_tokens` have been set but they serve the same purpose.", UserWarning
        )

    max_length = max_length if max_length is not None else hf_model.config.max_length

    eos_token_id = eos_token_id if eos_token_id is not None else hf_model.config.eos_token_id

    # Storing encoder_input_ids for logits_processor that could use them
    encoder_input_ids = input_ids if hf_model.config.is_encoder_decoder else None

    if input_ids.shape[-1] > max_length:
        input_ids_string = "decoder_input_ids" if hf_model.config.is_encoder_decoder else "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}."
            "You should check whether ``stopping_criteria`` works in your decoding method."
        )

    # get distribution pre_processing samplers
    logits_processor = hf_model._get_logits_processor(
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
        encoder_input_ids=encoder_input_ids,
        bad_words_ids=bad_words_ids,
        min_length=min_length,
        max_length=max_length,
        eos_token_id=eos_token_id,
        forced_bos_token_id=forced_bos_token_id,
        forced_eos_token_id=forced_eos_token_id,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
        remove_invalid_values=remove_invalid_values,
    )

    # determine generation mode
    # determine generation mode
    if use_mcts or use_pplmcts:
        is_mcts = True
        is_greedy_gen_mode = False
        is_sample_gen_mode = False
        is_beam_gen_mode = False
        is_beam_sample_gen_mode = False
        is_group_beam_gen_mode = False
        is_beam_stochastic_gen_mode = False
        is_beam_value_gen_mode = False
    else:
        is_mcts = False
        is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False
        is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True

        is_beam_gen_mode = False
        is_beam_sample_gen_mode = False
        is_beam_stochastic_gen_mode = False
        is_beam_value_gen_mode = False

        if do_sample and do_stochastic:
            raise ValueError(
                (
                    "Only one of do_sample and do_stochastic can be specified. "
                    "Found do_sample: {0}, do_stochastic: {1}".format(do_sample, do_stochastic)
                )
            )
        if do_sample and do_value_guided:
            raise ValueError(
                (
                    "Only one of do_sample and do_stochastic can be specified. "
                    "Found do_sample: {0}, do_stochastic: {1}".format(do_sample, do_value_guided)
                )
            )
        if do_stochastic and do_value_guided:
            raise ValueError(
                (
                    "Only one of do_sample and do_stochastic can be specified. "
                    "Found do_sample: {0}, do_stochastic: {1}".format(do_stochastic, do_value_guided)
                )
            )

        if not do_sample and not do_stochastic:
            is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1)
        elif do_sample:
            is_beam_sample_gen_mode = (num_beams > 1) and (num_beam_groups == 1)
        elif do_stochastic:
            is_beam_stochastic_gen_mode = (num_beams > 1) and (num_beam_groups == 1)
        elif do_value_guided:
            # VGBS outputs should be scored in the same way as BS
            # is_beam_value_gen_mode = (num_beams > 1) and (num_beam_groups == 1)
            is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1)

        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)
        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

    assert (
        sum(
            [
                is_mcts,
                is_greedy_gen_mode,
                is_sample_gen_mode,
                is_beam_gen_mode,
                is_beam_value_gen_mode,
                is_beam_sample_gen_mode,
                is_group_beam_gen_mode,
                is_beam_stochastic_gen_mode,
            ]
        )
        == 1
    )

    num_beams_warper = num_beams_warper if num_beams_warper is not None else num_beams

    if is_beam_stochastic_gen_mode:
        top_k = 0
        top_p = 1
    elif is_mcts:
        num_beams_warper = 2

    if is_sample_gen_mode or is_beam_sample_gen_mode or is_mcts or is_beam_stochastic_gen_mode:
        # get probability distribution warper
        logits_warper = GenerationMixin._get_logits_warper(
            hf_model,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams_warper,
        )
        for lp in logits_warper:
            logits_processor += [lp]

    return logits_processor


def get_forced_token_logits_processors(logits_processors):
    forced_token_lps = transformers.generation_logits_process.LogitsProcessorList()
    for lp in logits_processors:
        if isinstance(lp, transformers.generation_logits_process.ForcedEOSTokenLogitsProcessor) or isinstance(
            lp, transformers.generation_logits_process.ForcedBOSTokenLogitsProcessor
        ):
            forced_token_lps.append(lp)
    return forced_token_lps


# ~~~~~ MBART & BART ~~~~~
def bart_prepare_inputs_and_labels(mbart_for_cg_model, src_input, tgt_input):
    src_input = deepcopy(src_input)
    tgt_input = deepcopy(tgt_input)

    model_kwargs = {}

    assert "attention_mask" in src_input
    assert src_input["attention_mask"] is not None
    model_kwargs["input_ids"] = src_input["input_ids"].reshape((1, -1))
    model_kwargs["attention_mask"] = src_input["attention_mask"].reshape((1, -1))

    # Remove target padding
    if "attention_mask" in tgt_input:
        attention_mask = tgt_input.pop("attention_mask")
    else:
        attention_mask = tgt_input["input_ids"].squeeze() != mbart_for_cg_model.config.pad_token_id
        assert attention_mask[0] != 0
    tgt_input["input_ids"] = tgt_input["input_ids"].squeeze()[attention_mask.bool()]
    tgt_input["input_ids"] = tgt_input["input_ids"].reshape((1, -1))
    model_kwargs["labels"] = model_kwargs["decoder_input_ids"] = tgt_input["input_ids"]
    model_kwargs["decoder_attention_mask"] = None

    return model_kwargs


# ~~~~~ GPT-2 ~~~~~
def gpt2_prepare_inputs_and_labels_from_ids(gpt2_model, src_input, tgt_input):
    src_input = deepcopy(src_input)
    tgt_input = deepcopy(tgt_input)

    # Remove padding
    if "attention_mask" in src_input:
        attention_mask = src_input.pop("attention_mask").squeeze()
    else:
        attention_mask = gpt2_compute_attention_mask(src_input["input_ids"].squeeze(), gpt2_model.config.eos_token_id)
    src_input["input_ids"] = src_input["input_ids"].squeeze()[attention_mask.bool()]
    src_input["input_ids"] = src_input["input_ids"].reshape((1, -1))

    # Remove padding
    if "attention_mask" in tgt_input:
        attention_mask = tgt_input.pop("attention_mask")
    else:
        attention_mask = gpt2_compute_attention_mask(tgt_input["input_ids"].squeeze(), gpt2_model.config.eos_token_id)
    tgt_input["input_ids"] = tgt_input["input_ids"].squeeze()[attention_mask.bool()]
    tgt_input["input_ids"] = tgt_input["input_ids"].reshape((1, -1))

    prompt_kwargs = gpt2_model.prepare_inputs_for_generation(**src_input)
    model_kwargs = gpt2_model.prepare_inputs_for_generation(**tgt_input)

    prompt_len = prompt_kwargs["input_ids"].shape[-1]
    assert torch.all(prompt_kwargs["input_ids"] == model_kwargs["input_ids"][:, :prompt_len])

    labels = model_kwargs["input_ids"][:, prompt_len:]
    return prompt_kwargs, model_kwargs, labels


def gpt2_compute_attention_mask(input_ids, eos_token_id):
    """
    Compute (or rather guess) the attention mask for GPT2. GPT2 does not have
    a padding token so we use the EOS token instead. Computing the attention
    mask needs to be aware of this and cannot simpy do `input_ids==pad_token_id`.
    This methods computes the attention mask by guessing where the EOS token
    should be. See tests for examples.

    Parameters
    ----------
    input_ids: one dimensional torch tensor
    eos_token_id: int

    Returns
    -------
    bool tensor of same shape as input_ids

    """
    if input_ids.dim() != 1:
        raise NotImplementedError("Batches not supported")

    attention_mask = input_ids != eos_token_id

    # GPT2 does not have PAD tokens and we use EOS instead (PAD = EOS). The first PAD token on the right side
    # is not a PAD token but a EOS token and should have a 1 in the attention mask.
    # Example:
    # EOS 1 2 3 4 5 EOS EOS EOS --> Only the second EOS is a real EOS token, others are PAD
    eos_index = len(input_ids)
    while eos_index > 0 and input_ids[eos_index - 1] == eos_token_id:
        # TODO while loop might be slow. Maybe a torch.nonzero approach would be faster? Or C++/Cython?
        eos_index -= 1

    if eos_index != len(input_ids) and input_ids[eos_index] == eos_token_id:
        attention_mask[eos_index] = 1

    return attention_mask
