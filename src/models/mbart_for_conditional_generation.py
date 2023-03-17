import json
import os
from collections import defaultdict

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger, LoggerCollection

import src.utils.evaluation as evaluation_utils
import src.utils.general as general_utils
import transformers
from src.models.mcts_mixin import GenerationMixinWithGenericMCTSSupport
from src.utils.oracles import bart_get_evaluation_model_target_ids
from src.utils.score_helpers import (
    get_hf_generation_params,
    bart_prepare_inputs_and_labels,
)
from src.utils.scorers import get_encoder_decoder_scores
from transformers.models.bart.modeling_bart import shift_tokens_right

log = general_utils.get_logger(__name__)


class MBartForConditionalGenerationWithMCTSSupport(
    transformers.MBartForConditionalGeneration, GenerationMixinWithGenericMCTSSupport
):
    pass


class MBartForConditionalGeneration(pl.LightningModule):
    def __init__(self, random_initialization=False, from_checkpoint=False, evaluation_model=None, **kwargs):
        super().__init__()
        # TODO: Update outside logging and set hparams logging to false
        # self.save_hyperparameters(logger=False)
        self.save_hyperparameters(ignore="datamodule")

        if from_checkpoint or random_initialization:
            # if the model is initialized form a checkpoint
            # the weights from the checkpoint will be set after the object is constructed
            self.model = MBartForConditionalGenerationWithMCTSSupport(self.hparams.hf_config)
        else:
            self.model = MBartForConditionalGenerationWithMCTSSupport.from_pretrained(**self.hparams.hf_model)

        self.evaluation_model = evaluation_model
        self.tokenizer = self.hparams.tokenizer
        self.src_lang_code = self.tokenizer.src_lang.split("_")[0]
        self.tgt_lang_code = self.tokenizer.tgt_lang.split("_")[0]
        self.hparams["decoding"]["hf_generation_params"]["forced_bos_token_id"] = self.tokenizer.lang_code_to_id[
            self.tokenizer.tgt_lang
        ]

        self.hparams["from_checkpoint"] = True
        self.hparams["random_initialization"] = False

    def on_test_epoch_start(self):
        if self.hparams.get("scatter_accross_gpus", False):
            cuda_id = self.global_rank % torch.cuda.device_count()
            device = torch.device(f"cuda:{cuda_id}")
            self.to(device)
            if self.evaluation_model is not None:
                self.evaluation_model.reset_device(device)
                self.evaluation_model.reset_rank(self.global_rank)
                self.evaluation_model.reset_random_state()
            log.info(f"Thread with rank `{self.global_rank}` running on device cuda:{cuda_id}")

    def test_step(self, batch, batch_idx):
        raw_target = batch[self.tgt_lang_code]
        noisy_target = batch["noisy_" + self.tgt_lang_code]

        with self.tokenizer.as_target_tokenizer():
            target_tokenizer_output = {
                k: v.to(self.device)
                for k, v in self.tokenizer(
                    raw_target,
                    return_tensors="pt",
                    return_attention_mask=False,
                    padding=True,
                    truncation=True,
                ).items()
            }
            target_decoder_input_ids = target_tokenizer_output["input_ids"]

            full_target_ids = bart_get_evaluation_model_target_ids(
                self.model, self.tokenizer, target_decoder_input_ids
            )
            full_target_ids = full_target_ids.to(self.device)

            target_decoder_input_ids = shift_tokens_right(
                target_decoder_input_ids, self.model.config.pad_token_id, self.model.config.decoder_start_token_id
            )
        batch['target_decoder_input_ids'] = target_decoder_input_ids
        batch['full_target_ids'] = full_target_ids

        with self.tokenizer.as_target_tokenizer():
            target_tokenizer_output = {
                k: v.to(self.device)
                for k, v in self.tokenizer(
                    noisy_target,
                    return_tensors="pt",
                    return_attention_mask=False,
                    padding=True,
                    truncation=True,
                ).items()
            }
            noisy_target_decoder_input_ids = target_tokenizer_output["input_ids"]

            full_noisy_target_ids = bart_get_evaluation_model_target_ids(
                self.model, self.tokenizer, noisy_target_decoder_input_ids
            )
            full_noisy_target_ids = full_noisy_target_ids.to(self.device)

        batch['full_noisy_target_ids'] = full_noisy_target_ids

        if not self.hparams.get("resample_exp", False):
            self._test_step_standard(batch, batch_idx)
        else:
            self._test_step_resample_exp(batch, batch_idx)

    def _test_step_standard(self, batch, batch_idx):
        raw_input = batch[self.src_lang_code]
        raw_target = batch[self.tgt_lang_code]
        noisy_target = batch["noisy_" + self.tgt_lang_code]
        assert isinstance(raw_target[0], str)

        # ----- Generate predictions
        sample_output = self.sample(
            input_data=raw_input,
            return_generation_outputs=True,
            return_generation_inputs=True,
            value_model_kwargs={"target_txt": raw_target,
                                "noisy_target_txt": noisy_target,
                                "full_target_ids": batch["full_target_ids"],
                                "full_noisy_target_ids": batch["full_noisy_target_ids"]
                                }
        )

        self._write_step_output(batch, batch_idx, raw_input, raw_target, sample_output)

    def _test_step_resample_exp(self, batch, batch_idx):
        raw_input = batch[self.src_lang_code]
        raw_target = batch[self.tgt_lang_code]
        assert isinstance(raw_target[0], str)
        predictions = []

        # ----- Generate predictions
        for i in range(self.hparams.n_sim):
            seed = self.hparams.decoding["seed"] + i
            sample_output = self.sample(
                input_data=raw_input,
                seed=seed,
                return_generation_outputs=True,
                return_generation_inputs=True,
            )
            predictions.extend(sample_output["grouped_decoded_sequences"])

            self._write_step_output(
                batch, batch_idx, raw_input, raw_target, sample_output, seeds=[seed for _ in range(len(raw_input))]
            )

    def _write_step_output(self, batch, batch_idx, raw_input, raw_target, sample_output, seeds=None):
        # ----- Write prediction outputs to file
        num_return_sequences = len(sample_output["grouped_decoded_sequences"][0])
        prediction_ids = evaluation_utils.group_sequences(
            sample_output["generation_outputs"].sequences.tolist(), num_return_sequences
        )

        prediction_outputs = {
            "id": batch["id"].tolist(),
            "input": raw_input,
            "input_ids": sample_output["generation_inputs"]["input_ids"].tolist(),
            "target": raw_target,
            "target_ids": batch['target_decoder_input_ids'].tolist(),
            "prediction": sample_output["grouped_decoded_sequences"],
            "prediction_ids": prediction_ids,
        }
        if seeds is not None:
            prediction_outputs["seed"] = seeds

        prediction_outputs_path = f"testing_output_{self.global_rank}.prediction.jsonl.gz"
        if self.testing_output_parent_dir is not None:
            prediction_outputs_path = os.path.join(self.testing_output_parent_dir, prediction_outputs_path)
        prediction_outputs_summary = evaluation_utils.get_summary(prediction_outputs)
        evaluation_utils.write_outputs(prediction_outputs_path, prediction_outputs_summary)

        # ––––– Write other outputs to file
        # Compute other datapoint level outputs like log-likelihood, bleu, etc.

        # No need to add input or prediction as it can be retrieved from the main output file
        other_outputs = {"id": batch["id"].tolist()}
        if seeds is not None:
            other_outputs["seed"] = seeds

        if self.hparams.save_testing_output["save_log_likelihood"]:
            for prefix, decoder_input_ids, n, is_target in [
                ("prediction", sample_output["generation_outputs"].sequences, num_return_sequences, False),
                ("target", batch['target_decoder_input_ids'], 1, True),
            ]:
                self._add_loglikelihood_to_outputs(
                    prefix=prefix,
                    outputs=other_outputs,
                    encoder_input_ids=sample_output["generation_inputs"]["input_ids"],
                    encoder_attention_mask=sample_output["generation_inputs"]["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                    num_return_sequences=n,
                    is_target=is_target,
                )

        other_outputs_path = f"testing_output_{self.global_rank}.other.jsonl.gz"
        if self.testing_output_parent_dir is not None:
            other_outputs_path = os.path.join(self.testing_output_parent_dir, other_outputs_path)
        other_outputs_summary = evaluation_utils.get_summary(other_outputs)
        evaluation_utils.write_outputs(other_outputs_path, other_outputs_summary)

        # ––––– Log a few batches during evaluation as a sanity check
        if batch_idx in [0, 100] and self.global_rank == 0:
            pred_json = json.dumps(prediction_outputs_summary)
            other_json = json.dumps(other_outputs_summary)
            pred_df = pd.DataFrame(prediction_outputs)
            other_df = pd.DataFrame(other_outputs)
            log.info(f"Testing_output_summary/predictions_{batch_idx}:\n{pred_json}")
            log.info(f"Testing_output_summary/predictions_{batch_idx}:\n{other_json}")

            if isinstance(self.logger, LoggerCollection):
                loggers = self.logger
            else:
                loggers = [self.logger]

            for logger in loggers:
                if isinstance(logger, WandbLogger):
                    logger.experiment.log({f"Testing_output_summary/predictions_{batch_idx}": pred_df})
                    logger.experiment.log({f"Testing_output_summary/other_{batch_idx}": other_df})

        # return prediction_outputs, other_outputs

    def on_test_epoch_end(self):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        evaluation_utils.upload_outputs_to_wandb(getattr(self, 'hparams_to_log', {}))

    def _add_loglikelihood_to_outputs(
        self,
        prefix,
        outputs,
        encoder_input_ids,
        encoder_attention_mask,
        decoder_input_ids,
        num_return_sequences,
        is_target,
    ):
        # Prepare inputs for computation
        src_inputs = [
            {"input_ids": ids, "attention_mask": mask}
            for ids, mask in zip(encoder_input_ids, encoder_attention_mask)
            for _ in range(num_return_sequences)
        ]
        tgt_inputs = [{"input_ids": seq.clone()} for seq in decoder_input_ids]

        # Compute the likelihood of the predicted texts
        scores = self.compute_scores(src_inputs, tgt_inputs, is_target)

        # Fill the outputs dictionary
        def _group(sequences):
            return evaluation_utils.group_sequences(sequences, num_return_sequences)

        outputs.update(
            {
                f"{prefix}_log_likelihood": _group(scores["processed_log_prob"]),
                f"{prefix}_log_likelihood_untampered": _group(scores["untampered_score"]),
                f"{prefix}_log_likelihood_force_corrected_untampered": _group(
                    scores["force_corrected_untampered_score"]
                ),
            }
        )

    @torch.no_grad()
    def compute_scores(self, src_inputs, tgt_inputs, is_target=False):
        assert isinstance(src_inputs, list)
        assert isinstance(tgt_inputs, list)
        assert len(src_inputs) == len(tgt_inputs)

        # if the model is not in evaluation mode, set it and remember to reset it
        training = self.training
        if training:
            self.eval()

        hf_generation_params = get_hf_generation_params(self, is_target)

        scores = defaultdict(list)
        for src_input, tgt_input in zip(src_inputs, tgt_inputs):
            model_kwargs = bart_prepare_inputs_and_labels(self.model, src_input, tgt_input)
            scores_obj = get_encoder_decoder_scores(self.model, hf_generation_params, model_kwargs)

            scores["scores_obj"].append(scores_obj)
            scores["processed_log_prob"].append(scores_obj.get_processed_score(return_final_score_only=True))
            scores["untampered_score"].append(
                scores_obj.get_untampered_score(force_corrected=False, return_final_score_only=True)
            )
            scores["force_corrected_untampered_score"].append(
                scores_obj.get_untampered_score(force_corrected=True, return_final_score_only=True)
            )

        if training:
            self.train()

        return scores

    @staticmethod
    def _tokenize_input(input_data, tokenizer, device, **tokenizer_kwargs):
        default_tokenizer_kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
        default_tokenizer_kwargs.update(tokenizer_kwargs)
        tokenizer_output = {k: v.to(device) for k, v in tokenizer(input_data, **default_tokenizer_kwargs).items()}

        return tokenizer_output

    @torch.no_grad()
    def sample(
        self,
        input_data,
        seed=None,
        skip_special_tokens=True,
        return_generation_outputs=False,
        return_generation_inputs=False,
        **kwargs,
    ):
        """
        Input data is a list of strings

        (not supported currently) a processed batch (contains src_input_ids,
        and src_attention_mask as expected in training)
        """
        # if the model is not in evaluation mode, set it and remember to reset it
        training = self.training
        if training:
            self.eval()

        hf_generation_params = self.hparams.decoding["hf_generation_params"].copy()
        hf_generation_params.update(kwargs)
        hf_generation_params["return_dict_in_generate"] = True

        if seed is None:
            seed = self.hparams.decoding.get("seed", None)
        if seed:
            transformers.trainer_utils.set_seed(seed)

        tokenizer_output = self._tokenize_input(input_data, self.tokenizer, self.device)
        generation_inputs = {
            "input_ids": tokenizer_output["input_ids"],
            "attention_mask": tokenizer_output["attention_mask"],
            "value_model": self.evaluation_model,
            **hf_generation_params,
        }
        generation_outputs = self.model.generate(**generation_inputs)

        # Returns a list of `num_sentences` decoded (textual) sequences
        num_return_sequences = hf_generation_params.get("num_return_sequences", 1)

        sequences = generation_outputs.sequences
        decoded_sequences = self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens)
        grouped_decoded_sequences = evaluation_utils.group_sequences(decoded_sequences, num_return_sequences)

        if training:
            self.train()

        results = {"grouped_decoded_sequences": grouped_decoded_sequences}
        if return_generation_inputs:
            results["generation_inputs"] = generation_inputs
        if return_generation_outputs:
            results["generation_outputs"] = generation_outputs
        return results
