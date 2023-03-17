import json
import os
import pdb
from collections import defaultdict

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger, LoggerCollection

import src.utils.evaluation as evaluation_utils
import src.utils.general as general_utils
import transformers
from src.mdp.evaluation_models import EvaluationModel
from src.models.mcts_mixin import GenerationMixinWithGenericMCTSSupport
from src.utils.score_helpers import (
    get_hf_generation_params,
    gpt2_prepare_inputs_and_labels_from_ids,
)
from src.utils.scorers import get_decoder_only_scores
from transformers import AutoModelForCausalLM

log = general_utils.get_logger(__name__)

class AutoModelForCausalLMWithMCTSSupport(
    AutoModelForCausalLM, 
    GenerationMixinWithGenericMCTSSupport
    ):
    pass

class ProtGPT2ForGeneration(pl.LightningModule):
    def __init__(
        self, 
        random_initialization=False, 
        from_checkpoint=False,
        evaluation_model: EvaluationModel = None, 
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore="datamodule")

        max_length = self.hparams.decoding["hf_generation_params"].max_length
        if from_checkpoint or random_initialization:
            # if the model is initialized form a checkpoint
            # the weights from the checkpoint will be set after the object is constructed
            self.model = AutoModelForCausalLMWithMCTSSupport.from_pretrained("nferruz/ProtGPT2", max_length=max_length) #transformers.GPT2LMHeadModel(self.hparams.hf_config)
        else:
            self.model = AutoModelForCausalLMWithMCTSSupport.from_pretrained("nferruz/ProtGPT2", max_length=max_length) #transformers.GPT2LMHeadModel.from_pretrained(**self.hparams.hf_model)

        self.evaluation_model = evaluation_model
        self.tokenizer = self.hparams.tokenizer

        # for some reason for the GPT2 implementation in HF the pad_token coincides with the EOS token
        self.model.config.pad_token_id = self.model.config.eos_token_id
        assert self.tokenizer.padding_side == "left"
        assert self.tokenizer.eos_token_id == self.tokenizer.pad_token_id
        assert self.tokenizer.eos_token_id == self.model.config.eos_token_id
        assert self.tokenizer.pad_token_id == self.model.config.pad_token_id

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
        if not self.hparams.get("resample_exp", False):
            self._test_step_standard(batch, batch_idx)
        else:
            self._test_step_resample_exp(batch, batch_idx)

    def _test_step_standard(self, batch, batch_idx):
        raw_input = batch["input"]

        # ----- Generate predictions
        sample_output = self.sample(
            input_data=raw_input,
            return_generation_outputs=True,
            return_generation_inputs=True,
        )

        self._write_step_output(batch, batch_idx, raw_input, sample_output)

    def _test_step_resample_exp(self, batch, batch_idx):
        raw_input = batch["input"]
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
                batch, batch_idx, raw_input, sample_output, seeds=[seed for _ in range(len(raw_input))]
            )

    def _write_step_output(self, batch, batch_idx, raw_input, sample_output, seeds=None):
        # ----- Write prediction outputs to file
        num_return_sequences = len(sample_output["grouped_decoded_sequences"][0])
        prediction_ids = evaluation_utils.group_sequences(
            sample_output["generation_outputs"].sequences.tolist(), num_return_sequences
        )
        prediction_outputs = {
            "id": batch["id"].tolist(),
            "input": raw_input,
            "input_ids": sample_output["generation_inputs"]["input_ids"].tolist(),
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

        # ----- Write other outputs to file
        # No need to add input or prediction as it can be retrieved from the main output file
        other_outputs = {"id": batch["id"].tolist()}
        if seeds is not None:
            other_outputs["seed"] = seeds

        if self.hparams.save_testing_output["save_log_likelihood"]:
            self._add_loglikelihood_to_outputs(other_outputs, sample_output, num_return_sequences)

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

    @staticmethod
    def _tokenize_input(input_data, tokenizer, device, **tokenizer_kwargs):
        default_tokenizer_kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
        default_tokenizer_kwargs.update(tokenizer_kwargs)
        tokenizer_output = {k: v.to(device) for k, v in tokenizer(input_data, **default_tokenizer_kwargs).items()}

        return tokenizer_output

    @torch.no_grad()
    def compute_scores(self, src_inputs, tgt_inputs):
        assert isinstance(src_inputs, list)
        assert isinstance(tgt_inputs, list)
        assert len(src_inputs) == len(tgt_inputs)

        # if the model is not in evaluation mode, set it and remember to reset it
        training = self.training
        if training:
            self.eval()

        hf_generation_params = get_hf_generation_params(self)

        scores = defaultdict(list)
        for src_input, tgt_input in zip(src_inputs, tgt_inputs):
            if src_input["input_ids"].shape == tgt_input["input_ids"].shape:
                # Nothing was generated, no score to compute
                scores["scores_obj"].append(None)
                scores["processed_log_prob"].append(None)
                scores["untampered_score"].append(None)
                scores["force_corrected_untampered_score"].append(None)

            prompt_kwargs, model_kwargs, label = gpt2_prepare_inputs_and_labels_from_ids(
                self.model, src_input, tgt_input
            )
            scores_obj = get_decoder_only_scores(self.model, hf_generation_params, prompt_kwargs, model_kwargs, label)

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

    def _add_loglikelihood_to_outputs(self, outputs, sample_output, num_return_sequences):
        # Prepare inputs for computation
        prompt_input_ids = sample_output["generation_inputs"]["input_ids"]
        prompt_attention_mask = sample_output["generation_inputs"]["attention_mask"]

        src_inputs = [
            {"input_ids": ids, "attention_mask": mask}
            for ids, mask in zip(prompt_input_ids, prompt_attention_mask)
            for _ in range(num_return_sequences)
        ]

        tgt_inputs = [{"input_ids": seq.clone()} for seq in sample_output["generation_outputs"].sequences]

        # Compute the likelihood of the predicted texts
        scores = self.compute_scores(src_inputs, tgt_inputs)

        # Fill the outputs dictionary
        def _group(sequences):
            return evaluation_utils.group_sequences(sequences, num_return_sequences)

        outputs.update(
            {
                "prediction_log_likelihood": _group(scores["processed_log_prob"]),
                "prediction_log_likelihood_untampered": _group(scores["untampered_score"]),
                "prediction_log_likelihood_force_corrected_untampered": _group(
                    scores["force_corrected_untampered_score"]
                ),
            }
        )

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
