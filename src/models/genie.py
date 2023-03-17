import json
import os
from collections import defaultdict
from typing import List, Any

import pandas as pd
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import LoggerCollection, WandbLogger

import src.metrics as CustomMetrics
import src.utils.evaluation as evaluation_utils
import src.utils.general as general_utils
import transformers
from src.constrained_generation import Trie, get_information_extraction_prefix_allowed_tokens_fn_hf
from src.datamodules.utils import TripletUtils
from src.utils.score_helpers import (
    get_hf_generation_params,
    bart_prepare_inputs_and_labels,
)
from src.utils.scorers import get_encoder_decoder_scores
from .mcts_mixin import GenerationMixinWithGenericMCTSSupport
from .utils.general import label_smoothed_nll_loss
from ..mdp.evaluation_models import EvaluationModel
from ..utils.oracles import bart_get_evaluation_model_target_ids

log = general_utils.get_logger(__name__)


class BartForConditionalGenerationWithMCTSSupport(
    transformers.BartForConditionalGeneration, GenerationMixinWithGenericMCTSSupport
):
    pass


class GeniePL(LightningModule):
    def __init__(
        self,
        random_initialization=False,
        from_checkpoint=False,
        evaluation_model: EvaluationModel = None,
        hparams_overrides=None,
        **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters(ignore="datamodule")

        if hparams_overrides is not None:
            # Overriding the hyper-parameters of a checkpoint at an arbitrary depth using a dict structure
            hparams_overrides = self.hparams.pop("hparams_overrides")
            general_utils.update(self.hparams, hparams_overrides)
            log.info("Some values of the original hparams were overridden")
            log.info("Hyper-parameters:")
            log.info(self.hparams)

        if from_checkpoint or random_initialization:
            # Initialization from a local, pre-trained GenIE PL checkpoint
            if self.hparams.get("other_parameters", None):
                self.hparams.hf_config.update(self.hparams.other_parameters)

            self.model = BartForConditionalGenerationWithMCTSSupport(self.hparams.hf_config)
        else:
            # Initialization from a HF model
            if self.hparams.get("other_parameters", None):
                self.hparams.hf_model.config.update(self.hparams.other_parameters)

            self.model = BartForConditionalGenerationWithMCTSSupport.from_pretrained(**self.hparams.hf_model)

        log.info("HF model config:")
        log.info(self.hparams.hf_config)

        self.evaluation_model = evaluation_model
        self.tokenizer = self.hparams.tokenizer
        self.collator = self.hparams.collator

        self.ts_precision = CustomMetrics.TSPrecision()
        self.ts_recall = CustomMetrics.TSRecall()
        self.ts_f1 = CustomMetrics.TSF1()

        if not self.hparams.decoding["free_generation"]:
            self.entity_trie = Trie.load(self.hparams.decoding["entity_trie_path"])
            self.relation_trie = Trie.load(self.hparams.decoding["relation_trie_path"])

        self.testing_output_parent_dir = kwargs.get("testing_output_parent_dir", None)

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None, **kwargs):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            **kwargs,
        )

        return output

    def process_batch(self, batch):
        if self.hparams.get("bos_as_first_token_generated", True):
            return batch

        # remove the starting bos token from the target
        batch["tgt_input_ids"] = batch["tgt_input_ids"][:, 1:]
        batch["tgt_attention_mask"] = batch["tgt_attention_mask"][:, 1:]

        return batch

    def training_step(self, batch, batch_idx=None):
        batch = self.process_batch(batch)

        model_output = self(
            input_ids=batch["src_input_ids"],
            attention_mask=batch["src_attention_mask"],
            labels=batch["tgt_input_ids"],
            decoder_attention_mask=batch["tgt_attention_mask"],
            use_cache=False,
        )

        # the output from hf contains a loss term that can be used in training (see the function commented out above)
        logits = model_output.logits

        # Note that pad_token_id used in tgt_input_ids is 1, and not -100 used by the hugging face loss implementation
        loss, nll_loss = label_smoothed_nll_loss(
            logits.log_softmax(dim=-1),
            batch["tgt_input_ids"],
            batch["tgt_attention_mask"],
            epsilon=self.hparams.eps,
            ignore_index=self.tokenizer.pad_token_id,
        )

        self.log("train-nll_loss", nll_loss.item(), on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx=None):
        batch = self.process_batch(batch)

        model_output = self(
            input_ids=batch["src_input_ids"],
            attention_mask=batch["src_attention_mask"],
            labels=batch["tgt_input_ids"],
            decoder_attention_mask=batch["tgt_attention_mask"],
            use_cache=False,
        )

        logits = model_output.logits

        # Note that pad_token_id used in tgt_input_ids is 1, and not -100 used by the hugging face loss implementation
        loss, nll_loss = label_smoothed_nll_loss(
            logits.log_softmax(dim=-1),
            batch["tgt_input_ids"],
            batch["tgt_attention_mask"],
            epsilon=self.hparams.eps,
            ignore_index=self.tokenizer.pad_token_id,
        )

        self.log("val-nll_loss", nll_loss.item(), on_step=False, on_epoch=True, prog_bar=True)

        return {"val-nll_loss": nll_loss}

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
            return self._test_step_standard(batch, batch_idx)
        else:
            return self._test_step_resample_exp(batch, batch_idx)

    def _test_step_standard(self, batch, batch_idx):
        raw_input = [sample["src"] for sample in batch["raw"]]
        raw_target = [sample["tgt"] for sample in batch["raw"]]
        ids = [sample["id"] for sample in batch["raw"]]

        # ==== Prediction related ===

        # Generate predictions
        if self.hparams.decoding["free_generation"]:
            sample_prefix_allowed_tokens_fn = None
            target_prefix_allowed_tokens_fn = None
        else:
            sample_prefix_allowed_tokens_fn = get_information_extraction_prefix_allowed_tokens_fn_hf(
                self,
                raw_input,
                bos_as_first_token_generated=self.hparams.get("bos_as_first_token_generated", True),
                entities_trie=self.entity_trie,
                relations_trie=self.relation_trie,
            )
            target_prefix_allowed_tokens_fn = get_information_extraction_prefix_allowed_tokens_fn_hf(
                self,
                raw_target,
                bos_as_first_token_generated=self.hparams.get("bos_as_first_token_generated", True),
                entities_trie=self.entity_trie,
                relations_trie=self.relation_trie,
            )
            assert sample_prefix_allowed_tokens_fn is not None
            assert target_prefix_allowed_tokens_fn is not None

        sample_output = self.sample(
            batch,
            input_data_is_processed_batch=True,
            return_generation_inputs=True,
            return_generation_outputs=True,
            output_scores=True,
            prefix_allowed_tokens_fn=sample_prefix_allowed_tokens_fn,
            **self.hparams.decoding["hf_generation_params"],
        )

        self._write_step_output(
            batch_idx=batch_idx,
            ids=ids,
            raw_input=raw_input,
            raw_target=raw_target,
            sample_output=sample_output,
            prediction_prefix_allowed_tokens_fn=sample_prefix_allowed_tokens_fn,
            target_prefix_allowed_tokens_fn=target_prefix_allowed_tokens_fn,
        )

        return_object = {
            "ids": ids,
            "inputs": raw_input,
            "targets": raw_target,
            "predictions": sample_output["grouped_decoded_sequences"],
        }
        return return_object

    def _test_step_resample_exp(self, batch, batch_idx):
        raw_input = [sample["src"] for sample in batch["raw"]]
        raw_target = [sample["tgt"] for sample in batch["raw"]]
        ids = [sample["id"] for sample in batch["raw"]]
        seeds = []

        sample_outputs = []

        # ==== Prediction related ===
        if self.hparams.decoding["free_generation"]:
            sample_prefix_allowed_tokens_fn = None
            target_prefix_allowed_tokens_fn = None
        else:
            sample_prefix_allowed_tokens_fn = get_information_extraction_prefix_allowed_tokens_fn_hf(
                self,
                raw_input,
                bos_as_first_token_generated=self.hparams.get("bos_as_first_token_generated", True),
                entities_trie=self.entity_trie,
                relations_trie=self.relation_trie,
            )
            target_prefix_allowed_tokens_fn = get_information_extraction_prefix_allowed_tokens_fn_hf(
                self,
                raw_target,
                bos_as_first_token_generated=self.hparams.get("bos_as_first_token_generated", True),
                entities_trie=self.entity_trie,
                relations_trie=self.relation_trie,
            )
            assert sample_prefix_allowed_tokens_fn is not None
            assert target_prefix_allowed_tokens_fn is not None

        predictions = []
        # Generate predictions
        for i in range(self.hparams.n_sim):
            seed = self.hparams.decoding["seed"] + i
            seeds.append(seed)
            sample_output = self.sample(
                batch,
                input_data_is_processed_batch=True,
                seed=seed,
                return_generation_inputs=True,
                return_generation_outputs=True,
                output_scores=True,
                prefix_allowed_tokens_fn=sample_prefix_allowed_tokens_fn,
                **self.hparams.decoding["hf_generation_params"],
            )
            # sample_outputs.append(sample_output)  # Creates a memory leak
            predictions.extend(sample_output["grouped_decoded_sequences"])

            self._write_step_output(
                batch_idx=batch_idx,
                ids=ids,
                raw_input=raw_input,
                raw_target=raw_target,
                sample_output=sample_output,
                prediction_prefix_allowed_tokens_fn=sample_prefix_allowed_tokens_fn,
                target_prefix_allowed_tokens_fn=target_prefix_allowed_tokens_fn,
                seeds=[seed for _ in range(len(ids))],
            )

        # predictions = [
        #     decoded_sequence
        #     for sample_output in sample_outputs
        #     for decoded_sequence in sample_output["grouped_decoded_sequences"]
        # ]
        return_object = {
            "ids": ids * self.hparams.n_sim,
            "inputs": raw_input * self.hparams.n_sim,
            "targets": raw_target * self.hparams.n_sim,
            "predictions": predictions,
            "seeds": seeds,
        }
        return return_object

    def test_step_end(self, outputs: List[Any]):
        # Process the data in the format expected by the metrics
        predictions = [
            TripletUtils.convert_text_sequence_to_text_triples(
                texts[0], verbose=self.hparams.decoding["verbose_flag_in_convert_to_triple"]
            )
            for texts in outputs["predictions"]
        ]
        targets = [
            TripletUtils.convert_text_sequence_to_text_triples(
                text, verbose=self.hparams.decoding["verbose_flag_in_convert_to_triple"]
            )
            for text in outputs["targets"]
        ]

        # Update the metrics
        p = self.ts_precision(predictions, targets)
        r = self.ts_recall(predictions, targets)
        f1 = self.ts_f1(predictions, targets)

        # Log the loss
        self.log("test-precision_step", p, on_step=True, on_epoch=False, prog_bar=True)
        self.log("test-recall_step", r, on_step=True, on_epoch=False, prog_bar=True)
        self.log("test-f1_step", f1, on_step=True, on_epoch=False, prog_bar=True)

    def _write_step_output(
        self,
        batch_idx,
        ids,
        raw_input,
        raw_target,
        sample_output,
        prediction_prefix_allowed_tokens_fn,
        target_prefix_allowed_tokens_fn,
        seeds=None,
    ):
        # ----- Write prediction outputs to file
        num_return_sequences = len(sample_output["grouped_decoded_sequences"][0])
        sequences = sample_output["generation_outputs"].sequences
        assert isinstance(sequences, torch.Tensor)
        prediction_ids = evaluation_utils.group_sequences(sequences.tolist(), num_return_sequences)

        tokenizer_output = self.tokenize(self.model, self.tokenizer, raw_input, raw_target)
        target_decoder_input_ids = tokenizer_output["decoder_input_ids"]

        prediction_outputs = {
            "id": ids,
            "input": raw_input,
            "input_ids": sample_output["generation_inputs"]["input_ids"].tolist(),
            "target": raw_target,
            "target_ids": target_decoder_input_ids.tolist(),
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
        other_outputs = {"id": ids}
        if seeds is not None:
            other_outputs["seed"] = seeds

        if self.hparams.save_testing_output["save_log_likelihood"]:
            for prefix, decoder_input_ids, n, prefix_fn, is_target in [
                ("prediction", sequences, num_return_sequences, prediction_prefix_allowed_tokens_fn, False),
                ("target", target_decoder_input_ids, 1, target_prefix_allowed_tokens_fn, True),
            ]:
                self._add_loglikelihood_to_outputs(
                    prefix=prefix,
                    outputs=other_outputs,
                    encoder_input_ids=sample_output["generation_inputs"]["input_ids"],
                    encoder_attention_mask=sample_output["generation_inputs"]["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                    num_return_sequences=n,
                    is_target=is_target,
                    prefix_allowed_tokens_fn=prefix_fn,
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

    def test_epoch_end(self, outputs):
        """Outputs is a list of either test_step outputs outputs"""
        # Log metrics aggregated across steps and processes (in ddp)
        self.log("test-precision", self.ts_precision.compute())
        self.log("test-recall", self.ts_recall.compute())
        self.log("test-f1", self.ts_f1.compute())

        return {
            "test-acc": self.ts_precision.compute(),
            "test-recall": self.ts_precision.compute(),
            "test-f1": self.ts_precision.compute(),
        }

    def on_test_epoch_end(self):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        evaluation_utils.upload_outputs_to_wandb(getattr(self, 'hparams_to_log', {}))

    def configure_optimizers(self):
        # Apply weight decay to all parameters except for the biases and the weight for Layer Normalization
        no_decay = ["bias", "LayerNorm.weight"]

        # Per-parameter optimization.
        # Each dict defines a parameter group and contains the list of parameters to be optimized in a key `params`
        # Other keys should match keyword arguments accepted by the optimizers and
        # will be used as optimization params for the parameter group
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.optimizer.weight_decay,
                # "betas": self.hparams.optimizer.adam_betas,
                "betas": (0.9, 0.999),
                "eps": self.hparams.optimizer.adam_eps,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                # "betas": self.hparams.optimizer.adam_betas,
                "betas": (0.9, 0.999),
                "eps": self.hparams.optimizer.adam_eps,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.optimizer.lr,
            weight_decay=self.hparams.optimizer.weight_decay,
        )

        if self.hparams.optimizer.schedule_name == "linear":
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.optimizer.warmup_updates,
                num_training_steps=self.hparams.optimizer.total_num_updates,
            )
        elif self.hparams.optimizer.schedule_name == "polynomial":
            scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.optimizer.warmup_updates,
                num_training_steps=self.hparams.optimizer.total_num_updates,
                lr_end=self.hparams.optimizer.lr_end,
            )

        lr_dict = {
            "scheduler": scheduler,  # scheduler instance
            "interval": "step",  # The unit of the scheduler's step size. 'step' or 'epoch
            "frequency": 1,  # corresponds to updating the learning rate after every `frequency` epoch/step
            "name": f"LearningRateScheduler-{self.hparams.optimizer.schedule_name}",
            # Used by a LearningRateMonitor callback
        }

        return [optimizer], [lr_dict]

    @staticmethod
    def _convert_surface_form_triplets_to_ids(triplets, entity_name2id, relation_name2id):
        triplets = [[entity_name2id[s], relation_name2id[r], entity_name2id[o]] for s, r, o in triplets]

        return triplets

    @staticmethod
    def _convert_output_to_triplets(output_obj, entity_name2id, relation_name2id):
        if isinstance(output_obj[0], str):
            output = []
            for text in output_obj:
                triplets = TripletUtils.convert_text_sequence_to_text_triples(text)

                if entity_name2id is not None and relation_name2id is not None:
                    triplets = GeniePL._convert_surface_form_triplets_to_ids(triplets, entity_name2id, relation_name2id)

                output.append(triplets)

            return output

        for sample in output_obj:
            sample["textual_triplets"] = TripletUtils.convert_text_sequence_to_text_triples(sample["text"])
            if entity_name2id is not None and relation_name2id is not None:
                sample["id_triplets"] = GeniePL._convert_surface_form_triplets_to_ids(
                    sample["textual_triplets"], entity_name2id, relation_name2id
                )

        return output_obj

    @staticmethod
    def tokenize(bart_for_cg_model, bart_for_cg_tokenizer, src_text, tgt_text=None, truncation=False):
        if isinstance(src_text, str):
            src_text = [src_text]
        model_kwargs = {}

        model_inputs = {
            k: v.to(bart_for_cg_model.device)
            for k, v in bart_for_cg_tokenizer(
                src_text,
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
                truncation=truncation,
            ).items()
        }

        model_kwargs["input_ids"] = model_inputs["input_ids"]
        model_kwargs["attention_mask"] = model_inputs["attention_mask"]

        if isinstance(tgt_text, str):
            tgt_text = [tgt_text]

        if tgt_text is not None:
            with bart_for_cg_tokenizer.as_target_tokenizer():
                decoder_inputs = {
                    k: v.to(bart_for_cg_model.device)
                    for k, v in bart_for_cg_tokenizer(
                        tgt_text,
                        return_tensors="pt",
                        return_attention_mask=True,
                        padding=True,
                        truncation=truncation,
                    ).items()
                }

            model_kwargs["labels"] = decoder_inputs["input_ids"]
            model_kwargs["decoder_input_ids"] = bart_for_cg_model.prepare_decoder_input_ids_from_labels(
                model_kwargs["labels"]
            )
            model_kwargs["decoder_attention_mask"] = decoder_inputs["attention_mask"]

        return model_kwargs

    def get_prefix_allowed_fn(self, input_text, entity_trie, relation_trie):
        if entity_trie is None:
            entity_trie = getattr(self, "entity_trie", None)

        if relation_trie is None:
            relation_trie = getattr(self, "relation_trie", None)

        if entity_trie is None or relation_trie is None:
            return None

        prefix_allowed_tokens_fn = get_information_extraction_prefix_allowed_tokens_fn_hf(
            self,
            input_text,
            bos_as_first_token_generated=self.hparams.get("bos_as_first_token_generated", True),
            entities_trie=entity_trie,
            relations_trie=relation_trie,
        )

        return prefix_allowed_tokens_fn

    @torch.no_grad()
    def sample(
        self,
        input_data,
        input_data_is_processed_batch=False,
        seed=None,
        skip_special_tokens=True,
        return_generation_outputs=False,
        return_generation_inputs=False,
        convert_to_triplets=False,
        prefix_allowed_tokens_fn=None,
        entity_trie=None,
        relation_trie=None,
        surface_form_mappings={"entity_name2id": None, "relation_name2id": None},
        **kwargs,
    ):
        """Input data is a list of strings or a processed batch (contains src_input_ids,
        and src_attention_mask as expected in training)"""
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

        # Get input_ids and attention masks
        if input_data_is_processed_batch:
            input_ids = input_data["src_input_ids"].to(self.device)
            attention_mask = input_data["src_attention_mask"].to(self.device)
            if "raw" in input_data:
                input_text = [sample["src"] for sample in input_data["raw"]]
            else:
                input_text = None

            if self.evaluation_model is not None and "value_model_kwargs" not in hf_generation_params:
                if "tgt_input_ids" in input_data:
                    expected_output_ids = bart_get_evaluation_model_target_ids(
                        self.model, self.tokenizer, input_data["tgt_input_ids"]
                    )
                    expected_output_ids = expected_output_ids.to(self.device)
                    hf_generation_params["value_model_kwargs"] = {"target_ids": expected_output_ids}
        else:
            tokenizer_output = self.tokenize(self.model, self.tokenizer, input_data)
            input_ids = tokenizer_output["input_ids"]
            attention_mask = tokenizer_output["attention_mask"]
            input_text = input_data

        if prefix_allowed_tokens_fn is None:
            prefix_allowed_tokens_fn = self.get_prefix_allowed_fn(input_text, entity_trie, relation_trie)

        generation_inputs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
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

        if convert_to_triplets:
            triplets = self._convert_output_to_triplets(decoded_sequences, **surface_form_mappings)
            results["grouped_decoded_triplets"] = evaluation_utils.group_sequences(triplets, num_return_sequences)

        return results

    @torch.no_grad()
    def compute_scores(self, src_inputs, tgt_inputs, prefix_allowed_tokens_fn, is_target=False):
        assert isinstance(src_inputs, list)
        assert isinstance(tgt_inputs, list)
        assert len(src_inputs) == len(tgt_inputs)

        # if the model is not in evaluation mode, set it and remember to reset it
        training = self.training
        if training:
            self.eval()

        hf_generation_params = get_hf_generation_params(self, is_target)
        hf_generation_params["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn

        scores = defaultdict(list)
        for src_input, tgt_input in zip(src_inputs, tgt_inputs):
            if tgt_input["input_ids"].shape[-1] > self.model.config.max_position_embeddings:
                # Too long target input ids will crash the model. We want to exclude these
                # datapoints from the analysis, instead of enabling tokenizer truncation.
                print(
                    f"Datapoint found with `tgt_input['input_ids'].shape[-1] > "
                    f"self.model.config.max_position_embeddings`"
                    f"--> `{tgt_input['input_ids'].shape[-1]} > {self.model.config.max_position_embeddings}` "
                    f"during scores computation. Skipping the datapoint."
                )
                scores["scores_obj"].append(None)
                scores["processed_log_prob"].append(None)
                scores["untampered_score"].append(None)
                scores["force_corrected_untampered_score"].append(None)
                continue

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

    def _add_loglikelihood_to_outputs(
        self,
        prefix,
        outputs,
        encoder_input_ids,
        encoder_attention_mask,
        decoder_input_ids,
        num_return_sequences,
        is_target,
        prefix_allowed_tokens_fn,
    ):
        # Prepare inputs for computation
        src_inputs = [
            {"input_ids": ids, "attention_mask": mask}
            for ids, mask in zip(encoder_input_ids, encoder_attention_mask)
            for _ in range(num_return_sequences)
        ]
        tgt_inputs = [{"input_ids": seq.clone()} for seq in decoder_input_ids]

        # Compute the likelihood of the predicted texts
        scores = self.compute_scores(src_inputs, tgt_inputs, prefix_allowed_tokens_fn, is_target)

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
