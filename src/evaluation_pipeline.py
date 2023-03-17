import os
from pathlib import Path
from typing import List, Optional

import hydra
import wandb
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

import src.utils.general as general_utils
import src.utils.evaluation as evaluation_utils
from src.mdp.trees.lm_as_tree import LanguageModelAsTree

log = general_utils.get_logger(__name__)


def evaluate(config: DictConfig) -> Optional[float]:
    """Contains the evaluation pipeline.
    Instantiates all PyTorch Lightning objects from configs.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        None
    """
    # Set seed for random number generators in PyTorch, Numpy and Python (random)
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    if config.get("ckpt_path", False) and not os.path.isabs(config.ckpt_path):
        config.ckpt_path = os.path.join(hydra.utils.get_original_cwd(), config.ckpt_path)

    # Initialize the LIT model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config=config.model)

    # Initialize the evaluation model (i.e. the value function used during decoding)
    if "evaluation_model" in config:
        lm_as_tree = LanguageModelAsTree(model, model.tokenizer)
        evaluation_model = hydra.utils.instantiate(config=config.evaluation_model, tree=lm_as_tree)
        model.evaluation_model = evaluation_model

    # Initialize the LIT data module
    if config.model.get("resample_exp", False):
        wapi = wandb.Api()
        exp_dir = wapi.run(config.datamodule.wandb_run_path).config["exp_dir"]
        if os.path.isdir(os.path.join(config.work_dir, exp_dir)):
            config.datamodule.exp_dir = os.path.join(config.work_dir, exp_dir)
        else:
            config.datamodule.exp_dir = evaluation_utils.get_temp_exp_dir(config.work_dir, config.wandb_run_path)
            evaluation_utils.restore_outputs_from_wandb(config.datamodule.wandb_run_path, config.datamodule.exp_dir)

    log.info(f"Instantiating data module <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config=config.datamodule, _recursive_=False)

    # Use the model collator's collate_fn, if defined (otherwise proceed with the PyTorch's default collate_fn)
    if getattr(model, "collator", None):
        datamodule.set_collate_fn(model.collator.collate_fn)

    # Initialize LIT callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(config=cb_conf))

    # Init LIT loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config=config.trainer, callbacks=callbacks, logger=logger, _convert_="partial")

    # Send some parameters from configs to all lightning loggers
    log.info("Logging hyperparameters!")

    general_utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting testing!")
    model.testing_output_parent_dir = "testing_output"
    Path(model.testing_output_parent_dir).mkdir(exist_ok=True)
    if config.model.get("resample_exp", False):
        datamodule.dataset_parameters["test"]["dataset"]["testing_output_parent_dir"] = model.testing_output_parent_dir

    trainer.test(model=model, datamodule=datamodule)

    # Make sure everything closed properly
    log.info("Finalizing!")
    general_utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    scores = trainer.callback_metrics
    log.info("Metrics:")
    log.info(scores)
