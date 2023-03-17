#!/bin/bash

export PYTHONPATH=".:transformers/src:mctx"

# General parameters
#PRINT=true
PRINT=false

#DEBUG=true
DEBUG=false

NUM_DATAPOINTS=36 # Doesn't have an effect if DEBUG is false.
LOGGER=wandb_group

# GPUs and Multiprocessing
VISIBLE_GPUS_STRING="'0,1,2,3,4,5,6,7'"
DATAMODULE_NUM_WORKERS=3
BATCH_SIZE=6 # Increase if there is more memory, decrease if it is too much.
EVALUATION_MODEL_BATCH_SIZE=$((BATCH_SIZE * 100))

NUM_GPUS=8

# Experiment Parameters
EVALUATION_MODEL="detoxify_noisy_oracle_rmse_01976"
CONTRIBUTION_FACTOR=0.25


if [ $PRINT == true ]
then
  echo python -m run_evaluation evaluation=gpt2_toxicgen model/decoding=[gpt_generic,value_guided_beam_search] \
    model.decoding.hf_generation_params.num_beams=5 \
    model.decoding.hf_generation_params.tokens_considered_by_value_processor=20 \
    evaluation_model=$EVALUATION_MODEL \
    evaluation_model.batch_size=$EVALUATION_MODEL_BATCH_SIZE \
    datamodule.dataset_parameters.test.dataloader.batch_size=$BATCH_SIZE \
    datamodule.debug=$DEBUG datamodule.debug_k=$NUM_DATAPOINTS datamodule.num_workers=$DATAMODULE_NUM_WORKERS \
    trainer.progress_bar_refresh_rate=1 \
    trainer=ddp trainer.gpus=$NUM_GPUS +model.scatter_accross_gpus=True \
    model.decoding.hf_generation_params.contribution_factor=$CONTRIBUTION_FACTOR \
    logger=$LOGGER \
    +hydra.job.env_set.CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS_STRING \
    run_name=gpt2_toxicity_vgbs_em_${EVALUATION_MODEL}_cf_${CONTRIBUTION_FACTOR}
else
  python -m run_evaluation evaluation=gpt2_toxicgen model/decoding=[gpt_generic,value_guided_beam_search] \
    model.decoding.hf_generation_params.num_beams=5 \
    model.decoding.hf_generation_params.tokens_considered_by_value_processor=20 \
    evaluation_model=$EVALUATION_MODEL \
    evaluation_model.batch_size=$EVALUATION_MODEL_BATCH_SIZE \
    datamodule.dataset_parameters.test.dataloader.batch_size=$BATCH_SIZE \
    datamodule.debug=$DEBUG datamodule.debug_k=$NUM_DATAPOINTS datamodule.num_workers=$DATAMODULE_NUM_WORKERS \
    +datamodule.dataset_parameters.test.dataset.subsample=True \
    trainer.progress_bar_refresh_rate=1 \
    trainer=ddp trainer.gpus=$NUM_GPUS +model.scatter_accross_gpus=True \
    model.decoding.hf_generation_params.contribution_factor=$CONTRIBUTION_FACTOR \
    logger=$LOGGER \
    +hydra.job.env_set.CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS_STRING \
    run_name=gpt2_toxicity_vgbs_em_${EVALUATION_MODEL}_cf_${CONTRIBUTION_FACTOR}
fi
