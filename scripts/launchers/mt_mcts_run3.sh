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
VISIBLE_GPUS_STRING="'0,1,2,3'" # More GPUs can be added if available.
NUM_THREADS=8  # Can be increases to the number of GPUs or more if the GPUs fit more models at once.
DATAMODULE_NUM_WORKERS=2
BATCH_SIZE=1 # Can be increases as much as the GPUs / compute allows.

# Task Specific Parameters
EVALUATION_MODEL="mt_noisy_oracle_b2"

# Experiment Parameters
LAMBDA=0.25
PB_C_INIT=0.25

if [ $PRINT == true ]
then
  echo python -m run_evaluation evaluation=mbart_translation model/decoding=[mbart_generic,pplmcts] \
    model.decoding.hf_generation_params.mcts_topk_actions=20 \
    model.decoding.hf_generation_params.mcts_num_simulations=50 \
    evaluation_model=$EVALUATION_MODEL \
    datamodule.dataset_parameters.test.dataloader.batch_size=$BATCH_SIZE \
    datamodule.debug=$DEBUG datamodule.debug_k=$NUM_DATAPOINTS datamodule.num_workers=$DATAMODULE_NUM_WORKERS \
    trainer.progress_bar_refresh_rate=1 \
    trainer=ddp trainer.gpus=0 +trainer.devices=$NUM_THREADS trainer.accelerator='cpu' +model.scatter_accross_gpus=True \
    evaluation_model.noising_function_parameters.lambda=$LAMBDA \
    model.decoding.hf_generation_params.mcts_pb_c_init=$PB_C_INIT \
    logger=$LOGGER \
    +hydra.job.env_set.CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS_STRING \
    run_name=mbart_translation_mcts_lambda_${LAMBDA}_pb_c_init_${PB_C_INIT}
else
  TOKENIZERS_PARALLELISM='false' python -m run_evaluation evaluation=mbart_translation model/decoding=[mbart_generic,pplmcts] \
    model.decoding.hf_generation_params.mcts_topk_actions=20 \
    model.decoding.hf_generation_params.mcts_num_simulations=50 \
    evaluation_model=$EVALUATION_MODEL \
    datamodule.dataset_parameters.test.dataloader.batch_size=$BATCH_SIZE \
    datamodule.debug=$DEBUG datamodule.debug_k=$NUM_DATAPOINTS datamodule.num_workers=$DATAMODULE_NUM_WORKERS \
    trainer.progress_bar_refresh_rate=1 \
    trainer=ddp trainer.gpus=0 +trainer.devices=$NUM_THREADS trainer.accelerator='cpu' +model.scatter_accross_gpus=True \
    evaluation_model.noising_function_parameters.lambda=$LAMBDA \
    model.decoding.hf_generation_params.mcts_pb_c_init=$PB_C_INIT \
    logger=$LOGGER \
    +hydra.job.env_set.CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS_STRING \
    run_name=mbart_translation_mcts_lambda_${LAMBDA}_pb_c_init_${PB_C_INIT}
fi
