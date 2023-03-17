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
NUM_THREADS=4  # Can be increases to the number of GPUs or more if the GPUs fit more models at once.
DATAMODULE_NUM_WORKERS=2
BATCH_SIZE=6 # Can be increases as much as the GPUs / compute allows.

# For VGBS the ddp can be implemented in an alternative way. This might run faster depending on the hardware.
# In addition you might want to increase the batch size for:
#       - the LM (above) and
#       - the evaluation model (applies only to toxicity; updated via evaluation_model.batch_size).
# To use this setting: 1) set the appropriate number of GPUs; 2) uncomment the commented lines in the call; and 3) delete lines 41 and 56.
NUM_GPUS=4

# Task Specific Parameters
EVALUATION_MODEL="mt_noisy_oracle_b2"

# Experiment Parameters
LAMBDA=0.15
CONTRIBUTION_FACTOR=0.25

if [ $PRINT == true ]
then
  echo python -m run_evaluation evaluation=mbart_translation model/decoding=[mbart_generic,value_guided_beam_search] \
    model.decoding.hf_generation_params.tokens_considered_by_value_processor=20 \
    evaluation_model=$EVALUATION_MODEL \
    datamodule.dataset_parameters.test.dataloader.batch_size=$BATCH_SIZE \
    datamodule.debug=$DEBUG datamodule.debug_k=$NUM_DATAPOINTS datamodule.num_workers=$DATAMODULE_NUM_WORKERS \
    trainer.progress_bar_refresh_rate=1 \
    trainer=ddp trainer.gpus=0 +trainer.devices=$NUM_THREADS trainer.accelerator='cpu' +model.scatter_accross_gpus=True \
    evaluation_model.noising_function_parameters.lambda=$LAMBDA \
    model.decoding.hf_generation_params.contribution_factor=$CONTRIBUTION_FACTOR \
    logger=$LOGGER \
    +hydra.job.env_set.CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS_STRING \
    run_name=mbart_translation_vgbs_lambda_${LAMBDA}_cf_${CONTRIBUTION_FACTOR}
    #    trainer=ddp trainer.gpus=$NUM_GPUS \  # Uncomment this line and delete the one above for alternative ddp implementation.
    #    evaluation_model.batch_size=30 \  # Could speed-up evaluation if GPU memory allows for this. Applies only to toxicity.
else
  TOKENIZERS_PARALLELISM='false' python -m run_evaluation evaluation=mbart_translation model/decoding=[mbart_generic,value_guided_beam_search] \
    model.decoding.hf_generation_params.tokens_considered_by_value_processor=20 \
    evaluation_model=$EVALUATION_MODEL \
    datamodule.dataset_parameters.test.dataloader.batch_size=$BATCH_SIZE \
    datamodule.debug=$DEBUG datamodule.debug_k=$NUM_DATAPOINTS datamodule.num_workers=$DATAMODULE_NUM_WORKERS \
    trainer.progress_bar_refresh_rate=1 \
    trainer=ddp trainer.gpus=0 +trainer.devices=$NUM_THREADS trainer.accelerator='cpu' +model.scatter_accross_gpus=True \
    evaluation_model.noising_function_parameters.lambda=$LAMBDA \
    model.decoding.hf_generation_params.contribution_factor=$CONTRIBUTION_FACTOR \
    logger=$LOGGER \
    +hydra.job.env_set.CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS_STRING \
    run_name=mbart_translation_vgbs_lambda_${LAMBDA}_cf_${CONTRIBUTION_FACTOR}
    #    trainer=ddp trainer.gpus=$NUM_GPUS \  # Uncomment this line and delete the one above for alternative ddp implementation.
    #    evaluation_model.batch_size=30 \  # Could speed-up evaluation if GPU memory allows for this. Applies only to toxicity.
fi


