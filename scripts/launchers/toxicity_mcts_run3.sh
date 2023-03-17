#!/bin/bash

export PYTHONPATH=".:transformers/src:mctx"

# General parameters
#PRINT=true
PRINT=false

#DEBUG=true
DEBUG=false

NUM_DATAPOINTS=96 # Doesn't have an effect if DEBUG is false.
LOGGER=wandb_group

# GPUs and Multiprocessing
VISIBLE_GPUS_STRING="'0,1,2,3,4,5,6,7'"
DATAMODULE_NUM_WORKERS=1
BATCH_SIZE=4
EVALUATION_MODEL_BATCH_SIZE=20

NUM_THREADS=16

# Experiment Parameters
EVALUATION_MODEL="detoxify_noisy_oracle_rmse_02242"
PB_C_INIT=1.25
ABSOLUTE_PATH_TO_OLD_EXP_DIR="/vc_data_1/users/bapatra/code/understanding-decoding/logs/evaluation/runs/gpt2_toxicity_mcts_em_detoxify_noisy_oracle_rmse_02242_pb_c_init_1.25/2022-07-30_18-04-31"

if [ $PRINT == true ]
then
  echo python -m run_evaluation evaluation=gpt2_toxicgen model/decoding=[gpt_generic,pplmcts] \
    model.decoding.hf_generation_params.mcts_topk_actions=20 \
    model.decoding.hf_generation_params.mcts_num_simulations=50 \
    evaluation_model=$EVALUATION_MODEL \
    datamodule.dataset_parameters.test.dataloader.batch_size=$BATCH_SIZE \
    evaluation_model.batch_size=$EVALUATION_MODEL_BATCH_SIZE \
    evaluation_model.device=cpu \
    datamodule.debug=$DEBUG datamodule.debug_k=$NUM_DATAPOINTS datamodule.num_workers=$DATAMODULE_NUM_WORKERS \
    +datamodule.dataset_parameters.test.dataset.subsample=True \
    trainer.progress_bar_refresh_rate=1 \
    trainer=ddp trainer.gpus=0 +trainer.devices=$NUM_THREADS trainer.accelerator='cpu' +model.scatter_accross_gpus=True \
    model.decoding.hf_generation_params.mcts_pb_c_init=$PB_C_INIT \
    logger=$LOGGER \
    +hydra.job.env_set.CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS_STRING \
    run_name=gpt2_toxicity_mcts_em_${EVALUATION_MODEL}_pb_c_init_${PB_C_INIT}
else
  TOKENIZERS_PARALLELISM='false' python -m run_evaluation evaluation=gpt2_toxicgen model/decoding=[gpt_generic,pplmcts] \
    model.decoding.hf_generation_params.mcts_topk_actions=20 \
    model.decoding.hf_generation_params.mcts_num_simulations=50 \
    evaluation_model=$EVALUATION_MODEL \
    datamodule.dataset_parameters.test.dataloader.batch_size=$BATCH_SIZE \
    evaluation_model.batch_size=$EVALUATION_MODEL_BATCH_SIZE \
    evaluation_model.device=cpu \
    datamodule.debug=$DEBUG datamodule.debug_k=$NUM_DATAPOINTS datamodule.num_workers=$DATAMODULE_NUM_WORKERS \
    +datamodule.dataset_parameters.test.dataset.subsample=True \
    trainer.progress_bar_refresh_rate=1 \
    trainer=ddp trainer.gpus=0 +trainer.devices=$NUM_THREADS trainer.accelerator='cpu' +model.scatter_accross_gpus=True \
    model.decoding.hf_generation_params.mcts_pb_c_init=$PB_C_INIT \
    logger=$LOGGER \
    +hydra.job.env_set.CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS_STRING \
    run_name=gpt2_toxicity_mcts_em_${EVALUATION_MODEL}_pb_c_init_${PB_C_INIT}
fi

