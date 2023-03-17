DEBUG=True
NUM_DATAPOINTS=400
VISIBLE_GPUS_STRING="'0,1,2,3'"
DECODING_ALGORITHM="beam_search"

export PYTHONPATH="$PYTHONPATH:$PWD"
export PYTHONPATH="$PYTHONPATH:$PWD/transformers/src"
export PYTHONPATH="$PYTHONPATH:$PWD/mctx"

if [ $DEBUG == True ]
then
  PROT_NUM_DATAPOINTS=$NUM_DATAPOINTS
  echo "Running in debug mode using $NUM_DATAPOINTS datapoints."
else
  PROT_NUM_DATAPOINTS=10000
fi

echo $PWD

# Beam Search + DDP
python -m run_evaluation evaluation=mbart_translation \
          model/decoding=[mbart_generic,$DECODING_ALGORITHM] \
          datamodule.debug=$DEBUG datamodule.debug_k=$NUM_DATAPOINTS \
          trainer=ddp \
          +hydra.job.env_set.CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS_STRING \
          run_name="mt_"$DECODING_ALGORITHM \
          logger=wandb_group

python -m run_evaluation evaluation=gpt2_toxicgen \
          model/decoding=[gpt_generic,$DECODING_ALGORITHM] \
          datamodule.debug=$DEBUG datamodule.debug_k=$NUM_DATAPOINTS \
          trainer=ddp \
          +hydra.job.env_set.CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS_STRING \
          run_name="toxicity_"$DECODING_ALGORITHM \
          logger=wandb_group

python -m run_evaluation evaluation=genie_cie_large \
          model/decoding=[genie_generic,$DECODING_ALGORITHM] \
          datamodule.debug=$DEBUG datamodule.debug_k=$NUM_DATAPOINTS \
          trainer=ddp \
          +hydra.job.env_set.CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS_STRING \
          run_name="genie_"$DECODING_ALGORITHM \
          logger=wandb_group

python -m run_evaluation evaluation=protgpt2_design \
          model/decoding=[protgpt2_generic,$DECODING_ALGORITHM] \
          datamodule.debug=True datamodule.debug_k=$PROT_NUM_DATAPOINTS \
          trainer=ddp \
          +hydra.job.env_set.CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS_STRING \
          run_name="protein_"$DECODING_ALGORITHM \
          logger=wandb_group

# Greedy + DDP
DECODING_ALGORITHM="greedy_search"
python -m run_evaluation evaluation=mbart_translation \
          model/decoding=[mbart_generic,$DECODING_ALGORITHM] \
          datamodule.debug=$DEBUG datamodule.debug_k=$NUM_DATAPOINTS \
          trainer=ddp \
          +hydra.job.env_set.CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS_STRING \
          run_name="test_mt_"$DECODING_ALGORITHM \
          logger=wandb_group

python -m run_evaluation evaluation=gpt2_toxicgen \
          model/decoding=[gpt_generic,$DECODING_ALGORITHM] \
          datamodule.debug=$DEBUG datamodule.debug_k=$NUM_DATAPOINTS \
          trainer=ddp \
          +hydra.job.env_set.CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS_STRING \
          run_name="test_toxicity_"$DECODING_ALGORITHM \
          logger=wandb_group

python -m run_evaluation evaluation=genie_cie_large \
          model/decoding=[genie_generic,$DECODING_ALGORITHM] \
          datamodule.debug=$DEBUG datamodule.debug_k=$NUM_DATAPOINTS \
          trainer=ddp \
          +hydra.job.env_set.CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS_STRING \
          run_name="test_genie_"$DECODING_ALGORITHM \
          logger=wandb_group

python -m run_evaluation evaluation=protgpt2_design \
          model/decoding=[protgpt2_generic,$DECODING_ALGORITHM] \
          datamodule.debug=True datamodule.debug_k=$PROT_NUM_DATAPOINTS \
          trainer=ddp \
          +hydra.job.env_set.CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS_STRING \
          run_name="test_protein_"$DECODING_ALGORITHM \
          logger=wandb_group

# Stochastic Beams + DDP
DECODING_ALGORITHM="stochastic_beams"
python -m run_evaluation evaluation=mbart_translation \
          model/decoding=[mbart_generic,$DECODING_ALGORITHM] \
          datamodule.debug=$DEBUG datamodule.debug_k=$NUM_DATAPOINTS \
          trainer=ddp \
          +hydra.job.env_set.CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS_STRING \
          run_name="mt_"$DECODING_ALGORITHM \
          logger=wandb_group

python -m run_evaluation evaluation=gpt2_toxicgen \
          model/decoding=[gpt_generic,$DECODING_ALGORITHM] \
          datamodule.debug=$DEBUG datamodule.debug_k=$NUM_DATAPOINTS \
          trainer=ddp \
          +hydra.job.env_set.CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS_STRING \
          run_name="toxicity_"$DECODING_ALGORITHM \
          logger=wandb_group

python -m run_evaluation evaluation=genie_cie_large \
          model/decoding=[genie_generic,$DECODING_ALGORITHM] \
          datamodule.debug=$DEBUG datamodule.debug_k=$NUM_DATAPOINTS \
          trainer=ddp \
          +hydra.job.env_set.CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS_STRING \
          run_name="genie_"$DECODING_ALGORITHM \
          logger=wandb_group

python -m run_evaluation evaluation=protgpt2_design \
          model/decoding=[protgpt2_generic,$DECODING_ALGORITHM] \
          datamodule.debug=True datamodule.debug_k=$PROT_NUM_DATAPOINTS \
          trainer=ddp \
          +hydra.job.env_set.CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS_STRING \
          run_name="protein_"$DECODING_ALGORITHM \
          logger=wandb_group