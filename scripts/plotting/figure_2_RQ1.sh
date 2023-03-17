#!/bin/bash

export PYTHONPATH=".:transformers/src:mctx"

PRINT=false

GRIDSIZE=10

ENTITY=epfl-dlab
ROOT=epfl-dlab/understanding-decoding
VERSION=paper-v1

datasets=("rtp" "wmt14" "rebel" "swissprot")
paths=(
  "$ROOT/6yf8p424 $ROOT/34kq9nsf $ROOT/2mgcqsyk"
  "$ROOT/1y6yme2k $ROOT/3410qxd0 $ROOT/gigllz7f"
  "$ROOT/9lbtlj0l $ROOT/28vcj2b0 $ROOT/2cqpyxit"
  "$ROOT/1k2x9th9 $ROOT/2fhbfm2e $ROOT/3vh6a5h8"
)

for i in "${!datasets[@]}";
do
  if [ $PRINT == true ]
  then
    echo python scripts/visualize_joint_plots.py \
      --dataset ${datasets[i]} \
      --gridsize $GRIDSIZE \
      --share_colorbar \
      --wandb_entity_for_report $ENTITY \
      --wandb_run_paths ${paths[i]} \
      --wandb_id_for_report Fig2-$VERSION
  else
    python scripts/visualize_joint_plots.py \
      --dataset ${datasets[i]} \
      --gridsize $GRIDSIZE \
      --share_colorbar \
      --wandb_entity_for_report $ENTITY \
      --wandb_run_paths ${paths[i]} \
      --wandb_id_for_report Fig2-$VERSION
  fi
done
