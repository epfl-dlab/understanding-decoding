#!/bin/bash

export PYTHONPATH=".:transformers/src:mctx"

PRINT=false

ENTITY=epfl-dlab
VERSION=paper-v3

if [ $PRINT == true ]
then
  echo python scripts/visualize_figure5.py \
    --wandb_entity_for_report $ENTITY \
    --wandb_id_for_report Fig5-$VERSION
else
  python scripts/visualize_figure5.py \
    --wandb_entity_for_report $ENTITY \
    --wandb_id_for_report Fig5-$VERSION
fi
