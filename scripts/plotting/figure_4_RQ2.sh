#!/bin/bash

export PYTHONPATH=".:transformers/src:mctx"

PRINT=false

CI=0.95_confidence_level_50_bootstrap_samples

ENTITY=epfl-dlab
ROOT=epfl-dlab/understanding-decoding
VERSION=paper-v1

if [ $PRINT == true ]
then
  echo python -m scripts.visualize_figure4 \
    --confidence_interval_id $CI \
    --wandb_entity_for_report $ENTITY \
    --wandb_run_paths \
      $ROOT/2xogfbel $ROOT/38vrpbjt $ROOT/2gchwiy1 $ROOT/3jtpx0zz $ROOT/3g2q6r6x $ROOT/yqyctpgx \
      $ROOT/2xdt1zgn $ROOT/7y0pr9sw $ROOT/13niq0lx $ROOT/uykq2pgp $ROOT/37wx7n8u $ROOT/2yc65rt8 \
    --wandb_id_for_report Fig4-$VERSION;

  echo python -m scripts.visualize_figure4 \
    --confidence_interval_id $CI \
    --wandb_entity_for_report $ENTITY \
    --wandb_run_paths \
      $ROOT/3a5ydpr4 $ROOT/1yc972vw $ROOT/2an1uakn $ROOT/2lveviea \
      $ROOT/ilkuz1w4 $ROOT/2ah4esvq $ROOT/1jl4dhvh $ROOT/2flwrn8g \
    --wandb_id_for_report Fig4-$VERSION
else
  python scripts/visualize_figure4.py \
    --confidence_interval_id $CI \
    --wandb_entity_for_report $ENTITY \
    --wandb_run_paths \
      $ROOT/2xogfbel $ROOT/38vrpbjt $ROOT/2gchwiy1 $ROOT/3jtpx0zz $ROOT/3g2q6r6x $ROOT/yqyctpgx \
      $ROOT/2xdt1zgn $ROOT/7y0pr9sw $ROOT/13niq0lx $ROOT/uykq2pgp $ROOT/37wx7n8u $ROOT/2yc65rt8 \
    --wandb_id_for_report Fig4-$VERSION;

  python scripts/visualize_figure4.py \
    --confidence_interval_id $CI \
    --wandb_entity_for_report $ENTITY \
    --wandb_run_paths \
      $ROOT/3a5ydpr4 $ROOT/1yc972vw $ROOT/2an1uakn $ROOT/2lveviea \
      $ROOT/ilkuz1w4 $ROOT/2ah4esvq $ROOT/1jl4dhvh $ROOT/2flwrn8g \
    --wandb_id_for_report Fig4-$VERSION
fi
