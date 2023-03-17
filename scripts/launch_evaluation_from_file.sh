#!/bin/bash

export PYTHONPATH=".:transformers/src:mctx"

#PRINT=true
PRINT=false

# Greedy + BS
array=( "epfl-dlab/understanding-decoding/3410qxd0" "epfl-dlab/understanding-decoding/1y6yme2k" )
array2=( "translation" "translation" )

# Greedy + BS
array=( "epfl-dlab/understanding-decoding/34kq9nsf" "epfl-dlab/understanding-decoding/6yf8p424" )
array2=( "toxicity" "toxicity" )

# Greedy + BS
array=( "epfl-dlab/understanding-decoding/28vcj2b0" "epfl-dlab/understanding-decoding/9lbtlj0l")
array2=( "cie" "cie")

# Greedy + BS
array=( "epfl-dlab/understanding-decoding/2fhbfm2e" "epfl-dlab/understanding-decoding/1k2x9th9" )
array2=( "solubility" "solubility")

# Stochastic Beams (all tasks)
array=("epfl-dlab/understanding-decoding/gigllz7f" "epfl-dlab/understanding-decoding/2mgcqsyk" "epfl-dlab/understanding-decoding/2cqpyxit" "epfl-dlab/understanding-decoding/3vh6a5h8" )
array2=( "translation" "toxicity" "cie" "solubility" )

# MT + VGBS
array=( "epfl-dlab/understanding-decoding/2xogfbel" "epfl-dlab/understanding-decoding/38vrpbjt" "epfl-dlab/understanding-decoding/2gchwiy1" "epfl-dlab/understanding-decoding/3jtpx0zz" "epfl-dlab/understanding-decoding/3g2q6r6x" "epfl-dlab/understanding-decoding/yqyctpgx" )
array2=( "translation_bootstrap" "translation_bootstrap" "translation_bootstrap" "translation_bootstrap" "translation_bootstrap" "translation_bootstrap" )

# MT + MCTS
array=( "epfl-dlab/understanding-decoding/2xdt1zgn" "epfl-dlab/understanding-decoding/7y0pr9sw" "epfl-dlab/understanding-decoding/13niq0lx" "epfl-dlab/understanding-decoding/uykq2pgp" "epfl-dlab/understanding-decoding/37wx7n8u"  "epfl-dlab/understanding-decoding/2yc65rt8" )
array2=( "translation_bootstrap" "translation_bootstrap" "translation_bootstrap" "translation_bootstrap" "translation_bootstrap" "translation_bootstrap" )

# Toxicity + VGBS
array=( "epfl-dlab/understanding-decoding/3a5ydpr4" "epfl-dlab/understanding-decoding/2lveviea" "epfl-dlab/understanding-decoding/2an1uakn" "epfl-dlab/understanding-decoding/1yc972vw" )
array2=( "toxicity_bootstrap" "toxicity_bootstrap" "toxicity_bootstrap" "toxicity_bootstrap" )

# Toxicity + MCTS
array=( "epfl-dlab/understanding-decoding/ilkuz1w4" "epfl-dlab/understanding-decoding/2ah4esvq" "epfl-dlab/understanding-decoding/1jl4dhvh" "epfl-dlab/understanding-decoding/2flwrn8g" )
array2=( "toxicity_bootstrap" "toxicity_bootstrap" "toxicity_bootstrap" "toxicity_bootstrap" )

for i in "${!array[@]}";
do
  if [ $PRINT == true ]
  then
    echo python -m run_evaluation_from_file --wandb_run_path ${array[i]} --overrides evaluation_from_file=${array2[i]}
  else
    python -m run_evaluation_from_file --wandb_run_path ${array[i]} --overrides evaluation_from_file=${array2[i]}
  fi
done
