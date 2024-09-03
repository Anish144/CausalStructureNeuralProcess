#!/bin/bash

# Fixed parameters
NE=4
ND=4
DM=512
DF=1024
NH=8
DS=gplvm_20var

# List of learning rates to iterate over
learning_rates=(0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001)

# Iterate over each learning rate
for LR in "${learning_rates[@]}"
do
    # Submit the job with the appropriate variables including the learning rate
    qsub -q hx -v "NE=$NE,ND=$ND,DM=$DM,DF=$DF,NH=$NH,DS=$DS,LR=$LR" -N "NE${NE}_ND${ND}_DM${DM}_DF${DF}_NH${NH}_DS${DS}_LR${LR}" /gpfs/home/ad6013/Research/CausalStructureNeuralProcess/ml2_meta_causal_discovery/experiments/causal_classification/hyperparam_cv/learning_rate_submit.pbs
done
