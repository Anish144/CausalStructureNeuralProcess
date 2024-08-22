#!/bin/bash

# List of parameters as space-separated strings
param_list=(
    "10,10,256,512,8,gplvm_20var",
    "10,10,512,1024,8,gplvm_20var",
    "8,8,256,512,8,gplvm_20var",
    "8,8,512,1024,8,gplvm_20var",
    "8,8,1024,2048,8,gplvm_20var",
    "6,6,256,512,8,gplvm_20var",
    "6,6,512,1024,8,gplvm_20var",
    "6,6,1024,2048,8,gplvm_20var",
)
# Iterate over each parameter set
for i in "${param_list[@]}"
do
    # Split the string into individual parameters
    IFS="," read -r NE ND DM DF NH DS <<< "$i"

    # Submit the job with the appropriate variables
    qsub -q hx -v "NE=$NE,ND=$ND,DM=$DM,DF=$DF,NH=$NH,DS=$DS" -N "NE${NE}_ND${ND}_DM${DM}_DF${DF}_NH${NH}_DS${DS}" /gpfs/home/ad6013/Research/CausalStructureNeuralProcess/ml2_meta_causal_discovery/experiments/causal_classification/hx1_large_model.pbs
done
