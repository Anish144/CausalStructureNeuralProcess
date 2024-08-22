#!/bin/bash

# List of parameters as space-separated strings
param_list=(
    # "8,8,256,512,16,gplvm_20var",
    # "8,8,512,1024,8,gplvm_20var",
    # "6,6,512,1024,16,gplvm_20var",
    # "6,6,512,1024,8,gplvm_20var",
    # "6,6,256,512,16,gplvm_20var",
    # "4,4,512,1024,16,gplvm_20var",
    "10,6,256,512,16,gplvm_20var",
    "12,4,256,512,16,gplvm_20var",
    "14,2,256,512,16,gplvm_20var",
    "6,10,256,512,16,gplvm_20var",
    "4,12,256,512,16,gplvm_20var",
     # "10,10,512,1024,16,gplvm_20var_ER10",
    # "10,10,512,1024,16,gplvm_20var_ER40",
    # "10,10,512,1024,16,gplvm_20var_ER60",
    # "10,10,512,1024,16,gplvm_20var_ERL10_ERU60",
)
# Iterate over each parameter set
for i in "${param_list[@]}"
do
    # Split the string into individual parameters
    IFS="," read -r NE ND DM DF NH DS <<< "$i"

    # Submit the job with the appropriate variables
    qsub -q hx -v "NE=$NE,ND=$ND,DM=$DM,DF=$DF,NH=$NH,DS=$DS" -N "NE${NE}_ND${ND}_DM${DM}_DF${DF}_NH${NH}_DS${DS}" /gpfs/home/ad6013/Research/ml2_meta_causal_discovery/ml2_meta_causal_discovery/experiments/causal_classification/hx1_large_model.pbs
done
