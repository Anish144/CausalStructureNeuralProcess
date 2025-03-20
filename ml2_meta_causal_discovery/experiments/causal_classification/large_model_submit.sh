#!/bin/bash

# List of parameters as space-separated strings
param_list=(
    "4,4,1024,4096,16,gp_2var_ERL0U1,probabilistic,32,2,1,1000"
)

# Iterate over each parameter set
for i in "${param_list[@]}"
do
    # Split the string into individual parameters
    IFS="," read -r NE ND DM DF NH DS MT BS ER SSMIN SSMAX <<< "$i"

    # Submit the job with the appropriate variables
    qsub -q hx -v "NE=$NE,ND=$ND,DM=$DM,DF=$DF,NH=$NH,DS=$DS,MT=$MT,BS=$BS,ER=$ER,SSMIN=$SSMIN,SSMAX=$SSMAX" -N "MT${MT}_NE${NE}_ND${ND}_DM${DM}_DF${DF}_NH${NH}_DS${DS}_BS${BS}" /gpfs/home/ad6013/Research/CausalStructureNeuralProcess/ml2_meta_causal_discovery/experiments/causal_classification/hx1_large_model.pbs
done
