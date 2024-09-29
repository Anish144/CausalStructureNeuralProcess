#!/bin/bash

# List of parameters as space-separated strings
param_list=(
    # "4,4,512,1024,8,neuralnet_20var_ER20,probabilistic", # This model fits on compute
    # "4,4,512,1024,8,neuralnet_20var_ER40,probabilistic",
    # "4,4,512,1024,8,neuralnet_20var_ER60,probabilistic",
    # "4,4,512,1024,8,neuralnet_20var_ERL20U60,probabilistic",
    # "4,4,512,1024,8,neuralnet_20var_ER20,autoregressive", # This model fits on compute
    # "4,4,512,1024,8,neuralnet_20var_ER40,autoregressive",
    # "4,4,512,1024,8,neuralnet_20var_ER60,autoregressive",
    # "4,4,512,1024,8,neuralnet_20var_ERL20U60,autoregressive",
    # "4,4,512,1024,8,neuralnet_20var_ER20,autoregressive,8", # This model fits on compute
    # "4,4,512,1024,8,neuralnet_20var_ER40,autoregressive,8",
    # "4,4,512,1024,8,neuralnet_20var_ER60,autoregressive,8",
    # "4,4,512,1024,8,neuralnet_20var_ER20,transformer,32", # This model fits on compute
    # "4,4,512,1024,8,neuralnet_20var_ER40,transformer,32",
    # "4,4,512,1024,8,neuralnet_20var_ER60,transformer,32",
    # "4,4,512,1024,8,neuralnet_20var_ERL20U60,transformer",
    # "4,4,512,1024,8,linear_20var_ER20,probabilistic",
    # "4,4,512,1024,8,linear_20var_ER40,probabilistic",
    # "4,4,512,1024,8,linear_20var_ER60,probabilistic",
    # "4,4,512,1024,8,linear_20var_ERL20U60,probabilistic",
    # "4,4,512,1024,8,linear_20var_ER20,autoregressive,8",
    # "4,4,512,1024,8,linear_20var_ER40,autoregressive,8",
    # "4,4,512,1024,8,linear_20var_ER60,autoregressive,8",
    # "4,4,512,1024,8,linear_20var_ER20,transformer,32",
    # "4,4,512,1024,8,linear_20var_ER40,transformer,32",
    # "4,4,512,1024,8,linear_20var_ER60,transformer,32",
    # "4,4,512,1024,8,linear_20var_ERL20U60,transformer",
    # "4,4,512,1024,8,gplvm_neuralnet_20var_ERSFL20U60,probabilistic",
    "4,4,256,512,8,gplvm_20var_ER20,autoregressive,8",
    "4,4,256,512,8,gplvm_20var_ER40,autoregressive,8",
    "4,4,256,512,8,gplvm_20var_ER60,autoregressive,8",
    "4,4,512,1024,8,gplvm_20var_ER20,transformer,32",
    "4,4,512,1024,8,gplvm_20var_ER40,transformer,32",
    "4,4,512,1024,8,gplvm_20var_ER60,transformer,32",
    "4,4,512,1024,8,gplvm_20var_ER20,probabilistic,32",
    "4,4,512,1024,8,gplvm_20var_ER40,probabilistic,32",
    "4,4,512,1024,8,gplvm_20var_ER60,probabilistic,32",
    # "4,4,512,1024,8,gplvm_20var_ERL20U60,probabilistic,32",
    "4,4,256,512,8,gplvm_20var_ER20,transformer,32",
    "4,4,256,512,8,gplvm_20var_ER40,transformer,32",
    "4,4,256,512,8,gplvm_20var_ER60,transformer,32",
    "4,4,256,512,8,gplvm_20var_ER20,probabilistic,32",
    "4,4,256,512,8,gplvm_20var_ER40,probabilistic,32",
    "4,4,256,512,8,gplvm_20var_ER60,probabilistic,32",
    "4,4,256,512,8,neuralnet_20var_ER20,transformer,32",
    "4,4,256,512,8,neuralnet_20var_ER40,transformer,32",
    "4,4,256,512,8,neuralnet_20var_ER60,transformer,32",
    "4,4,256,512,8,neuralnet_20var_ER20,probabilistic,32",
    "4,4,256,512,8,neuralnet_20var_ER40,probabilistic,32",
    "4,4,256,512,8,neuralnet_20var_ER60,probabilistic,32",
)
# Iterate over each parameter set
for i in "${param_list[@]}"
do
    # Split the string into individual parameters
    IFS="," read -r NE ND DM DF NH DS MT BS <<< "$i"

    # Submit the job with the appropriate variables
    qsub -q hx -v "NE=$NE,ND=$ND,DM=$DM,DF=$DF,NH=$NH,DS=$DS,MT=$MT,BS=$BS" -N "MT${MT}_NE${NE}_ND${ND}_DM${DM}_DF${DF}_NH${NH}_DS${DS}_BS${BS}" /gpfs/home/ad6013/Research/CausalStructureNeuralProcess/ml2_meta_causal_discovery/experiments/causal_classification/hx1_large_model.pbs
done
