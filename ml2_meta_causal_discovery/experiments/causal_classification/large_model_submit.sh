#!/bin/bash

# List of parameters as space-separated strings
param_list=(
    # "4,4,512,1024,8,gplvm_20var_ER20,probabilistic,32,2,1000", # RUN
    # "4,4,512,1024,8,gplvm_20var_ER40,probabilistic,32,2,1000", # RUN
    # "4,4,512,1024,8,gplvm_20var_ER60,probabilistic,32,2,1000", # RUN
    # "4,4,512,1024,8,neuralnet_20var_ER20,probabilistic,32,2,1000", # RUN
    # "4,4,512,1024,8,neuralnet_20var_ER40,probabilistic,32,2,1000", # RUN
    # "4,4,512,1024,8,neuralnet_20var_ER60,probabilistic,32,2,1000", # RUN
    # "4,4,512,1024,8,linear_20var_ER20,probabilistic,32,2,1000", # RUN
    # "4,4,512,1024,8,linear_20var_ER40,probabilistic,32,2,1000", # RUN
    # "4,4,512,1024,8,linear_20var_ER60,probabilistic,32,2,1000", # RUN
    # "4,4,512,1024,8,gplvm_20var_ERL20U60,probabilistic,32,2,1000", # RUN
    # "4,4,512,1024,8,neuralnet_20var_ERL20U60,probabilistic,32,2,1000", # RUN
    # "4,4,512,1024,8,linear_20var_ERL20U60,probabilistic,32,2,1000", # RUN
    # "4,4,512,2048,16,gplvm_neuralnet_20var_ERSFL20U60,probabilistic,32,2,500", # RUN
    # "4,4,256,512,8,gplvm_20var_ER20,autoregressive,8", # RUN
    # "4,4,256,512,8,gplvm_20var_ER40,autoregressive,8", # RUN
    # "4,4,256,512,8,gplvm_20var_ER60,autoregressive,8", # RUN
    # "4,4,512,1024,8,gplvm_20var_ER20,transformer,32", # RUN
    # "4,4,512,1024,8,gplvm_20var_ER40,transformer,32", # RUN
    # "4,4,512,1024,8,gplvm_20var_ER60,transformer,32", # RUN
    # "4,4,512,1024,8,gplvm_20var_ER20,probabilistic,32", # RUN
    # "4,4,512,1024,8,gplvm_20var_ER40,probabilistic,32", # RUN
    # "4,4,512,1024,8,gplvm_20var_ER60,probabilistic,32", # RUN
    # "4,4,512,1024,8,gplvm_20var_ERL20U60,probabilistic,32",
    # "4,4,256,512,8,gplvm_20var_ER20,transformer,32", # RUN
    # "4,4,256,512,8,gplvm_20var_ER40,transformer,32", # RUN
    # "4,4,256,512,8,gplvm_20var_ER60,transformer,32", # RUN
    # "4,4,256,512,8,gplvm_20var_ER20,probabilistic,32", # RUN
    # "4,4,256,512,8,gplvm_20var_ER40,probabilistic,32", # RUN
    # "4,4,256,512,8,gplvm_20var_ER60,probabilistic,32", # RUN
    # "4,4,256,512,8,neuralnet_20var_ER20,transformer,32", # RUN
    # "4,4,256,512,8,neuralnet_20var_ER40,transformer,32", # RUN
    # "4,4,256,512,8,neuralnet_20var_ER60,transformer,32", # RUN
    # "4,4,256,512,8,neuralnet_20var_ER20,probabilistic,32", # RUN
    # "4,4,256,512,8,neuralnet_20var_ER40,probabilistic,32", # RUN
    # "4,4,256,512,8,neuralnet_20var_ER60,probabilistic,32", # RUN
    # "4,4,256,512,8,linear_20var_ER20,transformer,32", # RUN
    # "4,4,256,512,8,linear_20var_ER40,transformer,32", # RUN
    # "4,4,256,512,8,linear_20var_ER60,transformer,32", # RUN
    # "4,4,256,512,8,linear_20var_ER20,probabilistic,32", # RUN
    # "4,4,256,512,8,linear_20var_ER40,probabilistic,32", # RUN
    # "4,4,256,512,8,linear_20var_ER60,probabilistic,32", # RUN
    # "4,4,256,512,8,linear_20var_ER60,probabilistic,32", # RUN
#     "4,4,512,2048,16,gplvm_neuralnet_20var_ERSFL20U60,probabilistic,32,2,500", # RUN
#     "4,4,512,2048,16,gplvm_neuralnet_20var_ERSFL20U60,transformer,32,2,500", # RUN
#     "4,4,256,512,8,gplvm_neuralnet_20var_ERSFL20U60,autoregressive,8,5,500", # RUN
    "4,4,512,1024,8,lowdata_neuralnet_20var_ER40,probabilistic,32,2,50",
    "4,4,512,1024,8,lowdata_neuralnet_20var_ER40,transformer,32,2,50",
    "4,4,256,512,8,lowdata_neuralnet_20var_ER40,autoregressive,8,2,50",
    # "4,4,512,4096,16,challenge_training,probabilistic,32,2,1000"
    # "4,4,1024,2048,16,challenge_training,probabilistic,32,2,1000"
    # "4,4,1024,4096,16,challenge_training,probabilistic,32,2,1000"
)

# Iterate over each parameter set
for i in "${param_list[@]}"
do
    # Split the string into individual parameters
    IFS="," read -r NE ND DM DF NH DS MT BS ER SS <<< "$i"

    # Submit the job with the appropriate variables
    qsub -q hx -v "NE=$NE,ND=$ND,DM=$DM,DF=$DF,NH=$NH,DS=$DS,MT=$MT,BS=$BS,ER=$ER,SS=$SS" -N "MT${MT}_NE${NE}_ND${ND}_DM${DM}_DF${DF}_NH${NH}_DS${DS}_BS${BS}" /gpfs/home/ad6013/Research/CausalStructureNeuralProcess/ml2_meta_causal_discovery/experiments/causal_classification/hx1_large_model.pbs
done
