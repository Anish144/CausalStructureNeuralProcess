#!/bin/bash

# Define arrays of hyperparameters to explore
learning_rates=(1e-5 5e-5 1e-4 5e-6)
learning_rate_decay=(0.9 0.95 0.99)
max_epochs_list=(5 10 20)
batch_sizes=(24)
weight_decays=(0 1e-5 1e-4 1e-3)

# Fixed parameters
run_name_base="lab_run"
data_file="challenge_finetune"
num_workers=12
num_layers_encoder=4
num_layers_decoder=4
dim_model=256
dim_feedforward=512
decoder="probabilistic"
seed=0
lr_warmup_ratio=0.1
num_nodes=10
nhead=16
n_perm_samples=200
sinkhorn_iter=1000

# Set the total number of experiments
total_experiments=30  # Adjust this number as needed

# Experiment counter
counter=1

# Loop to run experiments
while [ $counter -le $total_experiments ]; do

  # Randomly select hyperparameters
  lr=${learning_rates[$RANDOM % ${#learning_rates[@]}]}
  lr_decay=${learning_rate_decay[$RANDOM % ${#learning_rate_decay[@]}]}
  epochs=${max_epochs_list[$RANDOM % ${#max_epochs_list[@]}]}
  batch_size=${batch_sizes[$RANDOM % ${#batch_sizes[@]}]}
  weight_decay=${weight_decays[$RANDOM % ${#weight_decays[@]}]}

  # Generate a unique run name for each experiment
  experiment_name="finetuneshuffle_lr${lr}_lrdecay${lr_decay}_epochs${epochs}_bs${batch_size}_wd${weight_decay}"

  # Construct the command
  cmd="python3 /vol/bitbucket/ad6013/Research/CausalStructureNeuralProcess/ml2_meta_causal_discovery/experiments/causal_classification/finetuning/run_finetuning.py \
    --learning_rate=${lr} \
    --batch_size=${batch_size} \
    --max_epochs=${epochs} \
    --run_name=${experiment_name} \
    --data_file=${data_file} \
    --num_workers=${num_workers} \
    --num_layers_encoder=${num_layers_encoder} \
    --num_layers_decoder=${num_layers_decoder} \
    --dim_model=${dim_model} \
    --dim_feedforward=${dim_feedforward} \
    --decoder=${decoder} \
    --seed=${seed} \
    --lr_warmup_ratio=${lr_warmup_ratio} \
    --num_nodes=${num_nodes} \
    --nhead=${nhead} \
    --n_perm_samples=${n_perm_samples} \
    --sinkhorn_iter=${sinkhorn_iter} \
    --weight_decay=${weight_decay} \
    --learning_rate_decay=${lr_decay}"

  # Display the experiment details
  echo "Running experiment ${counter}/${total_experiments}: ${experiment_name}"
  echo "${cmd}"
  echo ""

  # Execute the command
  eval ${cmd}

  echo "Experiment ${counter} completed."
  echo "---------------------------------------------"
  echo ""

  # Increment the counter
  counter=$((counter + 1))
done