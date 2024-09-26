#!/bin/bash

# Set default values for data_start and data_end
DATA_START=0
DATA_END=1  # Adjust as needed

# Set the work directory
WORK_DIR="/vol/bitbucket/ad6013/Research/CausalStructureNeuralProcess/ml2_meta_causal_discovery/"

# Define arrays of exp_edges_upper and exp_edges_lower
EXP_EDGES_UPPER=("40" "60")
EXP_EDGES_LOWER=("40" "60")

# Function to run the Python script with given parameters
run_script() {
    local folder_name=$1
    local batch_size=$2

    for i in ${!EXP_EDGES_UPPER[@]}; do
        echo "Running for exp_edges_upper=${EXP_EDGES_UPPER[$i]} and exp_edges_lower=${EXP_EDGES_LOWER[$i]}"
        UPPER=${EXP_EDGES_UPPER[$i]}
        LOWER=${EXP_EDGES_LOWER[$i]}
        python create_save_synth_data.py --work_dir $WORK_DIR \
            --data_start $DATA_START \
            --data_end $DATA_END \
            --batch_size $batch_size \
            --exp_edges_upper $UPPER \
            --exp_edges_lower $LOWER \
            --folder_name $folder_name
    done
}

# Run for folder_name=val with batch_size=100
run_script "val" 100

# Run for folder_name=test with batch_size=25
# run_script "test" 25
