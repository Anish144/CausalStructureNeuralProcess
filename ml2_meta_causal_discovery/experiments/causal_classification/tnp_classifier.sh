#!/bin/sh
# Description: Script to train a TNP classifier on the synthetic data.
source /vol/cuda/11.8.0/setup.sh
export CUDA_VISIBLE_DEVICES=0


# python3 train_causal_classify.py \
#     --learning_rate_max=5e-4 \
#     --batch_size=32 \
#     --max_epochs=2 \
#     --run_name="20var_auto_posembed" \
#     --data_file="gplvm_20var" \
#     --num_workers=12 \
#     --num_layers_encoder=4 \
#     --num_layers_decoder=4 \
#     --dim_model=128 \
#     --dim_feedforward=128 \
#     --decoder="autoregressive" \
#     --seed=0 \
#     --lr_warmup_steps=5000 \
#     --nhead=8 \
#     --num_nodes=20 \
#     --use_positional_encoding \


# python3 train_causal_classify.py \
#     --learning_rate_max=5e-4 \
#     --batch_size=32 \
#     --max_epochs=2 \
#     --run_name="3var_transformer" \
#     --data_file="gplvm_3var_unifprior" \
#     --num_workers=12 \
#     --num_layers_encoder=8 \
#     --num_layers_decoder=8 \
#     --dim_model=128 \
#     --dim_feedforward=128 \
#     --decoder="transformer" \
#     --seed=0 \
#     --lr_warmup_steps=5000 \


# python3 train_causal_classify.py \
#     --learning_rate_max=5e-4 \
#     --batch_size=32 \
#     --max_epochs=2000 \
#     --run_name="3var_prob_10sample1noiseHARD_sigmoidmultiplymask_TNPperm_1temp_NOCURL_LQLinked_try3_debug" \
#     --data_file="gplvm_3var_unifprior" \
#     --num_workers=12 \
#     --num_layers_encoder=8 \
#     --num_layers_decoder=8 \
#     --dim_model=128 \
#     --dim_feedforward=128 \
#     --decoder="probabilistic" \
#     --seed=3 \


python3 train_causal_classify.py \
    --learning_rate=2e-4 \
    --batch_size=32 \
    --max_epochs=2 \
    --run_name="test_large_model_float32" \
    --data_file="gplvm_neuralnet_20var_ERSFL20U60" \
    --num_workers=12 \
    --num_layers_encoder=4 \
    --num_layers_decoder=4 \
    --dim_model=128 \
    --dim_feedforward=256 \
    --decoder="probabilistic" \
    --seed=0 \
    --lr_warmup_ratio=0.1 \
    --num_nodes=20 \
    --nhead=8 \
    --n_perm_samples=25 \
    --sinkhorn_iter=1000 \
