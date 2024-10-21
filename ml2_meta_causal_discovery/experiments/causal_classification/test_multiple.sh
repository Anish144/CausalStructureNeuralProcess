python3 test_causal_classify.py \
    --work_dir="/vol/bitbucket/ad6013/Research/CausalStructureNeuralProcess/ml2_meta_causal_discovery" \
    --decoder="probabilistic" \
    --model_list="AVERAGING_probabilistic_gplvm_20var_ER20_NH8_NE4_ND4_DM512_DF1024_BS32_SS1000,AVERAGING_probabilistic_gplvm_20var_ER40_NH8_NE4_ND4_DM512_DF1024_BS32_SS1000,AVERAGING_probabilistic_gplvm_20var_ER60_NH8_NE4_ND4_DM512_DF1024_BS32_SS1000" \
    --num_nodes=20 \


# --model_list="transformer_linear_20var_ER20_NH8_NE4_ND4_DM512_DF1024,transformer_linear_20var_ER40_NH8_NE4_ND4_DM512_DF1024,transformer_linear_20var_ER60_NH8_NE4_ND4_DM512_DF1024,transformer_linear_20var_ERL20U60_NH8_NE4_ND4_DM512_DF1024"