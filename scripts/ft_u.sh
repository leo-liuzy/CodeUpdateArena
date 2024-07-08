export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export TOKENIZERS_PARALLELISM=false # this is just to suppress warning

# run with `usage=eval` to sample/generate programs
# run with `usage=exec` to execute generated programs

# FT (U), eval with no Update 
python src/test_beds/ft_test_bed.py --config-name=lora_zs usage=eval prompt.include_update=False model.model_name_or_path=${MODEL_PATH}


# FT (U), eval with Update 
python src/test_beds/ft_test_bed.py --config-name=lora_zs usage=eval prompt.include_update=True model.model_name_or_path=${MODEL_PATH}