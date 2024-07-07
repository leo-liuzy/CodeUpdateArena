export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export TOKENIZERS_PARALLELISM=false # this is just to suppress warning

# run with `usage=eval` to sample/generate programs
# run with `usage=exec` to execute generated programs

# FT (PS), eval with no Update (i.e. prepend docstring in-context)
python src/test_beds/ft_test_bed.py --config-name=lora_fs usage=eval prompt.include_update=False data.training_example_per_update=2 model.model_name_or_path=${MODEL_PATH}