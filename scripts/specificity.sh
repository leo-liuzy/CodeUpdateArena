export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export TOKENIZERS_PARALLELISM=false # this is just to suppress warning

# FT (U)
python src/test_beds/ft_test_bed.py --config-name=lora_zs_specificity usage=specificity prompt.include_update=False data.training_example_per_update=0 model.model_name_or_path=${MODEL_PATH}

# FT (PS)
# python src/test_beds/ft_test_bed.py --config-name=lora_fs_specificity usage=specificity prompt.include_update=False data.training_example_per_update=2 model.model_name_or_path=${MODEL_PATH}

# FT (U+PS)
# python src/test_beds/ft_test_bed.py --config-name=lora_fs_specificity usage=specificity prompt.include_update=True data.training_example_per_update=2 model.model_name_or_path=${MODEL_PATH}