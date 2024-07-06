export CUDA_VISIBLE_DEVICES=4
export TOKENIZERS_PARALLELISM=false # this is just to suppress warning

# FT (U)
python src/test_beds/ft_test_bed.py --config-name=lora_zs_specificity usage=specificity prompt.include_update=False data.training_example_per_update=0 model.model_name_or_path=/u/zliu/datastor1/shared_resources/models/deepseek/deepseek-coder-7b-instruct-v1.5

# FT (PS)
python src/test_beds/ft_test_bed.py --config-name=lora_fs_specificity usage=specificity prompt.include_update=False data.training_example_per_update=2 model.model_name_or_path=/u/zliu/datastor1/shared_resources/models/deepseek/deepseek-coder-7b-instruct-v1.5

# FT (U+PS)
python src/test_beds/ft_test_bed.py --config-name=lora_fs_specificity usage=specificity prompt.include_update=True data.training_example_per_update=2 model.model_name_or_path=/u/zliu/datastor1/shared_resources/models/deepseek/deepseek-coder-7b-instruct-v1.5