export CUDA_VISIBLE_DEVICES=4
export TOKENIZERS_PARALLELISM=false # this is just to suppress warning

# run with `usage=rand_eval` to sample/generate programs
# run with `usage=rand_exec` to execute generated programs

# Rand Update on FT (U), eval without Update
python src/test_beds/ft_test_bed.py --config-name=lora_zs usage=rand_eval prompt.include_update=False model.model_name_or_path=/u/zliu/datastor1/shared_resources/models/deepseek/deepseek-coder-7b-instruct-v1.5

# Rand Update on FT (PS), eval without Update
python src/test_beds/ft_test_bed.py --config-name=lora_fs usage=rand_eval prompt.include_update=False data.training_example_per_update=2 model.model_name_or_path=/u/zliu/datastor1/shared_resources/models/deepseek/deepseek-coder-7b-instruct-v1.5