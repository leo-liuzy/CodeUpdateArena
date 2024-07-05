export CUDA_VISIBLE_DEVICES=4
export TOKENIZERS_PARALLELISM=false # this is just to suppress 

python src/test_beds/ft_test_bed.py --config-name=lora_zs usage=eval prompt.include_update=False model.model_name_or_path=/u/zliu/datastor1/shared_resources/models/deepseek/deepseek-coder-7b-instruct-v1.5