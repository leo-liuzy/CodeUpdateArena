export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export TOKENIZERS_PARALLELISM=false # this is just to suppress warning

# run with `usage=eval` to sample/generate programs
# run with `usage=exec` to execute generated programs

python src/test_beds/test_bed.py --config-name=prepend usage=exec model.model_name_or_path=${MODEL_PATH}