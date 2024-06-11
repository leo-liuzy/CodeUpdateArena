# CodeUpdateArena

## Run prepend experiment

```bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python src/test_beds/test_bed.py --config-name=lora_ps prompt.include_update=True model.model_name_or_path=<path to model>
```

## Run finetune experiment
### FT(U)
```bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python src/test_beds/ft_test_bed.py --config-name=lora_u usage=eval prompt.include_update=True model.model_name_or_path=<path to model>
```
### FT(PS)
```bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python src/test_beds/ft_test_bed.py --config-name=lora_ps usage=eval prompt.include_update=False model.model_name_or_path=<path to model>
```

### FT(UPS)
```bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python src/test_beds/ft_test_bed.py --config-name=lora_ps usage=eval prompt.include_update=True model.model_name_or_path=<path to model>
```

## Run finetune experiment for spcificity
### FT(U)
```bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python src/experiments/ft_test_bed.py --config-name=lora_u_specificity usage=specificity prompt.include_update=True model.model_name_or_path=<path to model>
```

### FT(PS)
```bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python src/experiments/ft_test_bed.py --config-name=lora_ps_specificity usage=specificity prompt.include_update=True model.model_name_or_path=<path to model>
```