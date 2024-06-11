import re
import os
import json
from loguru import logger
import numpy as np

from openai import AsyncOpenAI, OpenAI
from typing import List, Optional
from omegaconf import DictConfig, OmegaConf
import transformers
from peft import LoraConfig, TaskType, get_peft_model, LoraModel, PeftModel
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, GenerationConfig, DataCollatorForLanguageModeling
)

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from src.utils.utils import call_openai_chat, set_random_seed

from abc import ABC, abstractmethod 

class FinetunedModel(ABC):
    
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        self.cfg = cfg
        self.model_name: str = None
    
    @abstractmethod
    def generate_solutions(self, prompt: str, num_solution: Optional[int] = None) -> List[str]:
        raise NotImplementedError("Abstract method `generate_solutions` needs implementing")


class FinetunedCodeLlama(FinetunedModel):
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__(cfg)
        
        self.cfg = cfg
        # set_random_seed(cfg.seed)
        self.is_peft_training = hasattr(cfg.model, "lora")
        
        # ! For SFT, it must be `padding_side="right"`.
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path, padding_side="right")
        self.hf_config = AutoConfig.from_pretrained(cfg.model.model_name_or_path)
        logger.info(f"Model default dtype: {self.hf_config.torch_dtype}")
        logger.info(
            f"FloatPoint(FP)-8: {cfg.model.load_in_8bit} " \
            f"FP-4: {cfg.model.load_in_4bit}" \
        )
        self.num_decoding = cfg.evaluation.n_decoding_example
        self.model_name = os.path.basename(cfg.model.model_name_or_path)
        logger.info(f"Model name: {self.model_name}")
        logger.info(f"Model max length: {self.hf_config.max_position_embeddings}")
        
        self.device_map = "auto"
        
        self.quantization_config = BitsAndBytesConfig(
            load_in_8bit=cfg.model.load_in_8bit, 
            load_in_4bit=cfg.model.load_in_4bit, 
            bnb_4bit_quant_type=cfg.model.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.model.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=cfg.model.bnb_4bit_compute_dtype,
        )
        
        # Initialize configs used for generation
        self.generation_config = GenerationConfig(
            do_sample=cfg.generation.do_sample,
            top_k=cfg.generation.top_k,
            top_p=cfg.generation.top_p,
            temperature=cfg.generation.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=cfg.generation.max_new_tokens,
            num_return_sequences=self.num_decoding,
        )
        
        # LoRA config
        if self.is_peft_training:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                # r, alpha, dropout are taken from the experiments on GPT2 original LoRA paper 
                # Note that the original tasks are relation extraction, table-to-text generation, etc.
                r=cfg.model.lora.r, 
                lora_alpha=cfg.model.lora.alpha, 
                lora_dropout=cfg.model.lora.dropout,
                # The following arguments are used to finetune on specific layers
                # Uncomment the following to finetune only the q and v of attention layers
                # This is the default way that the original LoRA paper uses
                target_modules = ["q_proj", "v_proj"],

                # Uncomment the following to finetune only the MLP layers
                # target_modules = ['gate_proj','up_proj','down_proj']
                # Uncomment the following to finetune only on partial layers (lower, middle, upper, etc.)
                # layers_to_transform = list(range(20,32))
            )
            logger.info(f"Training: peft")
            logger.info(f"PEFT config: {self.peft_config}")
        else:
            logger.info(f"Training: fully-finetuning")
        
        if cfg.data.training_example_per_update > 0:
            response_template = "\n### Response:\n"
            response_template_ids = self.tokenizer.encode(response_template, add_special_tokens=False)[2:]
            self.data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=self.tokenizer)
        else:
            self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        
        self.initialize_model()
        
    def _prepare_for_open_generation(self):
        # codellama's padding token id is -1, but the same logic doesn't hold for HF. 
        # Follow adaptation in `Tips` at https://huggingface.co/docs/transformers/main/model_doc/llama2
        self.tokenizer.add_special_tokens({"pad_token":"<pad>"})
        # Just to suppress tokenizer's warning. Supposedly do nothing.
        self.tokenizer.sep_token = self.tokenizer.cls_token = self.tokenizer.mask_token = self.tokenizer.pad_token
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        
    def initialize_model(self):
        self._load_model()
        logger.info(f"Set of parameter dtypes: {set([p.dtype for p in self.model.parameters()])}")
        if self.is_peft_training:
            self._load_peft()
            logger.info(f"Set of parameter dtypes (after load_peft): {set([p.dtype for p in self.model.parameters()])}")
        
        trainable_params, all_param = self.model.get_nb_trainable_parameters() if self.is_peft_training \
            else \
                map(sum, zip(*(np.array([p.requires_grad, 1]) * p.numel() for p in self.model.parameters()))) # one-liner trick to calcualte both the two quantity
        logger.info(
            f"trainable params: {trainable_params:,d} " + \
                f"|| all params: {all_param:,d} " + \
                    f"|| trainable%: {100 * trainable_params / all_param}"
        )
            
    def _load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.model_name_or_path,
            torch_dtype=self.hf_config.torch_dtype,
            quantization_config=self.quantization_config if self.is_peft_training else None,
            device_map=self.device_map,
        )
        self._prepare_for_open_generation()
    
    def _load_peft(self):
        assert self.is_peft_training
        self.model = get_peft_model(self.model, self.peft_config)
    
    def refresh(self):
        # unload peft
        if self.is_peft_training:
            assert isinstance(self.model, PeftModel), "Model is not PEFT."
            self.model = self.model.unload()
            if hasattr(self.model, "peft_config"):
                del self.model.peft_config
            self._load_peft()
        else:
            self._load_model()
    
    def generate_solutions(self, ctx: str, num_solution: Optional[int] = None) -> List[str]:
        
        if num_solution:
            self.generation_config.num_return_sequences = num_solution
    
        inputs = self.tokenizer(ctx, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        generation_output = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
            return_dict_in_generate=True,
        )
        del inputs
        generated_texts = self.tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
        assert all(ctx in t for t in generated_texts), "prompt is expected in `generated_texts`"
        generated_texts = [t.replace(ctx, "") for t in generated_texts]
        return generated_texts