import re
import os
import json
from loguru import logger
import numpy as np

from openai import AsyncOpenAI, OpenAI
from typing import List, Optional
from omegaconf import DictConfig, OmegaConf
import transformers
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, GenerationConfig
)
from src.utils.utils import call_openai_chat

from abc import ABC, abstractmethod 

class PrependModel(ABC):
    
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        self.cfg = cfg
        self.model_name: str = None
    
    @abstractmethod
    def generate_solutions(self, prompt: str, num_solution: Optional[int] = None) -> List[str]:
        raise NotImplementedError("Abstract method `generate_solutions` needs implementing")


class PrependCodeLlama(PrependModel):
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__(cfg)
        
        self.cfg = cfg
        
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path, padding_side="left")
        self.hf_config = AutoConfig.from_pretrained(cfg.model.model_name_or_path)
        logger.info(f"Model default dtype: {self.hf_config.torch_dtype}")
        logger.info(
            f"FloatPoint(FP)-8: {cfg.model.load_in_8bit} " \
            f"FP-4: {cfg.model.load_in_4bit}" \
        )
        logger.info(f"N few-shot examples: {cfg.prompt.num_few_shot_examples}")

        self.model_name = os.path.basename(cfg.model.model_name_or_path)
        logger.info(f"Model name: {self.model_name}")

        # calculate how many decoding rounds for batch decoding
        self.num_decoding = cfg.evaluation.n_decoding_example
        logger.info(f"#Decoding per test: {self.num_decoding}")
        self.n_seq_per_round = min(cfg.generation.num_return_sequences, self.num_decoding)
        self.num_decoding_rounds = int(np.ceil(self.num_decoding / self.n_seq_per_round))


        self.quantization_config = BitsAndBytesConfig(
            load_in_8bit=cfg.model.load_in_8bit, 
            load_in_4bit=cfg.model.load_in_4bit, 
            bnb_4bit_quant_type=cfg.model.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.model.bnb_4bit_use_double_quant,
            # bnb_4bit_compute_dtype=cfg.model.bnb_4bit_compute_dtype,
            bnb_4bit_compute_dtype=self.hf_config.torch_dtype,
        )
        
    def _prepare_for_open_generation(self):
        # codellama's padding token id is -1, but the same logic doesn't hold for HF. 
        # Follow adaptation in `Tips` at https://huggingface.co/docs/transformers/main/model_doc/llama2
        self.tokenizer.add_special_tokens({"pad_token":"<pad>"})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        # Just to suppress tokenizer's warning. Supposedly do nothing.
        self.tokenizer.sep_token = self.tokenizer.cls_token = self.tokenizer.mask_token = self.tokenizer.pad_token
    
    def _load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.model_name_or_path,
            # torch_dtype=hf_config.torch_dtype,
            quantization_config=self.quantization_config,
            device_map="auto",
        )
        self._prepare_for_open_generation()
        
        # Initialize configs used for generation
        self.generation_config = GenerationConfig(
            do_sample=self.cfg.generation.do_sample,
            top_k=self.cfg.generation.top_k,
            top_p=self.cfg.generation.top_p,
            temperature=self.cfg.generation.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.cfg.generation.max_new_tokens,
            num_return_sequences=self.num_decoding,
        )
        
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )
    
    def generate_solutions(self, prompt: str, num_solution: Optional[int] = None) -> List[str]:
        if not hasattr(self, "model"):
            self._load_model()
            
        if num_solution:
            self.generation_config.num_return_sequences = num_solution
        
        sequences = self.pipeline(
            prompt,
            generation_config=self.generation_config,
            return_full_text=False,
        )
        return [seq['generated_text'] for seq in sequences]
    
    
class PrependGPT4(PrependModel):
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__(cfg)
        
        self.cfg = cfg
        self.model_name = "gpt-4"
        self.client = OpenAI()
        self.num_generation_per_prompt = cfg.evaluation.n_decoding_example
        
    def generate_solutions(self, prompt: str, num_solution: Optional[int] = None) -> List[str]:
        num_generation_per_prompt = num_solution or self.num_generation_per_prompt 
        ret = []
        for i in range(num_generation_per_prompt):
            response = call_openai_chat(self.client, "", user_prompt=prompt, model=self.model_name)
            
            ret.append(response.choices[0].message.content)
        return ret
