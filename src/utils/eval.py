import re
import os
import ast
import json
from typing import Dict, Any, Union, Tuple, List
from collections import defaultdict

import numpy as np
from loguru import logger 
import pandas as pd
from src.utils import Config
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset

from src.execution.safe_execution_util import execute
from src.experiments.ft_model import FinetunedModel, FinetunedCodeLlama
from src.data.code_utils import concat_and_exec
from src.utils.prompt_tool import CodeGenTemplate, InstructTemplate


def pass_at_k(n: int, c: int, k: int):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k: return 1.0 
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def humaneval_execute(generated_texts: List[str], datum: Dict[str, str]):
    
    def get_import(datum):
        prompt_in_lines = datum["prompt"].split("\n")
        function_star_idx = [line.startswith(f"def {datum['entry_point']}") for line in prompt_in_lines].index(True)
        return ("\n".join(prompt_in_lines[:function_star_idx])).strip()
    assert f"def {datum['entry_point']}" in datum["prompt"]
    imports = get_import(datum)
    test_func = datum["test"]
    exec_results = []
    for generated_text in generated_texts:
        generated_code = InstructTemplate.solution_extractor(generated_text)
        file_content = [
            imports,
            generated_code,
            test_func,
            f"check(candidate={datum['entry_point']})",
        ]
        exec_result = concat_and_exec(*file_content)
        exec_result["empty_extract"] = len(generated_code) == 0
        exec_results.append(exec_result)
    return exec_results

def eval_on_humaneval(
    cfg: OmegaConf,
    ft_model: FinetunedModel,
    save_root: str,
    sampled_task_ids = None,
    batch_size: int = 4,
    ks: List[int] = [1,2,5],
    ):
    save_dir = f"{save_root}/{ft_model.model_name}"
    assert len(ft_model.model_name) > 0, "Model name must be non-empty."
    os.makedirs(save_dir, exist_ok=True)
    dataset = load_dataset("openai_humaneval")
    sampled_dataset = dataset
    if sampled_task_ids is not None:
        sampled_dataset = dataset.filter(lambda x: x["task_id"] in sampled_task_ids)
    # TODO: filter datasets to sampled ones
    
    dataloader = DataLoader(sampled_dataset["test"], batch_size=batch_size, shuffle=False)
    prompt_template = InstructTemplate.from_file(cfg.prompt.eval_source)
    
    task_id2genereted_texts = {}
    for batch in tqdm(dataloader):
        ctx_lst = [prompt_template.render(completion_context=prompt) for prompt in batch["prompt"]]
        
        inputs = ft_model.tokenizer(ctx_lst, return_tensors="pt", padding="longest")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        generation_output = ft_model.model.generate(
            **inputs,
            generation_config=ft_model.generation_config,
            return_dict_in_generate=True,
        )
        generated_texts = ft_model.tokenizer.batch_decode(
            generation_output.sequences,
            skip_special_tokens=True
        )
        assert len(generated_texts) % len(ctx_lst) == 0
        n_decoding = ft_model.num_decoding
        assert len(batch["task_id"]) == len(ctx_lst)
        for i in range(len(batch["task_id"])):
            ctx = ctx_lst[i]
            generated_texts_i = generated_texts[i*n_decoding:(i+1)*n_decoding]
            task_id2genereted_texts[batch["task_id"][i]] = [t.replace(ctx, "") for t in generated_texts_i]
        json.dump(
            task_id2genereted_texts, 
            open(f"{save_dir}/task_id2genereted_texts.json", "w")
        )
    df = []
    for task_id, generated_texts in tqdm(task_id2genereted_texts.items()):
        idx = [datum["task_id"] == task_id for datum in dataset["test"]].index(True)
        datum = dataset["test"][idx]

        exec_results = humaneval_execute(generated_texts, datum)
        c = sum([exec_result['result'] == "passed" for exec_result in exec_results])
        num_empty = sum([exec_result['empty_extract'] for exec_result in exec_results])
        ret = {"task_id": task_id, "#empty": num_empty}
        for k in ks:
            ret[f"pass@{k}"] = pass_at_k(n=len(exec_results), c=c, k=k)
        df.append(ret)
    df = pd.DataFrame(df)
    df.to_csv(f"{save_dir}/exec_results.csv", index=False)

