import os
import sys
import json
import glob
import pickle
from time import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from typing import List, Callable, Dict
from collections import defaultdict
from copy import deepcopy
import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

from src.data.manager_update import UpdateManager
from src.data.manager_prog_syn import ProgSynManager
from src.utils.update import UpdatedFunction
from src.utils.code import Function
from src.utils.prompt import InstructTemplate
from src.utils.eval import eval_on_humaneval
from src.utils.utils import set_random_seed
from src.test_beds.test_bed import TestBed
from src.models.ft_model import FinetunedModel, FinetunedCodeLlama



U_FILE_NAME = "update-content-w_ut.json"
PS_FILE_NAME = "prog_syn-content-w_ut.json"
proj_root = os.path.dirname(__file__) + "/../.."


def render(cfg, template, datum, example_datum, is_train):
    return template.render(
        # Optionally update information
        include_update=cfg.prompt.include_update,
        old_function_signature=datum["update"]["old_signature"],
        new_function_signature=datum["update"]["signature"],
        update_description=datum["update"]["description"],
        update_docstring=datum["update"]["docstring"],
        # Example datum to demo task format
        example_scenario=example_datum["prog_syn"]["scenario"],
        example_problem=example_datum["prog_syn"]["problem"],
        example_solution_signature=example_datum["prog_syn"]["solution_signature"],
        example_unit_tests="\n\n".join(
            f"# Unit Test {ut_i}\n{ut}" 
            for ut_i, ut in enumerate(example_datum["prog_syn"]["unit_tests"][:cfg.prompt.num_public_unit_tests])
        ),
        example_solution=example_datum["prog_syn"]["ref_solution"],
        # Test input
        scenario=datum["prog_syn"]["scenario"],
        problem=datum["prog_syn"]["problem"],
        solution_signature=datum["prog_syn"]["solution_signature"],
        unit_tests="\n\n".join(
            f"# Unit Test {ut_i}\n{ut}" 
            for ut_i, ut in enumerate(datum["prog_syn"]["unit_tests"][:cfg.prompt.num_public_unit_tests])
        ),
        solution_code=datum["prog_syn"]["ref_solution"] if is_train else None,
    )
    
def decompose_id(identifier):
    components = identifier.split(":")
    assert all(len(c) >= 2 for c in components)
    if all(c[0] == "[" and c[-1] == "]" for c in components):
        components = [c[1:-1] for c in components]
    return components

def prepare_arena_dataset(cfg: DictConfig):    
    dataset = load_dataset(cfg.data.data_dir)["test"]
    prorcessed_dataset = []
    for datum in dataset:
        specific_update_id = datum['update']["update_id"]
        api_path, _, _ = decompose_id(specific_update_id)
        
        updated_function = UpdatedFunction(api_path=api_path)
        
        update_dict = datum['update']
        update_dict["old_signature"] = updated_function.function_signature
        update_dict["unit_tests"] = update_dict["unit_tests"].split("\n\n")
        update_dict["imports"] = update_dict["imports"].split("\n")
        
        prog_syn_dict = deepcopy(datum)
        del prog_syn_dict["update"]
        
        prog_syn_id = prog_syn_dict["prog_syn_id"]
        prog_syn_dict["unit_tests"] = prog_syn_dict["unit_tests"].split("\n\n")
        prog_syn_dict["imports"] = prog_syn_dict["imports"].split("\n")
            
        prorcessed_dataset.append({
            "update": update_dict, 
            "prog_syn": prog_syn_dict,
            "specific_update_id": specific_update_id,
            "prog_syn_id": prog_syn_id,
            "package": api_path.split(".")[0]
        })
    
    n_train = cfg.data.training_example_per_update
    assert n_train >= 0, "training_example_per_update must be non-negative."
    logger.info(f"#PS for training per update: {n_train}")
    
    example_datum = json.load(open(f"{proj_root}/data/example_datum.json", "r"))
    grouped_dataset = defaultdict(list)
    for datum in prorcessed_dataset:
        grouped_dataset[datum["specific_update_id"]].append(datum)
    train_prompt_template = InstructTemplate.from_file(f"{proj_root}/{cfg.prompt.train_source}")
    eval_prompt_template = InstructTemplate.from_file(f"{proj_root}/{cfg.prompt.eval_source}")
    final_datasets = []
    
    logger.info(f"Preparing knowledge editing dataset")
    for u_id, u_data in tqdm(grouped_dataset.items()):
        u_data = sorted(u_data, key=lambda x: int(decompose_id(x["prog_syn_id"])[-1].split("-")[-1]))
        assert len(u_data) > n_train, f"#PS / update must be more than n_train={n_train}"
        
        for u_datum_i, u_datum in enumerate(u_data):
            if n_train == 0:
                u_dataset = {
                    "train": [deepcopy(u_datum)],
                    "test": [deepcopy(u_datum)],
                }
            else:
                # always take the previous n_train for training 
                if u_datum_i - n_train < 0:
                    u_dataset = {
                        "train": u_data[:u_datum_i] + u_data[u_datum_i-n_train:],
                    }
                else:
                    u_dataset = {
                        "train": u_data[u_datum_i-n_train:u_datum_i],
                    }
                assert len(u_dataset["train"]) == n_train
                u_dataset["test"] = [deepcopy(u_datum)]
            # First make a micro dataset for each specific update
            # # go through every train instance and instantiate prompt
            for train_i in range(len(u_dataset["train"])):
                datum = u_dataset["train"][train_i]
                # add instantiated prompt to each instance
                if example_datum:
                    u_dataset["train"][train_i]["text"] = render(
                        cfg, 
                        train_prompt_template, 
                        datum, 
                        example_datum, 
                        is_train=True
                    )
            # go through every test instance and instantiate prompt
            for test_i in range(len(u_dataset["test"])):
                datum = u_dataset["test"][test_i]
                # add instantiated prompt to each instance
                if example_datum:
                    u_dataset["test"][test_i]["text"] = render(
                        cfg,
                        eval_prompt_template, 
                        datum, 
                        example_datum, 
                        is_train=False
                    )
            # Finally, turn train and test into datasets
            u_dataset["train"] = Dataset.from_pandas(pd.DataFrame(u_dataset["train"]))
            u_dataset["test"] = Dataset.from_pandas(pd.DataFrame(u_dataset["test"]))
            
            final_datasets.append(u_dataset)
    
    return final_datasets
    
class FTTestBed(TestBed):
    def __init__(
        self,
        config_name: str,
        cfg: DictConfig,
    ) -> None:
        super().__init__(cfg,)
        self.proj_root = proj_root = os.path.dirname(__file__) + "/../.."
        self.output_dir = f"caches/{config_name}_k={cfg.data.training_example_per_update}_include-update={cfg.prompt.include_update}_num_epoch={cfg.training.num_epoch}"
        os.makedirs(self.output_dir, exist_ok=True)
        logger.add(f"{self.output_dir}/log.out")
        
        logger.info(f"Command runned: python {' '.join(sys.argv)}")
        logger.info(f"Config name: {config_name}")
        logger.info(f"DEBUGGING: {cfg.debug}")
        logger.info(f"GPU ID(s): {os.environ.get('CUDA_VISIBLE_DEVICES', None)}")
        logger.info(f"Seed: {cfg.seed}")
        
        set_random_seed(cfg.seed)
        self.arena_dataset = prepare_arena_dataset(cfg)
        
        # calculate how many decoding rounds for batch decoding
        self.num_decoding = cfg.evaluation.n_decoding_example
        n_seq_per_round = min(cfg.generation.num_return_sequences, self.num_decoding)
        num_decoding_rounds = int(np.ceil(self.num_decoding / n_seq_per_round))
        logger.info(f"#Decoding per test: {self.num_decoding}")
        logger.info(f"N few-shot examples: {cfg.prompt.num_few_shot_examples}")
        
        # self.prompt_template = InstructTemplate.from_file(f"{self.proj_root}/{cfg.prompt.source}")
        
        self.update_cfg = OmegaConf.load(f"{proj_root}/data/prelim/configs/update_generation_v2-1.yaml")
        
        self.training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            optim="adamw_torch",
            learning_rate=cfg.training.lr,
            weight_decay=cfg.training.decay,
            lr_scheduler_type="constant",
            warmup_ratio=cfg.training.warmup_ratio,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            eval_strategy="no",
            save_strategy="no",
            logging_strategy='steps',
            report_to="none",
            logging_first_step=True,
            logging_steps=1,
        )
        
        # self.accelerator = Accelerator()
        
    def get_trainer(self, u_dataset: DatasetDict, ft_model: FinetunedModel):
        training_args = deepcopy(self.training_args)
        training_args.per_device_train_batch_size=min(self.cfg.training.batch_size, u_dataset['train'].num_rows)
        training_args.per_device_eval_batch_size=min(self.cfg.training.batch_size, u_dataset['train'].num_rows)
        training_args.num_train_epochs=self.cfg.training.num_epoch // u_dataset['train'].num_rows
        
        trainer = SFTTrainer(
            ft_model.model,
            train_dataset=u_dataset['train'],
            # eval_dataset=u_dataset['test'],
            # Change this if your data has a different key than "text" in the dictionary
            dataset_text_field='text',
            tokenizer=ft_model.tokenizer,
            max_seq_length=ft_model.hf_config.max_position_embeddings,
            peft_config=ft_model.peft_config if ft_model.is_peft_training else None,
            data_collator=ft_model.data_collator,
            args=training_args,
        )
        return trainer

    def evaluate_arena(self, ft_model: FinetunedModel, save_root, rerun=False):
        logger.info(f"Save root: {save_root}")
        os.makedirs(save_root, exist_ok=True)
        
        # grouped_eval_results = defaultdict(list)
        for u_dataset in tqdm(self.arena_dataset):
            assert u_dataset["test"].num_rows > 0
            max_token_instance = max(len(ft_model.tokenizer(r['text'])['input_ids']) for r in u_dataset['train'])
            logger.info(f"Max #token / instance in train: {max_token_instance}")
            n_test = u_dataset["test"].num_rows
            assert n_test == 1
            test_datum = u_dataset["test"][0]
            prog_syn_id = test_datum["prog_syn_id"]
            save_dir = f"{save_root}/{prog_syn_id}/{ft_model.model_name}"
            logger.info(f"Save to : {save_dir}")
            os.makedirs(save_dir, exist_ok=True)
            open(f"{save_dir}/train_prompt.txt", "w").write(u_dataset["train"][0]["text"])
            if os.path.exists(f"{save_dir}/generated_texts.json") and not rerun:
                continue
            # Initialize the trainer and train
            try:
                trainer = self.get_trainer(u_dataset, ft_model)
                trainer.train()
            except:
                open(f"{save_root}/{ft_model.model_name}_failed_data.txt", "a").write(
                    f"Test datum: {str(u_dataset['train']['prog_syn_id'])}\n"
                )
            
            # generate with finetuned model and extract solution
            open(f"{save_dir}/test_prompt.txt", "w").write(test_datum["text"])
            generated_texts = ft_model.generate_solutions(test_datum["text"])
            json.dump(generated_texts, open(f"{save_dir}/generated_texts.json", "w"))
        
            # refresh peft for next edit
            ft_model.refresh()
    
    def evaluate_arena_w_random_test_fixed(
        self, 
        ft_model: FinetunedModel,
        save_root, 
        random_update_map,
        rerun=False,
    ):
        logger.info(f"Save root: {save_root}")
        os.makedirs(save_root, exist_ok=True)
        
        update2u_datasets = defaultdict(list)
        for u_dataset in self.arena_dataset:
            prog_syn_id = u_dataset["test"][0]["prog_syn_id"]
            specific_update_id = "/".join(prog_syn_id.split("/")[:-1])
            update2u_datasets[specific_update_id].append(u_dataset)
        update2u_datasets = {
            k: sorted(
                vs, 
                key=lambda x: int(x["test"][0]["prog_syn_id"].split("-")[-1])
            )
            for k, vs in update2u_datasets.items()
        }
        update2random_u_datatset = {
            specific_update_id: # always use the first exmaples in the random update
                update2u_datasets[random_update_id][0] 
            for specific_update_id, random_update_id in random_update_map.items()
        }
        
        # grouped_eval_results = defaultdict(list)
        for u_dataset in tqdm(self.arena_dataset):
            
            max_token_instance = max(len(ft_model.tokenizer(r['text'])['input_ids']) for r in u_dataset['train'])
            logger.info(f"Max #token / instance in train: {max_token_instance}")
            
            n_test = u_dataset["test"].num_rows
            assert n_test == 1
            test_datum = u_dataset["test"][0]
            specific_update_id = test_datum["specific_update_id"]
            prog_syn_id = test_datum["prog_syn_id"]
            
            random_u_dataset = update2random_u_datatset[specific_update_id]
            save_dir = f"{save_root}/{prog_syn_id}/{ft_model.model_name}"
            logger.info(f"Save to : {save_dir}")
            os.makedirs(save_dir, exist_ok=True)
            
            open(f"{save_dir}/train_prompt.txt", "w").write(random_u_dataset["train"][0]["text"])
            # all_generated_texts = list(glob.glob(f"{save_dir}/**/generated_texts-*.json", recursive=True))
            if os.path.exists(
                f"{save_dir}/generated_texts.json"
                ) and not rerun:
                continue
            # Initialize the trainer and train
            try:
                trainer = self.get_trainer(random_u_dataset, ft_model)
                trainer.train()
            except:
                open(f"{save_root}/{ft_model.model_name}_failed_data.txt", "a").write(
                    f"Test datum: {str(random_u_dataset['train']['prog_syn_id'])}\n"
                )
            
            # generate with finetuned model and extract solution
            open(f"{save_dir}/test_prompt.txt", "w").write(test_datum["text"])
            generated_texts = ft_model.generate_solutions(test_datum["text"])
            json.dump(generated_texts, open(f"{save_dir}/generated_texts.json", "w"))
        
            # refresh peft for next edit
            ft_model.refresh()
    
    def evaluate_arena_with_smaller_updates(
        self,
        ft_model: FinetunedModel,
        save_root, 
        sampled_updates_ids,
        rerun=False):
        
        update2u_datasets = defaultdict(list)
        for u_dataset in self.arena_dataset:
            prog_syn_id = u_dataset["test"][0]["prog_syn_id"]
            specific_update_id = "/".join(prog_syn_id.split("/")[:-1])
            update2u_datasets[specific_update_id].append(u_dataset)
        update2u_datasets = {
            k: sorted(
                vs, 
                key=lambda x: int(x["test"][0]["prog_syn_id"].split("-")[-1])
            )
            for k, vs in update2u_datasets.items()
        }
        sampled_u_datasets = [
            update2u_datasets[specific_update_id][0]
            for specific_update_id in sampled_updates_ids
        ]
        
        logger.info(f"Save root: {save_root}")
        os.makedirs(save_root, exist_ok=True)
        
        # grouped_eval_results = defaultdict(list)
        for u_dataset in tqdm(sampled_u_datasets):
            assert u_dataset["test"].num_rows > 0
            max_token_instance = max(len(ft_model.tokenizer(r['text'])['input_ids']) for r in u_dataset['train'])
            logger.info(f"Max #token / instance in train: {max_token_instance}")
            n_test = u_dataset["test"].num_rows
            assert n_test == 1
            test_datum = u_dataset["test"][0]
            prog_syn_id = test_datum["prog_syn_id"]
            save_dir = f"{save_root}/{prog_syn_id}/{ft_model.model_name}"
            logger.info(f"Save to : {save_dir}")
            os.makedirs(save_dir, exist_ok=True)
            open(f"{save_dir}/train_prompt.txt", "w").write(u_dataset["train"][0]["text"])
            if os.path.exists(f"{save_dir}/generated_texts.json") and not rerun:
                continue
            # Initialize the trainer and train
            try:
                trainer = self.get_trainer(u_dataset, ft_model)
                trainer.train()
            except:
                open(f"{save_root}/{ft_model.model_name}_failed_data.txt", "a").write(
                    f"Test datum: {str(u_dataset['train']['prog_syn_id'])}\n"
                )
            
            # generate with finetuned model and extract solution
            open(f"{save_dir}/test_prompt.txt", "w").write(test_datum["text"])
            generated_texts = ft_model.generate_solutions(test_datum["text"])
            json.dump(generated_texts, open(f"{save_dir}/generated_texts.json", "w"))
        
            # refresh peft for next edit
            ft_model.refresh()

    def evaluate_arena_specificity(
        self, 
        ft_model: FinetunedModel,
        save_root, 
        sampled_updates_ids,
        sampled_task_ids = None,
        rerun=False
    ):
        logger.info(f"Save root: {save_root}")
        os.makedirs(save_root, exist_ok=True)
        update2u_datasets = defaultdict(list)
        for u_dataset in self.arena_dataset:
            prog_syn_id = u_dataset["test"][0]["prog_syn_id"]
            specific_update_id = "/".join(prog_syn_id.split("/")[:-1])
            update2u_datasets[specific_update_id].append(u_dataset)
        update2u_datasets = {
            k: sorted(
                vs, 
                key=lambda x: int(x["test"][0]["prog_syn_id"].split("-")[-1])
            )
            for k, vs in update2u_datasets.items()
        }
        sampled_u_datasets = [
            update2u_datasets[specific_update_id][0]
            for specific_update_id in sampled_updates_ids
        ]
        
        for u_dataset in tqdm(sampled_u_datasets):
            assert u_dataset["test"].num_rows > 0
            max_token_instance = max(len(ft_model.tokenizer(r['text'])['input_ids']) for r in u_dataset['train'])
            logger.info(f"Max #token / instance in train: {max_token_instance}")
            n_test = u_dataset["test"].num_rows
            assert n_test == 1
            test_datum = u_dataset["test"][0]
            specific_update_id = test_datum["specific_update_id"]
            logger.info(f"Save to: {save_root}/{specific_update_id}")
            os.makedirs(f"{save_root}/{specific_update_id}", exist_ok=True)
            # save_dir = f"{save_root}/{prog_syn_id}/{ft_model.model_name}"
            save_dir = f"{save_root}/{specific_update_id}/{ft_model.model_name}"
            open(f"{save_root}/{specific_update_id}/train_prompt.txt", "w").write(u_dataset["train"][0]["text"])
            open(f"{save_root}/{specific_update_id}/test_prompt.txt", "w").write(test_datum["text"])
            logger.info(f"Evaluating {specific_update_id}")
                
            if all(
                os.path.exists(f"{save_dir}/{filename}") 
                for filename in ["task_id2genereted_texts.json", "exec_results.csv"]
            ):
                continue
            # Initialize the trainer and train
            # try:
            trainer = self.get_trainer(u_dataset, ft_model)
            trainer.train()
            
            eval_on_humaneval(
                self.cfg,
                ft_model,
                save_root=f"{save_root}/{specific_update_id}", 
                sampled_task_ids=sampled_task_ids
            )
        
            # refresh peft for next edit
            ft_model.refresh()
    
    def evaluate_arena_specificity_base(
        self, 
        ft_model: FinetunedModel,
        save_root, 
        sampled_updates_ids,
        sampled_task_ids = None,
        rerun=False
    ):
        logger.info(f"Save root: {save_root}")
        os.makedirs(save_root, exist_ok=True)
        update2u_datasets = defaultdict(list)
        for u_dataset in self.arena_dataset:
            prog_syn_id = u_dataset["test"][0]["prog_syn_id"]
            specific_update_id = "/".join(prog_syn_id.split("/")[:-1])
            update2u_datasets[specific_update_id].append(u_dataset)
        update2u_datasets = {
            k: sorted(
                vs, 
                key=lambda x: int(x["test"][0]["prog_syn_id"].split("-")[-1])
            )
            for k, vs in update2u_datasets.items()
        }
        sampled_u_datasets = [
            update2u_datasets[specific_update_id][0]
            for specific_update_id in sampled_updates_ids
        ]
        
        for u_dataset in tqdm(sampled_u_datasets[:3]):
            assert u_dataset["test"].num_rows > 0
            max_token_instance = max(len(ft_model.tokenizer(r['text'])['input_ids']) for r in u_dataset['train'])
            logger.info(f"Max #token / instance in train: {max_token_instance}")
            n_test = u_dataset["test"].num_rows
            assert n_test == 1
            test_datum = u_dataset["test"][0]
            specific_update_id = test_datum["specific_update_id"]
            logger.info(f"Save to: {save_root}/{specific_update_id}")
            os.makedirs(f"{save_root}/{specific_update_id}", exist_ok=True)
            # save_dir = f"{save_root}/{prog_syn_id}/{ft_model.model_name}"
            save_dir = f"{save_root}/{specific_update_id}/{ft_model.model_name}"
            open(f"{save_root}/{specific_update_id}/train_prompt.txt", "w").write(u_dataset["train"][0]["text"])
            open(f"{save_root}/{specific_update_id}/test_prompt.txt", "w").write(test_datum["text"])
            logger.info(f"Evaluating {specific_update_id}")
            
            if all(
                os.path.exists(f"{save_dir}/{filename}") 
                for filename in ["task_id2genereted_texts.json", "exec_results.csv"]
            ):
                continue
            # Initialize the trainer and train
            # try:
            trainer = self.get_trainer(u_dataset, ft_model)
            trainer.train()
            
            eval_on_humaneval(
                self.cfg,
                ft_model,
                save_root=f"{save_root}/{specific_update_id}", 
                sampled_task_ids=sampled_task_ids
            )
        
            # refresh peft for next edit
            ft_model.refresh()
    
    def execute_arena(self, save_root, model_name):
        print(f"Evaluating results from: {model_name}")
        print(f"Saving to: {save_root}")
        assert os.path.exists(save_root), f"Save dir: {save_root}"
        
        for u_dataset in tqdm(self.arena_dataset):
            assert u_dataset["test"].num_rows > 0
            n_test = u_dataset["test"].num_rows
            
            for test_i in range(n_test):
                # logger.info(f"Evaluating {test_i + 1} / {n_test} of {u_i}th update eval")
                test_datum = u_dataset["test"][test_i]
                prog_syn_id = test_datum["prog_syn_id"]
                save_dir = f"{save_root}/{prog_syn_id}/{model_name}"
                
                if not os.path.exists(f"{save_dir}/generated_texts.json"):
                    continue
                
                assert os.path.exists(f"{save_dir}/generated_texts.json")
                
                generated_solutions = json.load(open(f"{save_dir}/generated_texts.json", "r"))
                generated_programs = list(map(InstructTemplate.solution_extractor, generated_solutions))
                
                if all(
                    os.path.exists(f"{save_dir}/{filename}") 
                    for filename in ["test_reports.pkl", "eval_result.json"]
                ):
                    continue
                
                u_manager = UpdateManager(
                    cfg=self.update_cfg, 
                    api_path=test_datum["update"]["api_path"], 
                    update_tag=test_datum["update"]["update_type"]
                )
                u_manager.load_from_dict(test_datum["update"])
                unit_test_functions = [Function(unit_test) for unit_test in test_datum["prog_syn"]["unit_tests"]]
                test_reports = []
                for generated_program in generated_programs:
                    # assert generated_program, "Failed to extract generated_program"
                    test_report = self.check_unit_tests(
                        update_manager=u_manager, 
                        imports=test_datum["prog_syn"]["imports"], 
                        unit_tests=unit_test_functions, 
                        tested_function=generated_program,
                    )
                    test_reports.append(test_report.output)
                pickle.dump(test_reports, open(f"{save_dir}/test_reports.pkl", "wb"))
                
                eval_result = self.aggregate_test_reports(test_reports)
                eval_result["api_path"] = test_datum["update"]["api_path"]
                eval_result["update_type"] = test_datum["update"]["update_type"]
                eval_result["package"] = test_datum["package"]
                eval_result["specific_update_id"] = test_datum["specific_update_id"]
                eval_result["prog_syn_id"] = test_datum["prog_syn_id"]
                json.dump(eval_result, open(f"{save_dir}/eval_result.json", "w"))
        
    
    def evaluate(self, ft_model: FinetunedModel, run_exec: bool = False):
        dfs = []
        
        for u_i, (u_name, u_dataset) in tqdm(enumerate(self.named_dataset)):
            # go through every test instances
            assert u_dataset["test"].num_rows > 0
            max_token_instance = max(len(ft_model.tokenizer(r['text'])['input_ids']) for r in u_dataset['train'])
            logger.info(f"Max #token / instance in train: {max_token_instance}")
            
            # Initialize the trainer and train

            trainer = self.get_trainer(u_dataset, ft_model)
            trainer.train()
            
            
            u_eval_results = []
            n_test = u_dataset["test"].num_rows
            for test_i in tqdm(range(n_test), desc=f"{u_i}th update eval"):
                logger.info(f"Evaluating {test_i} / {n_test} of {u_i}th update eval")
                test_datum = u_dataset["test"][test_i]
                
                # generate with finetuned model and extract solution
                generated_texts = ft_model.generate_solutions(test_datum["text"])
                generated_programs = list(map(InstructTemplate.solution_extractor, generated_texts))
                
                individual_eval_results = self.evaluate_generated_programs(generated_programs, test_datum, run_exec)
                individual_eval_results["trainer_logs"] = trainer.state.log_history
                individual_eval_results["generated_texts"] = generated_texts
                u_eval_results.append(individual_eval_results)
            
            # record eval results for 1 update
            df = pd.DataFrame(u_eval_results)
            dfs.append(df)
            pickle.dump(dfs, open(f"{self.output_dir}/dfs_w{'' if run_exec else 'o'}_exec.pkl", "wb"))
            # prepare ft model for training on next update
            ft_model.refresh()
        logger.info(f"Result saved to {self.output_dir}")
        
    def evaluate_generated_programs(self, generated_programs: List[str], datum: Dict, run_exec: bool = False):
        eval_results = {}
        
        if run_exec:
            u_manager = UpdateManager(cfg=self.update_cfg, api_path=datum["update"]["api_path"], update_tag=datum["update"]["update_type"])
            u_manager.load_from_dict(datum["update"])

            # Run every program against unit tests
            unit_test_functions = [Function(unit_test) for unit_test in datum["prog_syn"]["unit_tests"]]
            test_reports = []
            for generated_program in generated_programs:
                if generated_program is None:
                    generated_program = ""
                test_report = self.check_unit_tests(update_manager=u_manager, imports=datum["prog_syn"]["imports"], unit_tests=unit_test_functions, tested_function=generated_program,)
                test_reports.append(test_report.output)
            eval_results = self.aggregate_test_reports(test_reports)
        # record
        eval_results["text"] = datum["text"]
        eval_results["generated_programs"] = generated_programs
        eval_results["api_path"] = datum["update"]["api_path"]
        eval_results["update_type"] = datum["update"]["update_type"]
        eval_results["update_id"] = datum["update"]["identifier"]
        eval_results["identifier"] = datum["identifier"]
        return eval_results

def rand_evaluate(cfg):
    running_config = HydraConfig.get()
    config_name = Path(running_config.job.config_name).stem
    ft_testbed = FTTestBed(config_name, cfg) 
    ft_model = FinetunedCodeLlama(cfg)

    proj_root = os.path.dirname(__file__) + "/../.."
    
    save_root=f"{proj_root}/evaluation_output/rand-FT-{cfg.data.training_example_per_update}_n={cfg.evaluation.n_decoding_example}"
    save_root += f"_lr={cfg.training.lr}"
    if not cfg.prompt.include_update:
        save_root += "_noUpdate"
    
    random_update_map = json.load(open(f"{proj_root}/evaluation_output/random_update_map_final.json", "r"))
    
    ft_testbed.evaluate_arena_w_random_test_fixed(
        ft_model,
        save_root=save_root,
        random_update_map=random_update_map,
    )

def rand_execute(cfg):
    running_config = HydraConfig.get()
    config_name = Path(running_config.job.config_name).stem
    ft_testbed = FTTestBed(config_name, cfg)
    
    model_name = os.path.basename(cfg.model.model_name_or_path) 

    proj_root = os.path.dirname(__file__) + "/../.."
    
    save_root=f"{proj_root}/evaluation_output/rand-FT-{cfg.data.training_example_per_update}_n={cfg.evaluation.n_decoding_example}"
    save_root += f"_lr={cfg.training.lr}"
    if not cfg.prompt.include_update:
        save_root += "_noUpdate"
    
    ft_testbed.execute_arena(
        save_root=save_root,
        model_name=model_name,
    )

def evaluate(cfg):
    running_config = HydraConfig.get()
    config_name = Path(running_config.job.config_name).stem
    ft_testbed = FTTestBed(config_name, cfg)
    ft_model = FinetunedCodeLlama(cfg)

    model_name = os.path.basename(cfg.model.model_name_or_path)
    assert model_name == ft_model.model_name
    
    proj_root = os.path.dirname(__file__) + "/../.."
    
    save_root=f"{proj_root}/evaluation_output/FT-{cfg.data.training_example_per_update}_n={cfg.evaluation.n_decoding_example}"
    if "Eval-wo-Update" in config_name:
        save_root += "_Eval-wo-Update"
    save_root += f"_lr={cfg.training.lr}"
    if not cfg.prompt.include_update:
        save_root += "_noUpdate"
    print(save_root)
    ft_testbed.evaluate_arena(
        ft_model,
        save_root=save_root
    )


def execute(cfg):
    model_name = os.path.basename(cfg.model.model_name_or_path)
    running_config = HydraConfig.get()
    config_name = Path(running_config.job.config_name).stem
    ft_testbed = FTTestBed(config_name, cfg)
    
    save_root=f"{proj_root}/evaluation_output/FT-{cfg.data.training_example_per_update}_n={cfg.evaluation.n_decoding_example}"
    if "Eval-wo-Update" in config_name:
        save_root += "_Eval-wo-Update"
    save_root += f"_lr={cfg.training.lr}"
    if not cfg.prompt.include_update:
        save_root += "_noUpdate"
        
    ft_testbed.execute_arena(
        save_root=save_root,
        model_name=model_name
    )


def specificity(cfg):
    running_config = HydraConfig.get()
    config_name = Path(running_config.job.config_name).stem
    ft_testbed = FTTestBed(config_name, cfg)
    ft_model = FinetunedCodeLlama(cfg)

    proj_root = os.path.dirname(__file__) + "/../.."
    
    sampled_updates_ids = json.load(open(f"{proj_root}/data/sampled_update_ids.json", "r"))
    sampled_humaneval_task_ids = json.load(open(f"{proj_root}/data/sampled_humaneval_task_ids.json", "r"))
    
    save_root=f"{proj_root}/evaluation_output/specificity/FT-{cfg.data.training_example_per_update}_n={cfg.evaluation.n_decoding_example}"
    save_root += f"_lr={cfg.training.lr}"
        
    if not cfg.prompt.include_update:
        save_root += "_noUpdate"
    
    ft_testbed.evaluate_arena_specificity(
        ft_model,
        save_root=save_root,
        sampled_updates_ids=sampled_updates_ids[:25],
        sampled_task_ids=sampled_humaneval_task_ids[:82],
    )

def specificity_base(cfg):
    running_config = HydraConfig.get()
    config_name = Path(running_config.job.config_name).stem
    ft_testbed = FTTestBed(config_name, cfg)

    ft_model = FinetunedCodeLlama(cfg)

    proj_root = os.path.dirname(__file__) + "/../.."
    
    sampled_updates_ids = json.load(open(f"{proj_root}/data/sampled_update_ids.json", "r"))
    sampled_humaneval_task_ids = json.load(open(f"{proj_root}/data/sampled_humaneval_task_ids.json", "r"))
    
    save_root=f"{proj_root}/evaluation_output/specificity_base/FT-{cfg.data.training_example_per_update}_n={cfg.evaluation.n_decoding_example}"
    save_root += f"_lr={cfg.training.lr}"
    if not cfg.prompt.include_update:
        save_root += "_noUpdate"
    
    ft_testbed.evaluate_arena_specificity_base(
        ft_model,
        save_root=save_root,
        sampled_updates_ids=sampled_updates_ids[:25],
        sampled_task_ids=sampled_humaneval_task_ids[:82],
    )

def hyper_specificity(cfg):
    running_config = HydraConfig.get()
    config_name = Path(running_config.job.config_name).stem
    ft_testbed = FTTestBed(config_name, cfg)
    ft_model = FinetunedCodeLlama(cfg)

    proj_root = os.path.dirname(__file__) + "/../.."
    
    sampled_updates_ids = json.load(open(f"{proj_root}/data/sampled_update_ids.json", "r"))
    sampled_humaneval_task_ids = json.load(open(f"{proj_root}/data/sampled_humaneval_task_ids.json", "r"))
    
    
    save_root=f"{proj_root}/hyper_search/specificity/FT-{cfg.data.training_example_per_update}_n={cfg.evaluation.n_decoding_example}_lr={cfg.training.lr}"
    if not cfg.prompt.include_update:
        save_root += "_noUpdate"
    
    ft_testbed.evaluate_arena_specificity(
        ft_model,
        save_root=save_root,
        sampled_updates_ids=sampled_updates_ids[:10],
        sampled_task_ids=sampled_humaneval_task_ids[:20],
    )

def hyper_search(cfg):
    running_config = HydraConfig.get()
    config_name = Path(running_config.job.config_name).stem
    ft_testbed = FTTestBed(config_name, cfg)
    ft_model = FinetunedCodeLlama(cfg)

    proj_root = os.path.dirname(__file__) + "/../.."
    
    sampled_updates_ids = json.load(open(f"{proj_root}/data/sampled_update_ids.json", "r"))

    save_root=f"{proj_root}/hyper_search/FT-{cfg.data.training_example_per_update}_n={cfg.evaluation.n_decoding_example}_lr={cfg.training.lr}"
    if not cfg.prompt.include_update:
        save_root += "_noUpdate"

    ft_testbed.evaluate_arena_with_smaller_updates(
        ft_model,
        save_root=save_root,
        sampled_updates_ids=sampled_updates_ids[:10]
    )
    model_name = os.path.basename(cfg.model.model_name_or_path)
    ft_testbed.execute_arena(
        save_root=save_root,
        model_name=model_name
    )
    

@hydra.main(version_base=None , config_path="../../configs", config_name="lora_zs.yaml")
def main(
    cfg
):
    if cfg.usage == "eval":
        evaluate(cfg)
    elif cfg.usage == "rand_eval":
        rand_evaluate(cfg)
    elif cfg.usage == "exec":
        execute(cfg)
    elif cfg.usage == "rand_exec":
        rand_execute(cfg)
    elif cfg.usage == "specificity":
        specificity(cfg)
    elif cfg.usage == "specificity_base":
        specificity_base(cfg)
    elif cfg.usage == "hyper_specificity":
        hyper_specificity(cfg)
    else:
        assert cfg.usage == "hyper"
        hyper_search(cfg)
    
if __name__ == "__main__":
    main()
