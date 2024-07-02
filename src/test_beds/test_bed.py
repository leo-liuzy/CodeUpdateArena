import os
import json
import glob
import hydra
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from time import time
from omegaconf import DictConfig, OmegaConf
from typing import List, Callable
from loguru import logger
from collections import defaultdict

from src.data.manager_update import UpdateManager
from src.data.manager_prog_syn import ProgSynManager
from src.utils.update import UpdatedFunction
from src.utils.code import (
    Function,
    CheckOutput,
    CheckErrorType,
    concat_and_exec,
    wrap_code_in_short_comment,
    UnitTestsReport
)
from src.utils.prompt import CodeGenTemplate, InstructTemplate
from src.experiments.prepend_model import PrependModel, PrependCodeLlama, PrependGPT4
from src.utils.io_utils import SQLiteCache
from src.utils.eval import pass_at_k
from src.utils.utils import set_random_seed

U_FILE_NAME = "update-content-w_ut.json"
PS_FILE_NAME = "prog_syn-content-w_ut.json"
proj_root = os.path.dirname(__file__) + "/../.."
U_ROOT = f"{proj_root}/data/prelim/CodeUpdateArena-pre-PS"
pilot_root_dir = f"{proj_root}/data/prelim/pilot_run/gpt-4"

def trim_unit_tests(content_dict):
    unit_tests_pass_w_update = content_dict["unit_tests_pass_w_update"]
    unit_tests = content_dict["unit_tests"]
    trimmed_unit_tests = [unit_tests[int(idx)] for idx, pass_w_update in unit_tests_pass_w_update.items() if pass_w_update]
    return trimmed_unit_tests


def prepare_arena_dataset(cfg: DictConfig):    
    proj_root = os.path.dirname(__file__) + "/../.."
    arena_root = f"{proj_root}/{cfg.data.data_dir}"
    logger.info(f"Loading data from: {arena_root}")
    all_updates = list(glob.glob(f"{arena_root}/**/{U_FILE_NAME}", recursive=True))
    
    dataset = []
    
    for update_path in all_updates[:]:
        update_dir = os.path.dirname(update_path)
        specific_update_id = update_dir.replace(f"{arena_root}/", "")
        api_path, update_type_tag, update_idx = specific_update_id.split("/")
        
        updated_function = UpdatedFunction(api_path=api_path)
        
        update_dict = json.load(open(update_path, "r"))
        update_dict["package"] = api_path.split(".")[0]
        update_dict["api_path"] = api_path
        update_dict["update_type"] = update_type_tag
        update_dict["identifier"] = update_idx
        update_dict["old_function_signature"] = updated_function.function_signature
        assert len(update_dict["unit_tests_pass_w_update"]) == len(update_dict["unit_tests"])
        update_dict["unit_tests"] = trim_unit_tests(update_dict)
            
        ps_dirs = next(os.walk(update_dir))[1] # get all immediate sub-folder
        ps_dirs = sorted(ps_dirs)
        assert all(p.startswith("ProgSyn-") for p in ps_dirs)
        ps_dirs = [f"{update_dir}/{p}" for p in ps_dirs]
        
        assert all(os.path.exists(p) for p in ps_dirs)
        
        ps_paths = list(glob.glob(f"{update_dir}/**/{PS_FILE_NAME}", recursive=True))
        assert len(ps_paths) == len(ps_dirs)
        for ps_path in ps_paths:
            prog_syn_dict = json.load(open(ps_path, "r"))
            prog_syn_idx = os.path.basename(os.path.dirname(ps_path))
            prog_syn_id = f"{specific_update_id}/{prog_syn_idx}"
            assert len(prog_syn_dict["unit_tests_pass_w_update"]) == len(prog_syn_dict["unit_tests"])
            prog_syn_dict["unit_tests"] = trim_unit_tests(prog_syn_dict)
            prog_syn_dict["identifier"] = prog_syn_idx
            
            dataset.append({
                "update": update_dict, 
                "prog_syn": prog_syn_dict,
                "specific_update_id": specific_update_id,
                "prog_syn_id": prog_syn_id,
                "package": api_path.split(".")[0]
            })
    return {"test": dataset}


class TestBed:
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        self.cfg = cfg
        self.prompt_template: CodeGenTemplate = None
    
    @classmethod
    def check_unit_tests(
        cls,
        update_manager: UpdateManager,
        imports: List[str],
        unit_tests: List[Function],
        tested_function: Function,
        pass_criterion: Callable = lambda x, y: x and not y,
    ) -> CheckOutput:
        # TODO: refactor unit test checker into a separate object?
        num_unit_tests = len(unit_tests)
        report = UnitTestsReport(num_unit_tests)
        report.tested_function = str(tested_function)
        ret = CheckOutput(type=CheckErrorType.TEST, output=report, check_pass=False)
        
        assert not update_manager.update_type.is_banned_type
        update_enforce_statement = update_manager.update_enforce_statement
        try: 
            
            unit_test_exec_result_pairs = []
            for unit_test in unit_tests:
                # run through each unit tests with and without update
                assert isinstance(unit_test, Function), "Unit tests needs to be a Function object"
                program_in_section = [
                    wrap_code_in_short_comment(
                        "\n".join(imports),
                        "Import statement(s)",
                    ),
                    wrap_code_in_short_comment(
                        str(update_manager.new_impl),
                        "Updated API implementation",
                    ),
                    # enforce / introduce function update to package
                    update_enforce_statement,
                    str(tested_function),
                    str(unit_test),
                    f"{unit_test.function_name}()",
                ]
                update_idx = program_in_section.index(update_enforce_statement)
                program_in_section_wo_update = program_in_section[:update_idx] + program_in_section[update_idx+1:]
                exec_result_w_update = concat_and_exec(*program_in_section)
                exec_result_wo_update = concat_and_exec(*program_in_section_wo_update)
                unit_test_exec_result_pairs.append((exec_result_w_update, exec_result_wo_update))

            
            for t_i, (exec_result_w_update, exec_result_wo_update) in enumerate(unit_test_exec_result_pairs):
                # inspect each unit tests results
                pass_w_update = exec_result_w_update["result"] == "passed"
                report.pass_w_update[t_i] = pass_w_update
                pass_wo_update = exec_result_wo_update["result"] == "passed"
                report.pass_wo_update[t_i] = pass_wo_update
                # we cares more about pass with update; so we don't have exclusive check from another direction
                exclusive_pass = pass_w_update and not pass_wo_update
                report.exclusive_pass[t_i] = exclusive_pass
                report.inclusive_pass[t_i] = pass_w_update and pass_wo_update
                report.unit_tests[t_i] = str(unit_tests[t_i])
                
                report.exec_result_w_update[t_i] = exec_result_w_update['result']
                report.exec_result_wo_update[t_i] = exec_result_wo_update['result']
            
            assert report.finish_testing, "Test needs to finish before checking if passes"
            # there are at least 1 exclusive unit tests
            ret.check_pass = pass_criterion(report.all_pass_w_update, report.all_pass_wo_update)
            
            if not ret.check_pass:
                ret.error_message = report.generate()
            else:
                ret.warn_message = report.generate()
                
        except Exception as error:
            ret.error_message = error
        return ret
    
    @classmethod
    def aggregate_test_reports(
        cls,
        test_reports: List[UnitTestsReport],
        ks: List[int] = [1,2,5,10]
    ):
        # eval_results = []
        
        def helper(helper_test_reports: List[UnitTestsReport], ks: List[int], program_api_usages: List[bool] = None):
            ret = {}
            prefix = ""
            # prefix = "general"
            # if program_api_usages is not None:
            #     prefix = "restricted"
            
            # if program_api_usages is not None:
            #     helper_test_reports = [test_report for test_report, usage in zip(helper_test_reports, program_api_usages) if usage]
            #     ret["restricted_n"] = len(helper_test_reports)
                
            n = len(helper_test_reports)
            # c = sum(test_report.all_pass_w_update for test_report in helper_test_reports)
            c_old_excl = sum(
                test_report.all_pass_wo_update and not test_report.all_pass_w_update
                for test_report in helper_test_reports
            )
            c_new_excl = sum(
                test_report.all_pass_w_update and not test_report.all_pass_wo_update 
                for test_report in helper_test_reports 
            )
            c_unsolved = n - c_old_excl - c_new_excl
            ret[f"{prefix}unsolved (%)"] = c_unsolved / n * 100
            c_old = sum(
                test_report.all_pass_wo_update 
                for test_report in helper_test_reports
            )
            c_new = sum(
                test_report.all_pass_w_update
                for test_report in helper_test_reports 
            )
            for k in ks:
                # ret[f"{prefix}_pass@{k}"] = np.nan if n == 0 or c > n else pass_at_k(n, c, k)
                ret[f"{prefix}pass@{k}(new)"] = np.nan if n == 0 or c_new > n else pass_at_k(n, c_new, k) * 100
                ret[f"{prefix}UPass@{k}"] = np.nan if n == 0 or c_new_excl > n else pass_at_k(n, c_new_excl, k) * 100
                
                ret[f"{prefix}pass@{k}(old)"] = np.nan if n == 0 or c_old > n else pass_at_k(n, c_old, k) * 100

                ret[f"{prefix}pass@{k}(old excl.)"] = np.nan if n == 0 or c_old_excl > n else pass_at_k(n, c_old_excl, k) * 100
    
                # ret[f"{prefix}pass@{k}(unsolved)"] = np.nan if n == 0 or c_unsolved > n else pass_at_k(n, c_unsolved, k)                
                
            accuracies = [test_report.pass_w_update_rate for test_report in helper_test_reports]
            ret[f"{prefix}accuracy_mean"] = np.mean(accuracies) * 100
            ret[f"{prefix}accuracy_std"] = np.std(accuracies) * 100
            unaffected = [
                test_report.num_pass_w_update == test_report.num_pass_wo_update # and test_report.num_pass_wo_update > 0
                for test_report in helper_test_reports
            ]
            affected = [
                test_report.num_pass_w_update != test_report.num_pass_wo_update # and test_report.num_pass_wo_update > 0
                for test_report in helper_test_reports
            ]
            ret[f"{prefix}unaffected"] = np.mean(unaffected) * 100
            ret[f"{prefix}affected"] = np.mean(affected) * 100
            return ret
        
        general_agg_result = helper(test_reports, ks, None)
        
        # TODO: this matching is super stupid... think of a better one.
        # program_api_usages = [f".{update_manager.updated_function.function_name}" in generated_program for generated_program in generated_programs]
        # restricted_agg_result = helper(test_reports, ks, program_api_usages)
        # assert all(k not in general_agg_result for k in restricted_agg_result.keys())
        
        return {
            **general_agg_result,
            # **restricted_agg_result,
        }
    
class FTTestBed(TestBed):
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__(cfg,)
        pass

        
class OneShotTestBed(TestBed):
    def __init__(
        self,
        cfg: DictConfig,
        cache_prefix: str = "",
    ) -> None:
        super().__init__(cfg,)
        self.proj_root = os.path.dirname(__file__) + "/../.."
        self.dataset_dict = prepare_arena_dataset(cfg)
        self.test_dataset = self.dataset_dict["test"]
        assert self.cfg.prompt.num_few_shot_examples == 1, "Not one shot config"
        assert self.cfg.prompt.num_public_unit_tests == 1, "num_public_unit_tests != 1"
        set_random_seed(cfg.seed)
        # self.rand_idx = 6 # np.random.choice(len(self.test_dataset))
        # self.example_datum = self.test_dataset.pop(self.rand_idx)
        self.example_datum = self.prepare_example_datum()
        self.num_public_unit_tests = cfg.prompt.num_public_unit_tests
        
        self.cache_prefix = cache_prefix
        self.prompt_template = InstructTemplate.from_file(f"{self.proj_root}/{cfg.prompt.source}")
        
        
        self.update_cfg = OmegaConf.load(f"{self.proj_root}/data/prelim/configs/update_generation_v2-1.yaml")
        self.progsyn_cfg = OmegaConf.load(f"{self.proj_root}/data/prelim/configs/prog_syn_generation_v2-1.yaml")
    
    def prepare_example_datum(self):
        assert os.path.exists(f"{U_ROOT}/removed_update2ps.json")
        
        removed_update2ps = json.load(open(f"{U_ROOT}/removed_update2ps.json", "r"))
        assert len(removed_update2ps.items()) == 1
        specific_update_id, ps_dirs = list(removed_update2ps.items())[0]
        ps_dirs = sorted(ps_dirs)
        
        update_dir = f"{U_ROOT}/{specific_update_id}"
        api_path, update_type_tag, update_idx = specific_update_id.split("/")
        
        updated_function = UpdatedFunction(api_path=api_path)
        
        update_dict = json.load(open(f"{update_dir}/{U_FILE_NAME}", "r"))
        
        update_dict["package"] = api_path.split(".")[0]
        update_dict["api_path"] = api_path
        update_dict["update_type"] = update_type_tag
        update_dict["update_idx"] = update_idx
        update_dict["old_function_signature"] = updated_function.function_signature
        assert len(update_dict["unit_tests_pass_w_update"]) == len(update_dict["unit_tests"])
        update_dict["unit_tests"] = trim_unit_tests(update_dict)
        
        ps_dir = ps_dirs[0]
        ps_paths = list(glob.glob(f"{pilot_root_dir}/{ps_dir}/**/{PS_FILE_NAME}", recursive=True))
        assert len(ps_paths) == 1
        ps_path = ps_paths[0]
        # assert len(ps_paths) == len(ps_dirs)
        prog_syn_dict = json.load(open(ps_path, "r"))
        prog_syn_idx = os.path.basename(os.path.dirname(ps_path))
        prog_syn_id = f"{specific_update_id}/{prog_syn_idx}"
        assert len(prog_syn_dict["unit_tests_pass_w_update"]) == len(prog_syn_dict["unit_tests"])
        prog_syn_dict["unit_tests"] = trim_unit_tests(prog_syn_dict)
        prog_syn_dict["prog_syn_idx"] = prog_syn_idx
        
        return {
            "update": update_dict, 
            "prog_syn": prog_syn_dict,
            "specific_update_id": specific_update_id,
            "prog_syn_id": prog_syn_id,
            "package": api_path.split(".")[0]
        }
         
    
    def prepare_prompt(self, datum):
        # add option to remove CoT in code
        solution_new = self.example_datum["prog_syn"]["solution_new"]
        # solution_new_no_comment = "\n".join([l for l in solution_new.split("\n") if not l.strip().startswith("#")])
        prompt = self.prompt_template.render(
            # Update information
            old_function_signature=datum["update"]["old_function_signature"],
            new_function_signature=datum["update"]["new_function_signature"],
            update_description=datum["update"]["update_description"],
            update_docstring=datum["update"]["update_docstring"],
            # one-shot example
            example_scenario=self.example_datum["prog_syn"]["scenario"],
            example_problem=self.example_datum["prog_syn"]["problem"],
            example_solution_signature=self.example_datum["prog_syn"]["solution_signature"],
            example_unit_tests="\n\n".join(
                f"# Unit Test {ut_i}\n{ut}" 
                for ut_i, ut in enumerate(self.example_datum["prog_syn"]["unit_tests"][:self.num_public_unit_tests])
            ),
            example_solution=solution_new,
            # test example
            scenario=datum["prog_syn"]["scenario"],
            problem=datum["prog_syn"]["problem"],
            solution_signature=datum["prog_syn"]["solution_signature"],
            unit_tests="\n\n".join(
                f"# Unit Test {ut_i}\n{ut}" 
                for ut_i, ut in enumerate(datum["prog_syn"]["unit_tests"][:self.num_public_unit_tests])
            ),
        )
        return prompt

    def _create_cache(self, model: PrependModel):
        cache_prefix = self.cache_prefix
        if len(cache_prefix) > 0:
            cache_prefix = f"{cache_prefix}_"
        self.cache = SQLiteCache(
            f"{self.proj_root}/caches/{cache_prefix}one_shot_test_bed"\
                f"#pub-ut={self.num_public_unit_tests}_model={model.model_name}.sqlite"
                    # f"inspection-dir={self.cfg.data.inspect_dir}"
        )
        
        self.cached_query_func = self.cache.cache_func(
            model.generate_solutions,
            hash_func=lambda *args, **kwargs: \
                f"{model.model_name}.generate_solutions:" + ', '.join([
                    kwargs["prompt"],
                    str(kwargs.get("num_solution", None)),
                ])
        )
        
        self.cached_exec_func = self.cache.cache_func(
            self.check_unit_tests,
            hash_func=lambda *args, **kwargs: \
                "check_unit_tests:" + ', '.join([
                    kwargs["update_manager"]._api_path + "::" + kwargs["update_manager"]._update_tag,
                    '\n'.join(kwargs["imports"]),
                    '\n'.join(str(f) for f in kwargs["unit_tests"]),
                    str(kwargs["tested_function"]),
                ])
        )
    
    def evaluate_arena(self, model: PrependModel, save_root):
        logger.info(f"Save root: {save_root}")
        self._create_cache(model)
        os.makedirs(save_root, exist_ok=True)
        
        # grouped_eval_results = defaultdict(list)
        for test_datum in tqdm(self.test_dataset[:]):
            prog_syn_id = test_datum["prog_syn_id"]
            save_dir = f"{save_root}/{prog_syn_id}/{model.model_name}"
            logger.info(f"Save to : {save_dir}")
            if os.path.exists(f"{save_dir}/generated_texts.json"):
                continue
            os.makedirs(save_dir, exist_ok=True)
            
            test_prompt = self.prepare_prompt(test_datum)
            open(f"{save_dir}/prompt.txt", "w").write(test_prompt)
            generated_solutions = self.cached_query_func(prompt=test_prompt)
            json.dump(generated_solutions, open(f"{save_dir}/generated_texts.json", "w"))
        
    def execute_arena(self, save_root, model_name):
        print(f"Evaluating results from: {model_name}")
        # grouped_eval_results = defaultdict(list)
        for test_datum in tqdm(self.test_dataset[:]):
            prog_syn_id = test_datum["prog_syn_id"]
            save_dir = f"{save_root}/{prog_syn_id}/{model_name}"
            # print(f"Save dir: {save_dir}")
            assert os.path.exists(f"{save_dir}/generated_texts.json")
            
            generated_solutions = json.load(open(f"{save_dir}/generated_texts.json", "r"))
            generated_programs = list(map(self.prompt_template.solution_extractor, generated_solutions))
            
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
            
    
def generate_text(cfg):
    test_bed = OneShotTestBed(cfg, cache_prefix="neurips_base")
    prepend_model = PrependCodeLlama(cfg)
    # prepend_model = PrependGPT4(cfg)
    proj_root = os.path.dirname(__file__) + "/../.."
    # df, grouped_df = test_bed.evaluate(prepend_model)
    # test_bed.evaluate_arena(prepend_model, save_root=f"{proj_root}/evaluation_output_final/evaluate_base/FT-0_n={cfg.evaluation.n_decoding_example}_lr=0.001_noUpdate")
    test_bed.evaluate_arena(prepend_model, save_root=f"{proj_root}/evaluation_output_final/evaluate_base/prepend_n={cfg.evaluation.n_decoding_example}")
    
    
def run_exec(cfg):
    test_bed = OneShotTestBed(cfg, cache_prefix="neurips_base")
    # model_name = "CodeLlama-7b-Instruct-hf"
    # model_name = "deepseek-coder-7b-instruct-v1.5"
    # model_name = "deepseek-coder-6.7b-instruct"
    model_name = "gpt-4"
    
    test_bed.execute_arena(
        # save_root=f"{proj_root}/evaluation_output/prepend_n={cfg.evaluation.n_decoding_example}",
        save_root=f"{proj_root}/evaluation_output_final/evaluate_base/FT-0_n={cfg.evaluation.n_decoding_example}_lr=0.001_noUpdate",
        model_name=model_name
    )
    
@hydra.main(version_base=None , config_path="../../configs", config_name="base.yaml")
def main(
    cfg
):
    
    generate_text(cfg)
    run_exec(cfg)
    print()
    
if __name__ == "__main__":
    main()
    # print("Eval summary:")
    # print(df.describe().T)
    # print("\n\n")
    # print("Grouped eval summary:")
    # print(pd.DataFrame([x.describe().loc["mean"] for x in grouped_df.values()]).describe().T)
    # print()
    # prepend dataset