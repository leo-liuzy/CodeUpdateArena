import json
import os
import re
from typing import Dict, List, Callable
import ast
from openai import AsyncOpenAI, OpenAI
from omegaconf import OmegaConf
import pickle
import numpy as np

from src.utils.update import Action, Aspect, Place, UpdatedFunction, UpdateType
from src.utils.code import (
    CheckErrorType, CheckOutput,
    CheckOutputSuite, Function,
    UnitTestsReport, RejectionSampleOutput,
    concat_and_exec,
    wrap_code_in_short_comment
)
from src.utils.utils import call_openai_chat, OPENAI_MODEL
from src.data.prompt_update import PYTHON_INDENT
from src.data.manager_update import Manager, UpdateManager


CLIENT = OpenAI()


class ProgSynManager(Manager):
    
    WRAPPER_L = WRAPPER_R = "@"
    
    def __init__(
        self,
        cfg: OmegaConf,
        update_manager: UpdateManager,
        num_params: int,
        ) -> None:
        # Update input
        self.cfg = cfg
        OmegaConf.resolve(cfg)
        self.update_manager: UpdateManager = update_manager
        self.updated_function = self.update_manager.updated_function
        self.answer_placeholder = "# " + self.WRAPPER_L + "ANSWER" + self.WRAPPER_R
        self.assert_placeholder = "# " + self.WRAPPER_L + "ASSERT" + self.WRAPPER_R
        self.num_params = num_params
        
        # first make sure the content of the dict ready for code checking
        self.prog_syn_dict = None
        
    def check_unit_tests(
        self,
        tested_function: Function,
        pass_criterion: Callable = lambda x, y: x and not y,
    ) -> CheckOutput:
        # TODO: refactor unit test checker into a separate object
        report = UnitTestsReport(self.num_unit_tests)
        ret = CheckOutput(type=CheckErrorType.TEST, output=report, check_pass=False)
        
        assert not self.update_manager.update_type.is_banned_type
        update_enforce_statement = self.update_manager.update_enforce_statement
        try: 
            
            unit_test_exec_result_pairs = []
            for unit_test in self.unit_tests:
                # run through each unit tests with and without update
                assert isinstance(unit_test, Function)
                program_in_section = [
                    self.imports,
                    str(self.update_manager.new_impl),
                    str(tested_function),  # TODO: make update_enforce_statement before tested_function
                    # enforce / introduce function update to package
                    update_enforce_statement,
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
                report.unit_tests[t_i] = str(self.unit_tests[t_i])
                
                report.exec_result_w_update[t_i] = exec_result_w_update['result']
                report.exec_result_wo_update[t_i] = exec_result_wo_update['result']
            
            assert report.finish_testing
            # there are at least 1 exclusive unit tests
            ret.check_pass = pass_criterion(report.all_pass_w_update, report.all_pass_wo_update)
            
            if not ret.check_pass:
                ret.error_message = report.generate()
            else:
                ret.warn_message = report.generate()
                
        except Exception as error:
            ret.error_message = error
        return ret
        
    def _sample_prog_syn_spec(self, save_dir):
        from src.data.prompt_prog_syn import (
            prog_syn_sys_prompt_template, prog_syn_input_prompt_template,
        )
        # TODO: adapt
        self.prog_syn_sys_prompt = prog_syn_sys_prompt_template.render()
        self.prog_syn_input_prompt = prog_syn_input_prompt_template.render(
            package_name = self.updated_function.package_name,
            api_path=self.updated_function.full_path,
            old_func=self.updated_function.function_signature,
            update_description=self.update_manager.update_description,
            update_rationale=self.update_manager.rationale,
            docstring_diff=self.update_manager.update_docstring,
            new_function_signature=self.update_manager.new_function_signature,
            num_param=self.num_params,
        )
        self._record_prompt(
            prompt_name="prog_syn_spec", 
            sys_prompt=self.prog_syn_sys_prompt, 
            input_prompt=self.prog_syn_input_prompt, 
            save_dir=self.prompt_dir,
        )
        max_rejection_sample=self.cfg.prog_syn_spec.max_rej_sample
        
        ret = RejectionSampleOutput(name='prog_syn_spec', max_rejection_sample=max_rejection_sample)
        assert ret.name, "RejectionOutput is unnamed."
        error_messages = []
        prog_syn_spec_dict = None
        prog_syn_spec_dict_keys = [
            "scenario", "problem", 
            "solution_signature",
        ]
        for rj_i in range(max_rejection_sample):
            try:
                prog_syn_response = self.load_response_before_call(
                    f"{save_dir}/prog_syn_spec_response-{rj_i}.json",
                    self.prog_syn_sys_prompt, 
                    self.prog_syn_input_prompt,
                )
                prog_syn_spec_dict = self.load_dict(
                    prog_syn_response["choices"][0]["message"]["content"], 
                    extra_required_keys=prog_syn_spec_dict_keys,
                )
                assert prog_syn_spec_dict is not None, "sample_prog_syn_spec: prog_syn_spec_dict is None"
                break
            except Exception as e:
                error_messages.append(str(e))
                continue
        
        ret.error_message = "\n".join(f"{i}.{m}" for i, m in enumerate(error_messages))
        ret.num_rejection_sample = rj_i + 1
        ret.output = prog_syn_spec_dict
        if prog_syn_spec_dict is not None:
            for k in prog_syn_spec_dict_keys:
                setattr(self, k, prog_syn_spec_dict[k])
        return ret
    
    def _sample_unit_test_skeletons(self, save_dir,):
        from src.data.prompt_prog_syn import (
            unit_test_skeleton_sys_prompt_template, unit_test_skeleton_input_prompt_template,
        )
        
        max_rejection_sample = self.cfg.unit_test_skeletons.max_rej_sample
        num_unit_tests = self.cfg.unit_test_skeletons.num_unit_tests
        assert num_unit_tests == 10, "Wrong num_unit_tests in config"
        
        self.unit_test_skeleton_sys_prompt = unit_test_skeleton_sys_prompt_template.render(
            num_unit_tests=num_unit_tests,
        )
        self.unit_test_skeleton_input_prompt = unit_test_skeleton_input_prompt_template.render(
            problem=self.prog_syn_dict["problem"],
            scenario=self.prog_syn_dict["scenario"],
            solution_signature=self.prog_syn_dict["solution_signature"],
        )
        # print(self.unit_test_skeleton_sys_prompt)
        # print(self.unit_test_skeleton_sys_prompt)
        self._record_prompt(
            prompt_name="unit_test_skeletons", 
            sys_prompt=self.unit_test_skeleton_sys_prompt, 
            input_prompt=self.unit_test_skeleton_input_prompt, 
            save_dir=self.prompt_dir,
        )
        
        ret = RejectionSampleOutput(name='unit_test_skeletons', max_rejection_sample=max_rejection_sample)
        assert ret.name, "RejectionOutput is unnamed."
        error_messages = []
        unit_test_skeletons = None
        
        for rj_i in range(max_rejection_sample):
            try:
                unit_test_skeleton_response = self.load_response_before_call(
                    f"{save_dir}/unit_test_skeleton_response-{rj_i}.json",
                    self.unit_test_skeleton_sys_prompt, 
                    self.unit_test_skeleton_input_prompt,
                )
                unit_test_skeletons = self.load_list(
                    unit_test_skeleton_response["choices"][0]["message"]["content"]
                )
                assert len(unit_test_skeletons) > 0, "Empty list generated"
                unit_test_skeletons = [
                    self.extract_python_code(i)
                    for i in unit_test_skeletons
                ]
                break
            except Exception as e:
                error_messages.append(str(e))
                continue
        
        ret.output = unit_test_skeletons
        ret.error_message = "\n".join(f"{i}.{m}" for i, m in enumerate(error_messages))
        ret.num_rejection_sample = rj_i + 1
        return ret
    
    def _prepare_infill(self, content):
        """Extract infill from code block and heuristically fix indentation using first line.

        Args:
            content (_type_): _description_
        """
        def _extract_infill_from_markdown(content):
            pattern = r'```(?:python\n)?(.*?)```'
            return re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        infill_str = _extract_infill_from_markdown(content)
        assert len(infill_str) > 0, "sample_unit_test_answer: Failed to extract unit_test_infill_content"
        infill_str = infill_str[0]
        infill_lines = infill_str.split('\n')
        first_nonspace_index = infill_lines[0].index(infill_lines[0].strip())
        final_infill = []
        for l_i, l in enumerate(infill_lines):
            if len(l.strip()) > 0:
                final_infill.append(l[first_nonspace_index:])
            else:
                final_infill.append("")
        return "\n".join(final_infill)
    
    def _sample_unit_test_answer(
        self, 
        unit_test_i, unit_test_skeleton, 
        save_dir,
    ) -> RejectionSampleOutput:
        
        from src.data.prompt_prog_syn import unit_test_ans_sys_prompt_template, unit_test_ans_input_prompt_template
        from src.data.prompt_package import PACKAGE2PROMPT_ANSWER
        
        self.unit_test_ans_sys_prompt = unit_test_ans_sys_prompt_template.render(
            scenario=self.prog_syn_dict["scenario"],
            problem=self.prog_syn_dict["problem"],
            solution_signature=self.prog_syn_dict["solution_signature"],
        )
        max_rejection_sample=self.cfg.unit_test_answer.max_rej_sample
        ret = RejectionSampleOutput(name=f"unit_test_answer-{unit_test_i}", max_rejection_sample=max_rejection_sample)
        error_messages = []
        assert ret.name, "RejectionOutput is unnamed."
        infill_content = None
        old_function_signature = self.updated_function.function_signature
        old_function_signature = old_function_signature.replace(f"{self.updated_function.parent_path}.", "old_")
        new_function_signature = self.update_manager.new_function_signature
        for rj_i in range(max_rejection_sample):
            try:
                response_name = f"answer_response-{rj_i}.json"
                
                unit_test_ans_input_prompt_i = unit_test_ans_input_prompt_template.render(
                    unit_test_skeleton=unit_test_skeleton,
                    include_api_info=True,
                    old_function_signature=old_function_signature,
                    new_function_signature=new_function_signature,
                    update_docstring=self.update_manager.update_docstring,
                    # package-specific addition
                    package_name=self.updated_function.package_name,
                    package_instruct=PACKAGE2PROMPT_ANSWER[self.updated_function.full_path],
                )
                if not hasattr(self, "unit_test_answer_input_prompt_i"):
                    self.unit_test_ans_input_prompt_i = unit_test_ans_input_prompt_i
                    self._record_prompt(
                        prompt_name="unit_test_answer", 
                        sys_prompt=self.unit_test_ans_sys_prompt, 
                        input_prompt=self.unit_test_ans_input_prompt_i, 
                        save_dir=self.prompt_dir,
                    )
                    
                infill_response = self.load_response_before_call(
                    f"{save_dir}/{response_name}",
                    self.unit_test_ans_sys_prompt, 
                    unit_test_ans_input_prompt_i
                )
                
                infill_content = self._prepare_infill(
                    infill_response["choices"][0]["message"]["content"],
                )
                
                ast.parse(infill_content)  # this is to check syntax error like indentation, 
                # assert f"old_{self.updated_function.full_path}" not in infill_content, "wrong call to old APIs"
                break
            except Exception as e:
                error_messages.append(str(e))
                continue
        ret.error_message = "\n".join(f"{i}.{m}" for i, m in enumerate(error_messages))
        ret.num_rejection_sample = rj_i + 1
        ret.output = infill_content
        return ret
    
    def _sample_unit_test_assert(
        self, 
        unit_test_i, unit_test_w_answer, 
        save_dir,
    ) -> RejectionSampleOutput:
        pass
        
        from src.data.prompt_prog_syn import unit_test_assert_sys_prompt_template, unit_test_assert_input_prompt_template
        from src.data.prompt_package import PACKAGE2PROMPT_ASSERT
        self.unit_test_ans_sys_prompt = unit_test_assert_sys_prompt_template.render(
            # function_name=self.updated_function.function_name,
            # old_function_signature=self.updated_function.function_signature,
            # update_description=self.update_dict["update_description"],
            # new_function_signature=self.update_dict["new_function_signature"],
        )
        max_rejection_sample=self.cfg.unit_test_answer.max_rej_sample
        ret = RejectionSampleOutput(name=f"unit_test_assert-{unit_test_i}", max_rejection_sample=max_rejection_sample)
        error_messages = []
        assert ret.name, "RejectionOutput is unnamed."
        infill_content = None
        for rj_i in range(max_rejection_sample):
            try:
                response_name = f"assert_response-{rj_i}.json"
                
                unit_test_ans_input_prompt_i = unit_test_assert_input_prompt_template.render(
                    unit_test_skeleton=unit_test_w_answer,
                    package_name=self.updated_function.package_name,
                    package_instruct=PACKAGE2PROMPT_ASSERT[self.updated_function.full_path],
                )
                if not hasattr(self, "unit_test_assert_input_prompt_i"):
                    self.unit_test_ans_input_prompt_i = unit_test_ans_input_prompt_i
                    self._record_prompt(
                        prompt_name="unit_test_assert", 
                        sys_prompt=self.unit_test_ans_sys_prompt, 
                        input_prompt=self.unit_test_ans_input_prompt_i, 
                        save_dir=self.prompt_dir,
                    )
                    
                infill_response = self.load_response_before_call(
                    f"{save_dir}/{response_name}",
                    self.unit_test_ans_sys_prompt, 
                    unit_test_ans_input_prompt_i,
                )
                
                infill_content = self._prepare_infill(
                    infill_response["choices"][0]["message"]["content"],
                )
                
                ast.parse(infill_content)  # this is to check syntax error like indentation, 
                if f"old_{self.updated_function.full_path}" in infill_content:
                    infill_content = infill_content.replace(f"old_{self.updated_function.full_path}", f"old_{self.updated_function.function_name}")
                # assert f"old_{self.updated_function.full_path}" not in infill_content, "wrong call to old APIs"
                break
            except Exception as e:
                error_messages.append(str(e))
                continue
        ret.error_message = "\n".join(f"{i}.{m}" for i, m in enumerate(error_messages))
        ret.num_rejection_sample = rj_i + 1
        ret.output = infill_content
        return ret
        
    def _save_unit_tests(self, unit_test_skeletons, answer_infill_list, assert_infill_list, unit_tests, save_dir):
        unit_tests_pack = []
        # record some intermediate results
        for i in range(self.num_unit_tests):
            unit_tests_pack.extend([
                "#" * 50,
                f"# Unit test {i}",
                "# Skeleton",
                unit_test_skeletons[i],
                "\n",
                "# answer_infill",
                str(answer_infill_list[i]),
                "\n",
                "# assert_infill",
                str(assert_infill_list[i]),
                "\n",
                "# Combined unit test",
                unit_tests[i],
                "\n",
                "#" * 50,
            ])
        open(f"{save_dir}/unit_tests_pack.py", "w").write("\n".join(unit_tests_pack))
    
    def _save_for_code_debug(self, file_name, save_dir):
        tmp = [(k, v.replace("\n", "\n# ")) for k, v in self.prog_syn_spec_dict.items()]
        debugging_files = [
            "\n".join([f"""# "{k}": {v}"""
                        for k, v in tmp]),
            self.imports,
            str(self.update_manager.new_impl),
            self.update_manager.update_enforce_statement,
            str(self.solution_new),
            # enforce / introduce function update to package
            "# Unit tests",
            *["\n".join([f"# Unit test {i}", unit_test] ) 
              for i, unit_test in enumerate(self.unit_tests_str)],
        ]
        open(f"{save_dir}/{file_name}", "w").write("\n\n".join(debugging_files))
    
    def _prepare_unit_test_skeleton(self, unit_test_skeleton, placeholders):
        assert any(sl in unit_test_skeleton for sl in placeholders)
        tmp = 0
        count_existence = 0
        placeholder = None
        # detect and select a sentinel_lines
        for sl in placeholders:
            tmp = sum([sl in l for l in unit_test_skeleton.split("\n")])
            if tmp == 1:
                placeholder = sl
            count_existence += tmp
        
        assert placeholder, f"Skeleton contains None of the sentinel_lines({[placeholders]})"
        assert count_existence == 1, \
            f"There should be only *one unique* sentinel_line among {placeholders} in and appearing once in the skeleton."
        ret = [
            (l[:l.find(placeholder)] + "[INFILL]" # just to truncate any trailing comments
             if placeholder in l else l)
            for l in unit_test_skeleton.split("\n")
        ]
        return "\n".join(ret)
    
    def _record_prompt(self, prompt_name, sys_prompt, input_prompt, save_dir=None):
        if not hasattr(self, "prompts"):
            self.prompts = {}
        
        self.prompts[prompt_name] = {
            "sys_prompt": sys_prompt,
            "input_prompt": input_prompt,
        }
        if save_dir:
            self.save_prompts(
                prompt_name=prompt_name, 
                **self.prompts[prompt_name],
                save_dir=save_dir
            )
            
    def _record_rej_sampling_stats(self, rej_sampling_output, suffix = "", save_dir=None):
        if not hasattr(self, "rejection_sample_stats"):
            self.rejection_sample_stats = {}
        assert rej_sampling_output.name and len(rej_sampling_output.name.strip()), \
            "Need to have name for rej_sampling_output"
            
        self.rejection_sample_stats[rej_sampling_output.name] = self.rej_output2dict(rej_sampling_output)
        if save_dir:
            json.dump(self.rejection_sample_stats, open(f"{save_dir}/rej_sample_stats{suffix}.json", "w"))
    
    def _sample_imports(self, code, code_name, save_dir=None):
        from src.data.prompt_update import import_sys_prompt_template, import_input_prompt_template
        
        self.import_sys_prompt = import_sys_prompt_template.render()
        tmp = import_input_prompt_template.render(code=code)
        if hasattr(self, "import_input_prompt"):
            self.import_input_prompt = tmp
            self._record_prompt(
                prompt_name="imports", 
                sys_prompt=self.import_sys_prompt,
                input_prompt=self.import_input_prompt, 
                save_dir=self.prompt_dir,
            )
        max_rejection_sample=self.cfg.imports.max_rej_sample
        prefix = f"{code_name}_" if len(code_name) > 0 else code_name
        
        self.import_input_prompt = tmp
        for rj_i in range(max_rejection_sample):
            try:
                response_name = f"{prefix}import_response-{rj_i}.json"
                import_response = self.load_response_before_call(
                    f"{save_dir}/{response_name}",
                    self.import_sys_prompt,
                    self.import_input_prompt,
                )
                imports_str = self.extract_python_code_from_markdown(import_response["choices"][0]["message"]["content"])
                assert len(imports_str) > 0, "Failed to generate in markdown format."
                # do some surface-level filtering of generated imports
                imports_str = imports_str[0]
                candidates = []
                for l in imports_str.split("\n"):
                    if len(l.strip()) == 0:
                        continue
                    candidates.append(l.strip())
                # execute each import to make sure there's no illegal import
                ret = []
                for l in candidates:
                    import_exec = concat_and_exec(*(ret + [l]))
                    assert import_exec["result"] == "passed"
                break
            except Exception as e:
                continue
        
        return ret
        
    def _sample_solution_new(self, 
        imports, 
        save_dir,
        rerun_exec=False,
    ) -> RejectionSampleOutput:
        from src.data.prompt_prog_syn import (
            solution_new_sys_prompt_template, solution_new_input_prompt_template,
        )
        from copy import deepcopy
        
        new_func_sign = self.update_manager.new_function_signature
        # new_func_sign_wo_full_ref = new_func_sign.replace(
        #     f"{self.updated_function.parent_path}.", ""
        # )
        max_rejection_sample = self.cfg.solution_new.max_rej_sample
        include_unit_tests = self.cfg.solution_new.include_unit_tests
        threshold = self.cfg.solution_new.threshold
        
        old_func_sign = self.updated_function.function_signature
        # renamed_old_func_sign = old_func_sign.replace(f"{self.updated_function.parent_path}.", "old_")
        self.solution_new_sys_prompt = solution_new_sys_prompt_template.render(
            old_function_signature=old_func_sign,
            update_description=self.update_manager.update_description,
            new_function_signature=new_func_sign,
            update_docstring=self.update_manager.update_docstring,
        )
        
        random_index = np.sort(np.random.choice(range(len(self.unit_tests_str)), size=len(self.unit_tests_str), replace=False))
        self.solution_new_input_prompt = solution_new_input_prompt_template.render(
            problem=self.problem,
            solution_signature=self.solution_signature,
            unit_tests=[self.unit_tests_str[i] for i in random_index],
        )
        base_name = "solution_new"
        if include_unit_tests:
            base_name += "-w_ut"
        self._record_prompt(
            prompt_name=base_name,
            sys_prompt=self.solution_new_sys_prompt,
            input_prompt=self.solution_new_input_prompt,
            save_dir=self.prompt_dir,
        )
        
        ret = RejectionSampleOutput(name='solution_new', max_rejection_sample=max_rejection_sample)
        assert ret.name, "RejectionOutput is unnamed."
        error_messages = []
        
        save_dir = f"{save_dir}/{base_name}"
        os.makedirs(save_dir, exist_ok=True)
        max_pass_rate = 0
        best_solution_new = None
        best_imports = imports
        for rj_i in range(max_rejection_sample):
            try:
                rj_i_save_dir = f"{save_dir}/sample-{rj_i}"
                os.makedirs(rj_i_save_dir, exist_ok=True)
                solution_new_response = self.load_response_before_call(
                    f"{rj_i_save_dir}/response.json",
                    self.solution_new_sys_prompt, 
                    self.solution_new_input_prompt,
                )
                solution_new = self.extract_python_code_from_markdown(
                    solution_new_response["choices"][0]["message"]["content"],
                )
                assert solution_new, "sample_solution_new: Failed to extract solution_new"
                solution_new = solution_new[0]
                assert len(solution_new.strip()) > 0, "sample_solution_new: extracted empty string"
                self.solution_new: Function = Function(wrap_code_in_short_comment(
                    solution_new, 
                    "Solution with new API",
                    )
                )
                # self.update_enforce_statement = self.update_manager.update_enforce_statement
                
                solution_new_imports = self._sample_imports(solution_new, f"", save_dir=rj_i_save_dir)
                # remove imports that contains the old function name, since that's heuristically added
                solution_new_imports = [
                    e for e in solution_new_imports 
                    if f"old_{self.updated_function.function_name}" not in e
                ]
                candidate_imports = self.filter_imports(imports + solution_new_imports)
                
                self.imports = wrap_code_in_short_comment(
                    "\n".join(self.sort_imports(candidate_imports)),
                    "Import statement(s)",
                )
                
                self._save_for_code_debug(f"debugging.py", rj_i_save_dir)
                
                exec_result_file = f"{rj_i_save_dir}/unit_tests_exec_check.pkl"
                if os.path.exists(exec_result_file) and not rerun_exec:
                    unit_tests_exec_check = pickle.load(open(exec_result_file, "rb"))
                else:
                    unit_tests_exec_check = self.check_unit_tests(self.solution_new)
                    pickle.dump(unit_tests_exec_check, open(exec_result_file, "wb"))
                    open(
                        f"{rj_i_save_dir}/unit_tests_report.txt","w").write(
                            unit_tests_exec_check.output.generate()
                    )
                # User pass with update as pass rate
                cur_pass_rate = unit_tests_exec_check.output.pass_w_update_rate
                if cur_pass_rate >= max_pass_rate and unit_tests_exec_check.output.num_exclusive_pass > 0:
                    max_pass_rate = cur_pass_rate
                    best_solution_new = solution_new
                    best_imports = deepcopy(candidate_imports)
                
                if max_pass_rate >= threshold and unit_tests_exec_check.output.num_exclusive_pass > 0:
                    break
            except Exception as e:
                error_messages.append(f"sample_id={rj_i}: " + str(e))
                # raise Exception("\n".join(f"{i}.{m}" for i, m in enumerate(error_messages)))
                continue
        ret.error_message = "\n".join(f"{i}.{m}" for i, m in enumerate(error_messages))
        ret.num_rejection_sample = rj_i + 1
        if max_pass_rate < threshold:
            return ret
        
        # TODO: add a step to remove failed unit tests, w/ high prob those are bad ones.
        ret.output = {
            "imports": best_imports,
            "solution_new": best_solution_new,
            "unit_tests_exec_check": unit_tests_exec_check,
        }
        
        self.solution_new: Function = Function(wrap_code_in_short_comment(
            best_solution_new,
            "New function implementation",
            )
        )
        self.imports = wrap_code_in_short_comment(
            "\n".join(self.sort_imports(best_imports)),
            "Import statement(s)",
        )
        return ret
    
    def generate_and_initialize(self, save_dir=None, rerun_exec=False):
        
        from copy import deepcopy
        
        assert os.path.exists(save_dir), "generate_and_initialize: save dir not exist"
        self.prompts: Dict[str, Dict]= {}
        self.rejection_sample_stats = {}
        suffix = ""
        self.prompt_dir = f"{save_dir}/prompt"
        os.makedirs(self.prompt_dir, exist_ok=True)
        if self.cfg.solution_new.include_unit_tests:
            suffix += "-w_ut"
        ######################### START: generate prog syn info #########################
        prog_syn_spec_output = self._sample_prog_syn_spec(
            save_dir, 
        )
        
        # TODO: write a generalized rejection_sampler
        assert isinstance(prog_syn_spec_output.output, dict), \
            "sample_prog_syn_spec: Failed to generate (run out of rejection sample)"
        self.prog_syn_spec_dict = prog_syn_spec_output.output
        self._record_rej_sampling_stats(prog_syn_spec_output, suffix=suffix, save_dir=save_dir)
        self.prog_syn_dict = deepcopy(self.prog_syn_spec_dict)
        ######################### END: generate prog syn info #########################
        # TODO: add a filter for prog syn
        
        ######################### START: generate unit tests skeleton #########################
        unit_test_skeletons_output = self._sample_unit_test_skeletons(
            save_dir,
        )
        unit_test_skeletons = unit_test_skeletons_output.output
        self._record_rej_sampling_stats(unit_test_skeletons_output, suffix=suffix, save_dir=save_dir)
        assert isinstance(unit_test_skeletons, list), \
            f"sample_unit_test_skeletons: Failed to generate (run out of rejection sample)"
        assert len(unit_test_skeletons) > 0, \
             "sample_unit_test_skeletons: Failed to extract unit_test_skeletons_output"
        self.prog_syn_dict["unit_test_skeletons"] = unit_test_skeletons
        ######################### END: generate unit tests skeleton #########################
        # TODO: add a filter for update
        
        ######################### START: generate unit tests answer #########################
        answer_infill_str_list = [None] * len(unit_test_skeletons)
        assert_infill_str_list = [None] * len(unit_test_skeletons)
        unit_tests = [None] * len(unit_test_skeletons)
        placeholders = [self.answer_placeholder, self.assert_placeholder]
        sample_funcs = [self._sample_unit_test_answer, self._sample_unit_test_assert]
        sampler_names = ["answer", "assert"]
        infill_str_lists = [answer_infill_str_list, assert_infill_str_list]
        for i, unit_test_skeleton in enumerate(unit_test_skeletons):
            unit_test = unit_test_skeleton
            # fill in answer and assert placeholder sequentially
            for placeholder, sampler_name, sample_func, infill_str_list in \
                zip(placeholders, sampler_names, sample_funcs, infill_str_lists):
                req_infill = placeholder in unit_test
                if not req_infill:
                    unit_tests[i] = unit_test
                    continue
                # get a directory to save responses
                # response_dir = f"{save_dir}/unit_test_{sampler_name}"
                response_dir = f"{save_dir}/unit_test/test-{i}"
                os.makedirs(response_dir, exist_ok=True)
                # rej sample the infill for correct answers
                infill_output = sample_func(
                    i, unit_test, 
                    save_dir=response_dir,
                )
                    
                self._record_rej_sampling_stats(infill_output, suffix=suffix, save_dir=save_dir)
                infill_str = infill_output.output
                assert infill_str, f"sample_unit_test_{sampler_name}: " \
                    "Failed to generate (run out of rejection sample)"
                assert len(infill_str.strip()) > 0, f"sample_unit_test_{sampler_name}: "\
                    f"Empty infill_content response"
                infill_str_list[i] = infill_str
                # fill in the answer to unit test skeleton
                infill = "\n".join([
                    (l if l_i == 0 else f"{PYTHON_INDENT}{l}")
                    for l_i, l in enumerate(infill_str.split("\n"))
                ])
                unit_test = unit_test.replace(placeholder, infill)
                
            unit_tests[i] = unit_test
            
        assert len(unit_tests) == len(unit_test_skeletons), \
            "len(unit_tests) != len(unit_test_skeletons)"
        self.answer_infills = answer_infill_str_list
        self.assert_infills = assert_infill_str_list
        ######################### END: generate unit tests answer #########################
        
        self.unit_tests_str = unit_tests
        self.unit_tests : List[Function] = [
            Function(
                wrap_code_in_short_comment(t, f"Unit Test {t_i}",)
            )
            for t_i, t in enumerate(self.unit_tests_str)
        ]
        self.num_unit_tests = len(self.unit_tests)
        self._save_unit_tests(unit_test_skeletons, answer_infill_str_list, assert_infill_str_list, unit_tests, save_dir)
        
        ######################### START: Generate and verify imports by exec #########################
        update_imports = self.update_manager.import_lst
        prog_syn_imports = self._sample_imports(
            self.prog_syn_dict["solution_signature"], f"solution_sign",
            save_dir=save_dir,
        )
        
        for u_i, unit_test in enumerate(unit_tests):
            os.makedirs(f"{save_dir}/unit_test/test-{u_i}", exist_ok=True)
            unit_test_imports = self._sample_imports(
                "\n".join(update_imports) + "\n\n" + unit_test, f"", 
                save_dir=f"{save_dir}/unit_test/test-{u_i}",
            )
            # remove imports that contains the old function name, since that's heuristically added
            unit_test_imports = [
                e for e in unit_test_imports 
                if f"old_{self.updated_function.function_name}" not in e
            ]
            prog_syn_imports += unit_test_imports
        # verify correctness of import by exec
        prog_syn_imports = self.filter_imports(prog_syn_imports, note="Unit tests", save_dir=save_dir)
        ######################### END: Generate and verify imports by exec #########################
        
        ######################### START: generate solution_new #########################
        imports = self.sort_imports(update_imports + prog_syn_imports)
        solution_new_output = self._sample_solution_new(
            imports,
            save_dir=save_dir,
            rerun_exec=rerun_exec,
        )
        self._record_rej_sampling_stats(solution_new_output, suffix=suffix, save_dir=save_dir)
        assert solution_new_output.output, \
            "sample_solution_new: Failed to generate (run out of rejection samples). " \
                f"Error: {solution_new_output.error_message}"
        ######################### END: generate solution_new #########################
        self.import_lst = solution_new_output.output["imports"]
        self.unit_tests_exec_result = solution_new_output.output["unit_tests_exec_check"].output
        # TODO: only add filtered unit tests and infill_content_list
        self.prog_syn_dict.update({
            "answer_infills": answer_infill_str_list,
            "assert_infills": assert_infill_str_list,
            "unit_tests": unit_tests,
            "unit_tests_pass_w_update": self.unit_tests_exec_result.pass_w_update,
            "imports": solution_new_output.output["imports"],
            "solution_new": solution_new_output.output["solution_new"],
        })
        
        # save content of all update to a json
        json.dump(self.prog_syn_dict, open(f"{save_dir}/prog_syn-content{suffix}.json", "w"))


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    cur_dir = os.getcwd()
    print(f"Current working dir: {cur_dir}")
    update_save_root = "<path to update save directory>"
    api_path = "math.cos"
    update_type_tag = "modify-function-name"
    update_save_dir = f"{update_save_root}/{api_path}/{update_type_tag}/update-0"
    print(f"Update save dir: {update_save_dir}")
    update_cfg = OmegaConf.load("configs/update_generation.yaml")
    update_cfg.new_impl.include_unit_tests=True
    u_manager = UpdateManager(cfg=update_cfg, api_path=api_path, update_tag=update_type_tag)
    
    # u_manager.generate_and_initialize(save_dir=update_save_dir,)
    u_manager.load_from_dir(save_dir=update_save_dir,)
    progsyn_cfg = OmegaConf.load("configs/prog_syn_generation.yaml")
    ps_manager = ProgSynManager(
        cfg=progsyn_cfg, 
        update_manager=u_manager,
        num_params=3
    )
    prog_syn_save_dir = f"<path to Program synthesis save directory>"
    os.makedirs(prog_syn_save_dir, exist_ok=True)
    print(f"ProgSyn save dir: {prog_syn_save_dir}")
    ps_manager.generate_and_initialize(save_dir=prog_syn_save_dir)
    
    print()
    