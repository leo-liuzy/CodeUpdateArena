import json
import os
import re
from typing import Dict, List
import pickle
from openai import OpenAI
from abc import ABC
import numpy as np
import ast
from omegaconf import OmegaConf

from src.utils.update import Action, Aspect, Place, UpdatedFunction, UpdateType
from src.utils.code import (
    CheckErrorType, CheckOutput,
    CheckOutputSuite, Function,
    RejectionSampleOutput,
    UnitTestsReport, concat_and_exec,
    wrap_code_in_short_comment
)
from src.utils.utils import call_openai_chat, OPENAI_MODEL
from src.data.prompt_update import PYTHON_INDENT



CLIENT = OpenAI()


class Manager(ABC):

    @classmethod
    def get_update_statements(
        cls,
        update_type: UpdateType,
        updated_function: UpdatedFunction, 
        new_function: Function,
        ) -> CheckOutput:
        
        old_f_name = updated_function.function_name
        new_f_name = new_function.function_name
        update_statements = []
        ret = CheckOutput(type=CheckErrorType.WRONG, check_pass=True)
        try:
            if update_type.place in [Place.Argument, Place.Output] or \
                (update_type.action, update_type.aspect) == (Action.Add, Aspect.Semantic):
                assert old_f_name == new_f_name, f"old_f_name == new_f_name for {update_type}"
                update_statements.append(f"{updated_function.full_path} = {new_f_name}")
            elif (update_type.action, update_type.aspect) == (Action.Deprecate, Aspect.Null):
                assert old_f_name == new_f_name, f"old_f_name == new_f_name for {update_type}"
                update_statements.append(f"del {updated_function.full_path}")
            elif (update_type.action, update_type.aspect) == (Action.Modify, Aspect.Name):
                assert old_f_name != new_f_name, f"old_f_name != new_f_name for {update_type}"
                update_statements.append(f"del {updated_function.parent_path}.{old_f_name}")
                update_statements.append(f"setattr({updated_function.parent_path}, '{new_f_name}', {new_f_name})")
            elif (update_type.action, update_type.aspect) == (Action.Add, Aspect.Null):
                assert old_f_name != new_f_name, f"old_f_name != new_f_name for {update_type}"
                update_statements.append(f"setattr({updated_function.parent_path}, '{new_f_name}', {new_f_name})")
            else:
                raise ValueError("Unexpected update scenario")
            ret.output = "\n".join(update_statements)
        except Exception as e:
            ret.check_pass = False
            ret.error_message = str(e)
        return ret

    @classmethod
    def contains_function_def(cls, code):
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return True
        return False
    
    @classmethod
    def inspect_function(cls, function: Function, name: str = "") -> CheckOutputSuite:
        
        ret = CheckOutputSuite()
        for chk, chk_output in function.check_results.items():
            if chk_output is None:
                continue
            chk_output.name = name
            if not chk_output.check_pass:
                # add location of the error types to disambiguate
                chk_output.error_message = name + (":" if name else "") + f"{chk}\n{chk_output.error_message}"
                ret.failed[chk] = chk_output
                # if anything goes wrong, we stop inspecting
                ret.last_failed = chk_output
                return ret
            else:
                ret.passed[chk] = chk_output
        
        return ret    
    
    @classmethod
    def extract_python_code_from_markdown(cls, content):
        pattern = r'```(?:python)?\s*(.*?)```'
        return re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
    
    @classmethod
    def extract_python_code(cls, text):
        def remove_wrapping_quotes(code_str: str):
            while code_str[0] == code_str[-1] and code_str[-1] in ["'", "\"", ]:
                code_str = code_str[1:-1]
            return code_str

        # def extract_python_code_from_markdown(content):
        #     pattern = r'```(?:python)?\s*(.*?)```'
        #     return re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        
        text = text.strip()
        if not text:
            raise ValueError("Empty string")
        
        text = remove_wrapping_quotes(text)
        if text[0] == text[-1] == "`":
            python_code_blocks = cls.extract_python_code_from_markdown(text)
            # always take the first entry
            text = python_code_blocks[0]
        
        text = text.strip()
        if not text:
            raise ValueError("Empty string")
        return text
    
    @classmethod
    def load_content_to_json(cls, generated_content: str) -> CheckOutput:
        """
        load generated string into json
        """
        ret = CheckOutput(
            type=CheckErrorType.LOAD,
            check_pass=False,
            warn_message="", 
            error_message="",
        )

        processing_messages = []
        try:
            ret.output = json.loads(fr"""{generated_content}""", strict=False)
        except json.decoder.JSONDecodeError as json_err:
            processing_messages.append(f"JSON decoding error: {json_err}")
            try:
                ret.output = eval(rf"""{generated_content}""",)
            except Exception as eval_err:
                processing_messages.append(
                    f"Error during `eval(generated_content)`: {eval_err}"
                )
                ret.error_message += "\n\n".join(processing_messages)
                return ret
        
        if not isinstance(ret.output, list):
            processing_messages.append(
                "generated_content is not a list of content"
            )
            ret.error_message += "\n\n".join(processing_messages)
            return ret
        
        if len(ret.output) == 0:
            processing_messages.append(
                "generated_content is an empty list"
            )
            ret.error_message += "\n\n".join(processing_messages)
            return ret

        if len(ret.output) > 1:
            ret.warn_message += "More than 1 update dictionary is generated\n\n"
        
        if not all(isinstance(d, dict) for d in ret.output):
            processing_messages.append(
                "update content is not in dictionary format"
            )
            ret.error_message = "\n\n".join(processing_messages)
            return ret
        
        ret.output = ret.output
        ret.check_pass = True
        ret.warn_message += "\n\n".join(processing_messages)
        return ret
    
    @classmethod
    def extract_json_str(cls, content):
        pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        return re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
    
    @classmethod
    def load_list(
        cls, 
        content: str, 
        element_type = str
    ) -> list:
        try: 
            lst = json.loads(content)
        except json.JSONDecodeError:
            json_str = cls.extract_json_str(content)
            assert len(json_str) > 0, "Failed to extract json from content"
            json_str = json_str[0]
            try:     
                lst = json.loads(json_str)
            except json.JSONDecodeError:
                try: 
                    lst = eval(json_str)
                except:
                    raise ValueError("Failed to turn json string into object")
        
        assert isinstance(lst, list), "json string is not a list"
        for e in lst:
            assert isinstance(e, element_type)
        return lst
    
    @classmethod
    def load_dict(cls, 
        content: str,
        list_val_keys: list = [],
        extra_required_keys: list = [],
    ):
        def extract_json_from_markdown(content):
            pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
            return re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        
        try:
            d = json.loads(content)
        except json.JSONDecodeError:
            try:
                json_content = extract_json_from_markdown(content)
                d = json.loads(json_content[0])
            except:
                return None
        try:
            assert isinstance(d, dict)
            assert all(k in d for k in list_val_keys)
            assert all(k in d for k in extra_required_keys)

            for k, v in d.items():
                assert isinstance(k, str)
                if k in list_val_keys:
                    assert isinstance(v, list)
                else:
                    assert isinstance(v, str)
            
            return d
        except Exception as e:
            print(e)
            return None
    
    @classmethod
    def save_prompts(cls, prompt_name, sys_prompt, input_prompt, save_dir):
        sep_line_len = 50

        f_out = open(f"{save_dir}/{prompt_name}.txt", "w")
        f_out.write("\n".join([
            "System prompt",
            "-" * sep_line_len,
            sys_prompt,
            "-" * sep_line_len,
            "\n\n",
            "Input prompt",
            "-" * sep_line_len,
            input_prompt,
            "-" * sep_line_len,
        ]))
        f_out.close()
    
    @classmethod
    def sort_imports(cls, imports: List[str]):
        def sort_func(x):
            if "=" in x:
                return max_import_len * 2
            elif "setattr(" in x:
                return max_import_len * 3
            else:
                return len(x)
        if len(imports) <= 1:
            return imports
        max_import_len = max(len(e) for e in imports)
        return sorted(
            set(imports), 
            key=sort_func,
        )
    
    @classmethod
    def filter_imports(cls, imports: List[str], note=None, save_dir=None):
        imports = cls.sort_imports(imports)
        legal_imports = []
        illegal_imports = []
        
        for l in imports:
            import_exec = concat_and_exec(*(legal_imports + [l]))
            if import_exec["result"] == "passed":
                legal_imports.append(l)
            else:
                illegal_imports.append(l)
        if save_dir and note is not None and len(illegal_imports) > 0:
            assert os.path.exists(save_dir), "filter_imports: Save dir not exist"
            with open(f"{save_dir}/illegal_imports.py", "a") as f:
                f.write(f"# Note for the following imports: {note}\n")
                f.write("\n".join(illegal_imports))
            
        return legal_imports
    
    @classmethod
    def rej_output2dict(cls, rej_output: RejectionSampleOutput):
        num_fail = rej_output.num_rejection_sample - (rej_output.output is not None)
        return {
            "num_rejection_sample": rej_output.num_rejection_sample,
            "max_rejection_sample": rej_output.max_rejection_sample,
            "yield_rate": \
                1 - num_fail / rej_output.max_rejection_sample,
            "yield_rate_str": \
                f"1 - {num_fail} / {rej_output.max_rejection_sample}",
            "error_messages": rej_output.error_message,
        }
    
    @classmethod
    def load_response_before_call(cls, file_path, sys_prompt, input_prompt, client = CLIENT, model: str = OPENAI_MODEL):
        if os.path.exists(file_path):
            response = json.load(open(file_path, "r"))
        else:
            response = call_openai_chat(client, sys_prompt, input_prompt, model)
            response = json.loads(response.model_dump_json())
            json.dump(response, open(file_path, "w"))
        return response
    
    @classmethod
    def remove_leading_imports(cls, code):
        assert "def" in code, "No `def` exists in code"
        lines = code.split("\n")
        idx = None
        for l_i, line in enumerate(lines):
            if "import " not in line:
                idx = l_i
                break
            
        assert idx is not None, "Failed to truncate leading imports"
        return "\n".join(lines[idx:])

    
class UpdateManager(Manager):
    
    WRAPPER_L = WRAPPER_R = "@"
    
    def __init__(
        self,
        cfg: OmegaConf,
        api_path: str,
        update_tag: str,
        ) -> None:
        # Update input
        self.cfg = cfg
        OmegaConf.resolve(cfg)
        self._api_path = api_path
        self._update_tag = update_tag
        self.updated_function: UpdatedFunction = UpdatedFunction(api_path)
        self.update_type: UpdateType = UpdateType.from_tag(update_tag)
        self.answer_placeholder = "# " + self.WRAPPER_L + "ANSWER" + self.WRAPPER_R
        self.assert_placeholder = "# " + self.WRAPPER_L + "ASSERT" + self.WRAPPER_R
        # first make sure the content of the dict ready for code checking
        self.update_dict = None

    def check_unit_tests(
        self,
        tested_function: Function,
    ) -> CheckOutput:
        # TODO: refactor unit test checker into a separate object
        assert isinstance(tested_function, Function)
        report = UnitTestsReport(self.num_unit_tests)
        ret = CheckOutput(type=CheckErrorType.TEST, output=report, check_pass=False)
        
        assert not self.update_type.is_banned_type
        try:
            
            unit_test_exec_result_pairs = []
            for unit_test in self.unit_tests:
                # run through each unit tests with and without update
                assert isinstance(unit_test, Function)
                program_in_section = [
                    self.imports,
                    str(tested_function),
                    # enforce / introduce function update to package
                    self.update_enforce_statement,
                    str(unit_test),
                    f"{unit_test.function_name}()",
                ]
                update_idx = program_in_section.index(self.update_enforce_statement)
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
                report.inclusive_pass[t_i] = pass_w_update and pass_w_update
                report.unit_tests[t_i] = str(self.unit_tests[t_i])
                
                report.exec_result_w_update[t_i] = exec_result_w_update['result']
                report.exec_result_wo_update[t_i] = exec_result_wo_update['result']
            
            assert report.finish_testing
            # there are at least 1 exclusive unit tests
            ret.check_pass = report.all_pass_w_update and not report.all_pass_wo_update
            
            if not ret.check_pass:
                ret.error_message = report.generate()
            else:
                ret.warn_message = report.generate()
                
        except Exception as error:
            ret.error_message = error
        return ret

    def _record_rej_sampling_stats(self, rej_sampling_output, suffix = "", save_dir=None):
        if not hasattr(self, "rejection_sample_stats"):
            self.rejection_sample_stats = {}
        assert rej_sampling_output.name and len(rej_sampling_output.name.strip()), \
            "Need to have name for rej_sampling_output"
            
        self.rejection_sample_stats[rej_sampling_output.name] = self.rej_output2dict(rej_sampling_output)
        if save_dir:
            json.dump(self.rejection_sample_stats, open(f"{save_dir}/rej_sample_stats{suffix}.json", "w"))
    
    def _sample_update_spec(self, save_dir):
        from src.data.prompt_update import (
            update_spec_sys_prompt_template, update_spec_input_prompt_template,
        )
        self.update_sys_prompt = update_spec_sys_prompt_template.render(update_description=self.update_type.description)
        self.update_input_prompt = update_spec_input_prompt_template.render(
            parent_path = self.updated_function.parent_path,
            function_signature = self.updated_function.function_signature,
            doc_string=self.updated_function.doc_string_indented,
        )
        self._record_prompt(
            prompt_name="update_spec", 
            sys_prompt=self.update_sys_prompt, 
            input_prompt=self.update_input_prompt, 
            save_dir=self.prompt_dir,
        )
        max_rejection_sample=self.cfg.update_spec.max_rej_sample
        
        ret = RejectionSampleOutput(name='update_spec', max_rejection_sample=max_rejection_sample)
        assert ret.name, "RejectionOutput is unnamed."
        error_messages = []
        update_spec_dict = None
        update_spec_dict_keys = [
            "description", "rationale", 
            "signature", "docstring",
        ]
        for rj_i in range(max_rejection_sample):
            try:
                update_response = self.load_response_before_call(
                    f"{save_dir}/update_spec_response-{rj_i}.json",
                    self.update_sys_prompt, 
                    self.update_input_prompt,
                )
                update_spec_dict = self.load_dict(
                    update_response["choices"][0]["message"]["content"], 
                    extra_required_keys=update_spec_dict_keys,
                )
                assert update_spec_dict is not None, "sample_update_spec: update_spec_dict is None"
                break
            except Exception as e:
                error_messages.append(str(e))
                continue
        
        ret.error_message = "\n".join(f"{i}.{m}" for i, m in enumerate(error_messages))
        ret.num_rejection_sample = rj_i + 1
        ret.output = update_spec_dict
        if update_spec_dict is not None:
            for k in update_spec_dict_keys:
                setattr(self, k, update_spec_dict[k])
        return ret
    
    def _sample_unit_test_skeletons(self, save_dir,):
        from src.data.prompt_update import (
            unit_test_skeleton_sys_prompt_template, unit_test_skeleton_input_prompt_template,
        )
        from src.data.prompt_package import PACKAGE2PROMPT_INPUT
        max_rejection_sample = self.cfg.unit_test_skeletons.max_rej_sample
        num_unit_tests = self.cfg.unit_test_skeletons.num_unit_tests
        assert num_unit_tests == 10, "Wrong num_unit_tests in config"
        
        self.unit_test_skeleton_sys_prompt = unit_test_skeleton_sys_prompt_template.render(
            old_function_signature=self.updated_function.function_signature,
            update_description=self.update_dict["description"],
            new_function_signature=self.update_dict["signature"],
            num_unit_tests=num_unit_tests,
            function_name=self.updated_function.function_name,
            full_api_path=self.updated_function.full_path,
        )
        self.unit_test_skeleton_input_prompt = unit_test_skeleton_input_prompt_template.render(
            update_docstring=self.update_dict["docstring"], 
            # package-specific addition
            package_name=self.updated_function.package_name,
            package_instruct=PACKAGE2PROMPT_INPUT[self.updated_function.full_path],
        )
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
                for skeleton in unit_test_skeletons:
                    assert skeleton.count(self.answer_placeholder) <= 1
                    assert skeleton.count(self.assert_placeholder) <= 1
                break
            except Exception as e:
                error_messages.append(str(e))
                continue
        
        ret.output = unit_test_skeletons
        ret.error_message = "\n".join(f"{i}.{m}" for i, m in enumerate(error_messages))
        ret.num_rejection_sample = rj_i + 1
        return ret
    
    def _sample_unit_test_answer(
        self, 
        unit_test_i, unit_test_skeleton, 
        save_dir,
    ) -> RejectionSampleOutput:
        
        from src.data.prompt_update import unit_test_ans_sys_prompt_template, unit_test_ans_input_prompt_template
        from src.data.prompt_package import PACKAGE2PROMPT_ANSWER
        
        self.unit_test_ans_sys_prompt = unit_test_ans_sys_prompt_template.render(
            function_name=self.updated_function.function_name,
            old_function_signature=self.updated_function.function_signature,
            update_description=self.update_dict["description"],
            new_function_signature=self.update_dict["signature"],
        )
        max_rejection_sample=self.cfg.unit_test_answer.max_rej_sample
        ret = RejectionSampleOutput(name=f"unit_test_answer-{unit_test_i}", max_rejection_sample=max_rejection_sample)
        error_messages = []
        assert ret.name, "RejectionOutput is unnamed."
        infill_content = None
        
        for rj_i in range(max_rejection_sample):
            try:
                response_name = f"answer_response-{rj_i}.json"
                
                unit_test_answer_input_prompt_i = unit_test_ans_input_prompt_template.render(
                    update_docstring=self.update_dict["docstring"], 
                    unit_test_skeleton=unit_test_skeleton,
                    function_name=self.updated_function.function_name,
                    # package-specific addition
                    package_name=self.updated_function.package_name,
                    package_instruct=PACKAGE2PROMPT_ANSWER[self.updated_function.full_path],
                )
                if not hasattr(self, "unit_test_answer_input_prompt_i"):
                    self.unit_test_ans_input_prompt_i = unit_test_answer_input_prompt_i
                    self._record_prompt(
                        prompt_name="unit_test_answer", 
                        sys_prompt=self.unit_test_ans_sys_prompt, 
                        input_prompt=self.unit_test_ans_input_prompt_i, 
                        save_dir=self.prompt_dir,
                    )
                    
                infill_response = self.load_response_before_call(
                    f"{save_dir}/{response_name}",
                    self.unit_test_ans_sys_prompt, 
                    unit_test_answer_input_prompt_i,
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
    
    def _sample_unit_test_assert(
        self, 
        unit_test_i, unit_test_w_answer, 
        save_dir,
    ) -> RejectionSampleOutput:
        
        from src.data.prompt_update import unit_test_assert_sys_prompt_template, unit_test_assert_input_prompt_template
        from src.data.prompt_package import PACKAGE2PROMPT_ASSERT
        self.unit_test_ans_sys_prompt = unit_test_assert_sys_prompt_template.render()
        max_rejection_sample=self.cfg.unit_test_answer.max_rej_sample
        ret = RejectionSampleOutput(name=f"unit_test_assert-{unit_test_i}", max_rejection_sample=max_rejection_sample)
        error_messages = []
        assert ret.name, "RejectionOutput is unnamed."
        infill_content = None
        for rj_i in range(max_rejection_sample):
            try:
                response_name = f"assert_response-{rj_i}.json"
                
                unit_test_assert_input_prompt_i = unit_test_assert_input_prompt_template.render(
                    unit_test_skeleton=unit_test_w_answer,
                    package_name=self.updated_function.package_name,
                    package_instruct=PACKAGE2PROMPT_ASSERT[self.updated_function.full_path],
                )
                if not hasattr(self, "unit_test_assert_input_prompt_i"):
                    self.unit_test_ans_input_prompt_i = unit_test_assert_input_prompt_i
                    self._record_prompt(
                        prompt_name="unit_test_assert", 
                        sys_prompt=self.unit_test_ans_sys_prompt, 
                        input_prompt=self.unit_test_ans_input_prompt_i, 
                        save_dir=self.prompt_dir,
                    )
                    
                infill_response = self.load_response_before_call(
                    f"{save_dir}/{response_name}",
                    self.unit_test_ans_sys_prompt, 
                    unit_test_assert_input_prompt_i,
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
    
    def _sample_new_impl(self, 
        imports, 
        save_dir,
        rerun_exec=False,
    ) -> RejectionSampleOutput:
        from src.data.prompt_update import (
            new_impl_sys_prompt_template, new_impl_input_prompt_template,
        )
        from copy import deepcopy
        
        new_func_sign = self.signature
        if self.updated_function.parent_path not in self.signature:
            assert self.updated_function.function_name in self.signature, \
            "sample_new_impl: function name not in self.signature"
            new_func_sign = f"{self.updated_function.parent_path}.{self.signature}"
        # new_func_sign_wo_full_ref = new_func_sign.replace(
        #     f"{self.updated_function.parent_path}.", ""
        # )
        max_rejection_sample = self.cfg.new_impl.max_rej_sample
        include_unit_tests = self.cfg.new_impl.include_unit_tests
        threshold = self.cfg.new_impl.threshold
        
        old_func_sign = self.updated_function.function_signature
        renamed_old_func_sign = old_func_sign.replace(f"{self.updated_function.parent_path}.", "old_")
        self.new_impl_sys_prompt = new_impl_sys_prompt_template.render(
            old_function_signature=old_func_sign,
            update_description=self.description,
            new_function_signature=new_func_sign,
            renamed_old_function_signature=renamed_old_func_sign,
            function_name=self.updated_function.function_name,
            full_api_path=self.updated_function.full_path,
        )
        
        random_index = np.sort(np.random.choice(range(len(self.unit_tests_str)), size=len(self.unit_tests_str), replace=False))
        self.new_impl_input_prompt = new_impl_input_prompt_template.render(
            update_docstring=self.docstring,
            unit_tests=[self.unit_tests_str[i] for i in random_index],
        )
        base_name = "new_impl"
        if include_unit_tests:
            base_name += "-w_ut"
        self._record_prompt(
            prompt_name=base_name, 
            sys_prompt=self.new_impl_sys_prompt,
            input_prompt=self.new_impl_input_prompt, 
            save_dir=self.prompt_dir,
        )
        
        ret = RejectionSampleOutput(name='new_impl', max_rejection_sample=max_rejection_sample)
        assert ret.name, "RejectionOutput is unnamed."
        error_messages = []
        
        save_dir = f"{save_dir}/{base_name}"
        os.makedirs(save_dir, exist_ok=True)
        max_pass_rate = 0
        best_new_impl = None
        best_imports = imports
        for rj_i in range(max_rejection_sample):
            try:
                rj_i_save_dir = f"{save_dir}/sample-{rj_i}"
                os.makedirs(rj_i_save_dir, exist_ok=True)
                new_impl_response = self.load_response_before_call(
                    f"{rj_i_save_dir}/response.json",
                    self.new_impl_sys_prompt, 
                    self.new_impl_input_prompt,
                )
                new_impl = self.extract_python_code_from_markdown(
                    new_impl_response["choices"][0]["message"]["content"],
                )
                assert new_impl, "sample_new_impl: Failed to extract new_impl"
                new_impl = new_impl[0]
                assert len(new_impl.strip()) > 0, "sample_new_impl: extracted empty string"
                # TODO truncate everything before def
                new_impl = self.remove_leading_imports(new_impl)
                self.implementation: Function = Function(wrap_code_in_short_comment(
                    new_impl, 
                    "New function implementation",
                    )
                )
                enforcement_output = self.get_update_statements(
                    self.update_type, 
                    self.updated_function,
                    self.implementation
                )
                assert enforcement_output.check_pass, "sample_new_impl: Failed to generate statement to enforce update"
                
                self.update_enforce_statement = wrap_code_in_short_comment(
                    enforcement_output.output,
                    "Update statement(s)",
                )
                new_impl_str = "\n".join(best_imports) + "\n" + str(new_impl)
                new_impl_imports = self._sample_imports(new_impl_str, f"", save_dir=rj_i_save_dir)
                # remove imports that contains the old function name, since that's heuristically added
                new_impl_imports = [
                    e for e in new_impl_imports 
                    if f"old_{self.updated_function.function_name}" not in e
                ]
                candidate_imports = self.filter_imports(imports + new_impl_imports)
                
                self.imports = wrap_code_in_short_comment(
                    "\n".join(self.sort_imports(candidate_imports)),
                    "Import statement(s)",
                )
                
                self._save_for_code_debug(f"debugging.py", rj_i_save_dir)
                
                exec_result_file = f"{rj_i_save_dir}/unit_tests_exec_check.pkl"
                if os.path.exists(exec_result_file) and not rerun_exec:
                    unit_tests_exec_check = pickle.load(open(exec_result_file, "rb"))
                else:
                    unit_tests_exec_check = self.check_unit_tests(self.implementation)
                    pickle.dump(unit_tests_exec_check, open(exec_result_file, "wb"))
                    open(
                        f"{rj_i_save_dir}/unit_tests_report.txt","w").write(
                            unit_tests_exec_check.output.generate()
                    )
                # User pass with update as pass rate
                cur_pass_rate = unit_tests_exec_check.output.pass_w_update_rate
                if cur_pass_rate >= max_pass_rate and unit_tests_exec_check.output.num_exclusive_pass > 0:
                    max_pass_rate = cur_pass_rate
                    best_new_impl = new_impl
                    best_imports = deepcopy(candidate_imports)
                
                if max_pass_rate >= threshold and unit_tests_exec_check.output.num_exclusive_pass > 0:
                    break
            except Exception as e:
                error_messages.append(f"sample_id={rj_i}: " + str(e))
                continue
        ret.error_message = "\n".join(f"{i}.{m}" for i, m in enumerate(error_messages))
        ret.num_rejection_sample = rj_i + 1
        if max_pass_rate < threshold:
            return ret
        
        # TODO: add a step to remove failed unit tests, w/ high prob those are bad ones.
        ret.output = {
            "imports": best_imports,
            "implementation": best_new_impl,
            "unit_tests_exec_check": unit_tests_exec_check,
        }
        
        self.implementation: Function = Function(wrap_code_in_short_comment(
            best_new_impl,
            "New function implementation",
            )
        )
        self.imports = wrap_code_in_short_comment(
            "\n".join(self.sort_imports(best_imports)),
            "Import statement(s)",
        )
        enforcement_output = self.get_update_statements(
            self.update_type, 
            self.updated_function,
            self.implementation
        )
        self.update_enforce_statement = wrap_code_in_short_comment(
            enforcement_output.output,
            "Update statement(s)",
        )
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
        tmp = [(k, v.replace("\n", "\n# ")) for k, v in self.update_spec_dict.items()]
        debugging_files = [
            "\n".join([f"""# "{k}": {v}"""
                        for k, v in tmp]),
            self.imports,
            str(self.implementation),
            # enforce / introduce function update to package
            self.update_enforce_statement,
            "# Unit tests",
            *["\n".join([f"# Unit test {i}", unit_test] ) 
              for i, unit_test in enumerate(self.unit_tests_str)],
        ]
        open(f"{save_dir}/{file_name}", "w").write("\n\n".join(debugging_files))
    
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
    
    def load_from_dict(self, update_dict):
        for k in ['description', 'rationale', 'signature', 'docstring', 'unit_tests', 'imports', 'implementation']:
            assert k in update_dict
        self.description = update_dict["description"]
        self.rationale = update_dict["rationale"]
        self.signature = update_dict["signature"]
        self.docstring = update_dict["docstring"]
        self.unit_tests_str = update_dict["unit_tests"]
        self.unit_tests : List[Function] = [
            Function(
                wrap_code_in_short_comment(t, f"Unit Test {t_i}",)
            )
            for t_i, t in enumerate(self.unit_tests_str)
        ]
        self.num_unit_tests = len(self.unit_tests)
        self.import_lst = update_dict["imports"]
        self.imports = wrap_code_in_short_comment(
            "\n".join(self.sort_imports(self.import_lst)),
            "Import statement(s)",
        )
        
        self.implementation: Function = Function(wrap_code_in_short_comment(
            update_dict["implementation"],
            "New function implementation",
            )
        )
        enforcement_output = self.get_update_statements(
            self.update_type, 
            self.updated_function,
            self.implementation
        )
        assert enforcement_output.check_pass, "Update_init: Failed to generate statement to enforce update"
        self.update_enforce_statement = wrap_code_in_short_comment(
            enforcement_output.output,
            "Update statement(s)",
        )
        self.update_dict = update_dict
    
    def load_from_dir(self, save_dir):
        suffix = ""
        self.save_dir = save_dir
        if self.cfg.new_impl.include_unit_tests:
            suffix += "-w_ut"
        assert os.path.exists(f"{save_dir}/update-content{suffix}.json")
        update_dict = json.load(open(f"{save_dir}/update-content{suffix}.json", "r"))
        self.load_from_dict(update_dict)
    
    def generate_and_initialize(self, save_dir=None, rerun_exec=False):
        
        from copy import deepcopy
        
        assert os.path.exists(save_dir), "generate_and_initialize: save dir not exist"
        self.prompts: Dict[str, Dict]= {}
        self.rejection_sample_stats = {}
        
        suffix = ""
        self.save_dir = save_dir
        self.prompt_dir = f"{save_dir}/prompt"
        os.makedirs(self.prompt_dir, exist_ok=True)
        if self.cfg.new_impl.include_unit_tests:
            suffix += "-w_ut"
        ######################### START: generate update info #########################
        update_spec_output = self._sample_update_spec(
            save_dir, 
        )
        
        # TODO: write a generalized rejection_sampler
        assert isinstance(update_spec_output.output, dict), \
            "sample_update_spec: Failed to generate (run out of rejection sample)"
        self.update_spec_dict = update_spec_output.output
        self._record_rej_sampling_stats(update_spec_output, suffix=suffix, save_dir=save_dir)
        self.update_dict = deepcopy(self.update_spec_dict)
        ######################### END: generate update info #########################
        # TODO: add a filter for update
        
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
        self.update_dict["unit_test_skeletons"] = unit_test_skeletons
        ######################### END: generate unit tests skeleton #########################
        
        ######################### START: generate unit tests infill #########################
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
        ######################### END: generate unit tests infill #########################
        
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
        imports = self.updated_function.imports 
        f_name = self.updated_function.function_name
        imports += [
            f"import {self.updated_function.package_name}",
            f"old_{f_name} = {self.updated_function.full_path}",
            f"setattr({self.updated_function.parent_path}, 'old_{f_name}', old_{f_name})",
        ]
        
        for u_i, unit_test in enumerate(unit_tests):
            os.makedirs(f"{save_dir}/unit_test/test-{u_i}", exist_ok=True)
            unit_test_imports = self._sample_imports(
                unit_test, code_name=f"",
                save_dir=f"{save_dir}/unit_test/test-{u_i}",
            )
            # remove imports that contains the old function name, since that's heuristically added
            unit_test_imports = [
                e for e in unit_test_imports 
                if f"old_{self.updated_function.function_name}" not in e
            ]
            imports += unit_test_imports
        imports = self.filter_imports(imports, note="Old API + Unit tests", save_dir=save_dir)
        ######################### END: Generate and verify imports by exec #########################
        
        ######################### START: generate new_impl #########################
        # Now treating unit tests as "gold"
        new_impl_output = self._sample_new_impl(
            imports,
            save_dir=save_dir,
            rerun_exec=rerun_exec,
        )
        self._record_rej_sampling_stats(new_impl_output, suffix=suffix, save_dir=save_dir)
        assert new_impl_output.output, "sample_new_impl: "\
            "Failed to generate (run out of rejection samples). " \
                f"Error: {new_impl_output.error_message}"
        ######################### END: generate new_impl #########################
        self.import_lst = new_impl_output.output["imports"]
        self.unit_tests_exec_result = new_impl_output.output["unit_tests_exec_check"].output
        # TODO: only add filtered unit tests and infill_content_list
        self.update_dict.update({
            "answer_infills": answer_infill_str_list,
            "assert_infills": assert_infill_str_list,
            "unit_tests": unit_tests,
            "unit_tests_pass_w_update": self.unit_tests_exec_result.pass_w_update,
            "imports": new_impl_output.output["imports"],
            "implementation": new_impl_output.output["implementation"],
        })
        
        # save content of all update to a json
        json.dump(self.update_dict, open(f"{save_dir}/update-content{suffix}.json", "w"))
        # pickle.dump(self, open(f"{save_dir}/update-manager{suffix}.pkl", "wb"))
        
        
if __name__ == "__main__":
    
    os.chdir(os.path.dirname(__file__)) 
    cur_dir = os.getcwd()
    print(f"Current working dir: {cur_dir}")
    save_root = "<path to save directory>"
    api_path = "requests.get"
    update_type_tag = "add-argument-default_value(s)"
    
    for pack_id in ["response-pack-0", "response-pack-1", "response-pack-2"][:1]:
        save_dir = f"{save_root}/{api_path}/{update_type_tag}/{pack_id}"
        print(f"Save dir: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        cfg = OmegaConf.load("configs/update_generation.yaml")
        cfg.new_impl.max_rej_sample=6
        u_manager = UpdateManager(cfg=cfg, api_path=api_path, update_tag=update_type_tag)
        u_manager.generate_and_initialize(save_dir=save_dir, rerun_exec=True)
        print()
    