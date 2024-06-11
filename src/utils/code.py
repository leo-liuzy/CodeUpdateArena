import ast
import inspect
import json
import os
import re
import traceback
from collections import Counter, defaultdict, OrderedDict
# from update_utils import UpdateType, UpdatedFunction, Action, Aspect, Place
import os, sys

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from radon.metrics import h_visit
from radon.raw import analyze
from radon.visitors import ComplexityVisitor
# from src.execution.program_tracing import get_function_final_state, get_statements_from_code
from transformers.utils import ModelOutput

from src.execution.safe_execution_util import execute


def concat_and_exec(*args: List[str]):
    program = "\n".join(args)
    return execute(program, timeout=15, output_locals=False)

def wrap_code_in_short_comment(
    code: str, 
    comment: str, 
    num_hashtag: int = 10, 
    return_str: bool = True,
    ) -> List | str:
    affix = "#" * num_hashtag
    lines = [
        f"{affix} Start: {comment} {affix}",
        code,
        f"{affix}  End : {comment} {affix}",
    ]
    
    return "\n".join(lines) if return_str else lines

def str_lst_to_str(str_lst: List[str], numbered: bool = True):
    ret = ""
    for i, s in enumerate(str_lst, start=1):
        if numbered:
            ret += f"{i}. {s}\n"
        else:
            ret += f"* {s}\n"
    return ret.strip()


class CheckErrorType(Enum):
    """name for different types of checks"""
    LOAD = "load-json"
    PARSE = 'ast-parse'
    BORING = 'boring-func'
    # v e.g. wrong name, wrong #param
    WRONG = 'wrong-func' 
    TEST = 'unit-tests'
    NULL = 'pass'

    @classmethod
    def get_error_order(cls, error_type: str = "string", error2order: bool = True) -> Dict:
        """Get difficulty order of different checks; 

        the larger the harder, or closer to pass
        Args:
            error_type (str, optional): _description_. Defaults to "string".
            error2order (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        assert error_type in ["string", "check_type"]
         
        if error2order:
            return {(c if error_type == "check_type" else c.value): i for i, c in enumerate(cls)}
        else:
            return {i: (c if error_type == "check_type" else c.value) for i, c in enumerate(cls)}


class CheckPlace(Enum):
    new_impl = "new-impl"
    prog_syn = "prog-syn"


@dataclass
class CheckOutput:
    type: Optional[CheckErrorType] = None
    # place
    name: Optional[str] = None # e.g. To differentiate different unit tests result
    check_pass: Optional[bool] = None
    output: Optional[Any] = None
    warn_message: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class RejectionSampleOutput:
    output: Optional[Any] = None
    max_rejection_sample: Optional[Any] = None,
    num_rejection_sample: int = 0,
    name: Optional[str] = None,
    error_message: Optional[str] = None
    
    

@dataclass
class CheckOutputSuite:
    passed: Dict = field(default_factory = lambda: {})
    failed: Dict = field(default_factory = lambda: {})
    last_failed: Optional[CheckOutput] = None

class Function:
    """Static checking + compilation check
    """
    def __init__(
        self, 
        source_code,
        check_boring=True,
    ) -> None:
        self.source_code = source_code
        # ast parse related attr
        self.source_parse = None
        self.parse_check_result = None
        # function name extraction
        self.function_name = None
        self.function_name_result = None
        # boring function related attr
        self.is_boring_func = None
        self.boring_check_result = None
        
        self.check_boring = check_boring
        self.check_results: Dict[str, CheckOutput] =  OrderedDict()
        
        self.failed_check = None
        self.check_results[CheckErrorType.PARSE.value] = self.run_parsible_check()
        if not self.parse_check_result.check_pass:
            self.failed_check = self.parse_check_result
            return
        self.check_results[CheckErrorType.WRONG.value] = self.extract_func_name()
        if not self.function_name_result.check_pass:
            self.failed_check = self.function_name_result
            return
        
        if not check_boring: 
            return
        self.check_results[CheckErrorType.BORING.value] = self.run_boring_check()
        if not self.boring_check_result.check_pass:
            self.failed_check = self.boring_check_result
    
    @classmethod
    def count_function_call(cls, node) -> Counter:
        """A custom traverse func

        Args:
            node (_type_): ast parse

        Raises:
            ValueError: unprocessed situation

        Returns:
            num_func_call, func_counts
            ret: number of 
        """
        func2count = Counter()
        if isinstance(node, ast.Call):
            # called_names.add(node)
            if isinstance(node.func, ast.Name):
                func_name = f"{node.func.id}()"
            elif isinstance(node.func, ast.Attribute):
                func_name = f".{node.func.attr}()"
            else:
                # raise ValueError("Unknown situation")
                return cls.count_function_call(node.func)
            func2count[func_name] += 1
        for child in ast.iter_child_nodes(node):
            child_func2count = cls.count_function_call(child)
            # ret += child_ret
            func2count += child_func2count
        return func2count
    
    @classmethod
    def is_vacuous_return(cls, source_code):
        # Interesing Note: GPT-3.5 doesn't deal well with regex with nested groups.
        # The following is a regex describing what is deemed as a function with vacuous return
        # function definition contains only return, pass, print
        pattern = r"""
        ^\s*
        def\s+\w+\s*\(.*\)\s*(->\s*\w+\s*)?: # function signature
        \s*[\n\s]*(?:return\s+.*|pass|print\(.*\))\s*$ # vacuous function body
        """
        # only have return + assignment statements
        return re.match(pattern, source_code, re.MULTILINE | re.VERBOSE) is not None
    
    @classmethod
    def remove_imports(cls, node):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return None
        
        if hasattr(node, '_fields'):
            for field_name in node._fields:
                field_value = getattr(node, field_name)
                if isinstance(field_value, list):
                    # Handle lists of child nodes
                    new_children = []
                    for child in field_value:
                        new_child = cls.remove_imports(child)
                        if new_child is not None:
                            new_children.append(new_child)
                    setattr(node, field_name, new_children)
                elif isinstance(field_value, ast.AST):
                    # Handle single child nodes
                    new_child = cls.remove_imports(field_value)
                    if new_child is not None:
                        setattr(node, field_name, new_child)
        
        return node
    
    @classmethod
    def check_boring_func(
        cls,
        code: str, 
        min_func_call: int = 2,
        min_cyclomatic_complexity: int = 2, 
        min_halstead_effort: float = 0,
    ) -> CheckOutput:
        """Statically check if the function is vacuous, meaning 
        it only has a single return or only assignment statement before return

        Args:
            function_content (str): _description_
            min_cyclomatic_complexity: min number of independent path in ast that's *required*
            min_halstead_difficulty: 
                min halstead difficulty that's required
                (by num_unique_operator, num_unique_operands, total_operands)
                Now this is set to be 0 --- any function should work.
        """
        # this is only to remove comment and import by ast.parse
        # otherwise, the following regex is not effective
        # initialize some value for check_pass, as different checks goes either way
        ret = CheckOutput(
            type=CheckErrorType.BORING,
            output=True,
            check_pass=False,
        )
        node = ast.parse(code)
        import_free_node = cls.remove_imports(node)
        slimmed_code = ast.unparse(import_free_node)
        
        # only have return + assignment statements
        # check for low cyclomatic_ complexity
        try:
            cc_v = ComplexityVisitor.from_code(slimmed_code)
        except:
            # TODO: check why it fails here
            ret.error_message = "Unexpected error in Complexity measure"
            return ret
        
        if len(cc_v.functions) < 1:
            ret.error_message = "Cyclomatic complexity < 1."
            return ret

        # assert len(cc_v.functions) >= 1
        complexity = cc_v.functions[0].complexity
        # check for low helstead efforts
        hc_v = h_visit(slimmed_code)
        h_effort = hc_v[0].effort
        f2c = cls.count_function_call(ast.parse(slimmed_code))
        
        warn_messages = []
        if sum(f2c.values()) >= min_func_call:
            # if it contains many function call, then it should be interesting
            ret.output = False
            ret.check_pass = True
            return ret
        warn_messages.append("Not enough function calls")
        
        if complexity >= min_cyclomatic_complexity and h_effort >= min_halstead_effort:
            # didn't uses many function call, but has interesting operations or branching
            ret.output = False
            ret.warn_message = str_lst_to_str(warn_messages)
            ret.check_pass = True
            return ret
        warn_messages.append("Low Cyclomatic complexity")
        
        # only have (return/pass/print) statement 
        is_vacuous_return = cls.is_vacuous_return(slimmed_code)
        if not is_vacuous_return:
            ret.output = False
            ret.warn_message = str_lst_to_str(warn_messages)
            ret.check_pass = True
            return ret
        warn_messages.append("Vacuous return")
        
        ret.warn_message = str_lst_to_str(warn_messages)
        ret.error_message = "Fail all the checks"
        return ret

    def run_boring_check(self, rerun=False) -> CheckOutput:
        if self.boring_check_result is None or rerun:
            self.boring_check_result = self.check_boring_func(self.source_code)
            self.is_boring_func = self.boring_check_result.output
        
        return self.boring_check_result
            
    @classmethod
    def check_parsible(cls, source_code) -> CheckOutput:
        # initialize some value for check_pass, as different checks goes either way
        ret = CheckOutput(type=CheckErrorType.PARSE, check_pass=True)
        try:
            f_parse = ast.parse(rf"{source_code}")
            ret.output = f_parse
        except Exception as ast_error:
            # print(f"Error Message:\n {ast_error}")
            ret.check_pass = False
            ret.error_message = ast_error
        return ret
    
    def run_parsible_check(self, rerun=False) -> CheckOutput:
        if self.parse_check_result is None or rerun:
            self.parse_check_result = self.check_parsible(self.source_code)
            self.source_parse = self.parse_check_result.output
        
        return self.parse_check_result
        
    
    def extract_func_name(self, rerun=False) -> CheckOutput:
        if self.function_name_result is None or rerun:
            self.function_name_result = self.get_function_name(deepcopy(self.source_parse))
            self.function_name = self.function_name_result.output
        return self.function_name_result
    
    @classmethod
    def get_function_name(cls, node) -> CheckOutput:
        ret = CheckOutput(type=CheckErrorType.WRONG, check_pass=False)
        import_free_node = cls.remove_imports(node)
        slimmed_code = ast.unparse(import_free_node)
        
        pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        function_names = re.findall(pattern, slimmed_code, re.DOTALL | re.IGNORECASE)
        if len(function_names) == 1:
            ret.output = function_names[0]
            ret.check_pass = True
        elif len(function_names) > 1:
            ret.error_message = "Multiple function names detected"
        else:
            ret.error_message = "No function name detected"
        return ret
    
    @classmethod
    def get_function_obj_from_source(cls, source_code):
        # Execute the source code in a temporary namespace
        namespace = {}
        exec(source_code, namespace)
        
        # Search for a function object in the namespace
        function_objects = [obj for obj in namespace.values() if inspect.isfunction(obj)]
        
        # If there's only one function object, return it
        if len(function_objects) == 1:
            return function_objects[0]
        elif len(function_objects) > 1:
            raise ValueError("Multiple function objects found in the source code.")
        else:
            raise ValueError("No function object found in the source code.")    

    def __str__(self) -> str:
        return self.source_code

    def __repr__(self) -> str:
        name = self.function_name or "????"
        return f"{name}(...)"
            


class UnitTestsReport:
    def __init__(
        self,
        num_unit_tests
        ) -> None:
        self.tested_function = None
        self.num_unit_tests = num_unit_tests
        self.unit_tests = {}
        
        self.pass_w_update = {}
        self.pass_wo_update = {}
        self.exclusive_pass = {} # num_pass that pass with update but not without
        self.inclusive_pass = {} # num_pass that pass with and without update
        
        self.exec_result_w_update = defaultdict(str)
        self.exec_result_wo_update = defaultdict(str)
        # self.non_exclusive_unit_tests = {}
    
    # def generate_report
    # def check_exec_result_pair(self, exec_result_w_update, exec_result_wo_update):
    @property
    def num_pass_w_update(self):
        if not hasattr(self, "_num_pass_with_update"):
            self._num_pass_with_update = sum(self.pass_w_update.values())
        return self._num_pass_with_update

    @property
    def pass_w_update_rate(self):
        assert len(self.pass_w_update) == self.num_unit_tests
        if not hasattr(self, "_pass_w_update_rate"):
            self._pass_w_update_rate = self.num_pass_w_update / self.num_unit_tests
        return self._pass_w_update_rate
    
    @property
    def num_pass_wo_update(self):
        if not hasattr(self, "_num_pass_without_update"):
            self._num_pass_without_update = sum(self.pass_wo_update.values())
        return self._num_pass_without_update
    
    @property
    def pass_wo_update_rate(self):
        assert len(self.pass_wo_update) == self.num_unit_tests
        if not hasattr(self, "_pass_wo_update_rate"):
            self._pass_wo_update_rate = self.num_pass_wo_update / self.num_unit_tests
        return self._pass_wo_update_rate
    
    @property
    def num_exclusive_pass(self):
        if not hasattr(self, "_total_exclusive_pass"):
            self._total_exclusive_pass = sum(self.exclusive_pass.values())
        return self._total_exclusive_pass
    
    @property
    def num_inclusive_pass(self):
        if not hasattr(self, "_total_inclusive_pass"):
            self._total_inclusive_pass = sum(self.inclusive_pass.values())
        return self._total_inclusive_pass

    @property
    def all_pass_w_update(self):
        assert len(self.pass_w_update) == self.num_unit_tests, \
            "Not all tests has been run *with* update"
        return self.num_pass_w_update == self.num_unit_tests
    
    @property
    def all_pass_wo_update(self):
        assert len(self.pass_wo_update) == self.num_unit_tests, \
            "Not all tests has been run *without* update"
        return self.num_pass_wo_update == self.num_unit_tests
    
    @property
    def finish_testing(self):
        return len(self.pass_w_update) == self.num_unit_tests == len(self.pass_wo_update)
    
    def generate(self) -> str:
        # generate final reports
        ret = "\n".join([
            f"Finish Testing: {self.finish_testing}",
            f"Pass w update: {self.num_pass_w_update} / {self.num_unit_tests}",
            f"Pass wo update: {self.num_pass_wo_update} / {self.num_unit_tests}",
            f"Exclusive pass: {self.num_exclusive_pass} / {self.num_unit_tests}",
            f"Inclusive pass: {self.num_inclusive_pass} / {self.num_unit_tests}",
            "", "",
        ])
        details = []
        assert all(t_i in self.unit_tests for t_i in range(self.num_unit_tests))
        for t_i in range(self.num_unit_tests):
            details.append(
                "\n".join(
                    [
                        f"Test {t_i}:",
                        # self.unit_tests[t_i],
                        f"Exclusive pass: {self.exclusive_pass[t_i]}",
                        f"Exec with update: {self.exec_result_w_update[t_i]}",
                        f"Exec without update: {self.exec_result_wo_update[t_i]}",
                    ]
                )
            )
        ret += "\n\n".join(details)
        return ret