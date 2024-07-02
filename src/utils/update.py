import ast
import importlib
import inspect
import json
import os
from os.path import dirname
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple, Dict
import string 

from bigtree import Node
from openai import AsyncOpenAI, OpenAI
import sys

# from src.execution.safe_execution_util import execute
# from data.prelim.prompt import doc_summarization_prompt
from src.utils.utils import call_openai_chat
from src.data.prompt_update import PYTHON_INDENT, doc_summarization_prompt, arguments_writer_sys_prompt, arguments_writer_input_prompt
from src.utils.code import concat_and_exec
from copy import deepcopy
from inspect import signature

import numpy as np

DOC_STRING_ROOT_NAME = "update_functions" # it assume py
DOC_STRING_DIR = f"{dirname(__file__)}/{DOC_STRING_ROOT_NAME}"
os.makedirs(DOC_STRING_DIR, exist_ok=Tuple)


class Action(Enum):
    Add = "add"
    Modify = "modify"
    Deprecate = "deprecate"

class Place(Enum):
    Function = "function"
    Argument = "argument"
    Output = "output"
    
class Aspect(Enum):
    Null = "null"
    Name = "name"
    Semantic = "semantics"
    Data_type = "data type"
    Default_values = "default value(s)"
    Supported_values = "supported value(s)"


class UpdateType:
    EMPTY = "[Empty Update]"
    CONNECTOR = "-"
    def __init__(
        self, 
        action: str | Action,
        place: str | Place,
        aspect: str | Aspect,
        ) -> None:
        # TODO: optionally: examples, update-specific constraints
        # TODO: if want to have optional parts, store them in an external files
        self.action = Action(action.lower()) if isinstance(action, str) else action
        self.place = Place(place.lower()) if isinstance(place, str) else place
        self.aspect = Aspect(aspect.lower()) if isinstance(aspect, str) else aspect
        assert isinstance(self.action, Action)
        assert isinstance(self.place, Place)
        assert isinstance(self.aspect, Aspect)
        
        assert " " not in self.action.value
        assert " " not in self.place.value
        self.tag = f"{self.action.value}{self.CONNECTOR}{self.place.value}"
        if self.aspect != Aspect.Null:
            self.tag += self.CONNECTOR + self.aspect.value.replace(' ', '_')
        
        self.description = self.describe(self.action, self.place, self.aspect)
        self.is_banned_type = self.description.startswith(self.EMPTY)
    
    @classmethod
    def describe(cls, action, place, aspect) -> str:
        action = Action(action.lower()) if isinstance(action, str) else action
        place = Place(place.lower()) if isinstance(place, str) else place
        aspect = Aspect(aspect.lower()) if isinstance(aspect, str) else aspect
        assert isinstance(action, Action)
        assert isinstance(place, Place)
        assert isinstance(aspect, Aspect)
        ret = ""
        if action == Action.Modify:
            # "[] changes"
            ret += "{0} changes{1}"
            if place == Place.Function:
                if aspect in [Aspect.Semantic, Aspect.Name]:
                    ret = ret.format(
                        f"the function's *existing* {aspect.value}", ""
                    )
                else:
                    ret = f"{cls.EMPTY}: Unlikely update combination."
            elif place == Place.Argument:
                if aspect not in [Aspect.Null]:
                    ret = ret.format(
                        f"one of the *existing* function arguments' {aspect.value}", 
                        "(i.e. one replaces another, WITHOUT adding new arguments)"
                    )
                else:
                    ret = f"{cls.EMPTY}: Unlikely update combination."
            else:
                assert place == Place.Output
                if aspect in [Aspect.Semantic, Aspect.Data_type]:
                    ret = ret.format(
                        f"the existing function output's {aspect.value}",
                        " (i.e. one replaces another)"
                    )
                else:
                    ret = f"{cls.EMPTY}: Unlikely update combination."
        elif action == Action.Add:
            # "[1] is added [2]"
            ret += "{0} is added{1}"
            if place == Place.Function:
                if aspect in [Aspect.Null]:
                    ret = ret.format("a new (previously non-existent) function", "")
                else:
                    return f"{cls.EMPTY}: Overlaps with Action.Modify."
            elif place == Place.Argument:
                if aspect == Aspect.Null:
                    ret = ret.format(f"a new function argument", "")
                elif aspect != Aspect.Name:
                    ret = ret.format(
                        f"a new {aspect.value}", " to one of the function arguments"
                    )
                else:
                    ret = f"{cls.EMPTY}: Unlikely update combination."
            else:
                assert place == Place.Output
                if aspect in [Aspect.Semantic, Aspect.Data_type]:
                    ret = ret.format(
                        f"a new {aspect.value}", " to the existing function output"
                    )
                else:
                    ret = f"{cls.EMPTY}: Overlaps with Action.Modify."
        else:
            # "[]  is deprecated"
            assert action == Action.Deprecate
            ret += "{0} is deprecated{1}"
            # TODO: remove the ban
            ret = f"{cls.EMPTY}: TEMPERARILY BANNED"
            return ret
            if place == Place.Function:
                if aspect in [Aspect.Null]:
                    ret = ret.format("the existing function", "")
                else:
                    ret = f"{cls.EMPTY}: Overlaps with Action.Modify."
            elif place == Place.Argument:
                if aspect in [Aspect.Null]:
                    ret = ret.format(f"one of the existing function argument", "")
                elif aspect in [Aspect.Name]:
                    ret = f"{cls.EMPTY}: Overlaps with Action.Modify."
                else:
                    ret = ret.format(
                        f"one of the function arguments' {aspect.value}", ""
                    )
            else:
                assert place == Place.Output
                if aspect in [Aspect.Semantic, Aspect.Data_type]:
                    ret = ret.format(
                        f"part of the existing {aspect.value}", " from function output"
                    )
                else:
                    ret = f"{cls.EMPTY}: Overlaps with Action.Modify."
            
        return ret
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self) -> str:    
        return f"{self.tag}: {self.description}"
    
    @classmethod
    def from_tag(cls, full_tag: str):
        assert "-" in full_tag
        factors = full_tag.split("-")
        assert 2 <= len(factors) <= 3
        
        if len(factors) == 2:
            factors.append(Aspect.Null.value)
        try:
            ret = cls(*factors)
        except Exception as e:
            # aspect might contains underscore
            factors[-1] = factors[-1].replace('_', ' ')
            ret = cls(*factors)
        return ret
    

class Taxonomy:
    def __init__(self) -> None:
        root = Node(name="root", node_name="")
        # indent = "\t"
        c = 0
        non_banned_c = 0

        for action in Action:
            # print(indent * 0 + f"{action.name}")
            action_node = Node(name=action.value, parent=root)
            for place in Place:
                place_node = Node(name=place.value.replace(' ', '_'), parent=action_node)
                # print(indent * 1 + f"{place.name}")
                for aspect in Aspect:
                    c += 1
                    # print(indent * 2 + f"{aspect.name}")
                    u = UpdateType(action, place, aspect)
                    if not u.is_banned_type:
                        non_banned_c += 1
                        # print(indent * 3 + f"{u} -- {u.description}")
                        aspect_node = Node(name=aspect.value, parent=place_node)
                        update_node = Node(name=u, parent=aspect_node)
        self.root = root
        self.logically_possible_update = c
        self.practically_possible_update = non_banned_c
        self.all_updates = self.get_all_legal_updates()

    def get_all_legal_updates(self,) -> List[UpdateType]:
        return [l.name for l in list(self.root.leaves) if isinstance(l.name, UpdateType)]
    
    def sample_updates(self, n):
        rand_ids = np.arange(len(self.all_updates))
        np.random.shuffle(rand_ids)
        rand_ids = rand_ids[:n]
        return [deepcopy(self.all_updates[i]) for i in rand_ids]


class UpdatedFunction:
    def __init__(
        self, 
        api_path: str,
        force_summarization: Optional[bool] = False,
        docstring_dir: Optional[str] = DOC_STRING_DIR,
        ) -> None:
        assert "." in api_path
        path_parts = api_path.split(".")
        self._api_path = api_path
        self.path_parts = api_path.split(".")
        self.package_name: str = path_parts[0]
        self.parent_path: str = ".".join(path_parts[:-1])
        self.function_name: str = path_parts[-1]
        self.function_obj = self.get_function_obj(api_path)
        
        json_dict: Dict = self.attempt_to_load_json(docstring_dir, api_path)
        self.imports = json_dict.get("imports", None)
        if self.imports is None:
            self.imports = self.get_imports_from_api_path(api_path)
        
        self.source_code = json_dict.get("source_code", None)
        if not self.source_code:
            try:
                self.source_code = inspect.getsource(self.function_obj)
            except Exception as e:
                # Sometimes, the function may be implemented in C, 
                # so no python source code is available
                self.source_code = None
         # this normalizes tab to spaces and left shifting the doc body 
         # to remove common leading spaces.
        self.doc_string = json_dict.get("doc_string", None) or \
            inspect.getdoc(self.function_obj)
        
        
        self.arguments_str = json_dict.get("arguments_str", None)
        self.return_type_hint = json_dict.get("return_type_hint", None)
        if (self.arguments_str is None or self.return_type_hint is None):
            try:
                self.init_function_signature(
                    self.source_code if self.source_code
                    else inspect.signature(self.function_obj) # as a backup choice
                )
            except:
                try:
                    # if source code is not available, func signature is likely to be unavailable
                    # ! very error prone: assuming the first line in docstring is function sign
                    sign_candidate = self.doc_string.strip().split("\n")[0]
                    if not sign_candidate.startswith("def"):
                        sign_candidate = f"def {sign_candidate}"
                    if not sign_candidate.startswith(":"):
                        sign_candidate = f"{sign_candidate} :"
                    self.init_function_signature(
                        sign_candidate
                    )
                except:
                    self.return_type_hint = ""
                    try:
                        self.arguments_str = str(signature(self.function_obj))
                        assert self.arguments_str[0] == "(" and self.arguments_str[-1] == ")"
                        self.arguments_str = self.arguments_str[1:-1].replace(" ", "")
                    except:
                        try:
                            self.arguments_str = self.arguments_writer(api_path)
                            self.arguments_str = self.arguments_str.replace(" ", "")
                        except Exception as e:
                            raise Exception(
                                f"Failed to *automatically* get function signature for `{api_path}`: {e}"
                            )
                    
            assert hasattr(self, "arguments_str") and \
                    hasattr(self, "return_type_hint")
        assert self.arguments_str is not None and self.return_type_hint is not None
        
        # attempt to load from existing processed json
        self.summarized_doc = json_dict.get("summarized_doc", None)
        if self.summarized_doc:
            self.doc_string_indented = self.get_indented_docstring(self.summarized_doc)
        
        if force_summarization or not self.summarized_doc:
            self.summarized_doc = self.docstring_summarizer(self.doc_string)
            self.doc_string_indented = self.get_indented_docstring(self.summarized_doc)
            self.save_as_json(docstring_dir)
    
    @classmethod
    def docstring_summarizer(cls, documentation: str) -> str:
        OPENAI_MODEL = "gpt-4"
        CLIENT = OpenAI()
        response = call_openai_chat(
            CLIENT,
            sys_prompt=doc_summarization_prompt,
            user_prompt=documentation,
            model=OPENAI_MODEL,
        )
        tmp = json.loads(response.model_dump_json())
        return tmp["choices"][0]["message"]["content"]
    
    @classmethod
    def arguments_writer(cls, api_path: str) -> str:
        OPENAI_MODEL = "gpt-4"
        CLIENT = OpenAI()
        func_obj = cls.get_function_obj(api_path)
        documentation = func_obj.__doc__
        instantiated_input_prompt = arguments_writer_input_prompt.render(
            full_api_path=api_path,
            documentation=documentation,
        )
        response = call_openai_chat(
            CLIENT,
            sys_prompt=arguments_writer_sys_prompt,
            user_prompt=instantiated_input_prompt,
            model=OPENAI_MODEL,
        )
        tmp = json.loads(response.model_dump_json())
        return tmp["choices"][0]["message"]["content"]

    @classmethod
    def attempt_to_load_json(self,
        doc_directory: str, 
        api_path: str,
        ) -> Optional[Dict]:
        file_path = f"{doc_directory}/{api_path}.json"
        if not os.path.exists(file_path):
            return {}
        return json.load(open(file_path, "r"))
    
    
    @classmethod
    def get_indented_docstring(cls, docstring: str) -> str:
        return "\n".join([
                PYTHON_INDENT + l
                for l in docstring.split("\n")
                ])
    
    @classmethod
    def load_from_json(cls, json_file_path):
        assert os.path.exists(json_file_path), f"File not exist: {json_file_path}"
        json_dict = json.load(open(json_file_path, "r"))
        return cls(**json_dict)
        
    def save_as_json(self, saved_directory, force_save=True):
        
        file_path = f"{saved_directory}/{self.full_path}.json"
        assert self.summarized_doc is not None
        if os.path.exists(file_path) and not force_save:
            return

        output = {
            "api_path": self.full_path,
            "source_code": self.source_code,
            "doc_string": self.doc_string,
            "arguments_str": self.arguments_str,
            "return_type_hint": self.return_type_hint,
            "summarized_doc": self.summarized_doc,
            "imports": self.imports,
        }
        with open(file_path, "w") as f:
            json.dump(output, f, indent=4, sort_keys=True)

    @property
    def function_signature(self) -> str:
        return f"{self.full_path}({self.arguments_str}){self.return_type_hint}:"
    
    def init_function_signature(self, source_code):
        signature_parts = self.get_function_signature(source_code)
        # >= bc there could be helper function 
        assert len(signature_parts) >= 1, \
            "There should be just at least one signature in the code"
        # arguments_str might contains special internal package variable
        # (e.g. np._NoValue), but using inspect.signature() will give 
        # `<no value>`, which is no even python
        match_idx = [
            i for i, (f, _, _) in enumerate(signature_parts)
            if f.strip() == self.function_name
        ]
        assert len(match_idx) >= 1, "Bug in signature extraction"
        match_idx = match_idx[0]
        _, self.arguments_str, self.return_type_hint = signature_parts[match_idx]
    
    @classmethod
    def get_function_signature(cls, source_code) -> List[Tuple[str, str, str]]:
        # prompt: Given a python function source code, write me a regex to extract the signature from it. Remember, 1. function signature could takes multiple line, 2. there could be type hint in arguments and output; 3. input arguments could optionally have be default value (e.g. function call, primitive values)
        # see some development examples https://regex101.com/r/W7IqUf/1
        # ! important: no guarantee is made for grammaticality
        pattern = r"def\s+(\w+)\s*\(([^)]*?[^:]*?)\)\s*((?:->\s*.+(?:\\\s*\n\s*.+)*)?\s*):"
        matches = re.findall(pattern, source_code, re.MULTILINE) # Doesn't work with re.DOTALL
        ret = []
        for match in matches:
            function_name = match[0].strip()
            # remove empty spaces, and the possible `,` at the end
            arguments = match[1].translate({ord(c): None for c in string.whitespace})
            # arguments = arguments[:-1] if arguments[:-1] == "," else arguments
            return_type = match[2].strip() if match[2].strip() else ""
            ret.append([function_name, arguments, return_type])
        return ret
    
    @classmethod
    def sort_imports(cls, imports: List[str]):
        if len(imports) <= 1:
            return imports
        max_import_len = max(len(e) for e in imports)
        return sorted(
            set(imports), 
            key=lambda x: max_import_len * 2 if "=" in x else len(x)
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
    def get_imports_from_api_path(cls, api_path):
        try:
            path_parts = api_path.split(".")
            path_parts = api_path.split(".")
            ret = ["import " + ".".join(path_parts[:i+1]) for i in range(len(path_parts) - 1)]
            
            ret = cls.filter_imports(ret)
            return ret
        except Exception as e:
            raise ValueError(f"Invalid API path: {api_path}")
    
    @classmethod
    def get_function_obj(cls, api_path):
        try:
            path_parts = api_path.split(".")
            ret = importlib.import_module(path_parts[0])
            for part_name in path_parts[1:]:
                ret = getattr(ret, part_name)
            return ret
        except Exception as e:
            raise ValueError(f"Invalid API path: {api_path}")
        
    @property
    def full_path(self):
        return self._api_path

    # @property
    def get_imports(self):
        return "Null"


def get_imports(self):
    return self.imports


if __name__ == "__main__":
    import numpy as np
    import inspect
    # uf = UpdatedFunction("pandas.DataFrame.groupby")
    # uf = UpdatedFunction("pandas.DataFrame.select_dtypes")
    # uf = UpdatedFunction("pandas.DataFrame.eq")
    # uf = UpdatedFunction("pandas.DataFrame.where")
    # uf = UpdatedFunction("pandas.DataFrame.value_counts")
    # uf = UpdatedFunction("pandas.DataFrame.apply")
    # uf = UpdatedFunction("itertools.chain")
    functions = [
        "re.match",
        "re.search",
        "re.findall",
        "re.split",
        "re.sub",
        "random.random",
        "random.randint",
        "random.choice",
        "random.shuffle",
        "itertools.chain",
        "itertools.product",
        "itertools.groupby",
        "akshare.stock_zh_a_daily",
        "ast.parse",
        "copy.deepcopy",
        "copy.copy",
        "pandas.DataFrame.describe",
        "pandas.concat",
        "pandas.merge",
        "pandas.DataFrame.head",
        "math.sqrt",
        "math.pow",
        "math.radians",
        "math.degrees",
    ]
    # for function in functions:
    #     uf = UpdatedFunction(function)
    # print()
    
    tx = Taxonomy()
    print()