import re
import os
from typing import Dict
import jinja2

from overrides import overrides
from abc import ABC, abstractmethod 

from src.utils import create_test_calls
from src.utils.prompt_utils import TextPromptTemplate


# below are Prompt template
environment = jinja2.Environment()


class CodeGenTemplate(TextPromptTemplate):
    
    @classmethod
    def solution_extractor(self, text):
        raise NotImplementedError("Abstract method `solution_extractor` needs implementing")


class InstructTemplate(CodeGenTemplate):
    
    @classmethod
    def solution_extractor(self, text):
        pattern = r'\[PYTHON\](.*?)\[/PYTHON\]'
        matches = re.findall(pattern, text, re.DOTALL)
        if len(matches) > 0:    
            return matches[0]
        # make a second attempt
        pattern = r'```(?:python)?(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        return matches[0] if len(matches) > 0 else ""

# if __name__ == "__main__":
    # open("prompt/instruction_style_one_shot.jinja2", "r").read()
    # a = InstructOneShotTemplate.from_file("prompt/instruction_style_one_shot.jinja2")
    # print()