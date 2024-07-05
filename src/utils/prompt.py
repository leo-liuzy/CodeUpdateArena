import re
import os
from typing import Dict
import jinja2

from overrides import overrides
from abc import ABC, abstractmethod 

from typing import (
    Any,
    List,
    Dict,
    Tuple,
    Optional,
    Union,
)

import re
from jinja2 import Environment, BaseLoader, Template

# below are Prompt template
environment = jinja2.Environment()

class Jinja2PromptTemplate(ABC):
    """Base class for prompt templates."""
    def __init__(self, template: Template):
        self.template = template

    @abstractmethod
    def render(self, data: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """Render the template with the given data."""
        pass

    @classmethod
    def from_file(cls, filename: str) -> "TextPromptTemplate":
        """Load a template from a file."""
        with open(filename, 'r') as f:
            text = f.read()
        template = Environment(loader=BaseLoader, keep_trailing_newline=True, trim_blocks=True).from_string(text)
        return cls(template)

    @classmethod
    def from_string(cls, text: str) -> "TextPromptTemplate":
        """Load a template from a string."""
        template = Environment(loader=BaseLoader, keep_trailing_newline=True, trim_blocks=True).from_string(text)
        return cls(template)

    @staticmethod
    def pretty_print(prompt: Any) -> str:
        """Pretty print messages."""
        raise NotImplementedError

class TextPromptTemplate(Jinja2PromptTemplate):
    """Template for text prompts."""

    def render(self, data: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        if data is not None:
            return self.template.render(**data, **kwargs)
        return self.template.render(**kwargs)

    @staticmethod
    def pretty_print(prompt: str) -> str:
        """Pretty print messages."""
        return prompt


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