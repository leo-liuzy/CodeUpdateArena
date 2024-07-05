import glob
import ast
import re
import os
import numpy as np

import tiktoken
import editdistance

from itertools import chain



class CodeProcessor(ast.NodeTransformer):
    def __init__(self, imported_packages):
        self.var_counter = 0
        self.var_mapping = {}
        self.imported_packages = imported_packages

    def visit_FunctionDef(self, node):
        node.returns = None

        # Anonymize variable names in function definition
        self.anonymize_funcid(node)
        node.args.args = [self.anonymize_arg(arg) for arg in node.args.args]
        node.body = [self.visit(child) for child in node.body]

        return node

    def visit_Name(self, node):
        # Anonymize variable names in function code
        if isinstance(node.ctx, ast.Store):
            return self.anonymize_name(node)
        
        if isinstance(node.ctx, ast.Load):
            return self.anonymize_name(node)

    def visit_arg(self, node):
        # Remove type annotations from function arguments
        node.annotation = None
        return node

    def visit_Return(self, node):
        # Remove type annotations from return value
        node.value = self.visit(node.value)
        return node

    def anonymize_funcid(self, node):
        if node.name in self.imported_packages:
            return node
        if node.name not in self.var_mapping:
            self.var_mapping[node.name] = f'var{self.var_counter}'
            self.var_counter += 1
        node.name = self.var_mapping[node.name]
        return node

    def anonymize_name(self, node):
        if node.id in self.imported_packages:
            return node
        if node.id not in self.var_mapping:
            self.var_mapping[node.id] = f'var{self.var_counter}'
            self.var_counter += 1
        node.id = self.var_mapping[node.id]
        return node
        
    def anonymize_arg(self, arg):
    
        if arg.arg not in self.var_mapping:
            self.var_mapping[arg.arg] = f'var{self.var_counter}'
            self.var_counter += 1
        arg.arg = self.var_mapping[arg.arg]
        arg.annotation = None
        return arg

def extract_imports(code):
    import_pattern = re.compile(r'import\s+(\w+(?:\.\w+)*)')
    from_import_pattern = re.compile(r'from\s+(\w+(?:\.\w+)*)\s+import\s+')

    imports = []
    
    # Extract 'import' statements
    for match in import_pattern.finditer(code):
        package = match.group(1)
        imports.append(package)
    
    # Extract 'from ... import' statements
    for match in from_import_pattern.finditer(code):
        package = match.group(1)
        imports.append(package)

    return imports

def process_code_ast(code):

    tree = ast.parse(code)
    imported_packages = extract_imports(code)

    processor = CodeProcessor(imported_packages)
    processed_tree = processor.visit(tree)
    processed_code = ast.unparse(processed_tree)
    return processed_code


def tokenize_code(code):
    # Load the "code llama" tokenizer
    global tokenizer

    # Tokenize the code string
    tokens = tokenizer.tokenize(code)

    return tokens

def tokenize_openai(code):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return encoding.encode(code)


def dir_listing(dirs):
    all_dirs = []
    for d in dirs:
        dirs_of_d = os.listdir(d)
        dirs_of_d = [os.path.join(d, x) for x in dirs_of_d if not x.startswith(".")]
        all_dirs.extend(dirs_of_d)
    return all_dirs


def main(original_codes):
    """_summary_

    Args:
        original_codes (List[str]): list of string; each entry is the original content of a python script.

    Returns:
        _type_: _description_
    """
    
    original_codes = []
    anonymized_codes = []
    for solution in original_codes:
        original_codes.append(solution)
        try:
            proc_solution = process_code_ast(solution)
        except:
            proc_solution =  solution

        anonymized_codes.append(proc_solution)

    
    similarity_pairs = []
    
    num_prog = len(anonymized_codes)
    for i in range(num_prog):
        orig_a, prog_a = original_codes[i], anonymized_codes[i]
        tok_prog_a = tokenize_openai(prog_a)
        for j in range(i + 1, num_prog):
            orig_b, prog_b = original_codes[j], anonymized_codes[j]
            tok_prog_b = tokenize_openai(prog_b)

            similarity_pairs.append({
                    "orig_a": orig_a,
                    "proc_a": prog_a,
                    "orig_b": orig_b,
                    "proc_b": prog_b,
                    "distance": editdistance.eval(tok_prog_a, tok_prog_b)
                })

    return similarity_pairs


