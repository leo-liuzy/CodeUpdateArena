import jinja2
from src.utils import create_test_calls
from typing import List
from src.data.prompt_update import NUM_UNIT_TESTS, PYTHON_INDENT, construct_instruction_series
# below are Prompt template
environment = jinja2.Environment()

NUM_UNIT_TESTS = 5
PYTHON_INDENT = 4 * " "


N_PROG_SYN_EXAMPLES = 1
# You will be given a function signature, optionally along with its docstring, and the Python library it belongs to. You will think what hypothetical but still realistic update could happen to the function signature.

prog_syn_description = [
    ["the problem scenario posed by the program synthesis example MUST follow the general functionality of the (old and new) API."],
    ["the problem scenario MUST be affected and preferably benefited by the API update. By benefit, it means the code complexity of the solution will be reduced."], # TODO: what about deprecation
    ["the problem MUST be at least medium hard, so that the solution MUST make *non-trivial* use of the API's functionality."],
    # ["the solution to the problem MUST make use of the API's functionality in a *non-trivial* way."],
    # ["Any solution (i.e. Python function) to the problem should have the same function name."],
    # ["Be given the desired difficulty of the problem on a scale from 1 (easiest) to 5 (hardest)."],
    ["Be given the number of parameters that the solution accepts."],
]

NUM_PROG_SYN_UNIT_TESTS = 6
prog_syn_json_output = [
    [""""scenario": (as string) a real-world scenario that the problem is situated in. Keep it medium short.""",
     """Avoid including information -- e.g. exact term -- about API changes, or package needs to be used in "problem"."""],
    [""""problem": (as string) problem specification that needs solving by a Python function. Keep it short.""",
     """Avoid giving imperative instruction on how to solve the problem. MUST Remain at high-level. Avoid including information -- e.g. exact term -- about API changes, or package needs to be used in "problem".""",
     """Make sure the description of the input is well connected and blend into the description of the scenario.""",
     """Design the problem such that each input to the solution is meaningfully used in the code.""",
     ],
    [""""solution_signature": (as string) the function signature of the solution function.""",
     """the function name should be derived from "scenario"."""],
    # [""""solution_": (as string) a solution that uses NEITHER *new* NOR *old* signatures."""],
    
]

prog_syn_sys_prompt_template = environment.from_string(f"""
You are a helpful assistant. You think deeply and creatively.
Your task is to think of and write interesting tutorial(s) for an API update. mainly <problem, solution>.

You will be given the full information about an update to an existing Python package. You should think of usage (i.e. program synthesis example) of the updated API signature that satisfy the following criteria:
{construct_instruction_series(prog_syn_description, numbered=False)}


Return the entire response in JSON format as a dictionary. Make sure nested brackets are closed correctly. Be careful with unterminated string literal. The dictionary should contain the following:
{construct_instruction_series(prog_syn_json_output)}

Give me {N_PROG_SYN_EXAMPLES} diverse program synthesis example(s).
""".strip())


prog_syn_input_prompt_template = environment.from_string("""
In Python package `{{package_name}}`, there’s an API function `{{api_path}}` as follows:
[OLD_SIGN]
{{old_func}}
[/OLD_SIGN]

Maintainer of the package thinks it's best to introduce the following update
[DESC]
{{update_description}}
[/DESC]

This is because 
[RATIONALE]
{{update_rationale}}
[/RATIONALE]

The function docstring now differs with previous version in the following way:
[DOC]
{{docstring_diff}}
[/DOC]

And the function has the following new signature:
[NEW_SIGN]
{{new_function_signature}}
[/NEW_SIGN]


The problem *MUST* non-trivially benefit from the update (i.e. new API); so that solving the problem with the old API is not possible, or requires more efforts (e.g. need to write longer code).
The solution of the problem must accept {{num_param}} parameter(s). 

Note:
Only output the JSON in raw text.
""".strip())

unit_test_skeleton_instruction = [
    ["""You MUST READ the problem specification (between "[PROBLEM]" and "[/PROBLEM]") WORD-BY-WORD and understand it PERFECTLY WELL.""", 
     """Also, IDENTIFY important arguments: the more important arguments are ranked to the front in the new function signature."""],
    ["""For unit tests, READ the scenario description (between [SCENARIO]...[/SCENARIO]) WORD-BY-WORD and understand it PERFECTLY WELL.""",
     """Contemplate, and think of a diverse set of representative inputs to solution function; this set of input should capture possible and interesting cases which solution function might encounter after deployment.""",
     """BE SURE to test ALL specified behaviors in the problem specification --- edge-case input, edge-case output, exception raised, etc.""",
     """You need to have different edge-case values for the update and each important arguments (e.g., multi-dimensional input array with different `axis` values).""",
     ],
    ["""When you generate a new unit test, look CAREFULLY at already generated unit tests, and make sure the inputs are different from previously generated unit tests as much as possible.""",
     """You MUST have proper setup code for solution function inputs: initialize variables for testing the updated --- literally, or randomly generated, etc. INCLUDE in-line comments.""",
     """PREFERABLY, the input to the solution function call SHOULD foreseeably lead to a *unique* execution result."""],
    [
        """The output of the solution function MUST be assigned to a variable `result`.""", 
        """You MUST call the solution function.""",
    ],
    # ["""After input is decided, do you think the input to solution function leads to *a unique answer*? Choose between one of the following strategies:
    # a. If Yes, check the equality of `result` and `expected_result`. 
    # b. Otherwise (i.e. if No), check if the value of `result` is among `expected_results` --- a list of all possible correct answers.""",
    # """You MUST write the right code to check the equality of `result` and `expected_result`, or `result` is within `expected_results`. REMEMBER: it's nontrivial to check equality of two variables. FOR EXAMPLE, the truth value of a Series (pandas) is ambiguous and one need to use a.empty, a.bool(), a.item(), a.any() or a.all(). Think VERY CAREFULLY before  using `==` to check equality.""",
    # ],
    # ["""Then, instead of figuring out the right answer, you MUST output a code block placeholder "expected_result(s) = [INFILL]". The place holder represent a code block that will eventually arrive at a correct answer or a set of correct answers.""",
    #  """If the test input is meant to testing error catching, check if the call to solution function will raise error. DON'T check error message. DON'T write the placeholder."""],
    [
        """If a unit test function is testing throwing exception, you should proceed with `try-except` and finish the unit test function.""",
        """If the test input is meant to testing error catching, check if the API call will raise error. DON'T check error message.""",
    ],
    [
        """If a unit test function is NOT testing throwing exception:""",
        """You MUST output a placeholder `# @ANSWER@` for the right answer to be filled in. Writing the right answer is forbidden.""",
        """Do not write any assertion. This is forbidden. Instead, put a placeholder `# @ASSERT@` at the end of the test function.""",
        """Within the unit function, the placeholders need to start at the left-most indent (i.e. 4 empty spaces --- "    ").""",
    ],
    ["""Each test MUST be a function without any input arguments. DON'T attempt to test I/O in each unit tests.""",],
    ["""The function name MUST be informative. Avoid it to include generic terms like "case1" or "test1"."""],
    ["""Use "\\n" as line break. Use 4 empty spaces ("    ") as Python code block indent."""],
]

unit_test_skeleton_sys_prompt_template = environment.from_string(f"""
You are a very experienced programer. You are good at algorithmic reasoning and writing super high quality code.

Your task is to write {{{{num_unit_tests}}}} *high-quality* and *comprehensive* unit tests skeletons for testing validity of any solution function to a problem specification. A unit test skeleton is a unit test function except the right answer being clearly specified. Each unit test skeleton MUST be in raw string, not in Python code block.

Return the set of unit tests skeletons in JSON code block as a list of string. For unit test skeletons generation, following the instructions below:
{construct_instruction_series(unit_test_skeleton_instruction)}
""".strip())

unit_test_skeleton_input_prompt_template = environment.from_string(f"""
In a real-world scenario, there exists some trouble to be solved:
[SCENARIO]
{{{{scenario}}}}
[/SCENARIO] 

Luckily, someone could solve this trouble by writing a function, as long as the solution function satisfy the following problem specification:
[PROBLEM]
{{{{problem}}}}
[/PROBLEM]

Additionally, the solution function should have the following function signature:
[SOLUTION_SIGN]
{{{{solution_signature}}}}
[/SOLUTION_SIGN]

{{% if package_instruct %}}
Some special notes for `{{{{package_name}}}}` package:
{{{{package_instruct}}}}
{{% endif %}}

Only output the set of unit tests skeletons (*a list of strings*) in JSON code block (```json...```).
""".strip())

unit_test_ans_instructions = [
    # [""""input_validity" (as string): check whether input in the unit test skeleton is valid. Return "True" or "False""""],
    ["""You MUST READ the problem specification (between "[PROBLEM]" and "[/PROBLEM]") WORD-BY-WORD, take a pause and, understand it PERFECTLY WELL.""", 
     """Now look at the values of input to the solution function, and contemplate on the expected behavior of the solution function given those inputs."""],
    ["""IDENTIFY whether you need to assign value to `expected_result` or `expected_results`. There is only one right choice."""],
    ["""Before arriving at an answer, ALWAYS take a deep breath and work on this problem STEP-BY-STEP. Then, wisely choose one of the strategies from below:
    a. an assignment of a Python literal value to the variable;
    b. if the literal is too long or it's best to use arithmetics to get the value, DON'T write literal value. INSTEAD, use step-by-step program code to express how to arrive at the answer.""",
     ],
    # ["""In the code block, DO NOT call the *new* API function. However, for your calculation, you CAN call the *old* API function by "old_[old API function name]"; [old API function name] is the function name without the API path. For example, "old_argmax", NOT "numpy.old_argmax" or "np.old_argmax"."""],
    ["""Within the code block, you MUST generate WITH NO leading indent. Use 4 empty spaces ("    ") as indent when writing if-else, for-loop, etc.""",],
]

unit_test_ans_sys_prompt_template = environment.from_string(f"""
You are a very experienced programer. You are good at algorithmic reasoning and writing super high quality code.

In a real-world scenario, there exists some trouble to be solved:
[SCENARIO]
{{{{scenario}}}}
[/SCENARIO] 

Luckily, someone could solve this trouble by writing a function, as long as the solution function satisfy the following problem specification:
[PROBLEM]
{{{{problem}}}}
[/PROBLEM]

An ideal solution function takes the following function signature:
[SOLUTION_SIGN]
{{{{solution_signature}}}}
[/SOLUTION_SIGN]

You will be a unit test skeleton with a `# @ANSWER@`. Your task is to generate a Python code block (```python...```) to replace "`# @ANSWER@". The purpose of the code block is to calculate a value for a variable called `expected_result` or `expected_results`. 

For generating the code block, following the instructions below:
{construct_instruction_series(unit_test_ans_instructions)}
""".strip())

unit_test_ans_input_prompt_template = environment.from_string(f"""
{{% if include_api_info %}}
To write code to calculate `expected_result` or `expected_results` (strategy b), maybe the following two functions are useful:

The first function comes from package `numpy`.
[FUNCTION1]
{{{{old_function_signature}}}}
[/FUNCTION1]

The second function is an updated version of the FUNCTION1
[FUNCTION2]
{{{{new_function_signature}}}}
[/FUNCTION2]

FUNCTION2 differs from FUNCTION1 in the following way:
[DOC]
{{{{update_docstring}}}}
[/DOC]
{{% endif %}}
[TEST]
{{{{unit_test_skeleton}}}}
[/TEST]
{{% if package_instruct %}}
Some special notes for `{{{{package_name}}}}` package:
{{{{package_instruct}}}}
{{% endif %}}
""".strip())

unit_test_assert_sys_prompt_template = environment.from_string(f"""
You are a very experienced programer. You are good at algorithmic reasoning and writing super high quality code.

You will be given a unit test function that misses assertion statements to either:
1. check equivalence between `result` and `expected_result`
2. or check equivalence between `result` and any values in `expected_results` ( i.e. multiple correct answer).

Your task is to generate a Python code block (```python...```) to replace `# @ASSERT@`.
""".strip())

unit_test_assert_input_prompt_template = environment.from_string(f"""
[TEST]
{{{{unit_test_skeleton}}}}
[/TEST]

{{% if package_instruct %}}
Remember some special features of `{{{{package_name}}}}` package:
{{{{package_instruct}}}}
{{% endif %}}
""".strip())

solution_new_instructions = [
    ["""First of all, you MUST CAREFULLY READ the problem specification (between "[PROBLEM]" and "[/PROBLEM]") WORD-BY-WORD and understand it PERFECTLY WELL.""",],
    ["""Before arriving at the solution function, take a deep breath and work on this problem STEP-BY-STEP.""",
     """INCLUDE in-line comments and improve readability.""",
     """If you are provided with unit tests, use them to understand expected behavior of the solution function.""",],
    ["""Notice any error handling specified in the problem specification. INCLUDE error handling when writing solution.""",],
    ["""The solution signature MUST follows the one specified between "[SOLUTION_SIGN]" and "[/SOLUTION_SIGN]".""",
     """You MUST NOT write documentation for the solution.""",],
    ["""To implement the solution, you MUST use the *new* API function AS MUCH AS POSSIBLE.""",
     ],
    ["""Use 4 empty spaces ("    ") as Python code block indent.""",],
]

solution_new_sys_prompt_template = environment.from_string(f"""
You are a very experienced programer. You are good at algorithmic reasoning and writing super high quality code.

The API of interest is
[OLD_SIGN]
{{{{old_function_signature}}}}
[/OLD_SIGN]

This API recently undergoes an update and it now has the following new function signature:
[NEW_SIGN]
{{{{new_function_signature}}}}
[/NEW_SIGN]

This is the documentation that details the behavior about the update:
[DOC]
{{{{update_docstring}}}}
[/DOC]

You will be given the detailed problem specification. Your task is to USE the new API (between "[NEW_SIGN]" and "[/NEW_SIGN]") to write high quality solution function that solve the problem specification in Python code block (```python...```).

To generate the code block, following the instructions below:
{construct_instruction_series(solution_new_instructions)}
""".strip())

solution_new_input_prompt_template = environment.from_string(f"""
[PROBLEM]
{{{{problem}}}}
[/PROBLEM]

Solution should take the following singautre
[SOLUTION_SIGN]
{{{{solution_signature}}}}
[/SOLUTION_SIGN]
{{% if unit_tests %}}
Unit tests for new update:
[PYTHON]
{{%- for test in unit_tests %}}
# Unit Test {{{{loop.index}}}}
{{{{test}}}}
{{% endfor -%}}
[/PYTHON]
{{% endif %}}
USE the new API (between "[NEW_SIGN]" and "[/NEW_SIGN]") to write high quality solution function that solve the problem specification in Python code block (```python...```).
Only output the new implementation in Python code block (```python...```).
""".strip())

# TODO: write prompts for solution_new
# TODO: write prompts for solution_none
# TODO: write prompts for solution_old



if __name__ == "__main__":
    sys_prompt = prog_syn_sys_prompt_template.render(
        update_description="The function now accepts a new argument: 'reverse' that allows the indices to be returned in descending order of the sorted array.",
        old_function_signature="numpy.argsort(a,axis=-1,kind=None,order=None):",
        new_function_signature="numpy.argsort(a, axis=-1, kind=None, order=None, reverse=False):",
        renamed_old_function_signature="old_argsort(a,axis=-1,kind=None,order=None):"
    )
    input_prompt = prog_syn_input_prompt_template.render(
        update_docstring="The 'reverse' argument is a boolean flag and defaults to False. If set to True, the sorted indices are returned in the descending order of their corresponding values.",
    )
    print()