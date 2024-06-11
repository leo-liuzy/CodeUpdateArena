import jinja2
# from src.utils import create_test_calls
from typing import List
# below are Prompt template
environment = jinja2.Environment()

NUM_UNIT_TESTS = 5
PYTHON_INDENT = 4 * " "

def construct_instruction_series(expected_output: List[List[str]], numbered: bool=True):
    # construct the instrucitons for JSON output
    ret = ""
    for i, entries in enumerate(expected_output, start=1):
        assert isinstance(entries, list) and len(entries) > 0
        for j, entry in enumerate(entries):
            if numbered:
                # first element in `entries` is the major description
                ret += f"{i}: {entry}" if j == 0 else f"{i}.{j}: {entry}"
                ret += "\n"
            else:
                assert j < 1, "Nested (unnumbered) points are not supported yet"
                ret += f"* {entry}\n"
    ret = ret.strip()
    return ret



doc_summarization_instructions = [
    ["You MUST extract descriptions about the functionality, input parameters, and output from the original documentation."],
    ["You could include some illustrative code in the summary if the summary is ambiguous."],
    ["You MUST keep the most important information, e.g. description, data type, etc."],
    ["The reader of your summary MUST be able to implement the function with summarized documentation."],
    ["You MUST maintain  the original structure, format, and the style of the documentation."],
    ["Output the summarized documentation in text."],
]

doc_summarization_prompt = f"""
You are a helpful assistant
You will be given documentation for an API in a popular Python library.

You need to do the following:
{construct_instruction_series(doc_summarization_instructions)}
""".strip()

arguments_writer_sys_prompt = f"""
Infer argument of a Python function signature from documentation (output of `[full_api_path].__doc__`).

Function signature takes the form of 
```
[full_api_path]([arguments])
```
Output the right [arguments].

Note:
* Output raw text. 
* DO NOT Wrap output in a Python code block.
* DO NOT include documentation in the output.
""".strip()

arguments_writer_input_prompt = environment.from_string(f"""
Full API path: 
{{{{full_api_path}}}}

Documentation:
{{{{documentation}}}}
""".strip()
)


update_descriptions = [
    ["""The update should make the call site of the old function to be un-executable and one need to follow the new function signature."""],
    ["""The update should be as atomic as possible. It only includes one of the three possible editing actions and only happens to one place of the functions. So that the new function signature and old signature only differs at one place."""],
    ["""The update should lead to a new function signature whose implementation is non-trivially different from the old ones. An undesirable result is that the new implementation trivially calls the old function."""],
    ["""The update should be a sensible change that fits the overall topic of the function and the Python library."""],
    ["""The update should NOT contradict existing functionality of the old function."""],
    ["""The update needs to be supported by a good reason for library designer to introduce it"""],
]

update_json_output = [
    [""""update_description": (as string) a short one-sentence description of the update."""],
    [""""rationale": (as string) why any hypothetical designer of the API might want to introduce these changes."""],
    [""""new_function_signature": (as string) the new function signature.""",
     """"new_function_signature" MUST start with the full reference to the function. For example, "numpy.mean" instead of "def mean"."""],
    [""""update_docstring": (as string) the added documentation that explains the new functionality of the atomic update. It MUST be self-contained, unambiguous, detailed but concise.""", 
     """You MUST succinctly explain the updated behavior of the new API, and how it differs from the old behavior.""",
     """The "update_docstring" MUST fully specify the behavior about the update. For example, how the changes in input would change the output of new API w.r.t. the old version.""",
     """A third-person MUST be able to develop a new implementation by just reading the "update_docstring" along with the old docstring.""",
     """"update_docstring" could take the form of natural language, numpy-style docstring, pseudo-code examples, etc. Make the most sensible choice. If it's a string with multiple lines, output "\\n" as line break.""",
     """DO NOT include example(s) of using the updated API in "update_docstring".""",
     ],
    # [""""update_demo": (as string) demonstrative code snippet of how to use the new function. If it's a string with multiple lines, output "\\n" as line break."""],
]

update_spec_sys_prompt_template = environment.from_string(f"""
You are a helpful assistant. You think deeply and creatively.
Your task is to assist users to think of and instantiate interesting cases of API update.

A desirable update should satisfy the following criteria:
{construct_instruction_series(update_descriptions, numbered=False)}

Return the entire response in JSON format as a dictionary. Make sure nested brackets are closed correctly. Be careful with unterminated string literal. The dictionary should contain the following:

{construct_instruction_series(update_json_output)}

You will be given a function signature, optionally along with its docstring, and the Python library it belongs to. You will think what realistic update could happen to the function signature.

Give me 1 example of possible update(s) that {{{{update_description}}}}. {{{{extra_update_description}}}}
""".strip()
)

utter_most_important_note = [
    [""""new_function_signature" MUST ONLY contain the function name, instead of the full reference to the function. For example, "mean" instead of "numpy.mean".""",],
    ["""Only output the JSON in raw text."""]
]

update_spec_input_prompt_template = environment.from_string(f"""
Package: {{{{parent_path}}}}

[DOC]
def {{{{function_signature}}}}
{{{{doc_string}}}}
[/DOC]

Note:
{construct_instruction_series(utter_most_important_note, numbered=False)}
""".strip()
)

unit_test_skeleton_instruction = [
    ["""You MUST READ the documentation (between "[DOC]" and "[/DOC]") WORD-BY-WORD and understand it PERFECTLY WELL.""", 
     """Also, IDENTIFY important arguments: the more important arguments are ranked to the front in the new function signature."""],
    ["""For unit tests, think of a diverse set of API update and the important arguments to test ALL specified behaviors in the documentation --- edge-case input, edge-case output, exception raised, etc.""",
     """You need to have different edge-case values for the update and each important arguments (e.g., multi-dimensional input array with different `axis` values).""",
     ],
    ["""When you generate a new unit test, look CAREFULLY at already generated unit tests, and make sure the inputs are different from previously generated unit tests as much as possible.""",
     """You MUST have proper setup code for API inputs: initialize variables for testing the updated --- literally, or randomly generated, etc. INCLUDE in-line comments.""",
     """PREFERABLY, the input to the updated API SHOULD foreseeably lead to a *unique* execution result."""],
    [
        """The output of the API call MUST be assigned to a variable `result`.""", 
        """You MUST call the updated API, instead of old API. If required, you are allowed to call the *old* API by directly calling `old_{{function_name}}`. ALL other ways to call the old function are FORBIDDEN.""",
    ],
    # ["""After input is decided, do you think the input to updated API leads to *a unique answer*? Choose between one of the following strategies:
    # a. If Yes, check the equality of `result` and `expected_result`. 
    # b. Otherwise (i.e. if No), check if the value of `result` is among `expected_results` --- a list of all possible correct answers.""",
    # """You MUST write the right code to check the equality of `result` and `expected_result`, or `result` is within `expected_results`. REMEMBER: it's nontrivial to check equality of two variables. FOR EXAMPLE, the truth value of a Series (pandas) is ambiguous and one need to use a.empty, a.bool(), a.item(), a.any() or a.all(). Think VERY CAREFULLY before  using `==` to check equality.""",
    # ],
    
    # ["""Then, instead of figuring out the right answer, you MUST output a code block placeholder "expected_result(s) = @INFILL@". The place holder represent a code block that will eventually arrive at a correct answer or a set of correct answers.""",
    #  """If the test input is meant to testing error catching, check if the API call will raise error. DON'T check error message. DON'T write the placeholder."""],
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
    ["""When you have Python string literal, you MUST use escape for quote --- `\\"` or `\\'`; for triple quote --- `\\"\\"\\"` or `\\'\\'\\'`"""],
]

unit_test_skeleton_sys_prompt_template = environment.from_string(f"""
You are a very experienced programer. You are good at algorithmic reasoning and writing super high quality code.

The API of interest is:
[OLD_SIGN]
{{{{old_function_signature}}}}
[/OLD_SIGN]

This API recently undergoes an update:
[DESC]
{{{{update_description}}}}
[/DESC]

The API now has the following new function signature: 
[NEW_SIGN]
{{{{new_function_signature}}}}
[/NEW_SIGN]

Your task is to write {{{{num_unit_tests}}}} *high-quality* and *comprehensive* unit tests skeletons for testing the validity of the update. A unit test skeleton is a unit test function that only specifies the test inputs. Each unit test skeleton MUST be in raw string, not in Python code block.

Return the set of unit tests skeletons in JSON code block as a list of string. For unit test skeletons generation, following the instructions below:
{construct_instruction_series(unit_test_skeleton_instruction)}
""".strip())

unit_test_skeleton_input_prompt_template = environment.from_string(f"""
This is the documentation that details the behavior about the update:
[DOC]
{{{{update_docstring}}}}
[/DOC]

{{% if package_instruct %}}
Some special notes for `{{{{package_name}}}}` package:
{{{{package_instruct}}}}
{{% endif %}}

Only output the set of unit tests skeletons (*a list of strings*) in JSON code block (```json...```).
Include `global {{{{package_name}}}}` as the first line of each unit test function.
If you want to call the old function, you MUST directly call `old_{{{{function_name}}}}`. All other ways to call the old function are FORBIDDEN.
""".strip())

unit_test_ans_instructions = [
    # [""""input_validity" (as string): check whether input in the unit test skeleton is valid. Return "True" or "False""""],
    ["""You MUST READ the documentation (between "[DOC]" and "[/DOC]") WORD-BY-WORD, take a pause and, understand it PERFECTLY WELL.""", 
     """Now look at the values of input to the API call, and contemplate on the expected behavior of the *new* API given those inputs."""],
    ["""IDENTIFY whether you need to assign value to `expected_result` or `expected_results` --- `expected_result` if there's only 1 correct answer; `expected_results` if there's only multiple correct answers. There is only one right choice."""],
    ["""Focus on the behavior of the *new* API. When deriving the expected value of `result`, work on this problem STEP-BY-STEP. Then, wisely choose one of the strategies from below:
    a. an assignment of a Python literal value to the variable;
    b. if the literal is too long or it's best to use arithmetics to get the value, DON'T write literal value. INSTEAD, use step-by-step program code to express how to arrive at the answer.""",
     ],
    ["""In the code block, DO NOT call the *new* API function. For calculating the answer, you CAN call the *old* API function. However, you MUST directly call `old_{{function_name}}`. ALL other ways to call the old function are FORBIDDEN.""",
     ],
    ["""Within the code block, you MUST generate WITH NO leading indent. Use 4 empty spaces ("    ") as indent when writing if-else, for-loop, etc.""",],
]

unit_test_ans_sys_prompt_template = environment.from_string(f"""
You are a very experienced programer. You are good at algorithmic reasoning and writing super high quality code.

The API of interest is
[OLD_SIGN]
{{{{old_function_signature}}}}
[/OLD_SIGN]

This API recently undergoes an update:
[DESC]
{{{{update_description}}}}
[/DESC]

The API now has the following new function signature:
[NEW_SIGN]
{{{{new_function_signature}}}}
[/NEW_SIGN]

You will be given the detailed documentation about the update, and a unit test skeleton with a `# @ANSWER@`. Your task is to generate a Python code block (```python...```) to replace `# @ANSWER@`. The purpose of the code block is to calculate a value for a variable called `expected_result` or `expected_results`. 

For generating the code block, following the instructions below:
{construct_instruction_series(unit_test_ans_instructions)}
""".strip())

unit_test_ans_input_prompt_template = environment.from_string(f"""
This is the documentation that details the behavior about the update:
[DOC]
{{{{update_docstring}}}}
[/DOC]

[TEST]
{{{{unit_test_skeleton}}}}
[/TEST]

If you want to call the old function, you MUST directly call `old_{{{{function_name}}}}`. All other ways to call the old function are FORBIDDEN.
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


new_impl_instructions = [
    ["""First of all, you MUST CAREFULLY READ the documentation about the update (between "[DOC]" and "[/DOC]") WORD-BY-WORD and understand it PERFECTLY WELL.""",],
    ["""Before arriving at the new implementation, take a deep breath and work on this problem STEP-BY-STEP.""",
     """INCLUDE in-line comments and improve readability.""",
     """If you are provided with unit tests, use them to understand expected behavior of the update.""",],
    ["""Notice any error handling specified in the documentation. INCLUDE error handling when writing new implementation.""",],
    ["""The new function's name should be the same as the name in new function signature, with API path removed.""",
     """You MUST NOT write documentation for the new implementation.""",
     """You MUST NOT output the old implementation.""",],
    ["""To implement the new function, you MUST use the *old* API function AS MUCH AS POSSIBLE.""",
     """Since the bulk part of the functionality is accomplished by the *old* API function, the new implementation MUST be as SUCCINCT as possible.""",
     """You MUST call the *old* API function by directly calling `old_{{function_name}}`. ALL other ways to call the old function are FORBIDDEN.""",],
    ["""DO NOT write imports.""",],
    ["""Use 4 empty spaces ("    ") as Python code block indent.""",],
]

new_impl_sys_prompt_template = environment.from_string(f"""
You are a very experienced programer. You are good at algorithmic reasoning and writing super high quality code.

The API of interest is
[OLD_SIGN]
{{{{old_function_signature}}}}
[/OLD_SIGN]

This API recently undergoes an update:
[DESC]
{{{{update_description}}}}
[/DESC]

The API now has the following new function signature:
[NEW_SIGN]
{{{{new_function_signature}}}}
[/NEW_SIGN]

And the old API is renamed to:
[OLD_SIGN]
{{{{renamed_old_function_signature}}}}
[/OLD_SIGN]

You will be given the detailed documentation about the update. Your task is to write high quality implementation for the *new* API function in Python code block (```python...```).

To generate the code block, following the instructions below:
{construct_instruction_series(new_impl_instructions)}
""".strip())

new_impl_input_prompt_template = environment.from_string(f"""
This is the documentation that details the behavior about the update:
[DOC]
{{{{update_docstring}}}}
[/DOC]
{{% if unit_tests %}}
Unit tests for new update:
[PYTHON]
{{%- for test in unit_tests %}}
# Unit Test {{{{loop.index}}}}
{{{{test}}}}
{{% endfor -%}}
[/PYTHON]
{{% endif %}}

If you want to call the old function, you MUST directly call `old_{{{{function_name}}}}`. All other ways to call the old function are FORBIDDEN.
You MUST NOT output the old implementation.
You MUST NOT implement `old_{{{{function_name}}}}`.
Only output the new implementation in Python code block (```python...```).
""".strip())



import_instructions = [
    ["""First of all, read the code WORD-BY-WORD and understand it PERFECTLY WELL.""",],
    ["""DO NOT miss type hints in function signature, function body, etc.""",],
    ["""If no import statements is required, output an empty Python code block.""",],
]

import_sys_prompt_template = environment.from_string(f"""
You are a very experienced programer. You are good at algorithmic reasoning and writing super high quality code.

Your task is to write import statements to include any package dependency before running the code. Return import statements in Python code block (```python...```). 

To generate the code block, following the instructions below:
{construct_instruction_series(import_instructions)}
""".strip())

import_input_prompt_template = environment.from_string(f"""
[PYTHON]
{{{{code}}}}
[/PYTHON]

Only output the Python code block (```python...```).
""".strip())



if __name__ == "__main__":
    sys_prompt = solution_new_sys_prompt_template.render(
        update_description="The function now accepts a new argument: 'reverse' that allows the indices to be returned in descending order of the sorted array.",
        old_function_signature="numpy.argsort(a,axis=-1,kind=None,order=None):",
        new_function_signature="numpy.argsort(a, axis=-1, kind=None, order=None, reverse=False):",
        renamed_old_function_signature="old_argsort(a,axis=-1,kind=None,order=None):"
    )
    input_prompt = solution_new_input_prompt_template.render(
        update_docstring="The 'reverse' argument is a boolean flag and defaults to False. If set to True, the sorted indices are returned in the descending order of their corresponding values.",
    )
    print()