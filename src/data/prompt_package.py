from typing import List, Dict
from collections import defaultdict
from src.data.prompt_update import construct_instruction_series

class PromptHandler:
    UNALIASED_PACKAGE_NAME: str = ""
    GENERAL: List[List[str]] = []
    SPECIFIC = defaultdict(lambda: None, {})
    
    def __getitem__(self, api_path):
        assert isinstance(api_path, str), "index must be an API path"
        assert api_path.lower().startswith(self.UNALIASED_PACKAGE_NAME), \
            f"API path must start with `{self.UNALIASED_PACKAGE_NAME}`"
        
        return construct_instruction_series(
            self.GENERAL + (self.SPECIFIC[api_path] or [])
        )
    
class ReAssertHandler(PromptHandler):
    """
    This class manage the speific prompt for `re`
    """
    GENERAL = [
        ["""To compare `re.Match` object, `==` doesn't work. One should use `group()` method to obtain the string and then compare, e.g. `m1.group() == m2.group()`."""],
        ["""When no match is found, the output will be None. Make sure this situation is dealt with."""],
    ]

class PandasAssertHandler(PromptHandler):
    """
    This class manage the speific prompt for `pandas`
    """
    UNALIASED_PACKAGE_NAME = "pandas"
    GENERAL = [
        ["""Using `==` to check equality of complex objects is ambiguous. For example, Series (pandas), Index (pandas), numpy array, and DataFrame. One should to use a.empty, a.bool(), a.item(), a.any() or a.all(), etc."""],
        [
            """In addition, you could check equality of two tables (DataFrame) by `equals` (df1.equals(df2)). However, not all objects in `pandas` has this function, e.g. DataFrameGroupBy, primitive Python types like tuple.""",
            """For `equals` to be True, the table must have rows in the same order. However, we also allow table that has the same content (in different order) to be equivalent.""",
            """You could check equivalence of two tables by directly comparing the value row-by-row or column-by-column.""",
        ],
    ]
    
    SPECIFIC = defaultdict(lambda: None, {
        "pandas.DataFrame.groupby": [
            ["When comparing two DataFrameGroupBy objects, you need to make sure you compare the main attributes (e.g. obj.groups).",
             "'DataFrameGroupBy' object has no attribute 'equals' for comparison.",
             "`groups` attribute is a dictionary of <key, `Index` object>; You MUST loop through each entry and compare one-by-one.",
             ],]
    })

class ItertoolsAnswerHandler(PromptHandler):
    """
    This class manage the speific prompt for `itertools`
    """
    UNALIASED_PACKAGE_NAME = "itertools"
    GENERAL = [
    ]
    SPECIFIC = defaultdict(lambda: None, {
        "itertools.groupby": [["If you make call to `old_groupby`, don't attempt to unwrap the function output (e.g. by list())."]]
    })

class ItertoolsAssertHandler(PromptHandler):
    """
    This class manage the speific prompt for `itertools`
    """
    UNALIASED_PACKAGE_NAME = "itertools"
    GENERAL = [
        ["""The output of `itertools` functions (e.g. `itertools._grouper` object) is not directly checkable by `==`. To compare the output of itertools, the most direct way is to unwrap the output into something directly checkable (e.g. list, tuple, dict)."""],
    ]
    SPECIFIC = defaultdict(lambda: None, {
        # "itertools.groupby": [["The output of `itertools.groupby` are consecutive keys and groups (`itertools._grouper` objects) from the iterable. You need to make sure the unwrapping is handle the most nested `itertools._grouper` object"]]
    })

class NumpyAssertHandler(PromptHandler):
    """
    This class manage the speific prompt for `pandas`
    """
    UNALIASED_PACKAGE_NAME = "numpy"
    GENERAL = [
        ["""Using `==` to check equality of numpy objects (e.g. numpy.array) is ambiguous. For example, you should use `numpy.equal` or `numpy.allclose` to check if two numpy array equal."""],
    ]

class TorchAssertHandler(PromptHandler):
    """
    This class manage the speific prompt for `pandas`
    """
    UNALIASED_PACKAGE_NAME = "torch"
    GENERAL = [
        ["""Using `==` to check equality of Tensor objects (e.g. numpy.array) is ambiguous. For example, you should use `torch.equal` or `torch.allclose` to check if two Tensor objects equal.""",
         """allclose(): argument 'input' (position 1) must be Tensor, not list.""",
         ],
    ]
    
class MatplotlibAssertHandler(PromptHandler):
    """
    This class manage the speific prompt for `pandas`
    """
    UNALIASED_PACKAGE_NAME = "matplotlib"
    GENERAL = [
        ["""IMPORTANT: The equality matplotlib objects (e.g. `Figure`, `PathCollection`, `BarContainer`) could be checked by simply using `==`. However, such check may be too strict. Alternative is to check if the object's *main* attributes affected by the update is behaving as expected. CHOOSE WISELY!""",],
        ["""Using `==` to check equality of numpy objects (e.g. numpy.array) is ambiguous. For example, you should use `numpy.equal` or `numpy.allclose` to check if two numpy array equal."""],
    ]
    SPECIFIC = defaultdict(lambda: None, {
        "matplotlib.pyplot.scatter": [[
            """The original behavior of `matplotlib.pyplot.scatter` will return a PathCollection object.""",
            """Focus on checking if the object's *main* attributes affected by the update is behaving as expected.""",
        ],],
        "matplotlib.pyplot.hist": [[
            """The original behavior of `matplotlib.pyplot.hist` will return three objects as a tuple --- (numpy.array, numpy.array, BarContainer). Check if each of them have expected values.""",
            """Focus on checking if the matplotlib object's *main* attributes affected by the update is behaving as expected.""",
        ],],
        "matplotlib.pyplot.plot": [[
            """The original behavior of `matplotlib.pyplot.plot` will return a list of `matplotlib.lines.Line2D` object(s).""",
            """Focus on checking if the object's *main* attributes affected by the update is behaving as expected.""",
        ],],
        "matplotlib.pyplot.bar": [[
            """The original behavior of `matplotlib.pyplot.bar` will return a `BarContainer` object.""",
            """Focus on checking if the object's *main* attributes affected by the update is behaving as expected.""",
        ],],
        "matplotlib.pyplot.figure": [[
            """The original behavior of `matplotlib.pyplot.bar` will return a `Figure` object.""",
            """Focus on checking if the object's *main* attributes affected by the update is behaving as expected (e.g. figsize, etc.)""",
        ],],
        "matplotlib.pyplot.yticks": [[
            """The original behavior of `matplotlib.pyplot.yticks` will return (locs, labels) ---  a list of yticks locations as numpy.array, and a list of ylabel `Text` objects.""",
            """For example, you should use `numpy.equal` or `numpy.allclose` to check if two numpy array equal.""",
            """Focus on checking if the object's *main* attributes affected by the update is behaving as expected (e.g. `Text` locaiton  --- `Text.x` and `Text.y`, `Text` content `Text.text` etc.)""",
        ],],
        "matplotlib.pyplot.ylabel": [[
            """The original behavior of `matplotlib.pyplot.ylabel` will return a `Text` object.""",
            """Focus on checking if the object's *main* attributes affected by the update is behaving as expected (e.g. `Text` locaiton  --- `Text.x` and `Text.y`, `Text` content `Text.text` etc.)""",
        ],],
         "matplotlib.pyplot.title": [[
            """The original behavior of `matplotlib.pyplot.title` will return a `Text` object.""",
            """Focus on checking if the object's *main* attributes affected by the update is behaving as expected (e.g. `Text` locaiton  --- `Text.x` and `Text.y`, `Text` content `Text.text` etc.)""",
        ],],
        
    })

class ScipyAssertHandler(PromptHandler):
    UNALIASED_PACKAGE_NAME = "scipy"
    GENERAL = [
        ["""The output of scipy functions may be complex. It's not checkable by `==`. For example, you should use `numpy.equal` or `numpy.allclose` to check if two numpy array equal.""",
         """The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()."""],
    ]
    
class RequestsInputHandler(PromptHandler):
    """
    This class manage the speific prompt for `itertools`
    """
    UNALIASED_PACKAGE_NAME = "requests"
    GENERAL = [
        ["""To test the API, you *MUST* use `http://www.testingmcafeesites.com` as url input. This is a url dedicated for testing purpose.""",],
        ["""Dont load files from the disk.""",],
    ]
    SPECIFIC = defaultdict(lambda: None, {
        
    })


class RequestsAnswerHandler(PromptHandler):
    """
    This class manage the speific prompt for `itertools`
    """
    UNALIASED_PACKAGE_NAME = "requests"
    GENERAL = [
        ["""If an argument name is same as a function name (like json) then make sure to use tricks like import json as js and then use js.loads(data) instead of json.loads(data) to avoid confusion and overwriting. This was just an example generalise this to make sure this issue does not persist.""",],
        ["""Dont load files from the disk.""",],
    ]
    SPECIFIC = defaultdict(lambda: None, {
        
    })
    
class RequestsAssertHandler(PromptHandler):
    """
    This class manage the speific prompt for `itertools`
    """
    head_and_delete = [[
          """You should check the equality of the response object's `status_code`, `encoding`, `url` and `headers` attributes not any other thing.
    for the header attribute make sure NOT to check the 'Age', 'Server','X-RateLimit-Used','X-RateLimit-Remaining','X-RateLimit-Reset','X-GitHub-Request-Id' fields as they are dynamic and can change with time, 
eg: 
    assert response1.status_code == response2.status_code
    headers1 = {{k: v for k, v in response1.headers.items() if k not in excluded_headers}}
    headers2 = {{k: v for k, v in response2.headers.items() if k not in excluded_headers}}
    assert headers1 == headers2,
    assert response1.url == response2.url
    assert response1.encoding == response2.encoding"""
      ],]
    UNALIASED_PACKAGE_NAME = "requests"
    # GENERAL = [
    #     [
    #     """If an argument name is same as a function name (like json) then make sure to use tricks like import json as js and then use js.loads(data) instead of json.loads(data) to avoid confusion and overwriting. This was just an example generalise this to make sure this issue does not persist.""",
    #     ],
    #     ["""Dont load files from the disk.""",]
    # ]
    SPECIFIC = defaultdict(lambda: None, {
        "requests.get": [
            ["""You should check the equality of the response object's `text`, `status_code`, and `headers` attributes."""],
        ],
        "requests.post": [
            [
            """You should load the results.text and expected_result.text into JSON objects and compare the feilds like args, data , files, form, json, origin and url. (DONT compare X-Amzn-Trace-Id)
    for example: import json as js
    result = js.loads(result.text)
    expected_result = js.loads(expected_result.text)
    assert result['url'] == expected_result['url']
    assert result['origin'] == expected_result['origin']
    assert result['args'] == expected_result['args']
    assert result['data'] == expected_result['data']"""
        ],],
      "requests.head": head_and_delete,
      "requests.delete": head_and_delete,
    })


class PromptAllocator:
    HANDLER_MAPPING: Dict[str, PromptHandler] = defaultdict(lambda: None, {})
    
    def __getitem__(self, api_path):
        assert isinstance(api_path, str), "index must be an API path"
        package_name = api_path.split(".")[0]
        prompt_class = self.HANDLER_MAPPING[package_name]
        if prompt_class is None:
            return 
        return prompt_class[api_path]

class InputPromptAllocator(PromptAllocator):
    HANDLER_MAPPING = defaultdict(
        lambda: None, 
        {
            "requests": RequestsInputHandler(), 
        }
    )

class AnswerPromptAllocator(PromptAllocator):
    HANDLER_MAPPING = defaultdict(
        lambda: None, 
        {
            "itertools": ItertoolsAnswerHandler(), 
            "requests": RequestsAnswerHandler(), 
        }
    )

class AssertPromptAllocator(PromptAllocator):
    HANDLER_MAPPING = defaultdict(
        lambda: None, 
        {
            "pandas": PandasAssertHandler(),
            "re": ReAssertHandler(),
            "itertools": ItertoolsAssertHandler(), 
            "scipy": ScipyAssertHandler(),
            "numpy": NumpyAssertHandler(),
            "torch": TorchAssertHandler(),
            "matplotlib": MatplotlibAssertHandler(),
            "requests": RequestsAssertHandler(),
        }
    )

PACKAGE2PROMPT_INPUT = InputPromptAllocator()

PACKAGE2PROMPT_ANSWER = AnswerPromptAllocator()

PACKAGE2PROMPT_ASSERT = AssertPromptAllocator()
# print()
# PACKAGE2PROMPT_ANSWER 
# PACKAGE2PROMPT_SKELTON = defaultdict(
#     lambda: None, 
#     {
#         # "re": construct_instruction_series(re_assert_instruct),
#     }
# )
