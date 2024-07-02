# Dataset Card for CodeUpdateArena

## Table of Contents
- [CodeUpdateArena](#codeupdatearena)
  - [Table of Contents](#table-of-contents)
  - [Dataset Description](#dataset-description)
    - [Dataset Summary](#dataset-summary)
    - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
    - [Languages](#languages)
  - [Dataset Structure](#dataset-structure)
    - [Data Instances](#data-instances)
    - [Data Fields](#data-fields)
    - [Data Splits](#data-splits)
  - [Dataset Creation](#dataset-creation)
    - [Curation Rationale](#curation-rationale)
    - [Source Data](#source-data)
      - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
      - [Who are the source language producers?](#who-are-the-source-language-producers)
    - [Annotations](#annotations)
      - [Annotation process](#annotation-process)
      - [Who are the annotators?](#who-are-the-annotators)
    - [Personal and Sensitive Information](#personal-and-sensitive-information)
  - [Considerations for Using the Data](#considerations-for-using-the-data)
    - [Social Impact of Dataset](#social-impact-of-dataset)
    - [Discussion of Biases](#discussion-of-biases)
    - [Other Known Limitations](#other-known-limitations)
  - [Additional Information](#additional-information)
    - [Dataset Curators](#dataset-curators)
    - [Licensing Information](#licensing-information)
    - [Citation Information](#citation-information)
    - [Contributions](#contributions)

## Dataset Description

- **Repository:** [GitHub Repository](https://github.com/leo-liuzy/CodeUpdateArena.git)
- **Paper:** [CodeUpdateArena:
Benchmarking Knowledge Editing on API Updates]()

### Dataset Summary

The CodeUpdateArena dataset, a benchmark for knowledge editing in the code domain. An instance in our benchmark consists of a synthetic API function update paired with a program synthesis example that uses the updated functionality.

### Supported Tasks and Leaderboards

### Languages
The programming problems are written in Python and contain English natural text in comments and docstrings.

## Dataset Structure

### Data Instances

An example of a dataset instance:

```
{'update': 
    {
        'description': "Renaming 'select_dtypes' to ...",
        'rationale': "The new function name 'filter_by_dtypes' better communicates the purpose...",
        'docstring': "The functionality remains the same as the original 'select_dtypes' function....",
        'signature': 'pandas.DataFrame.filter_by_dtypes(self, include=None, exclude=None) -> Self',
        'imports': "import numpy\nimport pandas\n...\nold_select_dtypes = pandas.DataFrame.select_dtypes\nsetattr(pandas.DataFrame, 'old_select_dtypes', old_select_dtypes)",
        'implementation': 'def filter_by_dtypes(self, include=None, exclude=None):\n...\nreturn self.old_select_dtypes(include=include, exclude=exclude)\n',
        'unit_tests': 'def test_filter_type_int64():\n    ....',
        'update_type': 'modify-function-name',
        'function_path': 'pandas.DataFrame.select_dtypes',
        'package': 'pandas',
        'update_id': '[pandas.DataFrame.select_dtypes]:[modify-function-name]:[update-0]'
    },
    'update_id': '[pandas.DataFrame.select_dtypes]:[modify-function-name]:[update-0]',
    'scenario': 'You are a data scientist at a tech company.....',
    'problem': 'Write a Python function that given a pandas DataFrame, a....',
    'solution_signature': 'def filter_dataframe_by_dtype(dataframe, include, exclude, n_cols)',
    'unit_tests': 'def test_filter_dataframe_by_dtype_no_exclude():\n    # Creating a DataFrame for testing\n    ...',
    'imports': "import numpy\nimport pandas\n...",
    'prog_syn_id': '[pandas.DataFrame.select_dtypes]:[modify-function-name]:[update-0]:[prog_syn-3]'
}
```

### Data Fields
`update` (dictionary): content of the specific Code API update (detailed below)

* `description`: The description of the update.
* `rationale`: The rationale of introducing the update.
* `docstring`: The docstring detailing the update.
* `signature`: The new signature of the update.
* `imports`: The expected imports to run the update.
* `implementation`: The implementation of the updated function. Imports separated by `\n`.
* `unit_tests`: The unit tests to verify the correctness of the implementation of the updated function. Unit tests separated by `\n\n`.
* `update_type`: The update type that the update belongs to.
* `function_path`: The full api path of the function (e.g. numpy.argsort).
* `package`: The Python package the function belongs to.
* `update_id`: The unique identifier for the specific update.



`update_id`: The unique identifier for the specific update, same as `update_id` in `update` dictionary. This is intended for clusterring program synthesis examples of the same updates.

`scenario`: the scenario that the program synthesis example (one of the examples per update) is situated in.

`problem`: The problem that the program synthesis example (one of the examples per update) is trying to tackle.

`solution_signature`: The signature of solution requried by problem statement.

`unit_tests`: The unit tests to verify the correctness of a predicated solution. Unit tests separated by `\n\n`.

`imports`: The imports to run the reference solution of program synthesis. Imports separated by `\n`.

`ref_solution`: The reference solution of the program synthesis example.

`prog_syn_id`: The unique identifier of the program synthesis example.

### Data Splits

The dataset consists of 670 samples.

## Dataset Creation


### Curation Rationale

Current code generation model are trained on past code corpus. However, code API constantly evolves and adherence to older APIs can cause failures. To be maximally useful, LLMs for code generation need to stay in sync with API updates, even those that occur after they are pre-trained. However, a benchmark to test API update is missing in the current community. To assist research in this direction, we propose the benchmark --- CodeUpdateArena.

### Source Data

The dataset was synthetically generated by a new generation pipeline (powered by `GPT-4-0613`) proposed by authors.

#### Initial Data Collection and Normalization

[More Information Needed]

#### Who are the source language producers?

[More Information Needed]

### Annotations

[More Information Needed]

#### Annotation process

[More Information Needed]

#### Who are the annotators?

[More Information Needed]

### Personal and Sensitive Information

None.

## Considerations for Using the Data
Make sure you execute generated Python code in a safe environment when evauating against this dataset as generated code could be harmful.

### Social Impact of Dataset
With this dataset, code generation models can be better evaluated for incoporating new code API update to problem solving.

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators
Zeyu Leo Liu, Shrey Pandit, Xi Ye, Eunsol Choi, Greg Durrett

### Licensing Information

MIT License

### Citation Information
```

```

### Contributions
We thank helpful discussion with Fangcong Yin, Manya Wadhwa, and members in TAUR and EUNSOL Lab