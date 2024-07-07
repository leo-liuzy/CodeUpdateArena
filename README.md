# `CodeUpdateArena`: Benchmarking Knowledge Editing on API Updates

Authors: Zeyu Leo Liu, Shrey Pandit, Xi Ye, Eunsol Choi, Greg Durrett

Please check out our work [TBD] 📃

![Dataset overview](./images/dataset-overview.png)

## Description

The goal of our benchmark is to update an LLM about code API update and be able to solve "related" program synthesis example *without providing documentation of the update at inference time*.

`CodeUpdateArena` benchmark contains **fictitious** and **executable** updates to 54 functions from 7 diverse Python packages. 

An instance in our benchmark consists of a synthetic API function update paired with a program synthesis example that is biased to use the updated functionality. Each fictitious update is paired with at least 3 (executable) program synthesis examples. 


### Benchmark Access

```python
from datasets import load_dataset

ds = load_dataset("leo-liuzy/CodeUpdateArena")
```

The *goal* of our benchmark is to update an LLM to be able to solve this program synthesis example *without providing documentation of the update at inference time*.Our Benchmark is available on HuggingFace 🤗 More benchmark details can be found [here](https://huggingface.co/datasets/leo-liuzy/CodeUpdateArena).


### Dataset statistics
<p align="center">
  <img src="./images/demographics-by-package.png" alt="Demographics by package" width="50%" />
</p>


<img src="./images/demographics-by-update.png" alt="Demographics by update type"/>

<br/>

<img src="./images/dataset-size.png" alt="Size"/>

<br/>

<img src="./images/dataset-token-stats.png" alt="Token statistics"/>

<br/>




## Synthetic Data Generation

<img src="./images/generation-pipeline.png" alt="Generation pipeline"/>

**Check out the details in our paper!**

We provide bash script to run experiment in [src/data](https://github.com/leo-liuzy/CodeUpdateArena/tree/main/src/data) directory. The core scripts are `manager_update.py` and `manager_prog_syn.py`, who are pipelines to generate update and program synthesis respectively. Both script follows similar generation procedure but uses different sets of prompts.


We also include the core code to automatically de-duplicate generated program synthesis examples. See `auto-dedup.py` in `scripts` directory.

## Reproducing knowledge editing baselines

We provide bash script to run experiment in [script](https://github.com/leo-liuzy/CodeUpdateArena/tree/main/scripts) directory. We **highly recommend** running code to [predict solution] and [execute solution] separately. We left more comment for how to use the script.

## Citation

If you found our work useful, please consider citing our work.
```
TBD
```