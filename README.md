# MicroGPT - Commented Edition

## Introduction

[MicroGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), from [Andrej Karpathy](https://karpathy.ai/), but commented and refactored for learning.


- Code heavily commented (mostly based on [Andrej's own explanatory post](https://karpathy.github.io/2026/02/12/microgpt/)). 
- Input download logic removed. It is cool that it auto-downloads the dataset, but I wanted to decouple the model from only using a given names list. `input.txt` is this file (remember to rename it): [https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt](https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt)
- Code split into classes and modules, and with type hints. 


## TODOs

- if `self.inference_only`, disallow methods that might break due to missing data
- when loading data, compare against config and exit if values differ (e.g. heads, layers...)
- Define at `__init__` all the attributes used in methods, e.g. self.docs
- Code comments were added with AI assistance, they are under review at the moment.
- I might do further code splits and refactors, as my goal is learning and thus, readability.


## What's missing

Some ideas, mostly extracted from the Author's blog post. Note that this is **not** a roadmap/TODO list, just concepts that larger systems have.

TENSORS: Every operation here is on scalars. Real implementations use NumPy/PyTorch tensors — parallelized matrix ops that are 100-1000× faster.

GPU EXECUTION: PyTorch moves tensors to GPU with .cuda(), getting another 100-1000× speedup via thousands of parallel cores.

BATCHING: Training on one document at a time is wasteful. Real training processes B=512 or B=4096 documents simultaneously.

MIXED PRECISION: float32 parameters here; production uses bfloat16/float16 (half memory, faster ops) with float32 optimizer state.

FLASH ATTENTION: The naive attention here is O(T^2) in memory. FlashAttention (Dao et al. 2022) computes the same result in O(T) memory via tiling.

GRADIENT CHECKPOINTING: We keep the entire computation graph in memory. Checkpointing trades memory for recomputation to handle long sequences.

DISTRIBUTED TRAINING: Trillion-parameter models need thousands of GPUs and complex parallelism strategies (tensor, pipeline, data parallel).

TOKENIZER: Character-level is too granular for long text. Production uses BPE (50K+ vocabulary) to get shorter sequences with richer tokens.

WEIGHT TYING: wte and lm_head often share weights (Inan et al. 2016), halving output layer parameters with no quality loss.

ADVANCED POSITIONAL ENCODING: RoPE (rotary position embedding) or ALiBi generalize better to sequences longer than those seen during training.