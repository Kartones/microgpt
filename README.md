# MicroGPT - Commented Edition

## Introduction

[MicroGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), from [Andrej Karpathy](https://karpathy.ai/), but commented and refactored for learning.


- Code heavily commented (mostly based on [Andrej's own explanatory post](https://karpathy.github.io/2026/02/12/microgpt/)). 
- Input download logic removed. It is cool that it auto-downloads the dataset, but I wanted to decouple the model from only using a given names list. `input.txt` is this file (remember to rename it and place it under `datasets/`): [https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt](https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt)
- Code split into classes and modules, and with type hints.
- And quite a few other improvements (see )

## Instructions

Normal run - trains first, then runs inference:
```bash
python3 main.py
```

Inference only run (needs to have run with same values) - generates N random samples/results:
```bash
python3 main.py --load
```

Inference only run (needs to have run with same values) - asks for starting sequence and completes it:
```bash
python3 main.py --inference-input --load
```


### Configuration parameters

Note that you also need to specify some also in inference only mode. Check `config.py` for the default values.

- `--steps=<value>`: Number of training steps
- `--temperature=<value>`: Temperature. Lower less "creative" so less chance of hallucination, but also more probably to repeat
- `--input=<filename>`: Text filename inside `datasets/` containing the input dataset
- `--inference-input`: Ask for a starting sequence/sample, instead of generating random ones
- `--num-inference-results=<value>`: How many results to generate at inference, if generating random ones
- `--random-seed`: Do not use a fixed seed for random number generation

## Examples

Training against a 5-chars long, lowercased words dataset extracted from the first Dune book:

```bash
sample  1: sales
sample  2: tanal
sample  3: alone
sample  4: sarer
sample  5: weads
sample  6: stese
sample  7: stors
sample  8: chald
sample  9: sheal
sample 10: stare
sample 11: wings
sample 12: camel
sample 13: tares
sample 14: dares
sample 15: gaxed
sample 16: bound
sample 17: witha
sample 18: stard
sample 19: lroms
sample 20: trers
```

Of which:
- `sales` is hallucinated (not in the dataset), but valid
- `alone`, `camel`, `bound` are in the dataset and were not seen during training
- `stare`, `wings`, `dares` are in the dataset and were seen during training
- The rest are hallucinated


## What's missing

Some ideas, mostly extracted from the Author's blog post. Note that this is **not** a roadmap/TODO list, just concepts that larger systems have.

TENSORS: Every operation here is on scalars. Real implementations use NumPy/PyTorch tensors — parallelized matrix ops that are 100-1000× faster.

BATCHING: Training on one document at a time is wasteful. Real training processes B=512 or B=4096 documents simultaneously.

MIXED PRECISION: float32 parameters here; production uses bfloat16/float16 (half memory, faster ops) with float32 optimizer state.

FLASH ATTENTION: The naive attention here is O(T^2) in memory. FlashAttention (Dao et al. 2022) computes the same result in O(T) memory via tiling.

GRADIENT CHECKPOINTING: We keep the entire computation graph in memory. Checkpointing trades memory for recomputation to handle long sequences.

WEIGHT TYING: wte and lm_head often share weights (Inan et al. 2016), halving output layer parameters with no quality loss.

ADVANCED POSITIONAL ENCODING: RoPE (rotary position embedding) or ALiBi generalize better to sequences longer than those seen during training.