import os
import sys
from dataclasses import dataclass

from config import (
    DEFAULT_NUM_TRAINING_STEPS, NUM_INFERENCE_RESULTS, TEMPERATURE, DEFAULT_INPUTS_FILE, INPUTS_FOLDER,
    NUM_TRANSFORMER_LAYERS, NUM_EMBEDDING_DIMENSIONS, MAX_CONTENT_LENGTH, NUM_ATTENTION_HEADS
)
from microgpt import MicroGPT
from model_data import (
    ModelData,
    KEY_NUM_LAYERS, KEY_BLOCK_SIZE, KEY_NUM_HEADS, KEY_NUM_EMBEDDING_DIMENSIONS,
    KEY_NUM_TRAINING_STEPS, KEY_VOCAB_SIZE,
)

@dataclass
class Args:
    load_data: bool
    training_steps: int
    temperature: float
    num_inference_results: int
    inputs_file: str
    inference_input: bool
    fixed_random_seed: bool


def process_args() -> Args:
    args = sys.argv[1:]

    load_data = False
    training_steps = DEFAULT_NUM_TRAINING_STEPS
    temperature = TEMPERATURE
    num_inference_results = NUM_INFERENCE_RESULTS
    inputs_file = DEFAULT_INPUTS_FILE
    inference_input = False
    fixed_random_seed = True

    if "--load" in args:
        load_data = True

    for arg in args:
        if arg.startswith("--steps="):
            try:
                training_steps = int(arg.split("=")[1])
                if training_steps <= 0:
                    raise ValueError("Training steps must be > 0")
            except ValueError:
                print(f"Invalid steps value: {arg}. Using default {DEFAULT_NUM_TRAINING_STEPS}")
                training_steps = DEFAULT_NUM_TRAINING_STEPS
        elif arg.startswith("--temperature="):
            try:
                temperature = float(arg.split("=")[1])
                if temperature <= 0:
                    raise ValueError("Temperature must be > 0")
            except ValueError:
                print(f"Invalid temperature value: {arg}. Using default {TEMPERATURE}")
                temperature = TEMPERATURE
        elif arg.startswith("--num-inference-results="):
            try:
                num_inference_results = int(arg.split("=")[1])
                if num_inference_results <= 0:
                    raise ValueError("Number of inference results must be > 0")
            except ValueError:
                print(f"Invalid num-inference-results value: {arg}. Using default {NUM_INFERENCE_RESULTS}")
                num_inference_results = NUM_INFERENCE_RESULTS
        elif arg.startswith("--input="):
            inputs_file = arg.split("=")[1] or DEFAULT_INPUTS_FILE
        elif arg == "--inference-input":
            inference_input = True
        elif arg == "--random-seed":
            fixed_random_seed = False

    if not inputs_file.startswith(os.path.join(INPUTS_FOLDER, "")):
        inputs_file = os.path.join(INPUTS_FOLDER, inputs_file)

    return Args(
        load_data, training_steps, temperature, num_inference_results, inputs_file, inference_input, fixed_random_seed
    )


def print_config(
    training_steps: int,
    temperature: float,
    inputs_file: str,
    data: dict | None = None,
    metadata: dict | None = None,
) -> None:
    n_layers     = data[KEY_NUM_LAYERS]              if data     else NUM_TRANSFORMER_LAYERS
    n_embd       = data[KEY_NUM_EMBEDDING_DIMENSIONS] if data     else NUM_EMBEDDING_DIMENSIONS
    block_size   = data[KEY_BLOCK_SIZE]              if data     else MAX_CONTENT_LENGTH
    n_heads      = data[KEY_NUM_HEADS]               if data     else NUM_ATTENTION_HEADS
    steps        = metadata[KEY_NUM_TRAINING_STEPS]  if metadata else training_steps
    vocab_size   = metadata[KEY_VOCAB_SIZE]          if metadata else None

    print(f"  input file          : {inputs_file}")
    print(f"  training steps      : {steps}")
    print(f"  temperature         : {temperature}")
    print(f"  transformer layers  : {n_layers}")
    print(f"  embedding dimensions: {n_embd}")
    print(f"  context length      : {block_size}")
    print(f"  attention heads     : {n_heads}")
    if vocab_size is not None:
        print(f"  vocab size          : {vocab_size}")


if __name__ == '__main__':
    args = process_args()

    model_data = ModelData(args.inputs_file)

    if args.load_data:
        print("--- loading model data ---")
        data, metadata = model_data.load(args.training_steps)
        microgpt = MicroGPT(
            num_training_steps = args.training_steps, inputs_file = args.inputs_file, data = data, metadata = metadata,
            fixed_random_seed = args.fixed_random_seed
        )
        print_config(args.training_steps, args.temperature, args.inputs_file, data, metadata)
        microgpt.infer(args.temperature, args.num_inference_results, args.inference_input, args.fixed_random_seed)
    else:
        print("--- preparing model data ---")
        microgpt = MicroGPT(
            num_training_steps = args.training_steps, inputs_file = args.inputs_file,
            fixed_random_seed = args.fixed_random_seed
        )
        print_config(args.training_steps, args.temperature, args.inputs_file)
        microgpt.train()
        print("--- saving model data ---")
        model_data.save(microgpt)
        microgpt.infer(args.temperature, args.num_inference_results, args.inference_input, args.fixed_random_seed)
