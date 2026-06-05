import os
import sys

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

def process_args() -> tuple[bool, int, float, int, str]:
    args = sys.argv[1:]

    load_data = False
    training_steps = DEFAULT_NUM_TRAINING_STEPS
    temperature = TEMPERATURE
    num_inference_results = NUM_INFERENCE_RESULTS
    inputs_file = DEFAULT_INPUTS_FILE

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

    if not inputs_file.startswith(os.path.join(INPUTS_FOLDER, "")):
        inputs_file = os.path.join(INPUTS_FOLDER, inputs_file)

    print(inputs_file)

    return load_data, training_steps, temperature, num_inference_results, inputs_file


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
    load_data, training_steps, temperature, num_inference_results, inputs_file = process_args()

    model_data = ModelData(inputs_file)

    if load_data:
        print("--- loading model data ---")
        data, metadata = model_data.load(training_steps)
        microgpt = MicroGPT(training_steps, inputs_file, data, metadata)
        print_config(training_steps, temperature, inputs_file, data, metadata)
        microgpt.infer(temperature, num_inference_results)
    else:
        print("--- preparing model data ---")
        microgpt = MicroGPT(training_steps, inputs_file)
        print_config(training_steps, temperature, inputs_file)
        microgpt.train()
        print("--- saving model data ---")
        model_data.save(microgpt)
        microgpt.infer(temperature, num_inference_results)
