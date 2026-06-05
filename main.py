import os
import sys

from config import DEFAULT_NUM_TRAINING_STEPS, NUM_INFERENCE_RESULTS, TEMPERATURE, DEFAULT_INPUTS_FILE, INPUTS_FOLDER
from microgpt import MicroGPT
from model_data import ModelData

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


if __name__ == '__main__':
    load_data, training_steps, temperature, num_inference_results, inputs_file = process_args()

    model_data = ModelData(inputs_file)

    if load_data:
        print("--- loading model data ---")
        data, metadata = model_data.load(training_steps)
        microgpt = MicroGPT(training_steps, inputs_file, data, metadata)
        microgpt.infer(temperature, num_inference_results)
    else:
        microgpt = MicroGPT(training_steps, inputs_file)
        microgpt.train()
        print("--- saving model data ---")
        model_data.save(microgpt)
        microgpt.infer(temperature, num_inference_results)
