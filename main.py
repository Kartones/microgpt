import sys

from config import DEFAULT_NUM_TRAINING_STEPS, NUM_INFERENCE_RESULTS, TEMPERATURE
from microgpt import MicroGPT
from model_data import ModelData

def process_args() -> tuple[bool, int, float, int]:
    args = sys.argv[1:]

    load_data = False
    training_steps = DEFAULT_NUM_TRAINING_STEPS
    temperature = TEMPERATURE
    num_inference_results = NUM_INFERENCE_RESULTS

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

    return load_data, training_steps, temperature, num_inference_results

if __name__ == '__main__':
    model_data = ModelData()

    load_data, training_steps, temperature, num_inference_results = process_args()

    if load_data:
        print("--- loading model data ---")
        data, metadata = model_data.load(training_steps)
        microgpt = MicroGPT(training_steps, data, metadata)
    else:
        microgpt = MicroGPT(training_steps)

    microgpt.run(temperature, num_inference_results)
    if not load_data:
        print("--- saving model data ---")
        model_data.save(microgpt)
