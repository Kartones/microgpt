import sys

from config import DEFAULT_NUM_TRAINING_STEPS
from microgpt import MicroGPT
from model_data import ModelData

def process_args() -> tuple[bool, int]:
    args = sys.argv[1:]

    load_data = False
    training_steps = DEFAULT_NUM_TRAINING_STEPS

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

    return load_data, training_steps

if __name__ == '__main__':
    model_data = ModelData()

    load_data, training_steps = process_args()

    if load_data:
        print("--- loading model data ---")
        microgpt = MicroGPT(training_steps, model_data.load("model_data.json"))
    else:
        microgpt = MicroGPT(training_steps)

    microgpt.run()

    if not load_data:
        print("--- saving model data ---")
        model_data.save(microgpt, "model_data.json")
