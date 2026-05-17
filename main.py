import sys

from microgpt import MicroGPT
from model_data import ModelData

if __name__ == '__main__':
    model_data = ModelData()

    load_data = len(sys.argv) == 2 and sys.argv[1] == "--load"

    if load_data:
        print("--- loading model data ---")
        microgpt = MicroGPT(model_data.load("model_data.json"))
    else:
        microgpt = MicroGPT()

    microgpt.run()

    if not load_data:
        print("--- saving model data ---")
        model_data.save(microgpt, "model_data.json")
