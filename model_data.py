
import json
from typing import Any

from microgpt import MicroGPT
from value import Value, ValueJSONEncoder


class ModelData:

    @staticmethod
    def load(path: str) -> dict[str, Any]:
        with open(path, "r") as file_handle:
            data = json.load(file_handle)
        data["state_dict"] = {
            key: [[Value(x) for x in row] for row in matrix]
            for key, matrix in data["state_dict"].items()
        }
        return data

    @staticmethod
    def save(model: MicroGPT, path: str):
        with open(path, "w") as file_handle:
            file_handle.write(json.dumps({
                "n_layer": model.n_layer,
                "bos": model.BOS,
                "block_size": model.block_size,
                "vocab_size": model.vocab_size,
                "uchars": model.uchars,
                "state_dict": model.state_dict,
                "n_head": model.n_head,
                "head_dim": model.head_dim
            }, cls=ValueJSONEncoder))
