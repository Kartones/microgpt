
import json
from typing import Any

from config import (
    NUM_EMBEDDING_DIMENSIONS, NUM_TRANSFORMER_LAYERS, MAX_CONTENT_LENGTH, NUM_ATTENTION_HEADS
)
from microgpt import MicroGPT
from value import Value, ValueJSONEncoder

DATA_FILENAME = "model_data.json"
METADATA_FILENAME = "model_metadata.json"

KEY_NUM_LAYERS = "n_layer"
KEY_BOS = "bos"
KEY_BLOCK_SIZE = "block_size"
KEY_VOCAB_SIZE = "vocab_size"
KEY_UCHARS = "uchars"
KEY_STATE_DICT = "state_dict"
KEY_NUM_HEADS = "n_head"
KEY_NUM_EMBEDDING_DIMENSIONS = "n_embd"

KEY_NUM_TRAINING_STEPS = "num_training_steps"

class ModelData:

    def load(self, training_steps: int) -> tuple[dict[str, Any], dict[str, Any]]:
        with open(DATA_FILENAME, "r") as file_handle:
            data = json.load(file_handle)
        data["state_dict"] = {
            key: [[Value(x) for x in row] for row in matrix]
            for key, matrix in data["state_dict"].items()
        }

        with open(METADATA_FILENAME, "r") as file_handle:
            metadata = json.load(file_handle)

        try:
            self._validate(data, metadata, training_steps)
        except ValueError as e:
            print(f"Could not load model data: {e}")
            exit(1)

        return data, metadata

    @staticmethod
    def save(model: MicroGPT) -> None:
        with open(DATA_FILENAME, "w") as file_handle:
            file_handle.write(json.dumps({
                KEY_NUM_LAYERS: model.n_layer,
                KEY_BOS: model.BOS,
                KEY_BLOCK_SIZE: model.block_size,
                KEY_VOCAB_SIZE: model.vocab_size,
                KEY_UCHARS: model.uchars,
                KEY_STATE_DICT: model.state_dict,
                KEY_NUM_HEADS: model.n_head,
                KEY_NUM_EMBEDDING_DIMENSIONS: model.n_embd,
            }, cls=ValueJSONEncoder))

        with open(METADATA_FILENAME, "w") as file_handle:
            file_handle.write(json.dumps({
                KEY_NUM_TRAINING_STEPS: model.num_training_steps,
            }))

    @staticmethod
    def _validate(data: dict[str, Any], metadata: dict[str, Any], training_steps: int) -> None:
        required_data_keys = {
            KEY_NUM_LAYERS, KEY_BOS, KEY_BLOCK_SIZE, KEY_VOCAB_SIZE, KEY_UCHARS, KEY_STATE_DICT, KEY_NUM_HEADS, KEY_NUM_EMBEDDING_DIMENSIONS
        }
        missing_keys = required_data_keys - data.keys()
        if missing_keys:
            raise ValueError(f"Missing keys in model data: {missing_keys}")

        comparisons = {
            KEY_NUM_LAYERS: NUM_TRANSFORMER_LAYERS,
            # TODO: BOS
            KEY_BLOCK_SIZE: MAX_CONTENT_LENGTH,
            # TODO: VOCAB_SIZE
            KEY_NUM_HEADS: NUM_ATTENTION_HEADS,
            KEY_NUM_EMBEDDING_DIMENSIONS: NUM_EMBEDDING_DIMENSIONS,
        }
        for key, expected_value in comparisons.items():
            if data.get(key) != expected_value:
                raise ValueError(f"{key} mismatch: expected {expected_value}, got {data.get(key)}")

        required_metadata_keys = {KEY_NUM_TRAINING_STEPS}
        missing_keys = required_metadata_keys - metadata.keys()
        if missing_keys:
            raise ValueError(f"Missing keys in model metadata: {missing_keys}")

        if metadata.get(KEY_NUM_TRAINING_STEPS) != training_steps:
            raise ValueError(
                f"{KEY_NUM_TRAINING_STEPS} mismatch: expected {training_steps}, got {metadata.get(KEY_NUM_TRAINING_STEPS)}"
            )
