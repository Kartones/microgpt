from docs_reader import read_docs
from model_data import ModelData, KEY_SEEN_TRAINING_DOCS


RED = "\033[31m"
RESET = "\033[0m"


class InferenceDecorator:

    def __init__(self, inputs_file: str) -> None:
       self.inputs_file = inputs_file
       self.docs = self._load_docs()
       self.seen_training_docs = self._load_metadata()

    def _load_docs(self) -> list[str]:
        return read_docs(self.inputs_file, shuffle=False)

    def _load_metadata(self) -> list[str]:
        model_data = ModelData(self.inputs_file)
        metadata = model_data.load_metadata()
        return metadata[KEY_SEEN_TRAINING_DOCS]

    @staticmethod
    def _colorize_dark_gray(text : str) -> str:
        return f"\033[90m{text}{RESET}"

    @staticmethod
    def _colorize_green(text : str) -> str:
        return f"\033[32m{text}{RESET}"

    @staticmethod
    def _colorize_yellow(text : str) -> str:
        return f"\033[33m{text}{RESET}"

    def decorate_result(self, result : str, index : int) -> str:
      if result not in self.docs:
          # leave standard gray for hallucinated results
          pass
      elif result in self.seen_training_docs:
          result = self._colorize_yellow(result)
      else:
          result = self._colorize_green(result)

      prefix = self._colorize_dark_gray(f"sample {index+1:2d}: ")

      return f"{prefix}{result}"
