import random

# Here each document is a single person's first name (e.g. "emma", "olivia").
# The model will learn the statistical patterns of names and generate new ones.
#
# Why names? They're short (avg ~5 chars), have clear structure (consonant-vowel
# patterns, cultural clusters), and 32K of them fit in memory trivially.
# It's a hello-world dataset for character-level language models.

def read_docs(inputs_file: str, shuffle: bool = True) -> list[str]:
  # Each non-empty line is one document. strip() removes the trailing newline.
  docs = [line.strip() for line in open(inputs_file) if line.strip()]

  if shuffle:
      # Shuffle the dataset so training steps see a diverse mix of names.
      # Without shuffling, the model would see all "A" names first, then "B" names,
      # which could bias early training. Shuffling breaks this correlation.
      # The seed above ensures the shuffle is the same every run.
      random.shuffle(docs)

  return docs
