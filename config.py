

# A "dataset" in language modelling is just a list of text documents.
# Here each document is a single person's first name (e.g. "emma", "olivia").
# The model will learn the statistical patterns of names and generate new ones.
#
# Why names? They're short (avg ~5 chars), have clear structure (consonant-vowel
# patterns, cultural clusters), and 32K of them fit in memory trivially.
# It's a hello-world dataset for character-level language models.
DEFAULT_INPUTS_FILE = "input.txt"

INPUTS_FOLDER = "datasets"
DATA_FOLDER = "data"

RANDOM_SEED = 42

# Hyperparameters — chosen to be small enough for CPU training in minutes.
# Number of transformer layers (depth). GPT-2 small has 12.
NUM_TRANSFORMER_LAYERS = 1
# Embedding dimension (width). GPT-2 small has 768.
NUM_EMBEDDING_DIMENSIONS = 16
# Maximum context length. The longest name is 15 chars + BOS = 16.
MAX_CONTENT_LENGTH = 16
# Number of attention heads. Must divide NUM_EMBEDDING_DIMENSIONS evenly.
NUM_ATTENTION_HEADS = 4

# Initial learning rate. Controls step size.
INITIAL_LEARNING_RATE = 0.01
# Decay rate for first moment. (Typical: 0.9)
# Slightly lower than usual — more responsive to recent grads.
BETA_1 = 0.85
# Decay rate for second moment. (Typical: 0.999)
# Slightly lower — faster adaptation of the variance estimate.
BETA_2 = 0.99
# Small constant to prevent division by zero in the update.
EPS_ADAM = 1e-8

# Default total training steps. With 1 layer and 16 dims, 1000 is enough to see meaningful learning.
# Production models train for billions.
DEFAULT_NUM_TRAINING_STEPS = 1000

# Temperature controls the "creativity" of generation.
# Applied by dividing logits by temperature before softmax.
#   temperature → 0: argmax (always pick the most likely token) — deterministic, safe
#   temperature = 1: sample from the true learned distribution — balanced
#   temperature > 1: flatten the distribution — more random, more creative/wrong
#
# Here temperature = 0.5: slightly below 1, biases toward more probable characters
# while still allowing diversity. Prevents generating the same name every time.
TEMPERATURE = 0.5

NUM_INFERENCE_RESULTS = 20

# If true, will store additional metadata information. Requires more memory.
VERBOSE_METADATA = True

# If you want to tweak the hyperparameters, do it inside config_override.py
try:
    from config_override import *
except:
    pass

# Maximum recursion depth for the autograd backward pass (build_topo).
# Graph depth scales with layers × dims × ~8 ops/layer (Q,K,V,proj,MLP*2,norms).
# 2x headroom factor guards against overrides and edge cases.
MAX_RECURSION_DEPTH = NUM_TRANSFORMER_LAYERS * NUM_EMBEDDING_DIMENSIONS * 8 * 2
