"""
A heavily annotated version of microgpt.py by @karpathy.

Original source: https://github.com/karpathy/microgpt
Blog post (source of many annotations): https://karpathy.github.io/2026/02/12/microgpt/

Reading guide:
  1. Dataset & Tokenizer  — how raw text becomes integer token sequences
  2. Autograd (Value)     — how gradients are computed automatically
  3. Model parameters     — what the model "knows" and how it's stored
  4. Architecture (gpt)   — how tokens flow through the network
  5. Training loop        — how the model learns from data
  6. Inference            — how the trained model generates text

Original core thesis (Karpathy):
  "This file is the complete algorithm. Everything else is just efficiency."
  Every production system (GPT-4, LLaMA, etc.) does exactly what this file does —
  it just does it faster (GPUs, batching, fused kernels, mixed precision, etc.).
  Strip those away and this is what remains.
"""

import random
from typing import Any

from config import (
  BETA_1, EPS_ADAM, BETA_2, INITIAL_LEARNING_RATE, MAX_CONTENT_LENGTH, NUM_ATTENTION_HEADS, NUM_EMBEDDING_DIMENSIONS, NUM_INFERENCE_RESULTS,
  NUM_TRANSFORMER_LAYERS, RANDOM_SEED, TEMPERATURE, VERBOSE_METADATA
)
from docs_reader import read_docs
from inference_decorator import InferenceDecorator
from tokenizer import unique_chars, get_BOS_token_id, get_vocabulary_size
from value import Value


class MicroGPT:

    def __init__(
            self, num_training_steps : int, inputs_file : str, data : dict[str, Any] | None = None,
            metadata : dict[str, Any] | None = None, fixed_random_seed : bool = True) -> None:

        if fixed_random_seed:
            # Fix the random seed so every run is deterministic.
            # This matters for reproducibility:
            #  same seed → same weight initialization → same training trajectory → same final model.
            random.seed(RANDOM_SEED)

        self.num_training_steps = num_training_steps

        self.inputs_file = inputs_file

        self.seen_training_docs : list[str] = []

        self.inference_only = False

        if data:
            self.n_layer = data["n_layer"]
            self.block_size = data["block_size"]
            self.uchars = data["uchars"]
            self.vocab_size = get_vocabulary_size(self.uchars)
            self.BOS = get_BOS_token_id(self.uchars)
            self.state_dict = data["state_dict"]
            self.n_head = data["n_head"]
            self.n_embd = data["n_embd"]
            self.head_dim = self.get_head_dimension(self.n_embd, self.n_head)
            self.inference_only = True
        if metadata:
            self.num_training_steps = metadata["num_training_steps"]

    def _dataset(self) -> None:
        # A "dataset" in language modelling is just a list of text documents.

        self.docs = read_docs(self.inputs_file)
        print(f"num docs: {len(self.docs)}")

    def _tokenizer(self) -> None:
        # A tokenizer converts strings <-> sequences of integers ("tokens").
        # The model never sees raw characters — only token IDs.
        #
        # Design choice: CHARACTER-LEVEL tokenization.
        # The simplest possible scheme: each unique character gets one integer ID.
        # Alternative: subword tokenization (BPE, SentencePiece) used by real GPTs.
        #   - Subword: "hello" → [15496] (one token per word-piece)
        #   - Character: "hello" → [7, 4, 11, 11, 14]  (one token per character)
        #
        # For names, character-level is ideal: the vocabulary is tiny (26 letters),
        # sequences are short, and we want the model to learn letter-level patterns.

        if self.inference_only:
            raise ValueError("Tokenizer should not be called in inference-only mode")

        self.uchars = unique_chars(self.docs)

        self.BOS = get_BOS_token_id(self.uchars)

        self.vocab_size = get_vocabulary_size(self.uchars)
        print(f"vocab size: {self.vocab_size}")

        # Tokenization examples:
        #   encode("emma") → [BOS, 4, 12, 12, 0, BOS]   (wrapped with BOS on both ends)
        #   decode([4, 12, 12, 0]) → "emma"
        # Note: there's no explicit encode/decode function here — it's done inline
        # in the training loop with `uchars.index(ch)` and `uchars[token_id]`.

    def _model_parameters(self) -> None:
        # The model's "knowledge" lives entirely in its parameters — floating point
        # numbers that are learned during training. Everything else (architecture,
        # optimizer, loss) is fixed; only the parameters change.

        if self.inference_only:
            raise ValueError("Model parameters should not be initialized in inference-only mode")

        # Hyperparameters — chosen to be small enough for CPU training in minutes.
        self.n_layer = NUM_TRANSFORMER_LAYERS
        self.n_embd = NUM_EMBEDDING_DIMENSIONS
        self.block_size = MAX_CONTENT_LENGTH
        self.n_head = NUM_ATTENTION_HEADS
        self.head_dim = self.get_head_dimension(self.n_embd, self.n_head)

        # Helper to create a 2D matrix of random Value nodes.
        # nout × nin matrix, initialized from N(0, std).
        # std=0.08 is a reasonable default — small enough to avoid saturating activations at initialization,
        # large enough to break symmetry between neurons.
        # Without random init, all neurons would learn the same features (symmetry problem).
        # Symmetry problem: when N neurons continue learning exactly the same representation, becoming redundant,
        # and thus the layer being less effective.
        def matrix(nout: int, nin: int, std: float = 0.08) -> list[list[Value]]:
            return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

        # state_dict: a plain Python dict mapping parameter names to 2D matrices.
        # This mirrors the convention from PyTorch's model.state_dict().
        # Shapes are chosen to match GPT-2 architecture (minus biases and layernorm params).
        self.state_dict = {
            # Token Embedding Table: vocab_size × n_embd
            # Maps each token ID to a learned vector. wte[token_id] is the embedding.
            'wte': matrix(self.vocab_size, self.n_embd),

            # Positional Embedding Table: block_size × n_embd
            # Maps each position index to a learned vector. wpe[pos_id] is the embedding.
            # Note: this is LEARNED positional encoding (like GPT-2), not sinusoidal (like
            # the original Transformer). Learned positions are simpler to implement here.
            'wpe': matrix(self.block_size, self.n_embd),

            # Language Model Head: vocab_size × n_embd
            # Projects the final hidden state back to vocabulary logits (for next-token prediction).
            # "Logits" are unnormalized log-probabilities — one per token in the vocabulary.
            'lm_head': matrix(self.vocab_size, self.n_embd),
        }

        # For each transformer layer, add the attention and MLP weight matrices.
        for i in range(self.n_layer):
            # Attention projections: all n_embd × n_embd (square matrices).
            # Q, K, V projections transform the input into queries, keys, and values.
            # attn_wo projects the concatenated head outputs back to n_embd.
            self.state_dict[f'layer{i}.attn_wq'] = matrix(self.n_embd, self.n_embd)  # Query projection
            self.state_dict[f'layer{i}.attn_wk'] = matrix(self.n_embd, self.n_embd)  # Key projection
            self.state_dict[f'layer{i}.attn_wv'] = matrix(self.n_embd, self.n_embd)  # Value projection
            self.state_dict[f'layer{i}.attn_wo'] = matrix(self.n_embd, self.n_embd)  # Output projection

            # Quick reminder:
            # Queries: representation of what the current token is looking for.
            # Keys: representation of what each token offers for matching.
            # Values: the information content that gets aggregated when a key matches a query.
            # Output projection: linear layer that mixes the concatenated outputs of all attention heads
            #   back into the model's embedding dimension.

            # MLP projections — expand to 4× width then contract back.
            # The 4× expansion ("intermediate size") is a GPT-2 convention.
            # More width = more capacity for storing knowledge in the MLP.
            self.state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * self.n_embd, self.n_embd)  # Expand: 16 → 64
            self.state_dict[f'layer{i}.mlp_fc2'] = matrix(self.n_embd, 4 * self.n_embd)  # Contract: 64 → 16

        # Flatten all parameters into a single list for the optimizer.
        # The optimizer needs to iterate over every scalar Value to update it.
        self.params = [p for mat in self.state_dict.values() for row in mat for p in row]
        print(f"num params: {len(self.params)}")
        # With these defaults: 27*16 + 16*16 + 27*16 + 4*(16*16*4 + 16*16*2) ≈ 3,776 params.
        # GPT-2 small: 117M params. GPT-4: estimated ~1.8T params.

    @staticmethod
    def get_head_dimension(n_embd: int, n_head: int) -> int:
        # Dimension per head
        if n_embd % n_head != 0:
            raise ValueError(f"Embedding dimension {n_embd} must be divisible by number of heads {n_head}")
        return n_embd // n_head

    @staticmethod
    def _linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
        """
        Linear (fully-connected) layer: y = W @ x
        x: list of n_in Value nodes (input vector)
        w: list of n_out lists of n_in Value nodes (weight matrix)
        Returns: list of n_out Value nodes

        Each output neuron is a dot product of one row of w with x.
        This is the fundamental operation of neural networks.
        """
        return [sum((wi * xi for wi, xi in zip(wo, x)), Value(0)) for wo in w]

    @staticmethod
    def _softmax(logits: list[Value]) -> list[Value]:
        """
        Softmax: converts raw logits to a probability distribution.
        Output values are in (0, 1) and sum to 1.

        Formula: softmax(x_i) = exp(x_i) / sum_j(exp(x_j))

        Numerical stability trick: subtract max(logits) before exponentiating.
        This prevents overflow (exp of large numbers → inf).
        Mathematically equivalent: exp(x-c) / sum(exp(x_j-c)) = exp(x) / sum(exp(x_j))
        because the constant c cancels in numerator and denominator.

        Note: we use val.data (the raw float) for the max computation, not val itself.
        We only need Value nodes in the graph for things we backpropagate through.
        Finding the max is just a data-dependent control flow decision.
        """
        max_val = max(val.data for val in logits)  # plain float, for numerical stability
        exps = [(val - max_val).exp() for val in logits]  # Value nodes
        total = sum(exps)  # sum() calls __radd__ repeatedly, works because __radd__ is defined
        return [e / total for e in exps]

    @staticmethod
    def _rmsnorm(x : list[Value]) -> list[Value]:
        """
        Root Mean Square Layer Normalization.
        Normalizes x so that its RMS (root mean square) is ~1.

        Standard LayerNorm: subtracts mean, divides by std, applies learned scale+bias.
        RMSNorm (Zhang & Sennrich 2019): skips mean subtraction and learned params.
        - Hypothesis: re-centering (mean subtraction) is not necessary.
        - Result: ~15% faster, similar quality.
        - Used by LLaMA, Mistral, and many modern models.

        Formula: RMSNorm(x_i) = x_i / sqrt(mean(x^2) + epsilon)

        Why normalize at all? Without normalization, activations can grow or shrink
        exponentially through layers (vanishing/exploding activations), making
        training unstable. Normalization keeps activations in a healthy range.

        epsilon (1e-5) prevents division by zero if x is all zeros.
        """
        ms = sum(xi * xi for xi in x) / len(x)  # mean of squares: E[x^2]
        scale = (ms + 1e-5) ** -0.5              # 1 / sqrt(E[x^2] + eps)
        return [xi * scale for xi in x]

    # =============================================================================
    # SECTION 5: MODEL ARCHITECTURE
    # =============================================================================
    # The model is a function: (token_id, pos_id, kv_cache) → logits[vocab_size]
    # It follows GPT-2 with three simplifications:
    #   1. LayerNorm → RMSNorm  (simpler: no mean subtraction, no learned scale/bias)
    #   2. GeLU → ReLU          (simpler: one less special function)
    #   3. No biases anywhere   (fewer parameters, still works fine at this scale)
    def _gpt(self, token_id: int, pos_id: int, keys: list[list[list[Value]]], values: list[list[list[Value]]]) -> list[Value]:
        """
        One forward step of the GPT: given one token at one position, return logits.

        This processes ONE token at a time (not a whole sequence at once).
        The KV cache (keys, values) accumulates past context across calls.

        Arguments:
        token_id: int — the current input token (index into vocabulary)
        pos_id:   int — the current position in the sequence (0-indexed)
        keys:     list[list] — per-layer cache of key vectors for all past positions
        values:   list[list] — per-layer cache of value vectors for all past positions

        Returns:
        logits: list of vocab_size Value nodes — raw scores for each next token

        Architecture (follows GPT-2 pre-norm variant):
        Input embedding
        → RMSNorm
        → N × (Attention block with residual + MLP block with residual)
        → Linear projection to vocab size
        """

        # -------------------------------------------------------------------------
        # Input Embedding
        # -------------------------------------------------------------------------
        # Look up the learned vector for this token.
        # wte[token_id] is a list of n_embd Value nodes.
        tok_emb : list[Value] = self.state_dict['wte'][token_id]

        # Look up the learned vector for this position.
        # wpe[pos_id] is a list of n_embd Value nodes.
        pos_emb : list[Value] = self.state_dict['wpe'][pos_id]

        # Combine token and position information by addition.
        # Why addition? It's the simplest operation that lets each dimension carry
        # both token identity and positional information. The model learns to
        # disentangle them during training.
        x = [t + p for t, p in zip(tok_emb, pos_emb)]

        # Apply RMSNorm to the initial embedding.
        # This is unusual — GPT-2 doesn't normalize here. The author notes it's
        # "not redundant due to backward pass via the residual connection":
        # because later residual connections add unnormalized vectors back in,
        # normalizing here helps keep the initial scale reasonable.
        x = self._rmsnorm(x)

        # -------------------------------------------------------------------------
        # Transformer Layers
        # Each layer has two sub-blocks: Attention and MLP.
        # Both use the pre-norm pattern (normalize before the sub-block, not after).
        # Both use residual connections (add input to output).
        # -------------------------------------------------------------------------
        for li in range(self.n_layer):

            # --- Attention Block ---
            # Attention is the "communication" mechanism: tokens look at each other
            # and selectively gather information from the past.

            x_residual = x   # save x for the residual connection

            x = self._rmsnorm(x)   # pre-norm before attention

            # Project x into Query, Key, Value spaces.
            # All three are the same shape (n_embd) — they just live in different
            # learned linear subspaces optimized for their respective roles:
            #   Q ("what am I looking for?"): drives the attention pattern
            #   K ("what do I contain?"):     gets matched against queries
            #   V ("what do I offer?"):       the actual information retrieved
            q = self._linear(x, self.state_dict[f'layer{li}.attn_wq'])  # query for current token
            k = self._linear(x, self.state_dict[f'layer{li}.attn_wk'])  # key for current token
            v = self._linear(x, self.state_dict[f'layer{li}.attn_wv'])  # value for current token

            # Append current k, v to the cache.
            # keys[li] grows by one entry per forward call (one per position).
            # This is the KV cache: it lets us reuse past k, v without recomputing them.
            # After processing position t, keys[li] = [k_0, k_1, ..., k_t].
            #
            # TRAINING vs. INFERENCE difference (important):
            # In production inference, KV cache entries are detached tensors (no grad).
            # Here, they're live Value nodes still connected to the computation graph.
            # So backprop flows THROUGH the cache into all past positions' parameters.
            # This is correct for training but expensive (the full sequence graph is live).
            keys[li].append(k)
            values[li].append(v)

            # Multi-head attention: split Q/K/V into n_head independent attention heads.
            # Each head operates on a head_dim-dimensional slice of the embedding.
            # Why multiple heads? Each head can attend to different aspects of context
            # (e.g., one head for syntax, one for semantics, one for position patterns).
            x_attn = []
            for h in range(self.n_head):
                hs = h * self.head_dim   # start index of this head's slice

                # Extract this head's slice from Q, K, V.
                q_h = q[hs:hs+self.head_dim]                           # [head_dim] query
                k_h = [ki[hs:hs+self.head_dim] for ki in keys[li]]    # [T, head_dim] keys
                v_h = [vi[hs:hs+self.head_dim] for vi in values[li]]  # [T, head_dim] values
                # T = number of tokens seen so far (grows from 1 to sequence length)

                # Compute attention scores: dot(Q, K^T) / sqrt(head_dim)
                # For each past position t, score_t = Q · K_t / sqrt(d_head)
                # The 1/sqrt(head_dim) scaling prevents dot products from growing too
                # large in magnitude (which would push softmax into near-zero gradient
                # regions). This is the "scaled" in "Scaled Dot-Product Attention".
                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(self.head_dim)) / self.head_dim**0.5
                    for t in range(len(k_h))
                ]

                # Softmax over attention logits → attention weights (sum to 1).
                # This is a probability distribution over past positions:
                # attn_weights[t] = "how much attention to pay to position t".
                # No explicit causal mask is needed here because keys[li] only
                # contains positions 0..pos_id (we append one at a time, so the
                # future is never in the cache during training).
                attn_weights = self._softmax(attn_logits)

                # Weighted sum of values: the attention output for this head.
                # For each output dimension j, sum over all past positions t.
                # This is "retrieving" information: blend past value vectors weighted
                # by how relevant they are (as determined by Q·K similarity).
                head_out = [
                    sum((attn_weights[t] * v_h[t][j] for t in range(len(v_h))), Value(0))
                    for j in range(self.head_dim)
                ]
                x_attn.extend(head_out)  # concatenate head outputs
                # After all heads: x_attn has shape [n_head * head_dim] = [n_embd]

            # Project concatenated heads back to n_embd.
            # This allows heads to communicate: attn_wo mixes information across heads.
            x = self._linear(x_attn, self.state_dict[f'layer{li}.attn_wo'])

            # Residual connection: add the pre-attention x back in.
            # Why residual connections? They provide a "gradient highway":
            # during backprop, gradients can flow directly from later layers to
            # earlier layers without passing through the attention computation.
            # This is why very deep networks (GPT-3 has 96 layers) can be trained.
            x = [a + b for a, b in zip(x, x_residual)]

            # --- MLP Block ---
            # The MLP is the "computation" mechanism: tokens process information
            # locally (no cross-token communication here, unlike attention).
            # Think of it as a per-token key-value store, or a lookup table.

            x_residual = x   # save again for MLP residual

            x = self._rmsnorm(x)   # pre-norm before MLP

            # FC1: expand from n_embd (16) → 4*n_embd (64) dimensions.
            # The 4× expansion gives the network more capacity to represent
            # complex functions. This ratio (4×) comes from the original Transformer
            # paper and has been retained in GPT-2/3/4 as a convention.
            x = self._linear(x, self.state_dict[f'layer{li}.mlp_fc1'])

            # ReLU activation: element-wise max(0, x).
            # Introduces non-linearity — without this, stacking linear layers
            # collapses to a single linear layer (no additional expressiveness).
            # GPT-2 uses GeLU (smoother than ReLU); microgpt uses ReLU for simplicity.
            x = [xi.relu() for xi in x]

            # FC2: contract from 4*n_embd (64) → n_embd (16) dimensions.
            x = self._linear(x, self.state_dict[f'layer{li}.mlp_fc2'])

            # Residual connection for MLP block.
            x = [a + b for a, b in zip(x, x_residual)]

        # -------------------------------------------------------------------------
        # Output Head
        # -------------------------------------------------------------------------
        # Project the final hidden state x (n_embd) to logits over the vocabulary.
        # lm_head: vocab_size × n_embd → output is vocab_size logits.
        # Logit[i] is the unnormalized score for token i being the next token.
        # Higher logit = model thinks that token is more likely to come next.
        logits = self._linear(x, self.state_dict['lm_head'])
        return logits

    def _optimizer(self) -> None:
        # Once we have gradients (from backward()), we update parameters with an optimizer.
        # Gradient descent: p -= lr * p.grad  (simplest, but slow to converge)
        # Adam (Adaptive Moment Estimation, Kingma & Ba 2014): the standard for LLMs.
        #
        # Adam maintains two moving averages per parameter:
        #   m_i: first moment  — exponential moving average of gradients (mean direction)
        #   v_i: second moment — exponential moving average of squared gradients (variance)
        #
        # Update rule: p -= lr * m_hat / (sqrt(v_hat) + eps)
        #   where m_hat and v_hat are bias-corrected estimates.
        #
        # Why Adam over plain SGD?
        #   - Adapts learning rate per-parameter: rare parameters get bigger updates.
        #   - Momentum (m): smooths out noisy gradients, accelerates in consistent directions.
        #   - RMS scaling (v): prevents oscillation in directions with high gradient variance.

        self.learning_rate = INITIAL_LEARNING_RATE

        # Optimizer state: one float per parameter (not Value — no autograd needed here).
        self.m = [0.0] * len(self.params)  # first moment buffers, initialized to 0
        self.v = [0.0] * len(self.params)  # second moment buffers, initialized to 0

    def _training(self) -> None:
        # The training loop is the engine that makes the model learn.

        if self.inference_only:
            raise ValueError("Training should not be called in inference-only mode")

        # Each step:
        #   1. Sample a document
        #   2. Forward pass: build computation graph, compute loss
        #   3. Backward pass: compute gradients via backprop
        #   4. Optimizer step: update parameters
        #   5. Zero gradients: reset for next step
        for step in range(self.num_training_steps):

            # -------------------------------------------------------------------------
            # 1. Sample and tokenize a document
            # -------------------------------------------------------------------------
            # Cycle through the shuffled dataset. With 32K docs and 1K steps, we see
            # only ~3% of the dataset — but names share patterns, so generalization occurs.
            doc = self.docs[step % len(self.docs)]

            if VERBOSE_METADATA:
                self.seen_training_docs.append(doc)

            # Tokenize: convert name characters to integer IDs.
            # Wrap with BOS on both sides: [BOS, char0, char1, ..., charN, BOS]
            # The trailing BOS is the target for the last character: "after the last
            # character of a name, predict end-of-sequence."
            tokens = [self.BOS] + [self.uchars.index(ch) for ch in doc] + [self.BOS]

            # n: number of (input, target) pairs we'll train on.
            # We predict tokens[1..n] from tokens[0..n-1].
            # Capped at block_size to respect the model's maximum context length.
            n = min(self.block_size, len(tokens) - 1)

            # -------------------------------------------------------------------------
            # 2. Forward pass: compute loss
            # -------------------------------------------------------------------------
            # Initialize KV cache for this document.
            # keys[li] and values[li] will accumulate one entry per position.
            keys_cache: list[list[list[Value]]]  = [[] for _ in range(self.n_layer)]
            values_cache: list[list[list[Value]]] = [[] for _ in range(self.n_layer)]

            losses: list[Value] = []
            for pos_id in range(n):
                token_id  = tokens[pos_id]      # current input token
                target_id = tokens[pos_id + 1]  # next token we want the model to predict

                # Run the forward pass for this position.
                # This builds the computation graph incrementally.
                logits = self._gpt(token_id, pos_id, keys_cache, values_cache)

                # Convert logits to probabilities.
                probs = self._softmax(logits)

                # Cross-entropy loss for this position: -log(p_target)
                # If the model assigns p=1.0 to the correct token → loss = 0 (perfect)
                # If the model assigns p=0.01 → loss = -log(0.01) ≈ 4.6 (bad)
                # Random chance for vocab_size=27: -log(1/27) ≈ 3.3 (baseline to beat)
                loss_t = -probs[target_id].log()
                losses.append(loss_t)

            # Average loss across all positions in this document.
            # Using the mean (1/n * sum) rather than sum ensures the loss scale doesn't
            # depend on document length — important for a consistent learning rate.
            loss = (1 / n) * sum(losses, Value(0))

            # -------------------------------------------------------------------------
            # 3. Backward pass: compute gradients
            # -------------------------------------------------------------------------
            # Traverse the entire computation graph backwards and fill in .grad for
            # every Value node, including all model parameters.
            loss.backward()

            # -------------------------------------------------------------------------
            # 4. Optimizer step: update parameters
            # -------------------------------------------------------------------------
            # Linear learning rate decay: starts at learning_rate, decays to 0.
            # lr_t(step=0) = learning_rate * 1.0  (full rate at start)
            # lr_t(step=999) = learning_rate * 0.001 (nearly zero at end)
            # Why decay? Early training benefits from large steps (exploration);
            # late training needs small steps to fine-tune without overshooting minima.
            lr_t = self.learning_rate * (1 - step / self.num_training_steps)

            for i, p in enumerate(self.params):
                # Adam update equations:
                # m_i ← β1 * m_i + (1 - β1) * grad           (EMA of gradient)
                self.m[i] = BETA_1 * self.m[i] + (1 - BETA_1) * p.grad
                # v_i ← β2 * v_i + (1 - β2) * grad^2          (EMA of squared gradient)
                self.v[i] = BETA_2 * self.v[i] + (1 - BETA_2) * p.grad ** 2

                # Bias correction: at early steps, m and v are biased toward 0
                # (they're initialized at 0). Dividing by (1 - β^t) corrects this.
                # As t→∞, (1 - β^t) → 1, so corrections disappear.
                m_hat = self.m[i] / (1 - BETA_1 ** (step + 1))
                v_hat = self.v[i] / (1 - BETA_2 ** (step + 1))

                # Parameter update: gradient descent with adaptive step size.
                # sqrt(v_hat) ≈ RMS of recent gradients → parameters with high gradient
                # variance get a smaller effective learning rate (more cautious updates).
                p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)

                # Zero the gradient manually. There's no .zero_grad() method here.
                # Must be done AFTER the optimizer step, BEFORE the next forward pass.
                # Failure to zero grads would accumulate gradients across steps (wrong).
                p.grad = 0

            # Print progress on same line (\r) to avoid flooding the terminal.
            print(f"step {step+1:4d} / {self.num_training_steps:4d} | loss {loss.data:.4f}", end='\r')

        print()  # newline after training is done

    def _next_token(
            self, token_id: int, pos_id: int, keys_cache: list[list[list[Value]]], values_cache: list[list[list[Value]]],
            temperature: float) -> int:

        # Get logits from the model given the current token and position.
        logits = self._gpt(token_id, pos_id, keys_cache, values_cache)

        # Apply temperature scaling to logits (not to probabilities).
        # Dividing logits by T < 1 makes differences between logits larger,
        # sharpening the softmax distribution (peakier → more deterministic).
        # Mathematically equivalent to: softmax(logits/T)
        probs = self._softmax([l / temperature for l in logits])

        # Sample next token from the probability distribution.
        # random.choices returns a weighted random sample — this is the
        # stochastic part of generation. Two identical prefixes can yield
        # different continuations.
        # Note: .data is needed to extract plain floats from Value nodes.
        next_token_id = random.choices(range(self.vocab_size), weights=[p.data for p in probs])[0]

        return next_token_id

    def _input_inference(
            self, inference_decorator: InferenceDecorator, temperature: float, num_inference_results: int) -> None:

        input_sequence = input(f"\nEnter starting sequence (max {self.block_size - 1} characters): ")
        input_sequence = input_sequence.strip().lower()
        if len(input_sequence) > self.block_size - 1:
            print(f"Input sequence too long (max {self.block_size - 1} characters)")
            exit(1)
        sequence = [self.uchars.index(ch) for ch in input_sequence]

        for sample_idx in range(num_inference_results):
            # Fresh KV cache for each generated name.
            keys_cache: list[list[list[Value]]]  = [[] for _ in range(self.n_layer)]
            values_cache: list[list[list[Value]]] = [[] for _ in range(self.n_layer)]

            sample = [self.uchars[token_id] for token_id in sequence]

            token_id = sequence[-1]
            for pos_id in range(len(sequence)-1, self.block_size):
                token_id = self._next_token(token_id, pos_id, keys_cache, values_cache, temperature)

                # If we sampled BOS, the model is signaling "end of name"
                if token_id == self.BOS:
                    break

                sample.append(self.uchars[token_id])

            print(inference_decorator.decorate_result(''.join(sample), sample_idx))


    def _random_inference(
            self, inference_decorator: InferenceDecorator, temperature: float, num_inference_results: int) -> None:
        # After training or on demand, use the model to generate new names.
        # The model might never have seen these names — it learned the underlying distribution
        # of character sequences and samples from it.

        for sample_idx in range(num_inference_results):
            # Fresh KV cache for each generated name.
            keys_cache: list[list[list[Value]]]  = [[] for _ in range(self.n_layer)]
            values_cache: list[list[list[Value]]] = [[] for _ in range(self.n_layer)]

            # Start generation with BOS: this signals "beginning of a new name".
            token_id = self.BOS
            sample = []

            for pos_id in range(self.block_size):
                token_id = self._next_token(token_id, pos_id, keys_cache, values_cache, temperature)

                # If we sampled BOS, the model is signaling "end of name"
                if token_id == self.BOS:
                    break

                sample.append(self.uchars[token_id])

            print(inference_decorator.decorate_result(''.join(sample), sample_idx))


    def train(self) -> None:
        if not self.inference_only:
            print("--- training ---")
            self._dataset()
            self._tokenizer()
            self._model_parameters()
            self._optimizer()
            self._training()

    def infer(
            self, temperature: float, num_inference_results: int, inference_input: bool, fixed_random_seed: bool
        ) -> None:

        print("--- inference ---")

        if fixed_random_seed:
            print(f"  Fixed random seed for inference: {RANDOM_SEED}")
            # Re-seeding here so inference is deterministic regardless of whether training ran beforehand.
            # Without this, the PRNG position differs between training mode and inference-only mode.
            random.seed(RANDOM_SEED)
        else:
            print("  NOT fixing random seed for inference")

        inference_decorator = InferenceDecorator(self.inputs_file)
        inference_decorator.draw_color_legend()

        if inference_input:
            self._input_inference(inference_decorator, temperature, num_inference_results)
        else:
            self._random_inference(inference_decorator, temperature, num_inference_results)
