from __future__ import annotations

import json
import math
import sys

from config import MAX_RECURSION_DEPTH

sys.setrecursionlimit(MAX_RECURSION_DEPTH)

# =============================================================================
# SECTION 3: AUTOGRAD ENGINE
# =============================================================================
# To train a neural network, we need gradients: ∂Loss/∂parameter for every
# parameter. Computing these by hand for a GPT would be intractable.
#
# Automatic differentiation ("autograd") does this automatically by:
#   1. During the forward pass (prediction/inference, input → output), recording every operation in a computation graph.
#   2. During the backward pass (loss gradient computation and propagation, output → input),
#      applying the chain rule through that graph.
#
# This is exactly what PyTorch's autograd does. Here we implement a scalar
# version from scratch — each Value node holds a single float, not a tensor.
# This makes the code transparent at the cost of speed (no vectorization).

class Value:
    # __slots__ is a Python memory optimization.
    # Normally, Python objects store attributes in a dict (__dict__), which has
    # significant per-object overhead. __slots__ replaces that dict with a fixed
    # struct, saving ~50-70% memory per Value object.
    # Critical here because a single forward pass creates tens of thousands of
    # Value objects (every scalar multiply/add = new Value).
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data: float | int, children: tuple[Value, ...] = (), local_grads: tuple[float, ...] = ()) -> None:
        # data: the scalar value computed in the forward pass (e.g. 0.42)
        self.data = data

        # grad: ∂Loss/∂self — how much the loss changes if self changes slightly.
        # Initialized to 0; filled in during backward().
        # After backward(), every leaf Value (model parameter) has its grad set.
        self.grad = 0

        # _children: the Value nodes that were inputs to the operation that
        # produced self. Forms the edges of the computation graph.
        # Example: if c = a + b, then c._children = (a, b)
        self._children = children

        # _local_grads: ∂self/∂child for each child.
        # Example for c = a + b: ∂c/∂a = 1, ∂c/∂b = 1 → local_grads = (1, 1)
        # Example for c = a * b: ∂c/∂a = b, ∂c/∂b = a → local_grads = (b, a)
        # These are plain floats (not Value), computed at forward-pass time.
        self._local_grads = local_grads

    # -------------------------------------------------------------------------
    # Forward-pass operations — each returns a new Value node and records
    # the local gradients needed for backprop.
    # -------------------------------------------------------------------------

    def __add__(self, other: Value | float | int) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        # ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1 — addition distributes gradients equally.
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other: Value | float | int) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        # ∂(a*b)/∂a = b, ∂(a*b)/∂b = a — the "swap" rule of multiplication.
        # Local grads are stored as plain floats (other.data, self.data),
        # not as Value nodes, because we don't need to differentiate through them.
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other: float | int) -> Value:
        # other is a plain int/float (not a Value) — we don't differentiate
        # with respect to the exponent, only the base.
        # ∂(x^n)/∂x = n * x^(n-1)
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self) -> Value:
        # Natural logarithm. ∂(ln x)/∂x = 1/x.
        # Used in cross-entropy loss: loss = -log(p_correct).
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self) -> Value:
        # Exponential. ∂(e^x)/∂x = e^x (its own derivative).
        # Used in softmax: exp(logit) / sum(exp(logits)).
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self) -> Value:
        # Rectified Linear Unit: max(0, x).
        # ∂ReLU/∂x = 1 if x > 0, else 0 (sub-gradient at x=0 is taken as 0).
        # Used in the MLP block as the activation function.
        # GPT-2 uses GeLU; microgpt uses ReLU for simplicity (one less dependency).
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    # -------------------------------------------------------------------------
    # Derived operations — implemented in terms of the primitives above.
    # These don't need custom backward logic; they compose existing ops.
    # -------------------------------------------------------------------------
    def __neg__(self) -> Value:    return self * -1
    def __radd__(self, other: Value | float | int) -> Value: return self + other     # supports: 0 + Value (used by sum())
    def __sub__(self, other: Value | float | int) -> Value:  return self + (-other)
    def __rsub__(self, other: Value | float | int) -> Value: return other + (-self)
    def __rmul__(self, other: float | int) -> Value: return self * other
    def __truediv__(self, other: Value | float | int) -> Value:  return self * other**-1
    def __rtruediv__(self, other: float | int) -> Value: return other * self**-1

    # -------------------------------------------------------------------------
    # Backward pass — reverse-mode automatic differentiation (backpropagation).
    # -------------------------------------------------------------------------
    def backward(self) -> None:
        # Step 1: Topological sort of the computation graph.
        # We need to process nodes in reverse dependency order:
        # a node must receive all gradient contributions from its consumers
        # before it propagates gradients to its own children.
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)  # depth-first: recurse into children first
                topo.append(v)         # append AFTER children → leaves come first
        build_topo(self)

        # Step 2: Seed the gradient at the root (the loss node).
        # ∂Loss/∂Loss = 1 by definition. This starts the chain rule.
        self.grad = 1

        # Step 3: Walk the graph in reverse topological order (root → leaves).
        # At each node v, distribute v.grad to v's children via the chain rule:
        #   child.grad += (∂v/∂child) * v.grad
        # The += is critical: if a Value is used in multiple places (shared node),
        # its gradient must accumulate contributions from all paths. This is the
        # multivariable chain rule.
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


class ValueJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Value):
            return obj.data
        return super().default(obj)
