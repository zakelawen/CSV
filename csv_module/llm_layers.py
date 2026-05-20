# """
# LLM layer manipulation utilities for CSV injection.
# Adapted from TSV/llm_layers.py with Gemma/LLaMA support.

# Key idea:
#     Do NOT manually re-implement decoder-layer internals.

# Different transformers versions may return different numbers of values from
# self_attn(), especially for Gemma2 with the newer cache API. Therefore, the
# wrapper delegates the full forward pass to the original decoder layer, then
# injects the steering vector into outputs[0], i.e. the hidden_states after the
# complete decoder layer.

# This keeps the injection position equivalent to:
#     hidden_states = original_decoder_layer(...)[0]
#     hidden_states = hidden_states + lambda * v

# and avoids architecture/version-specific unpacking bugs.
# """

# from torch import nn
# from transformers import PreTrainedModel


# class TSVLayer(nn.Module):
#     """
#     Steering vector injection layer.
#     Adds λ * v to all token positions.

#     v / tsv is stored in fp32 for optimizer stability.
#     It is cast to the hidden states' dtype and device during forward.
#     """

#     def __init__(self, tsv, lam):
#         super().__init__()
#         self.tsv = tsv      # nn.Parameter, usually fp32, shape [1, 1, D]
#         self.lam = lam      # list with one float, e.g. [5.0]

#     def forward(self, x):
#         if self.tsv is not None:
#             # Cast v to match hidden states dtype and device.
#             # This is important when using device_map="auto".
#             v = self.lam[0] * self.tsv.to(device=x.device, dtype=x.dtype)

#             # [1, 1, D] -> [1, seq_len, D], broadcast over batch.
#             v = v.expand(-1, x.shape[1], -1)

#             x = x + v

#         return x


# class LlamaDecoderLayerWrapper(nn.Module):
#     """
#     Wraps a decoder layer to inject TSV/CSV into the residual stream.

#     Works for LLaMA-like layers and Gemma2 layers by delegating the original
#     forward pass to the original decoder layer.

#     Important for Gemma2:
#         Gemma2Model.forward() may directly access decoder_layer.attention_type.
#         Since this wrapper replaces the original layer inside model.layers,
#         the wrapper must expose that attribute too.
#     """

#     def __init__(self, llama_decoder_layer, tsv_layer, model_name=''):
#         super().__init__()
#         self.llama_decoder_layer = llama_decoder_layer
#         self.tsv_layer = tsv_layer
#         self.model_name = model_name

#         # ------------------------------------------------------------------
#         # Compatibility attributes
#         # ------------------------------------------------------------------
#         # Gemma2Model.forward() directly reads decoder_layer.attention_type.
#         # After replacing the original decoder layer with this wrapper, this
#         # attribute must still exist on the wrapper.
#         #
#         # If future HF model code reads additional attributes directly from
#         # decoder_layer, add them to this list.
#         # ------------------------------------------------------------------
#         for attr in [
#             "attention_type",
#         ]:
#             if hasattr(llama_decoder_layer, attr):
#                 setattr(self, attr, getattr(llama_decoder_layer, attr))

#     def forward(self, *args, **kwargs):
#         """
#         Run the original decoder layer, then inject CSV into hidden_states.

#         The original decoder layer usually returns a tuple like:
#             (hidden_states,)
#             (hidden_states, self_attn_weights)
#             (hidden_states, self_attn_weights, present_key_value)

#         depending on output_attentions / use_cache / transformers version.

#         We preserve everything except outputs[0], which we replace with the
#         CSV-injected hidden states.
#         """

#         outputs = self.llama_decoder_layer(*args, **kwargs)

#         # Most HF decoder layers return a tuple.
#         if isinstance(outputs, tuple):
#             hidden_states = outputs[0]
#             hidden_states = self.tsv_layer(hidden_states)
#             return (hidden_states,) + outputs[1:]

#         # Defensive fallback in case a future layer returns a list.
#         if isinstance(outputs, list):
#             hidden_states = outputs[0]
#             hidden_states = self.tsv_layer(hidden_states)
#             outputs[0] = hidden_states
#             return outputs

#         # Very defensive fallback: if the layer returns just hidden_states.
#         hidden_states = self.tsv_layer(outputs)
#         return hidden_states


# # ---------------------------------------------------------------------------
# # Utility functions
# # ---------------------------------------------------------------------------

# def get_nested_attr(obj, attr_path):
#     for attr in attr_path.split("."):
#         obj = getattr(obj, attr)
#     return obj


# def find_longest_modulelist(model, path=""):
#     longest_path, longest_len = path, 0

#     for name, child in model.named_children():
#         child_path = f"{path}.{name}" if path else name

#         if isinstance(child, nn.ModuleList) and len(child) > longest_len:
#             longest_len, longest_path = len(child), child_path

#         cp, cl = find_longest_modulelist(child, child_path)
#         if cl > longest_len:
#             longest_len, longest_path = cl, cp

#     return longest_path, longest_len


# def get_layers(model: PreTrainedModel):
#     longest_path, _ = find_longest_modulelist(model)
#     return get_nested_attr(model, longest_path)


# def add_tsv_layers(
#     model: PreTrainedModel,
#     tsv,
#     alpha: list,
#     str_layer: int,
#     model_name: str = '',
# ):
#     """
#     Inject TSVLayer into the residual stream of the specified transformer layer.

#     Args:
#         model:
#             HuggingFace causal LM model.
#         tsv:
#             Trainable steering vector, usually shape [1, 1, hidden_size].
#         alpha:
#             A one-element list containing lambda, e.g. [5.0].
#         str_layer:
#             Transformer layer index, 0-based.
#         model_name:
#             Optional model key/name.
#     """

#     layers = get_layers(model)

#     assert str_layer < len(layers), (
#         f"str_layer={str_layer} >= num_layers={len(layers)}"
#     )

#     decoder_layer = layers[str_layer]

#     layers[str_layer] = LlamaDecoderLayerWrapper(
#         decoder_layer,
#         TSVLayer(tsv, alpha),
#         model_name,
#     )

# add qwen3-4b support

"""
LLM layer manipulation utilities for CSV injection.
Adapted from TSV/llm_layers.py with Gemma2 / LLaMA / Qwen3 support.

Key idea:
    Do NOT manually re-implement decoder-layer internals.

Different transformers versions may return different numbers of values from
self_attn(), especially for Gemma2 with the newer cache API. Therefore, the
wrapper delegates the full forward pass to the original decoder layer, then
injects the steering vector into the layer's hidden_states output.

Architecture-specific return types:
    - LLaMA / Gemma2 / Mistral / etc.   → returns a tuple (hidden_states, ...)
    - Qwen3 / Qwen3MoE                  → returns a Tensor (just hidden_states)

The wrapper records this distinction at __init__ time so forward() can preserve
the correct return type. This is critical because the parent model's forward
pass treats the return value differently (e.g., Qwen3Model does
`hidden_states = layer(...)` directly, while LlamaModel does
`hidden_states = layer(...)[0]`). Returning the wrong type would silently break
model forward passes.
"""

from torch import nn
from transformers import PreTrainedModel


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

# Decoder-layer classes whose forward() returns a single Tensor instead of
# a tuple. Add new architectures here as they are validated.
_TENSOR_RETURNING_LAYERS = {
    "Qwen3DecoderLayer",
    "Qwen3MoeDecoderLayer",
}


class TSVLayer(nn.Module):
    """
    Steering vector injection layer.
    Adds λ * v to all token positions.

    v / tsv is stored in fp32 for optimizer stability.
    It is cast to the hidden states' dtype and device during forward.
    """

    def __init__(self, tsv, lam):
        super().__init__()
        self.tsv = tsv      # nn.Parameter, usually fp32, shape [1, 1, D]
        self.lam = lam      # list with one float, e.g. [5.0]

    def forward(self, x):
        if self.tsv is not None:
            # Cast v to match hidden states dtype and device.
            # This is important when using device_map="auto".
            v = self.lam[0] * self.tsv.to(device=x.device, dtype=x.dtype)

            # [1, 1, D] -> [1, seq_len, D], broadcast over batch.
            v = v.expand(-1, x.shape[1], -1)

            x = x + v

        return x


class LlamaDecoderLayerWrapper(nn.Module):
    """
    Wraps a decoder layer to inject TSV/CSV into the residual stream.

    Works for LLaMA-like layers, Gemma2 layers (tuple return) and Qwen3 layers
    (Tensor return) by delegating the original forward pass to the original
    decoder layer.

    Important for Gemma2:
        Gemma2Model.forward() may directly access decoder_layer.attention_type.
        Since this wrapper replaces the original layer inside model.layers,
        the wrapper must expose that attribute too.

    Important for Qwen3:
        Qwen3DecoderLayer.forward returns a Tensor, not a tuple. The wrapper
        detects this at __init__ time via the layer's class name and forwards
        the correct type so Qwen3Model's forward loop works unchanged.
    """

    def __init__(self, llama_decoder_layer, tsv_layer, model_name=''):
        super().__init__()
        self.llama_decoder_layer = llama_decoder_layer
        self.tsv_layer = tsv_layer
        self.model_name = model_name

        # ------------------------------------------------------------------
        # Return-type signature
        # ------------------------------------------------------------------
        # Detect whether the original layer returns a Tensor (Qwen3 family) or
        # a tuple (LLaMA / Gemma2 / etc.). We use the class name rather than
        # isinstance() on the runtime output so that forward() is fully
        # deterministic and doesn't change behavior across calls.
        cls_name = type(llama_decoder_layer).__name__
        self._returns_tensor = cls_name in _TENSOR_RETURNING_LAYERS

        # ------------------------------------------------------------------
        # Compatibility attributes
        # ------------------------------------------------------------------
        # Gemma2Model.forward() directly reads decoder_layer.attention_type.
        # After replacing the original decoder layer with this wrapper, this
        # attribute must still exist on the wrapper. Qwen3 has no such
        # attribute, so hasattr() safely skips.
        #
        # If future HF model code reads additional attributes directly from
        # decoder_layer, add them to this list.
        # ------------------------------------------------------------------
        for attr in [
            "attention_type",
        ]:
            if hasattr(llama_decoder_layer, attr):
                setattr(self, attr, getattr(llama_decoder_layer, attr))

    def forward(self, *args, **kwargs):
        """
        Run the original decoder layer, then inject CSV into hidden_states.

        Return-type behavior is decided at __init__ time:
          - Tensor-returning layers (Qwen3) → return a Tensor
          - Tuple-returning layers (LLaMA/Gemma2) → return a tuple, with
            outputs[0] replaced by CSV-injected hidden_states.
        """

        outputs = self.llama_decoder_layer(*args, **kwargs)

        # Qwen3 and similar: layer returns a single Tensor.
        if self._returns_tensor:
            return self.tsv_layer(outputs)

        # Most HF decoder layers return a tuple
        # (hidden_states,) | (hidden_states, attn) | (hidden_states, attn, kv).
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            hidden_states = self.tsv_layer(hidden_states)
            return (hidden_states,) + outputs[1:]

        # Defensive fallback in case a future layer returns a list.
        if isinstance(outputs, list):
            hidden_states = outputs[0]
            hidden_states = self.tsv_layer(hidden_states)
            outputs[0] = hidden_states
            return outputs

        # Very defensive fallback for unknown architectures: assume Tensor.
        # If a new architecture lands here, add it to _TENSOR_RETURNING_LAYERS
        # so the behavior is explicit instead of relying on this branch.
        return self.tsv_layer(outputs)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_nested_attr(obj, attr_path):
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def find_longest_modulelist(model, path=""):
    longest_path, longest_len = path, 0

    for name, child in model.named_children():
        child_path = f"{path}.{name}" if path else name

        if isinstance(child, nn.ModuleList) and len(child) > longest_len:
            longest_len, longest_path = len(child), child_path

        cp, cl = find_longest_modulelist(child, child_path)
        if cl > longest_len:
            longest_len, longest_path = cl, cp

    return longest_path, longest_len


def get_layers(model: PreTrainedModel):
    longest_path, _ = find_longest_modulelist(model)
    return get_nested_attr(model, longest_path)


def add_tsv_layers(
    model: PreTrainedModel,
    tsv,
    alpha: list,
    str_layer: int,
    model_name: str = '',
):
    """
    Inject TSVLayer into the residual stream of the specified transformer layer.

    Args:
        model:
            HuggingFace causal LM model.
        tsv:
            Trainable steering vector, usually shape [1, 1, hidden_size].
        alpha:
            A one-element list containing lambda, e.g. [5.0].
        str_layer:
            Transformer layer index, 0-based.
        model_name:
            Optional model key/name.
    """

    layers = get_layers(model)

    assert str_layer < len(layers), (
        f"str_layer={str_layer} >= num_layers={len(layers)}"
    )

    decoder_layer = layers[str_layer]

    layers[str_layer] = LlamaDecoderLayerWrapper(
        decoder_layer,
        TSVLayer(tsv, alpha),
        model_name,
    )