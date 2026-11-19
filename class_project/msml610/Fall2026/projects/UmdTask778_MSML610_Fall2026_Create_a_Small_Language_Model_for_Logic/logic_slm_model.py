"""
Llama-style decoder configurations, built from scratch (random init).

Sizes follow the project plan: ~25M, ~127M and ~1.0B non-embedding
parameters, RoPE, RMSNorm, SwiGLU, no biases, tied input/output embeddings.

Import as:

import logic_slm_model as lsmod
"""

from typing import Dict, Tuple

import torch
import transformers

# name -> (hidden, layers, heads, intermediate)
SIZES: Dict[str, Tuple[int, int, int, int]] = {
    "tiny": (64, 2, 2, 192),  # unit tests / CPU smoke runs only
    "25m": (512, 8, 8, 1408),
    "125m": (768, 18, 12, 2048),
    "1b": (2048, 20, 16, 5504),
}
PEAK_LR = {"tiny": 1e-3, "25m": 1e-3, "125m": 6e-4, "1b": 3e-4}


def make_config(
    size: str,
    vocab_size: int,
    *,
    eos_token_id: int,
    pad_token_id: int,
    max_position_embeddings: int = 1024,
    dropout: float = 0.0,
) -> transformers.LlamaConfig:
    """
    Build the `LlamaConfig` for one of the named sizes.
    """
    hidden, layers, heads, inter = SIZES[size]
    return transformers.LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        intermediate_size=inter,
        hidden_act="silu",
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=dropout,
        tie_word_embeddings=True,
        bos_token_id=eos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        initializer_range=0.02,
    )


def make_model(config: transformers.LlamaConfig) -> transformers.LlamaForCausalLM:
    """
    Instantiate a randomly initialized model with SDPA attention.
    """
    config._attn_implementation = "sdpa"
    return transformers.LlamaForCausalLM(config)


def count_params(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count total and non-embedding parameters (tied weights counted once).
    """
    total = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight.numel()
    return {"total": total, "non_embedding": total - emb}


def analytic_non_embedding_params(size: str) -> int:
    """
    Non-embedding parameter count from the shape alone (no allocation).
    """
    h, n_layers, _, inter = SIZES[size]
    per_layer = 4 * h * h + 3 * h * inter + 2 * h  # attn + SwiGLU + 2 norms
    return n_layers * per_layer + h  # + final norm
