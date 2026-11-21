"""
Decoder-only model configurations, built from scratch (random init).

Two architecture families:

- **Qwen2.5-style (plan v2, default for new runs)**: RMSNorm, RoPE, SwiGLU,
  grouped-query attention (few key/value heads), bias on Q/K/V, tied
  embeddings. Sizes `q125m`, `q0.5b` (= Qwen2.5-0.5B layer stack, ~358M
  non-embedding params) and `q1.5b` (= Qwen2.5-1.5B / Qwen2.5-Math-1.5B layer
  stack, ~1.31B non-embedding params). Same layer shapes as Qwen2.5 but our own
  tokenizer, so not weight-compatible with Qwen checkpoints.
- **Llama-style (plan v1)**: multi-head attention, no biases. Sizes `25m`,
  `125m`, `1b`; kept so the v1 baselines stay reproducible.

`tiny` / `qtiny` are for unit tests and CPU smoke runs only.

Import as:

import logic_slm_model as lsmod
"""

from typing import Dict, Tuple

import torch
import transformers

# Llama-style: name -> (hidden, layers, heads, intermediate)
SIZES: Dict[str, Tuple[int, int, int, int]] = {
    "tiny": (64, 2, 2, 192),
    "25m": (512, 8, 8, 1408),
    "125m": (768, 18, 12, 2048),
    "1b": (2048, 20, 16, 5504),
}
# Qwen2.5-style: name -> (hidden, layers, query heads, key/value heads, intermediate)
QWEN_SIZES: Dict[str, Tuple[int, int, int, int, int]] = {
    "qtiny": (64, 2, 4, 2, 192),
    "q125m": (768, 18, 12, 4, 2176),  # FFN widened from 2048 to offset GQA's smaller attention (~119M)
    "q0.5b": (896, 24, 14, 2, 4864),
    "q1.5b": (1536, 28, 12, 2, 8960),
}
PEAK_LR = {
    "tiny": 1e-3, "25m": 1e-3, "125m": 6e-4, "1b": 3e-4,
    "qtiny": 1e-3, "q125m": 6e-4, "q0.5b": 4e-4, "q1.5b": 2.5e-4,
}
ALL_SIZES = sorted(SIZES) + sorted(QWEN_SIZES)


def make_config(
    size: str,
    vocab_size: int,
    *,
    eos_token_id: int,
    pad_token_id: int,
    max_position_embeddings: int = 1024,
    dropout: float = 0.0,
) -> transformers.PretrainedConfig:
    """
    Build the model config for one of the named sizes.
    """
    common = dict(
        vocab_size=vocab_size,
        hidden_act="silu",
        max_position_embeddings=max_position_embeddings,
        rope_theta=10_000.0,
        attention_dropout=dropout,
        tie_word_embeddings=True,
        bos_token_id=eos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        initializer_range=0.02,
    )
    if size in QWEN_SIZES:
        hidden, layers, heads, kv_heads, inter = QWEN_SIZES[size]
        return transformers.Qwen2Config(
            hidden_size=hidden,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            intermediate_size=inter,
            rms_norm_eps=1e-6,
            use_sliding_window=False,
            **common,
        )
    hidden, layers, heads, inter = SIZES[size]
    return transformers.LlamaConfig(
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        intermediate_size=inter,
        rms_norm_eps=1e-5,
        attention_bias=False,
        mlp_bias=False,
        **common,
    )


def make_model(config: transformers.PretrainedConfig) -> transformers.PreTrainedModel:
    """
    Instantiate a randomly initialized causal LM (SDPA attention) for `config`.
    """
    config._attn_implementation = "sdpa"
    return transformers.AutoModelForCausalLM.from_config(config)


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
    if size in QWEN_SIZES:
        h, n_layers, heads, kv_heads, inter = QWEN_SIZES[size]
        kv_dim = kv_heads * (h // heads)
        attn = h * h + 2 * h * kv_dim + h * h + (h + 2 * kv_dim)  # q, k, v, o + q/k/v bias
        per_layer = attn + 3 * h * inter + 2 * h  # + SwiGLU + 2 norms
        return n_layers * per_layer + h  # + final norm
    h, n_layers, _, inter = SIZES[size]
    per_layer = 4 * h * h + 3 * h * inter + 2 * h  # attn + SwiGLU + 2 norms
    return n_layers * per_layer + h  # + final norm
