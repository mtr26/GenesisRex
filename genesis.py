"""
Genesis — REX v2. A ~500M agentic-first SLM, built from scratch.

Design goals vs REX (decoder, GQA+RoPE, RMSNorm, "SwiGLU"):
  - Convergence:  Muon optimizer (see muon.py), *real* SwiGLU (REX's w3 was dead),
                  QK-norm, z-loss + logit soft-cap, no dropout.
  - Memory:       Cut-Cross-Entropy-style chunked loss (never materialize [N, vocab]
                  logits), tied embeddings, Muon's single momentum state.
  - Throughput:   fused-friendly module boundaries, SDPA/flash, torch.compile-clean.
  - Long ctx / agentic / KV cache:  local sliding-window RoPE layers + periodic
                  NoPE full-attention "global" layers. NoPE globals make KV caching
                  correct *by construction* (no rotary offset bookkeeping — the bug
                  that forced REX to disable its cache).

HF-compatible (PretrainedConfig / PreTrainedModel / GenerationMixin) so it drops
straight into an existing REX training stack.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class GenesisConfig(PretrainedConfig):
    model_type = "genesis"

    def __init__(
        self,
        vocab_size: int = 32000,          # Mistral 7B tokenizer
        max_len: int = 4096,
        n_layers: int = 22,
        n_heads: int = 21,                # head_dim = 1344 / 21 = 64
        n_kv_heads: int = 7,              # GQA 3:1
        n_embd: int = 1344,
        hidden_dim: int = 3968,           # SwiGLU inner (~3x d)
        sliding_window: int = 4096,       # window for local layers
        global_every: int = 0,            # 0 = RoPE every layer; N = every Nth layer NoPE
        rope_theta: float = 10000.0,
        z_loss: float = 1e-4,             # logit z-loss coefficient (0 disables)
        logit_softcap: float = 0.0,       # 0 disables; ~30 enables Gemma2-style cap
        ce_chunk: int = 4096,             # tokens per chunk in chunked CE
        norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_embd = n_embd
        self.hidden_dim = hidden_dim
        self.sliding_window = sliding_window
        self.global_every = global_every
        self.rope_theta = rope_theta
        self.z_loss = z_loss
        self.logit_softcap = logit_softcap
        self.ce_chunk = ce_chunk
        self.norm_eps = norm_eps
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


# ---------------------------------------------------------------------------
# Rotary embeddings (clean; positions are absolute via cache_position)
# ---------------------------------------------------------------------------
class Rotary(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor, dtype: torch.dtype):
        # positions: [T] absolute positions -> cos/sin: [T, head_dim]
        freqs = torch.outer(positions.float(), self.inv_freq)      # [T, hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)                    # [T, hd]
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: [B, H, T, hd]; cos,sin: [T, hd]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


# ---------------------------------------------------------------------------
# Attention: GQA + QK-norm. Local layers = RoPE + sliding window.
#            Global layers (every Nth) = NoPE + full causal (cache-trivial).
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, config: GenesisConfig, is_global: bool):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.n_embd // config.n_heads
        self.is_global = is_global
        self.window = None if is_global else config.sliding_window
        self.use_rope = not is_global
        kv_dim = self.head_dim * self.n_kv_heads

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, kv_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # QK-norm: per-head RMSNorm over head_dim (stability -> higher LR).
        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        # QK-norm before rotary.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # -> [B, H, T, hd]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_rope:
            q, k = apply_rope(q, k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None

        q_len, kv_len = q.size(2), k.size(2)
        if q_len == kv_len:
            # Training / prefill: causal (+ optional sliding window).
            if self.window is not None and kv_len > self.window:
                attn_mask = _sliding_causal_mask(q_len, kv_len, self.window, x.device)
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, scale=self.scale, enable_gqa=True
                )
            else:
                out = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=self.scale, enable_gqa=True
                )
        else:
            # Decode (q_len==1): the query is the newest token, attends to all
            # cached keys -> no causal mask needed. For local layers, slice the
            # attention view but keep the stored cache untrimmed so RoPE offsets
            # remain recoverable without an external cache_position.
            attn_k, attn_v = k, v
            if self.window is not None and kv_len > self.window:
                attn_k = k[:, :, -self.window:, :]
                attn_v = v[:, :, -self.window:, :]
            out = F.scaled_dot_product_attention(
                q, attn_k, attn_v, is_causal=False, scale=self.scale, enable_gqa=True
            )

        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out), new_kv


def _sliding_causal_mask(q_len, kv_len, window, device):
    # Boolean additive mask path (only used when seq > window). Short-seq
    # benchmark never hits this; long-ctx pretraining should swap in FlexAttention.
    i = torch.arange(q_len, device=device)[:, None] + (kv_len - q_len)
    j = torch.arange(kv_len, device=device)[None, :]
    keep = (j <= i) & (j > i - window)
    return keep[None, None, :, :]


# ---------------------------------------------------------------------------
# SwiGLU MLP (the real thing — gate * up, no dropout)
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# Block (pre-norm)
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config: GenesisConfig, layer_idx: int):
        super().__init__()
        is_global = config.global_every > 0 and ((layer_idx + 1) % config.global_every == 0)
        self.attn = Attention(config, is_global=is_global)
        self.mlp = SwiGLU(config.n_embd, config.hidden_dim)
        self.ln_attn = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.ln_mlp = RMSNorm(config.n_embd, eps=config.norm_eps)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        h, new_kv = self.attn(self.ln_attn(x), cos, sin, past_kv, use_cache)
        x = x + h
        x = x + self.mlp(self.ln_mlp(x))
        return x, new_kv


# ---------------------------------------------------------------------------
# Cut-Cross-Entropy-style chunked loss.
# Never materializes the full [N, vocab] logit tensor: streams chunks in the
# forward and *recomputes* them in the backward. Supports z-loss + soft-cap.
# (Pure-torch reference; a Triton port is the drop-in throughput upgrade.)
# ---------------------------------------------------------------------------
class _ChunkedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, chunk, z_coef, softcap):
        # hidden: [N, d] (already gathered to valid tokens), weight: [V, d]
        N = hidden.size(0)
        loss = hidden.new_zeros(())
        lse_store = hidden.new_empty(N)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            z = hidden[s:e] @ weight.t()                      # [c, V]
            if softcap > 0:
                z = softcap * torch.tanh(z / softcap)
            lse = torch.logsumexp(z, dim=-1)                  # [c]
            tgt = z.gather(1, labels[s:e, None]).squeeze(1)   # [c]
            loss = loss + (lse - tgt).sum()
            if z_coef > 0:
                loss = loss + z_coef * (lse * lse).sum()
            lse_store[s:e] = lse
        loss = loss / N
        ctx.save_for_backward(hidden, weight, labels, lse_store)
        ctx.chunk, ctx.z_coef, ctx.softcap, ctx.N = chunk, z_coef, softcap, N
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, labels, lse_store = ctx.saved_tensors
        chunk, z_coef, softcap, N = ctx.chunk, ctx.z_coef, ctx.softcap, ctx.N
        g = grad_out / N
        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            h = hidden[s:e]
            z = h @ weight.t()                                # [c, V]
            if softcap > 0:
                t = torch.tanh(z / softcap)
                z = softcap * t
            probs = torch.softmax(z, dim=-1)                  # [c, V]
            # d(CE)/dz = softmax - onehot ; plus z-loss term.
            grad_z = probs.clone()
            grad_z.scatter_add_(1, labels[s:e, None],
                                torch.full_like(labels[s:e, None], -1.0, dtype=z.dtype))
            if z_coef > 0:
                grad_z = grad_z + (2.0 * z_coef) * lse_store[s:e, None] * probs
            if softcap > 0:
                grad_z = grad_z * (1.0 - t * t)               # chain through tanh cap
            grad_z = grad_z * g
            grad_hidden[s:e] = grad_z @ weight
            grad_weight += grad_z.t() @ h
        return grad_hidden, grad_weight, None, None, None, None


def chunked_cross_entropy(hidden, weight, labels, chunk=4096, z_coef=0.0,
                          softcap=0.0, ignore_index=-100):
    """hidden: [..., d], labels: [...] -> scalar loss, no full-logit materialization."""
    hidden = hidden.reshape(-1, hidden.size(-1))
    labels = labels.reshape(-1)
    valid = labels != ignore_index
    if not torch.all(valid):
        hidden = hidden[valid]
        labels = labels[valid]
    if hidden.numel() == 0:
        return hidden.new_zeros((), requires_grad=True)
    return _ChunkedCE.apply(hidden, weight, labels, chunk, z_coef, softcap)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Genesis(PreTrainedModel, GenerationMixin):
    config_class = GenesisConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: GenesisConfig):
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.rotary = Rotary(config.n_embd // config.n_heads, config.rope_theta)
        self.blocks = nn.ModuleList(Block(config, i) for i in range(config.n_layers))
        self.ln_f = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.post_init()
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, new):
        self.embedding = new

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)
        # Scaled init on residual-output projections (GPT-2 / REX convention).
        for name, p in module.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down.weight"):
                init.normal_(p, mean=0.0, std=0.02 / (2 * self.config.n_layers) ** 0.5)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        attention_mask=None,
        position_ids=None,
        cache_position=None,
        inputs_embeds=None,
        return_dict: bool = True,
    ) -> CausalLMOutputWithPast:
        x = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        B, T, _ = x.shape

        if self.gradient_checkpointing and self.training:
            use_cache = False

        if position_ids is not None:
            positions = position_ids[0].to(device=x.device)
        elif cache_position is not None:
            positions = cache_position.to(device=x.device)
        else:
            past_len = 0
            if past_key_values is not None:
                # Use the longest stored cache as the absolute decode position.
                # Local layers slice the attention view, not the stored K/V, so
                # this stays valid even after the sliding window is exceeded.
                lengths = [kv[0].size(2) for kv in past_key_values if kv is not None]
                past_len = max(lengths) if lengths else 0
            positions = torch.arange(past_len, past_len + T, device=x.device)
        cos, sin = self.rotary(positions, x.dtype)

        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)
        new_past = [] if use_cache else None

        for blk, past_kv in zip(self.blocks, past_key_values):
            if self.gradient_checkpointing and self.training:
                x, new_kv = torch.utils.checkpoint.checkpoint(
                    blk, x, cos, sin, None, False, use_reentrant=False
                )
            else:
                x, new_kv = blk(x, cos, sin, past_kv, use_cache)
            if use_cache:
                new_past.append(new_kv)

        x = self.ln_f(x)

        loss = None
        logits = None
        if labels is not None:
            # Memory-frugal path: shift, then chunked CE straight from hidden states.
            shift_h = x[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = chunked_cross_entropy(
                shift_h, self.lm_head.weight, shift_labels,
                chunk=self.config.ce_chunk,
                z_coef=self.config.z_loss,
                softcap=self.config.logit_softcap,
            )
        else:
            logits = self.lm_head(x)
            if self.config.logit_softcap > 0:
                cap = self.config.logit_softcap
                logits = cap * torch.tanh(logits / cap)

        if not return_dict:
            return (loss, logits, new_past)
        return CausalLMOutputWithPast(
            loss=loss, logits=logits,
            past_key_values=new_past if use_cache else None,
        )
