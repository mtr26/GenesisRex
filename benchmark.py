"""
Genesis vs REX — benchmark on WikiText-103.

Measures:
  1. Peak GPU memory (training, per step)
  2. Throughput (tokens/sec)
  3. Convergence speed (loss curve over 15M tokens)

Both models target ~500M parameters. Genesis uses:
  - Muon (matrices) + AdamW (embeddings / norms)
  - Chunked cross-entropy (no logit materialization)
  - torch.compile

REX uses its original AdamW + standard CE (as shipped).

Usage:
    python benchmark.py \
        --seq_len 1024 \
        --batch_size 8 \
        --total_tokens 15_000_000 \
        --log_every 100   # steps

    # or just the genesis model:
    python benchmark.py --model genesis

Install:
    pip install datasets transformers torch
    # datasets 'wikitext' needs internet on first run; cached after.
"""

import argparse
import math
import time
from typing import Iterator

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from transformers import AutoTokenizer

# ------------------------------------------------------------------ models --
from genesis import Genesis, GenesisConfig
from muon import Muon
from rexmodel import REX, REXConfig


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def _load_wikitext(tokenizer, split="train") -> torch.Tensor:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    texts = [t for t in ds["text"] if t.strip()]
    joined = tokenizer.eos_token.join(texts)
    ids = tokenizer(joined, return_tensors="pt",
                    truncation=False, add_special_tokens=False).input_ids[0]
    return ids


class ChunkedTokens(IterableDataset):
    def __init__(self, ids: torch.Tensor, seq_len: int):
        n = len(ids) // seq_len * seq_len
        self.ids = ids[:n]
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        for idx, s in enumerate(range(0, len(self.ids), self.seq_len)):
            if idx % num_workers == worker_id:
                yield self.ids[s : s + self.seq_len]


# --------------------------------------------------------------------------
# REX scaling: ~500M
# REX defaults give ~283M. Scale up to ~500M.
#   REX:     d=1152, layers=22, heads=16, kv_heads=4 -> ~497M trainable
#   Genesis: d=1344, layers=22, heads=21, kv_heads=7 -> ~501M trainable
# --------------------------------------------------------------------------
REX_500M = dict(
    vocab_size=32000,   # Mistral tokenizer
    max_len=2048,
    n_layers=22,        # 22 layers @ d=1152 -> ~497M despite dead w3 overhead
    n_heads=16,
    n_kv_heads=4,
    n_embd=1152,
    dropout=0.0,        # disable dropout for fair comparison at 15M tokens
    tie_word_embeddings=False,
)

GENESIS_500M = dict(
    vocab_size=32000,
    max_len=4096,
    n_layers=22,
    n_heads=21,
    n_kv_heads=7,
    n_embd=1344,
    hidden_dim=3968,
    sliding_window=4096,
    global_every=0,
    z_loss=0.0,         # benchmark reports comparable CE loss; enable for pretraining if needed
    logit_softcap=0.0,
    ce_chunk=4096,
    tie_word_embeddings=True,
)


# --------------------------------------------------------------------------
# Optimizers
# --------------------------------------------------------------------------
def _make_genesis_optimizer(model: Genesis, lr: float, total_steps: int):
    # Embeddings + norms + scalars -> Adam.
    embed_ids = {id(model.embedding.weight)}
    scalar_params, matrix_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or id(p) in embed_ids:
            scalar_params.append(p)
        else:
            matrix_params.append(p)

    return Muon(
        [
            {"params": matrix_params, "use_muon": True},
            {
                "params": scalar_params,
                "use_muon": False,
                "adam_lr": 3e-4,
                "adam_betas": (0.9, 0.95),
                "adam_eps": 1e-8,
                "wd": 0.1,
            },
        ],
        lr=lr, wd=0.1,
    )


def _make_rex_optimizer(model: REX, lr: float):
    return torch.optim.AdamW(model.parameters(), lr=lr,
                             betas=(0.9, 0.95), weight_decay=0.1)


def _cosine_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float):
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------
def train_one(
    name: str,
    model: torch.nn.Module,
    optimizer,
    loader: DataLoader,
    device: torch.device,
    total_tokens: int,
    log_every: int,
    compile_model: bool,
    seq_len: int,
):
    model.train()
    if compile_model:
        print(f"[{name}] torch.compile …")
        model = torch.compile(model, mode="max-autotune", fullgraph=False)

    tokens_seen = 0
    peak_mem = 0.0
    t0 = time.perf_counter()
    step = 0
    tokens_per_batch = (seq_len - 1) * loader.batch_size
    total_steps = max(1, total_tokens // tokens_per_batch)
    warmup = max(1, total_steps // 20)
    lr_base = 0.02 if name == "Genesis" else 3e-4
    lr_min = lr_base * 0.1

    log_rows = []  # (step, tokens, loss, tok_per_sec, mem_gb)

    for batch in loader:
        if tokens_seen >= total_tokens:
            break

        # LR schedule.
        lr = _cosine_lr(step, warmup, total_steps, lr_base, lr_min)
        for g in optimizer.param_groups:
            if name == "Genesis":
                if g.get("use_muon", True):
                    g["lr"] = lr
                else:
                    g["adam_lr"] = lr * 0.15   # Adam LR tracks Muon
            else:
                g["lr"] = lr

        input_ids = batch.to(device)
        labels = input_ids

        torch.cuda.reset_peak_memory_stats(device)
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        mem = torch.cuda.max_memory_allocated(device) / 1e9
        peak_mem = max(peak_mem, mem)

        tokens_this = labels[:, 1:].numel()
        tokens_seen += tokens_this
        step += 1

        if step % log_every == 0:
            elapsed = time.perf_counter() - t0
            tok_s = tokens_seen / elapsed
            print(
                f"[{name}] step={step:>5d}  tokens={tokens_seen/1e6:.2f}M"
                f"  loss={loss.item():.4f}  {tok_s:,.0f} tok/s"
                f"  peak_mem={peak_mem:.2f} GB"
            )
            log_rows.append((step, tokens_seen, loss.item(), tok_s, peak_mem))

    elapsed = time.perf_counter() - t0
    return {
        "name": name,
        "total_tokens": tokens_seen,
        "elapsed_s": elapsed,
        "tok_per_s": tokens_seen / elapsed,
        "peak_mem_gb": peak_mem,
        "log": log_rows,
    }


# --------------------------------------------------------------------------
# Param count
# --------------------------------------------------------------------------
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["both", "genesis", "rex"], default="both")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_tokens", type=int, default=15_000_000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no_compile", dest="compile", action="store_false")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("Loading Mistral tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing WikiText-103 train split…")
    ids = _load_wikitext(tokenizer, split="train")
    print(f"  {len(ids):,} tokens available")

    dataset = ChunkedTokens(ids, args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2,
                        pin_memory=True, prefetch_factor=4)

    results = []

    if args.model in ("both", "genesis"):
        torch.cuda.empty_cache()
        cfg = GenesisConfig(**GENESIS_500M)
        model = Genesis(cfg).to(device)
        print(f"\nGenesis params: {count_params(model)/1e6:.1f}M")
        total_steps = max(1, args.total_tokens // ((args.seq_len - 1) * args.batch_size))
        opt = _make_genesis_optimizer(model, lr=0.02, total_steps=total_steps)
        r = train_one("Genesis", model, opt, loader, device,
                      args.total_tokens, args.log_every, args.compile, args.seq_len)
        results.append(r)
        del model, opt
        torch.cuda.empty_cache()

    if args.model in ("both", "rex"):
        torch.cuda.empty_cache()
        cfg = REXConfig(**REX_500M)
        model = REX(cfg).to(device)
        print(f"\nREX params: {count_params(model)/1e6:.1f}M")
        opt = _make_rex_optimizer(model, lr=3e-4)
        r = train_one("REX", model, opt, loader, device,
                      args.total_tokens, args.log_every, args.compile, args.seq_len)
        results.append(r)
        del model, opt
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"\n{r['name']}")
        print(f"  Throughput:   {r['tok_per_s']:,.0f} tokens/s")
        print(f"  Peak memory:  {r['peak_mem_gb']:.2f} GB")
        final_loss = r["log"][-1][2] if r["log"] else float("nan")
        print(f"  Final loss:   {final_loss:.4f}")

    if len(results) == 2:
        g, rx = results if results[0]["name"] == "Genesis" else results[::-1]
        print(f"\n  Throughput Δ: {g['tok_per_s'] / rx['tok_per_s']:.2f}× (Genesis/REX)")
        print(f"  Memory Δ:     {rx['peak_mem_gb'] / g['peak_mem_gb']:.2f}× less (Genesis vs REX)")
        print(f"  Loss Δ:       {g['log'][-1][2] - rx['log'][-1][2]:+.4f} (Genesis - REX)")


if __name__ == "__main__":
    main()
