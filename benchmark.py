"""
Single-model WikiText-103 benchmark for Genesis or REX.

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
    python benchmark_genesis.py \
        --seq_len 1024 \
        --batch_size 8 \
        --total_tokens 15_000_000 \
        --log_every 100   # steps

    python benchmark_rex.py \
        --seq_len 1024 \
        --batch_size 8 \
        --total_tokens 15_000_000 \
        --log_every 100

    # Direct common entrypoint:
    python benchmark.py --model genesis

    # Each run writes a JSON artifact by default:
    #   benchmark_genesis.json / benchmark_rex.json

Install:
    pip install datasets transformers torch
    # datasets 'wikitext' needs internet on first run; cached after.
"""

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
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
def _load_wikitext(
    tokenizer,
    split="train",
    token_cache: str = "wikitext103_mistral_tokens.pt",
    tokenize_num_proc: int = None,
) -> torch.Tensor:
    from datasets import load_dataset

    cache_path = Path(token_cache) if token_cache else None
    if cache_path is not None and cache_path.exists():
        print(f"Loading token cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    ds = ds.filter(lambda ex: bool(ex["text"].strip()), num_proc=tokenize_num_proc)

    eos = tokenizer.eos_token_id

    def tokenize_batch(batch):
        ids = tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        # Match the old `eos_token.join(texts)` behavior: insert exactly one EOS
        # between non-empty WikiText lines, not after the final line.
        joined = []
        for item in ids:
            if joined:
                joined.append(eos)
            joined.extend(item)
        return {"input_ids": [joined]}

    if tokenize_num_proc is None:
        tokenize_num_proc = max(1, os.cpu_count() or 1)
    tokenized = ds.map(
        tokenize_batch,
        batched=True,
        batch_size=1024,
        num_proc=tokenize_num_proc,
        remove_columns=ds.column_names,
        desc="Tokenizing WikiText-103",
    )
    pieces = tokenized["input_ids"]
    flat = []
    for piece in pieces:
        if flat:
            flat.append(eos)
        flat.extend(piece)
    ids = torch.tensor(flat, dtype=torch.long)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ids, cache_path)
        print(f"Saved token cache: {cache_path}")
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
    warmup_steps: int,
):
    model.train()
    if compile_model:
        print(f"[{name}] torch.compile(mode='reduce-overhead') …")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    tokens_seen = 0
    timed_tokens = 0
    peak_mem = 0.0
    t0 = None
    last_timed_time = None
    step = 0
    tokens_per_batch = (seq_len - 1) * loader.batch_size
    total_steps = max(1, total_tokens // tokens_per_batch)
    warmup = max(1, total_steps // 20)
    lr_base = 0.02 if name == "Genesis" else 3e-4
    lr_min = lr_base * 0.1

    log_rows = []
    step_rows = []
    last_loss = float("nan")
    if warmup_steps <= 0:
        t0 = time.perf_counter()
        last_timed_time = t0

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

        step_start = time.perf_counter()
        input_ids = batch.to(device)
        labels = input_ids

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        last_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            mem = torch.cuda.max_memory_allocated(device) / 1e9
        else:
            mem = 0.0
        now = time.perf_counter()

        tokens_this = labels[:, 1:].numel()
        tokens_seen += tokens_this
        step += 1

        timed = step > warmup_steps
        step_elapsed = now - step_start
        elapsed = None
        tok_s = 0.0
        step_tok_s = 0.0

        if step == warmup_steps:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            peak_mem = 0.0
            timed_tokens = 0
            t0 = now
            last_timed_time = now
            step_rows.append({
                "step": step,
                "tokens_seen": tokens_seen,
                "tokens_this_step": tokens_this,
                "timed_tokens": timed_tokens,
                "loss": last_loss,
                "lr": lr,
                "step_time_s": step_elapsed,
                "step_tok_per_s": step_tok_s,
                "cumulative_tok_per_s": tok_s,
                "elapsed_timed_s": elapsed,
                "step_peak_mem_gb": mem,
                "peak_mem_gb": peak_mem,
                "timed": False,
                "warmup_boundary": True,
            })
            print(f"[{name}] warmup complete after {step} step(s); timing starts now")
            continue

        if timed:
            timed_tokens += tokens_this
            peak_mem = max(peak_mem, mem)
            elapsed = now - t0 if t0 is not None else 0.0
            tok_s = timed_tokens / elapsed if elapsed > 0 else 0.0
            if last_timed_time is not None:
                step_dt = now - last_timed_time
                step_tok_s = tokens_this / step_dt if step_dt > 0 else 0.0
            last_timed_time = now
        else:
            peak_mem = max(peak_mem, mem)

        step_rows.append({
            "step": step,
            "tokens_seen": tokens_seen,
            "tokens_this_step": tokens_this,
            "timed_tokens": timed_tokens,
            "loss": last_loss,
            "lr": lr,
            "step_time_s": step_elapsed,
            "step_tok_per_s": step_tok_s,
            "cumulative_tok_per_s": tok_s,
            "elapsed_timed_s": elapsed,
            "step_peak_mem_gb": mem,
            "peak_mem_gb": peak_mem,
            "timed": timed,
            "warmup_boundary": False,
        })

        if step % log_every == 0:
            print(
                f"[{name}] step={step:>5d}  tokens={tokens_seen/1e6:.2f}M"
                f"  loss={last_loss:.4f}  {tok_s:,.0f} tok/s"
                f"  peak_mem={peak_mem:.2f} GB"
            )
            log_rows.append({
                "step": step,
                "tokens_seen": tokens_seen,
                "loss": last_loss,
                "cumulative_tok_per_s": tok_s,
                "peak_mem_gb": peak_mem,
            })

    if t0 is None:
        t0 = time.perf_counter()
    elapsed = time.perf_counter() - t0
    return {
        "name": name,
        "total_tokens": tokens_seen,
        "elapsed_s": elapsed,
        "tok_per_s": timed_tokens / elapsed if elapsed > 0 else 0.0,
        "peak_mem_gb": peak_mem,
        "final_loss": last_loss,
        "timed_tokens": timed_tokens,
        "warmup_steps": warmup_steps,
        "total_steps": step,
        "target_steps": total_steps,
        "lr_base": lr_base,
        "lr_min": lr_min,
        "lr_warmup_steps": warmup,
        "log": log_rows,
        "steps": step_rows,
    }


# --------------------------------------------------------------------------
# Param count
# --------------------------------------------------------------------------
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def param_stats(model, model_key: str):
    total = count_params(model)
    stats = {
        "trainable_params": total,
        "trainable_params_m": total / 1e6,
    }
    if hasattr(model, "embedding"):
        stats["embedding_params"] = model.embedding.weight.numel()
    if model_key == "genesis":
        stats["lm_head_params"] = model.lm_head.weight.numel()
        stats["tied_embeddings"] = model.embedding.weight is model.lm_head.weight
    if model_key == "rex":
        stats["lm_head_params"] = model.fc_out.weight.numel()
        stats["tied_embeddings"] = model.embedding.weight is model.fc_out.weight
        dead_w3 = sum(block.ff.w3.weight.numel() for block in model.blocks)
        stats["rex_dead_w3_params"] = dead_w3
        stats["active_params_excluding_dead_w3"] = total - dead_w3
        stats["active_params_excluding_dead_w3_m"] = (total - dead_w3) / 1e6
    return stats


def build_artifact(
    *,
    args,
    name: str,
    model_key: str,
    model_config: dict,
    params: dict,
    data_tokens_available: int,
    result: dict,
):
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_key,
        "name": name,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "tokenizer": "mistralai/Mistral-7B-v0.1",
        "dataset": {
            "name": "Salesforce/wikitext",
            "config": "wikitext-103-raw-v1",
            "split": "train",
            "tokens_available": data_tokens_available,
        },
        "benchmark": {
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "requested_total_tokens": args.total_tokens,
            "log_every": args.log_every,
            "compile": args.compile,
            "compile_mode": "reduce-overhead" if args.compile else None,
            "warmup_steps": args.warmup_steps,
            "num_workers": args.num_workers,
            "token_cache": args.token_cache,
            "tokenize_num_proc": args.tokenize_num_proc,
            "device": args.device,
            "dtype": args.dtype,
        },
        "model_config": model_config,
        "param_stats": params,
        "summary": {
            "total_tokens": result["total_tokens"],
            "timed_tokens": result["timed_tokens"],
            "total_steps": result["total_steps"],
            "target_steps": result["target_steps"],
            "elapsed_timed_s": result["elapsed_s"],
            "tok_per_s": result["tok_per_s"],
            "peak_mem_gb": result["peak_mem_gb"],
            "final_loss": result["final_loss"],
            "lr_base": result["lr_base"],
            "lr_min": result["lr_min"],
            "lr_warmup_steps": result["lr_warmup_steps"],
        },
        "logs": result["log"],
        "steps": result["steps"],
    }


def write_artifact(path: str, artifact: dict):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Wrote JSON artifact: {out}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(forced_model: str = None):
    parser = argparse.ArgumentParser()
    if forced_model is None:
        parser.add_argument("--model", choices=["genesis", "rex"], default="genesis")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_tokens", type=int, default=15_000_000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no_compile", dest="compile", action="store_false")
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--token_cache", default="wikitext103_mistral_tokens.pt")
    parser.add_argument("--tokenize_num_proc", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()
    if forced_model is not None:
        args.model = forced_model
    if args.output_json is None:
        args.output_json = f"benchmark_{args.model}.json"

    device = torch.device(args.device)
    if args.dtype == "bf16":
        train_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        train_dtype = torch.float16
    else:
        train_dtype = torch.float32
    if device.type == "cpu" and train_dtype != torch.float32:
        print(f"CPU device selected; overriding dtype={args.dtype} to fp32")
        train_dtype = torch.float32
        args.dtype = "fp32"

    print("Loading Mistral tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing WikiText-103 train split…")
    ids = _load_wikitext(
        tokenizer,
        split="train",
        token_cache=args.token_cache,
        tokenize_num_proc=args.tokenize_num_proc,
    )
    print(f"  {len(ids):,} tokens available")

    dataset = ChunkedTokens(ids, args.seq_len)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(dataset, **loader_kwargs)

    results = []

    if args.model == "genesis":
        torch.cuda.empty_cache()
        model_config = dict(GENESIS_500M)
        cfg = GenesisConfig(**model_config)
        model = Genesis(cfg).to(device=device, dtype=train_dtype)
        params = param_stats(model, args.model)
        print(f"\nGenesis params: {params['trainable_params_m']:.1f}M")
        total_steps = max(1, args.total_tokens // ((args.seq_len - 1) * args.batch_size))
        opt = _make_genesis_optimizer(model, lr=0.02, total_steps=total_steps)
        r = train_one("Genesis", model, opt, loader, device,
                      args.total_tokens, args.log_every, args.compile,
                      args.seq_len, args.warmup_steps)
        artifact = build_artifact(
            args=args,
            name="Genesis",
            model_key=args.model,
            model_config=model_config,
            params=params,
            data_tokens_available=len(ids),
            result=r,
        )
        write_artifact(args.output_json, artifact)
        results.append(r)
        del model, opt
        torch.cuda.empty_cache()

    if args.model == "rex":
        torch.cuda.empty_cache()
        model_config = dict(REX_500M)
        cfg = REXConfig(**model_config)
        model = REX(cfg).to(device=device, dtype=train_dtype)
        params = param_stats(model, args.model)
        print(f"\nREX params: {params['trainable_params_m']:.1f}M")
        opt = _make_rex_optimizer(model, lr=3e-4)
        r = train_one("REX", model, opt, loader, device,
                      args.total_tokens, args.log_every, args.compile,
                      args.seq_len, args.warmup_steps)
        artifact = build_artifact(
            args=args,
            name="REX",
            model_key=args.model,
            model_config=model_config,
            params=params,
            data_tokens_available=len(ids),
            result=r,
        )
        write_artifact(args.output_json, artifact)
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
        print(f"  Final loss:   {r['final_loss']:.4f}")


if __name__ == "__main__":
    main()
