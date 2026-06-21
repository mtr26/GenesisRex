"""
Single-model WikiText-103 benchmark for Genesis or REX.

Measures:
  1. Peak GPU memory (training, per step)
  2. Throughput (tokens/sec)
  3. Convergence speed (loss curve over 15M tokens)

Both models target ~500M parameters. Optimizer arms are selectable:
  - adamw:       AdamW over all trainable params
  - local_muon:  local Muon for hidden matrices + AdamW for embeddings / norms / heads
  - flash_muon:  nil0x9/flash-muon for hidden matrices + AdamW fallback

Genesis uses:
  - Chunked cross-entropy (no logit materialization)
  - torch.compile

REX uses standard CE (as shipped). REX's dead w3 weights are excluded from
Muon arms because they never receive gradients.

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
    python benchmark.py --model genesis --optimizer local_muon

    # Full isolated model x optimizer matrix:
    python run_full_benchmark.py --out_dir runs/full_benchmark

    # Each run writes a JSON artifact by default:
    #   benchmark_<model>_<optimizer>.json

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
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass
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
OPTIMIZER_CHOICES = ("adamw", "local_muon", "flash_muon")


@dataclass
class OptimizerSplit:
    muon_params: list
    adam_params: list
    excluded_params: list
    muon_param_count: int
    adam_param_count: int
    excluded_param_count: int


class OptimizerBundle:
    """Small adapter for recipes that use separate Muon and AdamW optimizers."""

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)

    @property
    def param_groups(self):
        groups = []
        for optimizer in self.optimizers:
            groups.extend(optimizer.param_groups)
        return groups

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self, *args, **kwargs):
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)


def _param_count(params) -> int:
    return sum(p.numel() for p in params)


def _module_param_ids(module) -> set:
    if module is None:
        return set()
    return {id(p) for p in module.parameters()}


def _rex_dead_w3_param_ids(model) -> set:
    if not hasattr(model, "blocks"):
        return set()
    ids = set()
    for block in model.blocks:
        ff = getattr(block, "ff", None)
        w3 = getattr(ff, "w3", None)
        if w3 is not None:
            ids.update(id(p) for p in w3.parameters())
    return ids


def split_optimizer_params(model: torch.nn.Module, model_key: str) -> OptimizerSplit:
    """Split trainable params into hidden matrices for Muon and Adam fallback params."""

    adam_param_ids = set()
    for attr in ("embedding", "lm_head", "fc_out"):
        adam_param_ids.update(_module_param_ids(getattr(model, attr, None)))

    excluded_ids = _rex_dead_w3_param_ids(model) if model_key == "rex" else set()

    muon_params, adam_params, excluded_params = [], [], []
    seen = set()
    for _, p in model.named_parameters():
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        if not p.requires_grad:
            continue
        if pid in excluded_ids:
            excluded_params.append(p)
        elif p.ndim >= 2 and pid not in adam_param_ids:
            muon_params.append(p)
        else:
            adam_params.append(p)

    return OptimizerSplit(
        muon_params=muon_params,
        adam_params=adam_params,
        excluded_params=excluded_params,
        muon_param_count=_param_count(muon_params),
        adam_param_count=_param_count(adam_params),
        excluded_param_count=_param_count(excluded_params),
    )


def _tag_group(group: dict, *, role: str, base_lr: float, min_lr: float, lr_key: str = "lr"):
    group["optimizer_role"] = role
    group["schedule_role"] = role
    group["schedule_lr_key"] = lr_key
    group["schedule_base_lr"] = base_lr
    group["schedule_min_lr"] = min_lr
    group[lr_key] = base_lr
    if lr_key != "lr":
        group["lr"] = base_lr


def _tag_optimizer_groups(optimizer, *, role: str, base_lr: float, min_lr: float, lr_key: str = "lr"):
    for group in optimizer.param_groups:
        _tag_group(group, role=role, base_lr=base_lr, min_lr=min_lr, lr_key=lr_key)


def _import_flash_muon():
    try:
        from flash_muon import Muon as FlashMuon
    except ImportError as exc:
        raise RuntimeError(
            "flash_muon optimizer requested but the package is not installed. "
            "Install with: git clone https://github.com/nil0x9/flash-muon.git && "
            "pip install -e flash-muon/"
        ) from exc
    return FlashMuon


def _ensure_single_process_group(device: torch.device):
    import torch.distributed as dist

    if not dist.is_available() or dist.is_initialized():
        return
    backend = "nccl" if device.type == "cuda" and dist.is_nccl_available() else "gloo"
    init_file = tempfile.NamedTemporaryFile(prefix="flash_muon_pg_", delete=False)
    init_file.close()
    dist.init_process_group(
        backend=backend,
        rank=0,
        world_size=1,
        init_method=f"file://{init_file.name}",
    )


def make_optimizer(
    model: torch.nn.Module,
    *,
    model_key: str,
    optimizer_key: str,
    muon_lr: float,
    adam_lr: float,
    weight_decay: float,
    device: torch.device,
    min_lr_ratio: float = 0.1,
):
    if optimizer_key not in OPTIMIZER_CHOICES:
        raise ValueError(f"Unknown optimizer '{optimizer_key}'. Choices: {OPTIMIZER_CHOICES}")

    muon_min_lr = muon_lr * min_lr_ratio
    adam_min_lr = adam_lr * min_lr_ratio

    if optimizer_key == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=adam_lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )
        _tag_optimizer_groups(optimizer, role="adam", base_lr=adam_lr, min_lr=adam_min_lr)
        total = count_params(model)
        return optimizer, {
            "optimizer": optimizer_key,
            "muon_param_count": 0,
            "adam_param_count": total,
            "excluded_param_count": 0,
            "muon_param_count_m": 0.0,
            "adam_param_count_m": total / 1e6,
            "excluded_param_count_m": 0.0,
        }

    split = split_optimizer_params(model, model_key=model_key)
    if not split.muon_params:
        raise RuntimeError(f"{optimizer_key} requested but no hidden matrix parameters were found")

    if optimizer_key == "local_muon":
        optimizer = Muon(
            [
                {
                    "params": split.muon_params,
                    "use_muon": True,
                    "lr": muon_lr,
                },
                {
                    "params": split.adam_params,
                    "use_muon": False,
                    "lr": adam_lr,
                    "adam_lr": adam_lr,
                    "adam_betas": (0.9, 0.95),
                    "adam_eps": 1e-8,
                    "wd": weight_decay,
                },
            ],
            lr=muon_lr,
            wd=weight_decay,
        )
        _tag_group(
            optimizer.param_groups[0],
            role="muon",
            base_lr=muon_lr,
            min_lr=muon_min_lr,
        )
        _tag_group(
            optimizer.param_groups[1],
            role="adam_aux",
            base_lr=adam_lr,
            min_lr=adam_min_lr,
            lr_key="adam_lr",
        )
    else:
        if device.type != "cuda":
            raise RuntimeError("flash_muon requires CUDA because its optimizer allocates CUDA update buffers")
        FlashMuon = _import_flash_muon()
        if device.index is not None:
            torch.cuda.set_device(device)
        _ensure_single_process_group(device)
        muon_optimizer = FlashMuon(
            split.muon_params,
            lr=muon_lr,
            weight_decay=weight_decay,
            momentum=0.95,
            rank=0,
            world_size=1,
        )
        adam_optimizer = torch.optim.AdamW(
            split.adam_params,
            lr=adam_lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )
        _tag_optimizer_groups(muon_optimizer, role="muon", base_lr=muon_lr, min_lr=muon_min_lr)
        _tag_optimizer_groups(adam_optimizer, role="adam_aux", base_lr=adam_lr, min_lr=adam_min_lr)
        optimizer = OptimizerBundle([muon_optimizer, adam_optimizer])

    return optimizer, {
        "optimizer": optimizer_key,
        "muon_param_count": split.muon_param_count,
        "adam_param_count": split.adam_param_count,
        "excluded_param_count": split.excluded_param_count,
        "muon_param_count_m": split.muon_param_count / 1e6,
        "adam_param_count_m": split.adam_param_count / 1e6,
        "excluded_param_count_m": split.excluded_param_count / 1e6,
    }


def _set_optimizer_lrs(optimizer, step: int, warmup: int, total_steps: int):
    snapshot = {}
    for idx, group in enumerate(optimizer.param_groups):
        base_lr = group.get("schedule_base_lr", group["lr"])
        min_lr = group.get("schedule_min_lr", base_lr * 0.1)
        lr_key = group.get("schedule_lr_key", "lr")
        lr = _cosine_lr(step, warmup, total_steps, base_lr, min_lr)
        group[lr_key] = lr
        if lr_key != "lr":
            group["lr"] = lr
        role = group.get("schedule_role", f"group_{idx}")
        if role in snapshot:
            role = f"{role}_{idx}"
        snapshot[role] = lr
    return snapshot


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
    grad_clip: float,
):
    model.train()
    if compile_model:
        print(f"[{name}] torch.compile(mode='reduce-overhead') …")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    tokens_seen = 0
    timed_tokens = 0
    peak_mem = 0.0
    peak_reserved_mem = 0.0
    t0 = None
    last_timed_time = None
    step = 0
    tokens_per_batch = (seq_len - 1) * loader.batch_size
    total_steps = max(1, total_tokens // tokens_per_batch)
    warmup = max(1, total_steps // 20)

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
        lrs = _set_optimizer_lrs(optimizer, step, warmup, total_steps)
        primary_lr = next(iter(lrs.values())) if lrs else 0.0

        step_start = time.perf_counter()
        input_ids = batch.to(device)
        labels = input_ids

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        last_loss = loss.item()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            mem = torch.cuda.max_memory_allocated(device) / 1e9
            reserved_mem = torch.cuda.max_memory_reserved(device) / 1e9
        else:
            mem = 0.0
            reserved_mem = 0.0
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
            peak_reserved_mem = 0.0
            timed_tokens = 0
            t0 = now
            last_timed_time = now
            step_rows.append({
                "step": step,
                "tokens_seen": tokens_seen,
                "tokens_this_step": tokens_this,
                "timed_tokens": timed_tokens,
                "loss": last_loss,
                "lr": primary_lr,
                "lrs": lrs,
                "step_time_s": step_elapsed,
                "step_tok_per_s": step_tok_s,
                "cumulative_tok_per_s": tok_s,
                "elapsed_timed_s": elapsed,
                "step_peak_mem_gb": mem,
                "step_peak_reserved_mem_gb": reserved_mem,
                "peak_mem_gb": peak_mem,
                "peak_reserved_mem_gb": peak_reserved_mem,
                "timed": False,
                "warmup_boundary": True,
            })
            print(f"[{name}] warmup complete after {step} step(s); timing starts now")
            continue

        if timed:
            timed_tokens += tokens_this
            peak_mem = max(peak_mem, mem)
            peak_reserved_mem = max(peak_reserved_mem, reserved_mem)
            elapsed = now - t0 if t0 is not None else 0.0
            tok_s = timed_tokens / elapsed if elapsed > 0 else 0.0
            if last_timed_time is not None:
                step_dt = now - last_timed_time
                step_tok_s = tokens_this / step_dt if step_dt > 0 else 0.0
            last_timed_time = now
        else:
            peak_mem = max(peak_mem, mem)
            peak_reserved_mem = max(peak_reserved_mem, reserved_mem)

        step_rows.append({
            "step": step,
            "tokens_seen": tokens_seen,
            "tokens_this_step": tokens_this,
            "timed_tokens": timed_tokens,
            "loss": last_loss,
            "lr": primary_lr,
            "lrs": lrs,
            "step_time_s": step_elapsed,
            "step_tok_per_s": step_tok_s,
            "cumulative_tok_per_s": tok_s,
            "elapsed_timed_s": elapsed,
            "step_peak_mem_gb": mem,
            "step_peak_reserved_mem_gb": reserved_mem,
            "peak_mem_gb": peak_mem,
            "peak_reserved_mem_gb": peak_reserved_mem,
            "timed": timed,
            "warmup_boundary": False,
        })

        if step % log_every == 0:
            print(
                f"[{name}] step={step:>5d}  tokens={tokens_seen/1e6:.2f}M"
                f"  loss={last_loss:.4f}  {tok_s:,.0f} tok/s"
                f"  peak_mem={peak_mem:.2f} GB"
                f"  reserved={peak_reserved_mem:.2f} GB"
            )
            log_rows.append({
                "step": step,
                "tokens_seen": tokens_seen,
                "loss": last_loss,
                "cumulative_tok_per_s": tok_s,
                "peak_mem_gb": peak_mem,
                "peak_reserved_mem_gb": peak_reserved_mem,
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
        "peak_reserved_mem_gb": peak_reserved_mem,
        "final_loss": last_loss,
        "timed_tokens": timed_tokens,
        "warmup_steps": warmup_steps,
        "total_steps": step,
        "target_steps": total_steps,
        "lr_bases": {
            group.get("schedule_role", f"group_{i}"): group.get("schedule_base_lr", group["lr"])
            for i, group in enumerate(optimizer.param_groups)
        },
        "lr_mins": {
            group.get("schedule_role", f"group_{i}"): group.get("schedule_min_lr", group["lr"])
            for i, group in enumerate(optimizer.param_groups)
        },
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
    optimizer_metadata: dict,
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
            "optimizer": args.optimizer,
            "muon_lr": args.muon_lr,
            "adam_lr": args.adam_lr,
            "min_lr_ratio": args.min_lr_ratio,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
        },
        "model_config": model_config,
        "param_stats": params,
        "optimizer": optimizer_metadata,
        "summary": {
            "total_tokens": result["total_tokens"],
            "timed_tokens": result["timed_tokens"],
            "total_steps": result["total_steps"],
            "target_steps": result["target_steps"],
            "elapsed_timed_s": result["elapsed_s"],
            "tok_per_s": result["tok_per_s"],
            "peak_mem_gb": result["peak_mem_gb"],
            "peak_reserved_mem_gb": result["peak_reserved_mem_gb"],
            "final_loss": result["final_loss"],
            "lr_bases": result["lr_bases"],
            "lr_mins": result["lr_mins"],
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
    parser.add_argument("--optimizer", choices=OPTIMIZER_CHOICES, default=None)
    parser.add_argument("--muon_lr", type=float, default=0.02)
    parser.add_argument("--adam_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()
    if forced_model is not None:
        args.model = forced_model
    if args.optimizer is None:
        args.optimizer = "local_muon" if args.model == "genesis" else "adamw"
    if args.output_json is None:
        args.output_json = f"benchmark_{args.model}_{args.optimizer}.json"

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
        opt, opt_metadata = make_optimizer(
            model,
            model_key=args.model,
            optimizer_key=args.optimizer,
            muon_lr=args.muon_lr,
            adam_lr=args.adam_lr,
            weight_decay=args.weight_decay,
            device=device,
            min_lr_ratio=args.min_lr_ratio,
        )
        name = f"Genesis + {args.optimizer}"
        r = train_one(name, model, opt, loader, device,
                      args.total_tokens, args.log_every, args.compile,
                      args.seq_len, args.warmup_steps, args.grad_clip)
        artifact = build_artifact(
            args=args,
            name=name,
            model_key=args.model,
            model_config=model_config,
            params=params,
            optimizer_metadata=opt_metadata,
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
        opt, opt_metadata = make_optimizer(
            model,
            model_key=args.model,
            optimizer_key=args.optimizer,
            muon_lr=args.muon_lr,
            adam_lr=args.adam_lr,
            weight_decay=args.weight_decay,
            device=device,
            min_lr_ratio=args.min_lr_ratio,
        )
        name = f"REX + {args.optimizer}"
        r = train_one(name, model, opt, loader, device,
                      args.total_tokens, args.log_every, args.compile,
                      args.seq_len, args.warmup_steps, args.grad_clip)
        artifact = build_artifact(
            args=args,
            name=name,
            model_key=args.model,
            model_config=model_config,
            params=params,
            optimizer_metadata=opt_metadata,
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
        print(f"  Reserved mem: {r['peak_reserved_mem_gb']:.2f} GB")
        print(f"  Final loss:   {r['final_loss']:.4f}")


if __name__ == "__main__":
    main()
