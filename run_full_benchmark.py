import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmark import OPTIMIZER_CHOICES


def _as_int_token_count(value: str) -> int:
    return int(value.replace("_", ""))


def _python_bool_flag(enabled: bool, true_flag: str, false_flag: str):
    return [true_flag] if enabled else [false_flag]


def _build_run_command(args, model: str, optimizer: str, output_json: Path):
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("benchmark.py")),
        "--model", model,
        "--optimizer", optimizer,
        "--seq_len", str(args.seq_len),
        "--batch_size", str(args.batch_size),
        "--total_tokens", str(args.total_tokens),
        "--log_every", str(args.log_every),
        "--warmup_steps", str(args.warmup_steps),
        "--num_workers", str(args.num_workers),
        "--token_cache", args.token_cache,
        "--tokenize_num_proc", str(args.tokenize_num_proc),
        "--device", args.device,
        "--dtype", args.dtype,
        "--muon_lr", str(args.muon_lr),
        "--adam_lr", str(args.adam_lr),
        "--min_lr_ratio", str(args.min_lr_ratio),
        "--weight_decay", str(args.weight_decay),
        "--grad_clip", str(args.grad_clip),
        "--output_json", str(output_json),
    ]
    cmd.extend(_python_bool_flag(args.compile, "--compile", "--no_compile"))
    return cmd


def _check_optional_optimizers(optimizers, skip_unavailable: bool):
    selected = list(optimizers)
    if "flash_muon" not in selected:
        return selected
    if importlib.util.find_spec("flash_muon") is not None:
        return selected
    msg = (
        "flash_muon was requested but is not installed. Install it with:\n"
        "  git clone https://github.com/nil0x9/flash-muon.git\n"
        "  pip install -e ./flash-muon\n"
    )
    if skip_unavailable:
        print(msg + "Skipping flash_muon because --skip_unavailable was set.")
        return [opt for opt in selected if opt != "flash_muon"]
    raise SystemExit(msg)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full Genesis/REX optimizer benchmark as isolated subprocesses."
    )
    parser.add_argument("--models", nargs="+", choices=["rex", "genesis"], default=["rex", "genesis"])
    parser.add_argument("--optimizers", nargs="+", choices=OPTIMIZER_CHOICES, default=list(OPTIMIZER_CHOICES))
    parser.add_argument("--out_dir", default="runs/full_benchmark")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_tokens", type=_as_int_token_count, default=15_000_000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no_compile", dest="compile", action="store_false")
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--token_cache", default="wikitext103_mistral_tokens.pt")
    parser.add_argument("--tokenize_num_proc", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--muon_lr", type=float, default=0.02)
    parser.add_argument("--adam_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true", help="Skip runs whose JSON already exists.")
    parser.add_argument("--skip_unavailable", action="store_true", help="Skip optional optimizers that are not installed.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--plot_out", default=None)
    parser.add_argument("--fail_fast", action="store_true", default=True)
    parser.add_argument("--keep_going", dest="fail_fast", action="store_false")
    args = parser.parse_args()

    optimizers = _check_optional_optimizers(args.optimizers, args.skip_unavailable)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": args.models,
        "optimizers": optimizers,
        "runs": [],
    }

    json_paths = []
    failures = []
    for model in args.models:
        for optimizer in optimizers:
            output_json = out_dir / f"{model}_{optimizer}.json"
            json_paths.append(output_json)
            cmd = _build_run_command(args, model, optimizer, output_json)
            run_entry = {
                "model": model,
                "optimizer": optimizer,
                "output_json": str(output_json),
                "command": cmd,
                "status": "pending",
            }
            manifest["runs"].append(run_entry)

            if args.resume and output_json.exists():
                print(f"Skipping existing run: {output_json}")
                run_entry["status"] = "skipped_existing"
                continue

            print("\n" + "=" * 80)
            print(f"Running {model} + {optimizer}")
            print(" ".join(cmd))
            print("=" * 80)
            if args.dry_run:
                run_entry["status"] = "dry_run"
                continue

            completed = subprocess.run(cmd, cwd=Path(__file__).parent)
            run_entry["returncode"] = completed.returncode
            if completed.returncode == 0:
                run_entry["status"] = "completed"
            else:
                run_entry["status"] = "failed"
                failures.append(run_entry)
                if args.fail_fast:
                    break
        if failures and args.fail_fast:
            break

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote manifest: {manifest_path}")

    existing_jsons = [path for path in json_paths if path.exists()]
    if not args.no_plot and existing_jsons and not failures:
        plot_out = Path(args.plot_out) if args.plot_out else out_dir / "comparison.png"
        plot_cmd = [
            sys.executable,
            str(Path(__file__).with_name("plot_benchmark_results.py")),
            *[str(path) for path in existing_jsons],
            "--out",
            str(plot_out),
        ]
        print("\n" + "=" * 80)
        print("Generating comparison plot")
        print(" ".join(plot_cmd))
        print("=" * 80)
        if not args.dry_run:
            subprocess.run(plot_cmd, cwd=Path(__file__).parent, check=True)

    if failures:
        print("\nFailed runs:")
        for failure in failures:
            print(f"  {failure['model']} + {failure['optimizer']} -> returncode {failure['returncode']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
