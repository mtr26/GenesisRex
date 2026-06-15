import argparse
import json
import os
import tempfile
from pathlib import Path


def load_run(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["_path"] = str(path)
    data["_label"] = data.get("name") or data.get("model") or Path(path).stem
    return data


def step_series(run, key, timed_only=False):
    xs, ys = [], []
    for row in run.get("steps", []):
        if timed_only and not row.get("timed", False):
            continue
        value = row.get(key)
        if value is None:
            continue
        xs.append(row["tokens_seen"] / 1e6)
        ys.append(value)
    return xs, ys


def choose_baseline(runs):
    for run in runs:
        if run.get("model") == "rex":
            return run
    return runs[0]


def summary_value(run, key, default=0.0):
    return float(run.get("summary", {}).get(key, default) or default)


def annotate_bars(ax, bars, values, fmt="{:.2f}"):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="Benchmark JSON files")
    parser.add_argument("--out", default="benchmark_comparison.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    mpl_cache = Path(os.environ.get("MPLCONFIGDIR", Path(tempfile.gettempdir()) / "matplotlib"))
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", Path(tempfile.gettempdir()) / "xdg-cache"))
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [load_run(path) for path in args.runs]
    baseline = choose_baseline(runs)
    baseline_name = baseline["_label"]
    baseline_tok_s = max(summary_value(baseline, "tok_per_s"), 1e-12)
    baseline_mem = max(summary_value(baseline, "peak_mem_gb"), 1e-12)
    baseline_loss = max(summary_value(baseline, "final_loss"), 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax_loss, ax_tps, ax_mem, ax_summary = axes.ravel()

    for run in runs:
        label = run["_label"]

        xs, ys = step_series(run, "loss", timed_only=False)
        if xs:
            ax_loss.plot(xs, ys, label=label, linewidth=1.8)

        xs, ys = step_series(run, "cumulative_tok_per_s", timed_only=True)
        if xs:
            ax_tps.plot(xs, ys, label=label, linewidth=1.8)

        xs, ys = step_series(run, "peak_mem_gb", timed_only=True)
        if xs:
            ax_mem.plot(xs, ys, label=label, linewidth=1.8)

    ax_loss.set_title("Convergence")
    ax_loss.set_xlabel("Tokens seen (M)")
    ax_loss.set_ylabel("Training loss")
    ax_loss.grid(True, alpha=0.25)
    ax_loss.legend()

    ax_tps.set_title("Throughput")
    ax_tps.set_xlabel("Tokens seen (M)")
    ax_tps.set_ylabel("Cumulative timed tokens/sec")
    ax_tps.grid(True, alpha=0.25)
    ax_tps.legend()

    ax_mem.set_title("Peak Training Memory")
    ax_mem.set_xlabel("Tokens seen (M)")
    ax_mem.set_ylabel("Peak allocated memory (GB)")
    ax_mem.grid(True, alpha=0.25)
    ax_mem.legend()

    metrics = [
        ("Throughput\nvs baseline", "tok_per_s"),
        ("Memory efficiency\nvs baseline", "memory_efficiency"),
        ("Loss efficiency\nvs baseline", "loss_efficiency"),
    ]
    x = list(range(len(metrics)))
    n = max(1, len(runs))
    width = min(0.8 / n, 0.35)

    for i, run in enumerate(runs):
        tok_ratio = summary_value(run, "tok_per_s") / baseline_tok_s
        mem_eff = baseline_mem / max(summary_value(run, "peak_mem_gb"), 1e-12)
        loss_eff = baseline_loss / max(summary_value(run, "final_loss"), 1e-12)
        values = [tok_ratio, mem_eff, loss_eff]
        offset = (i - (n - 1) / 2) * width
        bars = ax_summary.bar(
            [pos + offset for pos in x],
            values,
            width=width,
            label=run["_label"],
        )
        annotate_bars(ax_summary, bars, values)

    ax_summary.axhline(1.0, color="black", linewidth=1, alpha=0.35)
    ax_summary.set_title(f"Summary Ratios (baseline: {baseline_name})")
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels([m[0] for m in metrics])
    ax_summary.set_ylabel("Ratio, higher is better")
    ax_summary.grid(True, axis="y", alpha=0.25)
    ax_summary.legend()

    subtitle = []
    for run in runs:
        s = run.get("summary", {})
        p = run.get("param_stats", {})
        subtitle.append(
            f"{run['_label']}: "
            f"{s.get('tok_per_s', 0):,.0f} tok/s, "
            f"{s.get('peak_mem_gb', 0):.2f} GB, "
            f"loss {s.get('final_loss', 0):.4f}, "
            f"{p.get('trainable_params_m', 0):.1f}M params"
        )
    fig.suptitle("Genesis vs REX Benchmark\n" + "\n".join(subtitle), fontsize=12)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved comparison plot: {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
