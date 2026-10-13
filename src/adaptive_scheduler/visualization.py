"""CSV/PNG exports for experiment summaries and timeline explanation."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def _plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise RuntimeError("PNG charts require matplotlib; install project dependencies") from exc


def gantt_chart(result, path: str | Path) -> Path:
    plt = _plt()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    names = list(dict.fromkeys(s.pid for s in result.timeline))
    y = {pid: i for i, pid in enumerate(names)}
    colors = plt.get_cmap("tab20")
    for segment in result.timeline:
        ax.barh(y[segment.pid], segment.end-segment.start, left=segment.start, height=.58, color=colors(y[segment.pid] % 20), edgecolor="white")
        if segment.end - segment.start >= 4:
            ax.text((segment.start+segment.end)/2, y[segment.pid], segment.pid, ha="center", va="center", fontsize=8)
    ax.set_yticks(list(y.values()), list(y.keys())); ax.set_xlabel("Simulation time"); ax.set_title(f"{result.algorithm} execution timeline")
    ax.grid(axis="x", alpha=.25); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return path


def comparison_charts(frame: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    plt = _plt()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("average_waiting_time", "Average waiting time", "Time units"),
        ("average_turnaround_time", "Average turnaround time", "Time units"),
        ("average_response_time", "Average response time", "Time units"),
        ("throughput", "Throughput", "Processes per time unit"),
        ("context_switches", "Context switches", "Switches"),
        ("decision_overhead_seconds", "Scheduling decision overhead", "Wall-clock seconds"),
        ("cpu_utilization_percent", "CPU utilization", "Percent"),
    ]
    outputs = []
    for col, title, ylabel in metrics:
        if col not in frame or not frame[col].notna().any():
            continue
        grouped = frame.groupby("algorithm")[col].agg(["mean", "std"]).sort_index()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        std = grouped["std"].fillna(0).to_numpy()
        mean = grouped["mean"].to_numpy()
        lower = [min(s, max(0, m)) for s, m in zip(std, mean)]
        ax.bar(grouped.index, mean, yerr=[lower, std], capsize=4, color="#4178a8")
        ax.set_title(f"{title}\nBars show mean; error bars show ±1 SD"); ax.set_ylabel(ylabel); ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=.25)
        fig.tight_layout(); p = out / f"{col}.png"; fig.savefig(p, dpi=160); plt.close(fig); outputs.append(p)
    return outputs


def reward_chart(csv_path: str | Path, output_path: str | Path) -> Path:
    plt = _plt(); data = pd.read_csv(csv_path); p = Path(output_path); p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not data.empty:
        ax.plot(data["episode"], data["episode_reward"], alpha=.35, label="Episode reward")
        if len(data) >= 5:
            ax.plot(data["episode"], data["episode_reward"].rolling(5, min_periods=1).mean(), label="5-episode mean")
    ax.set_title("PPO training reward"); ax.set_xlabel("Episode"); ax.set_ylabel("Reward (higher is better)"); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); return p
