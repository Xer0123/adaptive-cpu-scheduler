"""Reproducible shared-workload comparisons with explicit failure records."""
from __future__ import annotations

import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import numpy as np
import pandas as pd
from .algorithms import fcfs, sjf, priority_scheduling, round_robin
from .metrics import calculate_metrics
from .models import Process
from .workloads import generate_workload, save_workload

BASELINES = ("FCFS", "SJF", "Priority", "Round Robin")


def run_experiments(output_dir: str | Path, *, scenarios: list[str], seeds: list[int], n: int = 30,
                    quantum: float = 3.0, model_path: str | Path | None = None,
                    split: str = "test") -> tuple[pd.DataFrame, pd.DataFrame]:
    if split not in {"demo", "validation", "test"}:
        raise ValueError("split must be one of: demo, validation, test")
    out = Path(output_dir); workloads_dir = out / "workloads"; process_dir = out / "process_results"
    workloads_dir.mkdir(parents=True, exist_ok=True); process_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in scenarios:
        for seed in seeds:
            ps = generate_workload(n, scenario, seed)
            workload_path = workloads_dir / f"{scenario}_seed{seed}.csv"
            save_workload(ps, workload_path, seed, scenario)
            runs = [("FCFS", lambda: fcfs(ps)), ("SJF", lambda: sjf(ps)),
                    ("Priority", lambda: priority_scheduling(ps)),
                    ("Round Robin", lambda: round_robin(ps, quantum))]
            if model_path is not None:
                from .ppo import evaluate_ppo
                runs.append(("PPO", lambda: evaluate_ppo(ps, model_path, deterministic=True)))
            for algorithm, run in runs:
                try:
                    result = run()
                    record = {**calculate_metrics(result), "split": split, "scenario": scenario, "seed": seed, "status": "ok", "error": "", "workload_file": str(workload_path)}
                    proc_rows = []
                    for p in result.processes:
                        proc_rows.append({"pid": p.pid, "arrival_time": p.arrival_time, "burst_time": p.burst_time,
                                          "priority": p.priority, "first_execution_time": p.first_execution_time,
                                          "completion_time": p.completion_time, "waiting_time": p.waiting_time,
                                          "turnaround_time": p.turnaround_time, "response_time": p.response_time})
                    pd.DataFrame(proc_rows).to_csv(process_dir / f"{scenario}_seed{seed}_{algorithm.lower().replace(' ', '-')}.csv", index=False)
                except Exception as exc:  # Failed runs remain visible in raw results.
                    record = {"algorithm": algorithm, "split": split, "scenario": scenario, "seed": seed, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "workload_file": str(workload_path)}
                rows.append(record)
    raw = pd.DataFrame(rows)
    raw_path = out / "per_run_results.csv"; raw.to_csv(raw_path, index=False)
    metric_cols = ["average_waiting_time", "average_turnaround_time", "average_response_time", "throughput", "context_switches", "decision_overhead_seconds", "cpu_utilization_percent"]
    successful = raw[raw.status == "ok"].copy() if not raw.empty else raw
    agg_rows = []
    if not successful.empty:
        for (scenario, algorithm), group in successful.groupby(["scenario", "algorithm"], sort=True):
            row = {"scenario": scenario, "algorithm": algorithm, "successful_runs": len(group)}
            for col in metric_cols:
                row[f"{col}_mean"] = group[col].mean()
                row[f"{col}_std"] = group[col].std(ddof=1) if len(group) > 1 else 0.0
            agg_rows.append(row)
    summary = pd.DataFrame(agg_rows)
    summary.to_csv(out / "aggregate_results.csv", index=False)
    if len(successful):
        # Overall summary keeps every tested scenario and seed equally represented.
        overall = successful.groupby("algorithm")[metric_cols].agg(["mean", "std"])
        overall.to_csv(out / "overall_summary.csv")
    packages = {}
    for package in ("numpy", "pandas", "matplotlib", "gymnasium", "stable-baselines3", "torch"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    (out / "run_metadata.json").write_text(json.dumps({"python_version": platform.python_version(), "platform": platform.platform(),
        "package_versions": packages, "split": split, "scenarios": scenarios, "seeds": seeds, "process_count": n,
        "round_robin_quantum": quantum, "model_path": str(model_path) if model_path else None,
        "timing_method": "time.perf_counter_ns around scheduling selection only; simulated CPU time excluded",
        "context_switch_rule": "count transition between different process IDs when timeline segments are contiguous; exclude initial dispatch and idle transitions"}, indent=2), encoding="utf-8")
    return raw, summary
