"""Reproducible synthetic workload scenarios."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from .models import Process

SCENARIOS = ("uniform", "mixed", "bursty", "priority", "cpu_intensive")


def _sample(rng: np.random.Generator, spec: dict, n: int, integer: bool) -> np.ndarray:
    kind = spec.get("kind", "uniform")
    if kind == "uniform":
        low, high = spec.get("low", 0), spec.get("high", 10)
        values = rng.integers(low, high, n) if integer else rng.uniform(low, high, n)
    elif kind == "normal":
        values = rng.normal(spec.get("mean", 5), spec.get("std", 1), n)
        values = np.maximum(values, spec.get("minimum", 0))
    elif kind == "poisson":
        values = rng.poisson(spec.get("lam", 5), n)
    elif kind == "categorical":
        values = rng.choice(spec["values"], n, p=spec.get("probabilities"))
    elif kind == "fixed":
        values = np.full(n, spec["value"])
    else:
        raise ValueError(f"unsupported distribution kind {kind!r}")
    if integer:
        values = np.rint(values).astype(int)
    return values


def generate_workload(n: int = 30, scenario: str = "mixed", seed: int = 1, *,
                      arrival_distribution: dict | None = None,
                      burst_distribution: dict | None = None,
                      priority_distribution: dict | None = None,
                      short_job_fraction: float | None = None) -> list[Process]:
    if n < 1:
        raise ValueError("n must be at least 1")
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {', '.join(SCENARIOS)}")
    rng = np.random.default_rng(seed)
    if scenario == "uniform":
        arrivals = np.sort(rng.integers(0, max(2, n // 5), n))
        bursts = rng.integers(4, 9, n)
        priorities = rng.integers(1, 6, n)
    elif scenario == "mixed":
        arrivals = np.sort(rng.integers(0, max(2, n // 3), n))
        fraction = 0.7 if short_job_fraction is None else short_job_fraction
        if not 0 <= fraction <= 1:
            raise ValueError("short_job_fraction must be between 0 and 1")
        short = rng.random(n) < fraction
        bursts = np.where(short, rng.integers(1, 6, n), rng.integers(12, 36, n))
        priorities = rng.integers(1, 6, n)
    elif scenario == "bursty":
        centers = rng.choice([0, max(1, n // 2), n], size=n, p=[0.55, 0.30, 0.15])
        arrivals = np.maximum(0, centers + rng.integers(-1, 2, n))
        bursts = rng.integers(1, 18, n)
        priorities = rng.integers(1, 6, n)
    elif scenario == "priority":
        arrivals = np.sort(rng.integers(0, max(2, n // 4), n))
        bursts = rng.integers(2, 18, n)
        priorities = rng.choice([0, 1, 2, 3, 4], n, p=[0.10, 0.15, 0.20, 0.25, 0.30])
    else:
        arrivals = np.sort(rng.integers(0, max(2, n // 2), n))
        bursts = rng.integers(15, 50, n)
        priorities = rng.integers(1, 6, n)
    if arrival_distribution is not None:
        arrivals = _sample(rng, arrival_distribution, n, integer=True)
        if np.any(arrivals < 0):
            raise ValueError("arrival distribution produced a negative arrival")
    if burst_distribution is not None:
        bursts = _sample(rng, burst_distribution, n, integer=True)
        if np.any(bursts <= 0):
            raise ValueError("burst distribution must produce positive values")
    if priority_distribution is not None:
        priorities = _sample(rng, priority_distribution, n, integer=True)
    order = np.argsort(arrivals, kind="stable")
    return [Process(f"P{i + 1:03d}", float(arrivals[k]), float(bursts[k]), int(priorities[k])) for i, k in enumerate(order)]


def save_workload(processes: list[Process], path: str | Path, seed: int | None = None, scenario: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"pid": p.pid, "arrival_time": p.arrival_time, "burst_time": p.burst_time, "priority": p.priority} for p in processes]).to_csv(path, index=False)
    if seed is not None or scenario is not None:
        path.with_suffix(".json").write_text(json.dumps({"seed": seed, "scenario": scenario, "process_count": len(processes)}, indent=2), encoding="utf-8")


def load_workload(path: str | Path) -> list[Process]:
    df = pd.read_csv(path)
    required = {"pid", "arrival_time", "burst_time", "priority"}
    if not required.issubset(df.columns):
        raise ValueError(f"workload CSV must contain columns: {', '.join(sorted(required))}")
    return [Process(str(r.pid), float(r.arrival_time), float(r.burst_time), int(r.priority)) for r in df.itertuples(index=False)]
