"""PPO training and deterministic policy evaluation (Stable-Baselines3)."""
from __future__ import annotations

from pathlib import Path
import csv
import json
import numpy as np
import platform
from importlib.metadata import PackageNotFoundError, version
from .environment import SchedulingEnv
from .models import Process, ScheduleResult, Segment
from .metrics import calculate_metrics


def _sb3():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:
        raise RuntimeError("PPO requires the optional runtime dependencies. Run: pip install -e '.[test]'") from exc
    return PPO, BaseCallback


def train_ppo(processes: list[Process], model_path: str | Path, *, timesteps: int = 10_000,
              learning_rate: float = 3e-4, seed: int = 1, n_steps: int = 256,
              batch_size: int = 64, gamma: float = 0.99, scenario: str = "mixed") -> Path:
    if timesteps < 1 or n_steps < 2 or batch_size < 1:
        raise ValueError("timesteps and batch_size must be positive; n_steps must be at least 2")
    PPO, BaseCallback = _sb3()
    from .workloads import generate_workload
    from stable_baselines3.common.monitor import Monitor

    n = len(processes)
    training_workload_seeds: list[int] = []
    def training_workload(seed_value: int):
        training_workload_seeds.append(seed_value)
        return generate_workload(n, scenario, seed_value)
    env = SchedulingEnv(processes, max_processes=n, workload_factory=training_workload)
    rewards: list[float] = []

    class RewardLogger(BaseCallback):
        def _on_step(self):
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    rewards.append(float(info["episode"]["r"]))
            return True

    batch = min(batch_size, n_steps)
    model = PPO("MlpPolicy", Monitor(env), learning_rate=learning_rate, n_steps=n_steps,
                batch_size=batch, gamma=gamma, seed=seed, verbose=0,
                policy_kwargs={"net_arch": [64, 64]})
    model.learn(total_timesteps=timesteps, callback=RewardLogger())
    dest = Path(model_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(dest))
    with dest.with_suffix(".rewards.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "episode_reward"])
        writer.writerows(enumerate(rewards, 1))
    packages = {}
    for package in ("numpy", "gymnasium", "stable-baselines3", "torch"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    dest.with_suffix(".metadata.json").write_text(json.dumps({
        "algorithm": "Stable-Baselines3 PPO", "split": "training", "python_version": platform.python_version(),
        "package_versions": packages, "training_scenario": scenario, "process_count": n,
        "seed": seed, "training_episode_workload_seeds": training_workload_seeds,
        "requested_timesteps": timesteps, "actual_timesteps": model.num_timesteps,
        "learning_rate": learning_rate, "n_steps": n_steps, "batch_size": batch,
        "gamma": gamma, "policy": "MlpPolicy, hidden layers [64, 64]",
        "reward": "negative normalized waiting-area cost; 0.02 switch penalty; 0.1 invalid-action penalty",
        "policy_actions": "non-preemptive process selection; fixed slot indices; invalid slots flagged, penalized, and safely replaced"
    }, indent=2), encoding="utf-8")
    return dest.with_suffix(".zip")


def evaluate_ppo(processes: list[Process], model_path: str | Path, deterministic: bool = True) -> ScheduleResult:
    PPO, _ = _sb3()
    model_file = Path(model_path)
    if not model_file.exists() and not model_file.with_suffix(".zip").exists():
        raise FileNotFoundError(f"PPO model not found: {model_file}")
    env = SchedulingEnv(processes, max_processes=len(processes))
    model = PPO.load(str(model_file), env=env)
    obs, _ = env.reset(seed=123)
    timeline: list[Segment] = []
    decision_overhead = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        from time import perf_counter_ns
        start = perf_counter_ns()
        action, _ = model.predict(obs, deterministic=deterministic)
        decision_overhead += (perf_counter_ns() - start) / 1e9
        before = env.now
        obs, _, terminated, truncated, info = env.step(int(np.asarray(action).item()))
        selected = env.processes[info["selected_index"]]
        timeline.append(Segment(selected.pid, before, selected.completion_time))
    ps = env.processes
    switches = sum(1 for a, b in zip(timeline, timeline[1:]) if a.pid != b.pid and abs(a.end - b.start) < 1e-9)
    return ScheduleResult("PPO", ps, timeline, decision_overhead, switches)
