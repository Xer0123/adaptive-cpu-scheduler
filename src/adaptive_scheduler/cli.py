"""Command line interface for workload, baseline, PPO, and experiment runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
from .algorithms import run_policy
from .experiments import run_experiments
from .metrics import calculate_metrics
from .visualization import gantt_chart, comparison_charts, reward_chart
from .workloads import SCENARIOS, generate_workload, load_workload, save_workload


def _parser():
    parser = argparse.ArgumentParser(prog="cpu-scheduler", description="Adaptive CPU scheduling simulator")
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="generate a reproducible workload CSV")
    g.add_argument("--n", type=int, default=20); g.add_argument("--scenario", choices=SCENARIOS, default="mixed")
    g.add_argument("--seed", type=int, default=1); g.add_argument("--output", default="results/workload.csv")
    b = sub.add_parser("baseline", help="run one classical scheduler")
    b.add_argument("--algorithm", choices=["fcfs", "sjf", "priority", "rr"], default="fcfs")
    b.add_argument("--input"); b.add_argument("--n", type=int, default=12); b.add_argument("--scenario", choices=SCENARIOS, default="mixed")
    b.add_argument("--seed", type=int, default=7); b.add_argument("--quantum", type=float, default=3.0); b.add_argument("--output-dir", default="results/baseline")
    t = sub.add_parser("train", help="train PPO on synthetic training workloads")
    t.add_argument("--n", type=int, default=20); t.add_argument("--scenario", choices=SCENARIOS, default="mixed")
    t.add_argument("--seed", type=int, default=11); t.add_argument("--timesteps", type=int, default=10000)
    t.add_argument("--learning-rate", type=float, default=3e-4); t.add_argument("--n-steps", type=int, default=256)
    t.add_argument("--batch-size", type=int, default=64); t.add_argument("--gamma", type=float, default=.99)
    t.add_argument("--model", default="results/models/ppo_scheduler")
    e = sub.add_parser("evaluate", help="evaluate an existing PPO checkpoint")
    e.add_argument("--model", required=True); e.add_argument("--input"); e.add_argument("--n", type=int, default=20)
    e.add_argument("--scenario", choices=SCENARIOS, default="bursty"); e.add_argument("--seed", type=int, default=101); e.add_argument("--output-dir", default="results/evaluation")
    x = sub.add_parser("compare", help="compare baselines and optional PPO on shared held-out workloads")
    x.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["uniform", "mixed", "bursty", "priority", "cpu_intensive"])
    x.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303]); x.add_argument("--n", type=int, default=30)
    x.add_argument("--quantum", type=float, default=3.0); x.add_argument("--model"); x.add_argument("--output-dir", default="results/experiments")
    x.add_argument("--split", choices=["validation", "test"], default="test", help="record and keep validation and final-test outputs separate")
    d = sub.add_parser("demo", help="run a compact baseline demonstration with exports")
    d.add_argument("--output-dir", default="results/demo"); d.add_argument("--n", type=int, default=10); d.add_argument("--seed", type=int, default=7); d.add_argument("--quantum", type=float, default=3.0)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "generate":
        ps = generate_workload(args.n, args.scenario, args.seed); save_workload(ps, args.output, args.seed, args.scenario)
        print(f"Saved {len(ps)} processes to {args.output}")
    elif args.command == "baseline":
        ps = load_workload(args.input) if args.input else generate_workload(args.n, args.scenario, args.seed)
        result = run_policy(args.algorithm, ps, args.quantum); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([calculate_metrics(result)]).to_csv(out / "metrics.csv", index=False)
        pd.DataFrame([{"pid": p.pid, "arrival": p.arrival_time, "burst": p.burst_time, "priority": p.priority,
                       "first_start": p.first_execution_time, "completion": p.completion_time, "waiting": p.waiting_time,
                       "turnaround": p.turnaround_time, "response": p.response_time} for p in result.processes]).to_csv(out / "processes.csv", index=False)
        gantt_chart(result, out / "gantt.png"); print(pd.DataFrame([calculate_metrics(result)]).to_string(index=False))
    elif args.command == "train":
        ps = generate_workload(args.n, args.scenario, args.seed)
        from .ppo import train_ppo
        checkpoint = train_ppo(ps, args.model, timesteps=args.timesteps, learning_rate=args.learning_rate,
                               seed=args.seed, n_steps=args.n_steps, batch_size=args.batch_size, gamma=args.gamma, scenario=args.scenario)
        reward_file = Path(args.model).with_suffix(".rewards.csv")
        reward_chart(reward_file, reward_file.with_suffix(".png"))
        print(f"Saved PPO checkpoint: {checkpoint}; training reward log: {reward_file}")
    elif args.command == "evaluate":
        ps = load_workload(args.input) if args.input else generate_workload(args.n, args.scenario, args.seed)
        from .ppo import evaluate_ppo
        result = evaluate_ppo(ps, args.model); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([calculate_metrics(result)]).to_csv(out / "metrics.csv", index=False); gantt_chart(result, out / "gantt.png")
        print(pd.DataFrame([calculate_metrics(result)]).to_string(index=False))
    elif args.command in ("compare", "demo"):
        if args.command == "demo":
            scenarios, seeds, n, output, model, quantum = ["mixed"], [args.seed], args.n, args.output_dir, None, args.quantum
            ps = generate_workload(n, "mixed", args.seed); save_workload(ps, Path(output) / "demo_workload.csv", args.seed, "mixed")
            for name in ("fcfs", "sjf", "priority", "rr"):
                result = run_policy(name, ps, quantum)
                gantt_chart(result, Path(output) / f"{name}_gantt.png")
        else:
            scenarios, seeds, n, output, model, quantum = args.scenarios, args.seeds, args.n, args.output_dir, args.model, args.quantum
        split = "demo" if args.command == "demo" else args.split
        raw, summary = run_experiments(output, scenarios=scenarios, seeds=seeds, n=n, quantum=quantum, model_path=model, split=split)
        comparison_charts(raw[raw.status == "ok"], Path(output) / "charts")
        print(f"Saved {len(raw)} run records and {len(summary)} aggregate rows under {output}")
        if not raw.empty and "status" in raw and (raw.status != "ok").any():
            print("Some runs failed; inspect per_run_results.csv for recorded errors.")


if __name__ == "__main__":
    main()
