from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import pandas as pd

from adaptive_scheduler.algorithms import fcfs, sjf, priority_scheduling, round_robin
from adaptive_scheduler.environment import SchedulingEnv
from adaptive_scheduler.experiments import run_experiments
from adaptive_scheduler.metrics import calculate_metrics
from adaptive_scheduler.models import Process
from adaptive_scheduler.workloads import generate_workload, load_workload, save_workload


def example():
    return [Process("A", 0, 5, 2), Process("B", 1, 3, 1), Process("C", 1, 1, 0)]


class SchedulingTests(unittest.TestCase):
    def test_hand_calculated_fcfs(self):
        result = fcfs(example())
        self.assertEqual([(s.pid, s.start, s.end) for s in result.timeline], [("A", 0, 5), ("B", 5, 8), ("C", 8, 9)])
        m = calculate_metrics(result)
        self.assertAlmostEqual(m["average_waiting_time"], 11 / 3)
        self.assertAlmostEqual(m["average_turnaround_time"], 20 / 3)
        self.assertAlmostEqual(m["average_response_time"], 11 / 3)

    def test_hand_calculated_sjf_nonpreemptive(self):
        result = sjf(example())
        self.assertEqual([s.pid for s in result.timeline], ["A", "C", "B"])
        self.assertEqual([p.completion_time for p in result.processes], [5, 9, 6])

    def test_hand_calculated_priority_lower_value_is_higher(self):
        result = priority_scheduling(example())
        self.assertEqual([s.pid for s in result.timeline], ["A", "C", "B"])

    def test_round_robin_arrivals_and_quantum_expiry(self):
        result = round_robin(example(), quantum=2)
        self.assertEqual([(s.pid, s.start, s.end) for s in result.timeline],
                         [("A", 0, 2), ("B", 2, 4), ("C", 4, 5), ("A", 5, 7), ("B", 7, 8), ("A", 8, 9)])
        self.assertEqual([p.completion_time for p in result.processes], [9, 8, 5])

    def test_idle_period_and_utilization_include_initial_idle(self):
        result = fcfs([Process("A", 4, 2), Process("B", 0, 1)])
        self.assertEqual(result.timeline[1].start, 4)
        self.assertAlmostEqual(calculate_metrics(result)["cpu_utilization_percent"], 50)

    def test_all_policies_complete_valid_workload_without_overrun(self):
        workload = generate_workload(25, "bursty", 3)
        for result in (fcfs(workload), sjf(workload), priority_scheduling(workload), round_robin(workload, 3)):
            self.assertEqual(len(result.processes), len(workload))
            self.assertTrue(all(p.completion_time is not None for p in result.processes))
            by_pid = {p.pid: p for p in result.processes}
            for segment in result.timeline:
                self.assertGreaterEqual(segment.start + 1e-9, by_pid[segment.pid].arrival_time)
            for p in result.processes:
                executed = sum(s.end - s.start for s in result.timeline if s.pid == p.pid)
                self.assertAlmostEqual(executed, p.burst_time)
                self.assertAlmostEqual(p.turnaround_time, p.completion_time - p.arrival_time)
                self.assertAlmostEqual(p.waiting_time, p.turnaround_time - p.burst_time)
                self.assertAlmostEqual(p.response_time, p.first_execution_time - p.arrival_time)

    def test_simultaneous_arrivals_are_deterministic(self):
        ps = [Process("x", 0, 2), Process("y", 0, 1), Process("z", 0, 3)]
        self.assertEqual([s.pid for s in fcfs(ps).timeline], ["x", "y", "z"])

    def test_round_robin_requires_positive_quantum(self):
        with self.assertRaises(ValueError):
            round_robin(example(), 0)

    def test_switches_exclude_initial_dispatch_and_idle_transition(self):
        from adaptive_scheduler.models import ScheduleResult, Segment
        result = ScheduleResult("sample", [], [Segment("A", 0, 1), Segment("B", 2, 3)], 0.0)
        from adaptive_scheduler.algorithms import _result
        self.assertEqual(_result("sample", [], result.timeline, 0).context_switches, 0)

    def test_empty_schedule_metrics_are_finite(self):
        from adaptive_scheduler.models import ScheduleResult
        metrics = calculate_metrics(ScheduleResult("empty", [], [], 0.0))
        self.assertEqual(metrics["throughput"], 0)
        self.assertEqual(metrics["cpu_utilization_percent"], 0)

    def test_workload_reproducibility_and_csv_roundtrip(self):
        first = generate_workload(20, "mixed", 42)
        second = generate_workload(20, "mixed", 42)
        self.assertEqual([(p.arrival_time, p.burst_time, p.priority) for p in first],
                         [(p.arrival_time, p.burst_time, p.priority) for p in second])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "w.csv"; save_workload(first, path, 42, "mixed")
            loaded = load_workload(path)
            self.assertEqual([(p.pid, p.arrival_time, p.burst_time, p.priority) for p in loaded],
                             [(p.pid, p.arrival_time, p.burst_time, p.priority) for p in first])

    def test_custom_workload_distributions(self):
        ps = generate_workload(10, "mixed", 12,
            arrival_distribution={"kind": "fixed", "value": 2},
            burst_distribution={"kind": "uniform", "low": 4, "high": 5},
            priority_distribution={"kind": "categorical", "values": [0, 3], "probabilities": [1, 0]},
            short_job_fraction=0.5)
        self.assertTrue(all(p.arrival_time == 2 and p.burst_time == 4 and p.priority == 0 for p in ps))

    def test_gymnasium_environment_validity_and_invalid_action(self):
        env = SchedulingEnv(example(), max_processes=4)
        obs, info = env.reset(seed=9)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(info["valid_actions"], [0])
        obs, reward, terminated, truncated, info = env.step(3)  # padded slot
        self.assertTrue(info["invalid_action"])
        self.assertEqual(info["selected_index"], 0)
        self.assertEqual(env.processes[0].completion_time, 5)
        self.assertFalse(terminated); self.assertFalse(truncated)
        self.assertEqual(info["valid_actions"], [1, 2])
        self.assertLess(reward, 0)

    def test_gymnasium_checker(self):
        from gymnasium.utils.env_checker import check_env
        check_env(SchedulingEnv(example(), max_processes=4), skip_render_check=True)

    def test_ppo_checkpoint_roundtrip_when_runtime_available(self):
        try:
            import stable_baselines3  # noqa: F401
        except ImportError:
            self.skipTest("Stable-Baselines3 optional dependency is not installed")
        from adaptive_scheduler.ppo import train_ppo, evaluate_ppo
        with tempfile.TemporaryDirectory() as d:
            model = train_ppo(example(), Path(d) / "tiny", timesteps=8, n_steps=4, batch_size=4, seed=2)
            self.assertTrue(model.exists())
            result = evaluate_ppo(example(), model)
            self.assertEqual(result.algorithm, "PPO")
            self.assertEqual(len(result.processes), len(example()))

    def test_experiment_outputs_use_common_workload(self):
        with tempfile.TemporaryDirectory() as d:
            raw, summary = run_experiments(d, scenarios=["uniform"], seeds=[4], n=5, split="validation")
            self.assertEqual(len(raw), 4)
            self.assertTrue((raw.split == "validation").all())
            self.assertTrue((raw.status == "ok").all())
            self.assertEqual(len(summary), 4)
            self.assertTrue((Path(d) / "per_run_results.csv").exists())
            self.assertTrue((Path(d) / "process_results" / "uniform_seed4_fcfs.csv").exists())
            fcfs_rows = pd.read_csv(Path(d) / "process_results" / "uniform_seed4_fcfs.csv")[["pid", "arrival_time", "burst_time", "priority"]]
            sjf_rows = pd.read_csv(Path(d) / "process_results" / "uniform_seed4_sjf.csv")[["pid", "arrival_time", "burst_time", "priority"]]
            pd.testing.assert_frame_equal(fcfs_rows, sjf_rows)


if __name__ == "__main__":
    unittest.main()
