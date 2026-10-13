"""Evaluation metrics independent of the learning reward."""
from __future__ import annotations

from time import perf_counter_ns
from .models import ScheduleResult


def calculate_metrics(result: ScheduleResult) -> dict[str, float | int | str]:
    done = [p for p in result.processes if p.completion_time is not None]
    # Simulation clock starts at zero, so any initial idle interval is included.
    elapsed = result.makespan
    busy = sum(s.end - s.start for s in result.timeline)
    n = len(done)
    return {
        "algorithm": result.algorithm,
        "completed_processes": n,
        "average_waiting_time": sum(p.waiting_time for p in done) / n if n else 0.0,
        "average_turnaround_time": sum((p.turnaround_time or 0.0) for p in done) / n if n else 0.0,
        "average_response_time": sum((p.response_time or 0.0) for p in done) / n if n else 0.0,
        "throughput": n / elapsed if elapsed > 0 else 0.0,
        "context_switches": result.context_switches,
        "decision_overhead_seconds": result.decision_overhead_seconds,
        "cpu_utilization_percent": 100.0 * busy / elapsed if elapsed > 0 else 0.0,
        "elapsed_simulation_time": elapsed,
    }


def measure_decision(callable_):
    """Time one scheduler decision with a monotonic high-resolution clock."""
    start = perf_counter_ns()
    value = callable_()
    return value, (perf_counter_ns() - start) / 1e9
