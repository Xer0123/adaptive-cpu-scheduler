"""Classical scheduling policies over a shared process model."""
from __future__ import annotations

from collections import deque
from time import perf_counter_ns
from .models import Process, ScheduleResult, Segment


def _prepare(processes: list[Process]) -> tuple[list[Process], dict[str, int]]:
    ps = [p.clone() for p in processes]
    order = {i: i for i, _ in enumerate(ps)}
    return ps, order


def _finish(p: Process, now: float) -> None:
    p.completion_time = now
    p.turnaround_time = now - p.arrival_time
    p.waiting_time = p.turnaround_time - p.burst_time
    p.response_time = (p.first_execution_time or 0.0) - p.arrival_time


def _result(name: str, ps: list[Process], timeline: list[Segment], overhead: int) -> ScheduleResult:
    switches = sum(1 for a, b in zip(timeline, timeline[1:]) if a.pid != b.pid and abs(a.end - b.start) < 1e-9)
    return ScheduleResult(name, ps, timeline, overhead / 1e9, switches)


def _non_preemptive(processes: list[Process], name: str, choose) -> ScheduleResult:
    ps, order = _prepare(processes)
    pending = set(range(len(ps)))
    timeline: list[Segment] = []
    now = 0.0
    overhead = 0
    while pending:
        ready = [i for i in pending if ps[i].arrival_time <= now]
        if not ready:
            now = min(ps[i].arrival_time for i in pending)
            ready = [i for i in pending if ps[i].arrival_time <= now]
        t0 = perf_counter_ns()
        chosen = choose(ready, ps, order)
        overhead += perf_counter_ns() - t0
        p = ps[chosen]
        if p.first_execution_time is None:
            p.first_execution_time = now
        end = now + p.burst_time
        timeline.append(Segment(p.pid, now, end))
        _finish(p, end)
        now = end
        pending.remove(chosen)
    return _result(name, ps, timeline, overhead)


def fcfs(processes: list[Process]) -> ScheduleResult:
    return _non_preemptive(processes, "FCFS", lambda ready, ps, order: min(ready, key=lambda i: (ps[i].arrival_time, order[i])))


def sjf(processes: list[Process]) -> ScheduleResult:
    """Non-preemptive SJF; ties break by arrival then input order."""
    return _non_preemptive(processes, "SJF", lambda ready, ps, order: min(ready, key=lambda i: (ps[i].burst_time, ps[i].arrival_time, order[i])))


def priority_scheduling(processes: list[Process]) -> ScheduleResult:
    """Non-preemptive priority; lower numeric values mean higher priority."""
    return _non_preemptive(processes, "Priority", lambda ready, ps, order: min(ready, key=lambda i: (ps[i].priority, ps[i].arrival_time, order[i])))


def round_robin(processes: list[Process], quantum: float = 3.0) -> ScheduleResult:
    if quantum <= 0:
        raise ValueError("quantum must be positive")
    ps, order = _prepare(processes)
    waiting = sorted(range(len(ps)), key=lambda i: (ps[i].arrival_time, order[i]))
    q: deque[int] = deque()
    timeline: list[Segment] = []
    now = 0.0
    overhead = 0
    while q or waiting:
        while waiting and ps[waiting[0]].arrival_time <= now:
            q.append(waiting.pop(0))
        if not q:
            now = max(now, ps[waiting[0]].arrival_time)
            continue
        t0 = perf_counter_ns()
        i = q.popleft()
        overhead += perf_counter_ns() - t0
        p = ps[i]
        if p.first_execution_time is None:
            p.first_execution_time = now
        run = min(float(p.remaining_burst), quantum)
        end = now + run
        timeline.append(Segment(p.pid, now, end))
        # Arrivals during a slice enter the tail before an unfinished process is requeued.
        while waiting and ps[waiting[0]].arrival_time <= end:
            q.append(waiting.pop(0))
        p.remaining_burst = float(p.remaining_burst) - run
        now = end
        if p.remaining_burst <= 1e-9:
            p.remaining_burst = 0.0
            _finish(p, now)
        else:
            q.append(i)
    return _result("Round Robin", ps, timeline, overhead)


def run_policy(name: str, processes: list[Process], quantum: float = 3.0) -> ScheduleResult:
    policies = {"fcfs": fcfs, "sjf": sjf, "priority": priority_scheduling, "rr": lambda x: round_robin(x, quantum)}
    try:
        return policies[name.lower()](processes)
    except KeyError as exc:
        raise ValueError(f"unknown baseline {name!r}") from exc
