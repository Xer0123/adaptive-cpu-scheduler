"""Core workload and execution timeline data models."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Process:
    pid: str
    arrival_time: float
    burst_time: float
    priority: int = 0
    remaining_burst: float | None = None
    first_execution_time: float | None = None
    completion_time: float | None = None
    waiting_time: float = 0.0
    turnaround_time: float | None = None
    response_time: float | None = None

    def __post_init__(self) -> None:
        if not self.pid or self.arrival_time < 0 or self.burst_time <= 0:
            raise ValueError("pid must be nonempty, arrival_time >= 0, and burst_time > 0")
        if self.remaining_burst is None:
            self.remaining_burst = float(self.burst_time)

    def clone(self) -> "Process":
        return Process(self.pid, float(self.arrival_time), float(self.burst_time), int(self.priority))


@dataclass(frozen=True)
class Segment:
    pid: str
    start: float
    end: float


@dataclass
class ScheduleResult:
    algorithm: str
    processes: list[Process]
    timeline: list[Segment]
    decision_overhead_seconds: float
    context_switches: int = 0

    @property
    def makespan(self) -> float:
        return max((p.completion_time or 0.0 for p in self.processes), default=0.0)
