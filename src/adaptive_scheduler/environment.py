"""Fixed-shape Gymnasium environment for process-level dispatch decisions."""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .models import Process


class SchedulingEnv(gym.Env):
    """Each selected ready process runs to completion (non-preemptive dispatch).

    Action is an input-workload slot index. Invalid/padded slots incur a penalty
    and dispatch the first valid ready slot, so state cannot be corrupted.
    Reward is negative incremental ready-queue waiting area, normalized by
    ``max_processes * max_burst``, plus a configurable switch penalty.
    """
    metadata = {"render_modes": []}

    def __init__(self, processes: list[Process], max_processes: int | None = None,
                 switch_penalty: float = 0.02, invalid_action_penalty: float = 0.1,
                 workload_factory=None):
        super().__init__()
        if not processes:
            raise ValueError("environment needs at least one process")
        self.template = [p.clone() for p in processes]
        self.workload_factory = None
        self._pending_workload_factory = workload_factory
        self.max_processes = max_processes or len(processes)
        if len(processes) > self.max_processes:
            raise ValueError("workload exceeds max_processes")
        self.switch_penalty = switch_penalty
        self.invalid_action_penalty = invalid_action_penalty
        # Per slot: ready, remaining, waiting so far, priority, time since arrival; globals: clock, queue size.
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.max_processes * 5 + 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.max_processes)
        self.reset()
        self.workload_factory = self._pending_workload_factory
        self._pending_workload_factory = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.workload_factory is not None:
            self.template = [p.clone() for p in self.workload_factory(int(self.np_random.integers(0, 2**31 - 1)))]
            if len(self.template) > self.max_processes:
                raise ValueError("generated workload exceeds max_processes")
        self.processes = [p.clone() for p in self.template]
        self.now = 0.0
        self.ready: list[int] = []
        self.done: set[int] = set()
        self.last_pid: int | None = None
        self.switches = 0
        self._refresh()
        return self._obs(), {"valid_actions": self.ready.copy(), "time": self.now}

    def _refresh(self):
        self.ready = [i for i, p in enumerate(self.processes) if i not in self.done and p.arrival_time <= self.now]
        if not self.ready and len(self.done) < len(self.processes):
            self.now = min(p.arrival_time for i, p in enumerate(self.processes) if i not in self.done)
            self.ready = [i for i, p in enumerate(self.processes) if i not in self.done and p.arrival_time <= self.now]

    def _obs(self):
        max_burst = max(p.burst_time for p in self.processes)
        max_priority = max(1, max(abs(p.priority) for p in self.processes))
        max_arrival = max(1.0, max(p.arrival_time for p in self.processes))
        out = np.zeros(self.observation_space.shape, dtype=np.float32)
        for i, p in enumerate(self.processes[:self.max_processes]):
            off = i * 5
            ready = float(i in self.ready)
            remaining = float(p.remaining_burst or 0.0) / max_burst if i not in self.done else 0.0
            waited = max(0.0, self.now - p.arrival_time) / max_arrival
            out[off:off+5] = [ready, remaining, min(waited, 1.0), np.clip(p.priority / max_priority, -1, 1), min(self.now / max_arrival, 1.0)]
        out[-2] = min(self.now / (max_arrival + sum(p.burst_time for p in self.processes)), 1.0)
        out[-1] = len(self.ready) / self.max_processes
        return out

    def step(self, action):
        if len(self.done) == len(self.processes):
            raise RuntimeError("step called after episode termination; call reset")
        invalid = not isinstance(action, (int, np.integer)) or int(action) not in self.ready
        index = self.ready[0] if invalid else int(action)
        p = self.processes[index]
        start = self.now
        switched = self.last_pid is not None and self.last_pid != index
        if switched:
            self.switches += 1
        waiting_area = 0.0
        end = start + p.burst_time
        for j, other in enumerate(self.processes):
            if j != index and j not in self.done and other.arrival_time < end:
                waiting_area += max(0.0, end - max(start, other.arrival_time))
        p.first_execution_time = start
        p.completion_time = end
        p.remaining_burst = 0.0
        p.turnaround_time = end - p.arrival_time
        p.waiting_time = p.turnaround_time - p.burst_time
        p.response_time = start - p.arrival_time
        self.now = end
        self.done.add(index)
        self.last_pid = index
        self._refresh()
        scale = max(1.0, self.max_processes * max(x.burst_time for x in self.processes))
        reward = -waiting_area / scale - (self.switch_penalty if switched else 0.0)
        if invalid:
            reward -= self.invalid_action_penalty
        terminated = len(self.done) == len(self.processes)
        return self._obs(), float(reward), terminated, False, {"valid_actions": self.ready.copy(), "selected_index": index, "invalid_action": invalid, "time": self.now, "waiting_area": waiting_area, "context_switches": self.switches}
