"""Simulation engine – drives time-domain simulations for marine systems.

The :class:`SimulationEngine` provides a generic integration loop that
connects an ocean environment with a mechanical model, collecting outputs
at each time step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SimulationResult:
    """Container for simulation output data."""

    system_type: str
    duration: float         # s
    time_step: float        # s
    steps: int
    time: list[float] = field(default_factory=list)
    outputs: dict[str, list[float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.time)

    def channel(self, name: str) -> list[float]:
        """Return a named output channel (raises KeyError if absent)."""
        return self.outputs[name]

    def summary(self) -> dict[str, Any]:
        """Return scalar summary statistics for all channels."""
        stats: dict[str, Any] = {"system_type": self.system_type, "duration_s": self.duration}
        for ch, values in self.outputs.items():
            if values:
                stats[f"{ch}_mean"] = sum(values) / len(values)
                stats[f"{ch}_max"] = max(values)
                stats[f"{ch}_min"] = min(values)
        return stats


class SimulationEngine:
    """Generic time-domain simulation engine.

    Drives a step-function-based system model in a fixed time-step loop,
    optionally calling a data-collection callback at each step.

    Example::

        from models.mechanical.auv import AUV

        auv = AUV()
        engine = SimulationEngine(dt=0.1, duration=120.0)

        def step_fn(t, state):
            auv.step(thrust=20.0, dt=engine.dt)
            return {"speed": auv.state.velocity[0], "x": auv.state.pose.x}

        result = engine.run("auv", step_fn)
    """

    def __init__(self, dt: float = 0.1, duration: float = 600.0) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        if duration <= 0:
            raise ValueError("duration must be positive")
        self.dt = dt
        self.duration = duration

    def run(
        self,
        system_type: str,
        step_fn: Callable[[float, dict[str, Any]], dict[str, float]],
        initial_state: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SimulationResult:
        """Execute the simulation loop.

        Args:
            system_type: Human-readable label for the simulated system.
            step_fn: Callable ``(t, state) -> {channel: value}`` executed at
                each time step.
            initial_state: Initial state dict passed to the first step.
            metadata: Arbitrary metadata to attach to the result.

        Returns:
            :class:`SimulationResult` with all collected outputs.
        """
        state = dict(initial_state or {})
        times: list[float] = []
        channels: dict[str, list[float]] = {}
        n_steps = int(self.duration / self.dt)

        for i in range(n_steps + 1):
            t = round(i * self.dt, 10)
            outputs = step_fn(t, state)
            times.append(t)
            for ch, val in outputs.items():
                channels.setdefault(ch, []).append(val)

        return SimulationResult(
            system_type=system_type,
            duration=self.duration,
            time_step=self.dt,
            steps=n_steps,
            time=times,
            outputs=channels,
            metadata=metadata or {},
        )
