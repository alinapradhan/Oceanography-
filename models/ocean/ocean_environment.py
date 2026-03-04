"""Ocean environment model providing physical conditions for marine systems."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class OceanState:
    """Snapshot of ocean conditions at a given location and time."""

    wave_height: float        # significant wave height (m)
    wave_period: float        # peak wave period (s)
    wave_direction: float     # degrees from North (0–360)
    current_speed: float      # m/s
    current_direction: float  # degrees from North (0–360)
    water_depth: float        # m
    temperature: float        # °C
    salinity: float           # PSU
    wind_speed: float         # m/s
    wind_direction: float     # degrees from North (0–360)

    @property
    def current_velocity(self) -> tuple[float, float]:
        """Return (u, v) current velocity components in m/s."""
        rad = math.radians(self.current_direction)
        return (
            self.current_speed * math.sin(rad),
            self.current_speed * math.cos(rad),
        )

    @property
    def wind_velocity(self) -> tuple[float, float]:
        """Return (u, v) wind velocity components in m/s."""
        rad = math.radians(self.wind_direction)
        return (
            self.wind_speed * math.sin(rad),
            self.wind_speed * math.cos(rad),
        )


class OceanEnvironment:
    """Model of the ocean environment that supports querying conditions.

    In a production system this would consume real-time or hindcast data.
    Here it provides deterministic conditions suitable for simulation and
    agent-based design/optimization loops.
    """

    WATER_DENSITY: float = 1025.0   # kg/m³ (sea water at 15 °C, 35 PSU)
    GRAVITY: float = 9.81           # m/s²

    def __init__(self, state: OceanState | None = None) -> None:
        if state is None:
            state = OceanState(
                wave_height=2.0,
                wave_period=8.0,
                wave_direction=270.0,
                current_speed=0.5,
                current_direction=90.0,
                water_depth=100.0,
                temperature=15.0,
                salinity=35.0,
                wind_speed=10.0,
                wind_direction=270.0,
            )
        self._state = state

    @property
    def state(self) -> OceanState:
        return self._state

    def update(self, **kwargs: float) -> None:
        """Update individual ocean state parameters."""
        for key, value in kwargs.items():
            if not hasattr(self._state, key):
                raise AttributeError(f"OceanState has no attribute '{key}'")
            setattr(self._state, key, value)

    def wave_angular_frequency(self) -> float:
        """Angular frequency of the dominant wave (rad/s)."""
        return 2.0 * math.pi / self._state.wave_period

    def wave_number(self) -> float:
        """Approximate wave number via the linear dispersion relation (rad/m)."""
        omega = self.wave_angular_frequency()
        d = self._state.water_depth
        g = self.GRAVITY

        # Initial deep-water guess: k ≈ ω² / g
        k = omega ** 2 / g
        # Newton–Raphson iteration
        for _ in range(50):
            tanh_kd = math.tanh(k * d)
            f = g * k * tanh_kd - omega ** 2
            df = g * (tanh_kd + k * d * (1.0 - tanh_kd ** 2))
            k -= f / df
            if abs(f) < 1e-10:
                break
        return k

    def wave_phase_speed(self) -> float:
        """Phase speed of the dominant wave (m/s)."""
        return self.wave_angular_frequency() / self.wave_number()

    def dynamic_pressure_amplitude(self, depth: float = 0.0) -> float:
        """Dynamic pressure amplitude at *depth* below surface (Pa).

        Args:
            depth: Depth below the free surface (positive downward, metres).

        Returns:
            Dynamic pressure amplitude in pascals.
        """
        a = self._state.wave_height / 2.0
        k = self.wave_number()
        d = self._state.water_depth
        numerator = math.cosh(k * (d - depth))
        denominator = math.cosh(k * d)
        return (
            self.WATER_DENSITY
            * self.GRAVITY
            * a
            * numerator
            / denominator
        )

    def beaufort_scale(self) -> int:
        """Return the Beaufort wind-force number based on wind speed."""
        speed = self._state.wind_speed
        thresholds = [0.3, 1.5, 3.3, 5.5, 7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6]
        for scale, threshold in enumerate(thresholds):
            if speed < threshold:
                return scale
        return 12

    def __repr__(self) -> str:
        s = self._state
        return (
            f"OceanEnvironment(Hs={s.wave_height}m, Tp={s.wave_period}s, "
            f"depth={s.water_depth}m, current={s.current_speed}m/s)"
        )
