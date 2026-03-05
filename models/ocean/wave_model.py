"""JONSWAP and linear wave theory models for ocean wave simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass
class WaveComponent:
    """Single sinusoidal wave component."""

    amplitude: float     # m
    frequency: float     # Hz
    phase: float         # radians
    direction: float     # degrees from North


class JONSWAPSpectrum:
    """JONSWAP (Joint North Sea Wave Project) wave energy spectrum.

    Reference: Hasselmann et al. (1973).
    """

    GAMMA_DEFAULT: float = 3.3   # peak enhancement factor

    def __init__(
        self,
        significant_wave_height: float,
        peak_period: float,
        gamma: float = GAMMA_DEFAULT,
    ) -> None:
        if significant_wave_height <= 0:
            raise ValueError("significant_wave_height must be positive")
        if peak_period <= 0:
            raise ValueError("peak_period must be positive")
        self.hs = significant_wave_height
        self.tp = peak_period
        self.fp = 1.0 / peak_period
        self.gamma = gamma

    def spectral_density(self, freq: float) -> float:
        """Return spectral energy density S(f) in m²/Hz.

        Args:
            freq: Frequency in Hz (must be > 0).

        Returns:
            Spectral energy density at *freq*.
        """
        if freq <= 0:
            return 0.0

        fp = self.fp
        g = 9.81

        # Phillips constant calibrated for JONSWAP
        alpha = 5.058 * (self.hs ** 2) * (fp ** 4)

        # Pierson-Moskowitz base
        pm = alpha * (g ** 2) * (2.0 * math.pi) ** (-4) * freq ** (-5)
        pm *= math.exp(-1.25 * (fp / freq) ** 4)

        # Peak enhancement
        sigma = 0.07 if freq <= fp else 0.09
        r = math.exp(-((freq - fp) ** 2) / (2.0 * sigma ** 2 * fp ** 2))
        return pm * (self.gamma ** r)

    def sample_components(
        self, n_components: int = 20, f_min: float = 0.02, f_max: float = 0.5
    ) -> list[WaveComponent]:
        """Discretise the spectrum into *n_components* wave components.

        Args:
            n_components: Number of frequency bands.
            f_min: Lower frequency bound (Hz).
            f_max: Upper frequency bound (Hz).

        Returns:
            List of :class:`WaveComponent` objects with random phases.
        """
        import random

        df = (f_max - f_min) / n_components
        components: list[WaveComponent] = []
        for i in range(n_components):
            f = f_min + (i + 0.5) * df
            s = self.spectral_density(f)
            a = math.sqrt(2.0 * s * df)
            phase = random.uniform(0.0, 2.0 * math.pi)
            components.append(WaveComponent(amplitude=a, frequency=f, phase=phase, direction=0.0))
        return components

    def surface_elevation(
        self,
        x: float,
        t: float,
        components: Sequence[WaveComponent],
    ) -> float:
        """Compute surface elevation at position *x* and time *t*.

        Args:
            x: Position along wave propagation direction (m).
            t: Time (s).
            components: Pre-computed wave components from :meth:`sample_components`.

        Returns:
            Free-surface elevation η(x, t) in metres.
        """
        g = 9.81
        eta = 0.0
        for wc in components:
            omega = 2.0 * math.pi * wc.frequency
            k = omega ** 2 / g   # deep-water approximation
            eta += wc.amplitude * math.cos(k * x - omega * t + wc.phase)
        return eta


class LinearWaveKinematics:
    """Velocity and acceleration field from linear (Airy) wave theory."""

    def __init__(self, amplitude: float, period: float, water_depth: float) -> None:
        if amplitude <= 0:
            raise ValueError("amplitude must be positive")
        if period <= 0:
            raise ValueError("period must be positive")
        if water_depth <= 0:
            raise ValueError("water_depth must be positive")
        self.a = amplitude
        self.T = period
        self.d = water_depth
        self.omega = 2.0 * math.pi / period
        self.k = self._solve_dispersion()

    def _solve_dispersion(self) -> float:
        """Solve linear dispersion relation ω² = g·k·tanh(k·d)."""
        g = 9.81
        k = self.omega ** 2 / g
        for _ in range(50):
            tanh_kd = math.tanh(k * self.d)
            f = g * k * tanh_kd - self.omega ** 2
            df = g * (tanh_kd + k * self.d * (1.0 - tanh_kd ** 2))
            k -= f / df
            if abs(f) < 1e-10:
                break
        return k

    def horizontal_velocity(self, z: float, t: float) -> float:
        """Horizontal water particle velocity u(z, t) at depth *z* (m/s).

        Args:
            z: Vertical position (0 at surface, positive upward; negative below).
            t: Time (s).

        Returns:
            Horizontal velocity in m/s.
        """
        g = 9.81
        numerator = math.cosh(self.k * (self.d + z))
        denominator = math.sinh(self.k * self.d)
        return (
            self.a * self.omega * numerator / denominator * math.cos(-self.omega * t)
        )

    def vertical_velocity(self, z: float, t: float) -> float:
        """Vertical water particle velocity w(z, t) (m/s)."""
        numerator = math.sinh(self.k * (self.d + z))
        denominator = math.sinh(self.k * self.d)
        return (
            self.a * self.omega * numerator / denominator * math.sin(-self.omega * t)
        )

    def horizontal_acceleration(self, z: float, t: float) -> float:
        """Horizontal water particle acceleration (m/s²)."""
        numerator = math.cosh(self.k * (self.d + z))
        denominator = math.sinh(self.k * self.d)
        return (
            self.a * self.omega ** 2 * numerator / denominator * math.sin(-self.omega * t)
        )
