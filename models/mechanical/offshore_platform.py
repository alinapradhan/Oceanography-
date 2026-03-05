"""Offshore platform structural model.

Implements a simplified jacket-type fixed offshore platform with:
- Wave force calculation (Morison equation)
- Wind load estimation
- Natural frequency estimation
- Fatigue damage accumulation (Palmgren–Miner rule, simplified)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PlatformGeometry:
    """Geometric and mass properties of a jacket-type platform."""

    leg_diameter: float = 1.2       # m
    leg_count: int = 4
    water_depth: float = 100.0      # m
    deck_height: float = 15.0       # m (above mean water level)
    deck_mass: float = 5.0e6        # kg (topsides + deck)
    jacket_mass: float = 2.0e6      # kg
    cd_cylinder: float = 1.0        # drag coefficient (Morison)
    cm_cylinder: float = 2.0        # inertia coefficient (Morison)


@dataclass
class PlatformState:
    """Accumulated structural state."""

    base_shear: float = 0.0          # N  (total horizontal wave/wind force)
    overturning_moment: float = 0.0  # N·m
    fatigue_damage: float = 0.0      # dimensionless (0–1, 1 = failure)


class OffshorePlatform:
    """Simplified fixed offshore jacket platform model.

    Provides wave and wind loading estimates using standard engineering
    methods (Morison equation, API RP 2A wind load model).
    """

    WATER_DENSITY: float = 1025.0  # kg/m³
    AIR_DENSITY: float = 1.225     # kg/m³
    GRAVITY: float = 9.81          # m/s²

    def __init__(self, geometry: PlatformGeometry | None = None) -> None:
        self.geometry = geometry or PlatformGeometry()
        self.state = PlatformState()

    # ------------------------------------------------------------------
    # Natural frequency
    # ------------------------------------------------------------------

    @property
    def total_mass(self) -> float:
        """Total structural mass (kg)."""
        return self.geometry.deck_mass + self.geometry.jacket_mass

    def natural_frequency(self) -> float:
        """Approximate first natural frequency (Hz) using a cantilever model.

        Uses a simplified single-degree-of-freedom representation.
        """
        # Equivalent stiffness: k = 3EI/L³ with steel E = 210 GPa
        E = 210e9  # Pa
        I = (math.pi / 64.0) * self.geometry.leg_diameter ** 4  # m⁴ (per leg)
        total_I = self.geometry.leg_count * I
        L = self.geometry.water_depth + self.geometry.deck_height
        k_equiv = 3.0 * E * total_I / L ** 3
        omega_n = math.sqrt(k_equiv / self.total_mass)
        return omega_n / (2.0 * math.pi)

    # ------------------------------------------------------------------
    # Wave loading (Morison equation)
    # ------------------------------------------------------------------

    def morison_wave_force(
        self, wave_height: float, wave_period: float
    ) -> float:
        """Estimate total horizontal wave force using the Morison equation (N).

        Integrates drag and inertia terms over the submerged leg length.

        Args:
            wave_height: Wave height (m).
            wave_period: Wave period (s).

        Returns:
            Total horizontal force amplitude (N).
        """
        a = wave_height / 2.0
        omega = 2.0 * math.pi / wave_period
        k = omega ** 2 / self.GRAVITY  # deep-water wave number

        D = self.geometry.leg_diameter
        rho = self.WATER_DENSITY
        cd = self.geometry.cd_cylinder
        cm = self.geometry.cm_cylinder
        d = self.geometry.water_depth
        n_legs = self.geometry.leg_count

        # Maximum horizontal velocity and acceleration at surface (deep water)
        u_max = a * omega
        du_max = a * omega ** 2

        # Integrate Morison over depth (simplified: use surface values × effective depth)
        # Effective depth factor from cosh decay
        effective_depth = (1.0 / k) * math.sinh(k * d) / math.cosh(k * d) if k * d > 0.01 else d

        f_drag = 0.5 * rho * cd * D * u_max ** 2 * effective_depth
        f_inertia = rho * cm * (math.pi * D ** 2 / 4.0) * du_max * effective_depth

        total_per_leg = math.sqrt(f_drag ** 2 + f_inertia ** 2)
        return n_legs * total_per_leg

    # ------------------------------------------------------------------
    # Wind loading (API RP 2A simplified)
    # ------------------------------------------------------------------

    def wind_force(self, wind_speed: float, exposed_area: float = 500.0) -> float:
        """Estimate wind force on the platform deck (N).

        Args:
            wind_speed: 1-hour mean wind speed at 10 m height (m/s).
            exposed_area: Wind-exposed projected area of topsides (m²).

        Returns:
            Wind force in newtons.
        """
        cd_wind = 1.3   # bluff body drag coefficient
        return 0.5 * self.AIR_DENSITY * cd_wind * exposed_area * wind_speed ** 2

    # ------------------------------------------------------------------
    # Combined load and fatigue
    # ------------------------------------------------------------------

    def update_loads(
        self,
        wave_height: float,
        wave_period: float,
        wind_speed: float,
    ) -> None:
        """Compute and store base shear and overturning moment.

        Args:
            wave_height: Wave height (m).
            wave_period: Wave period (s).
            wind_speed: Wind speed at 10 m (m/s).
        """
        f_wave = self.morison_wave_force(wave_height, wave_period)
        f_wind = self.wind_force(wind_speed)
        total_f = f_wave + f_wind

        arm_wave = self.geometry.water_depth / 2.0
        arm_wind = self.geometry.water_depth + self.geometry.deck_height / 2.0

        self.state.base_shear = total_f
        self.state.overturning_moment = (
            f_wave * arm_wave + f_wind * arm_wind
        )

    def accumulate_fatigue(
        self,
        stress_range: float,
        cycles: float,
        sn_constant: float = 1e14,
        sn_slope: float = 3.0,
    ) -> None:
        """Accumulate fatigue damage using the Palmgren–Miner rule.

        Args:
            stress_range: Stress range (Pa).
            cycles: Number of load cycles.
            sn_constant: S-N curve constant (A in N = A / S^m).
            sn_slope: S-N curve slope exponent (m).
        """
        if stress_range <= 0 or cycles <= 0:
            return
        n_failure = sn_constant / (stress_range ** sn_slope)
        self.state.fatigue_damage += cycles / n_failure

    def __repr__(self) -> str:
        g = self.geometry
        return (
            f"OffshorePlatform(depth={g.water_depth}m, legs={g.leg_count}, "
            f"leg_D={g.leg_diameter}m)"
        )
