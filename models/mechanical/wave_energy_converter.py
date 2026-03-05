"""Wave Energy Converter (WEC) mechanical model.

Models a point-absorber WEC using linear potential-flow theory with a
radiation-diffraction approximation and a power take-off (PTO) mechanism.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class WECGeometry:
    """Physical dimensions of a cylindrical point-absorber WEC."""

    radius: float = 5.0       # m (float radius)
    draft: float = 4.0        # m (submerged depth)
    mass: float = 2.5e5       # kg (total mass including ballast)


@dataclass
class PTOParameters:
    """Power Take-Off (PTO) system parameters."""

    damping: float = 1.0e5       # N·s/m (linear PTO damping)
    stiffness: float = 0.0       # N/m   (optional spring element)
    efficiency: float = 0.85     # mechanical-to-electrical conversion


@dataclass
class WECState:
    """Instantaneous dynamic state of the WEC."""

    displacement: float = 0.0    # m (heave from equilibrium, positive upward)
    velocity: float = 0.0        # m/s
    power_output: float = 0.0    # W (instantaneous electrical power)
    energy_captured: float = 0.0 # J (cumulative)


class WaveEnergyConverter:
    """Point-absorber Wave Energy Converter model.

    Uses a 1-DOF (heave) equation of motion with:
    - Added mass (frequency-independent approximation)
    - Radiation damping
    - Hydrostatic restoring force
    - Linear PTO damping
    - Froude-Krylov excitation force approximation
    """

    WATER_DENSITY: float = 1025.0  # kg/m³
    GRAVITY: float = 9.81          # m/s²

    def __init__(
        self,
        geometry: WECGeometry | None = None,
        pto: PTOParameters | None = None,
    ) -> None:
        self.geometry = geometry or WECGeometry()
        self.pto = pto or PTOParameters()
        self.state = WECState()

    # ------------------------------------------------------------------
    # Derived hydrostatic / hydrodynamic properties
    # ------------------------------------------------------------------

    @property
    def waterplane_area(self) -> float:
        """Water-plane area (m²)."""
        return math.pi * self.geometry.radius ** 2

    @property
    def displaced_volume(self) -> float:
        """Displaced volume at still-water draft (m³)."""
        return self.waterplane_area * self.geometry.draft

    @property
    def hydrostatic_stiffness(self) -> float:
        """Hydrostatic restoring stiffness C₃₃ (N/m)."""
        return self.WATER_DENSITY * self.GRAVITY * self.waterplane_area

    @property
    def added_mass(self) -> float:
        """Frequency-independent added mass approximation (kg).

        Uses the flat-cylinder formula m_a ≈ ρ π r³ / 3 for a
        submerged flat-bottom cylinder (Lamb, 1932).
        """
        return (self.WATER_DENSITY * math.pi * self.geometry.radius ** 3) / 3.0

    @property
    def radiation_damping(self) -> float:
        """Simplified radiation damping coefficient (N·s/m).

        Estimated as 10 % of the critical damping for illustration.
        """
        total_mass = self.geometry.mass + self.added_mass
        omega_n = math.sqrt(self.hydrostatic_stiffness / total_mass)
        critical = 2.0 * math.sqrt(total_mass * self.hydrostatic_stiffness)
        return 0.1 * critical

    # ------------------------------------------------------------------
    # Excitation force
    # ------------------------------------------------------------------

    def excitation_force(self, wave_amplitude: float, wave_period: float) -> float:
        """Approximate Froude-Krylov excitation force amplitude (N).

        Args:
            wave_amplitude: Incident wave amplitude (m).
            wave_period: Wave period (s).

        Returns:
            Vertical excitation force amplitude in newtons.
        """
        omega = 2.0 * math.pi / wave_period
        k = omega ** 2 / self.GRAVITY   # deep-water wave number
        # Scattering factor (Smith correction) – exponential attenuation with draft
        scattering = math.exp(-k * self.geometry.draft)
        return (
            self.WATER_DENSITY
            * self.GRAVITY
            * wave_amplitude
            * self.waterplane_area
            * scattering
        )

    # ------------------------------------------------------------------
    # Power calculation
    # ------------------------------------------------------------------

    def capture_width_ratio(self, wave_period: float) -> float:
        """Dimensionless capture width ratio (0–1 approximate).

        Based on an optimal PTO-damped resonant absorber analogy.

        Args:
            wave_period: Incident wave period (s).

        Returns:
            Ratio of absorbed power to incident wave power per metre of crest.
        """
        total_mass = self.geometry.mass + self.added_mass
        omega = 2.0 * math.pi / wave_period
        omega_n = math.sqrt(self.hydrostatic_stiffness / total_mass)
        # Simplified resonance bandwidth model
        ratio = 1.0 / (1.0 + ((omega - omega_n) / (omega_n + 1e-9)) ** 2)
        return min(ratio, 1.0)

    def average_power(self, wave_height: float, wave_period: float) -> float:
        """Estimate average absorbed power (W).

        Args:
            wave_height: Significant wave height (m).
            wave_period: Peak wave period (s).

        Returns:
            Average absorbed electrical power in watts.
        """
        # Incident wave power per metre of crest (W/m)
        wave_amplitude = wave_height / 2.0
        omega = 2.0 * math.pi / wave_period
        k = omega ** 2 / self.GRAVITY
        cg = self.GRAVITY / (2.0 * omega)   # deep-water group speed
        p_incident = (
            0.5 * self.WATER_DENSITY * self.GRAVITY * wave_amplitude ** 2 * cg
        )
        cwr = self.capture_width_ratio(wave_period)
        # Effective capture width limited by device diameter
        capture_width = min(cwr * (2.0 * self.geometry.radius), 2.0 * self.geometry.radius * math.pi)
        return p_incident * capture_width * self.pto.efficiency

    # ------------------------------------------------------------------
    # Time-stepping
    # ------------------------------------------------------------------

    def step(
        self,
        excitation: float,
        dt: float,
    ) -> float:
        """Advance the WEC dynamics by one time step using Euler integration.

        Args:
            excitation: Vertical excitation force (N).
            dt: Time step (s).

        Returns:
            Instantaneous absorbed power (W).
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        total_mass = self.geometry.mass + self.added_mass
        z = self.state.displacement
        zdot = self.state.velocity

        # Total damping
        b_total = self.radiation_damping + self.pto.damping
        # Total stiffness
        c_total = self.hydrostatic_stiffness + self.pto.stiffness

        # Equation of motion: M·z̈ = F_exc - b·ż - c·z
        zdotdot = (excitation - b_total * zdot - c_total * z) / total_mass

        # Euler update
        new_zdot = zdot + zdotdot * dt
        new_z = z + zdot * dt

        # PTO power = b_pto * ż²
        instantaneous_power = self.pto.damping * zdot ** 2 * self.pto.efficiency

        self.state.velocity = new_zdot
        self.state.displacement = new_z
        self.state.power_output = instantaneous_power
        self.state.energy_captured += instantaneous_power * dt

        return instantaneous_power

    def __repr__(self) -> str:
        g = self.geometry
        return (
            f"WaveEnergyConverter(radius={g.radius}m, draft={g.draft}m, "
            f"mass={g.mass:.1e}kg)"
        )
