"""Ocean monitoring buoy model.

Models a surface mooring buoy including:
- Motion response in waves (heave)
- Mooring tension
- Sensor payload management
- Power budget (solar + battery)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class BuoyGeometry:
    """Physical characteristics of a discus-type monitoring buoy."""

    diameter: float = 2.0         # m (hull diameter)
    height: float = 1.0           # m (hull height above water)
    mass: float = 1500.0          # kg (total deployed mass)
    waterplane_area: float = 3.14 # m² (≈ π·(d/2)²)
    mooring_depth: float = 50.0   # m (mooring line length)
    mooring_stiffness: float = 5000.0  # N/m


@dataclass
class PowerSystem:
    """Solar and battery power system of the buoy."""

    solar_panel_area: float = 2.0       # m²
    solar_efficiency: float = 0.18
    battery_capacity_wh: float = 1000.0
    battery_level: float = 1.0          # 0–1
    load_power_w: float = 20.0          # W (continuous sensor/comms load)


@dataclass
class SensorPayload:
    """Sensor suite carried by the buoy."""

    sensors: Dict[str, bool] = field(default_factory=lambda: {
        "CTD": True,            # conductivity, temperature, depth
        "ADCP": True,           # acoustic Doppler current profiler
        "wave_gauge": True,     # surface wave measurements
        "meteorological": True, # wind, air temperature, humidity
        "GPS": True,            # position
    })

    @property
    def active_count(self) -> int:
        return sum(self.sensors.values())


@dataclass
class BuoyState:
    """Dynamic state of the buoy."""

    heave: float = 0.0           # m (displacement from still-water)
    heave_velocity: float = 0.0  # m/s
    mooring_tension: float = 0.0 # N
    roll: float = 0.0            # rad
    pitch: float = 0.0           # rad


class MonitoringBuoy:
    """Oceanographic monitoring buoy model.

    Provides heave dynamics, mooring tension estimates, and a simple
    energy budget for solar-powered operations.
    """

    WATER_DENSITY: float = 1025.0  # kg/m³
    GRAVITY: float = 9.81          # m/s²
    SOLAR_IRRADIANCE: float = 1000.0  # W/m² (peak)

    def __init__(
        self,
        geometry: BuoyGeometry | None = None,
        power: PowerSystem | None = None,
        sensors: SensorPayload | None = None,
    ) -> None:
        self.geometry = geometry or BuoyGeometry()
        self.power = power or PowerSystem()
        self.sensors = sensors or SensorPayload()
        self.state = BuoyState()

    # ------------------------------------------------------------------
    # Hydrostatics
    # ------------------------------------------------------------------

    @property
    def hydrostatic_stiffness(self) -> float:
        """Heave hydrostatic restoring stiffness (N/m)."""
        return self.WATER_DENSITY * self.GRAVITY * self.geometry.waterplane_area

    @property
    def added_mass_heave(self) -> float:
        """Approximate added mass in heave for a circular disc (kg)."""
        r = self.geometry.diameter / 2.0
        return (8.0 / 3.0) * self.WATER_DENSITY * r ** 3

    # ------------------------------------------------------------------
    # Mooring
    # ------------------------------------------------------------------

    def mooring_tension(self, surge_offset: float = 0.0) -> float:
        """Estimate quasi-static mooring tension (N).

        Args:
            surge_offset: Horizontal offset from nominal position (m).

        Returns:
            Mooring line tension in newtons.
        """
        # Approximate catenary restoring as linear spring in the horizontal
        horizontal_tension = self.geometry.mooring_stiffness * abs(surge_offset)
        # Add static pretension from buoy net buoyancy
        net_buoyancy = (
            self.WATER_DENSITY
            * self.GRAVITY
            * self.geometry.waterplane_area
            * 0.5  # estimate submerged depth as half buoy height
        ) - self.geometry.mass * self.GRAVITY
        pretension = max(net_buoyancy, 0.0)
        return pretension + horizontal_tension

    # ------------------------------------------------------------------
    # Heave dynamics
    # ------------------------------------------------------------------

    def natural_period(self) -> float:
        """Natural period in heave (s)."""
        total_mass = self.geometry.mass + self.added_mass_heave
        omega_n = math.sqrt(self.hydrostatic_stiffness / total_mass)
        return 2.0 * math.pi / omega_n

    def step(self, wave_force: float, dt: float) -> None:
        """Advance heave dynamics by one time step.

        Args:
            wave_force: Vertical excitation force from waves (N).
            dt: Time step (s).
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        total_mass = self.geometry.mass + self.added_mass_heave
        z = self.state.heave
        zdot = self.state.heave_velocity

        # Damping: 5 % of critical (mooring + viscous)
        b = 0.05 * 2.0 * math.sqrt(total_mass * self.hydrostatic_stiffness)
        c = self.hydrostatic_stiffness + self.geometry.mooring_stiffness

        zdotdot = (wave_force - b * zdot - c * z) / total_mass
        self.state.heave_velocity = zdot + zdotdot * dt
        self.state.heave = z + zdot * dt
        self.state.mooring_tension = self.mooring_tension()

    # ------------------------------------------------------------------
    # Power budget
    # ------------------------------------------------------------------

    def solar_generation(self, solar_fraction: float = 1.0) -> float:
        """Instantaneous solar power generation (W).

        Args:
            solar_fraction: Fraction of peak irradiance (0–1).

        Returns:
            Generated power in watts.
        """
        return (
            self.power.solar_panel_area
            * self.SOLAR_IRRADIANCE
            * self.power.solar_efficiency
            * max(0.0, min(solar_fraction, 1.0))
        )

    def update_battery(self, dt: float, solar_fraction: float = 0.5) -> None:
        """Update battery state over a time step.

        Args:
            dt: Time step (s).
            solar_fraction: Fractional solar irradiance (0–1).
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        gen = self.solar_generation(solar_fraction)
        net_power = gen - self.power.load_power_w
        energy_wh = net_power * dt / 3600.0
        self.power.battery_level = max(
            0.0,
            min(
                1.0,
                self.power.battery_level
                + energy_wh / max(self.power.battery_capacity_wh, 1e-9),
            ),
        )

    def data_availability(self) -> bool:
        """Return True if battery has sufficient charge to transmit data."""
        return self.power.battery_level > 0.1

    def __repr__(self) -> str:
        g = self.geometry
        return (
            f"MonitoringBuoy(diameter={g.diameter}m, mass={g.mass}kg, "
            f"sensors={self.sensors.active_count})"
        )
