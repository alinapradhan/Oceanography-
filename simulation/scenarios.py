"""Pre-defined simulation scenarios for common marine engineering use cases.

Each scenario function wires together a mechanical model, an ocean
environment, and a simulation engine, returning a :class:`SimulationResult`.
"""

from __future__ import annotations

from typing import Any

from .engine import SimulationEngine, SimulationResult
from models.mechanical.auv import AUV, AUVGeometry, AUVPropulsion
from models.mechanical.wave_energy_converter import WaveEnergyConverter, WECGeometry
from models.mechanical.offshore_platform import OffshorePlatform, PlatformGeometry
from models.mechanical.monitoring_buoy import MonitoringBuoy, BuoyGeometry
from models.ocean.ocean_environment import OceanEnvironment, OceanState
from models.ocean.wave_model import LinearWaveKinematics
import math


def auv_transit_scenario(
    wave_height: float = 2.0,
    wave_period: float = 8.0,
    current_speed: float = 0.5,
    target_speed: float = 1.5,
    duration: float = 300.0,
    dt: float = 0.1,
) -> SimulationResult:
    """Simulate an AUV transiting against a current.

    Args:
        wave_height: Significant wave height (m).
        wave_period: Peak wave period (s).
        current_speed: Opposing current speed (m/s).
        target_speed: AUV commanded speed (m/s).
        duration: Simulation duration (s).
        dt: Time step (s).

    Returns:
        :class:`SimulationResult` with ``speed`` and ``position`` channels.
    """
    auv = AUV(
        geometry=AUVGeometry(length=2.0, diameter=0.2, mass=30.0),
        propulsion=AUVPropulsion(max_thrust=80.0, max_speed=2.5),
    )
    engine = SimulationEngine(dt=dt, duration=duration)

    def step(t: float, state: dict[str, Any]) -> dict[str, float]:
        thrust = min(auv.propulsion.max_thrust, auv.required_thrust(target_speed) * 1.1)
        auv.step(thrust=thrust, dt=dt, current_speed=current_speed)
        return {
            "speed_ms": auv.state.velocity[0],
            "position_m": auv.state.pose.x,
        }

    return engine.run(
        "auv",
        step,
        metadata={
            "wave_height": wave_height,
            "wave_period": wave_period,
            "current_speed": current_speed,
            "target_speed": target_speed,
        },
    )


def wec_power_scenario(
    wave_height: float = 2.0,
    wave_period: float = 8.0,
    water_depth: float = 50.0,
    float_radius: float = 5.0,
    duration: float = 300.0,
    dt: float = 0.1,
) -> SimulationResult:
    """Simulate a WEC generating power in regular waves.

    Args:
        wave_height: Wave height (m).
        wave_period: Wave period (s).
        water_depth: Water depth (m).
        float_radius: WEC float radius (m).
        duration: Simulation duration (s).
        dt: Time step (s).

    Returns:
        :class:`SimulationResult` with ``power_w`` and ``displacement_m`` channels.
    """
    wec = WaveEnergyConverter(geometry=WECGeometry(radius=float_radius, draft=float_radius * 0.8))
    kinematics = LinearWaveKinematics(wave_height / 2.0, wave_period, water_depth)
    engine = SimulationEngine(dt=dt, duration=duration)
    f_exc_amp = wec.excitation_force(wave_height / 2.0, wave_period)

    def step(t: float, state: dict[str, Any]) -> dict[str, float]:
        excitation = f_exc_amp * math.sin(2.0 * math.pi * t / wave_period)
        power = wec.step(excitation=excitation, dt=dt)
        return {
            "power_w": power,
            "displacement_m": wec.state.displacement,
        }

    return engine.run(
        "wec",
        step,
        metadata={
            "wave_height": wave_height,
            "wave_period": wave_period,
            "float_radius": float_radius,
        },
    )


def platform_storm_scenario(
    water_depth: float = 100.0,
    leg_diameter: float = 1.2,
    storm_wave_heights: list[float] | None = None,
    wave_period: float = 14.0,
    wind_speed: float = 35.0,
) -> SimulationResult:
    """Simulate a jacket platform under a sequence of storm sea states.

    Args:
        water_depth: Water depth (m).
        leg_diameter: Leg outer diameter (m).
        storm_wave_heights: List of wave heights to apply sequentially (m).
        wave_period: Wave period during storm (s).
        wind_speed: Wind speed (m/s).

    Returns:
        :class:`SimulationResult` with ``base_shear_n`` and ``fatigue_damage`` channels.
    """
    if storm_wave_heights is None:
        storm_wave_heights = [3.0, 6.0, 9.0, 12.0, 9.0, 6.0]

    platform = OffshorePlatform(
        geometry=PlatformGeometry(
            leg_diameter=leg_diameter,
            water_depth=water_depth,
        )
    )
    engine = SimulationEngine(dt=1.0, duration=float(len(storm_wave_heights)))

    def step(t: float, state: dict[str, Any]) -> dict[str, float]:
        idx = min(int(t), len(storm_wave_heights) - 1)
        hs = storm_wave_heights[idx]
        platform.update_loads(hs, wave_period, wind_speed)
        stress = platform.state.base_shear / (
            4.0 * math.pi * leg_diameter * 0.05 * 1e6
        )
        platform.accumulate_fatigue(stress_range=abs(stress), cycles=1.0)
        return {
            "base_shear_n": platform.state.base_shear,
            "overturning_moment_nm": platform.state.overturning_moment,
            "fatigue_damage": platform.state.fatigue_damage,
        }

    return engine.run(
        "offshore_platform",
        step,
        metadata={
            "water_depth": water_depth,
            "leg_diameter": leg_diameter,
            "wave_period": wave_period,
        },
    )


def buoy_deployment_scenario(
    wave_height: float = 1.5,
    wave_period: float = 7.0,
    water_depth: float = 50.0,
    duration: float = 86400.0,
    dt: float = 60.0,
) -> SimulationResult:
    """Simulate a monitoring buoy over a 24-hour deployment.

    Args:
        wave_height: Significant wave height (m).
        wave_period: Peak wave period (s).
        water_depth: Water depth / mooring length (m).
        duration: Simulation duration (s, default 24 h).
        dt: Time step (s, default 60 s).

    Returns:
        :class:`SimulationResult` with ``heave_m`` and ``battery_level`` channels.
    """
    buoy = MonitoringBuoy(geometry=BuoyGeometry(mooring_depth=water_depth))
    kinematics = LinearWaveKinematics(wave_height / 2.0, wave_period, water_depth)
    engine = SimulationEngine(dt=dt, duration=duration)

    def step(t: float, state: dict[str, Any]) -> dict[str, float]:
        fz = kinematics.vertical_velocity(z=0.0, t=t) * buoy.geometry.diameter
        buoy.step(wave_force=fz, dt=dt)
        # Solar fraction varies with time of day (sinusoidal)
        solar = max(0.0, math.sin(2.0 * math.pi * t / 86400.0))
        buoy.update_battery(dt=dt, solar_fraction=solar)
        return {
            "heave_m": buoy.state.heave,
            "battery_level": buoy.power.battery_level,
        }

    return engine.run(
        "monitoring_buoy",
        step,
        metadata={
            "wave_height": wave_height,
            "wave_period": wave_period,
            "water_depth": water_depth,
        },
    )
