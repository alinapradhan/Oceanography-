"""Simulation Agent – runs time-domain simulations of marine systems.

Executes a simulation scenario using the specified mechanical model and
ocean environment, collecting time-series outputs for downstream analysis.
"""

from __future__ import annotations

import math
from typing import Any

from .base_agent import BaseAgent


class SimulationAgent(BaseAgent):
    """Runs physics-based time-domain simulations.

    Accepts a fully-parametrised design and environmental conditions,
    executes a simulation, and returns structured results including
    time histories of key state variables.
    """

    def __init__(
        self,
        name: str = "SimulationAgent",
        default_duration: float = 600.0,
        default_dt: float = 0.1,
        verbose: bool = False,
    ) -> None:
        super().__init__(name=name, verbose=verbose)
        self.default_duration = default_duration
        self.default_dt = default_dt

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def perceive(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "system_type": context.get("system_type", "auv"),
            "design": context.get("design", {}),
            "wave_height": context.get("wave_height", 2.0),
            "wave_period": context.get("wave_period", 8.0),
            "current_speed": context.get("current_speed", 0.5),
            "water_depth": context.get("water_depth", 100.0),
            "duration": context.get("duration", self.default_duration),
            "dt": context.get("dt", self.default_dt),
        }

    def reason(self, observations: dict[str, Any]) -> dict[str, Any]:
        system = observations["system_type"]
        if system == "auv":
            return self._simulate_auv(observations)
        if system == "wec":
            return self._simulate_wec(observations)
        if system == "offshore_platform":
            return self._simulate_platform(observations)
        if system == "monitoring_buoy":
            return self._simulate_buoy(observations)
        raise ValueError(f"Unknown system_type: {system!r}")

    def act(self, decision: dict[str, Any]) -> dict[str, Any]:
        decision["agent"] = self.name
        decision["status"] = "simulation_complete"
        return decision

    # ------------------------------------------------------------------
    # System-specific simulations
    # ------------------------------------------------------------------

    def _simulate_auv(self, obs: dict[str, Any]) -> dict[str, Any]:
        from models.mechanical.auv import AUV, AUVGeometry, AUVPropulsion

        d = obs["design"]
        geom = AUVGeometry(
            length=d.get("length_m", 2.0),
            diameter=d.get("diameter_m", 0.2),
            mass=d.get("estimated_mass_kg", 30.0),
        )
        prop = AUVPropulsion(
            max_thrust=d.get("required_thrust_n", 50.0) * 1.5,
        )
        auv = AUV(geometry=geom, propulsion=prop)

        dt = obs["dt"]
        duration = obs["duration"]
        target_speed = obs.get("target_speed", 1.5)
        current_speed = obs["current_speed"]

        times, speeds, positions = [], [], []
        t = 0.0
        while t <= duration:
            speed = auv.state.velocity[0]
            thrust = min(prop.max_thrust, auv.required_thrust(target_speed) * 1.1)
            auv.step(thrust=thrust, dt=dt, current_speed=current_speed)
            times.append(round(t, 3))
            speeds.append(round(auv.state.velocity[0], 4))
            positions.append(round(auv.state.pose.x, 3))
            t += dt

        return {
            "system_type": "auv",
            "final_speed_ms": speeds[-1] if speeds else 0.0,
            "distance_covered_m": positions[-1] if positions else 0.0,
            "time_s": times,
            "speed_ms": speeds,
            "position_m": positions,
        }

    def _simulate_wec(self, obs: dict[str, Any]) -> dict[str, Any]:
        from models.mechanical.wave_energy_converter import (
            WaveEnergyConverter, WECGeometry, PTOParameters,
        )
        from models.ocean.wave_model import LinearWaveKinematics

        d = obs["design"]
        hs = obs["wave_height"]
        tp = obs["wave_period"]
        depth = obs["water_depth"]
        dt = obs["dt"]
        duration = obs["duration"]

        geom = WECGeometry(
            radius=d.get("float_radius_m", 5.0),
            draft=d.get("draft_m", 4.0),
        )
        wec = WaveEnergyConverter(geometry=geom)
        kinematics = LinearWaveKinematics(
            amplitude=hs / 2.0, period=tp, water_depth=depth
        )

        times, powers, displacements = [], [], []
        t = 0.0
        while t <= duration:
            fz = kinematics.vertical_velocity(z=0.0, t=t)
            excitation = wec.excitation_force(hs / 2.0, tp) * math.sin(
                2.0 * math.pi * t / tp
            )
            power = wec.step(excitation=excitation, dt=dt)
            times.append(round(t, 3))
            powers.append(round(power, 3))
            displacements.append(round(wec.state.displacement, 4))
            t += dt

        avg_power = sum(powers) / len(powers) if powers else 0.0
        return {
            "system_type": "wec",
            "average_power_w": round(avg_power, 2),
            "total_energy_j": round(wec.state.energy_captured, 2),
            "time_s": times,
            "power_w": powers,
            "displacement_m": displacements,
        }

    def _simulate_platform(self, obs: dict[str, Any]) -> dict[str, Any]:
        from models.mechanical.offshore_platform import OffshorePlatform, PlatformGeometry

        d = obs["design"]
        hs = obs["wave_height"]
        tp = obs["wave_period"]
        wind_speed = obs.get("wind_speed", 15.0)

        geom = PlatformGeometry(
            leg_diameter=d.get("leg_diameter_m", 1.2),
            leg_count=d.get("leg_count", 4),
            water_depth=d.get("water_depth_m", obs["water_depth"]),
        )
        platform = OffshorePlatform(geometry=geom)
        platform.update_loads(hs, tp, wind_speed)

        return {
            "system_type": "offshore_platform",
            "base_shear_n": round(platform.state.base_shear, 2),
            "overturning_moment_nm": round(platform.state.overturning_moment, 2),
            "natural_frequency_hz": round(platform.natural_frequency(), 4),
            "design_wave_height_m": d.get("design_wave_height_m", hs),
        }

    def _simulate_buoy(self, obs: dict[str, Any]) -> dict[str, Any]:
        from models.mechanical.monitoring_buoy import (
            MonitoringBuoy, BuoyGeometry, PowerSystem,
        )
        from models.ocean.wave_model import LinearWaveKinematics

        d = obs["design"]
        hs = obs["wave_height"]
        tp = obs["wave_period"]
        depth = obs["water_depth"]
        dt = obs["dt"]
        duration = obs["duration"]

        geom = BuoyGeometry(
            diameter=d.get("hull_diameter_m", 2.0),
            mooring_depth=depth,
        )
        power = PowerSystem(
            battery_capacity_wh=d.get("battery_capacity_wh", 1000.0),
            solar_panel_area=d.get("solar_panel_area_m2", 2.0),
        )
        buoy = MonitoringBuoy(geometry=geom, power=power)
        kinematics = LinearWaveKinematics(
            amplitude=hs / 2.0, period=tp, water_depth=depth
        )

        times, heaves, battery = [], [], []
        t = 0.0
        while t <= duration:
            fz = kinematics.vertical_velocity(z=0.0, t=t) * geom.diameter
            buoy.step(wave_force=fz, dt=dt)
            buoy.update_battery(dt=dt, solar_fraction=0.5)
            times.append(round(t, 3))
            heaves.append(round(buoy.state.heave, 4))
            battery.append(round(buoy.power.battery_level, 4))
            t += dt

        return {
            "system_type": "monitoring_buoy",
            "natural_period_s": round(buoy.natural_period(), 3),
            "max_heave_m": round(max(abs(h) for h in heaves), 3) if heaves else 0.0,
            "final_battery_level": battery[-1] if battery else 0.0,
            "time_s": times,
            "heave_m": heaves,
            "battery_level": battery,
        }
