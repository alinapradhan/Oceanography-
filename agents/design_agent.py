"""Design Agent – generates initial design proposals for marine systems.

The :class:`DesignAgent` takes environmental requirements (wave height, depth,
operational profile) and applies engineering heuristics to produce candidate
design parameters for AUVs, WECs, offshore platforms, and monitoring buoys.
"""

from __future__ import annotations

import math
from typing import Any

from .base_agent import BaseAgent


class DesignAgent(BaseAgent):
    """Generates initial parametric designs for marine mechanical systems.

    Supported ``system_type`` values in context:
    - ``"auv"``
    - ``"wec"``
    - ``"offshore_platform"``
    - ``"monitoring_buoy"``
    """

    def __init__(self, name: str = "DesignAgent", verbose: bool = False) -> None:
        super().__init__(name=name, verbose=verbose)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def perceive(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "system_type": context.get("system_type", "auv"),
            "wave_height": context.get("wave_height", 2.0),
            "wave_period": context.get("wave_period", 8.0),
            "water_depth": context.get("water_depth", 100.0),
            "current_speed": context.get("current_speed", 0.5),
            "payload_mass": context.get("payload_mass", 5.0),
            "target_speed": context.get("target_speed", 1.5),
            "deployment_duration_days": context.get("deployment_duration_days", 30),
        }

    def reason(self, observations: dict[str, Any]) -> dict[str, Any]:
        system = observations["system_type"]
        if system == "auv":
            return self._design_auv(observations)
        if system == "wec":
            return self._design_wec(observations)
        if system == "offshore_platform":
            return self._design_platform(observations)
        if system == "monitoring_buoy":
            return self._design_buoy(observations)
        raise ValueError(f"Unknown system_type: {system!r}")

    def act(self, decision: dict[str, Any]) -> dict[str, Any]:
        decision["agent"] = self.name
        decision["status"] = "design_proposed"
        return decision

    # ------------------------------------------------------------------
    # System-specific design heuristics
    # ------------------------------------------------------------------

    def _design_auv(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Heuristic AUV sizing rules."""
        payload_mass = obs["payload_mass"]
        target_speed = obs["target_speed"]
        current_speed = obs["current_speed"]

        # Hull size: payload drives volume estimate (0.5 kg/L electronic payload)
        payload_volume = payload_mass / 500.0   # m³ (rough)
        # Structure + systems ~4× payload volume
        total_volume = 5.0 * payload_volume
        diameter = max(0.1, (4.0 * total_volume / (math.pi * 8.0)) ** (1.0 / 3.0))
        length = 8.0 * diameter  # L/D ratio ≈ 8

        # Required thrust at max speed against current
        effective_speed = target_speed + current_speed
        area = math.pi * (diameter / 2.0) ** 2
        cd = 0.03
        rho = 1025.0
        thrust = 0.5 * rho * cd * area * effective_speed ** 2

        return {
            "system_type": "auv",
            "length_m": round(length, 3),
            "diameter_m": round(diameter, 3),
            "required_thrust_n": round(thrust, 2),
            "estimated_mass_kg": round(rho * total_volume * 0.6, 2),
        }

    def _design_wec(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Heuristic WEC sizing rules based on wave resonance."""
        hs = obs["wave_height"]
        tp = obs["wave_period"]
        d = obs["water_depth"]

        # Natural period matches peak wave period → radius from resonance condition
        g = 9.81
        rho = 1025.0
        # C₃₃ / (m + m_a) = (2π/Tp)²
        # For a cylinder: C₃₃ = ρg·πr², m_a ≈ ρπr³/3
        # Choosing draft ≈ 0.4·r
        # → simplified: r ≈ g·Tp²/(4π²·(1+ρπ/3)) — iterative
        r = 1.0
        for _ in range(30):
            area = math.pi * r ** 2
            stiffness = rho * g * area
            draft = max(0.3 * r, 1.0)
            m_a = rho * math.pi * r ** 3 / 3.0
            mass = rho * area * draft * 0.3  # 30 % solid cylinder
            omega_n = math.sqrt(stiffness / (mass + m_a))
            target_omega = 2.0 * math.pi / tp
            r += 0.1 * (target_omega - omega_n) / target_omega
            r = max(1.0, min(r, 20.0))

        draft = max(0.3 * r, 1.0)
        return {
            "system_type": "wec",
            "float_radius_m": round(r, 2),
            "draft_m": round(draft, 2),
            "rated_wave_height_m": round(hs, 2),
            "resonant_period_s": round(tp, 2),
        }

    def _design_platform(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Heuristic jacket platform leg sizing."""
        depth = obs["water_depth"]
        hs = obs["wave_height"]

        # Leg diameter: API RP 2A rule-of-thumb D ≈ 0.01·depth
        leg_d = max(0.5, 0.012 * depth)
        # Number of legs based on depth
        n_legs = 4 if depth < 150 else 8
        # Wall thickness ≈ D/20
        wall_t = leg_d / 20.0

        return {
            "system_type": "offshore_platform",
            "water_depth_m": depth,
            "leg_diameter_m": round(leg_d, 3),
            "leg_count": n_legs,
            "wall_thickness_m": round(wall_t, 4),
            "design_wave_height_m": round(hs * 1.86, 2),  # Hs → 100-yr extreme
        }

    def _design_buoy(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Heuristic monitoring buoy sizing."""
        days = obs["deployment_duration_days"]
        hs = obs["wave_height"]

        # Hull diameter scales with payload + battery needs
        diameter = max(0.8, 0.05 * days ** 0.5 + 0.5)
        # Battery capacity for 20 W load with 50 % solar contribution
        net_load_w = 20.0 * 0.5  # W (net after solar in daylight hours)
        battery_wh = net_load_w * 24.0 * days * 0.2  # 20 % buffer
        solar_area = min(2.0, 0.2 * diameter ** 2)

        return {
            "system_type": "monitoring_buoy",
            "hull_diameter_m": round(diameter, 2),
            "battery_capacity_wh": round(battery_wh, 1),
            "solar_panel_area_m2": round(solar_area, 2),
            "design_wave_height_m": round(hs, 2),
        }
