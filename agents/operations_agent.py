"""Operations Agent – monitors deployed systems and issues control actions.

The :class:`OperationsAgent` ingests sensor data from deployed marine
systems, detects anomalies, and generates control recommendations
(mission re-planning for AUVs, load-shedding for buoys, etc.).
"""

from __future__ import annotations

from typing import Any

from .base_agent import BaseAgent


class OperationsAgent(BaseAgent):
    """Monitors system health and generates operational recommendations.

    Decision logic is intentionally rule-based so it operates autonomously
    without external inference services.
    """

    # Thresholds that trigger alerts / actions
    THRESHOLDS: dict[str, dict[str, float]] = {
        "auv": {
            "battery_low": 0.15,
            "max_depth": 500.0,
            "collision_proximity_m": 10.0,
        },
        "wec": {
            "max_displacement_m": 3.0,
            "min_power_w": 100.0,
        },
        "offshore_platform": {
            "fatigue_warning": 0.5,
            "fatigue_critical": 0.8,
            "max_wave_height_m": 15.0,
        },
        "monitoring_buoy": {
            "battery_low": 0.10,
            "max_heave_m": 2.5,
        },
    }

    def __init__(self, name: str = "OperationsAgent", verbose: bool = False) -> None:
        super().__init__(name=name, verbose=verbose)
        self._alert_log: list[dict[str, Any]] = []

    def perceive(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "system_type": context.get("system_type", "auv"),
            "telemetry": context.get("telemetry", {}),
            "environment": context.get("environment", {}),
        }

    def reason(self, observations: dict[str, Any]) -> dict[str, Any]:
        system = observations["system_type"]
        telemetry = observations["telemetry"]
        env = observations["environment"]
        alerts: list[str] = []
        actions: list[str] = []

        thresholds = self.THRESHOLDS.get(system, {})

        if system == "auv":
            self._check_auv(telemetry, thresholds, alerts, actions)
        elif system == "wec":
            self._check_wec(telemetry, thresholds, alerts, actions)
        elif system == "offshore_platform":
            self._check_platform(telemetry, env, thresholds, alerts, actions)
        elif system == "monitoring_buoy":
            self._check_buoy(telemetry, thresholds, alerts, actions)

        self._alert_log.extend({"system": system, "alert": a} for a in alerts)

        return {
            "system_type": system,
            "alerts": alerts,
            "recommended_actions": actions,
            "operational_status": "degraded" if alerts else "nominal",
        }

    def act(self, decision: dict[str, Any]) -> dict[str, Any]:
        decision["agent"] = self.name
        decision["status"] = "operations_assessed"
        return decision

    # ------------------------------------------------------------------
    # System-specific checks
    # ------------------------------------------------------------------

    def _check_auv(
        self,
        tel: dict,
        thr: dict,
        alerts: list[str],
        actions: list[str],
    ) -> None:
        if tel.get("battery_level", 1.0) < thr.get("battery_low", 0.15):
            alerts.append("AUV battery critically low")
            actions.append("Abort mission – return to surface for recovery")
        if tel.get("depth_m", 0.0) > thr.get("max_depth", 500.0):
            alerts.append("AUV exceeding maximum rated depth")
            actions.append("Emergency ascent initiated")
        if tel.get("obstacle_proximity_m", 999) < thr.get("collision_proximity_m", 10.0):
            alerts.append("Collision risk detected")
            actions.append("Execute avoidance manoeuvre")

    def _check_wec(
        self,
        tel: dict,
        thr: dict,
        alerts: list[str],
        actions: list[str],
    ) -> None:
        if abs(tel.get("displacement_m", 0.0)) > thr.get("max_displacement_m", 3.0):
            alerts.append("WEC heave displacement exceeds safe limit")
            actions.append("Activate hydraulic end-stops")
        if tel.get("power_w", 1000.0) < thr.get("min_power_w", 100.0):
            alerts.append("WEC power output below minimum threshold")
            actions.append("Check PTO system and mooring")

    def _check_platform(
        self,
        tel: dict,
        env: dict,
        thr: dict,
        alerts: list[str],
        actions: list[str],
    ) -> None:
        fatigue = tel.get("fatigue_damage", 0.0)
        if fatigue > thr.get("fatigue_critical", 0.8):
            alerts.append("Platform fatigue damage critical")
            actions.append("Immediate structural inspection required")
        elif fatigue > thr.get("fatigue_warning", 0.5):
            alerts.append("Platform fatigue damage elevated")
            actions.append("Schedule inspection at next maintenance window")
        if env.get("wave_height_m", 0.0) > thr.get("max_wave_height_m", 15.0):
            alerts.append("Extreme wave event detected")
            actions.append("Activate storm-survival mode")

    def _check_buoy(
        self,
        tel: dict,
        thr: dict,
        alerts: list[str],
        actions: list[str],
    ) -> None:
        if tel.get("battery_level", 1.0) < thr.get("battery_low", 0.10):
            alerts.append("Buoy battery critically low")
            actions.append("Enter low-power mode – suspend non-essential sensors")
        if abs(tel.get("heave_m", 0.0)) > thr.get("max_heave_m", 2.5):
            alerts.append("Buoy heave exceeding survival limit")
            actions.append("Deploy sea-anchor / storm drogue")

    @property
    def alert_history(self) -> list[dict[str, Any]]:
        """Return all accumulated alerts."""
        return list(self._alert_log)
