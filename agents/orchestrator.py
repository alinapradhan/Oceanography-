"""Orchestrator – coordinates the multi-agent design-to-operations pipeline.

The :class:`Orchestrator` manages the full lifecycle of a marine system:

1. **Design** – :class:`DesignAgent` generates initial parameters.
2. **Optimization** – :class:`OptimizationAgent` refines them.
3. **Simulation** – :class:`SimulationAgent` validates performance.
4. **Operations** – :class:`OperationsAgent` monitors the deployed system.

Agents communicate via a shared context dictionary that the orchestrator
routes between stages.
"""

from __future__ import annotations

import logging
from typing import Any

from .base_agent import AgentMessage, BaseAgent
from .design_agent import DesignAgent
from .optimization_agent import OptimizationAgent
from .simulation_agent import SimulationAgent
from .operations_agent import OperationsAgent


class Orchestrator:
    """Coordinates the multi-agent design-to-operations pipeline.

    Usage::

        orch = Orchestrator()
        result = orch.run_design_pipeline(
            system_type="wec",
            wave_height=3.0,
            wave_period=9.0,
            water_depth=50.0,
        )
    """

    def __init__(self, verbose: bool = False) -> None:
        self.design_agent = DesignAgent(verbose=verbose)
        self.optimization_agent = OptimizationAgent(verbose=verbose)
        self.simulation_agent = SimulationAgent(verbose=verbose)
        self.operations_agent = OperationsAgent(verbose=verbose)
        self._logger = logging.getLogger("oceanmech.orchestrator")
        self._pipeline_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Message routing between agents
    # ------------------------------------------------------------------

    def _route_messages(self) -> None:
        """Flush each agent's outbox and deliver messages to recipients."""
        agents: dict[str, BaseAgent] = {
            self.design_agent.name: self.design_agent,
            self.optimization_agent.name: self.optimization_agent,
            self.simulation_agent.name: self.simulation_agent,
            self.operations_agent.name: self.operations_agent,
        }
        for agent in agents.values():
            for msg in agent.flush_outbox():
                target = agents.get(msg.recipient)
                if target:
                    target.receive(msg)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run_design_pipeline(self, **env_params: Any) -> dict[str, Any]:
        """Execute the full design → optimize → simulate pipeline.

        Args:
            **env_params: Environmental and mission parameters, e.g.::

                system_type="auv", wave_height=2.0, wave_period=8.0,
                water_depth=100.0, current_speed=0.5

        Returns:
            Final result dict containing outputs from all pipeline stages.
        """
        self._logger.info("Orchestrator: starting design pipeline for %s", env_params.get("system_type"))

        # Stage 1: Design
        design_result = self.design_agent.run(env_params)
        self._pipeline_log.append({"stage": "design", "result": design_result})
        self._logger.info("Design stage complete: %s", design_result.get("system_type"))

        # Stage 2: Optimization (uses a simple objective based on system type)
        system_type = env_params.get("system_type", "auv")
        obj_fn = self._build_objective(system_type, env_params)
        param_bounds = self._build_bounds(system_type, design_result)

        opt_context = {
            "design": design_result,
            "objective_fn": obj_fn,
            "param_bounds": param_bounds,
            "maximize": True,
        }
        opt_result = self.optimization_agent.run(opt_context)
        self._pipeline_log.append({"stage": "optimization", "result": opt_result})

        # Stage 3: Simulation using optimized design
        sim_context = dict(env_params)
        sim_context["design"] = opt_result.get("optimized_design", design_result)
        sim_result = self.simulation_agent.run(sim_context)
        self._pipeline_log.append({"stage": "simulation", "result": sim_result})

        # Route messages between agents
        self._route_messages()

        return {
            "system_type": system_type,
            "design": design_result,
            "optimization": opt_result,
            "simulation": sim_result,
            "pipeline_stages_completed": 3,
        }

    def run_operations_check(
        self, system_type: str, telemetry: dict[str, Any], environment: dict[str, Any]
    ) -> dict[str, Any]:
        """Run the operations agent to assess system health.

        Args:
            system_type: One of ``"auv"``, ``"wec"``, ``"offshore_platform"``,
                ``"monitoring_buoy"``.
            telemetry: Dict of live sensor readings from the system.
            environment: Dict of current environmental conditions.

        Returns:
            Operations assessment with alerts and recommended actions.
        """
        ops_context = {
            "system_type": system_type,
            "telemetry": telemetry,
            "environment": environment,
        }
        result = self.operations_agent.run(ops_context)
        self._pipeline_log.append({"stage": "operations", "result": result})
        return result

    # ------------------------------------------------------------------
    # Objective / bound builders
    # ------------------------------------------------------------------

    def _build_objective(
        self, system_type: str, env: dict[str, Any]
    ) -> Any:
        """Return an objective function for the given system type."""
        hs = env.get("wave_height", 2.0)
        tp = env.get("wave_period", 8.0)
        depth = env.get("water_depth", 100.0)

        if system_type == "auv":
            def auv_range(design: dict) -> float:
                from models.mechanical.auv import AUV, AUVGeometry, AUVPropulsion
                geom = AUVGeometry(
                    length=design.get("length_m", 2.0),
                    diameter=design.get("diameter_m", 0.2),
                    mass=design.get("estimated_mass_kg", 30.0),
                )
                auv = AUV(geometry=geom)
                return auv.range_estimate(500.0, 1.5)
            return auv_range

        if system_type == "wec":
            def wec_power(design: dict) -> float:
                from models.mechanical.wave_energy_converter import (
                    WaveEnergyConverter, WECGeometry,
                )
                geom = WECGeometry(
                    radius=design.get("float_radius_m", 5.0),
                    draft=design.get("draft_m", 4.0),
                )
                wec = WaveEnergyConverter(geometry=geom)
                return wec.average_power(hs, tp)
            return wec_power

        if system_type == "offshore_platform":
            def platform_freq_margin(design: dict) -> float:
                from models.mechanical.offshore_platform import (
                    OffshorePlatform, PlatformGeometry,
                )
                geom = PlatformGeometry(
                    leg_diameter=design.get("leg_diameter_m", 1.2),
                    leg_count=design.get("leg_count", 4),
                    water_depth=design.get("water_depth_m", depth),
                )
                platform = OffshorePlatform(geometry=geom)
                fn = platform.natural_frequency()
                wave_freq = 1.0 / tp
                return abs(fn - wave_freq)  # maximise separation
            return platform_freq_margin

        # monitoring_buoy or default: maximise battery duration
        def buoy_battery(design: dict) -> float:
            return design.get("battery_capacity_wh", 1000.0)
        return buoy_battery

    def _build_bounds(
        self, system_type: str, design: dict[str, Any]
    ) -> dict[str, tuple[float, float]]:
        """Return optimisation bounds for numeric design parameters."""
        if system_type == "auv":
            return {
                "length_m": (0.5, 6.0),
                "diameter_m": (0.05, 0.8),
            }
        if system_type == "wec":
            return {
                "float_radius_m": (1.0, 20.0),
                "draft_m": (1.0, 10.0),
            }
        if system_type == "offshore_platform":
            return {
                "leg_diameter_m": (0.5, 3.0),
            }
        return {
            "battery_capacity_wh": (500.0, 10000.0),
        }

    @property
    def pipeline_log(self) -> list[dict[str, Any]]:
        """Return the full pipeline execution log."""
        return list(self._pipeline_log)
