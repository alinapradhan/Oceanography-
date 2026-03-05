"""Tests for the agentic AI architecture."""

import pytest

from agents.base_agent import AgentMessage, AgentStatus
from agents.design_agent import DesignAgent
from agents.optimization_agent import OptimizationAgent
from agents.simulation_agent import SimulationAgent
from agents.operations_agent import OperationsAgent
from agents.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Base agent messaging
# ---------------------------------------------------------------------------

class ConcreteAgent:
    """Minimal concrete wrapper using DesignAgent as a proxy for BaseAgent."""

    def __init__(self):
        self._agent = DesignAgent(name="TestAgent")

    def test_send_receive(self):
        msg = self._agent.send("OtherAgent", "greeting", {"text": "hello"})
        assert isinstance(msg, AgentMessage)
        assert msg.sender == "TestAgent"
        assert msg.recipient == "OtherAgent"

    def test_flush_outbox(self):
        self._agent.send("X", "subject")
        messages = self._agent.flush_outbox()
        assert len(messages) == 1
        assert self._agent.flush_outbox() == []

    def test_receive_and_flush_inbox(self):
        msg = AgentMessage(sender="Y", recipient="TestAgent", subject="data")
        self._agent.receive(msg)
        inbox = self._agent.flush_inbox()
        assert len(inbox) == 1
        assert self._agent.flush_inbox() == []


class TestBaseAgentMessaging:
    def setup_method(self):
        self.agent = DesignAgent(name="MsgTest")

    def test_send_creates_message(self):
        msg = self.agent.send("Recipient", "test_subject", {"key": "value"})
        assert msg.subject == "test_subject"
        assert msg.payload["key"] == "value"

    def test_outbox_flush(self):
        self.agent.send("R", "s1")
        self.agent.send("R", "s2")
        msgs = self.agent.flush_outbox()
        assert len(msgs) == 2
        assert self.agent.flush_outbox() == []

    def test_inbox_receive(self):
        m = AgentMessage(sender="S", recipient="MsgTest", subject="hi")
        self.agent.receive(m)
        inbox = self.agent.flush_inbox()
        assert inbox[0].subject == "hi"

    def test_repr(self):
        r = repr(self.agent)
        assert "DesignAgent" in r
        assert "MsgTest" in r


# ---------------------------------------------------------------------------
# DesignAgent
# ---------------------------------------------------------------------------

class TestDesignAgent:
    def setup_method(self):
        self.agent = DesignAgent()

    def _run(self, system_type, **kwargs):
        ctx = {"system_type": system_type, **kwargs}
        return self.agent.run(ctx)

    def test_design_auv(self):
        result = self._run("auv", wave_height=2.0, water_depth=100.0, current_speed=0.5)
        assert result["system_type"] == "auv"
        assert "length_m" in result
        assert result["length_m"] > 0

    def test_design_wec(self):
        result = self._run("wec", wave_height=2.5, wave_period=9.0, water_depth=50.0)
        assert result["system_type"] == "wec"
        assert "float_radius_m" in result
        assert result["float_radius_m"] > 0

    def test_design_platform(self):
        result = self._run("offshore_platform", wave_height=5.0, water_depth=120.0)
        assert result["system_type"] == "offshore_platform"
        assert "leg_diameter_m" in result

    def test_design_buoy(self):
        result = self._run("monitoring_buoy", deployment_duration_days=60)
        assert result["system_type"] == "monitoring_buoy"
        assert "battery_capacity_wh" in result

    def test_unknown_system_type(self):
        with pytest.raises(ValueError):
            self._run("unknown_system")

    def test_status_after_run(self):
        self._run("auv")
        assert self.agent.status == AgentStatus.COMPLETED


# ---------------------------------------------------------------------------
# OptimizationAgent
# ---------------------------------------------------------------------------

class TestOptimizationAgent:
    def setup_method(self):
        self.agent = OptimizationAgent(max_iterations=20)

    def test_maximize_simple_function(self):
        """Maximize f(x) = -(x-3)² + 9, optimum at x=3."""

        def obj(d):
            x = d["x"]
            return -(x - 3.0) ** 2 + 9.0

        ctx = {
            "design": {"x": 0.0},
            "objective_fn": obj,
            "param_bounds": {"x": (0.0, 6.0)},
            "maximize": True,
        }
        result = self.agent.run(ctx)
        assert result["best_score"] > 0.0
        assert abs(result["optimized_design"]["x"] - 3.0) < 0.6

    def test_minimize_simple_function(self):
        """Minimise f(x) = x², optimum at x=0."""

        def obj(d):
            return d["x"] ** 2

        ctx = {
            "design": {"x": 5.0},
            "objective_fn": obj,
            "param_bounds": {"x": (0.0, 10.0)},
            "maximize": False,
        }
        result = self.agent.run(ctx)
        assert result["best_score"] <= 25.0
        # Should move toward 0
        assert result["optimized_design"]["x"] < 5.0

    def test_missing_context_keys(self):
        with pytest.raises(KeyError):
            self.agent.run({"design": {}})

    def test_history_non_empty(self):
        ctx = {
            "design": {"x": 1.0},
            "objective_fn": lambda d: d["x"],
            "param_bounds": {"x": (0.0, 5.0)},
        }
        result = self.agent.run(ctx)
        assert len(result["score_history"]) >= 1


# ---------------------------------------------------------------------------
# OperationsAgent
# ---------------------------------------------------------------------------

class TestOperationsAgent:
    def setup_method(self):
        self.agent = OperationsAgent()

    def _run(self, system_type, telemetry, environment=None):
        return self.agent.run({
            "system_type": system_type,
            "telemetry": telemetry,
            "environment": environment or {},
        })

    def test_nominal_auv(self):
        result = self._run("auv", {"battery_level": 0.8, "depth_m": 100.0})
        assert result["operational_status"] == "nominal"
        assert result["alerts"] == []

    def test_low_battery_auv(self):
        result = self._run("auv", {"battery_level": 0.05})
        assert "AUV battery critically low" in result["alerts"]
        assert len(result["recommended_actions"]) > 0

    def test_deep_dive_alert(self):
        result = self._run("auv", {"battery_level": 0.9, "depth_m": 600.0})
        assert any("depth" in a.lower() for a in result["alerts"])

    def test_wec_nominal(self):
        result = self._run("wec", {"displacement_m": 0.5, "power_w": 500.0})
        assert result["operational_status"] == "nominal"

    def test_wec_over_displacement(self):
        result = self._run("wec", {"displacement_m": 5.0, "power_w": 500.0})
        assert len(result["alerts"]) > 0

    def test_platform_fatigue_warning(self):
        result = self._run(
            "offshore_platform",
            {"fatigue_damage": 0.6},
            {"wave_height_m": 5.0},
        )
        assert any("fatigue" in a.lower() for a in result["alerts"])

    def test_buoy_low_battery(self):
        result = self._run("monitoring_buoy", {"battery_level": 0.05, "heave_m": 0.2})
        assert any("battery" in a.lower() for a in result["alerts"])

    def test_alert_history_accumulated(self):
        self._run("auv", {"battery_level": 0.05})
        self._run("auv", {"battery_level": 0.05})
        assert len(self.agent.alert_history) >= 2


# ---------------------------------------------------------------------------
# Orchestrator (integration test)
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def setup_method(self):
        self.orch = Orchestrator()

    def test_auv_pipeline(self):
        result = self.orch.run_design_pipeline(
            system_type="auv",
            wave_height=2.0,
            wave_period=8.0,
            water_depth=100.0,
            current_speed=0.5,
            duration=30.0,
            dt=0.5,
        )
        assert result["system_type"] == "auv"
        assert result["pipeline_stages_completed"] == 3
        assert "design" in result
        assert "optimization" in result
        assert "simulation" in result

    def test_wec_pipeline(self):
        result = self.orch.run_design_pipeline(
            system_type="wec",
            wave_height=2.5,
            wave_period=9.0,
            water_depth=50.0,
            duration=30.0,
            dt=0.5,
        )
        assert result["system_type"] == "wec"

    def test_platform_pipeline(self):
        result = self.orch.run_design_pipeline(
            system_type="offshore_platform",
            wave_height=4.0,
            wave_period=12.0,
            water_depth=80.0,
            duration=30.0,
            dt=0.5,
        )
        assert result["system_type"] == "offshore_platform"

    def test_buoy_pipeline(self):
        result = self.orch.run_design_pipeline(
            system_type="monitoring_buoy",
            wave_height=1.5,
            wave_period=7.0,
            water_depth=40.0,
            duration=30.0,
            dt=0.5,
        )
        assert result["system_type"] == "monitoring_buoy"

    def test_operations_check_nominal(self):
        result = self.orch.run_operations_check(
            system_type="auv",
            telemetry={"battery_level": 0.7, "depth_m": 50.0},
            environment={},
        )
        assert result["operational_status"] == "nominal"

    def test_pipeline_log_populated(self):
        self.orch.run_design_pipeline(
            system_type="auv", duration=10.0, dt=1.0
        )
        assert len(self.orch.pipeline_log) > 0
