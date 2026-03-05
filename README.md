
# OceanMech-Agent

**A production-ready modular research platform integrating Mechanical Engineering models, Oceanographic data systems, and Agentic AI for the autonomous design, optimization, simulation, and operation of marine systems.**



## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Marine Systems](#marine-systems)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Modules](#modules)
  - [Agents](#agents)
  - [Mechanical Models](#mechanical-models)
  - [Ocean Models](#ocean-models)
  - [Data Layer](#data-layer)
  - [Simulation](#simulation)
  - [Configuration](#configuration)
- [Running Tests](#running-tests)
- [Examples](#examples)
- [Design Decisions](#design-decisions)

---

## Overview

OceanMech-Agent is a research platform where **AI agents autonomously design, optimize, simulate, and operate marine mechanical systems**. It is built around four marine engineering domains:

| System | Description |
|---|---|
| **Autonomous Underwater Vehicles (AUVs)** | Hydrodynamic design, propulsion sizing, range estimation, and transit simulation |
| **Wave Energy Converters (WECs)** | Point-absorber sizing, JONSWAP spectrum modelling, power capture and PTO dynamics |
| **Offshore Platforms** | Jacket-type structural analysis, Morison wave loading, wind forces, and fatigue accumulation |
| **Ocean Monitoring Buoys** | Heave dynamics, mooring tension, solar power budget, and sensor payload management |

---

## Architecture

The platform uses a **multi-agent pipeline** built on a clean perception–reasoning–action loop:

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│DesignAgent  │────▶│OptimizationAgent │────▶│SimulationAgent   │────▶│OperationsAgent   │
│             │     │                  │     │                  │     │                  │
│ Heuristic   │     │ Hill-climbing    │     │ Time-domain      │     │ Rule-based       │
│ parametric  │     │ gradient-free    │     │ physics sim      │     │ anomaly          │
│ sizing      │     │ optimisation     │     │ (4 systems)      │     │ detection        │
└─────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
        │                    │                       │                        │
        └────────────────────┴───────────────────────┴────────────────────────┘
                                        │
                               ┌────────────────┐
                               │  Orchestrator  │
                               │  (pipeline     │
                               │   coordinator) │
                               └────────────────┘
```

Each agent implements three abstract methods from `BaseAgent`:
- **`perceive(context)`** – extract relevant inputs from the shared context
- **`reason(observations)`** – analyse and make a decision
- **`act(decision)`** – execute and return results

Agents communicate via `AgentMessage` objects routed by the `Orchestrator`.

---

## Marine Systems

### Autonomous Underwater Vehicle (AUV)
- Hull sizing from payload and speed requirements
- Axial and lateral hydrodynamic drag
- Surge equation of motion (Euler integration)
- Propulsion power and endurance/range estimation
- Battery-level tracking during mission

### Wave Energy Converter (WEC)
- Point-absorber geometry (cylindrical float)
- JONSWAP spectrum wave modelling
- Hydrostatic stiffness and added mass (frequency-independent)
- Linear PTO (power take-off) damping
- Froude–Krylov excitation force
- Average power and capture-width ratio calculation

### Offshore Platform
- Jacket-type fixed structure with configurable leg geometry
- **Morison equation** wave loading (drag + inertia)
- API RP 2A wind load model
- Natural frequency estimation (cantilever model)
- **Palmgren–Miner fatigue damage** accumulation

### Ocean Monitoring Buoy
- Heave dynamics with mooring restoring force
- Solar + battery power budget
- Sensor payload management (CTD, ADCP, wave gauge, met, GPS)
- Data availability gating on battery state

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/alinapradhan/Oceanography-.git
cd Oceanography-
pip install -e ".[dev]"

# 2. Run the full test suite
pytest tests/ -v

# 3. Run a design pipeline in Python
python - <<'EOF'
from agents.orchestrator import Orchestrator

orch = Orchestrator()

# Design → Optimise → Simulate a Wave Energy Converter
result = orch.run_design_pipeline(
    system_type="wec",
    wave_height=2.5,
    wave_period=9.0,
    water_depth=50.0,
)
print("Design:", result["design"])
print("Optimised:", result["optimization"]["optimized_design"])
print("Avg power:", result["simulation"]["average_power_w"], "W")

# Operations health check
ops = orch.run_operations_check(
    system_type="monitoring_buoy",
    telemetry={"battery_level": 0.08, "heave_m": 0.3},
    environment={},
)
print("Status:", ops["operational_status"])
print("Alerts:", ops["alerts"])
EOF
```

---

## Project Structure

```
OceanMech-Agent/
├── agents/                    # Agentic AI architecture
│   ├── base_agent.py          #   Abstract base (perception–reasoning–action)
│   ├── design_agent.py        #   Parametric design heuristics
│   ├── optimization_agent.py  #   Gradient-free hill-climbing optimiser
│   ├── simulation_agent.py    #   Physics-based time-domain simulation
│   ├── operations_agent.py    #   Anomaly detection & control recommendations
│   └── orchestrator.py        #   Pipeline coordinator & message router
│
├── models/
│   ├── mechanical/            # Marine engineering system models
│   │   ├── auv.py             #   Autonomous Underwater Vehicle
│   │   ├── wave_energy_converter.py  # Point-absorber WEC
│   │   ├── offshore_platform.py      # Jacket-type fixed platform
│   │   └── monitoring_buoy.py        # Surface mooring buoy
│   └── ocean/                 # Ocean environment & wave theory
│       ├── ocean_environment.py  # OceanState + OceanEnvironment
│       └── wave_model.py         # JONSWAP spectrum + linear kinematics
│
├── data/
│   ├── oceanographic/         # Oceanographic data access
│   │   └── data_fetcher.py    #   OceanographicDataFetcher + SyntheticBackend
│   └── sensors/               # Sensor data processing
│       └── sensor_data.py     #   SensorReading + SensorDataProcessor
│
├── simulation/
│   ├── engine.py              # Generic time-domain SimulationEngine
│   └── scenarios.py           # Pre-built scenarios (AUV, WEC, platform, buoy)
│
├── config/
│   └── settings.py            # Centralised configuration dataclasses
│
├── tests/
│   ├── test_models.py         # Unit tests for all mechanical & ocean models
│   ├── test_agents.py         # Unit + integration tests for all agents
│   └── test_simulation.py     # Tests for engine, scenarios, data layer
│
├── pyproject.toml             # Build config and test settings
└── requirements.txt           # Core runtime dependencies
```

---

## Modules

### Agents

#### `BaseAgent`
Abstract base class. Subclass and implement `perceive`, `reason`, `act`.

```python
class MyAgent(BaseAgent):
    def perceive(self, context): ...
    def reason(self, observations): ...
    def act(self, decision): ...
```

#### `DesignAgent`
Generates initial parametric designs using engineering heuristics for all four system types.

#### `OptimizationAgent`
Coordinate-wise hill-climbing optimiser. Accepts any callable objective function and parameter bounds.

```python
from agents.optimization_agent import OptimizationAgent

agent = OptimizationAgent(max_iterations=100)
result = agent.run({
    "design": {"x": 0.0},
    "objective_fn": lambda d: -(d["x"] - 3)**2,
    "param_bounds": {"x": (0.0, 6.0)},
    "maximize": True,
})
```

#### `SimulationAgent`
Runs time-domain simulations (Euler integration) for AUVs, WECs, platforms, and buoys.

#### `OperationsAgent`
Rule-based anomaly detection. Issues structured alerts and recommended actions based on telemetry thresholds.

#### `Orchestrator`
Coordinates the full pipeline: Design → Optimize → Simulate. Routes `AgentMessage` objects between agents.

---

### Mechanical Models

```python
from models.mechanical.auv import AUV, AUVGeometry, AUVPropulsion

auv = AUV(
    geometry=AUVGeometry(length=2.5, diameter=0.25, mass=35.0),
    propulsion=AUVPropulsion(max_thrust=80.0),
)
print(auv.drag_force(1.5))           # N at 1.5 m/s
print(auv.range_estimate(500.0, 1.5))  # m at 1.5 m/s with 500 Wh battery
auv.step(thrust=30.0, dt=0.1)        # advance dynamics
```

```python
from models.mechanical.wave_energy_converter import WaveEnergyConverter

wec = WaveEnergyConverter()
print(wec.average_power(wave_height=2.0, wave_period=8.0))  # W
```

---

### Ocean Models

```python
from models.ocean.ocean_environment import OceanEnvironment

env = OceanEnvironment()
print(env.wave_number())         # rad/m (dispersion relation)
print(env.wave_phase_speed())    # m/s
print(env.beaufort_scale())      # Beaufort wind force number
```

```python
from models.ocean.wave_model import JONSWAPSpectrum, LinearWaveKinematics

spec = JONSWAPSpectrum(significant_wave_height=3.0, peak_period=10.0)
components = spec.sample_components(n_components=50)
eta = spec.surface_elevation(x=0.0, t=100.0, components=components)

wave = LinearWaveKinematics(amplitude=1.5, period=8.0, water_depth=50.0)
u = wave.horizontal_velocity(z=-10.0, t=5.0)  # m/s at 10 m depth
```

---

### Data Layer

```python
from data.oceanographic.data_fetcher import OceanographicDataFetcher

fetcher = OceanographicDataFetcher()
record = fetcher.get(lat=51.5, lon=-3.2)
print(record.wave_height, record.sea_surface_temp)
```

Custom backend example:

```python
class MyERDDAPBackend:
    def fetch(self, latitude, longitude, depth, timestamp):
        # call real API here
        ...

fetcher = OceanographicDataFetcher(backend=MyERDDAPBackend())
```

---

### Simulation

```python
from simulation.scenarios import auv_transit_scenario, wec_power_scenario

result = auv_transit_scenario(current_speed=0.8, target_speed=2.0, duration=120.0)
print(result.summary())

result = wec_power_scenario(wave_height=3.0, wave_period=10.0, duration=300.0)
print("Avg power:", sum(result.channel("power_w")) / result.n_points, "W")
```

Custom simulation:

```python
from simulation.engine import SimulationEngine

engine = SimulationEngine(dt=0.5, duration=60.0)
result = engine.run(
    "my_system",
    step_fn=lambda t, state: {"value": t * 2},
    metadata={"description": "custom run"},
)
```

---

### Configuration

```python
from config.settings import DEFAULT_SETTINGS

print(DEFAULT_SETTINGS.simulation.water_density)  # 1025.0 kg/m³
print(DEFAULT_SETTINGS.ocean.significant_wave_height)  # 2.0 m
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_models.py -v
pytest tests/test_agents.py -v
pytest tests/test_simulation.py -v

# With coverage
pytest tests/ --cov=. --cov-report=term-missing
```

The test suite has **124 tests** covering all models, agents, and simulation scenarios.

---

## Examples

### Full AUV Mission Design

```python
from agents.orchestrator import Orchestrator

orch = Orchestrator()
result = orch.run_design_pipeline(
    system_type="auv",
    wave_height=1.5,
    wave_period=7.0,
    water_depth=200.0,
    current_speed=0.3,
    target_speed=2.0,
    payload_mass=8.0,
    deployment_duration_days=1,
    duration=120.0,
    dt=0.5,
)
print(f"AUV length: {result['design']['length_m']:.2f} m")
print(f"AUV diameter: {result['design']['diameter_m']:.3f} m")
print(f"Distance covered: {result['simulation']['distance_covered_m']:.1f} m")
```

### WEC Power Assessment

```python
from simulation.scenarios import wec_power_scenario

result = wec_power_scenario(
    wave_height=3.0, wave_period=10.0, water_depth=40.0, float_radius=6.0, duration=600.0
)
avg_kw = sum(result.channel("power_w")) / result.n_points / 1000.0
print(f"Average power: {avg_kw:.2f} kW")
```

### Operations Monitoring

```python
from agents.operations_agent import OperationsAgent

ops = OperationsAgent()
result = ops.run({
    "system_type": "offshore_platform",
    "telemetry": {"fatigue_damage": 0.72},
    "environment": {"wave_height_m": 8.0},
})
for alert in result["alerts"]:
    print(f"⚠ {alert}")
for action in result["recommended_actions"]:
    print(f"→ {action}")
```

---

## Design Decisions

- **Zero heavy dependencies** – the core platform requires only `numpy` and `scipy` (for future extensions). The base simulation engine is pure Python.
- **Extensible backends** – the `DataBackend` protocol lets you swap in NOAA ERDDAP, Copernicus Marine, or any other data source without touching agent code.
- **Protocol-based design** – agents communicate via `AgentMessage` dicts rather than direct method calls, making them loosely coupled and independently deployable.
- **Euler integration** – all time-stepping uses explicit Euler for simplicity and transparency. For higher-fidelity work, the `SimulationEngine` `step_fn` interface supports drop-in RK4 or other integrators.
- **Engineering validity** – physics formulas follow established references: Morison equation (API RP 2A), JONSWAP spectrum (Hasselmann 1973), linear wave theory (Airy), Palmgren–Miner fatigue rule.
=======
 
>>>>>>> main
