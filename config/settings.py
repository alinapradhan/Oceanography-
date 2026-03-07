"""Global configuration and default settings for OceanMech-Agent."""
   
from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    """Parameters that control simulation behaviour."""

    time_step: float = 0.1          # seconds
    max_duration: float = 3600.0    # seconds (1 hour)
    gravity: float = 9.81           # m/s²
    water_density: float = 1025.0   # kg/m³ (sea water)
    air_density: float = 1.225      # kg/m³


@dataclass
class AgentConfig:
    """Parameters controlling agent decision cycles."""

    max_iterations: int = 100
    convergence_threshold: float = 1e-4
    verbose: bool = False


@dataclass
class OceanConfig:
    """Default ocean environment parameters."""

    significant_wave_height: float = 2.0   # metres
    peak_wave_period: float = 8.0          # seconds
    current_speed: float = 0.5             # m/s
    current_direction: float = 0.0         # degrees (0 = North)
    water_depth: float = 100.0             # metres
    temperature: float = 15.0             # °C
    salinity: float = 35.0                # PSU


@dataclass
class Settings:
    """Top-level application settings."""

    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    ocean: OceanConfig = field(default_factory=OceanConfig)


# Module-level default instance – import and use directly.
DEFAULT_SETTINGS = Settings()
