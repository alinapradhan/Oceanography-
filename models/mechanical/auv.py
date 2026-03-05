"""Autonomous Underwater Vehicle (AUV) mechanical model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple


class Pose(NamedTuple):
    """6-DOF pose of the AUV."""

    x: float = 0.0       # m (East)
    y: float = 0.0       # m (North)
    z: float = 0.0       # m (depth, positive downward)
    roll: float = 0.0    # rad
    pitch: float = 0.0   # rad
    yaw: float = 0.0     # rad


@dataclass
class AUVGeometry:
    """Physical dimensions and mass properties of an AUV."""

    length: float = 2.0           # m
    diameter: float = 0.2         # m
    mass: float = 30.0            # kg
    buoyancy_volume: float = 0.03 # m³  (slightly positively buoyant)
    drag_coefficient_axial: float = 0.03
    drag_coefficient_lateral: float = 0.8


@dataclass
class AUVPropulsion:
    """Thruster characteristics."""

    max_thrust: float = 50.0      # N  (single thruster)
    efficiency: float = 0.65
    max_speed: float = 2.5        # m/s


@dataclass
class AUVState:
    """Dynamic state of the AUV."""

    pose: Pose = field(default_factory=Pose)
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    battery_level: float = 1.0    # 0–1 (fraction remaining)
    mission_elapsed: float = 0.0  # s


class AUV:
    """Autonomous Underwater Vehicle model.

    Provides geometry, hydrodynamics, and simplified 3-DOF (surge, sway, heave)
    equations of motion for simulation and agent-based design.
    """

    WATER_DENSITY = 1025.0  # kg/m³
    GRAVITY = 9.81          # m/s²

    def __init__(
        self,
        geometry: AUVGeometry | None = None,
        propulsion: AUVPropulsion | None = None,
    ) -> None:
        self.geometry = geometry or AUVGeometry()
        self.propulsion = propulsion or AUVPropulsion()
        self.state = AUVState()

    # ------------------------------------------------------------------
    # Geometric properties
    # ------------------------------------------------------------------

    @property
    def cross_sectional_area(self) -> float:
        """Frontal cross-sectional area (m²)."""
        return math.pi * (self.geometry.diameter / 2.0) ** 2

    @property
    def lateral_area(self) -> float:
        """Lateral projected area (m²)."""
        return self.geometry.length * self.geometry.diameter

    # ------------------------------------------------------------------
    # Hydrodynamic forces
    # ------------------------------------------------------------------

    def buoyancy_force(self) -> float:
        """Net buoyancy force (N, positive = upward)."""
        weight = self.geometry.mass * self.GRAVITY
        buoyancy = self.WATER_DENSITY * self.GRAVITY * self.geometry.buoyancy_volume
        return buoyancy - weight

    def drag_force(self, speed: float, lateral: bool = False) -> float:
        """Hydrodynamic drag force (N) at given speed.

        Args:
            speed: Vehicle speed through water (m/s).
            lateral: If True use lateral drag coefficient, else axial.

        Returns:
            Drag force magnitude in newtons.
        """
        cd = (
            self.geometry.drag_coefficient_lateral
            if lateral
            else self.geometry.drag_coefficient_axial
        )
        area = self.lateral_area if lateral else self.cross_sectional_area
        return 0.5 * self.WATER_DENSITY * cd * area * speed ** 2

    def required_thrust(self, target_speed: float) -> float:
        """Thrust required to maintain *target_speed* at steady state (N)."""
        return self.drag_force(target_speed)

    def power_consumption(self, thrust: float) -> float:
        """Estimate propulsion power consumption (W).

        Args:
            thrust: Propulsive thrust (N).

        Returns:
            Power in watts.
        """
        if thrust < 0:
            raise ValueError("thrust must be non-negative")
        speed = self.propulsion.max_speed * (thrust / self.propulsion.max_thrust) ** (1.0 / 3.0)
        return thrust * speed / max(self.propulsion.efficiency, 1e-6)

    def range_estimate(self, battery_capacity_wh: float, speed: float) -> float:
        """Estimate maximum range (m) at constant speed.

        Args:
            battery_capacity_wh: Battery energy capacity (Wh).
            speed: Cruise speed (m/s).

        Returns:
            Range in metres.
        """
        thrust = self.required_thrust(speed)
        power_w = self.power_consumption(thrust)
        if power_w <= 0:
            return float("inf")
        endurance_s = (battery_capacity_wh * 3600.0) / power_w
        return speed * endurance_s

    # ------------------------------------------------------------------
    # Simplified equations of motion (surge only)
    # ------------------------------------------------------------------

    def step(self, thrust: float, dt: float, current_speed: float = 0.0) -> None:
        """Advance the AUV dynamics by one time step.

        Args:
            thrust: Applied thrust (N, positive forward).
            dt: Time step (s).
            current_speed: Water current speed in surge direction (m/s).
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        vx = self.state.velocity[0]
        relative_speed = vx - current_speed
        drag = self.drag_force(abs(relative_speed)) * math.copysign(1.0, relative_speed)
        net_force = thrust - drag
        ax = net_force / self.geometry.mass
        self.state.velocity[0] = vx + ax * dt
        new_pose = Pose(
            x=self.state.pose.x + self.state.velocity[0] * dt,
            y=self.state.pose.y,
            z=self.state.pose.z,
            roll=self.state.pose.roll,
            pitch=self.state.pose.pitch,
            yaw=self.state.pose.yaw,
        )
        self.state.pose = new_pose
        self.state.mission_elapsed += dt

    def __repr__(self) -> str:
        g = self.geometry
        return (
            f"AUV(length={g.length}m, diameter={g.diameter}m, "
            f"mass={g.mass}kg, max_speed={self.propulsion.max_speed}m/s)"
        )
