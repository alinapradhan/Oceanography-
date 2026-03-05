"""Sensor data models and processing utilities.

Provides data classes for sensor readings from marine platforms and a
lightweight processing pipeline for validation and unit conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class SensorReading:
    """A single raw sensor reading."""

    sensor_id: str
    sensor_type: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    quality_flag: int = 1   # 0 = bad, 1 = good, 2 = uncertain

    def is_valid(self) -> bool:
        return self.quality_flag == 1


class SensorDataProcessor:
    """Simple processing pipeline for sensor readings.

    Supports:
    - Range-based quality control
    - Unit conversions
    - Summary statistics over a batch of readings
    """

    # Acceptable physical ranges per sensor type
    _RANGES: dict[str, tuple[float, float]] = {
        "temperature": (-2.0, 35.0),    # °C
        "salinity": (0.0, 45.0),        # PSU
        "pressure": (0.0, 11000.0),     # dbar
        "wave_height": (0.0, 30.0),     # m
        "current_speed": (0.0, 5.0),    # m/s
        "depth": (0.0, 11000.0),        # m
        "battery_voltage": (10.0, 16.0),  # V
        "heading": (0.0, 360.0),        # degrees
    }

    def validate(self, reading: SensorReading) -> SensorReading:
        """Apply range-based quality control to *reading*.

        Sets ``quality_flag = 0`` if the value is outside the known range
        for the sensor type; ``quality_flag = 1`` otherwise.

        Args:
            reading: Raw sensor reading to validate.

        Returns:
            The same reading object with updated ``quality_flag``.
        """
        sensor_type = reading.sensor_type.lower()
        if sensor_type in self._RANGES:
            lo, hi = self._RANGES[sensor_type]
            reading.quality_flag = 1 if lo <= reading.value <= hi else 0
        return reading

    def celsius_to_fahrenheit(self, celsius: float) -> float:
        return celsius * 9.0 / 5.0 + 32.0

    def knots_to_ms(self, knots: float) -> float:
        return knots * 0.514444

    def summarise(self, readings: list[SensorReading]) -> dict[str, Any]:
        """Compute summary statistics over a list of valid readings.

        Args:
            readings: Sensor readings (mixed types are handled separately).

        Returns:
            Dictionary mapping sensor type to ``{mean, min, max, count}``.
        """
        from collections import defaultdict

        buckets: dict[str, list[float]] = defaultdict(list)
        for r in readings:
            if r.is_valid():
                buckets[r.sensor_type].append(r.value)

        summary: dict[str, Any] = {}
        for stype, values in buckets.items():
            if values:
                summary[stype] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
        return summary
