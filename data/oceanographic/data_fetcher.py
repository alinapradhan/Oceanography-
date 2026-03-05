"""Oceanographic data access layer.

Provides a clean interface for retrieving ocean condition data.  The
``OceanographicDataFetcher`` class supports both a built-in *synthetic*
backend (useful for offline testing and agent training) and an extensible
plugin architecture for real data sources (NOAA ERDDAP, Copernicus Marine,
etc.).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@dataclass
class OceanDataRecord:
    """A single observation record."""

    timestamp: datetime
    latitude: float       # degrees
    longitude: float      # degrees
    depth: float          # m (positive downward)
    wave_height: float    # m
    wave_period: float    # s
    sea_surface_temp: float  # °C
    salinity: float       # PSU
    current_speed: float  # m/s
    current_direction: float  # degrees from North


@runtime_checkable
class DataBackend(Protocol):
    """Protocol that any data backend must implement."""

    def fetch(
        self,
        latitude: float,
        longitude: float,
        depth: float,
        timestamp: datetime,
    ) -> OceanDataRecord:
        """Return an :class:`OceanDataRecord` for the given coordinates."""
        ...


class SyntheticBackend:
    """Generates physically plausible synthetic ocean data.

    Uses a simple climatological model based on latitude to produce
    deterministic (seeded) results, making it suitable for reproducible
    agent training and unit tests.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def fetch(
        self,
        latitude: float,
        longitude: float,
        depth: float,
        timestamp: datetime,
    ) -> OceanDataRecord:
        """Generate a synthetic ocean data record.

        Args:
            latitude: Latitude in decimal degrees (-90 to 90).
            longitude: Longitude in decimal degrees (-180 to 180).
            depth: Depth in metres (positive downward).
            timestamp: Observation time (timezone-aware recommended).

        Returns:
            Synthetic :class:`OceanDataRecord`.
        """
        # SST decreases from ~28 °C at equator to ~2 °C at poles
        sst = 28.0 - 0.26 * abs(latitude)
        # Add seasonal variation (±2 °C)
        day_of_year = timestamp.timetuple().tm_yday
        seasonal = 2.0 * math.sin(2.0 * math.pi * (day_of_year - 80) / 365.0)
        sst = max(0.0, sst + seasonal + self._rng.gauss(0, 0.5))

        # Depth-temperature gradient (thermocline)
        temp_at_depth = sst * math.exp(-depth / 200.0)

        # Wave height increases with latitude (rougher seas)
        hs = 1.0 + 0.03 * abs(latitude) + self._rng.uniform(0, 0.5)
        tp = 6.0 + 0.05 * abs(latitude) + self._rng.uniform(0, 2.0)

        salinity = 35.0 + 0.01 * abs(latitude) + self._rng.gauss(0, 0.3)
        current_speed = self._rng.uniform(0.1, 1.2)
        current_direction = self._rng.uniform(0, 360)

        return OceanDataRecord(
            timestamp=timestamp,
            latitude=latitude,
            longitude=longitude,
            depth=depth,
            wave_height=round(hs, 2),
            wave_period=round(tp, 2),
            sea_surface_temp=round(temp_at_depth, 2),
            salinity=round(salinity, 2),
            current_speed=round(current_speed, 3),
            current_direction=round(current_direction, 1),
        )


class OceanographicDataFetcher:
    """High-level interface for fetching oceanographic data.

    Usage::

        fetcher = OceanographicDataFetcher()
        record = fetcher.get(lat=51.5, lon=-3.2, depth=0.0)

    To plug in a custom backend::

        fetcher = OceanographicDataFetcher(backend=MyERDDAPBackend())
    """

    def __init__(self, backend: DataBackend | None = None) -> None:
        self._backend: DataBackend = backend or SyntheticBackend()

    def get(
        self,
        lat: float,
        lon: float,
        depth: float = 0.0,
        timestamp: datetime | None = None,
    ) -> OceanDataRecord:
        """Fetch an ocean data record.

        Args:
            lat: Latitude (decimal degrees).
            lon: Longitude (decimal degrees).
            depth: Depth below surface (m, positive downward).
            timestamp: Observation time; defaults to current UTC time.

        Returns:
            :class:`OceanDataRecord` from the active backend.
        """
        if timestamp is None:
            timestamp = datetime.now(tz=timezone.utc)
        return self._backend.fetch(lat, lon, depth, timestamp)

    def get_time_series(
        self,
        lat: float,
        lon: float,
        depth: float = 0.0,
        timestamps: list[datetime] | None = None,
    ) -> list[OceanDataRecord]:
        """Fetch a sequence of records.

        Args:
            lat: Latitude (decimal degrees).
            lon: Longitude (decimal degrees).
            depth: Depth below surface (m).
            timestamps: List of datetimes to query.

        Returns:
            List of :class:`OceanDataRecord` objects.
        """
        if not timestamps:
            return []
        return [self.get(lat, lon, depth, ts) for ts in timestamps]
