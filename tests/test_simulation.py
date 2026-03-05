"""Tests for simulation engine, scenarios, and ocean models."""

import math
import pytest

from simulation.engine import SimulationEngine, SimulationResult
from simulation.scenarios import (
    auv_transit_scenario,
    wec_power_scenario,
    platform_storm_scenario,
    buoy_deployment_scenario,
)
from models.ocean.ocean_environment import OceanEnvironment, OceanState
from models.ocean.wave_model import JONSWAPSpectrum, LinearWaveKinematics
from data.oceanographic.data_fetcher import OceanographicDataFetcher, SyntheticBackend
from data.sensors.sensor_data import SensorReading, SensorDataProcessor
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# SimulationEngine
# ---------------------------------------------------------------------------

class TestSimulationEngine:
    def test_basic_run(self):
        engine = SimulationEngine(dt=1.0, duration=5.0)
        calls = []

        def step(t, state):
            calls.append(t)
            return {"value": t}

        result = engine.run("test", step)
        assert isinstance(result, SimulationResult)
        assert result.n_points == 6  # 0,1,2,3,4,5
        assert "value" in result.outputs

    def test_invalid_dt(self):
        with pytest.raises(ValueError):
            SimulationEngine(dt=0.0, duration=10.0)

    def test_invalid_duration(self):
        with pytest.raises(ValueError):
            SimulationEngine(dt=0.1, duration=0.0)

    def test_channel_access(self):
        engine = SimulationEngine(dt=1.0, duration=3.0)
        result = engine.run("x", lambda t, s: {"ch": t * 2})
        ch = result.channel("ch")
        assert len(ch) == 4

    def test_channel_missing(self):
        engine = SimulationEngine(dt=1.0, duration=2.0)
        result = engine.run("x", lambda t, s: {"a": t})
        with pytest.raises(KeyError):
            result.channel("b")

    def test_summary_keys(self):
        engine = SimulationEngine(dt=1.0, duration=4.0)
        result = engine.run("x", lambda t, s: {"speed": t})
        summary = result.summary()
        assert "speed_mean" in summary
        assert "speed_max" in summary
        assert "speed_min" in summary

    def test_metadata_stored(self):
        engine = SimulationEngine(dt=1.0, duration=2.0)
        result = engine.run("x", lambda t, s: {"v": t}, metadata={"foo": "bar"})
        assert result.metadata["foo"] == "bar"


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

class TestAUVTransitScenario:
    def test_returns_result(self):
        result = auv_transit_scenario(duration=5.0, dt=0.5)
        assert result.system_type == "auv"
        assert "speed_ms" in result.outputs
        assert "position_m" in result.outputs

    def test_position_increases(self):
        result = auv_transit_scenario(
            target_speed=1.5, current_speed=0.0, duration=10.0, dt=0.5
        )
        positions = result.channel("position_m")
        assert positions[-1] > positions[0]


class TestWECPowerScenario:
    def test_returns_result(self):
        result = wec_power_scenario(duration=5.0, dt=0.5)
        assert result.system_type == "wec"
        assert "power_w" in result.outputs

    def test_power_non_negative(self):
        result = wec_power_scenario(duration=10.0, dt=0.5)
        for p in result.channel("power_w"):
            assert p >= 0.0


class TestPlatformStormScenario:
    def test_returns_result(self):
        result = platform_storm_scenario(storm_wave_heights=[3.0, 5.0, 3.0])
        assert result.system_type == "offshore_platform"
        assert "base_shear_n" in result.outputs

    def test_base_shear_positive(self):
        result = platform_storm_scenario(storm_wave_heights=[5.0])
        for v in result.channel("base_shear_n"):
            assert v > 0.0


class TestBuoyDeploymentScenario:
    def test_returns_result(self):
        result = buoy_deployment_scenario(duration=300.0, dt=60.0)
        assert result.system_type == "monitoring_buoy"
        assert "heave_m" in result.outputs
        assert "battery_level" in result.outputs

    def test_battery_bounded(self):
        result = buoy_deployment_scenario(duration=300.0, dt=60.0)
        for b in result.channel("battery_level"):
            assert 0.0 <= b <= 1.0


# ---------------------------------------------------------------------------
# Ocean environment
# ---------------------------------------------------------------------------

class TestOceanEnvironment:
    def setup_method(self):
        self.env = OceanEnvironment()

    def test_default_state(self):
        s = self.env.state
        assert s.wave_height == 2.0
        assert s.wave_period == 8.0

    def test_wave_angular_frequency(self):
        omega = self.env.wave_angular_frequency()
        assert abs(omega - 2.0 * math.pi / 8.0) < 1e-10

    def test_wave_number_positive(self):
        k = self.env.wave_number()
        assert k > 0.0

    def test_wave_phase_speed_positive(self):
        c = self.env.wave_phase_speed()
        assert c > 0.0

    def test_dynamic_pressure_surface(self):
        p = self.env.dynamic_pressure_amplitude(depth=0.0)
        assert p > 0.0

    def test_dynamic_pressure_decreases_with_depth(self):
        p0 = self.env.dynamic_pressure_amplitude(depth=0.0)
        p50 = self.env.dynamic_pressure_amplitude(depth=50.0)
        assert p0 >= p50

    def test_beaufort_scale_calm(self):
        self.env.update(wind_speed=0.1)
        assert self.env.beaufort_scale() == 0

    def test_beaufort_scale_storm(self):
        self.env.update(wind_speed=33.0)
        assert self.env.beaufort_scale() >= 11

    def test_update_invalid_attr(self):
        with pytest.raises(AttributeError):
            self.env.update(nonexistent=1.0)

    def test_current_velocity_components(self):
        self.env.update(current_speed=1.0, current_direction=0.0)
        u, v = self.env.state.current_velocity
        assert abs(u) < 1e-10   # North → no East component
        assert abs(v - 1.0) < 1e-10

    def test_repr(self):
        assert "OceanEnvironment" in repr(self.env)


# ---------------------------------------------------------------------------
# Wave model
# ---------------------------------------------------------------------------

class TestJONSWAPSpectrum:
    def test_spectral_density_positive(self):
        spec = JONSWAPSpectrum(significant_wave_height=2.0, peak_period=8.0)
        s = spec.spectral_density(freq=0.125)
        assert s > 0.0

    def test_spectral_density_zero_freq(self):
        spec = JONSWAPSpectrum(2.0, 8.0)
        assert spec.spectral_density(0.0) == 0.0

    def test_peak_at_peak_frequency(self):
        spec = JONSWAPSpectrum(2.0, 8.0)
        fp = 1.0 / 8.0
        # S(fp) should be larger than nearby frequencies
        s_peak = spec.spectral_density(fp)
        s_low = spec.spectral_density(fp * 0.5)
        s_high = spec.spectral_density(fp * 2.0)
        assert s_peak > s_low
        assert s_peak > s_high

    def test_sample_components_count(self):
        spec = JONSWAPSpectrum(2.0, 8.0)
        components = spec.sample_components(n_components=10)
        assert len(components) == 10

    def test_surface_elevation_returns_float(self):
        spec = JONSWAPSpectrum(2.0, 8.0)
        comps = spec.sample_components(5)
        eta = spec.surface_elevation(x=0.0, t=0.0, components=comps)
        assert isinstance(eta, float)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            JONSWAPSpectrum(significant_wave_height=0.0, peak_period=8.0)
        with pytest.raises(ValueError):
            JONSWAPSpectrum(significant_wave_height=2.0, peak_period=0.0)


class TestLinearWaveKinematics:
    def setup_method(self):
        self.wave = LinearWaveKinematics(amplitude=1.0, period=8.0, water_depth=50.0)

    def test_horizontal_velocity_surface(self):
        u = self.wave.horizontal_velocity(z=0.0, t=0.0)
        assert isinstance(u, float)

    def test_vertical_velocity_surface(self):
        w = self.wave.vertical_velocity(z=0.0, t=0.0)
        assert isinstance(w, float)

    def test_wave_number_positive(self):
        assert self.wave.k > 0.0

    def test_invalid_amplitude(self):
        with pytest.raises(ValueError):
            LinearWaveKinematics(amplitude=0.0, period=8.0, water_depth=50.0)

    def test_invalid_period(self):
        with pytest.raises(ValueError):
            LinearWaveKinematics(amplitude=1.0, period=0.0, water_depth=50.0)

    def test_invalid_depth(self):
        with pytest.raises(ValueError):
            LinearWaveKinematics(amplitude=1.0, period=8.0, water_depth=0.0)


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

class TestOceanographicDataFetcher:
    def setup_method(self):
        self.fetcher = OceanographicDataFetcher()

    def test_get_returns_record(self):
        record = self.fetcher.get(lat=51.5, lon=-3.2)
        assert record.latitude == 51.5
        assert record.longitude == -3.2

    def test_wave_height_positive(self):
        record = self.fetcher.get(lat=0.0, lon=0.0)
        assert record.wave_height > 0.0

    def test_salinity_range(self):
        record = self.fetcher.get(lat=0.0, lon=0.0)
        assert 30.0 < record.salinity < 40.0

    def test_time_series_length(self):
        ts = [datetime(2024, 6, 1, h, tzinfo=timezone.utc) for h in range(5)]
        records = self.fetcher.get_time_series(lat=30.0, lon=-50.0, timestamps=ts)
        assert len(records) == 5

    def test_empty_time_series(self):
        records = self.fetcher.get_time_series(lat=0.0, lon=0.0, timestamps=[])
        assert records == []

    def test_synthetic_backend_reproducible(self):
        ts = datetime(2024, 1, 15, tzinfo=timezone.utc)
        b1 = SyntheticBackend(seed=99)
        b2 = SyntheticBackend(seed=99)
        r1 = b1.fetch(20.0, 10.0, 0.0, ts)
        r2 = b2.fetch(20.0, 10.0, 0.0, ts)
        assert r1.wave_height == r2.wave_height
        assert r1.sea_surface_temp == r2.sea_surface_temp


class TestSensorDataProcessor:
    def setup_method(self):
        self.proc = SensorDataProcessor()

    def _reading(self, sensor_type, value):
        return SensorReading(
            sensor_id="S1", sensor_type=sensor_type, value=value, unit="unit"
        )

    def test_valid_temperature(self):
        r = self.proc.validate(self._reading("temperature", 15.0))
        assert r.is_valid()

    def test_invalid_temperature(self):
        r = self.proc.validate(self._reading("temperature", 100.0))
        assert not r.is_valid()

    def test_unknown_type_passes(self):
        r = self.proc.validate(self._reading("unknown_sensor", 999.0))
        assert r.quality_flag == 1   # unknown types are not flagged

    def test_unit_conversion_c_to_f(self):
        f = self.proc.celsius_to_fahrenheit(0.0)
        assert f == 32.0
        f = self.proc.celsius_to_fahrenheit(100.0)
        assert abs(f - 212.0) < 1e-6

    def test_unit_conversion_knots_to_ms(self):
        ms = self.proc.knots_to_ms(1.0)
        assert abs(ms - 0.514444) < 1e-5

    def test_summarise_returns_stats(self):
        readings = [
            self._reading("temperature", 10.0),
            self._reading("temperature", 20.0),
        ]
        summary = self.proc.summarise(readings)
        assert "temperature" in summary
        assert summary["temperature"]["mean"] == 15.0
        assert summary["temperature"]["max"] == 20.0

    def test_summarise_excludes_bad_readings(self):
        r_good = self._reading("salinity", 35.0)
        r_bad = self._reading("salinity", 35.0)
        r_bad.quality_flag = 0
        summary = self.proc.summarise([r_good, r_bad])
        assert summary["salinity"]["count"] == 1
