"""Tests for mechanical system models."""

import math
import pytest

from models.mechanical.auv import AUV, AUVGeometry, AUVPropulsion, Pose
from models.mechanical.wave_energy_converter import WaveEnergyConverter, WECGeometry, PTOParameters
from models.mechanical.offshore_platform import OffshorePlatform, PlatformGeometry
from models.mechanical.monitoring_buoy import MonitoringBuoy, BuoyGeometry, PowerSystem


class TestAUV:
    def setup_method(self):
        self.auv = AUV()

    def test_default_geometry(self):
        assert self.auv.geometry.length == 2.0
        assert self.auv.geometry.diameter == 0.2
        assert self.auv.geometry.mass == 30.0

    def test_cross_sectional_area(self):
        expected = math.pi * 0.1 ** 2
        assert abs(self.auv.cross_sectional_area - expected) < 1e-10

    def test_drag_force_zero_speed(self):
        assert self.auv.drag_force(0.0) == 0.0

    def test_drag_force_positive(self):
        assert self.auv.drag_force(1.0) > 0.0

    def test_drag_force_increases_with_speed(self):
        assert self.auv.drag_force(2.0) > self.auv.drag_force(1.0)

    def test_buoyancy_force(self):
        # buoyancy_volume set so buoy > weight
        bf = self.auv.buoyancy_force()
        expected = (1025.0 * 9.81 * self.auv.geometry.buoyancy_volume
                    - self.auv.geometry.mass * 9.81)
        assert abs(bf - expected) < 1e-6

    def test_power_consumption_zero_thrust(self):
        assert self.auv.power_consumption(0.0) == 0.0

    def test_power_consumption_negative_thrust(self):
        with pytest.raises(ValueError):
            self.auv.power_consumption(-1.0)

    def test_range_estimate_positive(self):
        r = self.auv.range_estimate(500.0, 1.5)
        assert r > 0.0

    def test_step_advances_position(self):
        self.auv.step(thrust=20.0, dt=1.0)
        assert self.auv.state.pose.x > 0.0
        assert self.auv.state.mission_elapsed == 1.0

    def test_step_invalid_dt(self):
        with pytest.raises(ValueError):
            self.auv.step(thrust=20.0, dt=0.0)

    def test_step_with_current(self):
        auv = AUV()
        auv.step(thrust=20.0, dt=1.0, current_speed=0.5)
        assert auv.state.mission_elapsed == 1.0

    def test_repr(self):
        r = repr(self.auv)
        assert "AUV" in r


class TestWaveEnergyConverter:
    def setup_method(self):
        self.wec = WaveEnergyConverter()

    def test_waterplane_area(self):
        expected = math.pi * 5.0 ** 2
        assert abs(self.wec.waterplane_area - expected) < 1e-6

    def test_hydrostatic_stiffness_positive(self):
        assert self.wec.hydrostatic_stiffness > 0.0

    def test_added_mass_positive(self):
        assert self.wec.added_mass > 0.0

    def test_excitation_force_positive(self):
        f = self.wec.excitation_force(wave_amplitude=1.0, wave_period=8.0)
        assert f > 0.0

    def test_average_power_positive(self):
        p = self.wec.average_power(wave_height=2.0, wave_period=8.0)
        assert p > 0.0

    def test_capture_width_ratio_bounded(self):
        cwr = self.wec.capture_width_ratio(wave_period=8.0)
        assert 0.0 <= cwr <= 1.0

    def test_step_returns_power(self):
        power = self.wec.step(excitation=1e4, dt=0.1)
        assert isinstance(power, float)
        assert power >= 0.0

    def test_step_invalid_dt(self):
        with pytest.raises(ValueError):
            self.wec.step(excitation=1e4, dt=-0.1)

    def test_energy_accumulates(self):
        for _ in range(10):
            self.wec.step(excitation=1e4, dt=0.1)
        assert self.wec.state.energy_captured >= 0.0

    def test_repr(self):
        assert "WaveEnergyConverter" in repr(self.wec)


class TestOffshorePlatform:
    def setup_method(self):
        self.platform = OffshorePlatform()

    def test_total_mass(self):
        g = self.platform.geometry
        assert self.platform.total_mass == g.deck_mass + g.jacket_mass

    def test_natural_frequency_positive(self):
        fn = self.platform.natural_frequency()
        assert fn > 0.0

    def test_morison_wave_force_positive(self):
        f = self.platform.morison_wave_force(wave_height=5.0, wave_period=10.0)
        assert f > 0.0

    def test_wave_force_increases_with_height(self):
        f1 = self.platform.morison_wave_force(2.0, 10.0)
        f2 = self.platform.morison_wave_force(5.0, 10.0)
        assert f2 > f1

    def test_wind_force_positive(self):
        assert self.platform.wind_force(wind_speed=15.0) > 0.0

    def test_wind_force_zero_speed(self):
        assert self.platform.wind_force(wind_speed=0.0) == 0.0

    def test_update_loads(self):
        self.platform.update_loads(5.0, 10.0, 15.0)
        assert self.platform.state.base_shear > 0.0
        assert self.platform.state.overturning_moment > 0.0

    def test_fatigue_accumulates(self):
        self.platform.accumulate_fatigue(stress_range=1e6, cycles=1000)
        assert self.platform.state.fatigue_damage > 0.0

    def test_fatigue_zero_stress_skipped(self):
        damage_before = self.platform.state.fatigue_damage
        self.platform.accumulate_fatigue(stress_range=0.0, cycles=1000)
        assert self.platform.state.fatigue_damage == damage_before

    def test_repr(self):
        assert "OffshorePlatform" in repr(self.platform)


class TestMonitoringBuoy:
    def setup_method(self):
        self.buoy = MonitoringBuoy()

    def test_hydrostatic_stiffness_positive(self):
        assert self.buoy.hydrostatic_stiffness > 0.0

    def test_natural_period_positive(self):
        assert self.buoy.natural_period() > 0.0

    def test_mooring_tension_positive(self):
        assert self.buoy.mooring_tension() >= 0.0

    def test_step_advances_heave(self):
        self.buoy.step(wave_force=500.0, dt=0.1)
        # After one step from rest with a positive force the velocity becomes non-zero
        assert self.buoy.state.heave_velocity != 0.0

    def test_step_invalid_dt(self):
        with pytest.raises(ValueError):
            self.buoy.step(wave_force=500.0, dt=0.0)

    def test_solar_generation_max(self):
        p = self.buoy.solar_generation(solar_fraction=1.0)
        expected = (
            self.buoy.power.solar_panel_area
            * self.buoy.SOLAR_IRRADIANCE
            * self.buoy.power.solar_efficiency
        )
        assert abs(p - expected) < 1e-6

    def test_solar_generation_clamped(self):
        p_over = self.buoy.solar_generation(solar_fraction=2.0)
        p_max = self.buoy.solar_generation(solar_fraction=1.0)
        assert p_over == p_max

    def test_battery_charges(self):
        # With high solar fraction the battery should not decrease
        initial = self.buoy.power.battery_level = 0.5
        self.buoy.update_battery(dt=3600.0, solar_fraction=1.0)
        assert self.buoy.power.battery_level >= initial

    def test_battery_update_invalid_dt(self):
        with pytest.raises(ValueError):
            self.buoy.update_battery(dt=-1.0)

    def test_data_availability(self):
        self.buoy.power.battery_level = 0.5
        assert self.buoy.data_availability() is True

    def test_data_availability_low_battery(self):
        self.buoy.power.battery_level = 0.05
        assert self.buoy.data_availability() is False

    def test_repr(self):
        assert "MonitoringBuoy" in repr(self.buoy)
