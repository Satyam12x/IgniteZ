"""Well-to-wake emissions tests.

Maps to F-01 (grey vs green pathways), F-02 (methane slip), F-03 (ammonia N2O),
F-04 (hydrogen range), F-05 (tank volume displaces cargo), F-06 (fuel availability),
F-07 (shore power on a coal grid), F-08 (port and vessel both fitted), F-09 (factors
in config), F-10 (stated GWP horizon), P-09 (pilot fuel), and S-02/S-04/S-06.

The F-01 test is the one to show a judge: a stack-only model scores grey ammonia at
zero, and this suite asserts it comes out *worse than diesel*.
"""

from __future__ import annotations

import math

import pytest

from greenfleet.config.loader import load_fuel_registry
from greenfleet.emissions.kpi import HaritSagarTracker, carbon_cost_inr
from greenfleet.emissions.ports import FuelUnavailableError, load_port_registry
from greenfleet.emissions.tank import TankCalculator
from greenfleet.emissions.wtw import WellToWakeCalculator

pytestmark = pytest.mark.edgecase

ONE_TJ = 1_000_000.0


@pytest.fixture(scope="module")
def calc() -> WellToWakeCalculator:
    return WellToWakeCalculator(gwp_set="ar6_gwp100")


@pytest.fixture(scope="module")
def ports():
    return load_port_registry()


@pytest.fixture(scope="module")
def tank() -> TankCalculator:
    return TankCalculator()


class TestF01GreyVersusGreen:
    def test_grey_ammonia_is_worse_than_diesel(self, calc):
        """The headline lifecycle result. A funnel-only model scores this zero."""
        change = calc.relative_to(ONE_TJ, "ammonia", "grey")
        assert change > 0, "grey ammonia must come out worse than diesel on WTW"
        assert change > 0.2, f"expected a large penalty, got {100 * change:.1f}%"

    def test_green_ammonia_is_much_better_than_diesel(self, calc):
        assert calc.relative_to(ONE_TJ, "ammonia", "green") < -0.5

    def test_pathway_choice_dominates_the_answer(self, calc):
        """Same molecule, same engine: the pathway changes the result by >5x."""
        grey = calc.compute(ONE_TJ, "ammonia", "grey").total_t
        green = calc.compute(ONE_TJ, "ammonia", "green").total_t
        assert grey / green > 5.0

    def test_grey_hydrogen_is_worse_than_diesel(self, calc):
        assert calc.relative_to(ONE_TJ, "hydrogen", "grey") > 0

    def test_green_hydrogen_is_far_better(self, calc):
        assert calc.relative_to(ONE_TJ, "hydrogen", "green") < -0.85

    def test_ambiguous_pathway_must_be_named(self, calc):
        """Defaulting ammonia's pathway would silently pick a 5x-different answer."""
        with pytest.raises(ValueError, match="must be named explicitly"):
            calc.compute(ONE_TJ, "ammonia")

    def test_unknown_pathway_lists_the_valid_ones(self, calc):
        with pytest.raises(KeyError, match="available"):
            calc.compute(ONE_TJ, "ammonia", "turquoise")


class TestF02MethaneSlip:
    def test_slip_is_counted(self, calc):
        assert calc.compute(ONE_TJ, "lng", "fossil").tank_to_wake_ch4_t > 0

    def test_slip_erases_most_of_the_lng_advantage(self, calc):
        """LNG starts ~24% better on combustion CO2 and ends only a few % better."""
        lng = calc.compute(ONE_TJ, "lng", "fossil")
        mgo = calc.compute(ONE_TJ, "mgo", "fossil")
        co2_advantage = 1.0 - lng.tank_to_wake_co2_t / mgo.tank_to_wake_co2_t
        wtw_advantage = 1.0 - lng.total_t / mgo.total_t
        assert co2_advantage > 0.2, "LNG should start well ahead on combustion CO2"
        assert wtw_advantage < co2_advantage / 2, (
            "methane slip must erase most of the CO2 advantage"
        )

    def test_engine_technology_changes_the_answer(self, calc):
        otto = calc.compute(ONE_TJ, "lng", "fossil", engine="otto_medium_speed")
        diesel = calc.compute(ONE_TJ, "lng", "fossil", engine="diesel_slow_speed")
        assert otto.tank_to_wake_ch4_t > diesel.tank_to_wake_ch4_t * 10

    def test_default_engine_is_the_measured_fleet_not_the_worst_case(self):
        """Assuming Otto medium-speed for an unknown vessel overstates slip ~1.8x."""
        registry = load_fuel_registry()
        default = registry.methane_slip_fraction("lng")
        worst = registry.methane_slip_fraction("lng", "otto_medium_speed")
        assert default < worst
        assert default == pytest.approx(0.01716, rel=1e-6)

    def test_measured_slip_distribution_is_available(self):
        """A single point value hides a 4.7x spread across the real fleet."""
        low, median, high = load_fuel_registry().methane_slip_range("lng")
        assert low < median < high
        assert high / low > 4.0

    def test_conventional_fuels_have_negligible_slip(self, calc):
        mgo = calc.compute(ONE_TJ, "mgo", "fossil")
        assert mgo.tank_to_wake_ch4_t < mgo.tank_to_wake_co2_t * 0.01

    def test_unknown_engine_is_rejected(self):
        with pytest.raises(KeyError, match="unknown engine"):
            load_fuel_registry().methane_slip_fraction("lng", "steam_powered_unicorn")


class TestF03AmmoniaN2O:
    def test_n2o_is_counted_for_ammonia(self, calc):
        assert calc.compute(ONE_TJ, "ammonia", "green").tank_to_wake_n2o_t > 0

    def test_green_ammonia_is_not_zero(self, calc):
        """Carbon-free fuel, non-zero warming: N2O plus pilot diesel."""
        result = calc.compute(ONE_TJ, "ammonia", "green")
        assert result.tank_to_wake_co2_t == 0.0
        assert result.total_t > 0
        assert result.tank_to_wake_n2o_t + result.pilot_fuel_t > 0

    def test_n2o_uses_its_gwp(self, calc):
        """N2O is ~273x CO2, so a small mass matters.

        Computed from *fuel* energy, which is shaft energy divided by the engine's
        conversion efficiency - the same basis the calculator uses.
        """
        result = calc.compute(ONE_TJ, "ammonia", "green")
        registry = load_fuel_registry()
        props = registry.get("ammonia")
        pilot = props.pilot_fuel["energy_fraction"]
        fuel_energy = registry.input_energy_mj(ONE_TJ, "ammonia")
        expected = fuel_energy * (1 - pilot) * props.n2o_g_per_mj / 1e6 * calc.gwp.n2o
        assert result.tank_to_wake_n2o_t == pytest.approx(expected, rel=1e-9)

    def test_shaft_basis_scales_n2o_by_engine_efficiency(self, calc):
        """Guards the conversion: fuel energy is 1/0.45 of shaft energy for ammonia."""
        registry = load_fuel_registry()
        assert registry.input_energy_mj(ONE_TJ, "ammonia") == pytest.approx(
            ONE_TJ / registry.get("ammonia").conversion_efficiency, rel=1e-12
        )


class TestP09PilotFuel:
    def test_dual_fuel_engines_burn_pilot_diesel(self, calc):
        assert calc.compute(ONE_TJ, "ammonia", "green").pilot_fuel_t > 0
        assert calc.compute(ONE_TJ, "methanol", "e_fuel").pilot_fuel_t > 0

    def test_pilot_can_be_excluded_for_comparison(self, calc):
        with_pilot = calc.compute(ONE_TJ, "ammonia", "green")
        without = calc.compute(ONE_TJ, "ammonia", "green", include_pilot_fuel=False)
        assert with_pilot.total_t > without.total_t
        assert without.pilot_fuel_t == 0.0

    def test_single_fuel_engines_have_no_pilot(self, calc):
        assert calc.compute(ONE_TJ, "mgo", "fossil").pilot_fuel_t == 0.0

    def test_pilot_is_a_material_share_for_green_ammonia(self, calc):
        """When the main fuel is clean, the pilot diesel dominates what is left."""
        result = calc.compute(ONE_TJ, "ammonia", "green")
        assert result.pilot_fuel_t / result.total_t > 0.2


class TestF10GwpHorizon:
    def test_horizon_is_stated_on_every_result(self, calc):
        result = calc.compute(ONE_TJ, "lng", "fossil")
        assert result.horizon_years == 100
        assert result.gwp_set == "ar6_gwp100"

    def test_the_eu_regulatory_gwp_set_is_available(self):
        """Recovered exactly from 28,125 published filings."""
        registry = load_fuel_registry()
        eu = registry.gwp_sets["eu_mrv_ar5"]
        assert eu.ch4_fossil == 28.0
        assert eu.n2o == 265.0
        assert eu.confidence == "high"

    def test_gwp_set_changes_the_methane_penalty(self):
        ar6 = WellToWakeCalculator(gwp_set="ar6_gwp100").compute(ONE_TJ, "lng", "fossil")
        eu = WellToWakeCalculator(gwp_set="eu_mrv_ar5").compute(ONE_TJ, "lng", "fossil")
        assert ar6.tank_to_wake_ch4_t != eu.tank_to_wake_ch4_t
        assert ar6.tank_to_wake_ch4_t / eu.tank_to_wake_ch4_t == pytest.approx(
            29.8 / 28.0, rel=1e-9
        )

    def test_unknown_gwp_set_is_rejected(self):
        with pytest.raises(ValueError, match="unknown GWP set"):
            WellToWakeCalculator(gwp_set="ar99")


class TestF09FactorsComeFromConfig:
    def test_changing_a_factor_changes_the_result(self, tmp_path):
        """No emission factor may be hard-coded in Python."""
        import yaml

        from greenfleet.config.loader import FuelRegistry

        source = load_fuel_registry().source_path
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        before = WellToWakeCalculator(
            FuelRegistry(payload), gwp_set="ar6_gwp100"
        ).compute(ONE_TJ, "ammonia", "grey").total_t

        payload["fuels"]["ammonia"]["pathways"]["grey"]["wtt_gco2e_per_mj"] = 10.0
        after = WellToWakeCalculator(
            FuelRegistry(payload), gwp_set="ar6_gwp100"
        ).compute(ONE_TJ, "ammonia", "grey").total_t
        assert after < before / 2

    def test_result_reports_its_weakest_confidence(self, calc):
        """A result is only as trustworthy as its least certain input."""
        assert calc.compute(ONE_TJ, "ammonia", "grey").confidence == "provisional"
        assert calc.compute(ONE_TJ, "mgo", "fossil").confidence in {"high", "medium"}


class TestValidation:
    def test_negative_energy_rejected(self, calc):
        with pytest.raises(ValueError, match="non-negative"):
            calc.compute(-1.0, "mgo", "fossil")

    def test_zero_energy_gives_zero_emissions(self, calc):
        assert calc.compute(0.0, "mgo", "fossil").total_t == 0.0

    def test_unknown_fuel_lists_the_known_ones(self, calc):
        with pytest.raises(KeyError, match="configured"):
            calc.compute(ONE_TJ, "unobtainium", "fossil")

    def test_intensity_is_comparable_across_fuels(self, calc):
        """gCO2e/MJ is the fuel-agnostic comparison, per P-08."""
        for fuel, pathway in [("mgo", "fossil"), ("lng", "fossil"), ("ammonia", "green")]:
            result = calc.compute(ONE_TJ, fuel, pathway)
            assert result.intensity_gco2e_per_mj == pytest.approx(
                result.total_t * 1e6 / ONE_TJ, rel=1e-9
            )

    def test_breakdown_sums_to_the_total(self, calc):
        breakdown = calc.compute(ONE_TJ, "ammonia", "green").breakdown()
        parts = sum(
            v for k, v in breakdown.items() if k != "total"
        )
        assert parts == pytest.approx(breakdown["total"], rel=1e-12)


class TestF04HydrogenRange:
    def test_hydrogen_cannot_retrofit_a_long_haul_leg(self, tank):
        assessment = tank.assess("hydrogen", 5000.0, 3000.0)
        assert not assessment.feasible
        assert "short of" in assessment.reason

    def test_hydrogen_reaches_only_a_fraction_of_diesel_range(self, tank):
        volume = tank.required_tank_volume_m3(5000.0, 3000.0, "mgo")
        hydrogen = tank.max_route_nmi("hydrogen", volume, 3000.0)
        diesel = tank.max_route_nmi("mgo", volume, 3000.0)
        assert hydrogen / diesel < 0.15, "hydrogen retrofit range must collapse"

    def test_hydrogen_works_on_short_legs_and_fails_on_long_ones(self, tank):
        """The positive half of F-04: a fixed hydrogen tank is fine for a tug.

        One tank volume, two route lengths. This is the constraint that confines
        hydrogen to harbour tugs, ferries and short coastal legs rather than ruling
        it out altogether.
        """
        volume = tank.required_tank_volume_m3(100.0, 3000.0, "hydrogen")
        assert tank.assess("hydrogen", 80.0, 3000.0, tank_volume_m3=volume).feasible
        assert not tank.assess("hydrogen", 2000.0, 3000.0, tank_volume_m3=volume).feasible

    def test_a_newbuild_tank_always_covers_its_route(self, tank):
        for fuel in ("mgo", "lng", "methanol", "ammonia", "hydrogen"):
            for leg in (50.0, 1000.0, 12000.0):
                assert tank.assess(fuel, leg, 3000.0, size_tank_for="fuel").feasible

    def test_range_margin_reports_the_shortfall(self, tank):
        assert tank.assess("hydrogen", 5000.0, 3000.0).range_margin < 1.0
        assert tank.assess("mgo", 5000.0, 3000.0).range_margin >= 1.0

    def test_invalid_sizing_mode_rejected(self, tank):
        with pytest.raises(ValueError, match="size_tank_for"):
            tank.assess("mgo", 100.0, 3000.0, size_tank_for="wishful")


class TestF05TankDisplacesCargo:
    def test_low_density_fuels_cost_cargo(self, tank):
        for fuel in ("lng", "methanol", "ammonia", "hydrogen"):
            assert tank.assess(fuel, 5000.0, 3000.0, size_tank_for="fuel").cargo_displaced_t > 0

    def test_the_reference_fuel_costs_nothing(self, tank):
        assert tank.assess("mgo", 5000.0, 3000.0, size_tank_for="fuel").cargo_displaced_t == 0.0

    def test_hydrogen_costs_the_most_cargo(self, tank):
        losses = {
            fuel: tank.assess(fuel, 5000.0, 3000.0, size_tank_for="fuel").cargo_displaced_t
            for fuel in ("lng", "methanol", "ammonia", "hydrogen")
        }
        assert max(losses, key=losses.get) == "hydrogen"

    def test_longer_routes_cost_more_cargo(self, tank):
        short = tank.assess("ammonia", 1000.0, 3000.0, size_tank_for="fuel")
        long = tank.assess("ammonia", 5000.0, 3000.0, size_tank_for="fuel")
        assert long.cargo_displaced_t > short.cargo_displaced_t

    def test_electricity_has_no_tank_volume(self, tank):
        with pytest.raises(ValueError, match="energy carrier"):
            tank.usable_energy_mj(100.0, "electricity")


class TestF06FuelAvailability:
    def test_a_fuel_with_no_bunkering_is_not_usable(self, ports):
        assert not ports.can_bunker("mormugao", "hydrogen", 2024)

    def test_planned_does_not_count_as_available(self, ports):
        """Planning a fleet around announced-but-unbuilt infrastructure is the trap."""
        assert ports.availability("mormugao", "lng", 2030) == "planned"
        assert not ports.can_bunker("mormugao", "lng", 2030)

    def test_route_needs_every_port_to_supply_the_fuel(self, ports):
        route = ["mumbai", "mormugao", "cochin"]
        assert ports.route_feasible(route, "mgo", 2024)
        assert not ports.route_feasible(route, "hydrogen", 2024)

    def test_refusal_names_the_blocking_ports(self, ports):
        with pytest.raises(FuelUnavailableError, match="mormugao"):
            ports.require_route(["mumbai", "mormugao"], "hydrogen", 2030)

    def test_availability_is_a_step_not_an_interpolation(self, ports):
        """A bunkering facility is commissioned or it is not."""
        assert ports.availability("deendayal", "lng", 2027) == ports.availability(
            "deendayal", "lng", 2024
        )

    def test_availability_before_the_first_milestone_is_none(self, ports):
        assert ports.availability("mormugao", "mgo", 1990) == "none"

    def test_feasible_fuel_set_grows_over_time(self, ports):
        """S-04: multi-year scenarios must see infrastructure appear."""
        route = ["mumbai", "mormugao", "cochin"]
        early = set(ports.fuels_available_on_route(route, 2024))
        late = set(ports.fuels_available_on_route(route, 2047))
        assert early < late

    def test_earliest_year_answers_when_an_option_opens(self, ports):
        assert ports.earliest_year("mormugao", "electricity") == 2030
        assert ports.earliest_year("mormugao", "ammonia") is None

    def test_unknown_port_lists_the_known_ones(self, ports):
        with pytest.raises(KeyError, match="configured"):
            ports.get("atlantis")

    def test_empty_route_rejected(self, ports):
        with pytest.raises(ValueError, match="at least one port"):
            ports.route_feasible([], "mgo", 2024)


class TestF07ShorePowerOnACoalGrid:
    def test_the_grid_factor_is_not_zero(self, ports):
        """Shore power in India is not zero-emission, and saying so is the point."""
        assert ports.grid_factor(2024) > 0.5

    def test_the_grid_gets_cleaner_over_time(self, ports):
        """S-05: the shore-power benefit grows as the grid decarbonises."""
        assert ports.grid_factor(2047) < ports.grid_factor(2030) < ports.grid_factor(2024)

    def test_transmission_losses_raise_the_delivered_factor(self, ports):
        assert ports.grid_factor(2030, include_losses=True) > ports.grid_factor(
            2030, include_losses=False
        )

    def test_grid_factor_is_interpolated_between_milestones(self, ports):
        low, high = ports.grid_factor(2030), ports.grid_factor(2035)
        middle = ports.grid_factor(2032)
        assert high < middle < low

    def test_grid_factor_is_clamped_outside_the_table(self, ports):
        assert ports.grid_factor(1990) == ports.grid_factor(2023)
        assert ports.grid_factor(2100) == ports.grid_factor(2047)

    def test_electricity_requires_a_grid_factor(self, calc):
        with pytest.raises(ValueError, match="grid emission factor"):
            calc.compute(ONE_TJ, "electricity")

    def test_shore_power_saving_grows_as_the_grid_cleans(self, ports):
        early = ports.shore_power("jnpa", 2030, 5e5, vessel_equipped=True)
        late = ports.shore_power("jnpa", 2047, 5e5, vessel_equipped=True)
        assert late.saving_fraction > early.saving_fraction


class TestF08ShorePowerNeedsBothSides:
    def test_both_fitted_enables_shore_power(self, ports):
        result = ports.shore_power("jnpa", 2030, 5e5, vessel_equipped=True)
        assert result.available
        assert "both fitted" in result.reason

    def test_vessel_not_fitted_blocks_it(self, ports):
        result = ports.shore_power("jnpa", 2030, 5e5, vessel_equipped=False)
        assert not result.available
        assert "vessel" in result.reason
        assert result.saving_t == 0.0

    def test_port_not_fitted_blocks_it(self, ports):
        """Mormugao commissions OPS in 2028, so a 2024 call falls back to auxiliaries."""
        result = ports.shore_power("mormugao", 2024, 5e5, vessel_equipped=True)
        assert not result.available
        assert "port" in result.reason

    def test_the_same_port_works_once_commissioned(self, ports):
        """S-04: infrastructure appears over time, and the model must see it."""
        assert not ports.shore_power("mormugao", 2024, 5e5, vessel_equipped=True).available
        assert ports.shore_power("mormugao", 2028, 5e5, vessel_equipped=True).available

    def test_neither_fitted_says_so(self, ports):
        result = ports.shore_power("mormugao", 2024, 5e5, vessel_equipped=False)
        assert not result.available
        assert "neither" in result.reason

    def test_negative_energy_rejected(self, ports):
        with pytest.raises(ValueError, match="non-negative"):
            ports.shore_power("jnpa", 2030, -1.0, vessel_equipped=True)


class TestS06HaritSagar:
    @pytest.fixture(scope="class")
    def tracker(self, ports) -> HaritSagarTracker:
        return HaritSagarTracker(ports.policy)

    def test_targets_match_the_guidelines(self, tracker):
        assert tracker.required_reduction(2030) == pytest.approx(0.30)
        assert tracker.required_reduction(2047) == pytest.approx(0.70)

    def test_baseline_year_requires_no_reduction(self, tracker):
        assert tracker.required_reduction(tracker.baseline_year) == 0.0

    def test_glide_path_is_monotonic(self, tracker):
        years = list(range(2023, 2048))
        values = [tracker.required_reduction(y) for y in years]
        assert all(b >= a for a, b in zip(values, values[1:], strict=False))

    def test_beyond_the_last_milestone_holds_the_target(self, tracker):
        assert tracker.required_reduction(2060) == pytest.approx(0.70)

    def test_on_track_is_detected(self, tracker):
        status = tracker.assess(2030, emissions_t=70_000.0, cargo_t=1e6,
                                baseline_intensity=0.10)
        assert status.on_track
        assert status.achieved_reduction == pytest.approx(0.30)
        assert status.gap == 0.0

    def test_off_track_reports_the_gap(self, tracker):
        status = tracker.assess(2030, emissions_t=82_000.0, cargo_t=1e6,
                                baseline_intensity=0.10)
        assert not status.on_track
        assert status.gap == pytest.approx(0.12, abs=1e-9)
        assert status.headroom_intensity == pytest.approx(0.012, abs=1e-9)
        assert "OFF TRACK" in status.summary()

    def test_zero_cargo_is_rejected_not_divided_by(self, tracker):
        with pytest.raises(ValueError, match="cargo_t must be positive"):
            tracker.assess(2030, emissions_t=100.0, cargo_t=0.0, baseline_intensity=0.1)

    def test_glide_path_returns_intensities(self, tracker):
        path = tracker.glide_path([2030, 2047], baseline_intensity=0.10)
        assert path[2030] == pytest.approx(0.07)
        assert path[2047] == pytest.approx(0.03)


class TestS02CarbonPrice:
    def test_cost_scales_with_price(self):
        assert carbon_cost_inr(100.0, 5000.0) == 500_000.0

    def test_zero_price_costs_nothing(self):
        assert carbon_cost_inr(100.0, 0.0) == 0.0

    @pytest.mark.parametrize(("emissions", "price"), [(-1.0, 100.0), (100.0, -1.0)])
    def test_negative_inputs_rejected(self, emissions, price):
        with pytest.raises(ValueError):
            carbon_cost_inr(emissions, price)

    def test_a_carbon_price_reorders_fuels_by_cost(self, calc):
        """S-02: raising the price must make dirty fuels relatively more expensive."""
        price = 10_000.0
        grey = carbon_cost_inr(calc.compute(ONE_TJ, "ammonia", "grey").total_t, price)
        green = carbon_cost_inr(calc.compute(ONE_TJ, "ammonia", "green").total_t, price)
        assert grey > green


class TestIndiaSpecificConfiguration:
    def test_gttp_ports_are_identified(self, ports):
        assert set(ports.gttp_ports) >= {"deendayal", "jnpa", "visakhapatnam",
                                         "vo_chidambaranar"}

    def test_green_hydrogen_hubs_are_identified(self, ports):
        assert set(ports.green_hydrogen_hubs) == {"deendayal", "paradip",
                                                  "vo_chidambaranar"}

    def test_the_demo_site_is_mormugao(self, ports):
        assert ports.demo_site == "mormugao"

    def test_provenance_records_availability_confidence(self, ports):
        provenance = ports.as_provenance()
        assert provenance["confidence_availability"] == "provisional"
        assert "mormugao" in provenance["ports"]


class TestCompareAndRank:
    def test_compare_sorts_cleanest_first(self, calc):
        results = calc.compare(
            ONE_TJ,
            [("mgo", "fossil"), ("ammonia", "grey"), ("ammonia", "green")],
        )
        assert [r.pathway for r in results] == ["green", "fossil", "grey"]

    def test_every_result_is_finite(self, calc):
        for fuel, pathway in [("mgo", "fossil"), ("lng", "fossil"),
                              ("methanol", "e_fuel"), ("hydrogen", "green")]:
            assert math.isfinite(calc.compute(ONE_TJ, fuel, pathway).total_t)


class TestConversionEfficiency:
    """Shaft energy is the only fair basis for comparing fuels with batteries."""

    def test_a_diesel_needs_more_than_twice_the_fuel_energy(self):
        registry = load_fuel_registry()
        assert registry.input_energy_mj(1000.0, "mgo") == pytest.approx(1000.0 / 0.45)

    def test_an_electric_drivetrain_is_far_more_efficient(self):
        registry = load_fuel_registry()
        diesel = registry.input_energy_mj(1000.0, "mgo")
        electric = registry.input_energy_mj(1000.0, "electricity")
        assert diesel / electric > 1.9, (
            "a thermal engine needs roughly twice a drivetrain's input energy; "
            "ignoring this inverts the electrification recommendation"
        )

    def test_ignoring_efficiency_would_double_count_against_batteries(self, calc, ports):
        """The bug this guards: comparing input energy instead of shaft work."""
        shaft = ONE_TJ
        diesel = calc.compute(shaft, "mgo", "fossil").total_t
        electric = calc.compute(
            shaft, "electricity", grid_factor_t_per_mwh=ports.grid_factor(2035)
        ).total_t
        assert electric < diesel, (
            "by 2035 the Indian grid should make electric drive cleaner than diesel"
        )

    def test_negative_shaft_energy_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            load_fuel_registry().input_energy_mj(-1.0, "mgo")

    def test_electrification_crossover_is_reported(self, ports):
        """F-07: on a coal-heavy grid the carbon benefit is not immediate."""
        crossover = ports.electrification_crossover_year()
        assert crossover is not None
        assert 2030 < crossover < 2045, f"implausible crossover year {crossover}"

    def test_electric_is_worse_on_carbon_before_the_crossover(self, calc, ports):
        crossover = ports.electrification_crossover_year()
        diesel = calc.compute(ONE_TJ, "mgo", "fossil").total_t
        before = calc.compute(
            ONE_TJ, "electricity",
            grid_factor_t_per_mwh=ports.grid_factor(crossover - 5),
        ).total_t
        assert before > diesel


class TestLocalAirPollutants:
    """F-07's other half: what a port city actually breathes."""

    def test_diesel_emits_all_three_pollutants(self, calc):
        local = calc.local_pollutants(ONE_TJ, "mgo")
        assert local["nox_kg"] > 0
        assert local["sox_kg"] > 0
        assert local["pm_kg"] > 0

    def test_shore_power_is_zero_at_the_quayside(self, calc):
        local = calc.local_pollutants(ONE_TJ, "electricity")
        assert local == {"nox_kg": 0.0, "sox_kg": 0.0, "pm_kg": 0.0}

    def test_hydrogen_fuel_cell_is_zero_at_the_quayside(self, calc):
        assert calc.local_pollutants(ONE_TJ, "hydrogen")["nox_kg"] == 0.0

    def test_lng_cuts_local_pollution_sharply(self, calc):
        """Gas combustion removes sulphur entirely and most particulates."""
        diesel = calc.local_pollutants(ONE_TJ, "mgo")
        lng = calc.local_pollutants(ONE_TJ, "lng")
        assert lng["sox_kg"] == 0.0
        assert lng["nox_kg"] < diesel["nox_kg"] / 3
        assert lng["pm_kg"] < diesel["pm_kg"] / 3

    def test_local_pollutants_are_not_added_into_co2e(self, calc):
        """Adding an air-quality pollutant into a GHG total is a category error."""
        result = calc.compute(ONE_TJ, "mgo", "fossil")
        parts = sum(v for k, v in result.breakdown().items() if k != "total")
        assert parts == pytest.approx(result.breakdown()["total"], rel=1e-12)

    def test_dual_fuel_pilot_contributes_local_pollution(self, calc):
        """Green ammonia still burns pilot diesel, so it is not locally clean."""
        assert calc.local_pollutants(ONE_TJ, "ammonia")["nox_kg"] > 0

    def test_electrification_trades_carbon_for_air_quality_before_crossover(
        self, calc, ports
    ):
        """The finding that shapes the GTTP recommendation."""
        year = 2030
        diesel_co2 = calc.compute(ONE_TJ, "mgo", "fossil").total_t
        electric_co2 = calc.compute(
            ONE_TJ, "electricity", grid_factor_t_per_mwh=ports.grid_factor(year)
        ).total_t
        assert electric_co2 > diesel_co2, "2030 grid is still too dirty for a CO2 win"
        assert calc.local_pollutants(ONE_TJ, "mgo")["nox_kg"] > 0
        assert calc.local_pollutants(ONE_TJ, "electricity")["nox_kg"] == 0.0
