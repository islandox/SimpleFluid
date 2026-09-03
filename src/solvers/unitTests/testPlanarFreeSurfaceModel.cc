/**
 * @file testPlanarFreeSurfaceModel.cc
 * @brief Serial tests for the fixed-grid planar free-surface core.
 */

#include <gtest/gtest.h>

#include "geometry/unitTests/test_mesh_helpers.hh"
#include "solvers/PlanarFreeSurfaceModel.hh"
#include "utils/testing_environment.hh"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::HeadspaceOptions vented_options(double total_volume)
{
    SimpleFluid::HeadspaceOptions options;
    options.mode = SimpleFluid::HeadspaceMode::Vented;
    options.total_internal_volume = total_volume;
    return options;
}

SimpleFluid::HeadspaceOptions closed_options(double total_volume)
{
    auto options = vented_options(total_volume);
    options.mode = SimpleFluid::HeadspaceMode::Closed;
    options.initial_pressure = 1.0e5;
    options.initial_temperature = 300.0;
    return options;
}

SimpleFluid::FreeSurfaceUpdate constant_update(double liquid_volume, double bubble_volume)
{
    SimpleFluid::FreeSurfaceUpdate update;
    update.liquid_volume_at_pressure = [liquid_volume](double) { return liquid_volume; };
    update.bubble_volume_at_pressure = [bubble_volume](double) { return bubble_volume; };
    return update;
}

SimpleFluid::FreeSurfaceOptions enabled_constant_area_options()
{
    SimpleFluid::FreeSurfaceOptions options;
    options.enabled = true;
    options.mode = SimpleFluid::FreeSurfaceMode::PlanarVolumeBudget;
    options.initial_liquid_volume = 1.0;
    options.vessel.mode = SimpleFluid::VesselVolumeMapMode::ConstantArea;
    options.vessel.bottom_elevation = 0.0;
    options.vessel.top_elevation = 2.0;
    options.vessel.cross_section_area = 1.0;
    options.vessel.total_internal_volume = 2.0;
    return options;
}

} // namespace

TEST(ConstantAreaVesselVolumeMapTest, MatchesAnalyticForwardAndInverse)
{
    SimpleFluid::ConstantAreaVesselVolumeMap map(-1.0, 3.0, 2.5);

    EXPECT_DOUBLE_EQ(map.bottomElevation(), -1.0);
    EXPECT_DOUBLE_EQ(map.topElevation(), 3.0);
    EXPECT_DOUBLE_EQ(map.totalUsableVolume(), 10.0);
    EXPECT_DOUBLE_EQ(map.volumeBelow(-1.0), 0.0);
    EXPECT_DOUBLE_EQ(map.volumeBelow(3.0), 10.0);
    EXPECT_DOUBLE_EQ(map.volumeBelow(1.0), 5.0);
    EXPECT_DOUBLE_EQ(map.levelForVolume(5.0), 1.0);
    EXPECT_DOUBLE_EQ(map.areaAt(2.0), 2.5);
}

TEST(ConstantAreaVesselVolumeMapTest, RejectsMalformedAndNonFiniteData)
{
    EXPECT_THROW(SimpleFluid::ConstantAreaVesselVolumeMap(0.0, 0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(SimpleFluid::ConstantAreaVesselVolumeMap(0.0, 1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(SimpleFluid::ConstantAreaVesselVolumeMap(0.0, std::numeric_limits<double>::infinity(), 1.0),
        std::invalid_argument);
}

TEST(VesselVolumeMapRangeTest, ErrorPolicyRejectsEveryOutOfRangeRequest)
{
    SimpleFluid::ConstantAreaVesselVolumeMap map(0.0, 2.0, 1.0);
    EXPECT_THROW(static_cast<void>(map.volumeBelow(-0.1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(map.areaAt(2.1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(map.levelForVolume(-0.1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(map.levelForVolume(2.1)), std::out_of_range);
    EXPECT_THROW(
        static_cast<void>(map.levelForVolume(std::numeric_limits<double>::quiet_NaN())), std::invalid_argument);
}

TEST(VesselVolumeMapRangeTest, ClampPolicyReturnsAnExplicitReport)
{
    SimpleFluid::ConstantAreaVesselVolumeMap map(0.0, 2.0, 1.0, SimpleFluid::FreeSurfaceRangePolicy::ClampAndReport);

    const auto overflow = map.evaluateLevelForVolume(2.25);
    EXPECT_TRUE(overflow.clamped());
    EXPECT_DOUBLE_EQ(overflow.accepted, 2.0);
    EXPECT_DOUBLE_EQ(overflow.value, 2.0);
    EXPECT_DOUBLE_EQ(overflow.overflow, 0.25);

    EXPECT_DOUBLE_EQ(map.volumeBelow(-1.0), 0.0);
    EXPECT_DOUBLE_EQ(map.lastRangeEvaluation().underflow, 1.0);
}

TEST(TabulatedVesselVolumeMapTest, BracketedInterpolationIsMonotoneAndConsistent)
{
    SimpleFluid::TabulatedVesselVolumeMap map({0.0, 1.0, 3.0}, {0.0, 2.0, 8.0});

    EXPECT_DOUBLE_EQ(map.areaAt(0.5), 2.0);
    EXPECT_DOUBLE_EQ(map.areaAt(2.0), 3.0);
    EXPECT_DOUBLE_EQ(map.volumeBelow(2.0), 5.0);
    EXPECT_DOUBLE_EQ(map.levelForVolume(5.0), 2.0);
    for (const auto height : {0.0, 0.25, 1.0, 1.75, 3.0})
    {
        EXPECT_NEAR(map.levelForVolume(map.volumeBelow(height)), height, 1.0e-14);
    }
}

TEST(TabulatedVesselVolumeMapTest, RecoversAnalyticConicalProfileKnots)
{
    const auto pi = std::acos(-1.0);
    // Unit-height similar cones with r(h)=h have V(h)=pi*h^3/3.
    SimpleFluid::TabulatedVesselVolumeMap map({0.0, 0.5, 1.0, 2.0}, {0.0, pi / 24.0, pi / 3.0, 8.0 * pi / 3.0});

    for (const auto height : {0.0, 0.5, 1.0, 2.0})
    {
        const auto analytic_volume = pi * height * height * height / 3.0;
        EXPECT_NEAR(map.volumeBelow(height), analytic_volume, 1.0e-14);
        EXPECT_NEAR(map.levelForVolume(analytic_volume), height, 1.0e-14);
    }
}

TEST(TabulatedVesselVolumeMapTest, ValidatesTablesAndReportsZeroArea)
{
    EXPECT_THROW(SimpleFluid::TabulatedVesselVolumeMap({0.0}, {0.0}), std::invalid_argument);
    EXPECT_THROW(SimpleFluid::TabulatedVesselVolumeMap({0.0, 0.0}, {0.0, 1.0}), std::invalid_argument);
    EXPECT_THROW(SimpleFluid::TabulatedVesselVolumeMap({0.0, 1.0}, {0.0, -1.0}), std::invalid_argument);
    EXPECT_THROW(SimpleFluid::TabulatedVesselVolumeMap({0.0, 1.0}, {1.0, 2.0}), std::invalid_argument);

    SimpleFluid::TabulatedVesselVolumeMap plateau({0.0, 1.0, 2.0, 3.0}, {0.0, 1.0, 1.0, 3.0});
    EXPECT_THROW(static_cast<void>(plateau.areaAt(1.5)), std::domain_error);
    EXPECT_THROW(static_cast<void>(plateau.levelForVolume(1.0)), std::domain_error);

    SimpleFluid::TabulatedVesselVolumeMap bottom_plateau({0.0, 1.0, 2.0}, {0.0, 0.0, 1.0});
    SimpleFluid::TabulatedVesselVolumeMap top_plateau({0.0, 1.0, 2.0}, {0.0, 1.0, 1.0});
    EXPECT_THROW(static_cast<void>(bottom_plateau.levelForVolume(0.0)), std::domain_error);
    EXPECT_THROW(static_cast<void>(top_plateau.levelForVolume(1.0)), std::domain_error);
}

TEST(FreeSurfaceOptionsTest, DefaultIsDisabledAndFactoryReturnsNull)
{
    SimpleFluid::Database database;
    const auto options = SimpleFluid::free_surface_options_from_database(database);
    EXPECT_FALSE(options.enabled);
    EXPECT_EQ(options.mode, SimpleFluid::FreeSurfaceMode::Fixed);
    EXPECT_EQ(SimpleFluid::make_planar_free_surface_model(options), nullptr);
}

TEST(FreeSurfaceOptionsTest, PlanarAleUsesStableUnavailableSolverDiagnostic)
{
    SimpleFluid::Database database;
    database.set("free_surface_model", std::string("planarALE"));
    try
    {
        static_cast<void>(SimpleFluid::free_surface_options_from_database(database));
        FAIL() << "Expected unsupported planarALE configuration to fail.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_EQ(std::string(error.what()), SimpleFluid::planar_ale_unavailable_diagnostic);
        EXPECT_EQ(SimpleFluid::planar_ale_unavailable_diagnostic, SimpleFluid::planar_ale_immutable_mesh_diagnostic);
    }
}

TEST(FreeSurfaceOptionsTest, ParsesFlatKeysAndInitialClearLevel)
{
    SimpleFluid::Database database;
    database.set("free_surface_enabled", true);
    database.set("free_surface_model", std::string("planarVolumeBudget"));
    database.set("free_surface_vessel_model", std::string("constantArea"));
    database.set("free_surface_bottom_elevation", 1.0);
    database.set("free_surface_top_elevation", 3.0);
    database.set("free_surface_cross_section_area", 2.0);
    database.set("free_surface_total_internal_volume", 5.0);
    database.set("free_surface_initial_clear_level", 1.5);
    database.set("free_surface_overflow_policy", std::string("clampAndReport"));

    const auto options = SimpleFluid::free_surface_options_from_database(database);
    ASSERT_TRUE(options.initial_clear_level);
    EXPECT_DOUBLE_EQ(*options.initial_clear_level, 1.5);
    EXPECT_EQ(options.range_policy, SimpleFluid::FreeSurfaceRangePolicy::ClampAndReport);
    const auto initial_volume = SimpleFluid::configured_initial_liquid_volume(options);
    ASSERT_TRUE(initial_volume);
    EXPECT_DOUBLE_EQ(*initial_volume, 1.0);
    EXPECT_NE(SimpleFluid::make_planar_free_surface_model(options), nullptr);
}

TEST(FreeSurfaceOptionsTest, RequiresAnExplicitInitialLiquidInventory)
{
    auto options = enabled_constant_area_options();
    options.initial_liquid_volume.reset();
    EXPECT_THROW(SimpleFluid::validate_free_surface_options(options), std::invalid_argument);
}

TEST(FreeSurfaceOptionsTest, AcceptsCellMassInventoryOnlyWithErrorDepletion)
{
    auto options = enabled_constant_area_options();
    options.liquid_mass.mode = SimpleFluid::LiquidVolumeMode::CellMassInventory;
    EXPECT_NO_THROW(SimpleFluid::validate_free_surface_options(options));

    options.range_policy = SimpleFluid::FreeSurfaceRangePolicy::ClampAndReport;
    options.liquid_mass.depletion_policy = options.range_policy;
    EXPECT_THROW(SimpleFluid::validate_free_surface_options(options), std::invalid_argument);

    SimpleFluid::Database database;
    database.set("free_surface_enabled", true);
    database.set("free_surface_model", std::string("planarVolumeBudget"));
    database.set("free_surface_vessel_model", std::string("constantArea"));
    database.set("free_surface_bottom_elevation", 0.0);
    database.set("free_surface_top_elevation", 2.0);
    database.set("free_surface_cross_section_area", 1.0);
    database.set("free_surface_initial_liquid_volume", 1.0);
    database.set("free_surface_liquid_volume_model", std::string("cellMassInventory"));
    const auto parsed = SimpleFluid::free_surface_options_from_database(database);
    EXPECT_EQ(parsed.liquid_mass.mode, SimpleFluid::LiquidVolumeMode::CellMassInventory);
}

TEST(FreeSurfaceOptionsTest, ClampPolicyRetainsConfiguredInitialLevelRangeReport)
{
    auto options = enabled_constant_area_options();
    options.range_policy = SimpleFluid::FreeSurfaceRangePolicy::ClampAndReport;
    options.liquid_mass.depletion_policy = options.range_policy;
    options.initial_liquid_volume.reset();
    options.initial_clear_level = 2.25;
    auto model = SimpleFluid::make_planar_free_surface_model(options);
    ASSERT_NE(model, nullptr);
    const auto initial_volume = SimpleFluid::configured_initial_liquid_volume(options);
    ASSERT_TRUE(initial_volume);
    model->initialize(constant_update(*initial_volume, 0.0));

    EXPECT_DOUBLE_EQ(model->diagnostics().configured_level_overflow, 0.25);
    EXPECT_DOUBLE_EQ(model->diagnostics().configured_level_underflow, 0.0);
}

TEST(FreeSurfaceOptionsTest, RejectsAmbiguousHeadspaceInventoriesAndCapacities)
{
    {
        auto options = enabled_constant_area_options();
        options.headspace.initial_moles = {{"air", 1.0}};
        options.headspace.infer_initial_moles = false;
        EXPECT_THROW(SimpleFluid::validate_free_surface_options(options), std::invalid_argument);
    }
    {
        auto options = enabled_constant_area_options();
        options.headspace.mode = SimpleFluid::HeadspaceMode::Closed;
        options.headspace.initial_moles = {{"air", 1.0}};
        EXPECT_THROW(SimpleFluid::validate_free_surface_options(options), std::invalid_argument);
    }
    {
        auto options = enabled_constant_area_options();
        options.headspace.total_internal_volume = 3.0;
        EXPECT_THROW(SimpleFluid::validate_free_surface_options(options), std::invalid_argument);
    }
}

TEST(HeadspaceTemperatureTest, PrescribedHistoryInterpolatesAndValidates)
{
    auto options = vented_options(2.0);
    options.temperature_mode = SimpleFluid::HeadspaceTemperatureMode::Prescribed;
    options.prescribed_temperature_times = {0.0, 2.0, 5.0};
    options.prescribed_temperature_values = {300.0, 320.0, 350.0};

    EXPECT_DOUBLE_EQ(SimpleFluid::prescribed_headspace_temperature(options, 1.0), 310.0);
    EXPECT_DOUBLE_EQ(SimpleFluid::prescribed_headspace_temperature(options, 5.0), 350.0);
    EXPECT_THROW(SimpleFluid::prescribed_headspace_temperature(options, 6.0), std::out_of_range);

    options.prescribed_temperature_times = {0.0, 0.0};
    options.prescribed_temperature_values = {300.0, 310.0};
    EXPECT_THROW(SimpleFluid::prescribed_headspace_temperature(options, 0.0), std::invalid_argument);
}

TEST(VentedHeadspaceModelTest, HoldsPressureAndAccumulatesEscapedSpecies)
{
    SimpleFluid::VentedHeadspaceModel model(vented_options(4.0));
    model.initialize(1.0, 999.0);
    const auto trial = model.trialState(1.5, 999.0, {{"H2", 0.25}, {"steam", 0.5}});
    EXPECT_DOUBLE_EQ(trial.pressure, 101325.0);
    EXPECT_DOUBLE_EQ(trial.volume, 2.5);
    EXPECT_DOUBLE_EQ(trial.temperature, 293.15);
    model.commit(trial, {{"H2", 0.25}, {"steam", 0.5}});
    EXPECT_DOUBLE_EQ(model.ventedMoles().at("H2"), 0.25);
    EXPECT_DOUBLE_EQ(model.ventedMoles().at("steam"), 0.5);
    EXPECT_TRUE(model.headspaceMoles().empty());
}

TEST(ClosedIdealGasHeadspaceModelTest, MatchesAnalyticPressureAndTransfer)
{
    auto options = closed_options(3.0);
    SimpleFluid::ClosedIdealGasHeadspaceModel model(options);
    model.initialize(1.0, 300.0);
    EXPECT_NEAR(model.state().pressure, 1.0e5, 1.0e-10);
    const auto initial_moles = model.state().total_moles;

    const auto trial = model.trialState(1.5, 300.0, {{"H2", 1.0}});
    const auto expected = (initial_moles + 1.0) * options.gas_constant * 300.0 / 1.5;
    EXPECT_NEAR(trial.pressure, expected, 1.0e-10 * expected);
    model.commit(trial, {{"H2", 1.0}});
    EXPECT_DOUBLE_EQ(model.headspaceMoles().at("H2"), 1.0);
}

TEST(ClosedIdealGasHeadspaceModelTest, ExplicitInventoryMustMatchInitialThermodynamicState)
{
    auto options = closed_options(3.0);
    options.infer_initial_moles = false;
    options.initial_moles = {
        {"air", options.initial_pressure * 2.0 / (options.gas_constant * options.initial_temperature)}};
    SimpleFluid::ClosedIdealGasHeadspaceModel consistent(options);
    EXPECT_NO_THROW(consistent.initialize(1.0, options.initial_temperature));

    options.initial_moles.at("air") *= 1.1;
    SimpleFluid::ClosedIdealGasHeadspaceModel inconsistent(options);
    const auto before = inconsistent.state();
    EXPECT_THROW(inconsistent.initialize(1.0, options.initial_temperature), std::invalid_argument);
    EXPECT_TRUE(inconsistent.headspaceMoles().empty());
    EXPECT_DOUBLE_EQ(inconsistent.state().pressure, before.pressure);
    EXPECT_DOUBLE_EQ(inconsistent.state().volume, before.volume);
}

TEST(ClosedIdealGasHeadspaceModelTest, RejectsZeroHeadspace)
{
    SimpleFluid::ClosedIdealGasHeadspaceModel model(closed_options(1.0));
    EXPECT_THROW(model.initialize(1.0, 300.0), std::domain_error);
}

TEST(PlanarFreeSurfaceModelTest, VentedUpdateTracksLevelsEscapeAndGasClosure)
{
    auto map = std::make_shared<SimpleFluid::ConstantAreaVesselVolumeMap>(1.0, 3.0, 2.0);
    auto headspace = std::make_unique<SimpleFluid::VentedHeadspaceModel>(vented_options(4.0));
    SimpleFluid::PlanarFreeSurfaceModel model(map, std::move(headspace), {}, 0.01);

    auto initial = constant_update(1.0, 0.5);
    initial.gas.submerged_moles = {{"H2", 0.2}};
    initial.gas.submerged_population_moles = {{"microbubble", {{"H2", 0.15}}}, {"largeBubble", {{"H2", 0.05}}}};
    model.initialize(initial);
    auto first = model.diagnostics();
    EXPECT_DOUBLE_EQ(first.clear_level, 1.5);
    EXPECT_DOUBLE_EQ(first.pool_level, 1.75);
    EXPECT_DOUBLE_EQ(first.gas_closure_residual, 0.0);

    auto next = constant_update(1.2, 0.6);
    next.time = 0.5;
    next.time_step = 0.5;
    next.gas.submerged_moles = {{"H2", 0.1}};
    next.gas.submerged_population_moles = {{"microbubble", {{"H2", 0.075}}}, {"largeBubble", {{"H2", 0.025}}}};
    next.gas.escaped_moles_this_step = {{"H2", 0.1}};
    model.update(next);
    const auto diagnostics = model.diagnostics();
    EXPECT_DOUBLE_EQ(diagnostics.old_pool_level, 1.75);
    EXPECT_DOUBLE_EQ(diagnostics.clear_level, 1.6);
    EXPECT_DOUBLE_EQ(diagnostics.pool_level, 1.9);
    EXPECT_DOUBLE_EQ(diagnostics.time, 0.5);
    EXPECT_DOUBLE_EQ(diagnostics.time_step, 0.5);
    EXPECT_NEAR(diagnostics.clear_level_rate, 0.2, 1.0e-14);
    EXPECT_NEAR(diagnostics.pool_level_rate, 0.3, 1.0e-14);
    EXPECT_DOUBLE_EQ(diagnostics.vented_gas_moles.at("H2"), 0.1);
    EXPECT_DOUBLE_EQ(diagnostics.submerged_population_gas_moles.at("microbubble").at("H2"), 0.075);
    EXPECT_DOUBLE_EQ(model.committedEscapedMoles().at("H2"), 0.1);
    EXPECT_DOUBLE_EQ(diagnostics.gas_closure_residual, 0.0);
    EXPECT_TRUE(diagnostics.validity_warning);
    EXPECT_DOUBLE_EQ(model.headspacePressure(), 101325.0);
}

TEST(PlanarFreeSurfaceModelTest, ClampPolicyReportsOverflowWithoutHidingResidual)
{
    auto map = std::make_shared<SimpleFluid::ConstantAreaVesselVolumeMap>(
        0.0, 2.0, 1.0, SimpleFluid::FreeSurfaceRangePolicy::ClampAndReport);
    auto headspace = std::make_unique<SimpleFluid::VentedHeadspaceModel>(vented_options(2.0));
    SimpleFluid::PlanarFreeSurfaceModel model(map, std::move(headspace));
    model.initialize(constant_update(1.8, 0.5));

    const auto diagnostics = model.diagnostics();
    EXPECT_DOUBLE_EQ(diagnostics.pool_level, 2.0);
    EXPECT_NEAR(diagnostics.overflow_volume, 0.3, 1.0e-14);
    EXPECT_NEAR(diagnostics.volume_closure_residual, -0.3, 1.0e-14);
}

TEST(PlanarFreeSurfaceModelTest, ClampPolicyReportsDryoutDeficit)
{
    auto map = std::make_shared<SimpleFluid::ConstantAreaVesselVolumeMap>(
        0.0, 2.0, 1.0, SimpleFluid::FreeSurfaceRangePolicy::ClampAndReport);
    auto headspace = std::make_unique<SimpleFluid::VentedHeadspaceModel>(vented_options(2.0));
    SimpleFluid::PlanarFreeSurfaceModel model(map, std::move(headspace));
    model.initialize(constant_update(0.5, 0.0));
    model.update(constant_update(-0.2, 0.0));

    const auto diagnostics = model.diagnostics();
    EXPECT_DOUBLE_EQ(diagnostics.liquid_volume, 0.0);
    EXPECT_DOUBLE_EQ(diagnostics.pool_level, 0.0);
    EXPECT_DOUBLE_EQ(diagnostics.dryout_deficit, 0.2);
}

TEST(PlanarFreeSurfaceModelTest, ClosedClosureRaisesPressureAsPoolFalls)
{
    auto map = std::make_shared<SimpleFluid::ConstantAreaVesselVolumeMap>(0.0, 5.0, 1.0);
    auto headspace = std::make_unique<SimpleFluid::ClosedIdealGasHeadspaceModel>(closed_options(10.0));
    SimpleFluid::PlanarFreeSurfaceModel model(map, std::move(headspace));

    SimpleFluid::FreeSurfaceUpdate initial;
    initial.liquid_volume_at_pressure = [](double) { return 1.8; };
    initial.bubble_volume_at_pressure = [](double pressure) { return 2.0e4 / pressure; };
    initial.gas.submerged_moles = {{"H2", 20.0}};
    model.initialize(initial);
    const auto old = model.diagnostics();
    EXPECT_NEAR(old.headspace.pressure, 1.0e5, 1.0e-4);

    SimpleFluid::FreeSurfaceUpdate next;
    next.liquid_volume_at_pressure = [](double) { return 1.8; };
    next.bubble_volume_at_pressure = [](double pressure) { return 1.0e4 / pressure; };
    next.gas.submerged_moles = {{"H2", 10.0}};
    next.gas.escaped_moles_this_step = {{"H2", 10.0}};
    model.update(next);
    const auto diagnostics = model.diagnostics();
    EXPECT_GT(diagnostics.headspace.pressure, old.headspace.pressure);
    EXPECT_LT(diagnostics.pool_level, old.pool_level);
    EXPECT_GT(diagnostics.nonlinear_iterations, 0);
    EXPECT_LE(std::abs(diagnostics.nonlinear_residual), 1.0e-6);
    EXPECT_NEAR(diagnostics.gas_closure_residual, 0.0, 1.0e-12);
}

TEST(PlanarFreeSurfaceModelTest, ClosedClosureAcceptsRootAtPressureBound)
{
    auto map = std::make_shared<SimpleFluid::ConstantAreaVesselVolumeMap>(0.0, 2.0, 1.0);
    auto headspace = std::make_unique<SimpleFluid::ClosedIdealGasHeadspaceModel>(closed_options(3.0));
    SimpleFluid::FreeSurfaceCouplingOptions coupling;
    coupling.minimum_absolute_pressure = 1.0e5;
    coupling.maximum_absolute_pressure = 2.0e5;
    SimpleFluid::PlanarFreeSurfaceModel model(map, std::move(headspace), coupling);

    model.initialize(constant_update(1.0, 0.0));
    EXPECT_NEAR(model.diagnostics().headspace.pressure, 1.0e5, 1.0e-10);
    EXPECT_EQ(model.diagnostics().nonlinear_iterations, 0);
}

TEST(PlanarFreeSurfaceModelTest, ClosedClosureHonorsCallbackPressureDomain)
{
    auto map = std::make_shared<SimpleFluid::ConstantAreaVesselVolumeMap>(0.0, 2.0, 1.0);
    auto headspace = std::make_unique<SimpleFluid::ClosedIdealGasHeadspaceModel>(closed_options(3.0));
    SimpleFluid::PlanarFreeSurfaceModel model(map, std::move(headspace));
    auto update = constant_update(1.0, 0.0);
    update.minimum_valid_absolute_pressure = 9.0e4;
    update.bubble_volume_at_pressure = [](double pressure)
    {
        if (pressure < 9.0e4)
        {
            throw std::domain_error("pressure below callback domain");
        }
        return 0.0;
    };

    EXPECT_NO_THROW(model.initialize(update));
    EXPECT_NEAR(model.diagnostics().headspace.pressure, 1.0e5, 1.0e-10);
}

TEST(PlanarFreeSurfaceModelTest, ClosedClosureKeepsAcceptedPressureConsistentWithVolumeCallbacks)
{
    auto map = std::make_shared<SimpleFluid::ConstantAreaVesselVolumeMap>(0.0, 3.0, 1.0);
    auto headspace = std::make_unique<SimpleFluid::ClosedIdealGasHeadspaceModel>(closed_options(4.0));
    SimpleFluid::FreeSurfaceCouplingOptions coupling;
    coupling.absolute_tolerance = 1.0e6; // Deliberately looser than the EOS consistency gate.
    coupling.relative_tolerance = 0.0;
    SimpleFluid::PlanarFreeSurfaceModel model(map, std::move(headspace), coupling);
    double bubble_evaluation_pressure = 0.0;
    SimpleFluid::FreeSurfaceUpdate update;
    update.liquid_volume_at_pressure = [](double) { return 1.0; };
    update.bubble_volume_at_pressure = [&bubble_evaluation_pressure](double pressure)
    {
        bubble_evaluation_pressure = pressure;
        return 2.0e4 / pressure;
    };

    model.initialize(update);
    const auto diagnostics = model.diagnostics();
    EXPECT_NEAR(diagnostics.headspace.pressure, bubble_evaluation_pressure, 2.0e-5);
    EXPECT_LE(std::abs(diagnostics.nonlinear_residual), 2.0e-5);
}

TEST(PlanarFreeSurfaceModelTest, GasClosureMismatchRejectsAcceptedUpdate)
{
    auto map = std::make_shared<SimpleFluid::ConstantAreaVesselVolumeMap>(0.0, 2.0, 1.0);
    auto headspace = std::make_unique<SimpleFluid::VentedHeadspaceModel>(vented_options(2.0));
    SimpleFluid::PlanarFreeSurfaceModel model(map, std::move(headspace));
    auto initial = constant_update(1.0, 0.0);
    initial.gas.submerged_moles = {{"H2", 1.0}};
    model.initialize(initial);

    auto inconsistent = constant_update(1.0, 0.0);
    inconsistent.gas.submerged_moles = {{"H2", 0.5}};
    EXPECT_THROW(model.update(inconsistent), std::runtime_error);
    EXPECT_TRUE(model.headspace().ventedMoles().empty());
    EXPECT_TRUE(model.committedEscapedMoles().empty());
}

TEST(PlanarFreeSurfaceModelTest, FailedInitializationRestoresHeadspaceState)
{
    auto map = std::make_shared<SimpleFluid::ConstantAreaVesselVolumeMap>(0.0, 2.0, 1.0);
    auto headspace = std::make_unique<SimpleFluid::VentedHeadspaceModel>(vented_options(2.0));
    SimpleFluid::PlanarFreeSurfaceModel model(map, std::move(headspace));
    const auto before = model.headspace().state();
    auto inconsistent = constant_update(1.0, 0.0);
    inconsistent.gas.initial_moles = {{"H2", 1.0}};
    inconsistent.gas.submerged_moles = {{"H2", 0.5}};

    EXPECT_THROW(model.initialize(inconsistent), std::runtime_error);
    EXPECT_FALSE(model.initialized());
    EXPECT_TRUE(model.headspace().ventedMoles().empty());
    EXPECT_DOUBLE_EQ(model.headspace().state().pressure, before.pressure);
    EXPECT_DOUBLE_EQ(model.headspace().state().volume, before.volume);
}

TEST(PlanarFreeSurfaceModelTest, ReportsClosedNonlinearConvergenceFailure)
{
    auto map = std::make_shared<SimpleFluid::ConstantAreaVesselVolumeMap>(0.0, 2.0, 1.0);
    auto headspace = std::make_unique<SimpleFluid::ClosedIdealGasHeadspaceModel>(closed_options(3.0));
    SimpleFluid::FreeSurfaceCouplingOptions coupling;
    coupling.maximum_correctors = 1;
    coupling.absolute_tolerance = 1.0e-30;
    coupling.relative_tolerance = 0.0;
    SimpleFluid::PlanarFreeSurfaceModel model(map, std::move(headspace), coupling);

    try
    {
        model.initialize(constant_update(1.0, 0.0));
        FAIL() << "Expected the one-corrector closed solve to fail.";
    }
    catch (const std::runtime_error& error)
    {
        EXPECT_NE(std::string(error.what()).find("failed to converge after 1 correctors"), std::string::npos);
    }
}

TEST(LiquidMassInventoryTest, PureDensityControlsVolumeAndPhaseChangeOnce)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_single_hex_database());
    SimpleFluid::LiquidMassInventory<Pack> inventory(mesh);
    inventory.initialize(1.0, [](Pack::local_ordinal_type) { return 1000.0; });
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 1000.0);
    EXPECT_DOUBLE_EQ(inventory.liquidVolume(), 1.0);
    EXPECT_EQ(inventory.rhoLiquid().name(), "rhoLiquid");

    inventory.updatePureLiquidDensity([](Pack::local_ordinal_type) { return 800.0; });
    EXPECT_DOUBLE_EQ(inventory.liquidVolume(), 1.25);
    inventory.updatePhaseChange(100.0);
    const auto diagnostics = inventory.diagnostics();
    EXPECT_DOUBLE_EQ(diagnostics.total_mass, 900.0);
    EXPECT_DOUBLE_EQ(diagnostics.cumulative_evaporated_mass, 100.0);
    EXPECT_DOUBLE_EQ(diagnostics.liquid_volume, 1.125);
    EXPECT_DOUBLE_EQ(diagnostics.mass_balance_residual, 0.0);
}

TEST(LiquidMassInventoryTest, PhaseChangePreviewDoesNotMutateUntilCommitted)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_single_hex_database());
    SimpleFluid::LiquidMassInventory<Pack> inventory(mesh);
    inventory.initialize(1.0, [](Pack::local_ordinal_type) { return 1000.0; });

    const auto preview = inventory.previewPhaseChange(125.0, 25.0);
    const auto& diagnostics = preview.diagnostics();
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 1000.0);
    EXPECT_DOUBLE_EQ(diagnostics.total_mass, 900.0);
    EXPECT_DOUBLE_EQ(diagnostics.liquid_volume, 0.9);
    inventory.commitPhaseChange(preview);
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 900.0);
    EXPECT_DOUBLE_EQ(inventory.diagnostics().mass_balance_residual, 0.0);
    EXPECT_THROW(inventory.commitPhaseChange(preview), std::logic_error);
}

TEST(LiquidMassInventoryTest, DensityChangeInvalidatesGlobalPhaseChangePreview)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_single_hex_database());
    SimpleFluid::LiquidMassInventory<Pack> inventory(mesh);
    inventory.initialize(1.0, [](Pack::local_ordinal_type) { return 1000.0; });

    const auto stale = inventory.previewPhaseChange(100.0);
    inventory.updatePureLiquidDensity([](Pack::local_ordinal_type) { return 500.0; });

    EXPECT_THROW(inventory.commitPhaseChange(stale), std::logic_error);
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 1000.0);
    EXPECT_DOUBLE_EQ(inventory.liquidVolume(), 2.0);

    const auto current = inventory.previewPhaseChange(100.0);
    EXPECT_NO_THROW(inventory.commitPhaseChange(current));
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 900.0);
    EXPECT_DOUBLE_EQ(inventory.liquidVolume(), 1.8);
}

TEST(LiquidMassInventoryTest, NonuniformDensityUsesFixedMassWeights)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database());
    SimpleFluid::LiquidMassInventory<Pack> inventory(mesh);
    inventory.initialize(2.0, [](Pack::local_ordinal_type cell) { return cell == 0 ? 1000.0 : 500.0; });
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 1500.0);
    EXPECT_DOUBLE_EQ(inventory.liquidVolume(), 2.0);

    inventory.updatePureLiquidDensity([](Pack::local_ordinal_type cell) { return cell == 0 ? 800.0 : 400.0; });
    EXPECT_NEAR(inventory.liquidVolume(), 2.5, 1.0e-14);
}

TEST(LiquidMassInventoryTest, VoidOnlyChangeCannotAlterLiquidMaterialVolume)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_single_hex_database());
    SimpleFluid::LiquidMassInventory<Pack> inventory(mesh);
    inventory.initialize(1.0, [](Pack::local_ordinal_type) { return 1000.0; });
    const auto before = inventory.liquidVolume();
    double unrelated_void_fraction = 0.0;
    unrelated_void_fraction = 0.75;
    inventory.updatePureLiquidDensity(
        [unrelated_void_fraction](Pack::local_ordinal_type)
        {
            static_cast<void>(unrelated_void_fraction);
            return 1000.0;
        });
    EXPECT_DOUBLE_EQ(inventory.liquidVolume(), before);
}

TEST(LiquidMassInventoryTest, ClampPolicyReportsMassDryout)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_single_hex_database());
    SimpleFluid::LiquidMassInventoryOptions options;
    options.initial_liquid_mass = 10.0;
    options.depletion_policy = SimpleFluid::FreeSurfaceRangePolicy::ClampAndReport;
    SimpleFluid::LiquidMassInventory<Pack> inventory(mesh, options);
    inventory.initialize(0.01, [](Pack::local_ordinal_type) { return 1000.0; });
    inventory.updatePhaseChange(12.0);
    const auto diagnostics = inventory.diagnostics();
    EXPECT_DOUBLE_EQ(diagnostics.total_mass, 0.0);
    EXPECT_DOUBLE_EQ(diagnostics.cumulative_evaporated_mass, 10.0);
    EXPECT_DOUBLE_EQ(diagnostics.dryout_mass_deficit, 2.0);
    EXPECT_DOUBLE_EQ(diagnostics.mass_balance_residual, 0.0);
}

TEST(LiquidMassInventoryTest, CellwisePhaseChangeIsTransactionalAndSpatiallyResolved)
{
    using Inventory = SimpleFluid::LiquidMassInventory<Pack>;
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database());
    SimpleFluid::LiquidMassInventoryOptions options;
    options.mode = SimpleFluid::LiquidVolumeMode::CellMassInventory;
    Inventory inventory(mesh, options);
    inventory.initialize(2.0, [](Pack::local_ordinal_type) { return 100.0; });
    EXPECT_EQ(inventory.mode(), SimpleFluid::LiquidVolumeMode::CellMassInventory);
    EXPECT_EQ(inventory.cellMassInventory().name(), "liquidMassInventory");

    Inventory::face_flux_field_type flux(mesh, 0.0, "liquidMassFlux");
    Inventory::field_type evaporation(mesh, 0.0, "evaporationMassRate");
    Inventory::field_type condensation(mesh, 0.0, "condensationMassRate");
    evaporation.set_owned_value(0, 10.0);
    condensation.set_owned_value(1, 4.0);
    evaporation.sync_ghosts();
    condensation.sync_ghosts();

    const auto before_0 = inventory.cellMassInventory().value(0);
    const auto before_1 = inventory.cellMassInventory().value(1);
    const auto preview = inventory.previewCellwiseAdvance(0.5, flux, &evaporation, &condensation);
    ASSERT_TRUE(preview.transportStatistics().has_value());
    EXPECT_TRUE(preview.transportStatistics()->converged);
    EXPECT_DOUBLE_EQ(inventory.cellMassInventory().value(0), before_0);
    EXPECT_DOUBLE_EQ(inventory.cellMassInventory().value(1), before_1);
    EXPECT_NEAR(preview.diagnostics().total_mass, 197.0, 1.0e-10);
    EXPECT_NEAR(preview.diagnostics().cumulative_evaporated_mass, 5.0, 1.0e-12);
    EXPECT_NEAR(preview.diagnostics().cumulative_condensed_mass, 2.0, 1.0e-12);
    EXPECT_NEAR(preview.diagnostics().liquid_volume, 1.97, 1.0e-12);
    EXPECT_NEAR(preview.diagnostics().step_mass_balance_residual, 0.0, 1.0e-10);

    inventory.commitPhaseChange(preview);
    EXPECT_NEAR(inventory.cellMassInventory().value(0), 95.0, 1.0e-10);
    EXPECT_NEAR(inventory.cellMassInventory().value(1), 102.0, 1.0e-10);
    EXPECT_NEAR(inventory.totalMass(), 197.0, 1.0e-10);
    EXPECT_THROW(inventory.commitPhaseChange(preview), std::logic_error);
}

TEST(LiquidMassInventoryTest, CellwiseInternalAdvectionRedistributesButConservesMass)
{
    using Inventory = SimpleFluid::LiquidMassInventory<Pack>;
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database());
    SimpleFluid::LiquidMassInventoryOptions options;
    options.mode = SimpleFluid::LiquidVolumeMode::CellMassInventory;
    Inventory inventory(mesh, options);
    inventory.initialize(2.0, [](Pack::local_ordinal_type cell) { return cell == 0 ? 200.0 : 100.0; });
    inventory.updatePureLiquidDensity([](Pack::local_ordinal_type) { return 100.0; });

    Inventory::face_flux_field_type flux(mesh, 0.0, "liquidMassFlux");
    Pack::local_ordinal_type interior_face = -1;
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid = static_cast<Pack::local_ordinal_type>(face);
        if (mesh->is_interior_face(face_lid) && flux.is_owned_face(face_lid))
        {
            interior_face = face_lid;
            break;
        }
    }
    ASSERT_GE(interior_face, 0);
    flux.set_value(interior_face, 0.2);
    const auto old_owner = inventory.cellMassInventory().value(mesh->owner_cell(interior_face));
    const auto old_neighbor = inventory.cellMassInventory().value(mesh->neighbor_cell(interior_face));

    const auto preview = inventory.previewCellwiseAdvance(0.5, flux);
    EXPECT_NEAR(preview.diagnostics().total_mass, 300.0, 1.0e-8);
    EXPECT_NEAR(preview.diagnostics().liquid_volume, 3.0, 1.0e-10);
    inventory.commitPhaseChange(preview);
    EXPECT_LT(inventory.cellMassInventory().value(mesh->owner_cell(interior_face)), old_owner);
    EXPECT_GT(inventory.cellMassInventory().value(mesh->neighbor_cell(interior_face)), old_neighbor);
    EXPECT_NEAR(inventory.totalMass(), 300.0, 1.0e-8);
}

TEST(LiquidMassInventoryTest, CellwiseDryoutAndBoundaryFluxRejectWithoutCommit)
{
    using Inventory = SimpleFluid::LiquidMassInventory<Pack>;
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_single_hex_database());
    SimpleFluid::LiquidMassInventoryOptions options;
    options.mode = SimpleFluid::LiquidVolumeMode::CellMassInventory;
    Inventory inventory(mesh, options);
    inventory.initialize(1.0, [](Pack::local_ordinal_type) { return 10.0; });
    Inventory::face_flux_field_type flux(mesh, 0.0, "liquidMassFlux");
    Inventory::field_type evaporation(mesh, 20.0, "evaporationMassRate");
    const auto mass_before = inventory.totalMass();
    EXPECT_THROW(static_cast<void>(inventory.previewCellwiseAdvance(1.0, flux, &evaporation)), std::out_of_range);
    EXPECT_DOUBLE_EQ(inventory.totalMass(), mass_before);
    EXPECT_DOUBLE_EQ(inventory.cellMassInventory().value(0), 10.0);

    ASSERT_FALSE(flux.owned_face_ids().empty());
    flux.set_value(flux.owned_face_ids().front(), 1.0e-3);
    EXPECT_THROW(static_cast<void>(inventory.previewCellwiseAdvance(0.1, flux)), std::invalid_argument);
    EXPECT_DOUBLE_EQ(inventory.totalMass(), mass_before);

    flux.put_scalar(0.0);
    SimpleFluid::LinearSolverOptions cg_options;
    cg_options.backend = SimpleFluid::LinearSolverBackend::Cg;
    EXPECT_THROW(static_cast<void>(inventory.previewCellwiseAdvance(0.1, flux, nullptr, nullptr, cg_options)),
        std::invalid_argument);
}

TEST(LiquidMassInventoryTest, NewCellwisePreviewInvalidatesOlderTrialToken)
{
    using Inventory = SimpleFluid::LiquidMassInventory<Pack>;
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_single_hex_database());
    SimpleFluid::LiquidMassInventoryOptions options;
    options.mode = SimpleFluid::LiquidVolumeMode::CellMassInventory;
    Inventory inventory(mesh, options);
    inventory.initialize(1.0, [](Pack::local_ordinal_type) { return 1000.0; });
    Inventory::face_flux_field_type flux(mesh, 0.0, "liquidMassFlux");

    const auto first = inventory.previewCellwiseAdvance(0.1, flux);
    const auto second = inventory.previewCellwiseAdvance(0.1, flux);
    EXPECT_THROW(inventory.commitPhaseChange(first), std::logic_error);
    EXPECT_NO_THROW(inventory.commitPhaseChange(second));
}

TEST(LiquidMassInventoryTest, DensityChangeInvalidatesCellwiseTransportPreview)
{
    using Inventory = SimpleFluid::LiquidMassInventory<Pack>;
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database());
    SimpleFluid::LiquidMassInventoryOptions options;
    options.mode = SimpleFluid::LiquidVolumeMode::CellMassInventory;
    Inventory inventory(mesh, options);
    inventory.initialize(2.0, [](Pack::local_ordinal_type) { return 100.0; });
    Inventory::face_flux_field_type flux(mesh, 0.0, "liquidMassFlux");

    const auto stale = inventory.previewCellwiseAdvance(0.1, flux);
    inventory.updatePureLiquidDensity([](Pack::local_ordinal_type) { return 50.0; });

    EXPECT_THROW(inventory.commitPhaseChange(stale), std::logic_error);
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 200.0);
    EXPECT_DOUBLE_EQ(inventory.liquidVolume(), 4.0);
    EXPECT_DOUBLE_EQ(inventory.cellMassInventory().value(0), 100.0);
    EXPECT_DOUBLE_EQ(inventory.cellMassInventory().value(1), 100.0);

    const auto current = inventory.previewCellwiseAdvance(0.1, flux);
    EXPECT_NO_THROW(inventory.commitPhaseChange(current));
    EXPECT_NEAR(inventory.liquidVolume(), 4.0, 1.0e-12);
}

TEST(LiquidMassInventoryTest, LooseLinearToleranceCannotRelaxPhysicalMassClosure)
{
    using Inventory = SimpleFluid::LiquidMassInventory<Pack>;
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database());
    SimpleFluid::LiquidMassInventoryOptions options;
    options.mode = SimpleFluid::LiquidVolumeMode::CellMassInventory;
    Inventory inventory(mesh, options);
    inventory.initialize(2.0, [](Pack::local_ordinal_type) { return 100.0; });
    Inventory::face_flux_field_type flux(mesh, 0.0, "liquidMassFlux");
    Inventory::field_type evaporation(mesh, 0.0, "evaporationMassRate");
    evaporation.set_owned_value(0, 50.0);
    evaporation.sync_ghosts();

    SimpleFluid::LinearSolverOptions loose;
    loose.max_iterations = 1;
    loose.tolerance = 2.0;
    try
    {
        static_cast<void>(inventory.previewCellwiseAdvance(1.0, flux, &evaporation, nullptr, loose));
        FAIL() << "Expected strict physical mass closure to reject the loose transport solve.";
    }
    catch (const std::runtime_error& error)
    {
        EXPECT_NE(std::string(error.what()).find("strict physical tolerance"), std::string::npos);
    }
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 200.0);
    EXPECT_DOUBLE_EQ(inventory.cellMassInventory().value(0), 100.0);
    EXPECT_DOUBLE_EQ(inventory.cellMassInventory().value(1), 100.0);
}
