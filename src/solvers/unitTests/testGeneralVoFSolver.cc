/** @file testGeneralVoFSolver.cc */
#include "general_vof_solver_test_helpers.hh"
#include <iostream>
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
TEST(GeneralVoFSolverTest, ClosedTwoByTwoSingleLayerConstantCarrierAdvancesAndRejectsAtomically)
{
    GeneralVoFSolver solver(phases(0, false, true), inviscid());
    const auto before = solver.diagnostics();
    const auto accepted = solver.phases();
    EXPECT_THROW((void) solver.step(-1), std::runtime_error);
    EXPECT_EQ(solver.phases(), accepted);
    EXPECT_EQ(solver.time(), 0);
    const auto after = solver.step(.01);
    EXPECT_EQ(after.gas_volume, 0);
    EXPECT_EQ(after.gas_mass, 0);
    EXPECT_EQ(after.liquid_mass, before.liquid_mass);
    EXPECT_EQ(after.maximum_velocity, 0);
    EXPECT_EQ(after.maximum_divergence, 0);
    EXPECT_EQ(solver.time(), .01);
    const auto preparation = solver.prepared_transport();
    ASSERT_TRUE(preparation);
    EXPECT_EQ(preparation->family, VoF::GeneralVoFTransportFamily::AlgebraicFaceVolume);
    EXPECT_EQ(preparation->source_identity, accepted->accepted_identity());
    EXPECT_EQ(solver.phases()->accepted_identity(), accepted->accepted_identity());
    EXPECT_EQ(preparation->candidate_identity, accepted->accepted_identity());
    EXPECT_TRUE(preparation->streams.empty());
    EXPECT_TRUE(preparation->momentum_preparation);
    EXPECT_EQ(preparation->momentum_builds, 1u);
    EXPECT_EQ(preparation->momentum_reuses, 2u);
    for (const auto& result : solver.momentum_transport_diagnostics())
        EXPECT_EQ(result.physical_relative_residual, 0);
    const auto phases_after = solver.phases();
    const auto velocity_after = solver.velocity();
    const auto face_velocity_after = solver.face_velocity();
    const auto pressure_after = solver.pressure();
    EXPECT_THROW((void) solver.step(-1), std::runtime_error);
    EXPECT_EQ(solver.phases(), phases_after);
    EXPECT_EQ(solver.prepared_transport(), preparation);
    EXPECT_EQ(solver.velocity(), velocity_after);
    EXPECT_EQ(solver.face_velocity(), face_velocity_after);
    EXPECT_EQ(solver.pressure(), pressure_after);
    EXPECT_EQ(solver.time(), .01);
}
TEST(GeneralVoFSolverTest, SingleLayerNormalGravityHasWallPressureReactionAndZeroAcceptedMomentum)
{
    auto material = inviscid();
    material.gas_density = material.liquid_density = 1;
    GeneralVoFSolverOptions options;
    options.flow.gravity = {0, 0, 1};
    GeneralVoFSolver solver(phases(0, false, true), material, options);
    const auto before = solver.diagnostics();
    EXPECT_EQ(before.maximum_velocity, 0);
    const auto after = solver.step(.125);
    EXPECT_DOUBLE_EQ(solver.time(), .125);
    EXPECT_EQ(after.liquid_mass, before.liquid_mass);
    for (const auto& velocity : solver.velocity())
        for (const auto component : velocity)
            EXPECT_NEAR(component, 0, 1e-12);
    for (const auto velocity : solver.face_velocity())
        EXPECT_NEAR(velocity, 0, 1e-12);
    for (const auto momentum : after.momentum)
        EXPECT_NEAR(momentum, 0, 1e-12);
    EXPECT_LT(after.maximum_divergence, options.flow.divergence_tolerance);
    EXPECT_NEAR(after.maximum_velocity, 0, 1e-12);
}
TEST(GeneralVoFSolverTest, SingleLayerInitialNormalMotionUsesPressureWithoutApplyingGravity)
{
    auto material = inviscid();
    material.gas_density = material.liquid_density = 1;
    GeneralVoFSolverOptions options;
    options.flow.gravity = {0, 0, 17};
    GeneralVoFSolver solver(phases(0, false, true), material, options);
    EXPECT_EQ(solver.diagnostics().maximum_velocity, 0);
    std::vector<VoF::Vector> initial(solver.mesh()->num_owned_cells(), {0, 0, .25});
    solver.initialize(initial);
    EXPECT_EQ(solver.time(), 0);
    for (const auto& velocity : solver.velocity())
        for (const auto component : velocity)
            EXPECT_NEAR(component, 0, 1e-12);
    for (const auto velocity : solver.face_velocity())
        EXPECT_NEAR(velocity, 0, 1e-12);
    for (const auto momentum : solver.diagnostics().momentum)
        EXPECT_NEAR(momentum, 0, 1e-12);
}
TEST(GeneralVoFSolverTest, FiniteSolverKeepsSelectedGeometryThroughStepAndRegrid)
{
    const auto source = phases(0, false, true);
    const auto comm = Tpetra::getDefaultComm();
    const auto reset = []
    {
        PLIC::reset_geometry_statistics();
        VoF::GeneralVoF::reset_geometry_statistics();
        GeneralVoFSolver::reset_geometry_statistics();
    };
    const auto selected = [&](const char* stage)
    {
        SCOPED_TRACE(stage);
        const auto fvm = VoF::GeneralVoF::geometry_statistics().bounded;
        const auto solver = GeneralVoFSolver::geometry_statistics().bounded;
        EXPECT_EQ(fvm.runtime_dispatches, 0u);
        EXPECT_EQ(solver.runtime_dispatches, 0u);
        // Empty owners participate in all operations. Establish actual selected
        // geometry/arithmetic work while checking dispatch on every rank.
        const double local = double(
            fvm.volume_evaluations + solver.volume_evaluations + fvm.ledger_operations + solver.ledger_operations);
        double global = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local, &global);
        EXPECT_GT(global, 0);
    };
    reset();
    GeneralVoFSolver solver(source, inviscid());
    const int local_owners = solver.mesh()->num_owned_cells() != 0;
    const int local_empty = !local_owners;
    int owners = 0, empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_owners, &owners);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(owners, std::min(2, comm->getSize()));
    EXPECT_EQ(empty, comm->getSize() - std::min(2, comm->getSize()));
    selected("construction");
    const auto initial = solver.diagnostics();
    const auto accepted = solver.phases();
    reset();
    EXPECT_THROW((void) solver.step(-1), std::runtime_error);
    EXPECT_EQ(solver.phases(), accepted);
    const auto advanced = solver.step(.01);
    selected("step and rejection recovery");
    EXPECT_EQ(advanced.gas_mass, 0);
    EXPECT_EQ(advanced.liquid_mass, initial.liquid_mass);
    EXPECT_EQ(advanced.maximum_velocity, 0);

    std::vector<AMR::AdaptiveMark> refine;
    for (const auto& key : solver.phases()->owned_order())
        refine.push_back({key, AMR::AdaptiveSplit::Isotropic});
    reset();
    solver.regrid(refine);
    selected("regrid");
    EXPECT_EQ(solver.time(), .01);
    EXPECT_EQ(solver.diagnostics().gas_mass, 0);
    EXPECT_EQ(solver.diagnostics().liquid_mass, initial.liquid_mass);
    EXPECT_EQ(solver.diagnostics().maximum_velocity, 0);
    const double local_cells = double(solver.mesh()->num_owned_cells());
    double global_cells = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_cells, &global_cells);
    EXPECT_EQ(global_cells, 32);
}

TEST(GeneralVoFSolverTest, NativeSourceFreeStationaryAdvanceAndRejectedStepRetainAcceptedPublication)
{
    GeneralVoFSolver solver(phases(0), inviscid());
    const auto initial = solver.phases();
    const auto fraction = solver.gas_fraction();
    EXPECT_THROW((void) solver.step(-1), std::runtime_error);
    EXPECT_EQ(solver.phases(), initial);
    EXPECT_EQ(solver.gas_fraction(), fraction);
    EXPECT_EQ(solver.time(), 0);
    for (int step = 0; step < 3; ++step)
    {
        const auto diagnostic = solver.step(.01);
        EXPECT_NEAR(diagnostic.time, .01 * (step + 1), 1e-15);
        EXPECT_EQ(diagnostic.gas_volume, 0);
        EXPECT_EQ(diagnostic.maximum_velocity, 0);
        EXPECT_EQ(diagnostic.maximum_divergence, 0);
    }
}
TEST(GeneralVoFSolverTest, IndependentMixedPlaneRetainsBothPhaseMassesInSourceFreeStationaryFlow)
{
    GeneralVoFSolver solver(phases(2), inviscid());
    const auto before = solver.diagnostics();
    for (int step = 0; step < 3; ++step)
    {
        const auto after = solver.step(.01);
        EXPECT_NEAR(after.gas_volume, before.gas_volume, 1e-12);
        EXPECT_NEAR(after.gas_mass, before.gas_mass, 1e-12);
        EXPECT_NEAR(after.liquid_mass, before.liquid_mass, 1e-10);
        EXPECT_EQ(after.maximum_velocity, 0);
    }
}
TEST(GeneralVoFSolverTest, SweptPlanarBinaryInterfaceAdmitsPhysicalZeroCurvature)
{
    const auto source = phases(1);
    auto material = inviscid();
    material.surface_tension = .07;
    GeneralVoFSolver solver(source, material);
    for (const auto curvature : solver.curvature())
        EXPECT_EQ(curvature, 0);
    const auto before = solver.diagnostics();
    const auto after = solver.step(.01);
    EXPECT_EQ(after.maximum_velocity, 0);
    EXPECT_EQ(after.gas_mass, before.gas_mass);
    EXPECT_EQ(after.liquid_mass, before.liquid_mass);
}
TEST(GeneralVoFSolverTest, SymmetricNativeRegridRetainsAcceptedPhysicalMomentumAtFixedClock)
{
    GeneralVoFSolver solver(phases(0), inviscid());
    std::vector<VoF::Vector> velocity(solver.mesh()->num_owned_cells());
    for (std::size_t c = 0; c < velocity.size(); ++c)
    {
        const auto x = solver.mesh()->cell_centroid(DefaultTpetraTypes::local_ordinal_type(c));
        const double a = std::numbers::pi * x.x / 3, b = std::numbers::pi * x.y / 3;
        velocity[c] = {std::sin(a) * std::sin(a) * std::sin(2 * b), -std::sin(b) * std::sin(b) * std::sin(2 * a), 0};
    }
    solver.initialize(velocity);
    const auto immutable_pre_step = solver.phases();
    (void) solver.step(.001);
    const auto before = solver.diagnostics();
    ASSERT_GT(before.time, 0);
    const auto source = solver.phases();
    EXPECT_NE(source, immutable_pre_step);
    std::vector<AMR::AdaptiveMark> refine;
    for (const auto& key : source->owned_order())
    {
        const auto center = source->hierarchy()->geometry(source->hierarchy()->owned().at(key))->centroid();
        if (std::abs(center[0] - 1.5) < 1e-12 && std::abs(center[1] - 1.5) < 1e-12 && std::abs(center[2] - 1.5) < 1e-12)
            refine.push_back({key, AMR::AdaptiveSplit::Isotropic});
    }
    const auto count = [](const GeneralVoFSolver& flow)
    {
        double local = double(flow.mesh()->num_owned_cells()), global = 0;
        Teuchos::reduceAll(*flow.mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
        return global;
    };
    const double old_cells = count(solver);
    EXPECT_NO_THROW(solver.regrid(refine));
    EXPECT_GT(count(solver), old_cells);
    EXPECT_NE(solver.phases(), source);
    EXPECT_EQ(solver.time(), before.time);
    EXPECT_NEAR(solver.diagnostics().liquid_mass, before.liquid_mass, 2e-11);
    for (unsigned a = 0; a < 3; ++a)
        EXPECT_NEAR(solver.diagnostics().momentum[a], before.momentum[a], 2e-11);
    const auto remapped = solver.phases();
    const auto after_regrid = solver.diagnostics();
    std::vector<int> destination(remapped->owned_order().size(),
        (Tpetra::getDefaultComm()->getRank() + 1) % Tpetra::getDefaultComm()->getSize());
    EXPECT_NO_THROW(solver.repartition(destination));
    EXPECT_EQ(solver.time(), before.time);
    EXPECT_NEAR(solver.diagnostics().gas_mass, after_regrid.gas_mass, 2e-11);
    EXPECT_NEAR(solver.diagnostics().liquid_mass, after_regrid.liquid_mass, 2e-11);
    for (unsigned a = 0; a < 3; ++a)
        EXPECT_NEAR(solver.diagnostics().momentum[a], after_regrid.momentum[a], 2e-11);
    // Retained sources from both accepted epochs remain valid and unchanged.
    EXPECT_NE(immutable_pre_step, source);
    EXPECT_NE(source, remapped);
    EXPECT_EQ(immutable_pre_step->gas_volumes(), std::vector<double>(immutable_pre_step->owned_order().size(), 0));
}
TEST(GeneralVoFSolverTest, PhysicalHydrostaticPressureBalancesSkewNativeCentersOverFourSteps)
{
    auto material = inviscid();
    material.gas_density = material.liquid_density = 1;
    GeneralVoFSolverOptions options;
    options.flow.gravity = {.3, -.2, .1};
    GeneralVoFSolver solver(phases(0, true), material, options);
    for (unsigned step = 0; step < 4; ++step)
        (void) solver.step(.01);
    using GO = DefaultTpetraTypes::global_ordinal_type;
    using LO = DefaultTpetraTypes::local_ordinal_type;
    GO local_gid = std::numeric_limits<GO>::max(), reference_gid = 0;
    for (std::size_t c = 0; c < solver.velocity().size(); ++c)
        local_gid = std::min(local_gid, solver.mesh()->cell_global_id(LO(c)));
    const auto comm = solver.mesh()->owned_cell_map()->getComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_gid, &reference_gid);
    double local[4]{}, reference[4]{};
    for (std::size_t c = 0; c < solver.velocity().size(); ++c)
        if (solver.mesh()->cell_global_id(LO(c)) == reference_gid)
        {
            const auto x = solver.mesh()->cell_centroid(LO(c));
            local[0] = solver.pressure()[c];
            local[1] = x.x;
            local[2] = x.y;
            local[3] = x.z;
        }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 4, local, reference);
    for (std::size_t c = 0; c < solver.velocity().size(); ++c)
    {
        const auto x = solver.mesh()->cell_centroid(LO(c));
        const double expected = .3 * (x.x - reference[1]) - .2 * (x.y - reference[2]) + .1 * (x.z - reference[3]);
        EXPECT_NEAR(solver.pressure()[c] - reference[0], expected, 2e-7);
        for (double value : solver.velocity()[c])
            EXPECT_NEAR(value, 0, 2e-8);
    }
    EXPECT_LT(solver.diagnostics().maximum_divergence, 1e-8);
}
} // namespace

namespace
{
TEST(GeneralVoFSolverTest, IndependentlyIntegratedReflectedCylinderRetainsFiniteDurationCapillaryBalance)
{
    auto material = inviscid();
    material.gas_density = material.liquid_density = 1;
    material.surface_tension = 1;
    GeneralVoFSolverOptions options;
    options.flow.pressure_relative_tolerance = 1e-12;
    GeneralVoFSolver solver(cylinder_sector_phases(), material, options);
    const auto initial = solver.diagnostics();
    constexpr double physical_end = .02, step_ceiling = .0025, stability_fraction = .9;
    const double initial_limit = solver.stable_timestep();
    ASSERT_TRUE(std::isfinite(initial_limit));
    ASSERT_GT(initial_limit, 0);
    const double initial_step = std::min(step_ceiling, stability_fraction * initial_limit);
    ASSERT_GT(initial_step, 0);
    const double nominal_steps = std::ceil(physical_end / initial_step);
    ASSERT_LT(nominal_steps, double(std::numeric_limits<std::size_t>::max() / 2));
    // Twice the initial-limit step count permits moderate subsequent variation
    // while detecting a collapsing timestep instead of shortening physical time.
    const auto maximum_steps = 2 * std::size_t(nominal_steps);
    std::size_t steps = 0;
    double peak = 0;
    while (solver.time() < physical_end)
    {
        ASSERT_LT(steps, maximum_steps) << "initial stability limit=" << initial_limit;
        const double before = solver.time(), limit = solver.stable_timestep();
        ASSERT_TRUE(std::isfinite(limit));
        ASSERT_GT(limit, 0);
        const double dt = std::min({step_ceiling, stability_fraction * limit, physical_end - before});
        ASSERT_TRUE(std::isfinite(dt));
        ASSERT_GT(dt, 0);
        ASSERT_GT(before + dt, before);
        SCOPED_TRACE(::testing::Message() << "time=" << before << ", dt=" << dt << ", stable limit=" << limit);
        VoFStepDiagnostics after;
        ASSERT_NO_THROW(after = solver.step(dt));
        ++steps;
        peak = std::max(peak, after.maximum_velocity);
        EXPECT_LT(after.maximum_divergence, 1e-8);
        EXPECT_LT(std::abs(after.gas_mass - initial.gas_mass) / initial.gas_mass, 2e-11);
        EXPECT_LT(std::abs(after.liquid_mass - initial.liquid_mass) / initial.liquid_mass, 2e-11);
    }
    double local[4] = {}, global[4] = {};
    for (std::size_t c = 0; c < solver.mesh()->num_owned_cells(); ++c)
    {
        const auto x = solver.mesh()->cell_centroid(DefaultTpetraTypes::local_ordinal_type(c));
        const double radial = std::hypot(x.x, x.y),
                     volume = solver.mesh()->cell_volume(DefaultTpetraTypes::local_ordinal_type(c));
        if (radial < .95)
        {
            local[0] += volume * solver.pressure()[c];
            local[1] += volume;
        }
        if (radial > 1.05)
        {
            local[2] += volume * solver.pressure()[c];
            local[3] += volume;
        }
    }
    Teuchos::reduceAll(*solver.mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 4, local, global);
    ASSERT_GT(global[1], 0);
    ASSERT_GT(global[3], 0);
    const double jump = global[0] / global[1] - global[2] / global[3];
    EXPECT_LT(std::abs(jump - 1), .08);
    EXPECT_LT(peak, .05);
    EXPECT_DOUBLE_EQ(solver.time(), physical_end);
}
} // namespace

namespace
{
TEST(GeneralVoFSolverTest, ClosedSymmetryViscousModeRetainsAnalyticFiniteDurationRate)
{
    auto material = inviscid();
    material.gas_density = material.liquid_density = 1;
    material.gas_viscosity = material.liquid_viscosity = .01;
    GeneralVoFSolver solver(pure_cartesian_phases(8, 8, 2), material);
    const auto comm = solver.mesh()->owned_cell_map()->getComm();
    const int local_cells = static_cast<int>(solver.mesh()->num_owned_cells());
    int global_cells = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_cells, &global_cells);
    ASSERT_EQ(global_cells, 128);
    constexpr double amplitude = 1e-6;
    std::vector<VoF::Vector> velocity(solver.mesh()->num_owned_cells());
    for (std::size_t c = 0; c < velocity.size(); ++c)
    {
        const auto x = solver.mesh()->cell_centroid(DefaultTpetraTypes::local_ordinal_type(c));
        velocity[c] = {amplitude * std::sin(std::numbers::pi * x.x) * std::cos(std::numbers::pi * x.y),
            -amplitude * std::cos(std::numbers::pi * x.x) * std::sin(std::numbers::pi * x.y), 0};
    }
    solver.initialize(velocity);
    const auto mode_amplitude = [&]
    {
        double local[2] = {}, global[2] = {};
        for (std::size_t c = 0; c < velocity.size(); ++c)
        {
            const auto x = solver.mesh()->cell_centroid(DefaultTpetraTypes::local_ordinal_type(c));
            const double shape = std::sin(std::numbers::pi * x.x) * std::cos(std::numbers::pi * x.y);
            const double volume = solver.mesh()->cell_volume(DefaultTpetraTypes::local_ordinal_type(c));
            local[0] += volume * solver.velocity()[c][0] * shape;
            local[1] += volume * shape * shape;
        }
        Teuchos::reduceAll(*solver.mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 2, local, global);
        return global[0] / global[1];
    };
    const auto before = solver.diagnostics();
    const double initial = mode_amplitude();
    for (unsigned step = 0; step < 10; ++step)
    {
        const auto after = solver.step(.002);
        EXPECT_LT(after.maximum_divergence, 1e-8);
    }
    const double final = mode_amplitude();
    EXPECT_GT(final, 0);
    EXPECT_LT(final, initial);
    const double rate = -std::log(final / initial) / solver.time();
    EXPECT_NEAR(rate / (2 * std::numbers::pi * std::numbers::pi * .01), 1, .05);
    EXPECT_LT(solver.diagnostics().kinetic_energy, before.kinetic_energy);
    for (unsigned a = 0; a < 3; ++a)
        EXPECT_NEAR(solver.diagnostics().momentum[a], before.momentum[a], 2e-11);
}
} // namespace

namespace
{
TEST(GeneralVoFSolverTest, MicroscopicDeclaredTraceAggregatesPhaseMassBeforeDiagnosticNarrowing)
{
    const auto base = phases(0);
    const double eta = std::numeric_limits<double>::denorm_min();
    std::vector<double> gas(base->owned_order().size(), eta), carrier = base->carrier_volumes();
    std::vector<VoF::GeneralVoFPhaseGeometry> geometry(gas.size(), VoF::GeneralVoFPhaseGeometry::PureCarrier);
    auto source = std::make_shared<const VoF::GeneralVoF>(
        base->hierarchy(), base->owned_order(), gas, carrier, base->options(), geometry);
    auto material = inviscid();
    material.gas_density = .5;
    material.liquid_density = 1;
    GeneralVoFSolver solver(source, material);
    // The accepted phase eta is representable. Its weighted mass eta/2 is a
    // transient product; aggregate before narrowing rather than dropping it.
    const auto expected = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::from_value(eta)
                              .scaled(13.5)
                              .certified()
                              .as_double();
    EXPECT_EQ(solver.diagnostics().gas_mass, expected);
    const auto before = solver.phases();
    const auto after = solver.step(.001);
    EXPECT_EQ(after.gas_mass, expected);
    EXPECT_EQ(after.maximum_velocity, 0);
    for (double value : solver.phases()->gas_volumes())
        EXPECT_EQ(value, eta);
    EXPECT_NE(solver.phases(), before);
}
} // namespace

namespace
{
TEST(GeneralVoFSolverTest, NonzeroSkewCoarseFinePhasePassagePreservesIndependentInventories)
{
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    reset_observed_geometry_statistics();
    const auto record = [&](const std::string& name, auto value)
    {
        std::ostringstream text;
        text << std::setprecision(20) << value;
        RecordProperty(name, text.str());
        if (std::getenv("SIMPLEFLUID_VOF_TEST_PROGRESS") && Tpetra::getDefaultComm()->getRank() == 0)
            std::cerr << name << '=' << text.str() << std::endl;
    };
    const auto* exact_setting = std::getenv("SIMPLEFLUID_VOF_EXACT_STATS");
    const bool exact_enabled = exact_setting && std::string(exact_setting) == "1";
    struct ExactStatisticsScope
    {
        bool enabled;
        ~ExactStatisticsScope()
        {
            if (enabled)
                enable_observed_exact_geometry_statistics(false);
        }
    } exact_scope{exact_enabled};
    if (exact_enabled)
    {
        enable_observed_exact_geometry_statistics(true);
        reset_observed_geometry_statistics();
    }
    record("exact_statistics_enabled", exact_enabled);
    std::set<const void*> exact_domains, bounded_domains;
    for (const auto& image : geometry_image_snapshots())
    {
        exact_domains.insert(image.exact_domain);
        bounded_domains.insert(image.bounded_domain);
    }
    record("observed_exact_counter_domains", exact_domains.size());
    record("observed_bounded_counter_domains", bounded_domains.size());
    const auto record_exact = [&](const std::string& stage)
    {
        if (!exact_enabled)
            return;
        const auto stats = observed_geometry_statistics().exact;
        record("exact_api_" + stage + "_source_coordinate_exports", stats.source_coordinate_exports);
        record("exact_api_" + stage + "_source_support_exports", stats.source_support_exports);
        record("exact_api_" + stage + "_volume_evaluations", stats.volume_evaluations);
        record("exact_api_" + stage + "_moment_evaluations", stats.moment_evaluations);
        record("exact_api_" + stage + "_clip_constructions", stats.clip_constructions);
        record("exact_api_" + stage + "_intersection_constructions", stats.intersection_constructions);
        record("exact_api_" + stage + "_face_measure_evaluations", stats.face_measure_evaluations);
        record("exact_api_" + stage + "_source_membership_predicates", stats.source_membership_predicates);
        record("exact_api_" + stage + "_measure_values", stats.measure_values);
        record("exact_api_" + stage + "_measure_arithmetic", stats.measure_arithmetic);
        record("exact_api_" + stage + "_cover_certifications", stats.cover_certifications);
        record("exact_api_" + stage + "_maximum_result_numerator_bits", stats.maximum_result_numerator_bits);
        record("exact_api_" + stage + "_maximum_result_denominator_bits", stats.maximum_result_denominator_bits);
    };
    record_exact("before_construction");
    auto material = inviscid();
    material.gas_density = .1;
    material.liquid_density = 1;
    const auto fixture_started = Clock::now();
    const auto initial_phases = skew_coarse_fine_plane(true);
    const auto fixture_finished = Clock::now();
    GeneralVoFSolver solver(initial_phases, material);
    const auto constructor_finished = Clock::now();
    record("fixture_seconds", std::chrono::duration<double>(fixture_finished - fixture_started).count());
    record("solver_constructor_seconds",
        std::chrono::duration<double>(constructor_finished - fixture_finished).count());
    record("phase_initialization", "ordinary_fp");
    record("arithmetic_mode", "Ordinary");
    record_exact("after_construction");
    double local_structure[2] = {double(solver.mesh()->num_owned_cells()), 0}, structure[2]{};
    for (std::size_t f = 0; f < solver.mesh()->num_owned_faces(); ++f)
    {
        const auto face = DefaultTpetraTypes::local_ordinal_type(f);
        if (solver.mesh()->is_geometry_exterior_face(face))
            continue;
        const double left = solver.mesh()->cell_volume(solver.mesh()->owner_cell(face)),
                     right = solver.mesh()->cell_volume(solver.mesh()->neighbor_cell(face));
        if (std::max(left, right) > 1.5 * std::min(left, right))
            local_structure[1] += 1;
    }
    Teuchos::reduceAll(*solver.mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 2, local_structure, structure);
    // The provider also introduces conforming transition cells. Check its
    // actual refined native topology, rather than assuming only marked splits.
    record("global_native_cells", structure[0]);
    record("global_coarse_fine_face_incidences", structure[1]);
    EXPECT_DOUBLE_EQ(structure[0], 50);
    EXPECT_GT(structure[0], 16);
    EXPECT_LE(structure[0], 64);
    EXPECT_GT(structure[1], 0);
    std::vector<VoF::Vector> velocity(solver.mesh()->num_owned_cells());
    for (std::size_t c = 0; c < velocity.size(); ++c)
    {
        const auto x = solver.mesh()->cell_centroid(DefaultTpetraTypes::local_ordinal_type(c));
        const double xi = x.x - .2 * x.y, eta = x.y;
        const double vx = .05 * std::pow(std::sin(std::numbers::pi * xi), 2) * std::sin(2 * std::numbers::pi * eta);
        const double vy = -.05 * std::pow(std::sin(std::numbers::pi * eta), 2) * std::sin(2 * std::numbers::pi * xi);
        velocity[c] = {vx + .2 * vy, vy, 0};
    }
    record_exact("before_initialize");
    const auto explicit_initialize_started = Clock::now();
    solver.initialize(velocity);
    record_exact("after_initialize");
    const auto initialized = Clock::now();
    record("explicit_initialize_seconds",
        std::chrono::duration<double>(initialized - explicit_initialize_started).count());
    // Historical property includes fixture, constructor and explicit initialize.
    record("initialization_seconds", std::chrono::duration<double>(initialized - started).count());
    const auto region_gas = [&]
    {
        double local = 0, global = 0;
        for (std::size_t c = 0; c < solver.mesh()->num_owned_cells(); ++c)
        {
            const auto x = solver.mesh()->cell_centroid(DefaultTpetraTypes::local_ordinal_type(c));
            if (x.x - .2 * x.y > .5)
                local += solver.phases()->gas_volumes()[c];
        }
        Teuchos::reduceAll(*solver.mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
        return global;
    };
    const auto before = solver.diagnostics();
    const double initial_region = region_gas();
    double gas_drift = 0, liquid_drift = 0, momentum_residual = 0, maximum_divergence = 0;
    double gas_minimum = std::numeric_limits<double>::infinity(), carrier_minimum = gas_minimum;
    double fraction_minimum = 1, fraction_maximum = 0;
    unsigned accepted_steps = 0;
    std::array<unsigned long long, 15> owner_request_counts{};
    double owner_reconstruction_seconds = 0;
    record_exact("before_loop");
    for (unsigned step = 0; step < 10; ++step)
    {
        const auto previous_exact = observed_geometry_statistics().exact;
        const auto step_started = Clock::now();
        const auto after = solver.step(.002);
        record_exact("step_" + std::to_string(step));
        if (exact_enabled)
        {
            const auto measured = observed_geometry_statistics().exact;
            EXPECT_EQ(measured.volume_evaluations, previous_exact.volume_evaluations);
            EXPECT_EQ(measured.moment_evaluations, previous_exact.moment_evaluations);
            EXPECT_EQ(measured.clip_constructions, previous_exact.clip_constructions);
            EXPECT_EQ(measured.intersection_constructions, previous_exact.intersection_constructions);
            EXPECT_EQ(measured.face_measure_evaluations, previous_exact.face_measure_evaluations);
            EXPECT_EQ(measured.measure_values, previous_exact.measure_values);
            EXPECT_EQ(measured.measure_arithmetic, previous_exact.measure_arithmetic);
            EXPECT_EQ(measured.cover_certifications, previous_exact.cover_certifications);
        }
        record("step_" + std::to_string(step) + "_seconds",
            std::chrono::duration<double>(Clock::now() - step_started).count());
        const auto& prepared = solver.prepared_transport();
        EXPECT_TRUE(prepared);
        if (prepared)
        {
            const auto& requests = prepared->diagnostics.owner_plane_requests;
            EXPECT_EQ(requests.owner_reconstruction_calls,
                requests.mixed_cell_fits + requests.retained_interface_returns + requests.pure_declaration_returns +
                    requests.zero_phase_offset_returns);
            EXPECT_EQ(requests.owner_reconstruction_calls + requests.local_result_hits + requests.peer_cache_hits,
                requests.local_owned_requests + requests.incoming_peer_requests);
            if (requests.cache_admission_denials == 0)
                EXPECT_EQ(requests.owner_reconstruction_calls,
                    requests.local_owned_requests + requests.peer_cache_admissions);
            EXPECT_LE(requests.workspace_peak_bytes, initial_phases->options().byte_budget);
            const std::array<unsigned long long, 15> counts{requests.local_owned_requests,
                requests.incoming_peer_requests, requests.local_result_hits, requests.peer_cache_hits,
                requests.peer_cache_admissions, requests.cache_admission_denials, requests.owner_reconstruction_calls,
                requests.mixed_cell_fits, requests.retained_interface_returns, requests.pure_declaration_returns,
                requests.zero_phase_offset_returns, requests.geometry_volume_evaluations,
                requests.geometry_intersections, requests.peer_cache_peak_bytes, requests.workspace_peak_bytes};
            for (std::size_t i = 0; i < 13; ++i)
                owner_request_counts[i] += counts[i];
            for (std::size_t i = 13; i < counts.size(); ++i)
                owner_request_counts[i] = std::max(owner_request_counts[i], counts[i]);
            owner_reconstruction_seconds += requests.reconstruction_seconds;
        }
        gas_drift = std::max(gas_drift, std::abs(after.gas_mass - before.gas_mass) / before.gas_mass);
        liquid_drift = std::max(liquid_drift, std::abs(after.liquid_mass - before.liquid_mass) / before.liquid_mass);
        maximum_divergence = std::max(maximum_divergence, after.maximum_divergence);
        EXPECT_LT(std::abs(after.gas_mass - before.gas_mass) / before.gas_mass, 2e-11);
        EXPECT_LT(std::abs(after.liquid_mass - before.liquid_mass) / before.liquid_mass, 2e-11);
        for (const auto& ledger : solver.momentum_transport_diagnostics())
        {
            momentum_residual = std::max(momentum_residual, ledger.physical_relative_residual);
            EXPECT_LE(ledger.physical_relative_residual, 1e-12);
            EXPECT_LE(ledger.physical_relative_residual + ledger.cleanup_identity_relative_residual, 1e-12);
        }
        EXPECT_LT(after.maximum_divergence, 1e-8);
        for (std::size_t c = 0; c < solver.phases()->gas_volumes().size(); ++c)
        {
            gas_minimum = std::min(gas_minimum, solver.phases()->gas_volumes()[c]);
            carrier_minimum = std::min(carrier_minimum, solver.phases()->carrier_volumes()[c]);
            fraction_minimum = std::min(fraction_minimum, solver.gas_fraction()[c]);
            fraction_maximum = std::max(fraction_maximum, solver.gas_fraction()[c]);
            EXPECT_GE(solver.phases()->gas_volumes()[c], 0);
            EXPECT_GE(solver.phases()->carrier_volumes()[c], 0);
            EXPECT_GE(solver.gas_fraction()[c], 0);
            EXPECT_LE(solver.gas_fraction()[c], 1);
        }
        ++accepted_steps;
        record("accepted_steps", accepted_steps);
        record("accepted_time", solver.time());
        const auto& cleanup = solver.phases()->phase_cleanup_state();
        record("cleanup_cells", cleanup.cleaned_cells);
        record("cleanup_signed_gas_volume", cleanup.signed_volume[0]);
        record("cleanup_signed_carrier_volume", cleanup.signed_volume[1]);
        record("cleanup_absolute_gas_volume", cleanup.absolute_volume[0]);
        record("cleanup_absolute_carrier_volume", cleanup.absolute_volume[1]);
        record("cleanup_largest_gas_volume", cleanup.largest_absolute_volume[0]);
        record("cleanup_largest_carrier_volume", cleanup.largest_absolute_volume[1]);
        record("cleanup_relative_cutoff", initial_phases->options().phase_cleanup.relative_cutoff);
        record("cleanup_gas_budget", initial_phases->options().phase_cleanup.maximum_absolute_volume[0]);
        record("cleanup_carrier_budget", initial_phases->options().phase_cleanup.maximum_absolute_volume[1]);
        for (unsigned axis = 0; axis < 3; ++axis)
            record("cleanup_momentum_" + std::to_string(axis),
                solver.momentum_transport_diagnostics()[axis].accumulated_cleanup_momentum);
        for (unsigned phase = 0; phase < 2; ++phase)
            EXPECT_LE(
                cleanup.absolute_volume[phase], initial_phases->options().phase_cleanup.maximum_absolute_volume[phase]);
    }
    // The region's only open internal boundary is the actual refined stripe
    // interface xi=.5; external walls are impermeable. The initial analytic
    // velocity carries gas into this region, at O(1e−4) volume over .02s.
    const auto passage = region_gas() - initial_region;
    EXPECT_GT(passage, 1e-5);
    EXPECT_DOUBLE_EQ(solver.time(), .02);
    const double local_minimum[3] = {gas_minimum, carrier_minimum, fraction_minimum};
    double global_minimum[3]{};
    Teuchos::reduceAll(
        *solver.mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 3, local_minimum, global_minimum);
    record("maximum_gas_mass_relative_drift", gas_drift);
    record("maximum_liquid_mass_relative_drift", liquid_drift);
    record("maximum_original_momentum_relative_residual", momentum_residual);
    record("maximum_physical_divergence", maximum_divergence);
    record("minimum_gas_volume", global_minimum[0]);
    record("minimum_carrier_volume", global_minimum[1]);
    record("minimum_gas_fraction", global_minimum[2]);
    double global_fraction_maximum = 0;
    Teuchos::reduceAll(*solver.mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &fraction_maximum,
        &global_fraction_maximum);
    record("maximum_gas_fraction", global_fraction_maximum);
    record("regional_gas_passage", passage);
    record("complete_case_seconds", std::chrono::duration<double>(Clock::now() - started).count());
    // Gather rank-local work only after the timed physical horizon. Work counters
    // may decrease with reuse; physical and cleanup receipts remain the parity oracle.
    const auto comm = Tpetra::getDefaultComm();
    std::vector<unsigned long long> all_owner_counts(owner_request_counts.size() * comm->getSize());
    std::vector<double> all_owner_seconds(comm->getSize());
    Teuchos::gatherAll(*comm, int(owner_request_counts.size()), owner_request_counts.data(),
        int(all_owner_counts.size()), all_owner_counts.data());
    Teuchos::gatherAll(
        *comm, 1, &owner_reconstruction_seconds, int(all_owner_seconds.size()), all_owner_seconds.data());
    constexpr std::array<const char*, 15> owner_count_names{"local_owned_requests", "incoming_peer_requests",
        "local_result_hits", "peer_cache_hits", "peer_cache_admissions", "cache_admission_denials",
        "owner_reconstruction_calls", "mixed_cell_fits", "retained_interface_returns", "pure_declaration_returns",
        "zero_phase_offset_returns", "geometry_volume_evaluations", "geometry_intersections", "peer_cache_peak_bytes",
        "workspace_peak_bytes"};
    for (int rank = 0; rank < comm->getSize(); ++rank)
    {
        const auto prefix = "owner_requests_rank_" + std::to_string(rank) + "_";
        for (std::size_t i = 0; i < owner_count_names.size(); ++i)
            record(prefix + owner_count_names[i], all_owner_counts[rank * owner_count_names.size() + i]);
        record(prefix + "reconstruction_seconds", all_owner_seconds[rank]);
    }
    const auto operations = observed_geometry_statistics().bounded;
    // These are bounded-kernel counters. A zero here alone is not a global
    // proof that other infrastructure performed no rational operation.
    record("bounded_kernel_source_imports", operations.source_imports);
    record("bounded_kernel_constructions", operations.constructions);
    record("bounded_kernel_intersections", operations.intersections);
    record("bounded_kernel_volume_evaluations", operations.volume_evaluations);
    record("bounded_kernel_moment_evaluations", operations.moment_evaluations);
    record("bounded_kernel_partition_checks", operations.partition_checks);
    record("bounded_kernel_structural_partition_reuses", operations.structural_partition_reuses);
    record("bounded_kernel_floating_operations", operations.floating_operations);
    record("bounded_kernel_interval_operations", operations.interval_operations);
    record("bounded_kernel_ledger_operations", operations.ledger_operations);
    record("bounded_kernel_wider_routes", operations.wider_routes);
    record("bounded_kernel_rejections", operations.rejections);
    record("bounded_kernel_exact_predicates", operations.exact_predicates);
    record("bounded_kernel_reported_rational_constructions", operations.rational_constructions);
    record("bounded_kernel_reported_rational_measures", operations.rational_measures);
    record("bounded_kernel_reported_rational_ledgers", operations.rational_ledgers);
}
} // namespace

namespace
{
TEST(GeneralVoFSolverTest, DominantGasOccupancyDefectUsesSharedBoundedGeometricView)
{
    const auto base = phases(0);
    std::vector<double> gas = base->carrier_volumes(), carrier(gas.size(), std::numeric_limits<double>::denorm_min());
    for (auto& amount : gas)
        amount = std::nextafter(amount, std::numeric_limits<double>::infinity());
    std::vector<VoF::GeneralVoFPhaseGeometry> geometry(gas.size(), VoF::GeneralVoFPhaseGeometry::PureGas);
    auto source = std::make_shared<const VoF::GeneralVoF>(
        base->hierarchy(), base->owned_order(), gas, carrier, base->options(), geometry);
    auto material = inviscid();
    material.gas_density = material.liquid_density = 1;
    GeneralVoFSolver solver(source, material);
    for (std::size_t c = 0; c < gas.size(); ++c)
    {
        EXPECT_GT(
            solver.phases()->gas_volumes()[c], solver.mesh()->cell_volume(DefaultTpetraTypes::local_ordinal_type(c)));
        EXPECT_EQ(solver.gas_fraction()[c], 1);
    }
    (void) solver.step(.001);
    EXPECT_EQ(solver.phases()->gas_volumes(), gas);
    EXPECT_EQ(solver.phases()->carrier_volumes(), carrier);
    for (double fraction : solver.gas_fraction())
        EXPECT_EQ(fraction, 1);
}
} // namespace

namespace
{
TEST(GeneralVoFSolverTest, TinyCellRepartitionRetainsIndependentZeroRatesAndSubnormalAggregateMomentum)
{
    const double length = std::ldexp(1.0, -358);
    const double eta = std::numeric_limits<double>::denorm_min();
    AMR::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.extent = {2 * length, 2 * length, 2 * length};
    auto provider = std::make_shared<AMR::CartesianAMR>(grid);
    const auto comm = Tpetra::getDefaultComm();
    const int owning_ranks = std::min(2, comm->getSize());
    std::vector<std::uint64_t> owned;
    for (std::uint64_t c = 0; c < 8; ++c)
        if (int(c % owning_ranks) == comm->getRank())
            owned.push_back(c);
    auto bootstrap = provider->detach_owned(owned, {'t', 'i', 'n', 'y', 'm', 'o', 'm', 'e', 'n', 't', 'u', 'm'});
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    std::vector<AMR::AdaptiveCellKey> order;
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        const auto exact_volume =
            SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::volume(*hierarchy->geometry(leaf));
        ASSERT_EQ(exact_volume.compare(
                      SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::from_value(eta)),
            0);
        order.push_back(key);
        gas.push_back(0);
        carrier.push_back(eta);
    }
    auto phase =
        std::make_shared<const VoF::GeneralVoF>(hierarchy, std::move(order), std::move(gas), std::move(carrier));
    auto material = inviscid();
    material.gas_density = material.liquid_density = 1;
    GeneralVoFSolver solver(phase, material);
    std::vector<VoF::Vector> velocity(solver.mesh()->num_owned_cells(), {.5, 0, 0});
    std::vector<double> independent_faces(solver.mesh()->num_faces(), 0);
    ASSERT_NO_THROW(solver.initialize(velocity, independent_faces));
    const auto check_owners = [&]
    {
        const int local_present = solver.mesh()->num_owned_cells() > 0;
        const int local_empty = !local_present;
        int present = 0, empty = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_present, &present);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
        EXPECT_EQ(present, owning_ranks);
        EXPECT_EQ(empty, comm->getSize() - owning_ranks);
    };
    check_owners();
    const auto before = solver.diagnostics();
    EXPECT_EQ(before.momentum[0], 4 * eta);
    EXPECT_EQ(before.kinetic_energy, eta);
    const auto source = solver.phases();
    const auto source_finite = source->phase_volumes();
    const auto source_pressure = solver.pressure();
    std::vector<int> destination(source->owned_order().size(), (comm->getRank() + 1) % comm->getSize());
    ASSERT_NO_THROW(solver.repartition(destination));
    check_owners();
    EXPECT_EQ(solver.time(), before.time);
    EXPECT_EQ(solver.diagnostics().momentum, before.momentum);
    EXPECT_EQ(solver.diagnostics().liquid_mass, before.liquid_mass);
    EXPECT_EQ(solver.diagnostics().kinetic_energy, before.kinetic_energy);
    for (const auto& u : solver.velocity())
    {
        EXPECT_EQ(u[0], .5);
        EXPECT_EQ(u[1], 0);
        EXPECT_EQ(u[2], 0);
    }
    for (double value : solver.face_velocity())
        EXPECT_EQ(value, 0);
    for (double value : solver.pressure())
        EXPECT_EQ(value, 0);
    EXPECT_EQ(solver.diagnostics().maximum_divergence, 0);
    for (const auto& p : solver.phases()->phase_volumes())
    {
        EXPECT_EQ(p.gas, 0);
        EXPECT_EQ(p.carrier, eta);
    }
    // Immutable accepted sources survive a later ownership publication.
    ASSERT_EQ(source->phase_volumes().size(), source_finite.size());
    for (std::size_t c = 0; c < source_finite.size(); ++c)
    {
        EXPECT_EQ(source->phase_volumes()[c].gas, source_finite[c].gas);
        EXPECT_EQ(source->phase_volumes()[c].carrier, source_finite[c].carrier);
    }
    EXPECT_EQ(source_pressure, std::vector<double>(source_finite.size(), 0));
}
} // namespace
