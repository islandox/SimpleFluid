/** @file testGeneralVoFFaceCapillary.cc
 * @brief Explicit algebraic-backend curved capillary qualification prerequisites.
 */
#include "general_vof_solver_test_helpers.hh"
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using Backend = VoF::GeneralVoFTransportBackend;
class GeneralVoFFaceCapillary : public testing::TestWithParam<Backend>
{
};
TEST_P(GeneralVoFFaceCapillary, SpatialPartitionRetainsCompleteCurvedSupport)
{
    QualificationPhaseReference raw;
    const auto source = cylinder_sector_phases(true, GetParam(), qualification_phase_cleanup_enabled(), &raw);
    const auto comm = Tpetra::getDefaultComm();
    const int active = std::min(3, comm->getSize());
    int local_empty = source->owned_order().empty(), empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, comm->getSize() - active);
    // Three axial layers and two halo rings contain the entire curved component
    // on every active owner. This does not claim support for larger components.
    if (!local_empty)
        EXPECT_EQ(source->support_cells().size(), 36u);
    for (const auto& [key, leaf] : source->hierarchy()->owned())
    {
        const auto layer = std::get<AMR::ProductCellAncestry>(key.ancestry).axial.root;
        EXPECT_EQ(int(layer * active / 3), comm->getRank());
    }
    auto material = inviscid();
    material.gas_density = material.liquid_density = material.surface_tension = 1;
    const auto catalog = source->capillary_surface();
    const auto identity = source->accepted_identity();
    const auto gas = source->gas_volumes(), liquid = source->carrier_volumes();
    GeneralVoFSolver solver(source, material);
    record_qualification_cleanup(solver);
    std::map<VoF::GeneralVoFSurfaceKey, double> area;
    for (const auto& patch : catalog.patches)
        area.emplace(patch.key, patch.area);
    double local[3]{}, global[3]{};
    for (const auto& fit : solver.capillary_diagnostics().fits)
    {
        local[0] += area.at(fit.key);
        local[1] += area.at(fit.key) * std::abs(fit.curvature - 1);
        local[2] += 1;
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 3, local, global);
    ASSERT_GT(global[0], 0);
    EXPECT_EQ(global[2], 12);
    EXPECT_LE(global[1] / global[0], .05);
    EXPECT_EQ(source->accepted_identity(), identity);
    EXPECT_EQ(source->gas_volumes(), gas);
    EXPECT_EQ(source->carrier_volumes(), liquid);
    EXPECT_EQ(solver.time(), 0);
    RecordProperty("configured_backend", GetParam() == Backend::DonorCell ? "DonorCell" : "GeometricFace");
    RecordProperty("actual_prepared_family", "unpublished_initialization_only");
    RecordProperty("ranks", comm->getSize());
    RecordProperty("asserted_empty_owners", empty);
    RecordProperty("owned_curved_patches", global[2]);
    RecordProperty("area_weighted_curvature_relative_error", global[1] / global[0]);
}
TEST_P(GeneralVoFFaceCapillary, NativeCylinderRetainsCompletePhysicalEndpoint)
{
    if (!std::getenv("SIMPLEFLUID_RUN_ALGEBRAIC_CAPILLARY"))
        GTEST_SKIP() << "Opt-in Q4 physical qualification; see verification/vof/algebraic_capillary_manifest.json";
    auto material = inviscid();
    material.gas_density = material.liquid_density = 1;
    material.surface_tension = 1;
    GeneralVoFSolverOptions options;
    options.flow.pressure_relative_tolerance = 1e-12;
    QualificationPhaseReference raw;
    const auto source = cylinder_sector_phases(false, GetParam(), qualification_phase_cleanup_enabled(), &raw);
    GeneralVoFSolver solver(source, material, options);
    record_qualification_cleanup(solver, "initial_");
    RecordProperty("configured_backend", GetParam() == Backend::DonorCell ? "DonorCell" : "GeometricFace");
    RecordProperty("ranks", Tpetra::getDefaultComm()->getSize());
    // R=1, rho=1 and sigma=1 give kappa=1, pressure jump=1,
    // capillary velocity=1 and capillary time=1 in this cylindrical geometry.
    // This unchanged .02 s prerequisite is not the separate five-time gate.
    const auto catalog = source->capillary_surface(options.capillary);
    std::map<VoF::GeneralVoFSurfaceKey, double> areas;
    for (const auto& patch : catalog.patches)
        areas.emplace(patch.key, patch.area);
    double local_curvature[2]{}, global_curvature[2]{};
    for (const auto& fit : solver.capillary_diagnostics().fits)
    {
        const auto area = areas.at(fit.key);
        local_curvature[0] += area * std::abs(fit.curvature - 1);
        local_curvature[1] += area;
    }
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 2, local_curvature, global_curvature);
    ASSERT_GT(global_curvature[1], 0);
    const auto curvature_error = global_curvature[0] / global_curvature[1];
    RecordProperty("area_weighted_curvature_relative_error", curvature_error);
    EXPECT_LE(curvature_error, .05);
    // Independent polar-integral input is the physical reference, before cleanup.
    const auto initial = raw.physical_reference(solver.diagnostics(), material);
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
        RecordProperty("accepted_steps", int(steps));
        RecordProperty("completed_time", solver.time());
        const auto prepared = solver.prepared_transport();
        ASSERT_TRUE(prepared);
        EXPECT_EQ(prepared->backend, GetParam());
        EXPECT_EQ(prepared->family, VoF::GeneralVoFTransportFamily::AlgebraicFaceVolume);
        RecordProperty("actual_prepared_family", "AlgebraicFaceVolume");
        EXPECT_EQ(prepared->arithmetic_mode, ArithmeticMode::Ordinary);
        record_qualification_cleanup(solver, "step_" + std::to_string(steps) + "_");
        qualification_cleanup_metric(
            "raw_gas_mass_relative_drift", std::abs(after.gas_mass - initial.gas_mass) / initial.gas_mass);
        qualification_cleanup_metric(
            "raw_liquid_mass_relative_drift", std::abs(after.liquid_mass - initial.liquid_mass) / initial.liquid_mass);
        for (const auto& momentum : solver.momentum_transport_diagnostics())
            EXPECT_LE(momentum.physical_relative_residual, 1e-12);
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
    RecordProperty("pressure_jump", jump);
    RecordProperty("peak_velocity", peak);
    EXPECT_LT(std::abs(jump - 1), .08);
    EXPECT_LT(peak, .05);
    EXPECT_DOUBLE_EQ(solver.time(), physical_end);
}
INSTANTIATE_TEST_SUITE_P(AlgebraicCapillary, GeneralVoFFaceCapillary,
    testing::Values(Backend::DonorCell, Backend::GeometricFace), [](const testing::TestParamInfo<Backend>& parameter)
    { return parameter.param == Backend::DonorCell ? "DonorCell" : "GeometricFace"; });
} // namespace
