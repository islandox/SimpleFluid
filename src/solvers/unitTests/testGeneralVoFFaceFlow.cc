#include "general_vof_solver_test_helpers.hh"
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
std::shared_ptr<const VoF::GeneralVoF> face_backend(
    std::shared_ptr<const VoF::GeneralVoF> source, bool pure_gas = false)
{
    VoF::GeneralVoFOptions options = source->options();
    options.transport_backend = VoF::GeneralVoFTransportBackend::GeometricFace;
    auto gas = source->gas_volumes(), liquid = source->carrier_volumes();
    if (pure_gas)
        std::swap(gas, liquid);
    return std::make_shared<const VoF::GeneralVoF>(source->hierarchy(), source->owned_order(), gas, liquid, options);
}
TEST(GeneralVoFFaceFlow, MovingPurePhasesReuseMomentumWithoutInterfaceWork)
{
    for (bool gas : {false, true})
    {
        GeneralVoFSolver solver(face_backend(phases(0, false, true), gas), inviscid());
        std::vector<double> face_velocity(solver.mesh()->num_faces());
        for (const auto& tile : solver.phases()->tiles())
        {
            const auto f = solver.mesh()->overlap_face_map()->getLocalElement(tile.face_gid);
            const auto c = tile.geometry->centroid(), n = tile.geometry->area_vector();
            if (tile.exterior || f < 0)
                continue;
            if (c.x == 1 && n.x != 0)
                face_velocity[f] = (c.y < 1 ? .05 : -.05) * (n.x > 0 ? 1 : -1);
            if (c.y == 1 && n.y != 0)
                face_velocity[f] = (c.x > 1 ? .05 : -.05) * (n.y > 0 ? 1 : -1);
        }
        solver.initialize({}, face_velocity);
        const auto before = solver.diagnostics();
        EXPECT_GT(before.maximum_velocity, 0);
        for (unsigned step = 0; step < 3; ++step)
        {
            const auto after = solver.step(.001);
            EXPECT_NEAR(after.gas_mass, before.gas_mass, 1e-13);
            EXPECT_NEAR(after.liquid_mass, before.liquid_mass, 1e-13);
            const auto p = solver.prepared_transport();
            ASSERT_TRUE(p);
            ASSERT_TRUE(p->momentum_preparation);
            EXPECT_EQ(p->diagnostics.plane_measurements, 0u);
            EXPECT_LE(p->momentum_builds, 1u);
            EXPECT_GE(p->momentum_reuses, 2u);
            EXPECT_EQ(p->momentum_builds + p->momentum_reuses, 3u);
            for (const auto& m : solver.momentum_transport_diagnostics())
                EXPECT_LE(m.physical_relative_residual, 1e-12);
        }
        const auto accepted = solver.phases();
        const auto preparation = solver.prepared_transport();
        EXPECT_THROW((void) solver.step(1e8), std::runtime_error);
        EXPECT_EQ(solver.phases(), accepted);
        EXPECT_EQ(solver.prepared_transport(), preparation);
    }
}
TEST(GeneralVoFFaceFlow, PureLiquidHydrostaticPressureRetainsPhysicalBalance)
{
    auto material = inviscid();
    material.liquid_density = 1;
    GeneralVoFSolverOptions options;
    options.flow.gravity = {.3, -.2, .1};
    GeneralVoFSolver solver(face_backend(phases(0, true)), material, options);
    for (unsigned step = 0; step < 4; ++step)
        (void) solver.step(.01);
    double local[2]{}, global[2]{};
    for (std::size_t c = 0; c < solver.velocity().size(); ++c)
    {
        const auto x = solver.mesh()->cell_centroid(int(c));
        local[0] += solver.pressure()[c] - (.3 * x.x - .2 * x.y + .1 * x.z);
        local[1] += 1;
    }
    Teuchos::reduceAll(*solver.mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 2, local, global);
    for (std::size_t c = 0; c < solver.velocity().size(); ++c)
    {
        const auto x = solver.mesh()->cell_centroid(int(c));
        EXPECT_NEAR(solver.pressure()[c], .3 * x.x - .2 * x.y + .1 * x.z + global[0] / global[1], 2e-7);
        for (const auto u : solver.velocity()[c])
            EXPECT_NEAR(u, 0, 2e-8);
    }
    EXPECT_LT(solver.diagnostics().maximum_divergence, 1e-8);
}
TEST(GeneralVoFFaceFlow, HighDensityRatioSkewCoarseFinePassage)
{
    auto material = inviscid();
    material.gas_density = 1;
    material.liquid_density = 1000;
    GeneralVoFSolver solver(face_backend(skew_coarse_fine_plane()), material);
    std::vector<VoF::Vector> velocity(solver.mesh()->num_owned_cells());
    for (std::size_t c = 0; c < velocity.size(); ++c)
    {
        const auto x = solver.mesh()->cell_centroid(int(c));
        const double xi = x.x - .2 * x.y, eta = x.y;
        const double vx = .05 * std::pow(std::sin(std::numbers::pi * xi), 2) * std::sin(2 * std::numbers::pi * eta);
        const double vy = -.05 * std::pow(std::sin(std::numbers::pi * eta), 2) * std::sin(2 * std::numbers::pi * xi);
        velocity[c] = {vx + .2 * vy, vy, 0};
    }
    solver.initialize(velocity);
    const auto before = solver.diagnostics();
    for (unsigned step = 0; step < 10; ++step)
    {
        const auto after = solver.step(.002);
        EXPECT_LE(std::abs(after.gas_mass - before.gas_mass), 2e-11 * before.gas_mass);
        EXPECT_LE(std::abs(after.liquid_mass - before.liquid_mass), 2e-11 * before.liquid_mass);
        EXPECT_LT(after.maximum_divergence, 1e-8);
        for (const auto& m : solver.momentum_transport_diagnostics())
            EXPECT_LE(m.physical_relative_residual, 1e-12);
        const auto p = solver.prepared_transport();
        ASSERT_TRUE(p);
        EXPECT_EQ(p->momentum_builds, 1u);
        EXPECT_EQ(p->momentum_reuses, 2u);
        for (const auto& cell : solver.phases()->phase_volumes())
        {
            EXPECT_GE(cell.gas, 0);
            EXPECT_GE(cell.carrier, 0);
            EXPECT_NEAR(cell.gas + cell.carrier, cell.bulk, 1e-13 * cell.bulk);
        }
        const auto metric = [&](const std::string& name, long double value)
        {
            std::ostringstream output;
            output << std::setprecision(20) << value;
            RecordProperty(name + "_" + std::to_string(step), output.str());
        };
        metric("gas_mass_relative_drift", std::abs(after.gas_mass - before.gas_mass) / before.gas_mass);
        metric("liquid_mass_relative_drift", std::abs(after.liquid_mass - before.liquid_mass) / before.liquid_mass);
        metric("maximum_divergence", after.maximum_divergence);
        metric("maximum_occupancy_defect", p->diagnostics.maximum_occupancy_defect);
        metric("maximum_outgoing_courant", p->diagnostics.maximum_outgoing_courant);
        long double residual = 0;
        for (const auto& m : solver.momentum_transport_diagnostics())
            residual = std::max(residual, (long double) m.physical_relative_residual);
        metric("momentum_physical_relative_residual", residual);
        metric("plane_measurements", p->diagnostics.plane_measurements);
        metric("momentum_builds", p->momentum_builds);
        metric("momentum_reuses", p->momentum_reuses);
        RecordProperty("fallback_streams_" + std::to_string(step), p->diagnostics.fallback_streams);
        RecordProperty("limited_streams_" + std::to_string(step), p->diagnostics.limited_streams);
    }
    EXPECT_NEAR(solver.time(), .02, 1e-16);
}
} // namespace
