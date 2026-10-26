/** Homogeneous moving swept transport retains the coupled solver contract. */
#include "general_vof_solver_test_helpers.hh"
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

std::shared_ptr<const VoF::GeneralVoF> homogeneous_phases(bool gas, bool trace)
{
    const auto source = phases(0, false, true);
    auto gas_volume = source->gas_volumes(), liquid_volume = source->carrier_volumes();
    if (gas)
        std::swap(gas_volume, liquid_volume);
    if (trace)
        std::fill(
            gas ? liquid_volume.begin() : gas_volume.begin(), gas ? liquid_volume.end() : gas_volume.end(), 1e-25);
    std::vector<VoF::GeneralVoFPhaseGeometry> declarations(source->owned_order().size(),
        gas ? VoF::GeneralVoFPhaseGeometry::PureGas : VoF::GeneralVoFPhaseGeometry::PureCarrier);
    return std::make_shared<const VoF::GeneralVoF>(
        source->hierarchy(), source->owned_order(), gas_volume, liquid_volume, source->options(), declarations);
}

TEST(GeneralVoFHomogeneousFlow, MovingGasAndLiquidReuseMomentumWithoutSweptPartitionWork)
{
    struct Statistics
    {
        Statistics() { VoF::GeneralVoF::enable_geometry_statistics(true); }
        ~Statistics() { VoF::GeneralVoF::enable_geometry_statistics(false); }
    } statistics;
    for (bool gas : {false, true})
        for (bool trace : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << "gas=" << gas << ", trace=" << trace);
            auto material = inviscid();
            material.gas_density = 1;
            material.liquid_density = 1000;
            GeneralVoFSolver solver(homogeneous_phases(gas, trace), material);
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
            const auto stable = solver.stable_timestep();
            EXPECT_TRUE(std::isfinite(stable));
            EXPECT_GT(stable, 0);
            for (unsigned step = 0; step < 3; ++step)
            {
                const auto source = solver.phases();
                VoF::GeneralVoF::reset_geometry_statistics();
                const auto after = solver.step(.001);
                const auto geometry = VoF::GeneralVoF::geometry_statistics().bounded;
                EXPECT_EQ(geometry.intersections, 0u);
                EXPECT_EQ(geometry.partition_checks, 0u);
                EXPECT_NEAR(after.gas_mass, before.gas_mass, 1e-12 * std::abs(before.gas_mass));
                EXPECT_NEAR(after.liquid_mass, before.liquid_mass, 1e-12 * std::abs(before.liquid_mass));
                EXPECT_LT(after.maximum_divergence, 1e-8);
                const auto p = solver.prepared_transport();
                ASSERT_TRUE(p);
                EXPECT_EQ(p->backend, VoF::GeneralVoFTransportBackend::SweptPartition);
                EXPECT_EQ(p->family, VoF::GeneralVoFTransportFamily::AlgebraicFaceVolume);
                EXPECT_EQ(p->source_identity, source->accepted_identity());
                EXPECT_EQ(p->candidate_identity, solver.phases()->accepted_identity());
                EXPECT_EQ(p->density, (std::array<double, 2>{1, 1000}));
                EXPECT_TRUE(p->node_trajectories.empty());
                EXPECT_EQ(p->diagnostics.plane_measurements, 0u);
                ASSERT_TRUE(p->momentum_preparation);
                EXPECT_EQ(p->momentum_builds, 1u);
                EXPECT_EQ(p->momentum_reuses, 2u);
                for (const auto& m : solver.momentum_transport_diagnostics())
                    EXPECT_LE(m.physical_relative_residual, 1e-12);
            }
            EXPECT_NEAR(solver.time(), .003, 1e-18);
            const auto accepted = solver.phases();
            const auto preparation = solver.prepared_transport();
            const auto velocity = solver.velocity();
            const auto face = solver.face_velocity();
            const auto pressure = solver.pressure();
            const auto histories = accepted->accumulated_phase_errors();
            const auto diagnostics = solver.diagnostics();
            EXPECT_THROW((void) solver.step(1e8), std::runtime_error);
            EXPECT_EQ(solver.phases(), accepted);
            EXPECT_EQ(solver.prepared_transport(), preparation);
            EXPECT_EQ(solver.velocity(), velocity);
            EXPECT_EQ(solver.face_velocity(), face);
            EXPECT_EQ(solver.pressure(), pressure);
            EXPECT_EQ(solver.time(), diagnostics.time);
            EXPECT_EQ(solver.diagnostics().gas_mass, diagnostics.gas_mass);
            EXPECT_EQ(solver.diagnostics().liquid_mass, diagnostics.liquid_mass);
            EXPECT_EQ(solver.diagnostics().momentum, diagnostics.momentum);
            ASSERT_EQ(accepted->accumulated_phase_errors().size(), histories.size());
            for (std::size_t c = 0; c < histories.size(); ++c)
            {
                EXPECT_EQ(accepted->accumulated_phase_errors()[c].gas, histories[c].gas);
                EXPECT_EQ(accepted->accumulated_phase_errors()[c].carrier, histories[c].carrier);
            }
        }
}
} // namespace
