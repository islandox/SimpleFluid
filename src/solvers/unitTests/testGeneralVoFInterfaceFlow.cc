#include "general_vof_solver_test_helpers.hh"
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
std::shared_ptr<const VoF::GeneralVoF> interface_phases(unsigned initial)
{
    const auto comm = Tpetra::getDefaultComm();
    AMR::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.extent = {2, 2, 2};
    AMR::CartesianAMR provider(grid);
    std::vector<std::uint64_t> owned;
    for (std::uint64_t c = 0; c < 8; ++c)
        if (int(c % std::min(2, comm->getSize())) == comm->getRank())
            owned.push_back(c);
    auto bootstrap = provider.detach_owned(owned, {'i', 'n', 't', 'e', 'r', 'f', 'l', 'o', 'w'});
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    std::vector<AMR::AdaptiveCellKey> order;
    std::vector<double> gas, liquid;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        const auto x = hierarchy->geometry(leaf)->centroid();
        const double g = x[0] < 1 ? (initial == 1 ? 1 : .75) : 0;
        order.push_back(key);
        gas.push_back(g);
        liquid.push_back(1 - g);
    }
    int local_empty = order.empty(), empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, std::max(0, comm->getSize() - 2));
    return std::make_shared<const VoF::GeneralVoF>(hierarchy, order, gas, liquid);
}
TEST(GeneralVoFInterfaceFlow, SingleStepAlignedAndMixedPassageUsesSignedMomentumAtomically)
{
    // This checks one accepted phase/momentum/projection transaction. Repeated
    // coupled steps still need the Stage4 thin-fragment robustness qualification.
    for (unsigned initial : {1u, 2u})
    {
        SCOPED_TRACE(::testing::Message() << "initial=" << initial);
        auto material = inviscid();
        material.gas_density = 1;
        material.liquid_density = 1000;
        GeneralVoFSolver solver(interface_phases(initial), material);
        std::vector<double> face_velocity(solver.mesh()->num_faces());
        for (const auto& tile : solver.phases()->tiles())
        {
            if (tile.exterior)
                continue;
            const auto face = solver.mesh()->overlap_face_map()->getLocalElement(tile.face_gid);
            const auto c = tile.geometry->centroid(), n = tile.geometry->area_vector();
            if (c.x == 1 && c.y < 2 && n.x != 0)
                face_velocity[face] = (c.y < 1 ? 1 : -1) * (n.x > 0 ? 1 : -1) / 16.;
            if (c.y == 1 && c.x < 2 && n.y != 0)
                face_velocity[face] = (c.x > 1 ? 1 : -1) * (n.y > 0 ? 1 : -1) / 16.;
        }
        std::vector<VoF::Vector> initial_velocity(solver.mesh()->num_owned_cells());
        for (std::size_t c = 0; c < initial_velocity.size(); ++c)
        {
            const auto x = solver.mesh()->cell_centroid(int(c));
            initial_velocity[c] = {(1 - x.y) / 8, (x.x - 1) / 8, 0};
        }
        solver.initialize(initial_velocity, face_velocity);
        const auto before = solver.diagnostics();
        const auto original_gas = solver.phases()->gas_volumes();
        EXPECT_GT(before.maximum_velocity, 0);
        for (unsigned step = 0; step < 1; ++step)
        {
            SCOPED_TRACE(::testing::Message() << "step=" << step);
            VoFStepDiagnostics after;
            ASSERT_NO_THROW(after = solver.step(1. / 64));
            EXPECT_LE(std::abs(after.gas_mass - before.gas_mass), 2e-11 * before.gas_mass);
            EXPECT_LE(std::abs(after.liquid_mass - before.liquid_mass), 2e-11 * before.liquid_mass);
            EXPECT_LT(after.maximum_divergence, 1e-8);
            const auto prepared = solver.prepared_transport();
            ASSERT_TRUE(prepared);
            EXPECT_EQ(prepared->family, VoF::GeneralVoFTransportFamily::InterfaceLocalSwept);
            EXPECT_EQ(prepared->backend, VoF::GeneralVoFTransportBackend::SweptPartition);
            EXPECT_TRUE(prepared->positive_transfers.empty());
            EXPECT_EQ(prepared->momentum_builds, 1u);
            EXPECT_EQ(prepared->momentum_reuses, 2u);
            for (const auto& residual : solver.momentum_transport_diagnostics())
                EXPECT_LE(residual.physical_relative_residual, 1e-12);
            for (const auto& amount : solver.phases()->phase_volumes())
            {
                EXPECT_GE(amount.gas, 0);
                EXPECT_GE(amount.carrier, 0);
            }
        }
        EXPECT_EQ(solver.time(), 1. / 64);
        double local = 0, passage = 0;
        for (std::size_t c = 0; c < original_gas.size(); ++c)
            local += std::abs(solver.phases()->gas_volumes()[c] - original_gas[c]);
        Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local, &passage);
        EXPECT_GT(passage, 1e-7);
        const auto accepted = solver.phases();
        const auto prepared = solver.prepared_transport();
        const auto velocity = solver.velocity();
        const auto pressure = solver.pressure();
        const auto flux = solver.face_velocity();
        EXPECT_THROW((void) solver.step(1e8), std::runtime_error);
        EXPECT_EQ(solver.phases(), accepted);
        EXPECT_EQ(solver.prepared_transport(), prepared);
        EXPECT_EQ(solver.velocity(), velocity);
        EXPECT_EQ(solver.pressure(), pressure);
        EXPECT_EQ(solver.face_velocity(), flux);
        EXPECT_EQ(solver.time(), 1. / 64);
    }
}
} // namespace
