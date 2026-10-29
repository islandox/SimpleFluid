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
TEST(GeneralVoFInterfaceFlow, ThreeStepAlignedAndMixedPassageUsesSignedMomentumAtomically)
{
    // Repeated accepted transactions exercise thin computational pieces created
    // after both aligned and mixed interfaces enter neighboring pure cells.
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
        for (unsigned step = 0; step < 3; ++step)
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
            const int local_quality = int(prepared->diagnostics.quality_limited_faces);
            int quality_faces = 0;
            Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_quality, &quality_faces);
            RecordProperty(
                "quality_limited_faces_" + std::to_string(initial) + "_" + std::to_string(step), quality_faces);
            EXPECT_GT(prepared->diagnostics.minimum_quality_theta, 0);
            EXPECT_LE(prepared->diagnostics.minimum_quality_theta, 1);
            EXPECT_LE(prepared->diagnostics.minimum_theta, prepared->diagnostics.minimum_quality_theta);
            for (const auto& residual : solver.momentum_transport_diagnostics())
                EXPECT_LE(residual.physical_relative_residual, 1e-12);
            for (const auto& amount : solver.phases()->phase_volumes())
            {
                EXPECT_GE(amount.gas, 0);
                EXPECT_GE(amount.carrier, 0);
            }
        }
        EXPECT_EQ(solver.time(), 3. / 64);
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
        EXPECT_EQ(solver.time(), 3. / 64);
    }
}
void check_cleanup_coupling(VoF::GeneralVoFTransportBackend backend)
{
    constexpr double dt = 0x1p-44, q = dt / 16;
    const auto base = interface_phases(1);
    VoF::GeneralVoFOptions raw_options;
    raw_options.transport_backend = backend;
    raw_options.phase_cleanup.relative_cutoff = 0;
    const auto initial = std::make_shared<const VoF::GeneralVoF>(
        base->hierarchy(), base->owned_order(), base->gas_volumes(), base->carrier_volumes(), raw_options);
    VoF::GeneralVoFOptions options;
    options.transport_backend = backend;
    options.phase_cleanup = {1e-14, {6 * q, 6 * q}};
    auto cleaned = std::make_shared<const VoF::GeneralVoF>(
        initial->hierarchy(), initial->owned_order(), initial->gas_volumes(), initial->carrier_volumes(), options);
    auto material = inviscid();
    material.gas_density = 1;
    material.liquid_density = 1000;
    GeneralVoFSolver raw(initial, material), solver(cleaned, material);
    std::vector<double> faces(solver.mesh()->num_faces());
    for (const auto& tile : solver.phases()->tiles())
    {
        if (tile.exterior)
            continue;
        const auto local = solver.mesh()->overlap_face_map()->getLocalElement(tile.face_gid);
        const auto c = tile.geometry->centroid(), n = tile.geometry->area_vector();
        if (c.x == 1 && n.x != 0)
            faces[local] = (c.y < 1 ? 1 : -1) * (n.x > 0 ? 1 : -1) / 16.;
        if (c.y == 1 && n.y != 0)
            faces[local] = (c.x > 1 ? 1 : -1) * (n.y > 0 ? 1 : -1) / 16.;
    }
    std::vector<VoF::Vector> velocity(solver.mesh()->num_owned_cells());
    for (std::size_t c = 0; c < velocity.size(); ++c)
    {
        const auto x = solver.mesh()->cell_centroid(int(c));
        velocity[c] = {(1 - x.y) / 8, (x.x - 1) / 8, 0};
    }
    raw.initialize(velocity, faces);
    solver.initialize(velocity, faces);
    ASSERT_NO_THROW((void) raw.step(dt));
    ASSERT_NO_THROW((void) solver.step(dt));
    const auto prepared = solver.prepared_transport();
    ASSERT_TRUE(prepared);
    EXPECT_GT(solver.phases()->phase_cleanup_state().cleaned_cells, 0u);
    double local_nonzero = 0, global_nonzero = 0;
    std::array<double, 3> local_momentum{}, global_momentum{};
    for (std::size_t c = 0; c < velocity.size(); ++c)
    {
        const long double dg =
            static_cast<long double>(solver.phases()->gas_volumes()[c]) - raw.phases()->gas_volumes()[c];
        const long double dl =
            static_cast<long double>(solver.phases()->carrier_volumes()[c]) - raw.phases()->carrier_volumes()[c];
        const long double dm = dg + 1000 * dl;
        EXPECT_EQ(prepared->cleanup_mass[c], dm);
        EXPECT_EQ(prepared->raw_new_mass[c], raw.prepared_transport()->new_mass[c]);
        for (unsigned a = 0; a < 3; ++a)
        {
            const auto expected = dm * raw.velocity()[c][a];
            // The independent raw solve includes its pressure projection; its
            // O(q) difference from the transported trace is below this relative oracle tolerance.
            if (a == 2)
                EXPECT_EQ(prepared->cleanup_momentum[c][a], 0); // The raw transported axial momentum is exactly zero.
            else
                EXPECT_NEAR(prepared->cleanup_momentum[c][a], expected, 1e-10L * std::abs(expected));
            local_nonzero += prepared->cleanup_momentum[c][a] != 0;
            local_momentum[a] += static_cast<double>(prepared->cleanup_momentum[c][a]);
        }
    }
    const auto comm = Tpetra::getDefaultComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_nonzero, &global_nonzero);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 3, local_momentum.data(), global_momentum.data());
    EXPECT_GT(global_nonzero, 0);
    for (unsigned a = 0; a < 3; ++a)
    {
        const auto& diagnostic = solver.momentum_transport_diagnostics()[a];
        EXPECT_LE(diagnostic.physical_relative_residual + diagnostic.cleanup_identity_relative_residual, 1e-12);
        EXPECT_NEAR(diagnostic.cleanup_momentum, global_momentum[a], 1e-14L * std::abs(diagnostic.cleanup_momentum));
        EXPECT_EQ(diagnostic.accumulated_cleanup_momentum, diagnostic.cleanup_momentum);
    }
    const auto accepted = solver.phases();
    const auto accepted_velocity = solver.velocity();
    const auto accepted_pressure = solver.pressure();
    const auto ledger = solver.phases()->phase_cleanup_checkpoint();
    std::string rejection;
    try
    {
        (void) solver.step(dt);
        ADD_FAILURE() << "The second cleanup batch must exceed the cumulative cap.";
    }
    catch (const std::exception& error)
    {
        rejection = error.what();
    }
    EXPECT_NE(rejection.find("phase cleanup"), std::string::npos) << rejection;
    EXPECT_EQ(solver.time(), dt);
    EXPECT_EQ(solver.phases(), accepted);
    EXPECT_EQ(solver.prepared_transport(), prepared);
    EXPECT_EQ(solver.velocity(), accepted_velocity);
    EXPECT_EQ(solver.pressure(), accepted_pressure);
    EXPECT_EQ(solver.phases()->phase_cleanup_checkpoint(), ledger);
}
TEST(GeneralVoFInterfaceFlow, CleanupUsesRawMassAndExplicitMomentumAndRejectsWithoutSpending)
{
    for (const auto backend :
        {VoF::GeneralVoFTransportBackend::SweptPartition, VoF::GeneralVoFTransportBackend::GeometricFace})
    {
        SCOPED_TRACE(int(backend));
        check_cleanup_coupling(backend);
    }
}
} // namespace
