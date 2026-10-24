/** @file testExactGeneralVoFSolver.cc */
#include "general_vof_solver_test_helpers.hh"
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
std::shared_ptr<const VoF::GeneralVoF> inventory_fixture(
    VoF::PhaseInventoryPolicy policy, bool seed_remainder, bool cell_local_refinement = false)
{
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    const auto owners = std::min(2, comm->getSize());
    for (std::uint64_t c = 0; c < 8; ++c)
        if (int(c % owners) == comm->getRank())
            owned.push_back(c);
    auto boot = [&]
    {
        const AMR::AdaptiveBytes domain = {'o', 'p', 't', '_', 'i', 'n', 'v', 'e', 'n', 't', 'o', 'r', 'y'};
        if (cell_local_refinement)
        {
            AMR::ExtrudedAMRGrid grid;
            for (unsigned y = 0; y < 3; ++y)
                for (unsigned x = 0; x < 3; ++x)
                    grid.xy_nodes.push_back({double(x), double(y), 0});
            for (unsigned y = 0; y < 2; ++y)
                for (unsigned x = 0; x < 2; ++x)
                {
                    const unsigned a = 3 * y + x;
                    grid.base_cells.push_back({a, a + 1, a + 4, a + 3});
                }
            grid.z_edges = {0, 1, 2};
            return AMR::ExtrudedAMR(grid).detach_owned(owned, domain);
        }
        AMR::AMRGrid grid;
        grid.block_cells = {2, 2, 2};
        grid.extent = {2, 2, 2};
        return AMR::CartesianAMR(grid).detach_owned(owned, domain);
    }();
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(boot.owned), std::move(boot.families), boot.kernel, comm);
    std::vector<AMR::AdaptiveCellKey> order;
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        const auto x = hierarchy->geometry(leaf)->centroid();
        const double g = seed_remainder && x[0] < 1 && x[1] < 1 && x[2] < 1 ? 1 : 0;
        order.push_back(key);
        gas.push_back(g);
        carrier.push_back(1 - g);
    }
    VoF::GeneralVoFOptions options;
    options.arithmetic_mode =
        policy == VoF::PhaseInventoryPolicy::ExperimentalExact ? ArithmeticMode::Exact : ArithmeticMode::Ordinary;
    auto source = std::make_shared<const VoF::GeneralVoF>(hierarchy, order, gas, carrier, options);
    int local_empty = source->mesh()->num_owned_cells() == 0, empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, comm->getSize() - owners);
    if (!seed_remainder)
        return source;
    std::vector<VoF::GeneralVoFNodeVelocity> nodes;
    for (const auto& node : source->nodes())
        nodes.push_back({node.gid, {0, 0, 0}});
    std::vector<VoF::GeneralVoFFaceVolume> controls;
    for (const auto& face : source->tiles())
    {
        const auto x = face.geometry->centroid(), area = face.geometry->area_vector();
        double phi = 0;
        if (!face.exterior && x.x == 1 && area.x != 0)
            phi = (x.y < 1 ? 1 : -1) * (area.x > 0 ? 1 : -1) * .1;
        if (!face.exterior && x.y == 1 && area.y != 0)
            phi = (x.x > 1 ? 1 : -1) * (area.y > 0 ? 1 : -1) * .1;
        controls.push_back({face.face_gid, phi});
    }
    return source->apply(source->plan_transport(nodes, controls, 1));
}
TEST(ExactGeneralVoFSolverTest, ExperimentalInventoryIsExplicitAndRollbackPreservesItsSelectedAuthority)
{
    using Policy = VoF::PhaseInventoryPolicy;
    auto finite = phases(0);
    EXPECT_EQ(finite->inventory_policy(), Policy::Finite);
    EXPECT_THROW((void) finite->experimental_phases(), std::exception);
    auto source = inventory_fixture(Policy::ExperimentalExact, true);
    const auto before = source->experimental_phases();
    int local_remainder = 0, remainders = 0;
    for (std::size_t c = 0; c < before.size(); ++c)
        local_remainder +=
            before[c].gas.compare(Meshes::ExactGeometryMeasure::from_value(source->gas_volumes()[c])) != 0 ||
            before[c].carrier.compare(Meshes::ExactGeometryMeasure::from_value(source->carrier_volumes()[c])) != 0;
    Teuchos::reduceAll(
        *source->mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_remainder, &remainders);
    ASSERT_GT(remainders, 0);
    GeneralVoFSolver solver(source, inviscid());
    EXPECT_EQ(solver.phases()->inventory_policy(), Policy::ExperimentalExact);
    EXPECT_THROW((void) solver.step(-.001), std::exception);
    EXPECT_EQ(solver.phases(), source);
    EXPECT_EQ(solver.time(), 0);
    ASSERT_NO_THROW((void) solver.step(.001));
    EXPECT_DOUBLE_EQ(solver.time(), .001);
    ASSERT_EQ(solver.phases()->experimental_phases().size(), before.size());
    for (std::size_t c = 0; c < before.size(); ++c)
    {
        EXPECT_EQ(solver.phases()->experimental_phases()[c].gas.compare(before[c].gas), 0);
        EXPECT_EQ(solver.phases()->experimental_phases()[c].carrier.compare(before[c].carrier), 0);
    }
    const auto snapshot = solver.phases();
    std::vector<int> destination(snapshot->owned_order().size(),
        (Tpetra::getDefaultComm()->getRank() + 1) % Tpetra::getDefaultComm()->getSize());
    namespace wire = AMR::adaptive_detail;
    std::vector<OwnerPacket> expected_packets;
    for (std::size_t c = 0; c < snapshot->owned_order().size(); ++c)
    {
        wire::Writer w;
        w.bytes(AMR::encode_adaptive_key(snapshot->owned_order()[c]));
        w.bytes(snapshot->experimental_phases()[c].gas.encode());
        w.bytes(snapshot->experimental_phases()[c].carrier.encode());
        expected_packets.push_back({destination[c], std::move(w.data)});
    }
    const auto expected_packets_received =
        exchange_owner_packets(expected_packets, snapshot->mesh()->owned_cell_map()->getComm(), 1024 * 1024);
    std::map<AMR::AdaptiveCellKey, std::pair<Meshes::ExactGeometryMeasure, Meshes::ExactGeometryMeasure>> expected;
    bool expected_valid = true;
    for (const auto& packet : expected_packets_received.packets)
    {
        wire::Reader r{packet.bytes};
        auto key = AMR::decode_adaptive_key(r.bytes());
        auto gas = Meshes::ExactGeometryMeasure::decode(r.bytes());
        auto carrier = Meshes::ExactGeometryMeasure::decode(r.bytes());
        r.done();
        expected_valid =
            expected.emplace(std::move(key), std::pair{std::move(gas), std::move(carrier)}).second && expected_valid;
    }
    int local_valid = expected_valid, global_valid = 0;
    Teuchos::reduceAll(
        *snapshot->mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_valid, &global_valid);
    ASSERT_TRUE(global_valid);
    ASSERT_NO_THROW(solver.repartition(destination));
    EXPECT_EQ(solver.phases()->inventory_policy(), Policy::ExperimentalExact);
    EXPECT_DOUBLE_EQ(solver.time(), .001);
    const auto& migrated = solver.phases()->experimental_phases();
    ASSERT_EQ(expected.size(), migrated.size());
    for (std::size_t c = 0; c < migrated.size(); ++c)
    {
        const auto& value = expected.at(solver.phases()->owned_order()[c]);
        EXPECT_EQ(migrated[c].gas.compare(value.first), 0);
        EXPECT_EQ(migrated[c].carrier.compare(value.second), 0);
    }
    EXPECT_TRUE(snapshot->hierarchy()->geometry_unchanged());
}

TEST(ExactGeneralVoFSolverTest, ExperimentalInventorySamePolicyRegridRetainsClockAndPhaseAuthority)
{
    using Policy = VoF::PhaseInventoryPolicy;
    // A Cartesian mark refines its entire 8-cell block, so the 15-cell oracle
    // requires the equivalent XY/Z provider's cell-local isotropic split.
    GeneralVoFSolver solver(inventory_fixture(Policy::ExperimentalExact, false, true), inviscid());
    const auto cells = [&]
    {
        double local = solver.mesh()->num_owned_cells(), global = 0;
        Teuchos::reduceAll(*solver.mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
        return global;
    };
    ASSERT_EQ(cells(), 8);
    ASSERT_NO_THROW((void) solver.step(.001));
    const auto before = solver.diagnostics();
    const auto source = solver.phases();
    std::vector<AMR::AdaptiveMark> marks;
    for (const auto& key : source->owned_order())
    {
        const auto x = source->hierarchy()->geometry(source->hierarchy()->owned().at(key))->centroid();
        if (x[0] < 1 && x[1] < 1 && x[2] < 1)
            marks.push_back({key, AMR::AdaptiveSplit::Isotropic});
    }
    EXPECT_NO_THROW(solver.regrid(marks));
    EXPECT_EQ(solver.phases()->inventory_policy(), Policy::ExperimentalExact);
    EXPECT_DOUBLE_EQ(solver.time(), before.time);
    EXPECT_EQ(solver.diagnostics().liquid_mass, before.liquid_mass);
    EXPECT_EQ(solver.diagnostics().momentum, before.momentum);
    EXPECT_EQ(cells(), 15);
}

TEST(ExactGeneralVoFSolverTest, ExperimentalInventoryRejectsForeignPolicyTrialWithoutChangingAcceptedSolver)
{
    using Policy = VoF::PhaseInventoryPolicy;
    auto finite = inventory_fixture(Policy::Finite, false);
    auto exact = inventory_fixture(Policy::ExperimentalExact, false);
    GeneralVoFSolver solver(exact, inviscid());
    std::vector<VoF::GeneralVoFNodeVelocity> nodes;
    for (const auto& node : finite->nodes())
        nodes.push_back({node.gid, {0, 0, 0}});
    std::vector<VoF::GeneralVoFFaceVolume> controls;
    for (const auto& face : finite->tiles())
        controls.push_back({face.face_gid, 0});
    const auto foreign = finite->plan_transport(nodes, controls, 1);
    EXPECT_THROW((void) solver.phases()->apply(foreign), std::exception);
    EXPECT_EQ(solver.phases(), exact);
    EXPECT_EQ(solver.time(), 0);
    EXPECT_EQ(solver.phases()->inventory_policy(), Policy::ExperimentalExact);
}

TEST(ExactGeneralVoFSolverTest, GeometryInstrumentationObservesActualFiniteAndExactLibraryOperations)
{
    enable_observed_exact_geometry_statistics(true);
    reset_observed_geometry_statistics();
    struct DisableOnExit
    {
        ~DisableOnExit() { enable_observed_exact_geometry_statistics(false); }
    } disable;
    const auto comm = Tpetra::getDefaultComm();
    const auto global_count = [&](std::uint64_t local)
    {
        std::uint64_t global = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local, &global);
        return global;
    };
    for (const auto& image : geometry_image_snapshots())
    {
        EXPECT_NE(image.exact_domain, nullptr);
        EXPECT_NE(image.bounded_domain, nullptr);
    }
    const auto finite_source = phases(0);
    const auto fvm_finite = VoF::GeneralVoF::geometry_statistics();
    EXPECT_GT(global_count(fvm_finite.bounded.constructions), 0);
    const auto before_finite_solver = GeneralVoFSolver::geometry_statistics();
    GeneralVoFSolver finite_solver(finite_source, inviscid());
    const auto after_finite_solver = GeneralVoFSolver::geometry_statistics();
    EXPECT_GT(global_count(
                  after_finite_solver.bounded.floating_operations - before_finite_solver.bounded.floating_operations),
        0);
    auto exact_options = finite_source->options();
    exact_options.arithmetic_mode = ArithmeticMode::Exact;
    const auto before_exact_fvm = VoF::GeneralVoF::geometry_statistics();
    const auto exact_source = std::make_shared<const VoF::GeneralVoF>(finite_source->hierarchy(),
        finite_source->owned_order(), finite_source->gas_volumes(), finite_source->carrier_volumes(), exact_options);
    const auto after_exact_fvm = VoF::GeneralVoF::geometry_statistics();
    EXPECT_GT(global_count(after_exact_fvm.exact.measure_values - before_exact_fvm.exact.measure_values), 0);
    const auto before_exact_solver = GeneralVoFSolver::geometry_statistics();
    GeneralVoFSolver exact_solver(exact_source, inviscid());
    const auto after_exact_solver = GeneralVoFSolver::geometry_statistics();
    EXPECT_GT(
        global_count(after_exact_solver.exact.measure_arithmetic - before_exact_solver.exact.measure_arithmetic), 0);
    // The aggregate represents these observed linked images only. A shared
    // domain is counted once even if symbol binding makes two bridges alias.
    const auto combined = observed_geometry_statistics();
    EXPECT_GE(combined.exact.measure_values, after_exact_fvm.exact.measure_values);
    EXPECT_GE(combined.exact.measure_arithmetic, after_exact_solver.exact.measure_arithmetic);
    enable_observed_exact_geometry_statistics(false);
    const auto disabled_before = observed_geometry_statistics();
    (void) Meshes::ExactGeometryMeasure::from_value(13).scaled(3);
    const auto disabled_after = observed_geometry_statistics();
    EXPECT_EQ(disabled_after.exact.measure_values, disabled_before.exact.measure_values);
    EXPECT_EQ(disabled_after.exact.measure_arithmetic, disabled_before.exact.measure_arithmetic);
}
} // namespace
