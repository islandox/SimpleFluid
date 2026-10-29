#include "FVM/GeneralVoFWork.hh"
#include "general_vof_solver_test_helpers.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"

namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
namespace wire = AMR::adaptive_detail;
using Rows = std::map<AMR::AdaptiveBytes, AMR::AdaptiveBytes>;
constexpr double cleanup_dt = 0x1p-44, cleanup_q = cleanup_dt / 16;

std::shared_ptr<const VoF::GeneralVoF> live_phases(
    bool all_on_zero = false, bool capture = false, double cleanup_cap = 6 * cleanup_q, bool cleanup = true)
{
    const auto comm = Tpetra::getDefaultComm();
    AMR::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.extent = {2, 2, 2};
    AMR::CartesianAMR provider(grid);
    std::vector<std::uint64_t> owned;
    for (std::uint64_t c = 0; c < 8; ++c)
        if ((all_on_zero ? 0 : int(c % std::min(2, comm->getSize()))) == comm->getRank())
            owned.push_back(c);
    auto bootstrap = provider.detach_owned(owned, {'l', 'i', 'v', 'e', 'p', 'a', 'r', 't'});
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    std::vector<AMR::AdaptiveCellKey> order;
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        const double amount = hierarchy->geometry(leaf)->centroid()[0] < 1 ? 1 : 0;
        order.push_back(key);
        gas.push_back(amount);
        carrier.push_back(1 - amount);
    }
    VoF::GeneralVoFOptions options;
    options.record_balance_work = capture;
    if (cleanup)
        options.phase_cleanup = {1e-14, {cleanup_cap, cleanup_cap}};
    return std::make_shared<const VoF::GeneralVoF>(hierarchy, order, gas, carrier, options);
}
std::shared_ptr<const VoF::GeneralVoF> initial_phases_on_rank_zero()
{
    const auto source = live_phases(false, true, 32 * cleanup_q);
    // Publish a phase-only initial ownership remap before attaching any solver
    // fields. This gives the pilot an explicitly verified gross owner imbalance.
    const std::vector<int> destinations(source->owned_order().size(), 0);
    const auto plan = source->hierarchy()->plan_migration(destinations);
    const auto raw = source->hierarchy()->apply(plan);
    const auto native = AMR::assemble_adaptive_native(*raw, raw->configuration_descriptor());
    const auto target =
        source->hierarchy()->apply(plan, raw->with_partition(native.partition, native.mesh->owned_cell_map()));
    const auto migrated = source->plan_remap(plan, target).candidate();
    EXPECT_EQ(migrated->owned_order().size(), Tpetra::getDefaultComm()->getRank() == 0 ? 8u : 0u);
    EXPECT_TRUE(migrated->options().record_balance_work);
    EXPECT_EQ(migrated->phase_cleanup_state().cleaned_cells, 0u);
    return migrated;
}
VoFMaterial live_material()
{
    auto result = inviscid();
    result.gas_density = 1;
    result.liquid_density = 1000;
    return result;
}
void initialize_circulation(GeneralVoFSolver& solver)
{
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
    solver.initialize(velocity, faces);
}
void real(wire::Writer& output, double value)
{
    EXPECT_TRUE(std::isfinite(value));
    output.word(std::bit_cast<std::uint64_t>(value));
}
Rows gather_rows(const Rows& local)
{
    const auto comm = Tpetra::getDefaultComm();
    wire::Writer output;
    output.word(local.size());
    for (const auto& [key, value] : local)
    {
        output.bytes(key);
        output.bytes(value);
    }
    std::vector<OwnerPacket> packets;
    for (int rank = 0; rank < comm->getSize(); ++rank)
        packets.push_back({rank, output.data});
    const auto received = exchange_owner_packets(packets, comm, 8 * 1024 * 1024);
    Rows result;
    for (const auto& packet : received.packets)
    {
        wire::Reader input{packet.bytes};
        const auto count = input.word();
        for (std::uint64_t row = 0; row < count; ++row)
        {
            auto key = input.bytes();
            auto value = input.bytes();
            EXPECT_TRUE(result.emplace(std::move(key), std::move(value)).second);
        }
        input.done();
    }
    return result;
}
Rows cell_rows(const GeneralVoFSolver& solver)
{
    Rows rows;
    const auto& phases = *solver.phases();
    const auto declarations = phases.phase_geometry_declarations();
    for (std::size_t c = 0; c < phases.owned_order().size(); ++c)
    {
        wire::Writer output;
        for (double value : solver.velocity()[c])
            real(output, value);
        for (double value : {solver.pressure()[c], solver.curvature()[c], solver.gas_fraction()[c]})
            real(output, value);
        const auto volumes = phases.phase_volumes()[c];
        for (double value : {volumes.bulk, volumes.gas, volumes.carrier})
            real(output, value);
        for (const auto* error : {&phases.phase_roundoff_bounds(), &phases.accumulated_phase_errors()})
        {
            real(output, (*error)[c].gas);
            real(output, (*error)[c].carrier);
        }
        const auto error = phases.initial_occupancy_errors()[c];
        for (double value : {error.difference, error.admitted_bound, error.native_conversion, error.phase_input_bound,
                 error.fit_residual, error.fit_bound, error.transferred_bound, error.fit_allocation_bound})
            real(output, value);
        output.word(std::uint64_t(declarations[c]));
        rows.emplace(AMR::encode_adaptive_key(phases.owned_order()[c]), std::move(output.data));
    }
    return gather_rows(rows);
}
Rows face_rows(const GeneralVoFSolver& solver)
{
    Rows rows;
    for (const auto& tile : solver.phases()->tiles())
    {
        const auto local = solver.mesh()->owned_face_map()->getLocalElement(tile.face_gid);
        if (local < 0)
            continue;
        wire::Writer identity, output;
        identity.bytes(tile.domain);
        identity.bytes(tile.surface);
        identity.bytes(tile.identity);
        // Native IDs and first-cell orientation may change across ownership.
        // Compare each physical face in persistent lexicographic key orientation.
        const double sign = tile.second && *tile.second < tile.first ? -1 : 1;
        const double value = sign * solver.face_velocity()[local];
        real(output, value == 0 ? 0 : value);
        EXPECT_TRUE(rows.emplace(std::move(identity.data), std::move(output.data)).second);
    }
    return gather_rows(rows);
}
struct Snapshot
{
    std::shared_ptr<const VoF::GeneralVoF> phases;
    std::shared_ptr<const MeshHandle<>> mesh;
    std::shared_ptr<const VoF::GeneralVoFPreparedTransport> prepared;
    Rows cells, faces;
    std::vector<char> cleanup;
    double time = 0;
    std::array<GeneralVoFMomentumDiagnostics, 3> momentum;
    VoFStepDiagnostics diagnostics;
};
Snapshot snapshot(const GeneralVoFSolver& solver)
{
    return {solver.phases(), solver.mesh(), solver.prepared_transport(), cell_rows(solver), face_rows(solver),
        solver.phases()->phase_cleanup_checkpoint(), solver.time(), solver.momentum_transport_diagnostics(),
        solver.diagnostics()};
}
void expect_fields(const GeneralVoFSolver& solver, const Snapshot& expected)
{
    EXPECT_EQ(cell_rows(solver), expected.cells);
    EXPECT_EQ(face_rows(solver), expected.faces);
    EXPECT_EQ(solver.phases()->phase_cleanup_checkpoint(), expected.cleanup);
    EXPECT_EQ(solver.time(), expected.time);
    for (unsigned a = 0; a < 3; ++a)
    {
        const auto& actual = solver.momentum_transport_diagnostics()[a];
        const auto& before = expected.momentum[a];
        EXPECT_EQ(actual.physical_relative_residual, before.physical_relative_residual);
        EXPECT_EQ(actual.implicit_iterations, before.implicit_iterations);
        EXPECT_EQ(actual.defect_corrections, before.defect_corrections);
        EXPECT_EQ(actual.cleanup_momentum, before.cleanup_momentum);
        EXPECT_EQ(actual.accumulated_cleanup_momentum, before.accumulated_cleanup_momentum);
        EXPECT_EQ(actual.cleanup_identity_relative_residual, before.cleanup_identity_relative_residual);
    }
    const auto& actual = solver.diagnostics();
    const auto& before = expected.diagnostics;
    EXPECT_EQ(actual.time, before.time);
    EXPECT_EQ(actual.gas_volume, before.gas_volume);
    EXPECT_EQ(actual.gas_mass, before.gas_mass);
    EXPECT_EQ(actual.liquid_mass, before.liquid_mass);
    EXPECT_EQ(actual.momentum, before.momentum);
    EXPECT_EQ(actual.kinetic_energy, before.kinetic_energy);
    EXPECT_EQ(actual.maximum_velocity, before.maximum_velocity);
    EXPECT_EQ(actual.maximum_divergence, before.maximum_divergence);
    EXPECT_EQ(actual.pressure_residual, before.pressure_residual);
    EXPECT_EQ(actual.pressure_iterations, before.pressure_iterations);
}
void expect_unchanged(const GeneralVoFSolver& solver, const Snapshot& expected)
{
    EXPECT_EQ(solver.phases(), expected.phases);
    EXPECT_EQ(solver.mesh(), expected.mesh);
    EXPECT_EQ(solver.prepared_transport(), expected.prepared);
    expect_fields(solver, expected);
}
std::vector<int> rotate(const GeneralVoFSolver& solver)
{
    const auto comm = Tpetra::getDefaultComm();
    return std::vector<int>(solver.phases()->owned_order().size(), (comm->getRank() + 1) % comm->getSize());
}
void prove_nonzero_state(const GeneralVoFSolver& solver)
{
    const auto& cleanup = solver.phases()->phase_cleanup_state();
    EXPECT_GT(cleanup.cleaned_cells, 0u);
    EXPECT_GT(cleanup.absolute_volume[0] + cleanup.absolute_volume[1], 0);
    double local_pressure = 0, pressure = 0;
    for (double value : solver.pressure())
        local_pressure = std::max(local_pressure, std::abs(value));
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &local_pressure, &pressure);
    EXPECT_GT(pressure, 0);
    double momentum = 0;
    for (const auto& diagnostic : solver.momentum_transport_diagnostics())
        momentum = std::max(momentum, double(std::abs(diagnostic.accumulated_cleanup_momentum)));
    // Opposite local contributions can cancel globally; the actual retained
    // per-cell cleanup momentum is independently required to be nonzero.
    std::uint64_t local_components = 0, components = 0;
    for (const auto& cell : solver.prepared_transport()->cleanup_momentum)
        for (auto value : cell)
            local_components += value != 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_components, &components);
    EXPECT_GT(components, 0u);
    EXPECT_TRUE(std::isfinite(momentum));
    EXPECT_GT(momentum, 0);
}
TEST(GeneralVoFLivePartition, CleanupMigrationPreservesExactStateAndCumulativeRejection)
{
    GeneralVoFSolver solver(live_phases(), live_material());
    initialize_circulation(solver);
    ASSERT_NO_THROW((void) solver.step(cleanup_dt));
    prove_nonzero_state(solver);
    const auto before = snapshot(solver);
    ASSERT_EQ(before.cells.size(), 8u);
    ASSERT_EQ(before.faces.size(), 36u);
    const auto historical = solver.prepared_transport();
    const auto historical_sample = historical->balance_work;
    ASSERT_NO_THROW(solver.repartition(rotate(solver)));
    expect_fields(solver, before);
    if (Tpetra::getDefaultComm()->getSize() > 1)
    {
        EXPECT_NE(solver.phases(), before.phases);
        EXPECT_NE(solver.mesh(), before.mesh);
        EXPECT_EQ(solver.prepared_transport(), nullptr);
        EXPECT_TRUE(solver.phases()->latest_phase_cleanup().empty());
        EXPECT_EQ(historical->balance_work, historical_sample);
        EXPECT_EQ(historical->candidate_identity, before.phases->accepted_identity());
    }
    else
        expect_unchanged(solver, before);
    const auto migrated = snapshot(solver);
    std::string rejection;
    try
    {
        (void) solver.step(cleanup_dt);
        ADD_FAILURE() << "The next cleanup batch must exceed the original cumulative cap.";
    }
    catch (const std::exception& error)
    {
        rejection = error.what();
    }
    EXPECT_NE(rejection.find("phase cleanup"), std::string::npos) << rejection;
    expect_unchanged(solver, migrated);
}
TEST(GeneralVoFLivePartition, RoomyCleanupBudgetAllowsFreshStepAfterOwnershipMigration)
{
    GeneralVoFSolver solver(live_phases(false, false, 1e-10), live_material());
    initialize_circulation(solver);
    ASSERT_NO_THROW((void) solver.step(cleanup_dt));
    prove_nonzero_state(solver);
    const auto before = snapshot(solver);
    ASSERT_NO_THROW(solver.repartition(rotate(solver)));
    expect_fields(solver, before);
    ASSERT_NO_THROW((void) solver.step(cleanup_dt));
    EXPECT_EQ(solver.time(), 2 * cleanup_dt);
    ASSERT_NE(solver.prepared_transport(), nullptr);
    EXPECT_GT(solver.phases()->phase_cleanup_state().absolute_volume[0] +
                  solver.phases()->phase_cleanup_state().absolute_volume[1],
        0);
    for (const auto& diagnostic : solver.momentum_transport_diagnostics())
        EXPECT_LE(diagnostic.physical_relative_residual + diagnostic.cleanup_identity_relative_residual, 1e-12);
}
TEST(GeneralVoFLivePartition, RankLocalInvalidDestinationsAndMissingWorkPreservePublication)
{
    GeneralVoFSolver solver(live_phases(), live_material());
    initialize_circulation(solver);
    const auto initialized = snapshot(solver);
    EXPECT_THROW((void) solver.rebalance(), std::exception);
    expect_unchanged(solver, initialized);
    ASSERT_NO_THROW((void) solver.step(cleanup_dt));
    const auto accepted = snapshot(solver);
    for (unsigned fault = 0; fault < 2; ++fault)
    {
        auto destinations = rotate(solver);
        if (Tpetra::getDefaultComm()->getRank() == 0)
        {
            if (fault == 0)
                destinations[0] = Tpetra::getDefaultComm()->getSize();
            else
                destinations.pop_back();
        }
        EXPECT_THROW(solver.repartition(destinations), std::exception);
        expect_unchanged(solver, accepted);
    }
    const std::vector<int> identity(solver.phases()->owned_order().size(), Tpetra::getDefaultComm()->getRank());
    EXPECT_NO_THROW(solver.repartition(identity));
    expect_unchanged(solver, accepted);
    // An accepted step without optional capture is insufficient for rebalance.
    EXPECT_THROW((void) solver.rebalance(), std::exception);
    expect_unchanged(solver, accepted);
}
TEST(GeneralVoFLivePartition, CapturedStepBindsAcceptedSuccessorAndPreservesFields)
{
    const auto original = initial_phases_on_rank_zero();
    GeneralVoFSolver solver(original, live_material());
    initialize_circulation(solver);
    EXPECT_THROW((void) solver.rebalance(), std::exception);
    ASSERT_NO_THROW((void) solver.step(cleanup_dt));
    prove_nonzero_state(solver);
    const auto accepted = snapshot(solver);
    const auto ledger = solver.prepared_transport();
    ASSERT_NE(ledger, nullptr);
    ASSERT_NE(ledger->balance_work, nullptr);
    const auto declarations = solver.phases()->phase_geometry_declarations();
    EXPECT_EQ(solver.phases()->owned_order().size(), Tpetra::getDefaultComm()->getRank() == 0 ? 8u : 0u);
    for (std::size_t c = 0; c < declarations.size(); ++c)
    {
        EXPECT_NE(declarations[c], VoF::GeneralVoFPhaseGeometry::Infer);
        EXPECT_TRUE(solver.phases()->gas_volumes()[c] == 0 || solver.phases()->carrier_volumes()[c] == 0);
    }
    for (const auto& row : ledger->balance_work->cells)
        EXPECT_FALSE(row.mixed);
    EXPECT_EQ(ledger->source_identity, original->accepted_identity());
    EXPECT_NE(ledger->source_identity, accepted.phases->accepted_identity());
    EXPECT_NO_THROW(accepted.phases->validate_prepared_transport_candidate(*ledger));
    EXPECT_THROW(accepted.phases->validate_prepared_transport_source(*ledger), std::invalid_argument);
    EXPECT_EQ(ledger->candidate_identity, accepted.phases->accepted_identity());
    EXPECT_THROW(
        (void) VoF::plan_partition_proposal(original, *ledger, VoF::GeneralVoFBalanceBinding::AcceptedSuccessor),
        std::invalid_argument);
    for (unsigned fault = 0; fault < 5; ++fault)
    {
        auto forged = *ledger;
        if (Tpetra::getDefaultComm()->getRank() == 0)
        {
            if (fault == 0)
                forged.candidate_identity = original->accepted_identity();
            if (fault == 1)
                ++forged.transaction;
            if (fault == 2)
                ++forged.time_stage;
            if (fault == 3)
                forged.density[0] *= 2;
            if (fault == 4)
                forged.balance_work = std::make_shared<const VoF::GeneralVoFWorkSample>(*ledger->balance_work);
        }
        EXPECT_THROW((void) VoF::plan_partition_proposal(
                         accepted.phases, forged, VoF::GeneralVoFBalanceBinding::AcceptedSuccessor),
            std::invalid_argument);
        expect_unchanged(solver, accepted);
    }
    for (unsigned fault = 0; fault < 3; ++fault)
    {
        VoF::GeneralVoFBalanceOptions options;
        if (Tpetra::getDefaultComm()->getRank() == 0)
        {
            if (fault == 0)
                options.objective.communication_weight = 1;
            if (fault == 1)
                options.objective.byte_budget = 1;
            if (fault == 2)
                options.objective.compute_weight = std::numeric_limits<double>::quiet_NaN();
        }
        EXPECT_THROW((void) solver.rebalance(options), std::exception);
        expect_unchanged(solver, accepted);
    }
    GeneralVoFRebalanceResult result;
    ASSERT_NO_THROW(result = solver.rebalance());
    EXPECT_EQ(result.proposal.binding, VoF::GeneralVoFBalanceBinding::AcceptedSuccessor);
    EXPECT_EQ(result.proposal.current_phase_identity, accepted.phases->accepted_identity());
    EXPECT_EQ(result.proposal.current_phase_control, accepted.phases->accepted_control());
    EXPECT_EQ(result.proposal.observed_source_identity, ledger->source_identity);
    EXPECT_NE(result.proposal.observed_source_identity, result.proposal.current_phase_identity);
    EXPECT_EQ(result.proposal.observed_source_control, ledger->source_control);
    EXPECT_EQ(result.proposal.observed_work, ledger->balance_work);
    EXPECT_EQ(result.applied, Tpetra::getDefaultComm()->getSize() > 1);
    EXPECT_TRUE(result.proposal.measured_before.compute_available);
    EXPECT_TRUE(result.proposal.forecast_proposed.compute_available);
    EXPECT_EQ(result.proposal.measured_before.total_receiver_seconds, 0);
    EXPECT_EQ(result.proposal.forecast_proposed.total_receiver_seconds, 0);
    EXPECT_EQ(result.applied, result.proposal.accepted);
    for (double value : {result.proposal_seconds, result.migration_seconds})
    {
        EXPECT_TRUE(std::isfinite(value));
        EXPECT_GE(value, 0);
    }
    expect_fields(solver, accepted);
    if (result.applied)
    {
        EXPECT_NE(solver.phases(), accepted.phases);
        EXPECT_EQ(solver.prepared_transport(), nullptr);
        EXPECT_TRUE(solver.phases()->latest_phase_cleanup().empty());
        EXPECT_TRUE(result.proposal.native_plan.final.wire_bytes_exact);
        EXPECT_EQ(result.local_native_application_bytes, result.proposal.local_native_application_bytes);
        EXPECT_EQ(result.global_native_application_bytes, result.proposal.global_native_application_bytes);
        EXPECT_EQ(result.global_native_application_bytes, result.proposal.native_plan.final.moved_bytes);
        EXPECT_GT(result.global_native_application_bytes, 0u);
        const auto migrated = snapshot(solver);
        EXPECT_THROW((void) solver.rebalance(), std::exception);
        expect_unchanged(solver, migrated);
        EXPECT_THROW(solver.phases()->validate_prepared_transport_candidate(*ledger), std::invalid_argument);
        ASSERT_NO_THROW((void) solver.step(cleanup_dt));
        EXPECT_EQ(solver.time(), 2 * cleanup_dt);
        ASSERT_NE(solver.prepared_transport(), nullptr);
        EXPECT_NE(solver.prepared_transport(), ledger);
        ASSERT_NE(solver.prepared_transport()->balance_work, nullptr);
        const auto fresh_ledger = solver.prepared_transport();
        const auto fresh_identity = solver.phases()->accepted_identity();
        GeneralVoFRebalanceResult fresh;
        EXPECT_NO_THROW(fresh = solver.rebalance());
        EXPECT_EQ(fresh.proposal.binding, VoF::GeneralVoFBalanceBinding::AcceptedSuccessor);
        EXPECT_EQ(fresh.proposal.current_phase_identity, fresh_identity);
        EXPECT_EQ(fresh.proposal.observed_work, fresh_ledger->balance_work);
    }
    else
        expect_unchanged(solver, accepted);
}
TEST(GeneralVoFLivePartition, PureObservationRejectsNewMixedSuccessorWithoutChangingState)
{
    const auto original = live_phases(true, true, 0, false);
    GeneralVoFSolver solver(original, live_material());
    initialize_circulation(solver);
    ASSERT_NO_THROW((void) solver.step(1. / 64));
    const auto accepted = snapshot(solver);
    const auto ledger = solver.prepared_transport();
    ASSERT_NE(ledger, nullptr);
    ASSERT_NE(ledger->balance_work, nullptr);
    std::uint64_t local_mixed = 0, mixed = 0;
    const auto declarations = solver.phases()->phase_geometry_declarations();
    for (std::size_t c = 0; c < declarations.size(); ++c)
    {
        local_mixed += declarations[c] == VoF::GeneralVoFPhaseGeometry::Infer &&
                       solver.phases()->gas_volumes()[c] > 0 && solver.phases()->carrier_volumes()[c] > 0;
    }
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_mixed, &mixed);
    EXPECT_GT(mixed, 0u);
    for (const auto& row : ledger->balance_work->cells)
        EXPECT_FALSE(row.mixed);
    GeneralVoFRebalanceResult result;
    ASSERT_NO_THROW(result = solver.rebalance());
    EXPECT_FALSE(result.applied);
    EXPECT_FALSE(result.proposal.accepted);
    EXPECT_EQ(result.proposal.rejection, VoF::GeneralVoFBalanceRejection::ObservationClassificationChanged);
    EXPECT_EQ(result.proposal.binding, VoF::GeneralVoFBalanceBinding::AcceptedSuccessor);
    EXPECT_EQ(result.proposal.current_phase_identity, accepted.phases->accepted_identity());
    EXPECT_EQ(result.proposal.observed_source_identity, original->accepted_identity());
    EXPECT_EQ(result.proposal.observed_work, ledger->balance_work);
    EXPECT_EQ(result.local_native_application_bytes, 0u);
    EXPECT_EQ(result.global_native_application_bytes, 0u);
    expect_unchanged(solver, accepted);
}
TEST(GeneralVoFLivePartition, InitializeInvalidatesCapturedStepAndRetainsCleanup)
{
    GeneralVoFSolver solver(live_phases(true, true, 1e-10), live_material());
    initialize_circulation(solver);
    ASSERT_NO_THROW((void) solver.step(cleanup_dt));
    const auto ledger = solver.prepared_transport();
    ASSERT_NE(ledger, nullptr);
    ASSERT_NE(ledger->balance_work, nullptr);
    const auto cleanup = solver.phases()->phase_cleanup_checkpoint();
    EXPECT_EQ(solver.time(), cleanup_dt);
    const auto phases_before = solver.phases();
    const auto velocity = solver.velocity();
    const auto faces = solver.face_velocity();
    ASSERT_NO_THROW(solver.initialize(velocity, faces));
    EXPECT_EQ(solver.prepared_transport(), nullptr);
    EXPECT_EQ(solver.time(), 0);
    EXPECT_EQ(solver.phases(), phases_before);
    EXPECT_EQ(solver.phases()->phase_cleanup_checkpoint(), cleanup);
    const auto initialized = snapshot(solver);
    EXPECT_THROW((void) solver.rebalance(), std::exception);
    expect_unchanged(solver, initialized);
}
TEST(GeneralVoFLivePartition, UnsampledWallPressureContinuesHydrostaticEquilibriumAfterMigration)
{
    auto material = inviscid();
    material.gas_density = material.liquid_density = 1;
    GeneralVoFSolverOptions options;
    options.flow.gravity = {0, 0, 1};
    GeneralVoFSolver solver(phases(0, false, true), material, options);
    ASSERT_NO_THROW((void) solver.step(.125));
    const auto before = snapshot(solver);
    ASSERT_NO_THROW(solver.repartition(rotate(solver)));
    expect_fields(solver, before);
    // There is only one z layer: wall reaction cannot be recovered from a
    // sampled neighboring cell gradient and must survive the ownership handoff.
    ASSERT_NO_THROW((void) solver.step(.125));
    EXPECT_EQ(solver.time(), .25);
    EXPECT_EQ(solver.diagnostics().liquid_mass, before.diagnostics.liquid_mass);
    EXPECT_NEAR(solver.diagnostics().maximum_velocity, 0, 1e-12);
    EXPECT_LT(solver.diagnostics().maximum_divergence, options.flow.divergence_tolerance);
    for (const auto& velocity : solver.velocity())
        for (double value : velocity)
            EXPECT_NEAR(value, 0, 1e-12);
    for (double value : solver.face_velocity())
        EXPECT_NEAR(value, 0, 1e-12);
    for (double value : solver.diagnostics().momentum)
        EXPECT_NEAR(value, 0, 1e-12);
}
} // namespace
