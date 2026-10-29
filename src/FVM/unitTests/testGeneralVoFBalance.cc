#include "FVM/GeneralVoFBalance.hh"
#include "FVM/GeneralVoFWork.hh"
#include "FVM/details/GeneralVoF/BalanceForecast.hh"
#include "general_vof_finite_test_helpers.hh"
#include "geometry/PLIC/GeometryError.hh"
#include <algorithm>
#include <array>
#include <limits>
#include <set>

namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using general_vof_detail::forecast_partition_work;
AdaptiveCellKey key(unsigned id)
{
    return {0, StaticCellAncestry{0, id}};
}
GeneralVoFWorkDonor donor(unsigned id, int owner)
{
    return {key(id), 7, owner};
}
GeneralVoFWorkCell cell(unsigned id, int owner, double cost)
{
    GeneralVoFWorkCell result;
    result.key = key(id);
    result.owner = owner;
    result.revision = 7;
    result.calls = 1;
    result.reconstruction.seconds = cost;
    return result;
}
GeneralVoFWorkFace face(unsigned id, int owner, double cost)
{
    GeneralVoFWorkFace result;
    result.identity = {char(id)};
    result.first = donor(id, owner);
    result.second = donor(0, 0);
    result.geometric = true;
    result.setup.seconds = cost / 4;
    result.aggregate.seconds = 3 * cost / 4;
    // Repeated face dependencies must also be deduplicated per requester rank.
    result.donors = {donor(0, 0), donor(0, 0)};
    return result;
}
GeneralVoFWorkSample fanout_sample()
{
    GeneralVoFWorkSample result;
    const int rank = Tpetra::getDefaultComm()->getRank();
    if (rank == 0)
    {
        result.cells = {cell(0, 0, 4)};
        result.cells[0].mixed = true;
        result.cells[0].request_payload_bytes = 64;
        result.cells[0].reply_payload_bytes = 192;
    }
    if (rank == 1)
    {
        result.cells = {cell(1, 1, 1), cell(2, 1, 1)};
        result.faces = {face(1, 1, 2), face(2, 1, 9)};
    }
    if (rank == 2)
    {
        result.cells = {cell(3, 2, 1)};
        result.faces = {face(3, 2, 5)};
    }
    if (rank == 1 || rank == 2)
    {
        GeneralVoFWorkReceiver receiver;
        receiver.donor = donor(0, 0);
        receiver.requester = rank;
        receiver.mixed = true;
        receiver.remeasurement.seconds = rank == 1 ? 2 : 6;
        receiver.request_payload_bytes = 64;
        receiver.reply_payload_bytes = 192;
        result.receivers.push_back(receiver);
    }
    return result;
}
std::vector<int> destinations(const GeneralVoFWorkSample& sample)
{
    return std::vector<int>(sample.cells.size(), Tpetra::getDefaultComm()->getRank());
}
void expect_metrics(const GeneralVoFWorkMetrics& metrics)
{
    EXPECT_TRUE(std::isfinite(metrics.seconds));
    EXPECT_GE(metrics.seconds, 0);
}
TEST(GeneralVoFBalanceForecast, ExplicitMovePreservesObservedReceiversAndDeduplicatesFanout)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 4)
        GTEST_SKIP() << "Independent four-rank oracle includes an initially empty rank.";
    const auto sample = fanout_sample();
    auto target = destinations(sample);
    const auto identity = forecast_partition_work(sample, target, comm);
    const std::array<double, 4> core{4, 13, 6, 0}, receiver{0, 2, 6, 0};
    EXPECT_EQ(identity.local_core_before, core[comm->getRank()]);
    EXPECT_EQ(identity.local_receiver_before, receiver[comm->getRank()]);
    EXPECT_EQ(identity.local_core_proposed, identity.local_core_before);
    EXPECT_EQ(identity.local_receiver_proposed, identity.local_receiver_before);
    EXPECT_TRUE(identity.measured_before.compute_available);
    EXPECT_TRUE(identity.forecast_proposed.compute_available);
    EXPECT_EQ(identity.measured_before.maximum_compute_seconds, 15);
    EXPECT_EQ(identity.measured_before.mean_compute_seconds, 31. / 4);
    EXPECT_EQ(identity.measured_before.total_core_seconds, 23);
    EXPECT_EQ(identity.measured_before.remote_donor_consumers, 2u);
    EXPECT_EQ(identity.measured_before.request_payload_bytes, 128u);
    EXPECT_EQ(identity.measured_before.reply_payload_bytes, 384u);
    EXPECT_EQ(identity.measured_before.canonical_cut_faces, 3u);
    EXPECT_EQ(identity.forecast_proposed.maximum_compute_seconds, 15);
    EXPECT_EQ(identity.forecast_proposed.remote_donor_consumers, 2u);
    EXPECT_EQ(identity.forecast_proposed.request_payload_bytes, 128u);
    EXPECT_EQ(identity.forecast_proposed.reply_payload_bytes, 384u);
    if (comm->getRank() == 1)
        target[1] = 3; // B owns its canonical face work after migration.
    for (const auto estimator : {GeneralVoFReceiverEstimator::Mean, GeneralVoFReceiverEstimator::Maximum})
    {
        SCOPED_TRACE(int(estimator));
        GeneralVoFBalanceOptions options;
        options.receiver_estimator = estimator;
        const auto moved = forecast_partition_work(sample, target, comm, options);
        const bool mean = estimator == GeneralVoFReceiverEstimator::Mean;
        const std::array<double, 4> after_core{4, 3, 6, 10}, after_receiver{0, 2, 6, mean ? 4. : 6.};
        EXPECT_FALSE(moved.incomplete_receiver_model);
        EXPECT_EQ(moved.local_core_proposed, after_core[comm->getRank()]);
        EXPECT_EQ(moved.local_receiver_proposed, after_receiver[comm->getRank()]);
        EXPECT_EQ(moved.forecast_proposed.total_core_seconds, 23);
        EXPECT_EQ(moved.forecast_proposed.total_receiver_seconds, mean ? 12 : 14);
        EXPECT_EQ(moved.forecast_proposed.maximum_compute_seconds, mean ? 14 : 16);
        EXPECT_EQ(moved.forecast_proposed.mean_compute_seconds, mean ? 35. / 4 : 37. / 4);
        EXPECT_EQ(moved.forecast_proposed.remote_donor_consumers, 3u);
        EXPECT_EQ(moved.forecast_proposed.request_payload_bytes, 192u);
        EXPECT_EQ(moved.forecast_proposed.reply_payload_bytes, 576u);
        EXPECT_EQ(moved.forecast_proposed.canonical_cut_faces, 3u);
    }
}
TEST(GeneralVoFBalanceForecast, UnobservedMixedReceiverIsUnavailableAndPureReceiverCostsZero)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "A new remote consumer requires MPI.";
    for (bool mixed : {true, false})
    {
        GeneralVoFWorkSample sample;
        if (comm->getRank() == 0)
        {
            sample.cells = {cell(0, 0, 4), cell(1, 0, 1)};
            sample.cells[0].mixed = mixed;
            sample.cells[0].request_payload_bytes = 64;
            sample.cells[0].reply_payload_bytes = 192;
            sample.faces = {face(1, 0, 2)};
        }
        auto target = destinations(sample);
        if (comm->getRank() == 0)
            target[1] = 1;
        const auto forecast = forecast_partition_work(sample, target, comm);
        EXPECT_EQ(forecast.incomplete_receiver_model, mixed);
        EXPECT_EQ(forecast.forecast_proposed.compute_available, !mixed);
        EXPECT_EQ(forecast.forecast_proposed.remote_donor_consumers, 1u);
        EXPECT_EQ(forecast.forecast_proposed.request_payload_bytes, 64u);
        if (!mixed)
        {
            EXPECT_EQ(forecast.forecast_proposed.total_receiver_seconds, 0);
            EXPECT_EQ(forecast.forecast_proposed.maximum_compute_seconds, 4);
        }
    }
}
TEST(GeneralVoFBalanceForecast, RankLocalMalformedRowsDependenciesAndBudgetRejectCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    for (unsigned fault = 0; fault < 8; ++fault)
    {
        SCOPED_TRACE(fault);
        GeneralVoFWorkSample sample;
        if (comm->getRank() == 0)
        {
            sample.cells = {cell(0, 0, 4), cell(1, 0, 1)};
            sample.cells[0].request_payload_bytes = 64;
            sample.cells[0].reply_payload_bytes = 192;
            sample.faces = {face(1, 0, 2)};
            if (fault == 0)
                sample.cells[1].key = sample.cells[0].key;
            if (fault == 1)
                ++sample.faces[0].first.revision;
            if (fault == 2)
                ++sample.faces[0].second.revision;
            if (fault == 3)
                ++sample.faces[0].donors[0].revision;
            if (fault == 4)
                sample.faces.push_back(sample.faces[0]);
            if (fault == 5)
                sample.cells[0].reconstruction.seconds = -1;
            if (fault == 6)
                sample.cells[0].reconstruction.seconds = std::numeric_limits<double>::infinity();
        }
        auto target = destinations(sample);
        GeneralVoFBalanceOptions options;
        if (fault == 7 && comm->getRank() == 0)
            options.objective.byte_budget = 1;
        EXPECT_THROW((void) forecast_partition_work(sample, target, comm, options), std::invalid_argument);
    }
}

Fixture real_fixture(bool planar, bool tails = false)
{
    const auto comm = Tpetra::getDefaultComm();
    AMRGrid grid;
    grid.block_cells = {planar ? 4u : 2u, 2, 2};
    grid.extent = {planar ? 4. : 2., 2, 2};
    CartesianAMR mesh(grid);
    std::vector<std::uint64_t> owned;
    for (std::uint64_t id = 0; id < (planar ? 16u : 8u); ++id)
        if ((planar ? int(id % std::min(2, comm->getSize())) : 0) == comm->getRank())
            owned.push_back(id);
    auto native = mesh.detach_owned(owned, {'b', 'a', 'l', 'a', 'n', 'c', 'e'});
    Fixture result;
    result.hierarchy = std::make_shared<const DistributedAdaptiveHierarchy>(
        std::move(native.owned), std::move(native.families), native.kernel, comm);
    for (const auto& [key, leaf] : result.hierarchy->owned())
    {
        const auto x = result.hierarchy->geometry(leaf)->centroid();
        const auto gas = planar ? std::clamp(1.5 - int(x[0]), 0., 1.)
                                : (x[0] < 1 && x[1] < 1 && x[2] < 1 ? 1. : (tails ? 0x1p-60 : 0.));
        result.order.push_back(key);
        result.gas.push_back(gas);
        result.carrier.push_back(planar ? 1 - gas : (gas == 1 ? 0 : 1));
    }
    return result;
}
// Same four-cell circulation at each z, padded by stationary cells when nx=4.
std::vector<GeneralVoFFaceVolume> pilot_flux(const GeneralVoF& source)
{
    auto result = circulation(source);
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        const auto p = source.tiles()[i].geometry->centroid();
        if (p.x >= 2 || p.y >= 2)
            result[i].integrated_volume = 0;
    }
    return result;
}
TEST(GeneralVoFBalanceCapture, MixedPlanarPilotPreservesPhysicalUpdateAndReconcilesExecutedWork)
{
    const auto input = real_fixture(true);
    GeneralVoFOptions options;
    options.record_balance_work = true;
    auto enabled = std::make_shared<const GeneralVoF>(input.hierarchy, input.order, input.gas, input.carrier, options);
    GeneralVoF disabled(input.hierarchy, input.order, input.gas, input.carrier);
    const auto trial = enabled->plan_transport(stationary(*enabled), pilot_flux(*enabled), 1);
    const auto reference = disabled.plan_transport(stationary(disabled), pilot_flux(disabled), 1);
    const auto prepared = trial.prepare_transport({1, 1000}, 29);
    ASSERT_NE(prepared->balance_work, nullptr);
    EXPECT_EQ(reference.prepare_transport({1, 1000}, 29)->balance_work, nullptr);
    const auto& sample = *prepared->balance_work;
    // Isolate retained capture storage from all unrelated ledger vectors. This
    // checks the public byte boundary, including captures on empty owners.
    GeneralVoFPreparedTransport accounting;
    accounting.balance_work = prepared->balance_work;
    const auto exact_bytes = sizeof(accounting) + prepared->balance_work->retained_bytes;
    EXPECT_NO_THROW(accounting.validate_byte_budget(exact_bytes));
    try
    {
        accounting.validate_byte_budget(exact_bytes - 1);
        ADD_FAILURE() << "Capture retention must be charged to the public ledger byte budget.";
    }
    catch (const PLIC::GeometryError& error)
    {
        EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::ResourceLimit);
    }
    accounting.balance_work.reset();
    EXPECT_NO_THROW(accounting.validate_byte_budget(sizeof(accounting)));
    EXPECT_EQ(sample.source_identity, enabled->accepted_identity());
    EXPECT_EQ(sample.source_control, enabled->accepted_control());
    EXPECT_EQ(sample.time_stage, 29u);
    EXPECT_EQ(sample.timestep, 1);
    EXPECT_TRUE(sample.fully_deduplicated);
    EXPECT_GT(sample.retained_bytes, 0u);
    ASSERT_EQ(sample.cells.size(), input.order.size());
    std::size_t calls = 0, mixed = 0, retained = 0, pure = 0, zero = 0;
    std::uint64_t volumes = 0, intersections = 0;
    double reconstruction_seconds = 0;
    for (std::size_t i = 0; i < sample.cells.size(); ++i)
    {
        const auto& row = sample.cells[i];
        EXPECT_EQ(row.key, input.order[i]);
        EXPECT_EQ(row.owner, Tpetra::getDefaultComm()->getRank());
        EXPECT_EQ(row.revision, input.hierarchy->owned().at(row.key).revision);
        EXPECT_LE(row.calls, 1u);
        expect_metrics(row.reconstruction);
        if (row.calls == 0)
        {
            EXPECT_EQ(row.reconstruction.seconds, 0);
            EXPECT_EQ(row.reconstruction.floating_operations, 0u);
            EXPECT_EQ(row.reconstruction.volume_evaluations, 0u);
            EXPECT_EQ(row.reconstruction.intersections, 0u);
            EXPECT_EQ(row.reconstruction.constructions, 0u);
        }
        EXPECT_EQ(row.calls, row.mixed_fits + row.retained_returns + row.pure_returns + row.zero_phase_returns);
        calls += row.calls;
        mixed += row.mixed_fits;
        retained += row.retained_returns;
        pure += row.pure_returns;
        zero += row.zero_phase_returns;
        volumes += row.reconstruction.volume_evaluations;
        intersections += row.reconstruction.intersections;
        reconstruction_seconds += row.reconstruction.seconds;
    }
    const auto& diagnostic = prepared->diagnostics.owner_plane_requests;
    EXPECT_EQ(calls, diagnostic.owner_reconstruction_calls);
    EXPECT_EQ(mixed, diagnostic.mixed_cell_fits);
    EXPECT_EQ(retained, diagnostic.retained_interface_returns);
    EXPECT_EQ(pure, diagnostic.pure_declaration_returns);
    EXPECT_EQ(zero, diagnostic.zero_phase_offset_returns);
    EXPECT_EQ(volumes, diagnostic.geometry_volume_evaluations);
    EXPECT_EQ(intersections, diagnostic.geometry_intersections);
    EXPECT_EQ(calls, diagnostic.local_owned_requests + diagnostic.peer_cache_admissions);
    EXPECT_EQ(diagnostic.cache_admission_denials, 0u);
    EXPECT_TRUE(std::isfinite(reconstruction_seconds));
    std::set<AdaptiveBytes> expected_faces;
    for (const auto& tile : enabled->tiles())
        if (!tile.exterior && tile.second && tile.first_owner == Tpetra::getDefaultComm()->getRank())
            expected_faces.insert(tile.identity);
    std::set<AdaptiveBytes> faces;
    std::set<AdaptiveCellKey> requested_donors;
    for (const auto& row : sample.faces)
    {
        EXPECT_TRUE(faces.insert(row.identity).second);
        EXPECT_EQ(row.first.owner, Tpetra::getDefaultComm()->getRank());
        expect_metrics(row.setup);
        expect_metrics(row.aggregate);
        for (const auto& dependency : row.donors)
            requested_donors.insert(dependency.key);
        if (!row.geometric)
            EXPECT_TRUE(row.donors.empty());
    }
    EXPECT_EQ(faces, expected_faces);
    EXPECT_EQ(requested_donors.size(), prepared->diagnostics.plane_requests);
    EXPECT_EQ(sample.receivers.size() + diagnostic.local_owned_requests, prepared->diagnostics.plane_requests);
    const std::array<std::uint64_t, 2> local_rows{sample.cells.size(), sample.faces.size()};
    std::array<std::uint64_t, 2> global_rows{};
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 2, local_rows.data(), global_rows.data());
    EXPECT_EQ(global_rows[0], 16u);
    EXPECT_EQ(global_rows[1], 28u); // 12 x-faces + 8 y-faces + 8 z-faces.
    std::set<AdaptiveCellKey> receivers;
    for (const auto& row : sample.receivers)
    {
        EXPECT_TRUE(receivers.insert(row.donor.key).second);
        EXPECT_EQ(row.requester, Tpetra::getDefaultComm()->getRank());
        EXPECT_NE(row.donor.owner, row.requester);
        expect_metrics(row.remeasurement);
        EXPECT_GT(row.request_payload_bytes, 0u);
        EXPECT_GT(row.reply_payload_bytes, 0u);
        if (!row.mixed)
        {
            EXPECT_EQ(row.remeasurement.seconds, 0);
            EXPECT_EQ(row.remeasurement.volume_evaluations, 0u);
            EXPECT_EQ(row.remeasurement.intersections, 0u);
        }
    }
    const auto actual = trial.candidate(), expected = reference.candidate();
    EXPECT_EQ(actual->gas_volumes(), expected->gas_volumes());
    EXPECT_EQ(actual->carrier_volumes(), expected->carrier_volumes());
    for (std::size_t i = 0; i < input.order.size(); ++i)
    {
        const auto x = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[i]))->centroid();
        const auto analytic = int(x[0]) == 1 ? .5 + (x[1] < 1 ? 1. / 128 : -1. / 128) : input.gas[i];
        EXPECT_NEAR(actual->gas_volumes()[i], analytic, 8e-14);
        EXPECT_NEAR(actual->carrier_volumes()[i], 1 - analytic, 8e-14);
        EXPECT_EQ(actual->phase_roundoff_bounds()[i].gas, expected->phase_roundoff_bounds()[i].gas);
        EXPECT_EQ(actual->phase_roundoff_bounds()[i].carrier, expected->phase_roundoff_bounds()[i].carrier);
        EXPECT_EQ(actual->accumulated_phase_errors()[i].gas, expected->accumulated_phase_errors()[i].gas);
        EXPECT_EQ(actual->accumulated_phase_errors()[i].carrier, expected->accumulated_phase_errors()[i].carrier);
    }
    EXPECT_EQ(enabled->gas_volumes(), input.gas);
    EXPECT_EQ(enabled->carrier_volumes(), input.carrier);
}
void expect_cleanup(const GeneralVoFPhaseCleanupState& actual, const GeneralVoFPhaseCleanupState& expected)
{
    EXPECT_EQ(actual.cleaned_cells, expected.cleaned_cells);
    EXPECT_EQ(actual.signed_volume, expected.signed_volume);
    EXPECT_EQ(actual.absolute_volume, expected.absolute_volume);
    EXPECT_EQ(actual.largest_absolute_volume, expected.largest_absolute_volume);
    EXPECT_EQ(actual.policy_bound, expected.policy_bound);
    EXPECT_EQ(actual.policy.relative_cutoff, expected.policy.relative_cutoff);
    EXPECT_EQ(actual.policy.maximum_absolute_volume, expected.policy.maximum_absolute_volume);
}
std::map<AdaptiveCellKey, std::array<double, 15>> broadcast_initial_rows(const GeneralVoF& source)
{
    const auto comm = Tpetra::getDefaultComm();
    std::map<AdaptiveCellKey, std::array<double, 15>> rows;
    const auto declarations = source.phase_geometry_declarations();
    for (unsigned i = 0; i < 8; ++i)
    {
        AdaptiveBytes bytes;
        std::array<double, 15> values{};
        if (comm->getRank() == 0)
        {
            bytes = encode_adaptive_key(source.owned_order()[i]);
            const auto& occupancy = source.initial_occupancy_errors()[i];
            values = {source.gas_volumes()[i], source.carrier_volumes()[i], source.phase_roundoff_bounds()[i].gas,
                source.phase_roundoff_bounds()[i].carrier, source.accumulated_phase_errors()[i].gas,
                source.accumulated_phase_errors()[i].carrier, double(declarations[i]), occupancy.difference,
                occupancy.admitted_bound, occupancy.native_conversion, occupancy.phase_input_bound,
                occupancy.fit_residual, occupancy.fit_bound, occupancy.transferred_bound,
                occupancy.fit_allocation_bound};
        }
        int size = bytes.size();
        Teuchos::broadcast(*comm, 0, 1, &size);
        bytes.resize(size);
        Teuchos::broadcast(*comm, 0, size, bytes.data());
        Teuchos::broadcast(*comm, 0, int(values.size()), values.data());
        rows.emplace(decode_adaptive_key(bytes), values);
    }
    return rows;
}
TEST(GeneralVoFBalance, InitialMigrationPreservesKeyedIndependentPhasesHistoryAndCleanup)
{
    const auto comm = Tpetra::getDefaultComm();
    for (bool tails : {false, true})
    {
        SCOPED_TRACE(tails);
        const auto input = real_fixture(false, tails);
        GeneralVoFOptions options;
        options.record_balance_work = true;
        if (tails)
            options.phase_cleanup = {1e-14, {1e-12, 1e-12}};
        auto source =
            std::make_shared<const GeneralVoF>(input.hierarchy, input.order, input.gas, input.carrier, options);
        const auto identity = source->accepted_identity();
        const auto gas = source->gas_volumes(), carrier = source->carrier_volumes();
        const auto cleanup = source->phase_cleanup_state();
        if (tails)
        {
            EXPECT_EQ(cleanup.cleaned_cells, 7u);
            EXPECT_EQ(cleanup.absolute_volume[0], 7 * 0x1p-60L);
            EXPECT_EQ(cleanup.absolute_volume[1], 0);
        }
        const auto rows = broadcast_initial_rows(*source);
        const auto trial = source->plan_transport(stationary(*source), pilot_flux(*source), 1);
        const auto prepared = trial.prepare_transport({1, 1000}, 43);
        ASSERT_NE(prepared->balance_work, nullptr);
        const auto result = plan_initial_partition(source, *prepared);
        EXPECT_EQ(source->accepted_identity(), identity);
        EXPECT_EQ(source->gas_volumes(), gas);
        EXPECT_EQ(source->carrier_volumes(), carrier);
        expect_cleanup(source->phase_cleanup_state(), cleanup);
        ASSERT_NE(result.candidate, nullptr);
        EXPECT_EQ(result.accepted, comm->getSize() > 1);
        if (comm->getSize() == 1)
        {
            EXPECT_EQ(result.candidate, source);
            EXPECT_EQ(result.rejection, GeneralVoFBalanceRejection::NoCoreImprovement);
        }
        else if (result.accepted)
        {
            EXPECT_NE(result.candidate, source);
            EXPECT_TRUE(result.candidate->latest_phase_cleanup().empty());
            EXPECT_TRUE(result.native_plan.accepted);
            EXPECT_TRUE(result.native_plan.final.wire_bytes_exact);
            EXPECT_GT(result.native_application_bytes, 0u);
            EXPECT_EQ(result.native_application_bytes, result.native_plan.final.moved_bytes);
            const auto native = source->hierarchy()->plan_migration(result.native_plan);
            const auto local_bytes = native.actual_migration_application_bytes();
            std::uint64_t global_bytes = 0;
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_bytes, &global_bytes);
            EXPECT_EQ(global_bytes, result.native_application_bytes);
            EXPECT_TRUE(result.measured_before.compute_available);
            EXPECT_TRUE(result.forecast_proposed.compute_available);
            EXPECT_LT(result.forecast_proposed.maximum_compute_seconds, result.measured_before.maximum_compute_seconds);
        }
        for (double seconds :
            {result.native_migration_seconds, result.phase_migration_seconds, result.total_migration_seconds})
        {
            EXPECT_TRUE(std::isfinite(seconds));
            EXPECT_GE(seconds, 0);
        }
        const auto& migrated = *result.candidate;
        expect_cleanup(migrated.phase_cleanup_state(), cleanup);
        EXPECT_EQ(migrated.options().record_balance_work, options.record_balance_work);
        EXPECT_EQ(migrated.options().transport_backend, options.transport_backend);
        EXPECT_EQ(migrated.options().arithmetic_mode, options.arithmetic_mode);
        const auto declarations = migrated.phase_geometry_declarations();
        for (std::size_t i = 0; i < migrated.owned_order().size(); ++i)
        {
            const auto found = rows.find(migrated.owned_order()[i]);
            EXPECT_NE(found, rows.end());
            if (found == rows.end())
                continue;
            const auto& expected = found->second;
            EXPECT_EQ(migrated.gas_volumes()[i], expected[0]);
            EXPECT_EQ(migrated.carrier_volumes()[i], expected[1]);
            EXPECT_EQ(migrated.phase_roundoff_bounds()[i].gas, expected[2]);
            EXPECT_EQ(migrated.phase_roundoff_bounds()[i].carrier, expected[3]);
            EXPECT_EQ(migrated.accumulated_phase_errors()[i].gas, expected[4]);
            EXPECT_EQ(migrated.accumulated_phase_errors()[i].carrier, expected[5]);
            EXPECT_EQ(double(declarations[i]), expected[6]);
            const auto& occupancy = migrated.initial_occupancy_errors()[i];
            EXPECT_EQ(occupancy.difference, expected[7]);
            EXPECT_EQ(occupancy.admitted_bound, expected[8]);
            EXPECT_EQ(occupancy.native_conversion, expected[9]);
            EXPECT_EQ(occupancy.phase_input_bound, expected[10]);
            EXPECT_EQ(occupancy.fit_residual, expected[11]);
            EXPECT_EQ(occupancy.fit_bound, expected[12]);
            EXPECT_EQ(occupancy.transferred_bound, expected[13]);
            EXPECT_EQ(occupancy.fit_allocation_bound, expected[14]);
        }
        const std::uint64_t local_count = migrated.owned_order().size();
        std::uint64_t count = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_count, &count);
        EXPECT_EQ(count, 8u);
        EXPECT_NO_THROW(source->validate_prepared_transport_source(*prepared));
    }
}
TEST(GeneralVoFBalance, MissingForgedAndStaleSnapshotsRejectIncludingEmptyRanks)
{
    const auto input = real_fixture(false);
    GeneralVoFOptions options;
    options.record_balance_work = true;
    auto source = std::make_shared<const GeneralVoF>(input.hierarchy, input.order, input.gas, input.carrier, options);
    const auto prepared = source->plan_transport(stationary(*source), pilot_flux(*source), 1).prepare_transport();
    ASSERT_NE(prepared->balance_work, nullptr);
    const auto identity = source->accepted_identity();
    const auto new_publication =
        std::make_shared<const GeneralVoF>(input.hierarchy, input.order, input.gas, input.carrier, options);
    EXPECT_EQ(new_publication->hierarchy()->identity(), source->hierarchy()->identity());
    EXPECT_EQ(new_publication->hierarchy()->topology_generation(), source->hierarchy()->topology_generation());
    EXPECT_EQ(new_publication->hierarchy()->partition_generation(), source->hierarchy()->partition_generation());
    EXPECT_EQ(new_publication->owned_order(), source->owned_order());
    for (const auto& cell_key : source->owned_order())
        EXPECT_EQ(new_publication->hierarchy()->owned().at(cell_key).revision,
            source->hierarchy()->owned().at(cell_key).revision);
    EXPECT_NE(new_publication->accepted_identity(), identity);
    EXPECT_THROW((void) plan_initial_partition(new_publication, *prepared), std::invalid_argument);
    for (unsigned fault = 0; fault < 7; ++fault)
    {
        SCOPED_TRACE(fault);
        auto forged = *prepared;
        if (Tpetra::getDefaultComm()->getRank() == 0)
        {
            if (fault == 0)
                forged.balance_work.reset();
            else if (fault < 4)
            {
                auto snapshot = std::make_shared<GeneralVoFWorkSample>(*forged.balance_work);
                if (fault == 2)
                    snapshot->cells.pop_back();
                if (fault == 3)
                    ++snapshot->source_control;
                forged.balance_work = std::move(snapshot);
            }
            else if (fault == 4)
                ++forged.source_control;
            else if (fault == 5)
                ++forged.mesh_generation;
            else
                ++forged.mesh_topology_generation;
        }
        EXPECT_THROW((void) plan_initial_partition(source, forged), std::invalid_argument);
    }
    GeneralVoFOptions disabled_options;
    const auto disabled =
        std::make_shared<const GeneralVoF>(input.hierarchy, input.order, input.gas, input.carrier, disabled_options);
    const auto missing = disabled->plan_transport(stationary(*disabled), pilot_flux(*disabled), 1).prepare_transport();
    EXPECT_EQ(missing->balance_work, nullptr);
    EXPECT_THROW((void) plan_initial_partition(disabled, *missing), std::invalid_argument);
    EXPECT_EQ(source->accepted_identity(), identity);
    EXPECT_EQ(source->gas_volumes(), input.gas);
    EXPECT_EQ(source->carrier_volumes(), input.carrier);
}
TEST(GeneralVoFBalance, InvalidOrRankDivergentObjectivesRejectCollectively)
{
    const auto input = real_fixture(false);
    GeneralVoFOptions capture;
    capture.record_balance_work = true;
    auto source = std::make_shared<const GeneralVoF>(input.hierarchy, input.order, input.gas, input.carrier, capture);
    const auto prepared = source->plan_transport(stationary(*source), pilot_flux(*source), 1).prepare_transport();
    for (unsigned fault = 0; fault < 11; ++fault)
    {
        SCOPED_TRACE(fault);
        GeneralVoFBalanceOptions options;
        if (Tpetra::getDefaultComm()->getRank() == 0)
        {
            if (fault == 0)
                options.objective.work_unit = AdaptiveWorkUnit::Operations;
            if (fault == 1)
                options.objective.communication_weight = 1;
            if (fault == 2)
                options.objective.migration_weight = 1;
            if (fault == 3)
                options.objective.compute_weight = -1;
            if (fault == 4)
                options.objective.compute_weight = std::numeric_limits<double>::quiet_NaN();
            if (fault == 5)
                options.objective.minimum_relative_improvement = -1;
            if (fault == 6)
                options.objective.minimum_relative_improvement = std::numeric_limits<double>::infinity();
            if (fault == 7)
                options.objective.byte_budget = 1;
            if (fault == 8)
                options.receiver_estimator = static_cast<GeneralVoFReceiverEstimator>(123);
            if (fault == 9)
                options.objective.communication_weight = std::numeric_limits<double>::quiet_NaN();
            if (fault == 10)
                options.objective.migration_weight = -1;
        }
        EXPECT_THROW((void) plan_initial_partition(source, *prepared, options), std::invalid_argument);
    }
}
TEST(GeneralVoFBalance, SuccessorForecastKeepsHistoricalPinsAndRejectsUnmeasuredMixedState)
{
    const auto input = real_fixture(false);
    GeneralVoFOptions options;
    options.record_balance_work = true;
    auto source = std::make_shared<const GeneralVoF>(input.hierarchy, input.order, input.gas, input.carrier, options);
    const auto trial = source->plan_transport(stationary(*source), pilot_flux(*source), 1);
    const auto prepared = trial.prepare_transport();
    const auto successor = trial.candidate();
    const auto source_identity = source->accepted_identity(), successor_identity = successor->accepted_identity();
    EXPECT_NO_THROW(source->validate_prepared_transport_source(*prepared));
    EXPECT_NO_THROW(successor->validate_prepared_transport_candidate(*prepared));
    EXPECT_THROW(successor->validate_prepared_transport_source(*prepared), std::invalid_argument);
    EXPECT_THROW(source->validate_prepared_transport_candidate(*prepared), std::invalid_argument);
    int local_mixed = 0, global_mixed = 0;
    for (std::size_t i = 0; i < successor->owned_order().size(); ++i)
        local_mixed += successor->gas_volumes()[i] > 0 && successor->carrier_volumes()[i] > 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_mixed, &global_mixed);
    EXPECT_GT(global_mixed, 0);
    const auto proposal = plan_partition_proposal(successor, *prepared, GeneralVoFBalanceBinding::AcceptedSuccessor);
    EXPECT_FALSE(proposal.accepted);
    EXPECT_EQ(proposal.rejection, GeneralVoFBalanceRejection::ObservationClassificationChanged);
    EXPECT_EQ(proposal.current_phase_identity, successor_identity);
    EXPECT_EQ(proposal.observed_source_identity, source_identity);
    EXPECT_EQ(proposal.observed_work, prepared->balance_work);
    EXPECT_EQ(prepared->balance_work->source_identity, source_identity);
    EXPECT_EQ(successor->accepted_identity(), successor_identity);
    for (unsigned fault = 0; fault < 3; ++fault)
    {
        auto altered = *prepared;
        if (Tpetra::getDefaultComm()->getRank() == 0)
        {
            if (fault == 0)
                altered.candidate_identity = source_identity;
            else if (fault == 1)
                ++altered.transaction;
            else
                altered.timestep = 0;
        }
        EXPECT_THROW(successor->validate_prepared_transport_candidate(altered), std::invalid_argument);
    }
}
} // namespace
