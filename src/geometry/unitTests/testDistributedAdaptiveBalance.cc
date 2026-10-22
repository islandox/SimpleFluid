/** @file testDistributedAdaptiveBalance.cc
 * @brief Actual operation recording, independent objective oracles and pinned migration.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveBalance.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptivePackets.hh"
#include "geometry/mesh/AMR/DistributedAncestryDirectory.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
::SimpleFluid::Meshes::AMR::AdaptiveCellKey key(std::uint64_t id)
{
    return {0, ::SimpleFluid::Meshes::AMR::ConvexCellAncestry{id, {}}};
}
::SimpleFluid::Meshes::AMR::AdaptiveBalanceSource source()
{
    return {std::make_shared<const char>(0), 3, 7};
}
std::shared_ptr<const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy> cartesian_source()
{
    ::SimpleFluid::Meshes::AMR::AMRGrid config;
    config.block_cells = {2, 2, 2};
    ::SimpleFluid::Meshes::AMR::CartesianAMR original(config);
    auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    if (comm->getRank() == 0)
        for (std::uint64_t i = 0; i < 8; ++i)
            owned.push_back(i);
    auto detached = original.detach_owned(owned, {'d'});
    return std::make_shared<const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy>(
        std::move(detached.owned), std::move(detached.families), std::move(detached.kernel), comm);
}
TEST(DistributedAdaptiveBalance, RecorderUsesActualClippingFaceAndPopulationOperations)
{
    ::SimpleFluid::Meshes::AMR::AdaptiveWorkRecorder recorder(::SimpleFluid::Meshes::AMR::AdaptiveWorkUnit::Operations);
    auto shape = ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1});
    double clipping = 0, faces = 0, population = 0;
    for (int i = 0; i < 3; ++i)
    {
        auto clipped = shape->clip({{1, 0, 0}, {0, 0, 0}, .25 + .125 * i});
        (void) clipped;
        ++clipping;
    }
    for (std::size_t f = 0; f < shape->num_faces(); ++f)
    {
        EXPECT_GT(shape->face_geometry(f)->area().value, 0);
        ++faces;
    }
    std::vector<double> bins{1, 2, 3, 4};
    for (auto& bin : bins)
    {
        bin *= .5;
        ++population;
    }
    recorder.observe(key(0), 11, clipping, faces, population);
    const auto bytes = shape->encode_exact_certificate();
    recorder.record_payload(key(0), 11, bytes.size());
    auto measured = recorder.snapshot(key(0), 11);
    EXPECT_EQ(measured.reconstruction, 3);
    EXPECT_EQ(measured.interface, 6);
    EXPECT_EQ(measured.population, 4);
    EXPECT_EQ(measured.total(), 13);
    EXPECT_EQ(measured.migration_bytes, bytes.size());
    EXPECT_THROW(recorder.snapshot(key(0), 12), std::invalid_argument);
    EXPECT_THROW(recorder.observe(key(0), 12, 1, 0, 0), std::invalid_argument);
    EXPECT_THROW(recorder.observe(key(0), 11, NAN, 0, 0), std::invalid_argument);
    EXPECT_EQ(recorder.snapshot(key(0), 11).total(), 13);
}
TEST(DistributedAdaptiveBalance, MeasuredOperationsDrivePinnedHierarchyMigration)
{
    auto comm = Tpetra::getDefaultComm();
    auto state = cartesian_source();
    ::SimpleFluid::Meshes::AMR::AMRGrid config;
    config.block_cells = {2, 2, 2};
    auto kernel = ::SimpleFluid::Meshes::AMR::cartesian_adaptive_kernel(config);
    ::SimpleFluid::Meshes::AMR::AdaptiveWorkRecorder recorder(::SimpleFluid::Meshes::AMR::AdaptiveWorkUnit::Operations);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBalanceCell> cells;
    for (const auto& [cell, leaf] : state->owned())
    {
        auto shape = kernel->geometry(leaf);
        auto moments = shape->moments();
        GeometryPoint center = moments.origin;
        const auto volume = moments.volume.as_double();
        for (std::size_t a = 0; a < 3; ++a)
            center[a] += moments.first_moment[a].as_double() / volume;
        auto clipped = shape->clip({{1, 0, 0}, center, 0});
        (void) clipped;
        recorder.observe(cell, leaf.revision, 1, 0, 0);
        auto bytes = ::SimpleFluid::Meshes::AMR::encode_adaptive_leaf(leaf);
        recorder.record_payload(cell, leaf.revision, bytes.size());
        cells.push_back({cell, leaf.revision, recorder.snapshot(cell, leaf.revision)});
    }
    auto plan = ::SimpleFluid::Meshes::AMR::balance_owned_cells(
        cells, {}, {state->identity(), state->topology_generation(), state->partition_generation()}, comm);
    EXPECT_EQ(plan.source.identity, state->identity());
    EXPECT_FALSE(plan.proposed.wire_bytes_exact);
    EXPECT_EQ(plan.accepted, comm->getSize() > 1);
    if (comm->getSize() > 1)
        EXPECT_LT(plan.final.maximum_compute, plan.before.maximum_compute);
    // This geometry-only hierarchy has no named state/history. Pack the actual
    // candidate destination batches and include the owner-exchange record and
    // peer headers, rather than summing standalone per-leaf recipe bytes.
    std::map<int, std::vector<::SimpleFluid::Meshes::AMR::AdaptiveLeaf>> batches;
    std::size_t index = 0;
    for (const auto& [cell, leaf] : state->owned())
    {
        const int destination = plan.localdestinations[index++];
        if (destination != comm->getRank())
            batches[destination].push_back(leaf);
    }
    std::uint64_t wire = 0;
    for (const auto& [destination, leaves] : batches)
        wire += ::SimpleFluid::Meshes::AMR::encode_adaptive_batch(leaves, {}, 64 * 1024 * 1024).size() +
                3 * sizeof(std::uint64_t);
    plan = ::SimpleFluid::Meshes::AMR::qualify_balance_wire_bytes(std::move(plan), wire, comm);
    auto migration = state->plan_migration(plan);
    auto moved = state->apply(migration);
    int local = int(moved->owned().size()), global = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local, &global);
    EXPECT_EQ(global, 8);
    EXPECT_EQ(state->owned().size(), comm->getRank() == 0 ? 8 : 0);
    EXPECT_EQ(moved->topology_generation(), state->topology_generation());
    if (plan.accepted)
        EXPECT_THROW(moved->plan_migration(plan), std::invalid_argument);
    else
        EXPECT_EQ(moved->identity(), state->identity());
}
TEST(DistributedAdaptiveBalance, SyntheticWeightedProposalImprovesUnevenOwners)
{
    auto comm = Tpetra::getDefaultComm();
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBalanceCell> cells;
    if (comm->getRank() == 0)
        for (std::uint64_t i = 0; i < 8; ++i)
            cells.push_back(
                {key(i), 0, {double(i + 1), 0, 0, 128, ::SimpleFluid::Meshes::AMR::AdaptiveWorkUnit::Operations}});
    auto plan = ::SimpleFluid::Meshes::AMR::balance_owned_cells(cells, {}, source(), comm);
    EXPECT_EQ(plan.before.total_compute, 36);
    EXPECT_EQ(plan.before.maximum_compute, 36);
    EXPECT_EQ(plan.accepted, comm->getSize() > 1);
    if (plan.accepted)
    {
        EXPECT_LT(plan.final.maximum_compute, 36);
        EXPECT_GT(plan.final.moved_bytes, 0);
        EXPECT_GT(plan.final.moved_cells, 0);
    }
}
TEST(DistributedAdaptiveBalance, ActiveRankBudgetsPreserveGreedyAssignmentsAndIdentityValidation)
{
    using namespace SimpleFluid::Meshes::AMR;
    const auto comm = Tpetra::getDefaultComm();
    for (int pattern = 0; pattern < 3; ++pattern)
    {
        std::vector<AdaptiveBalanceCell> cells;
        const bool populated = pattern == 0 ? comm->getRank() == 0 : (pattern == 1 ? comm->getRank() < 2 : true);
        const auto count = pattern == 2 ? 17 * (comm->getRank() + 1) : 65 - 16 * comm->getRank();
        if (populated)
            for (std::uint64_t i = 0; i < std::uint64_t(count); ++i)
                cells.push_back({key(1000 * std::uint64_t(comm->getRank()) + i), 0,
                    {pattern == 2 ? double(i % 7) / 10 : double(i % 7), 0, 0, 64, AdaptiveWorkUnit::Operations}});
        double local = 0;
        for (const auto& cell : cells)
            local += cell.work.total();
        std::vector<double> loads(comm->getSize()), budgets(comm->getSize());
        Teuchos::gatherAll(*comm, 1, &local, int(loads.size()), loads.data());
        const auto mean = std::accumulate(loads.begin(), loads.end(), 0.0) / loads.size();
        // Independent reference retains the former rank-wide greedy allocation.
        for (int donor = 0; donor < comm->getSize(); ++donor)
        {
            double excess = std::max(0.0, loads[donor] - mean);
            for (int receiver = 0; receiver < comm->getSize() && excess > 0; ++receiver)
            {
                if (receiver == donor)
                    continue;
                const auto transfer = std::min(excess, std::max(0.0, mean - loads[receiver]));
                loads[receiver] += transfer;
                loads[donor] -= transfer;
                excess -= transfer;
                if (donor == comm->getRank())
                    budgets[receiver] += transfer;
            }
        }
        std::vector<std::size_t> order(cells.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(
            order.begin(), order.end(), [&](auto a, auto b) { return cells[a].work.total() > cells[b].work.total(); });
        std::vector<int> expected(cells.size(), comm->getRank());
        for (const auto index : order)
        {
            const auto work = cells[index].work.total();
            auto target = std::max_element(budgets.begin(), budgets.end());
            if (*target > 0 && work > 0 && work <= 2 * *target)
            {
                expected[index] = int(target - budgets.begin());
                *target = std::max(0.0, *target - work);
            }
        }
        const auto plan = balance_owned_cells(cells, {}, source(), comm);
        if (!plan.accepted)
            std::fill(expected.begin(), expected.end(), comm->getRank());
        EXPECT_EQ(plan.localdestinations, expected);
        std::vector<int> identity(cells.size(), comm->getRank());
        const auto unchanged = balance_owned_cells(cells, {}, source(), comm, {}, std::span<const int>(identity));
        EXPECT_FALSE(unchanged.accepted);
        EXPECT_EQ(unchanged.localdestinations, identity);
        EXPECT_EQ(unchanged.proposed.objective, unchanged.before.objective);
        EXPECT_EQ(unchanged.proposed.moved_cells, 0);
        if (!cells.empty())
            identity[0] = comm->getSize();
        EXPECT_THROW(
            balance_owned_cells(cells, {}, source(), comm, {}, std::span<const int>(identity)), std::invalid_argument);
    }
}
TEST(DistributedAdaptiveBalance, EmptyBalancedOwnersNeedNoDestinationBudgetStorage)
{
    using namespace SimpleFluid::Meshes::AMR;
    AdaptiveBalanceOptions options;
    options.byte_budget = std::max<std::size_t>(192, sizeof(DistributedAncestryDirectory));
    const auto plan = balance_owned_cells({}, {}, source(), Tpetra::getDefaultComm(), options);
    EXPECT_FALSE(plan.accepted);
    EXPECT_TRUE(plan.localdestinations.empty());
    EXPECT_EQ(plan.before.objective, 0);
    EXPECT_EQ(plan.proposed.moved_cells, 0);
}
TEST(DistributedAdaptiveBalance, SyntheticCommunicationAndMigrationObjectiveOracle)
{
    auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() == 1)
        GTEST_SKIP();
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBalanceCell> cells;
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBalanceEdge> edges;
    std::vector<int> proposal;
    if (comm->getRank() < 2)
    {
        cells.push_back(
            {key(comm->getRank()), 0, {1, 0, 0, 256, ::SimpleFluid::Meshes::AMR::AdaptiveWorkUnit::Operations}});
        proposal.push_back(0);
    }
    if (comm->getRank() == 0)
        edges.push_back({{'e'}, key(0), key(1), 0, 1, 100});
    ::SimpleFluid::Meshes::AMR::AdaptiveBalanceOptions options;
    options.communication_weight = 1;
    auto accepted = ::SimpleFluid::Meshes::AMR::balance_owned_cells(
        cells, edges, source(), comm, options, std::span<const int>(proposal));
    EXPECT_TRUE(accepted.accepted);
    EXPECT_EQ(accepted.before.objective, 101);
    EXPECT_EQ(accepted.proposed.objective, 2);
    EXPECT_EQ(accepted.proposed.communication_cut_bytes, 0);
    EXPECT_EQ(accepted.proposed.moved_bytes, 256);
    options.migration_weight = 1;
    auto rejected = ::SimpleFluid::Meshes::AMR::balance_owned_cells(
        cells, edges, source(), comm, options, std::span<const int>(proposal));
    EXPECT_FALSE(rejected.accepted);
    EXPECT_EQ(rejected.proposed.objective, 258);
    EXPECT_EQ(rejected.final.objective, 101);
    for (int rank : rejected.localdestinations)
        EXPECT_EQ(rank, comm->getRank());
    options.migration_weight = .001;
    auto estimated = ::SimpleFluid::Meshes::AMR::balance_owned_cells(
        cells, edges, source(), comm, options, std::span<const int>(proposal));
    ASSERT_TRUE(estimated.accepted);
    auto exact = ::SimpleFluid::Meshes::AMR::qualify_balance_wire_bytes(
        std::move(estimated), comm->getRank() == 1 ? 200000 : 0, comm, options);
    EXPECT_FALSE(exact.accepted);
    EXPECT_TRUE(exact.proposed.wire_bytes_exact);
    EXPECT_EQ(exact.proposed.moved_bytes, 200000);
    for (int rank : exact.localdestinations)
        EXPECT_EQ(rank, comm->getRank());
}
TEST(DistributedAdaptiveBalance, InvalidInputsRejectCollectivelyWithoutSourceChange)
{
    auto comm = Tpetra::getDefaultComm();
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBalanceCell> cells;
    if (comm->getRank() == 0)
        cells.push_back({key(0), 0, {1, 0, 0, 32, ::SimpleFluid::Meshes::AMR::AdaptiveWorkUnit::Operations}});
    std::vector<int> bad(cells.size(), comm->getSize());
    EXPECT_THROW(
        ::SimpleFluid::Meshes::AMR::balance_owned_cells(cells, {}, source(), comm, {}, std::span<const int>(bad)),
        std::invalid_argument);
    if (!cells.empty())
        cells[0].work.reconstruction = -1;
    EXPECT_THROW(::SimpleFluid::Meshes::AMR::balance_owned_cells(cells, {}, source(), comm), std::invalid_argument);
    if (!cells.empty())
        cells[0].work.reconstruction = 1;
    ::SimpleFluid::Meshes::AMR::AdaptiveBalanceOptions tiny;
    tiny.byte_budget = 1;
    EXPECT_THROW(
        ::SimpleFluid::Meshes::AMR::balance_owned_cells(cells, {}, source(), comm, tiny), std::invalid_argument);
    if (comm->getSize() > 1)
    {
        std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBalanceCell> duplicate;
        if (comm->getRank() < 2)
            duplicate.push_back({key(9), 0, {1, 0, 0, 32, ::SimpleFluid::Meshes::AMR::AdaptiveWorkUnit::Operations}});
        EXPECT_THROW(
            ::SimpleFluid::Meshes::AMR::balance_owned_cells(duplicate, {}, source(), comm), std::invalid_argument);
    }
    EXPECT_EQ(cells.size(), comm->getRank() == 0 ? 1 : 0);
}
} // namespace
