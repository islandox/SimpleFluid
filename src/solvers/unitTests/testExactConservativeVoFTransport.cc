/** @file testExactConservativeVoFTransport.cc
 * @brief Direct passive-tracer bounds and local momentum-ledger checks before projection.
 */
#include "FVM/AdaptiveVoF.hh"
#include "geometry/PLIC/SweptGeometry.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include "solvers/details/ConservativeVoFTransport.hh"
#include "solvers/details/GeneralVoFExactStorage.hh"
#include "solvers/details/GeneralVoFTransferBinding.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace
{
using namespace SimpleFluid;
using LO = DefaultTpetraTypes::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

SP<const MeshHandle<>> preparation_pair_mesh()
{
    namespace AMR = Meshes::AMR;
    AMR::ExtrudedAMRGrid grid;
    for (unsigned y = 0; y < 2; ++y)
        for (unsigned x = 0; x < 3; ++x)
            grid.xy_nodes.push_back({double(x), double(y), 0});
    grid.base_cells = {{0, 1, 4, 3}, {1, 2, 5, 4}};
    grid.z_edges = {0, 1};
    AMR::ExtrudedAMR source(grid);
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    for (std::uint64_t c = 0; c < 2; ++c)
        if (int(c % std::min(2, comm->getSize())) == comm->getRank())
            owned.push_back(c);
    auto boot = source.detach_owned(owned, {'p', 'r', 'e', 'p'});
    const auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(boot.owned), std::move(boot.families), boot.kernel, comm);
    return AMR::assemble_adaptive_native(*hierarchy, hierarchy->configuration_descriptor()).mesh;
}


TEST(ExactConservativeVoFTransportTest, ExactOppositeGroupsRetainMicroscopicNetBeforeImplicitTrace)
{
    using GO = DefaultTpetraTypes::global_ordinal_type;
    using Measure = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
    using Transfer = detail::ConservativeVoFTransport::Transfer;
    using Stream = detail::ConservativeVoFTransport::FaceContribution;
    SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 2, 2};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const SimpleFluid::Meshes::AMR::CartesianAMR>(grid));
    const auto mesh = geometry.mesh();
    const auto n = mesh->num_owned_cells();
    std::array<GO, 16> local_ids, ids;
    local_ids.fill(-1);
    const auto index = [](const auto& x)
    { return int(std::floor(x.x)) + 4 * int(std::floor(x.y)) + 8 * int(std::floor(x.z)); };
    for (std::size_t c = 0; c < n; ++c)
        local_ids[index(mesh->cell_centroid(LO(c)))] = mesh->cell_global_id(LO(c));
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 16, local_ids.data(), ids.data());
    constexpr double tiny = 1e-310;
    const auto forward = Measure::from_value(1), reverse = forward - Measure::from_value(tiny);
    detail::ConservativeVoFTransport transport(mesh);
    std::vector<double> old(n), capacity(n, 1), outgoing(n), source(n);
    std::vector<Transfer> incoming;
    std::vector<Stream> streams;
    for (std::size_t c = 0; c < n; ++c)
    {
        const auto x = mesh->cell_centroid(LO(c));
        const int ix = int(std::floor(x.x));
        old[c] = ix == 0 ? 1 : ix == 1 ? .5 : 0;
        if (ix == 0)
            outgoing[c] = tiny;
        if (ix == 2)
        {
            capacity[c] = 2 * tiny;
            incoming.push_back({ids[index(x) - 2], mesh->cell_global_id(LO(c)), tiny});
        }
    }
    for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
    {
        const auto x = mesh->face_centroid(LO(f));
        const auto normal = mesh->face_normal(LO(f));
        if (std::abs(normal.x) > .5 && (std::abs(x.x - 1) < 1e-12 || std::abs(x.x - 2) < 1e-12))
        {
            const int donor = 4 * int(std::floor(x.y)) + 8 * int(std::floor(x.z));
            streams.push_back({LO(f), ids[donor], normal.x, std::make_shared<const Measure>(forward.scaled(normal.x))});
            streams.push_back(
                {LO(f), ids[donor], -normal.x, std::make_shared<const Measure>(reverse.scaled(-normal.x))});
        }
    }
    const auto result =
        transport.solve_transfers(old, source, capacity, outgoing, incoming, streams, true, 3000, 1e-12);
    for (std::size_t c = 0; c < n; ++c)
        if (int(std::floor(mesh->cell_centroid(LO(c)).x)) == 2)
            EXPECT_NEAR(result.amount[c] / capacity[c], .5, 1e-12);
    for (const auto& stream : streams)
    {
        const double sign = mesh->face_normal(stream.face).x;
        EXPECT_EQ(result.flux[stream.face], sign * tiny);
    }
    // The displayed doubles still agree, but this exact net is materially
    // different from the certified microscopic routed amount. Its provenance
    // must reject before a global residual can hide the tiny recipient defect.
    auto inconsistent = streams;
    for (std::size_t i = 0; i < inconsistent.size(); i += 2)
    {
        const double sign = mesh->face_normal(inconsistent[i].face).x;
        inconsistent[i].exact_amount =
            std::make_shared<const Measure>((forward + Measure::from_value(1e-20)).scaled(sign));
    }
    EXPECT_THROW(
        (void) transport.solve_transfers(old, source, capacity, outgoing, incoming, inconsistent, true, 3000, 1e-12),
        std::runtime_error);
    auto duplicated = incoming;
    duplicated.insert(duplicated.end(), incoming.begin(), incoming.end());
    EXPECT_THROW(
        (void) transport.solve_transfers(old, source, capacity, outgoing, duplicated, streams, true, 3000, 1e-12),
        std::runtime_error);
    EXPECT_NO_THROW(
        (void) transport.solve_transfers(old, source, capacity, outgoing, incoming, streams, true, 3000, 1e-12));
}

TEST(ExactConservativeVoFTransportTest, ExactInputNormalizesBeforeNarrowingAndRecoversFromRangeFailure)
{
    using GO = DefaultTpetraTypes::global_ordinal_type;
    using M = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
    using Transport = detail::ConservativeVoFTransport;
    SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 2, 2};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const SimpleFluid::Meshes::AMR::CartesianAMR>(grid));
    const auto mesh = geometry.mesh();
    const auto n = mesh->num_owned_cells();
    std::array<GO, 16> local, ids;
    local.fill(-1);
    const auto index = [](const auto& x)
    { return int(std::floor(x.x)) + 4 * int(std::floor(x.y)) + 8 * int(std::floor(x.z)); };
    for (std::size_t c = 0; c < n; ++c)
        local[index(mesh->cell_centroid(LO(c)))] = mesh->cell_global_id(LO(c));
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 16, local.data(), ids.data());
    Transport transport(mesh);
    const double eta = std::numeric_limits<double>::denorm_min();
    for (int scenario : {0, 1, 0, 2, 3, 4})
    {
        const auto mu = scenario == 4  ? M::from_value(.1)
                        : scenario < 2 ? M::from_value(eta).divided_by(M::from_value(scenario == 1 ? 3 : 4))
                                       : M::from_value((scenario == 2 ? 3 : 24) * eta);
        std::vector<M> old(n), source(n), capacity(n, M::from_value(1)), outgoing(n);
        std::vector<Transport::ExactTransfer> routes;
        std::vector<Transport::ExactFaceContribution> faces;
        for (std::size_t c = 0; c < n; ++c)
        {
            const auto x = mesh->cell_centroid(LO(c));
            const int ix = int(std::floor(x.x));
            if (ix == 0)
            {
                outgoing[c] = mu;
                capacity[c] = scenario < 2 ? M::from_value(1) - mu : M::from_value((scenario == 2 ? 1 : 8) * eta);
                old[c] = scenario < 2 ? M::from_value(8) : capacity[c];
                if (scenario == 4)
                {
                    capacity[c] = M::from_value(.9);
                    old[c] = M::from_value(.7);
                }
            }
            if (ix == 2)
            {
                capacity[c] = scenario < 2 ? M::from_value(1e-315) + mu : M::from_value((scenario == 2 ? 4 : 32) * eta);
                if (scenario == 4)
                    capacity[c] = M::from_value(1.1);
                routes.push_back({ids[index(x) - 2], mesh->cell_global_id(LO(c)), mu});
            }
        }
        for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
        {
            const auto x = mesh->face_centroid(LO(f));
            const auto normal = mesh->face_normal(LO(f));
            if (std::abs(normal.x) > .5 && (std::abs(x.x - 1) < 1e-12 || std::abs(x.x - 2) < 1e-12))
            {
                const int donor = 4 * int(std::floor(x.y)) + 8 * int(std::floor(x.z));
                faces.push_back({LO(f), ids[donor], mu.scaled(normal.x)});
            }
        }
        if (scenario == 2)
        {
            EXPECT_THROW((void) detail::Exact::ConservativeVoFTransport::solve(
                             transport, old, source, capacity, outgoing, routes, faces, true, 3000),
                std::runtime_error);
            continue;
        }
        if (scenario == 4)
        {
            EXPECT_THROW((void) detail::Exact::ConservativeVoFTransport::solve(
                             transport, old, source, capacity, outgoing, routes, faces, true, 3000, 1e-30),
                std::runtime_error);
            EXPECT_THROW((void) detail::Exact::ConservativeVoFTransport::solve(
                             transport, old, source, capacity, outgoing, routes, faces, true, 3000, 1e-12, 1024),
                std::runtime_error);
            const auto comm = mesh->owned_cell_map()->getComm();
            if (comm->getSize() > 1)
            {
                EXPECT_THROW((void) detail::Exact::ConservativeVoFTransport::solve(transport,
                                 old, source, capacity, outgoing, routes, faces, comm->getRank() == 0, 3000),
                    std::runtime_error);
                EXPECT_THROW((void) detail::Exact::ConservativeVoFTransport::solve(transport, old, source, capacity,
                                 outgoing, routes, faces, true, comm->getRank() == 0 ? 2999 : 3000),
                    std::runtime_error);
                EXPECT_THROW((void) detail::Exact::ConservativeVoFTransport::solve(transport, old, source, capacity,
                                 outgoing, routes, faces, true, 3000, comm->getRank() == 0 ? 1e-11 : 1e-12),
                    std::runtime_error);
            }
        }
        const auto result = detail::Exact::ConservativeVoFTransport::solve(
            transport, old, source, capacity, outgoing, routes, faces, true, 3000);
        EXPECT_TRUE(result.exact_input);
        EXPECT_TRUE(result.flux.empty());
        EXPECT_LE(result.physical_relative_residual, 1e-12);
        if (n > 0)
            EXPECT_FALSE(result.inner_conversion_enclosures.empty());
        for (std::size_t c = 0; c < n; ++c)
        {
            const int ix = int(std::floor(mesh->cell_centroid(LO(c)).x));
            if (ix == 2 && scenario < 4)
                EXPECT_EQ(result.amount[c], (scenario == 1 ? 3 : scenario == 3 ? 6 : 2) * eta);
            if (scenario == 4 && ix == 0)
                EXPECT_NEAR(result.cell_trace[c], .7, 1e-12);
            if (scenario == 4 && ix == 2)
                EXPECT_NEAR(result.cell_trace[c], .07 / 1.1, 1e-12);
            if (ix == 0 && scenario == 3)
                EXPECT_EQ(result.amount[c], 2 * eta);
        }
    }
}

TEST(ExactConservativeVoFTransportTest, ExactStorageDeduplicatesSharedPayloadAndSeparatesDomains)
{
    using Domain = detail::GeneralVoFStorageDomain;
    using M = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
    const auto value = M::from_value(1e-310).squared();
    detail::GeneralVoFExactStorage audit(16384);
    audit.capacity(Domain::Input, 10, sizeof(M));
    audit.measure(Domain::Input, value);
    audit.measure(Domain::Scratch, value);
    audit.measure(Domain::Result, value);
    const auto shared = audit.finish();
    EXPECT_EQ(shared.bytes[0], 10 * sizeof(M) + value.storage_bytes());
    EXPECT_EQ(shared.bytes[1], 0u);
    EXPECT_EQ(shared.bytes[2], 0u);
    audit.release(Domain::Input, value);
    const auto moved = audit.finish();
    EXPECT_EQ(moved.bytes[0], 10 * sizeof(M));
    EXPECT_EQ(moved.bytes[1], value.storage_bytes());
    EXPECT_THROW(audit.capacity(Domain::Scratch, 16384, 1), std::runtime_error);
    EXPECT_EQ(audit.finish().total(), moved.total());
    EXPECT_THROW(audit.release(Domain::Input, value), std::runtime_error);
}

TEST(ExactConservativeVoFTransportTest, DistinctExactPayloadsExhaustCumulativeBudgetWhileSharedValuesFit)
{
    using Domain = detail::GeneralVoFStorageDomain;
    using M = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
    std::vector<M> distinct, shared;
    distinct.reserve(128);
    shared.reserve(128);
    const auto common = M::from_value(1e-310).squared();
    for (unsigned i = 1; i <= 128; ++i)
    {
        distinct.push_back(common.scaled(double(i)));
        shared.push_back(common);
    }
    constexpr std::size_t budget = 16384;
    for (const auto& value : distinct)
        ASSERT_LT(value.storage_bytes(), budget);
    detail::GeneralVoFExactStorage sharing(budget);
    sharing.capacity(Domain::Input, shared.capacity(), sizeof(M));
    for (const auto& value : shared)
        sharing.measure(Domain::Input, value);
    EXPECT_LT(sharing.finish().total(), budget);
    detail::GeneralVoFExactStorage exhausted(budget);
    exhausted.capacity(Domain::Input, distinct.capacity(), sizeof(M));
    EXPECT_THROW(
        {
            for (const auto& value : distinct)
                exhausted.measure(Domain::Input, value);
        },
        std::runtime_error);
    EXPECT_LE(exhausted.finish().total(), budget);
    detail::GeneralVoFExactStorage recovered(256 * 1024);
    recovered.capacity(Domain::Input, distinct.capacity(), sizeof(M));
    for (const auto& value : distinct)
        recovered.measure(Domain::Input, value);
    EXPECT_GT(recovered.finish().total(), budget);
    EXPECT_LT(recovered.finish().total(), 256 * 1024u);
}

TEST(ExactConservativeVoFTransportTest, ExactRawBindingReusesOnlyAuthenticatedLocalCoefficients)
{
    using M = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
    using GO = DefaultTpetraTypes::global_ordinal_type;
    using Transport = detail::ConservativeVoFTransport;
    namespace AMR = Meshes::AMR;
    AMR::ExtrudedAMRGrid grid;
    for (unsigned y = 0; y < 2; ++y)
        for (unsigned x = 0; x < 4; ++x)
            grid.xy_nodes.push_back({double(x), double(y), 0});
    for (unsigned x = 0; x < 3; ++x)
        grid.base_cells.push_back({x, x + 1, x + 5, x + 4});
    grid.z_edges = {0, 1};
    AMR::ExtrudedAMR root(grid);
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    for (std::uint64_t gid = 0; gid < 3; ++gid)
        if (int(gid % std::min(2, comm->getSize())) == comm->getRank())
            owned.push_back(gid);
    auto boot = root.detach_owned(owned, {'b', 'i', 'n', 'd', 'i', 'n', 'g'});
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(boot.owned), std::move(boot.families), boot.kernel, comm);
    const auto native = AMR::assemble_adaptive_native(*hierarchy, hierarchy->configuration_descriptor());
    const auto mesh = native.mesh;
    // An exact, unrepresentably small non-dyadic body. Its size, rather than
    // a diagnostic floating view, stresses repeated self-route serialization.
    auto positive = M::from_value(1).divided_by(M::from_value(3));
    for (unsigned i = 0; i < 17; ++i)
        positive = positive.squared();
    const auto negative = positive.scaled(-1);
    auto plus = std::make_shared<const M>(positive), minus = std::make_shared<const M>(negative);
    std::vector<Transport::FaceContribution> faces;
    int local_self = 0, local_remote = 0;
    for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
    {
        const auto normal = mesh->face_normal(LO(f));
        if (mesh->is_geometry_exterior_face(LO(f)) || std::abs(normal.x) < .5)
            continue;
        const auto first = mesh->cell_global_id(mesh->owner_cell(LO(f)));
        const auto second = mesh->cell_global_id(mesh->neighbor_cell(LO(f)));
        for (unsigned repeat = 0; repeat < 2; ++repeat)
        {
            faces.push_back({LO(f), first, 0, plus});
            faces.push_back({LO(f), first, 0, minus});
        }
        for (const auto endpoint : {first, second})
            (mesh->owned_cell_map()->getLocalElement(endpoint) >= 0 ? local_self : local_remote)++;
    }
    int global_self = 0, global_remote = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_self, &global_self);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_remote, &global_remote);
    EXPECT_GT(global_self, 0);
    if (comm->getSize() > 1)
        EXPECT_GT(global_remote, 0);
    const auto n = mesh->num_owned_cells();
    int local_empty = n == 0, global_empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &global_empty);
    if (comm->getSize() == 4)
        EXPECT_GT(global_empty, 0);
    std::vector<M> outgoing(n);
    std::vector<std::map<GO, M>> incoming(n);
    constexpr std::size_t budget = 4 * 1024 * 1024;
    const auto verify = [&]
    { return detail::verify_general_vof_transfer_binding(mesh, {}, {}, faces, budget, &outgoing, &incoming, true); };
    const auto rows = verify();
    ASSERT_EQ(rows.size(), n);
    for (const auto& row : rows)
        for (const auto& [donor, coefficient] : row.coefficients)
            EXPECT_EQ(coefficient.sign(), 0);
    if (!outgoing.empty())
        outgoing.front() = M::from_value(1);
    EXPECT_THROW((void) verify(), std::runtime_error);
    for (auto& amount : outgoing)
        amount = {};
    EXPECT_THROW((void) detail::verify_general_vof_transfer_binding(mesh, {}, {}, faces, 1, &outgoing, &incoming, true),
        std::exception);
    EXPECT_NO_THROW((void) verify());
}

TEST(ExactConservativeVoFTransportTest, ExactBindingSharesRemainingCapacityBetweenOperationAndRetainedResult)
{
    using M = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
    using GO = DefaultTpetraTypes::global_ordinal_type;
    using Transport = detail::ConservativeVoFTransport;
    namespace AMR = Meshes::AMR;
    AMR::ExtrudedAMRGrid grid;
    grid.xy_nodes = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}};
    grid.base_cells = {{0, 1, 4, 3}, {1, 2, 5, 4}};
    grid.z_edges = {0, 1};
    AMR::ExtrudedAMR root(grid);
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    if (!comm->getRank())
        owned = {0, 1};
    auto boot = root.detach_owned(owned, {'b', 'i', 'n', 'd', '_', 'l', 'e', 'a', 's', 'e'});
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(boot.owned), std::move(boot.families), boot.kernel, comm);
    const auto mesh = AMR::assemble_adaptive_native(*hierarchy, hierarchy->configuration_descriptor()).mesh;
    auto quantum = M::from_value(1).divided_by(M::from_value(3));
    auto other = M::from_value(1).divided_by(M::from_value(5));
    for (unsigned i = 0; i < 15; ++i)
    {
        quantum = quantum.squared();
        other = other.squared();
    }
    const auto positive = std::make_shared<const M>(quantum);
    const auto second = std::make_shared<const M>(other);
    const auto negative = std::make_shared<const M>((quantum + other).scaled(-1));
    std::vector<Transport::FaceContribution> contributions;
    for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
        if (!mesh->is_geometry_exterior_face(LO(f)))
        {
            const auto donor = mesh->cell_global_id(mesh->owner_cell(LO(f)));
            // Distinct coprime denominators grow the nonzero partial sum before
            // its exact cancellation; shared-body doubling misses that growth.
            contributions.push_back({LO(f), donor, 0, positive});
            contributions.push_back({LO(f), donor, 0, second});
            contributions.push_back({LO(f), donor, 0, negative});
        }
    const auto n = mesh->num_owned_cells();
    int local_empty = n == 0, global_empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &global_empty);
    EXPECT_EQ(global_empty, comm->getSize() - 1);
    std::vector<M> outgoing(n);
    std::vector<std::map<GO, M>> incoming(n);
    constexpr std::size_t budget = 2 * 1024 * 1024;
    const auto verify = [&](std::size_t cap)
    {
        return detail::verify_general_vof_transfer_binding(
            mesh, {}, {}, contributions, cap, &outgoing, &incoming, true);
    };
    const auto rows = verify(budget);
    // Matched old/new admission:640KiB retains the exact result and operation,
    // while the old half-remainder subdivision rejects this same fixed input.
    EXPECT_NO_THROW((void) verify(640 * 1024));
    for (const auto& row : rows)
        for (const auto& [donor, amount] : row.coefficients)
            EXPECT_EQ(amount.sign(), 0);
    if (!outgoing.empty())
        outgoing.front() = quantum;
    EXPECT_THROW((void) verify(budget), std::runtime_error);
    for (auto& amount : outgoing)
        amount = {};
    EXPECT_THROW((void) verify(comm->getRank() == 0 ? 1 : budget), std::exception);
    EXPECT_NO_THROW((void) verify(budget));
}

TEST(ExactConservativeVoFTransportTest, LeasedPhysicalNormSeparatesAggregateOperationAndCommunicationStorage)
{
    using M = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
    namespace E = detail::vof_exact;
    const auto comm = Tpetra::getDefaultComm();
    DefaultTpetraTypes::map_type map(3, 0, comm);
    auto p = M::from_value(1).divided_by(M::from_value(3));
    auto q = M::from_value(1).divided_by(M::from_value(5));
    for (unsigned i = 0; i < 13; ++i)
    {
        p = p.squared();
        q = q.squared();
    }
    std::vector<M> residual(map.getLocalNumElements()), rhs(residual.size());
    for (std::size_t i = 0; i < residual.size(); ++i)
    {
        const auto gid = map.getGlobalElement(LO(i));
        residual[i] = gid == 0 ? p : gid == 1 ? q : M::from_value(1);
    }
    // Original zero-RHS reference is exactly1: 1+p²+q² >1, despite a
    // rounded relative diagnostic of1. No tiny squared defect may disappear.
    constexpr std::size_t budget = 512 * 1024;
    const auto rejected = E::leased_physical_norm(residual, rhs, 1, map, budget);
    EXPECT_FALSE(rejected.accepted);
    EXPECT_DOUBLE_EQ(rejected.relative_residual, 1);
    const auto accepted = E::leased_physical_norm(residual, rhs, 2, map, budget);
    EXPECT_TRUE(accepted.accepted);
    EXPECT_THROW((void) E::physical_norm(residual, rhs, 1, map, budget / 8), std::exception);
    EXPECT_THROW(
        (void) E::leased_physical_norm(residual, rhs, 1, map, comm->getRank() == 0 ? 1 : budget), std::exception);
    EXPECT_FALSE(E::leased_physical_norm(residual, rhs, 1, map, budget).accepted);
    int local_empty = residual.empty(), empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    if (comm->getSize() > 3)
        EXPECT_EQ(empty, comm->getSize() - 3);
}

TEST(ExactConservativeVoFTransportTest, ExactPreparationRequiresSharedProvenanceAndRetainsEveryRejectionGate)
{
    using M = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
    using Transport = detail::ConservativeVoFTransport;
    const auto mesh = preparation_pair_mesh();
    const auto n = mesh->num_owned_cells();
    Transport transport(mesh);
    std::vector<M> old(n, M::from_value(7)), source(n), capacity(n, M::from_value(1)), outgoing(n);
    const auto solve = [&](std::size_t budget = 64 * 1024 * 1024)
    {
        return detail::Exact::ConservativeVoFTransport::solve(
            transport, old, source, capacity, outgoing, {}, {}, true, 3000, 1e-12, budget);
    };
    for (double amount : solve().amount)
        EXPECT_NEAR(amount, 7, 32 * std::numeric_limits<double>::epsilon() * 7);
    EXPECT_EQ(transport.preparation_statistics().exact_builds, 1u);
    EXPECT_EQ(transport.preparation_statistics().exact_hits, 0u);
    for (auto& value : old)
        value = M::from_value(14);
    for (double amount : solve().amount)
        EXPECT_NEAR(amount, 14, 32 * std::numeric_limits<double>::epsilon() * 14);
    EXPECT_EQ(transport.preparation_statistics().exact_hits, 1u);
    EXPECT_THROW((void) solve(1), std::runtime_error);
    EXPECT_NO_THROW((void) solve());
    EXPECT_EQ(transport.preparation_statistics().exact_hits, 2u);
    // A different exact capacity with the same binary64 view must rebuild on
    // every rank, including ranks with no changed or owned row.
    const auto tail = M::from_value(std::numeric_limits<double>::denorm_min()).divided_by(M::from_value(3));
    if (!capacity.empty())
    {
        const auto changed = capacity.front() + tail;
        EXPECT_EQ(double(changed.enclosure().value), double(capacity.front().enclosure().value));
        capacity.front() = changed;
    }
    EXPECT_NO_THROW((void) solve());
    EXPECT_EQ(transport.preparation_statistics().exact_builds, 2u);
    // An unauthenticated outgoing amount cannot publish a partial preparation.
    if (!outgoing.empty())
        outgoing.front() = M::from_value(1);
    EXPECT_THROW((void) solve(), std::runtime_error);
    EXPECT_EQ(transport.preparation_statistics().exact_builds, 2u);
    for (auto& value : outgoing)
        value = {};
    EXPECT_NO_THROW((void) solve());
    EXPECT_EQ(transport.preparation_statistics().exact_builds, 3u);
    EXPECT_NO_THROW((void) solve());
    EXPECT_EQ(transport.preparation_statistics().exact_hits, 3u);
}

TEST(ExactConservativeVoFTransportTest, ExactPreparationReusesAuthenticatedNonzeroRoutesAcrossThreeComponents)
{
    using M = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
    using GO = DefaultTpetraTypes::global_ordinal_type;
    using Transport = detail::ConservativeVoFTransport;
    const auto mesh = preparation_pair_mesh();
    const auto n = mesh->num_owned_cells();
    GO local[2] = {-1, -1}, ids[2]{};
    for (std::size_t c = 0; c < n; ++c)
        local[mesh->cell_centroid(LO(c)).x < 1 ? 0 : 1] = mesh->cell_global_id(LO(c));
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 2, local, ids);
    std::vector<M> old(n), source(n), capacity(n), outgoing(n);
    std::vector<Transport::ExactTransfer> incoming;
    std::vector<Transport::ExactFaceContribution> faces;
    const auto quarter = M::from_value(.25);
    for (std::size_t c = 0; c < n; ++c)
        if (mesh->cell_global_id(LO(c)) == ids[0])
        {
            capacity[c] = M::from_value(.75);
            outgoing[c] = quarter;
        }
        else
        {
            capacity[c] = M::from_value(1.25);
            incoming.push_back({ids[0], ids[1], quarter});
        }
    for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
        if (!mesh->is_geometry_exterior_face(LO(f)))
            faces.push_back({LO(f), ids[0], quarter.scaled(mesh->face_normal(LO(f)).x)});
    Transport transport(mesh);
    for (unsigned component = 1; component <= 3; ++component)
    {
        for (std::size_t c = 0; c < n; ++c)
            old[c] = M::from_value(component * (mesh->cell_global_id(LO(c)) == ids[0] ? 2 : 6));
        const auto result = detail::Exact::ConservativeVoFTransport::solve(
            transport, old, source, capacity, outgoing, incoming, faces, true, 3000);
        for (std::size_t c = 0; c < n; ++c)
            EXPECT_NEAR(result.amount[c], component * (mesh->cell_global_id(LO(c)) == ids[0] ? 1.5 : 6.5), 2e-12);
        EXPECT_EQ(transport.preparation_statistics().exact_builds, 1u);
        EXPECT_EQ(transport.preparation_statistics().exact_hits, component - 1);
    }
    for (std::size_t c = 0; c < n; ++c)
        old[c] = M::from_value(-1);
    EXPECT_THROW((void) detail::Exact::ConservativeVoFTransport::solve(
                     transport, old, source, capacity, outgoing, incoming, faces, true, 3000),
        std::runtime_error);
    EXPECT_EQ(transport.preparation_statistics().exact_hits, 2u);
    for (std::size_t c = 0; c < n; ++c)
        old[c] = M::from_value(mesh->cell_global_id(LO(c)) == ids[0] ? 2 : 6);
    EXPECT_NO_THROW((void) detail::Exact::ConservativeVoFTransport::solve(
        transport, old, source, capacity, outgoing, incoming, faces, true, 3000));
    EXPECT_EQ(transport.preparation_statistics().exact_hits, 3u);
}
} // namespace
