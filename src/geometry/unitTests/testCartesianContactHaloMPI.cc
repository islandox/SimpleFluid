/** Distributed Cartesian contact: exact stencil owner resolution and parity with
 * the explicit canonical session on the same block partition. */
#include "cartesian_incidence_test_helpers.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/mesh/AMR/CartesianContactBackend.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include "utils/testing_environment.hh"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <exception>
#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace
{
using namespace cartesian_incidence_test;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
namespace AMR = SimpleFluid::Meshes::AMR;
namespace D = AMR::adaptive_detail;
using Halo = D::DistributedAdaptiveContactHalo;
using Idx = std::array<std::uint64_t, 3>;

constexpr std::uint64_t side = 4; // 4x4x4 = 64 uniform level-0 cells
constexpr std::size_t cap64 = 67'108'864;

AMR::AMRGrid config()
{
    AMR::AMRGrid result;
    result.block_cells = {side, side, side};
    result.max_level = 0;
    return result;
}
struct Slab
{
    std::uint64_t begin, end;
};
Slab slab(int rank, int ranks)
{
    const auto base = side / std::uint64_t(ranks), extra = side % std::uint64_t(ranks);
    const auto offset = std::uint64_t(rank);
    const auto begin = offset * base + std::min(offset, extra);
    return {begin, begin + base + (offset < extra ? 1 : 0)};
}
std::vector<AMR::AdaptiveLeaf> range_leaves(const AMR::AMRGrid& grid, std::uint64_t begin, std::uint64_t end)
{
    std::vector<AMR::AdaptiveLeaf> result;
    for (std::uint64_t z = 0; z != side; ++z)
        for (std::uint64_t y = 0; y != side; ++y)
            for (std::uint64_t x = begin; x != end; ++x)
                result.push_back(AMR::cartesian_adaptive_leaf(
                    grid, {grid.region_id, AMR::CartesianCellAncestry{0, {x, y, z}}}, {'d'}));
    return result;
}
std::vector<AMR::AdaptiveLeaf> owned_leaves(const AMR::AMRGrid& grid, int rank, int ranks, bool punch_hole = false)
{
    const auto range = slab(rank, ranks);
    auto result = range_leaves(grid, range.begin, range.end);
    if (punch_hole && rank == 0 && !result.empty())
        result.pop_back();
    return result;
}
/** Rank ownership of the canonical slab partition; the hole variant is never
 * consulted because the session must reject it before grading. */
int owner_of(std::uint64_t x, int ranks)
{
    for (int r = 0; r != ranks; ++r)
    {
        const auto range = slab(r, ranks);
        if (range.begin <= x && x < range.end)
            return r;
    }
    return -1;
}
std::vector<std::pair<Idx, Idx>> touching_pairs()
{
    std::set<std::pair<Idx, Idx>> unique;
    for (std::uint64_t z = 0; z != side; ++z)
        for (std::uint64_t y = 0; y != side; ++y)
            for (std::uint64_t x = 0; x != side; ++x)
                for (const auto& offset : D::cartesian_contact_offsets())
                {
                    const auto nz = std::int64_t(z) + offset[2];
                    const auto ny = std::int64_t(y) + offset[1];
                    const auto nx = std::int64_t(x) + offset[0];
                    if (nx < 0 || ny < 0 || nz < 0 || std::uint64_t(nx) >= side || std::uint64_t(ny) >= side ||
                        std::uint64_t(nz) >= side)
                        continue;
                    const Idx first{x, y, z}, second{std::uint64_t(nx), std::uint64_t(ny), std::uint64_t(nz)};
                    unique.emplace(first < second ? std::pair{first, second} : std::pair{second, first});
                }
    return {unique.begin(), unique.end()};
}
std::size_t shared_vertices(const std::pair<Idx, Idx>& pair)
{
    std::size_t result = 1;
    for (std::size_t axis = 0; axis != 3; ++axis)
    {
        const auto a = pair.first[axis], b = pair.second[axis];
        result *= 2 - std::size_t(a > b ? a - b : b - a);
    }
    return result;
}
std::vector<std::vector<char>> encoded_points(const D::ContactIncidenceRow& row)
{
    std::vector<std::vector<char>> result;
    result.reserve(row.points.size());
    for (const auto& point : row.points)
    {
        std::vector<char> bytes;
        PLIC::details::GeometryIdentityAccess::emit_canonical(point.physical, &bytes,
            [](void* context, std::span<const char> span)
            {
                auto& target = *static_cast<std::vector<char>*>(context);
                target.insert(target.end(), span.begin(), span.end());
            });
        PLIC::details::GeometryIdentityAccess::emit_canonical(point.normalized, &bytes,
            [](void* context, std::span<const char> span)
            {
                auto& target = *static_cast<std::vector<char>*>(context);
                target.insert(target.end(), span.begin(), span.end());
            });
        result.push_back(std::move(bytes));
    }
    return result;
}
using ExportedRows = std::map<AMR::AdaptiveCellKey, std::vector<std::vector<char>>>;
ExportedRows exported_rows(const Halo& halo, std::size_t& count)
{
    ExportedRows result;
    count = 0;
    halo.for_each_incidence_row(
        [&](const D::ContactIncidenceRow& row)
        {
            ++count;
            EXPECT_EQ(row.points.size(), 8u);
            result.emplace(row.key, encoded_points(row));
        });
    return result;
}
struct Census
{
    std::size_t payload_bytes = 0, row_bytes = 0;
};
Census contact_census(const Halo& halo)
{
    Census result;
    halo.for_each_borrowed_contact_payload(
        [&](const D::ContactPayloadAllocation& allocation)
        {
            result.payload_bytes += allocation.payload_bytes;
            if (allocation.identity.role == D::GeometryContactPayloadRole::ContactIncidenceRowObject ||
                allocation.identity.role == D::GeometryContactPayloadRole::ContactIncidenceRowDomain)
                result.row_bytes += allocation.payload_bytes;
        });
    return result;
}
D::ContactOwnedLeafLookup root_lookup(const std::vector<AMR::AdaptiveLeaf>& leaves)
{
    return [&leaves](const AMR::AdaptiveCellKey& key) -> const AMR::AdaptiveLeaf&
    {
        const auto found = std::ranges::find(leaves, key, &AMR::AdaptiveLeaf::key);
        if (found == leaves.end())
            throw std::invalid_argument("Fixture lookup reached a nonowner.");
        return *found;
    };
}

void expect_rollback(Halo& halo, const std::vector<AMR::AdaptiveLeaf>& owned,
    std::span<const D::ContactRetiredLeaf> retired, const D::ContactOwnedLeafLookup& lookup,
    bool require_allocation_history = false)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto generation = halo.accepted_contact_generation();
    const auto admitted = halo.allocation().used;
    const auto before = halo.statistics();
    std::size_t count = 0;
    const auto rows = exported_rows(halo, count);
    const auto history_before_attempt = halo.allocation().cumulative;
    const auto peak_before_attempt = halo.allocation().peak;
    int rejected = 0, all_rejected = 0;
    try
    {
        halo.apply_delta(retired, {}, {7, 11, 13, 1}, lookup);
    }
    catch (const std::exception&)
    {
        rejected = 1;
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &rejected, &all_rejected);
    EXPECT_EQ(all_rejected, comm->getSize());
    EXPECT_EQ(halo.accepted_contact_generation(), generation);
    EXPECT_EQ(halo.allocation().used, admitted);
    EXPECT_GE(halo.allocation().cumulative, history_before_attempt);
    EXPECT_GE(halo.allocation().peak, peak_before_attempt);
    if (require_allocation_history)
        EXPECT_GT(halo.allocation().cumulative, history_before_attempt);
    EXPECT_TRUE(halo.uses_cartesian_contact());
    const auto after = halo.statistics();
    EXPECT_EQ(after.owned_records, before.owned_records);
    EXPECT_EQ(after.unique_cell_pairs, before.unique_cell_pairs);
    EXPECT_EQ(after.work, before.work);
    EXPECT_EQ(after.owner_query_pages, before.owner_query_pages);
    EXPECT_EQ(after.owner_reply_pages, before.owner_reply_pages);
    EXPECT_EQ(after.full_owned_scans_after_bootstrap, before.full_owned_scans_after_bootstrap);
    EXPECT_EQ(exported_rows(halo, count), rows);
    EXPECT_EQ(count, owned.size());
    halo.for_each_incidence_row(
        [&](const D::ContactIncidenceRow& row) { EXPECT_EQ(row.incidence_generation, generation); });
}

std::uint64_t packet_word(const AMR::AdaptiveBytes& bytes, std::size_t offset)
{
    std::uint64_t result = 0;
    for (unsigned i = 0; i != 8; ++i)
        result |= std::uint64_t(static_cast<unsigned char>(bytes.at(offset + i))) << (8 * i);
    return result;
}
void set_packet_word(AMR::AdaptiveBytes& bytes, std::size_t offset, std::uint64_t value)
{
    for (unsigned i = 0; i != 8; ++i)
        bytes.at(offset + i) = char(value >> (8 * i));
}
} // namespace

TEST(CartesianContactHaloMPITest, RankLocalStaleRetirementRejectsWithoutMutation)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto value = config();
    const auto kernel = AMR::cartesian_adaptive_kernel(value);
    const auto owned = owned_leaves(value, comm->getRank(), comm->getSize());
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo halo(*kernel, comm, options, {7, 11, 13, 0});
    halo.bootstrap_canonical(owned, root_lookup(owned));
    std::vector<D::ContactRetiredLeaf> retired;
    if (comm->getRank() == 0)
        retired.push_back({owned.front().key, owned.front().revision + 1});
    expect_rollback(halo, owned, retired, root_lookup(owned));
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}

TEST(CartesianContactHaloMPITest, RankLocalReplacementLookupFailureRejectsWithoutMutation)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto value = config();
    const auto kernel = AMR::cartesian_adaptive_kernel(value);
    const auto owned = owned_leaves(value, comm->getRank(), comm->getSize());
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo halo(*kernel, comm, options, {7, 11, 13, 0});
    halo.bootstrap_canonical(owned, root_lookup(owned));
    const auto normal = root_lookup(owned);
    expect_rollback(halo, owned, {},
        [&](const AMR::AdaptiveCellKey& key) -> const AMR::AdaptiveLeaf&
        {
            if (comm->getRank() == 0)
                throw std::bad_alloc();
            return normal(key);
        });
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}

TEST(CartesianContactHaloMPITest, RankLocalPreparationFailureRejectsWithoutMutation)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto value = config();
    const auto kernel = AMR::cartesian_adaptive_kernel(value);
    const auto owned = owned_leaves(value, comm->getRank(), comm->getSize());
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo halo(*kernel, comm, options, {7, 11, 13, 0});
    halo.bootstrap_canonical(owned, root_lookup(owned));
    halo.set_cartesian_fault_hook(
        [&](D::CartesianContactStage stage)
        {
            if (comm->getRank() == 0 && stage == D::CartesianContactStage::PrepareReplacement)
                throw std::bad_alloc();
        });
    expect_rollback(halo, owned, {}, root_lookup(owned));
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}

TEST(CartesianContactHaloMPITest, PublicationAllocationFailurePreservesAcceptedSnapshot)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto value = config();
    const auto kernel = AMR::cartesian_adaptive_kernel(value);
    const auto owned = owned_leaves(value, comm->getRank(), comm->getSize());
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo halo(*kernel, comm, options, {7, 11, 13, 0});
    halo.bootstrap_canonical(owned, root_lookup(owned));
    const auto generation = halo.accepted_contact_generation();
    halo.set_cartesian_fault_hook(
        [&](D::CartesianContactStage stage)
        {
            if (comm->getRank() == 0 && stage == D::CartesianContactStage::PreparePublication)
                throw std::bad_alloc();
        });
    expect_rollback(halo, owned, {}, root_lookup(owned), true);
    halo.set_cartesian_fault_hook({});
    // Reusing the same next context proves a failed publication did not advance it.
    halo.apply_delta({}, {}, {7, 11, 13, 1}, root_lookup(owned));
    EXPECT_EQ(halo.accepted_contact_generation(), generation + 1);
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}

TEST(CartesianContactHaloMPITest, DroppedTerminalReplyRejectsCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() == 1)
        GTEST_SKIP() << "A remote terminal reply requires multiple ranks.";
    const auto value = config();
    const auto kernel = AMR::cartesian_adaptive_kernel(value);
    const auto owned = owned_leaves(value, comm->getRank(), comm->getSize());
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    bool dropping = false;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo halo(*kernel, comm, options, {7, 11, 13, 0},
        [&](const D::ContactHaloContext&, D::ContactHaloMessageKind kind,
            std::vector<SimpleFluid::OwnerPacket>& packets)
        {
            if (dropping && comm->getRank() == 0 && kind == D::ContactHaloMessageKind::CellOwnerReply &&
                !packets.empty() && packet_word(packets.front().bytes, 96))
                packets.clear();
        });
    halo.bootstrap_canonical(owned, root_lookup(owned));
    dropping = true;
    expect_rollback(halo, owned, {}, root_lookup(owned));
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}

TEST(CartesianContactHaloMPITest, DroppedTerminalProbeRejectsCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() == 1)
        GTEST_SKIP() << "A remote terminal probe requires multiple ranks.";
    const auto value = config();
    const auto kernel = AMR::cartesian_adaptive_kernel(value);
    const auto owned = owned_leaves(value, comm->getRank(), comm->getSize());
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    bool dropping = false;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo halo(*kernel, comm, options, {7, 11, 13, 0},
        [&](const D::ContactHaloContext&, D::ContactHaloMessageKind kind,
            std::vector<SimpleFluid::OwnerPacket>& packets)
        {
            if (dropping && comm->getRank() == 0 && kind == D::ContactHaloMessageKind::CellOwners && !packets.empty() &&
                packet_word(packets.front().bytes, 96))
                packets.clear();
        });
    halo.bootstrap_canonical(owned, root_lookup(owned));
    dropping = true;
    expect_rollback(halo, owned, {}, root_lookup(owned));
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}

TEST(CartesianContactHaloMPITest, RankLocalOwnerInsertionFailureRejectsWithoutMutation)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() == 1)
        GTEST_SKIP() << "A remote insertion requires multiple ranks.";
    const auto value = config();
    const auto kernel = AMR::cartesian_adaptive_kernel(value);
    const auto owned = owned_leaves(value, comm->getRank(), comm->getSize());
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo halo(*kernel, comm, options, {7, 11, 13, 0});
    halo.bootstrap_canonical(owned, root_lookup(owned));
    halo.set_cartesian_fault_hook(
        [&](D::CartesianContactStage stage)
        {
            if (comm->getRank() == 0 && stage == D::CartesianContactStage::BeforeOwnerInsert)
                throw std::bad_alloc();
        });
    expect_rollback(halo, owned, {}, root_lookup(owned));
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}

TEST(CartesianContactHaloMPITest, ConflictingClaimForLocallyOwnedCellRejectsCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() == 1)
        GTEST_SKIP() << "A conflicting remote claim requires multiple ranks.";
    const auto value = config();
    const auto kernel = AMR::cartesian_adaptive_kernel(value);
    const auto owned = owned_leaves(value, comm->getRank(), comm->getSize());
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    bool injecting = false;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo halo(*kernel, comm, options, {7, 11, 13, 0},
        [&](const D::ContactHaloContext&, D::ContactHaloMessageKind kind,
            std::vector<SimpleFluid::OwnerPacket>& packets)
        {
            if (!injecting || comm->getRank() != 0 || kind != D::ContactHaloMessageKind::CellOwnerReply ||
                packets.empty() || !packet_word(packets.front().bytes, 104))
                return;
            auto& packet = packets.front();
            set_packet_word(packet.bytes, 112, slab(packet.peer, comm->getSize()).begin);
            set_packet_word(packet.bytes, 120, 0);
            set_packet_word(packet.bytes, 128, 0);
        });
    halo.bootstrap_canonical(owned, root_lookup(owned));
    injecting = true;
    expect_rollback(halo, owned, {}, root_lookup(owned));
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}

TEST(CartesianContactHaloMPITest, DuplicateOwnerClaimRejectsCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() == 1)
        GTEST_SKIP() << "A remote owner claim requires multiple ranks.";
    const auto value = config();
    const auto kernel = AMR::cartesian_adaptive_kernel(value);
    const auto owned = owned_leaves(value, comm->getRank(), comm->getSize());
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    bool injecting = false;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo halo(*kernel, comm, options, {7, 11, 13, 0},
        [&](const D::ContactHaloContext&, D::ContactHaloMessageKind kind,
            std::vector<SimpleFluid::OwnerPacket>& packets)
        {
            if (!injecting || comm->getRank() != 0 || kind != D::ContactHaloMessageKind::CellOwnerReply ||
                packets.empty())
                return;
            auto& bytes = packets.front().bytes;
            const auto count = packet_word(bytes, 104);
            if (!count)
                return;
            const std::array<char, 24> repeated = [&]
            {
                std::array<char, 24> result{};
                std::copy_n(bytes.begin() + 112, 24, result.begin());
                return result;
            }();
            bytes.insert(bytes.end(), repeated.begin(), repeated.end());
            set_packet_word(bytes, 104, count + 1);
        });
    halo.bootstrap_canonical(owned, root_lookup(owned));
    injecting = true;
    expect_rollback(halo, owned, {}, root_lookup(owned));
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}

TEST(CartesianContactHaloMPITest, EmptyOwnersParticipateInTerminalProtocol)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto value = config();
    const auto kernel = AMR::cartesian_adaptive_kernel(value);
    const auto owned = range_leaves(value, 0, comm->getRank() == 0 ? side : 0);
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo halo(*kernel, comm, options, {7, 11, 13, 0});
    halo.bootstrap_canonical(owned, root_lookup(owned));
    EXPECT_TRUE(halo.uses_cartesian_contact());
    EXPECT_EQ(halo.statistics().owned_records, owned.size());
    EXPECT_TRUE(halo.required_refinements().empty());
    if (comm->getRank() != 0)
    {
        EXPECT_EQ(halo.statistics().unique_cell_pairs, 0u);
        EXPECT_EQ(halo.statistics().resolved_remote_cells, 0u);
    }
    halo.apply_delta({}, {}, {7, 11, 13, 1}, root_lookup(owned));
    EXPECT_EQ(halo.accepted_contact_generation(), 2u);
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}

namespace
{
void check_distributed_census(std::uint64_t n, bool compare_explicit)
{
    const auto comm = Tpetra::getDefaultComm();
    const int ranks = comm->getSize(), rank = comm->getRank();
    auto value = config();
    value.block_cells = {n, n, n};
    const auto kernel = AMR::cartesian_adaptive_kernel(value);
    std::vector<AMR::AdaptiveLeaf> owned;
    for (std::uint64_t z = 0; z < n; ++z)
        for (std::uint64_t y = 0; y < n; ++y)
            for (std::uint64_t x = n * rank / ranks; x < n * (rank + 1) / ranks; ++x)
                owned.push_back(AMR::cartesian_adaptive_leaf(
                    value, {value.region_id, AMR::CartesianCellAncestry{0, {x, y, z}}}, {'d'}));
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo accepted(*kernel, comm, options, {7, 11, 13, 0});
    accepted.bootstrap_canonical(owned, root_lookup(owned));
    ASSERT_TRUE(accepted.uses_cartesian_contact());
    std::map<D::ContactPayloadRole, std::size_t> structural_bytes;
    accepted.for_each_borrowed_contact_payload([&](const D::ContactPayloadAllocation& allocation)
        { structural_bytes[allocation.identity.role] += allocation.payload_bytes; });
    const std::array<D::ContactPayloadRole, 12> regular_roles{D::GeometryContactPayloadRole::ContactIncidenceRowObject,
        D::GeometryContactPayloadRole::ContactIncidenceRowDomain, D::BasicContactPayloadRole::PointArrayPhysical,
        D::BasicContactPayloadRole::PointArrayNormalized, D::GeometryContactPayloadRole::CellPointCounts,
        D::ParallelSupportPayloadRole::FanLocal, D::ParallelSupportPayloadRole::FanRemote,
        D::ParallelSupportPayloadRole::FanBlockObject, D::GeometryContactPayloadRole::TilePageObject,
        D::GeometryContactPayloadRole::TilePageRecords, D::GeometryContactPayloadRole::TilePageRings,
        D::GeometryContactPayloadRole::LineSubscriberValues};
    for (const auto role : regular_roles)
        EXPECT_EQ(structural_bytes[role], 0u);
    EXPECT_EQ(accepted.statistics().point_groups, 0u);
    EXPECT_EQ(accepted.statistics().record_fetches, 0u);
    EXPECT_EQ(accepted.statistics().retained_explicit_line_records, 0u);
    const auto retained = accepted.allocation().used;
    const auto snapshot_pin = accepted.pin_incidence_rows().row(0).snapshot;
    const auto* snapshot = static_cast<const D::CartesianContactState*>(snapshot_pin.get());
    std::size_t payload = 0;
    snapshot->visit_borrowed_payload(
        [&](const D::ContactPayloadAllocation& allocation) { payload += allocation.payload_bytes; });
    EXPECT_EQ(payload + snapshot->admission_allowance_bytes(), snapshot->metadata_bytes());
    auto candidate = accepted.fork_candidate({7, 11, 13, 1});
    const auto private_bytes = accepted.allocation().used - retained;
    auto sibling = accepted.fork_candidate({7, 11, 13, 2});
    const auto family_retained = accepted.allocation().used;
    EXPECT_EQ(&candidate->allocation(), &accepted.allocation());
    EXPECT_EQ(&sibling->allocation(), &accepted.allocation());
    EXPECT_EQ(family_retained, retained + 2 * private_bytes);
    EXPECT_LT(private_bytes, 16 * 1024u);
    EXPECT_EQ(candidate->pin_incidence_rows().row(0).snapshot.get(), snapshot);
    EXPECT_EQ(sibling->pin_incidence_rows().row(0).snapshot.get(), snapshot);
    EXPECT_LT(accepted.allocation().peak, cap64);
    const auto print = std::string("CARTESIAN_CENSUS n=") + std::to_string(n) + " rank=" + std::to_string(rank) +
                       " ranks=" + std::to_string(ranks) + " snapshot_payload=" + std::to_string(payload) +
                       " snapshot_admitted=" + std::to_string(snapshot->metadata_bytes()) +
                       " accepted=" + std::to_string(retained) + " private_candidate=" + std::to_string(private_bytes) +
                       " family=" + std::to_string(family_retained) +
                       " peak=" + std::to_string(accepted.allocation().peak);
    std::cout << print << std::endl;
    std::cout << "CARTESIAN_STRUCTURAL n=" << n << " rank=" << rank << " ranks=" << ranks
              << " regular_role_payload_bytes=0 point_groups=0 fetched_contact_records=0" << std::endl;
    sibling.reset();
    candidate.reset();
    EXPECT_EQ(accepted.allocation().used, retained);
    if (compare_explicit)
    {
        ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
        Halo oracle(*kernel, comm, options, {7, 11, 13, 0});
        oracle.bootstrap_canonical(owned, root_lookup(owned));
        ASSERT_FALSE(oracle.uses_cartesian_contact());
        EXPECT_TRUE(oracle.required_refinements().empty());
        EXPECT_TRUE(accepted.required_refinements().empty());
        EXPECT_EQ(oracle.statistics().unique_cell_pairs, accepted.statistics().unique_cell_pairs);
        EXPECT_EQ(oracle.statistics().logical_contact_pairs, accepted.statistics().logical_contact_pairs);
        std::size_t count = 0;
        const auto expected = exported_rows(oracle, count);
        EXPECT_EQ(exported_rows(accepted, count), expected);
        std::cout << "CARTESIAN_N8_PARITY rank=" << rank << " ranks=" << ranks
                  << " explicit_admitted=" << oracle.cache_storage_bytes() << " cartesian_admitted=" << retained
                  << std::endl;
        ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    }
    std::uint64_t local = family_retained, maximum = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local, &maximum);
    options.byte_budget = maximum - 1; // Identical options; largest owner must fail admission.
    EXPECT_THROW(
        {
            Halo below(*kernel, comm, options, {7, 11, 13, 0});
            below.bootstrap_canonical(owned, root_lookup(owned));
            auto first = below.fork_candidate({7, 11, 13, 1});
            auto second = below.fork_candidate({7, 11, 13, 2});
        },
        std::invalid_argument);
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}
} // namespace

TEST(CartesianContactHaloMPITest, N8FamilyCensusAndExplicitParity)
{
    check_distributed_census(8, true);
}
TEST(CartesianContactHaloMPITest, N32FamilyCensusAndCapRollback)
{
    check_distributed_census(32, false);
}

TEST(CartesianContactHaloMPITest, DistributedStencilMatchesExplicitOracle)
{
    const auto comm = Tpetra::getDefaultComm();
    const int ranks = comm->getSize(), rank = comm->getRank();
    ASSERT_LE(ranks, int(side));
    const auto grid = config();
    const auto kernel = AMR::cartesian_adaptive_kernel(grid);
    const auto owned = owned_leaves(grid, rank, ranks);
    ASSERT_FALSE(owned.empty());
    D::ContactOwnedLeafLookup lookup = [&](const AMR::AdaptiveCellKey& key) -> const AMR::AdaptiveLeaf&
    {
        const auto found = std::ranges::find(owned, key, &AMR::AdaptiveLeaf::key);
        if (found == owned.end())
            throw std::invalid_argument("Fixture lookup reached a nonowner.");
        return *found;
    };
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;

    // Closed-contact oracle: every touching unordered pair is counted by each
    // rank that owns at least one of its cells.
    const auto pairs = touching_pairs();
    std::size_t expected_unique = 0, expected_logical = 0, global_unique = 0, global_logical = 0;
    for (const auto& pair : pairs)
    {
        const auto shared = shared_vertices(pair);
        const bool first_local = owner_of(pair.first[0], ranks) == rank;
        const bool second_local = owner_of(pair.second[0], ranks) == rank;
        global_unique += 1;
        global_logical += shared;
        if (first_local || second_local)
        {
            ++expected_unique;
            expected_logical += shared;
        }
    }

    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo cartesian(*kernel, comm, options, {7, 11, 13, 0});
    cartesian.bootstrap_canonical(owned, lookup);
    ASSERT_TRUE(cartesian.uses_cartesian_contact());
    const auto& statistics = cartesian.statistics();
    EXPECT_EQ(statistics.owned_records, owned.size());
    EXPECT_EQ(statistics.unique_cell_pairs, expected_unique);
    EXPECT_EQ(statistics.logical_contact_pairs, expected_logical);
    EXPECT_EQ(statistics.span_pair_evaluations, expected_unique);
    EXPECT_EQ(statistics.trace_pair_evaluations, expected_unique);
    EXPECT_EQ(statistics.work, 2 * expected_unique);
    if (ranks > 1)
        EXPECT_GT(statistics.resolved_remote_cells, 0u);
    EXPECT_EQ(statistics.owner_query_pages, std::size_t(ranks - 1));
    EXPECT_EQ(statistics.owner_reply_pages, std::size_t(ranks - 1));
    EXPECT_TRUE(cartesian.required_refinements().empty());

    std::size_t cartesian_rows = 0;
    const auto cartesian_export = exported_rows(cartesian, cartesian_rows);
    EXPECT_EQ(cartesian_rows, owned.size());

    // A candidate shares the accepted immutable state without new contact objects.
    const auto candidate = cartesian.fork_candidate({7, 11, 13, 1}, 0);
    EXPECT_TRUE(candidate->required_refinements().empty());
    EXPECT_EQ(candidate->statistics().unique_cell_pairs, statistics.unique_cell_pairs);

    // The explicit canonical session on the same partition is the oracle for
    // per-rank counters, marks and exported rows.
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
    Halo explicit_halo(*kernel, comm, options, {7, 11, 13, 0});
    explicit_halo.bootstrap_canonical(owned, lookup);
    ASSERT_FALSE(explicit_halo.uses_cartesian_contact());
    EXPECT_TRUE(explicit_halo.required_refinements().empty());
    const auto& oracle = explicit_halo.statistics();
    EXPECT_EQ(oracle.owned_records, owned.size());
    EXPECT_EQ(oracle.unique_cell_pairs, statistics.unique_cell_pairs);
    EXPECT_EQ(oracle.logical_contact_pairs, statistics.logical_contact_pairs);
    EXPECT_EQ(oracle.span_pair_evaluations, statistics.span_pair_evaluations);
    EXPECT_EQ(oracle.trace_pair_evaluations, statistics.trace_pair_evaluations);
    EXPECT_EQ(oracle.work, statistics.work);
    std::size_t explicit_rows = 0;
    const auto explicit_export = exported_rows(explicit_halo, explicit_rows);
    EXPECT_EQ(explicit_rows, cartesian_rows);
    EXPECT_EQ(explicit_export.size(), cartesian_export.size());
    for (const auto& [key, points] : cartesian_export)
    {
        const auto found = explicit_export.find(key);
        ASSERT_TRUE(found != explicit_export.end());
        EXPECT_EQ(found->second, points);
    }

    // The contact graph is never materialized on the Cartesian path.
    EXPECT_LT(cartesian.cache_storage_bytes(), explicit_halo.cache_storage_bytes());
    const auto lazy = contact_census(cartesian);
    EXPECT_EQ(lazy.row_bytes, 0u);
    const auto eager = contact_census(explicit_halo);
    EXPECT_GT(eager.row_bytes, 0u);
    EXPECT_LT(lazy.payload_bytes * 8, eager.payload_bytes);
    ::testing::Test::RecordProperty("cartesian_payload_bytes", std::to_string(lazy.payload_bytes));
    ::testing::Test::RecordProperty("explicit_payload_bytes", std::to_string(eager.payload_bytes));
    ::testing::Test::RecordProperty("global_unique_pairs", std::to_string(global_unique));
    ::testing::Test::RecordProperty("global_logical_pairs", std::to_string(global_logical));
    ::testing::Test::RecordProperty("resolved_remote_cells", std::to_string(statistics.resolved_remote_cells));
}

TEST(CartesianContactHaloMPITest, IncompleteGlobalCoverageRejectsCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    const int ranks = comm->getSize(), rank = comm->getRank();
    ASSERT_LE(ranks, int(side));
    const auto grid = config();
    const auto kernel = AMR::cartesian_adaptive_kernel(grid);
    const auto owned = owned_leaves(grid, rank, ranks, true);
    D::ContactOwnedLeafLookup lookup = [&](const AMR::AdaptiveCellKey& key) -> const AMR::AdaptiveLeaf&
    {
        const auto found = std::ranges::find(owned, key, &AMR::AdaptiveLeaf::key);
        if (found == owned.end())
            throw std::invalid_argument("Fixture lookup reached a nonowner.");
        return *found;
    };
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo cartesian(*kernel, comm, options, {7, 11, 13, 0});
    EXPECT_THROW(cartesian.bootstrap_canonical(owned, lookup), std::invalid_argument);
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}

/** All geometric answers are the original factory's; only the private source
 * capability is absent, so the halo must elect the explicit backend. */
class ExplicitPath final : public AMR::AdaptiveProviderKernel
{
public:
    explicit ExplicitPath(const AMR::AdaptiveProviderKernel& inner) : inner_(inner) {}
    AMR::AdaptiveBytes descriptor() const override { return inner_.descriptor(); }
    SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary> geometry(
        const AMR::AdaptiveLeaf& leaf) const override
    {
        return inner_.geometry(leaf);
    }
    std::vector<AMR::AdaptiveSurface> surfaces(const AMR::AdaptiveLeaf& leaf) const override
    {
        return inner_.surfaces(leaf);
    }
    AMR::AdaptiveSplitResult split(const AMR::AdaptiveLeaf& leaf, AMR::AdaptiveSplit kind) const override
    {
        return inner_.split(leaf, kind);
    }
    AMR::AdaptiveTrace traces(const AMR::AdaptiveLeaf& leaf) const override { return inner_.traces(leaf); }
    AMR::AdaptiveContactCapabilities contact_capabilities(const AMR::AdaptiveLeaf& leaf) const override
    {
        return inner_.contact_capabilities(leaf);
    }
    std::vector<AMR::AdaptiveContactImage> closed_contact_images(const AMR::AdaptiveLeaf& leaf) const override
    {
        return inner_.closed_contact_images(leaf);
    }
    SimpleFluid::PLIC::PointIdentity<SimpleFluid::ArithmeticMode::Ordinary> contact_point(const AMR::AdaptiveLeaf& leaf,
        const SimpleFluid::PLIC::PointIdentity<SimpleFluid::ArithmeticMode::Ordinary>& point) const override
    {
        return inner_.contact_point(leaf, point);
    }

private:
    const AMR::AdaptiveProviderKernel& inner_;
};

TEST(CartesianContactHaloMPITest, MissingCapabilityKeepsEveryRankOnTheExplicitBackend)
{
    const auto comm = Tpetra::getDefaultComm();
    const int ranks = comm->getSize(), rank = comm->getRank();
    ASSERT_LE(ranks, int(side));
    const auto grid = config();
    const auto factory = AMR::cartesian_adaptive_kernel(grid);
    const ExplicitPath kernel(*factory);
    const auto owned = owned_leaves(grid, rank, ranks);
    D::ContactOwnedLeafLookup lookup = [&](const AMR::AdaptiveCellKey& key) -> const AMR::AdaptiveLeaf&
    {
        const auto found = std::ranges::find(owned, key, &AMR::AdaptiveLeaf::key);
        if (found == owned.end())
            throw std::invalid_argument("Fixture lookup reached a nonowner.");
        return *found;
    };
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo halo(kernel, comm, options, {7, 11, 13, 0});
    halo.bootstrap_canonical(owned, lookup);
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
    // The election fell back collectively: no owner probes and no remote stencil.
    const auto& statistics = halo.statistics();
    EXPECT_EQ(statistics.owned_records, owned.size());
    EXPECT_EQ(statistics.owner_query_pages, 0u);
    EXPECT_EQ(statistics.resolved_remote_cells, 0u);
    EXPECT_TRUE(halo.required_refinements().empty());
    std::size_t exported = 0;
    halo.for_each_incidence_row([&](const D::ContactIncidenceRow&) { ++exported; });
    EXPECT_EQ(exported, owned.size());
}

TEST(CartesianContactHaloMPITest, OwnershipMigrationRepublishesTheCertifiedState)
{
    const auto comm = Tpetra::getDefaultComm();
    const int ranks = comm->getSize(), rank = comm->getRank();
    if (ranks != 2)
        GTEST_SKIP() << "The migration fixture keeps both owners non-empty at two ranks.";
    const auto grid = config();
    const auto kernel = AMR::cartesian_adaptive_kernel(grid);
    const auto before = owned_leaves(grid, rank, ranks);
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo halo(*kernel, comm, options, {7, 11, 13, 0});
    halo.bootstrap_canonical(before, root_lookup(before));
    const auto before_statistics = halo.statistics();

    // Rank zero hands its x=1 plane to rank one: a same-level ownership delta
    // with no cell added or removed globally.
    const Slab after{rank == 0 ? 0u : 1u, rank == 0 ? 1u : std::uint64_t(side)};
    const auto owned = range_leaves(grid, after.begin, after.end);
    std::vector<D::ContactRetiredLeaf> retired;
    for (const auto& leaf : before)
        if (std::ranges::find(owned, leaf.key, &AMR::AdaptiveLeaf::key) == owned.end())
            retired.push_back({leaf.key, leaf.revision});
    std::vector<AMR::AdaptiveCellKey> inserted;
    for (const auto& leaf : owned)
        if (std::ranges::find(before, leaf.key, &AMR::AdaptiveLeaf::key) == before.end())
            inserted.push_back(leaf.key);
    halo.apply_delta(retired, inserted, {7, 11, 13, 1}, root_lookup(owned));
    const auto& migrated = halo.statistics();
    EXPECT_EQ(migrated.owned_records, owned.size());
    EXPECT_EQ(migrated.delta_owned_visits, inserted.size());
    EXPECT_EQ(migrated.full_owned_scans_after_bootstrap, 1u);
    EXPECT_EQ(migrated.resolved_remote_cells, before_statistics.resolved_remote_cells);
    EXPECT_TRUE(halo.required_refinements().empty());

    // The migrated accepted state matches a fresh session on the same partition.
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
    Halo fresh(*kernel, comm, options, {7, 11, 13, 1});
    fresh.bootstrap_canonical(owned, root_lookup(owned));
    ASSERT_FALSE(fresh.uses_cartesian_contact());
    EXPECT_TRUE(fresh.required_refinements().empty());
    const auto& fresh_statistics = fresh.statistics();
    EXPECT_EQ(fresh_statistics.unique_cell_pairs, migrated.unique_cell_pairs);
    EXPECT_EQ(fresh_statistics.logical_contact_pairs + before_statistics.logical_contact_pairs,
        migrated.logical_contact_pairs);
    EXPECT_EQ(fresh_statistics.span_pair_evaluations + before_statistics.span_pair_evaluations,
        migrated.span_pair_evaluations);
    EXPECT_EQ(fresh_statistics.trace_pair_evaluations + before_statistics.trace_pair_evaluations,
        migrated.trace_pair_evaluations);
    EXPECT_EQ(fresh_statistics.work + before_statistics.work, migrated.work);
    std::size_t migrated_rows = 0;
    const auto migrated_export = exported_rows(halo, migrated_rows);
    std::size_t fresh_rows = 0;
    const auto fresh_export = exported_rows(fresh, fresh_rows);
    EXPECT_EQ(migrated_rows, owned.size());
    EXPECT_EQ(migrated_rows, fresh_rows);
    EXPECT_EQ(migrated_export, fresh_export);
    EXPECT_EQ(contact_census(halo).row_bytes, 0u);
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}
