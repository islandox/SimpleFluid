#include "cartesian_incidence_test_helpers.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/mesh/AMR/CartesianContactBackend.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include "utils/testing_environment.hh"
#include <algorithm>
#include <cstdlib>
#include <gtest/gtest.h>
#include <map>
#include <new>
#include <optional>
#include <set>
#include <span>
#include <vector>

namespace cartesian_contact_allocation_probe
{
thread_local bool enabled = false;
thread_local std::size_t allocations = 0;
} // namespace cartesian_contact_allocation_probe
void* operator new(std::size_t bytes)
{
    if (cartesian_contact_allocation_probe::enabled)
        ++cartesian_contact_allocation_probe::allocations;
    if (auto* pointer = std::malloc(bytes ? bytes : 1))
        return pointer;
    throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept
{
    std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept
{
    std::free(pointer);
}

namespace
{
using namespace cartesian_incidence_test;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
constexpr std::size_t cap64 = 67'108'864;
namespace AMR = SimpleFluid::Meshes::AMR;
namespace D = AMR::adaptive_detail;
using Halo = D::DistributedAdaptiveContactHalo;

template<class Row> std::vector<std::vector<char>> encoded_row_points(const Row& row)
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

struct RefinedFrame
{
    std::vector<AMR::AdaptiveLeaf> leaves;
    std::vector<D::ContactRetiredLeaf> retired;
    std::vector<AMR::AdaptiveCellKey> inserted;
};
RefinedFrame refine_frame(const AMR::AdaptiveProviderKernel& kernel, const std::vector<AMR::AdaptiveLeaf>& before)
{
    RefinedFrame result;
    for (const auto& leaf : before)
    {
        result.retired.push_back({leaf.key, leaf.revision});
        for (const auto& child : kernel.split(leaf, AMR::AdaptiveSplit::Isotropic).children)
        {
            result.leaves.push_back(child);
            result.inserted.push_back(child.key);
        }
    }
    return result;
}

TEST(CartesianContactHaloSerialTest, UnchangedQueriesReuseNonzeroCumulativeWork)
{
    const auto value = grid();
    const auto native = leaves(value);
    const auto kernel = cartesian_adaptive_kernel(value);
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo halo(*kernel, Tpetra::getDefaultComm(), options, {7, 11, 13, 0});
    halo.bootstrap_canonical(native, root_lookup(native));
    const auto initial = halo.statistics();
    ASSERT_GT(initial.work, 0u);
    for (unsigned query = 0; query != 3; ++query)
    {
        EXPECT_TRUE(halo.required_refinements().empty());
        const auto current = halo.statistics();
        EXPECT_EQ(current.work, initial.work);
        EXPECT_EQ(current.span_pair_evaluations, initial.span_pair_evaluations);
        EXPECT_EQ(current.trace_pair_evaluations, initial.trace_pair_evaluations);
        EXPECT_EQ(current.cumulative_bytes, initial.cumulative_bytes);
    }
    const auto refined = refine_frame(*kernel, native);
    halo.apply_delta(refined.retired, refined.inserted, {7, 11, 13, 1}, root_lookup(refined.leaves));
    const auto updated = halo.statistics();
    EXPECT_GT(updated.work, initial.work);
    EXPECT_TRUE(halo.required_refinements().empty());
    EXPECT_EQ(halo.statistics().work, updated.work);
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}

TEST(CartesianContactHaloSerialTest, RowConstructionAllocatesNoHeapOrBudget)
{
    const auto value = grid();
    const auto native = leaves(value);
    const auto kernel = cartesian_adaptive_kernel(value);
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo halo(*kernel, Tpetra::getDefaultComm(), options, {7, 11, 13, 0});
    halo.bootstrap_canonical(native, root_lookup(native));
    const auto before = halo.allocation().cumulative;
    cartesian_contact_allocation_probe::allocations = 0;
    cartesian_contact_allocation_probe::enabled = true;
    const auto pinned = halo.pin_incidence_rows();
    std::size_t points = 0, generation = 0;
    bool ordered = true;
    for (std::size_t i = 0; i != pinned.size(); ++i)
    {
        const auto row = pinned.row(i);
        generation = row.incidence_generation;
        points += row.points.size();
        for (std::size_t j = 1; j < row.points.size(); ++j)
            ordered = ordered && row.points[j - 1].physical < row.points[j].physical;
    }
    cartesian_contact_allocation_probe::enabled = false;
    EXPECT_EQ(cartesian_contact_allocation_probe::allocations, 0u);
    EXPECT_EQ(halo.allocation().cumulative - before, 0u);
    EXPECT_EQ(points, 8 * native.size());
    EXPECT_TRUE(ordered);
    EXPECT_EQ(generation, 1u);
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}

TEST(CartesianContactHaloSerialTest, OwningCompatibilityExportAdmitsBeforeAllocationAndRollsBack)
{
    const auto value = grid();
    const auto native = leaves(value);
    const auto kernel = cartesian_adaptive_kernel(value);
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo halo(*kernel, Tpetra::getDefaultComm(), options, {7, 11, 13, 0});
    halo.bootstrap_canonical(native, root_lookup(native));
    auto& budget = halo.allocation();
    const auto retained = budget.used;
    budget.limit = retained + 8 * sizeof(D::ContactPointIdentity); // One domain byte short.
    std::size_t emitted = 0;
    EXPECT_THROW(halo.for_each_incidence_row([&](const D::ContactIncidenceRow&) { ++emitted; }), std::invalid_argument);
    EXPECT_EQ(emitted, 0u);
    EXPECT_EQ(budget.used, retained);
    EXPECT_EQ(halo.accepted_contact_generation(), 1u);
    budget.limit = cap64;
    halo.for_each_incidence_row([&](const D::ContactIncidenceRow&) { ++emitted; });
    EXPECT_EQ(emitted, native.size());
    EXPECT_EQ(budget.used, retained);
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}

TEST(CartesianContactHaloSerialTest, PinnedSnapshotReadAfterEditKeepsOldRows)
{
    const auto value = grid();
    const auto native = leaves(value);
    const auto kernel = cartesian_adaptive_kernel(value);
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo halo(*kernel, Tpetra::getDefaultComm(), options, {7, 11, 13, 0});
    halo.bootstrap_canonical(native, root_lookup(native));
    std::map<AMR::AdaptiveCellKey, std::vector<std::vector<char>>> original;
    halo.for_each_incidence_row(
        [&](const D::ContactIncidenceRow& row) { original.emplace(row.key, encoded_row_points(row)); });
    const auto pin = halo.pin_incidence_rows(); // No row consumed until after publication.
    const auto refined = refine_frame(*kernel, native);
    halo.apply_delta(refined.retired, refined.inserted, {7, 11, 13, 1}, root_lookup(refined.leaves));
    EXPECT_EQ(halo.accepted_contact_generation(), 2u);
    EXPECT_EQ(pin.size(), native.size());
    for (std::size_t i = 0; i < pin.size(); ++i)
    {
        const auto row = pin.row(i);
        ASSERT_NE(row.key, nullptr);
        EXPECT_EQ(row.incidence_generation, 1u);
        EXPECT_EQ(encoded_row_points(row), original.at(*row.key));
    }
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}

TEST(CartesianContactHaloSerialTest, LiveRowAndCopiedPointsSurvivePublication)
{
    const auto value = grid();
    const auto native = leaves(value);
    const auto kernel = cartesian_adaptive_kernel(value);
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo halo(*kernel, Tpetra::getDefaultComm(), options, {7, 11, 13, 0});
    halo.bootstrap_canonical(native, root_lookup(native));
    const auto row = halo.pin_incidence_rows().row(0);
    const auto copied_points = row.points;
    const auto before = encoded_row_points(row);
    const auto refined = refine_frame(*kernel, native);
    halo.apply_delta(refined.retired, refined.inserted, {7, 11, 13, 1}, root_lookup(refined.leaves));
    EXPECT_EQ(encoded_row_points(row), before);
    EXPECT_EQ(row.incidence_generation, 1u);
    EXPECT_EQ(copied_points.size(), 8u);
    EXPECT_EQ(copied_points[7].physical, row.points[7].physical);
    EXPECT_EQ(std::get<AMR::CartesianCellAncestry>(row.key->ancestry).level, 0u);
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}

TEST(CartesianContactHaloSerialTest, PinnedSnapshotSurvivesSourceDestructionBeforeRead)
{
    D::ContactIncidenceRowsView pinned;
    {
        const auto value = grid();
        const auto native = leaves(value);
        const auto kernel = cartesian_adaptive_kernel(value);
        AMR::AdaptiveClosureOptions options;
        options.byte_budget = cap64;
        ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
        Halo halo(*kernel, Tpetra::getDefaultComm(), options, {7, 11, 13, 0});
        halo.bootstrap_canonical(native, root_lookup(native));
        pinned = halo.pin_incidence_rows();
    }
    ASSERT_EQ(pinned.size(), 8u);
    const auto row = pinned.row(0);
    EXPECT_EQ(row.incidence_generation, 1u);
    EXPECT_EQ(row.points.size(), 8u);
    EXPECT_EQ(row.points[0].physical, point(0, 0, 0));
    EXPECT_EQ(row.points[7].physical, point(0.5, 0.5, 0.5));
    EXPECT_EQ(row.domain.size(), 1u);
    EXPECT_EQ(row.domain.front(), 'd');
    const auto* snapshot = static_cast<const D::CartesianContactState*>(row.snapshot.get());
    EXPECT_EQ(
        snapshot->allocation_budget()->used, snapshot->metadata_bytes() + snapshot->allocation_budget()->owner_bytes);
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}

TEST(CartesianContactHaloSerialTest, ResidencyCounterCreditOutlivesControllerAndReleasesWithLease)
{
    const auto value = grid();
    const auto native = leaves(value);
    const auto kernel = cartesian_adaptive_kernel(value);
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    auto accepted =
        std::make_unique<Halo>(*kernel, Tpetra::getDefaultComm(), options, D::ContactHaloContext{7, 11, 13, 0});
    accepted->bootstrap_canonical(native, root_lookup(native));
    auto& arena = accepted->allocation();
    const auto one_controller = arena.used;
    auto sibling = accepted->fork_candidate({7, 11, 13, 1});
    std::optional<D::ContactExternalResidencyLease> lease(accepted->external_residency_lease());
    lease->resize(17);
    lease->resize(0); // Only the counter's metadata remains resident.
    accepted.reset();
    const auto with_counter_pin = arena.used;
    EXPECT_GT(with_counter_pin, one_controller);
    lease.reset();
    EXPECT_LT(arena.used, with_counter_pin);
    EXPECT_EQ(arena.used, one_controller);
    EXPECT_TRUE(sibling->required_refinements().empty());
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}

TEST(CartesianContactHaloSerialTest, PinnedCoordinatesKeepSourceArithmeticAndRestoreCallerEnvironment)
{
    RestoreEnvironment restore;
    auto value = grid();
    value.block_cells = {6, 2, 2};
    std::vector<AMR::AdaptiveLeaf> native;
    for (std::uint64_t z = 0; z < 2; ++z)
        for (std::uint64_t y = 0; y < 2; ++y)
            for (std::uint64_t x = 0; x < 6; ++x)
                native.push_back(
                    cartesian_adaptive_leaf(value, {value.region_id, AMR::CartesianCellAncestry{0, {x, y, z}}}, {'d'}));
    const auto kernel = cartesian_adaptive_kernel(value);
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo halo(*kernel, Tpetra::getDefaultComm(), options, {7, 11, 13, 0});
    halo.bootstrap_canonical(native, root_lookup(native));
    const auto pin = halo.pin_incidence_rows();
    const auto before = encoded_row_points(pin.row(2));
    ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);
    const auto environment_before = cartesian_evaluation_environment();
    EXPECT_EQ(encoded_row_points(pin.row(2)), before);
    EXPECT_EQ(cartesian_evaluation_environment(), environment_before);
    EXPECT_EQ(std::fegetround(), FE_DOWNWARD);
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}

bool cartesian_state_role(D::ContactPayloadRole role)
{
    return role.category() == D::ContactPayloadCategory::Geometry &&
           role.ordinal() >= static_cast<std::uint8_t>(D::GeometryContactPayloadRole::CartesianContactStateObject) &&
           role.ordinal() <= static_cast<std::uint8_t>(D::GeometryContactPayloadRole::CartesianContactMarks);
}
using PayloadUnion = std::map<std::pair<std::uintptr_t, D::ContactPayloadRole>, std::size_t>;
void include_cartesian_payload(const Halo& halo, PayloadUnion& allocations)
{
    halo.for_each_borrowed_contact_payload(
        [&](const D::ContactPayloadAllocation& allocation)
        {
            if (!cartesian_state_role(allocation.identity.role))
                return;
            const auto identity =
                std::pair{reinterpret_cast<std::uintptr_t>(allocation.identity.address), allocation.identity.role};
            const auto entry = allocations.emplace(identity, allocation.payload_bytes).first;
            EXPECT_EQ(entry->second, allocation.payload_bytes);
        });
}
void check_family_census(std::uint64_t side)
{
    auto value = grid();
    value.block_cells = {side, side, side};
    std::vector<AMR::AdaptiveLeaf> native;
    for (std::uint64_t z = 0; z < side; ++z)
        for (std::uint64_t y = 0; y < side; ++y)
            for (std::uint64_t x = 0; x < side; ++x)
                native.push_back(
                    cartesian_adaptive_leaf(value, {value.region_id, AMR::CartesianCellAncestry{0, {x, y, z}}}, {'d'}));
    const auto kernel = cartesian_adaptive_kernel(value);
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo accepted(*kernel, Tpetra::getDefaultComm(), options, {7, 11, 13, 0});
    accepted.bootstrap_canonical(native, root_lookup(native));
    ASSERT_TRUE(accepted.uses_cartesian_contact());
    const auto retained = accepted.allocation().used;
    PayloadUnion accepted_payload;
    include_cartesian_payload(accepted, accepted_payload);
    std::size_t payload_bytes = 0;
    for (const auto& [identity, bytes] : accepted_payload)
        payload_bytes += bytes;
    auto candidate = accepted.fork_candidate({7, 11, 13, 1});
    ASSERT_EQ(&candidate->allocation(), &accepted.allocation());
    const auto with_candidate = accepted.allocation().used;
    const auto candidate_private = with_candidate - retained;
    // Private controller metadata is independent of lattice size; a duplicated
    // snapshot charge is larger than this independently bounded envelope.
    EXPECT_LT(candidate_private, 16 * 1024u);
    auto sibling = accepted.fork_candidate({7, 11, 13, 2});
    ASSERT_EQ(&sibling->allocation(), &accepted.allocation());
    const auto overlap = accepted.allocation().used;
    EXPECT_EQ(overlap - with_candidate, candidate_private);
    PayloadUnion family = accepted_payload;
    include_cartesian_payload(*candidate, family);
    include_cartesian_payload(*sibling, family);
    EXPECT_EQ(family, accepted_payload);
    EXPECT_LT(overlap, cap64);
    const auto snapshot =
        static_cast<const D::CartesianContactState*>(accepted.pin_incidence_rows().row(0).snapshot.get());
    EXPECT_EQ(payload_bytes + snapshot->admission_allowance_bytes(), snapshot->metadata_bytes());
    EXPECT_GE(retained, snapshot->metadata_bytes() + sizeof(D::Budget));
    ::testing::Test::RecordProperty("accepted_retained_bytes", std::to_string(retained));
    ::testing::Test::RecordProperty("snapshot_payload_bytes", std::to_string(payload_bytes));
    ::testing::Test::RecordProperty("snapshot_admitted_bytes", std::to_string(snapshot->metadata_bytes()));
    ::testing::Test::RecordProperty("candidate_private_bytes", std::to_string(candidate_private));
    ::testing::Test::RecordProperty("accepted_candidate_sibling_bytes", std::to_string(overlap));
    ::testing::Test::RecordProperty("bootstrap_and_overlap_peak_bytes", std::to_string(accepted.allocation().peak));
    sibling.reset();
    candidate.reset();
    EXPECT_EQ(accepted.allocation().used, retained);
    options.byte_budget = overlap - 1;
    EXPECT_THROW(
        {
            Halo below(*kernel, Tpetra::getDefaultComm(), options, {7, 11, 13, 0});
            below.bootstrap_canonical(native, root_lookup(native));
            auto first = below.fork_candidate({7, 11, 13, 1});
            auto second = below.fork_candidate({7, 11, 13, 2});
        },
        std::invalid_argument);
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}

TEST(CartesianContactHaloSerialTest, N8FamilyCensusAccountsAcceptedCandidateSiblingOnce)
{
    check_family_census(8);
}
TEST(CartesianContactHaloSerialTest, N32FamilyCensusAccountsAcceptedCandidateSiblingOnce)
{
    check_family_census(32);
}

TEST(CartesianContactHaloSerialTest, OrderedComparisonDetectsPhysicalAndNormalizedPermutation)
{
    D::ContactIncidenceRow original;
    original.points = {{point(0), point(0)}, {point(1), point(1)}};
    auto permuted_normalized = original;
    permuted_normalized.points = {{point(1), point(0)}, {point(0), point(1)}};
    auto permuted_physical = original;
    permuted_physical.points = {{point(0), point(1)}, {point(1), point(0)}};
    EXPECT_NE(encoded_row_points(original), encoded_row_points(permuted_normalized));
    EXPECT_NE(encoded_row_points(original), encoded_row_points(permuted_physical));
}

TEST(CartesianContactHaloSerialTest, SixCellAxisHaloMatchesExplicitForEveryAcceptedGradingOption)
{
    auto value = grid();
    value.block_cells = {6, 2, 2};
    std::vector<AMR::AdaptiveLeaf> native;
    for (std::uint64_t z = 0; z < 2; ++z)
        for (std::uint64_t y = 0; y < 2; ++y)
            for (std::uint64_t x = 0; x < 6; ++x)
                native.push_back(
                    cartesian_adaptive_leaf(value, {value.region_id, AMR::CartesianCellAncestry{0, {x, y, z}}}, {'d'}));
    const auto kernel = cartesian_adaptive_kernel(value);
    const auto marks = [](Halo& halo)
    {
        std::vector<std::pair<AMR::AdaptiveCellKey, AMR::AdaptiveSplit>> result;
        for (const auto& mark : halo.required_refinements())
            result.emplace_back(mark.key, mark.split);
        return result;
    };
    for (const auto ratio : {1.0, 1.25, 2.0, 4.0})
    {
        AMR::AdaptiveClosureOptions options;
        options.byte_budget = cap64;
        options.maximum_span_ratio = ratio;
        options.maximum_trace_jump = 1; // The complete public option set.
        ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
        Halo actual(*kernel, Tpetra::getDefaultComm(), options, {7, 11, 13, 0});
        actual.bootstrap_canonical(native, root_lookup(native));
        ASSERT_TRUE(actual.uses_cartesian_contact());
        ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
        Halo oracle(*kernel, Tpetra::getDefaultComm(), options, {7, 11, 13, 0});
        oracle.bootstrap_canonical(native, root_lookup(native));
        ASSERT_FALSE(oracle.uses_cartesian_contact());
        EXPECT_EQ(marks(actual), marks(oracle)) << "ratio=" << ratio;
        if (ratio == 1)
            EXPECT_FALSE(marks(actual).empty());
        std::map<AMR::AdaptiveCellKey, std::vector<std::vector<char>>> expected;
        oracle.for_each_incidence_row(
            [&](const D::ContactIncidenceRow& row) { expected.emplace(row.key, encoded_row_points(row)); });
        actual.for_each_incidence_row_view(
            [&](const D::ContactIncidenceRowView& row) { EXPECT_EQ(encoded_row_points(row), expected.at(*row.key)); });
    }
}

TEST(CartesianContactHaloSerialTest, BootstrapGradesAndExportsExactlyLikeExplicitOracle)
{
    // The opt-in fixture and explicit oracle use the same authoritative leaves.
    auto value = grid();
    auto leaves_v = leaves(value);
    std::map<AMR::AdaptiveCellKey, const AMR::AdaptiveLeaf*> lookup_map;
    for (const auto& leaf : leaves_v)
        lookup_map.emplace(leaf.key, &leaf);
    D::ContactOwnedLeafLookup lookup = [&](const AMR::AdaptiveCellKey& key) -> const AMR::AdaptiveLeaf&
    { return *lookup_map.at(key); };

    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    const auto comm = Tpetra::getDefaultComm();
    ASSERT_EQ(comm->getSize(), 1);

    // Brute-force oracles for the 2x2x2 nonperiodic box: every distinct pair
    // touches, so unique pairs is C(8,2)=28 and shared vertices per pair follow
    // the index difference product (2 - |d|).
    std::uint64_t expected_unique = 0, expected_incidence = 0;
    for (std::size_t i = 0; i != leaves_v.size(); ++i)
        for (std::size_t j = i + 1; j != leaves_v.size(); ++j)
        {
            ++expected_unique;
            const auto* a = std::get_if<AMR::CartesianCellAncestry>(&leaves_v[i].key.ancestry);
            const auto* b = std::get_if<AMR::CartesianCellAncestry>(&leaves_v[j].key.ancestry);
            std::size_t shared = 1;
            for (std::size_t axis = 0; axis != 3; ++axis)
            {
                const auto delta =
                    a->index[axis] > b->index[axis] ? a->index[axis] - b->index[axis] : b->index[axis] - a->index[axis];
                shared *= 2 - delta;
            }
            expected_incidence += shared;
        }

    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    const auto kernel = cartesian_adaptive_kernel(value);
    Halo cartesian(*kernel, comm, options, {7, 11, 13, 0});
    cartesian.bootstrap_canonical(leaves_v, lookup);
    ASSERT_TRUE(cartesian.uses_cartesian_contact());
    const auto& cartesian_stats = cartesian.statistics();
    EXPECT_EQ(cartesian_stats.owned_records, 8u);
    EXPECT_EQ(cartesian_stats.unique_cell_pairs, expected_unique);
    EXPECT_EQ(cartesian_stats.logical_contact_pairs, expected_incidence);
    EXPECT_EQ(cartesian_stats.span_pair_evaluations, expected_unique);
    EXPECT_EQ(cartesian_stats.trace_pair_evaluations, expected_unique);
    EXPECT_EQ(cartesian_stats.work, 2 * expected_unique);
    EXPECT_GT(cartesian_stats.source_coordinate_points, 0u);
    EXPECT_EQ(cartesian.required_refinements().size(), 0u);

    std::map<AMR::AdaptiveCellKey, std::vector<std::vector<char>>> cartesian_rows;
    std::size_t cartesian_row_count = 0;
    cartesian.for_each_incidence_row(
        [&](const D::ContactIncidenceRow& row)
        {
            ++cartesian_row_count;
            EXPECT_EQ(row.points.size(), 8u);
            cartesian_rows.emplace(row.key, encoded_row_points(row));
        });
    EXPECT_EQ(cartesian_row_count, 8u);

    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
    Halo explicit_halo(*kernel, comm, options, {7, 11, 13, 0});
    explicit_halo.bootstrap_canonical(leaves_v, lookup);
    ASSERT_FALSE(explicit_halo.uses_cartesian_contact());
    EXPECT_EQ(explicit_halo.required_refinements().size(), 0u);
    const auto& explicit_stats = explicit_halo.statistics();
    EXPECT_EQ(explicit_stats.unique_cell_pairs, cartesian_stats.unique_cell_pairs);
    EXPECT_EQ(explicit_stats.logical_contact_pairs, cartesian_stats.logical_contact_pairs);
    EXPECT_EQ(explicit_stats.span_pair_evaluations, cartesian_stats.span_pair_evaluations);
    EXPECT_EQ(explicit_stats.trace_pair_evaluations, cartesian_stats.trace_pair_evaluations);
    EXPECT_EQ(explicit_stats.work, cartesian_stats.work);
    std::size_t explicit_row_count = 0;
    explicit_halo.for_each_incidence_row(
        [&](const D::ContactIncidenceRow& row)
        {
            ++explicit_row_count;
            const auto found = cartesian_rows.find(row.key);
            ASSERT_TRUE(found != cartesian_rows.end()) << "explicit row key missing from Cartesian export";
            EXPECT_EQ(encoded_row_points(row), found->second);
        });
    EXPECT_EQ(explicit_row_count, 8u);

    // Retained state: the Cartesian backend keeps capability/keys only, so it
    // must stay far below the explicit session's record/row/group storage.
    const auto cartesian_cache = cartesian.cache_storage_bytes();
    const auto explicit_cache = explicit_halo.cache_storage_bytes();
    ::testing::Test::RecordProperty("cartesian_cache_bytes", std::to_string(cartesian_cache));
    ::testing::Test::RecordProperty("explicit_cache_bytes", std::to_string(explicit_cache));
    EXPECT_LT(cartesian_cache, explicit_cache);

    // The Cartesian candidate forks by sharing the compact state.
    auto candidate = cartesian.fork_candidate({7, 11, 13, 1}, 0);
    EXPECT_TRUE(candidate->required_refinements().empty());

    // An edit that does not advance the round is rejected before any rebuild.
    EXPECT_THROW((void) cartesian.apply_delta({}, {}, {7, 11, 13, 0}, lookup), std::invalid_argument);
    EXPECT_EQ(cartesian.statistics().owned_records, 8u);
}

/** Uniform level transitions are transactional: the certified frame is rebuilt
 * from authoritative leaves, re-graded, and only then published. */
TEST(CartesianContactHaloSerialTest, UniformLevelDeltasRebuildAndRejectLocalRefinement)
{
    auto value = grid(); // 2x2x2 at level zero, one legal refinement level
    const auto kernel = cartesian_adaptive_kernel(value);
    const auto comm = Tpetra::getDefaultComm();
    ASSERT_EQ(comm->getSize(), 1);
    const auto level0 = leaves(value);
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = cap64;
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    Halo halo(*kernel, comm, options, {7, 11, 13, 0});
    halo.bootstrap_canonical(level0, root_lookup(level0));
    const auto original_statistics = halo.statistics();
    ASSERT_EQ(halo.statistics().owned_records, level0.size());

    // A local refinement mixes levels inside one owner and is rejected without
    // disturbing the accepted state of the candidate it was applied to.
    auto mixed_candidate = halo.fork_candidate({7, 11, 13, 5}, 0);
    std::vector<D::ContactRetiredLeaf> mixed_retired{{level0.front().key, level0.front().revision}};
    std::vector<AMR::AdaptiveCellKey> mixed_inserted;
    auto mixed_leaves = level0;
    mixed_leaves.erase(mixed_leaves.begin());
    for (const auto& child : kernel->split(level0.front(), AMR::AdaptiveSplit::Isotropic).children)
    {
        mixed_inserted.push_back(child.key);
        mixed_leaves.push_back(child);
    }
    std::size_t child_lookups = 0;
    const auto mixed_lookup = root_lookup(mixed_leaves);
    try
    {
        mixed_candidate->apply_delta(mixed_retired, mixed_inserted, {7, 11, 13, 6},
            [&](const AMR::AdaptiveCellKey& key) -> const AMR::AdaptiveLeaf&
            {
                const auto& leaf = mixed_lookup(key);
                child_lookups += std::get<AMR::CartesianCellAncestry>(key.ancestry).level == 1;
                return leaf;
            });
        FAIL() << "Mixed levels must decline the uniform prototype";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("uniform certified level"), std::string::npos);
    }
    EXPECT_EQ(child_lookups, mixed_inserted.size());
    EXPECT_EQ(mixed_candidate->statistics().owned_records, level0.size());
    EXPECT_EQ(mixed_candidate->statistics().full_owned_scans_after_bootstrap, 0u);
    std::size_t mixed_rows = 0;
    mixed_candidate->for_each_incidence_row([&](const D::ContactIncidenceRow&) { ++mixed_rows; });
    EXPECT_EQ(mixed_rows, level0.size());

    // Refine every cell once: 8 level-zero cells -> 64 level-one children.
    std::vector<AMR::AdaptiveLeaf> level1;
    std::vector<D::ContactRetiredLeaf> retired;
    std::vector<AMR::AdaptiveCellKey> inserted;
    for (const auto& leaf : level0)
    {
        retired.push_back({leaf.key, leaf.revision});
        for (const auto& child : kernel->split(leaf, AMR::AdaptiveSplit::Isotropic).children)
        {
            level1.push_back(child);
            inserted.push_back(child.key);
        }
    }
    ASSERT_EQ(level1.size(), 64u);
    auto lookup1 = root_lookup(level1);
    halo.apply_delta(retired, inserted, {7, 11, 13, 1}, lookup1);
    const auto& refined = halo.statistics();
    EXPECT_EQ(refined.owned_records, 64u);
    EXPECT_EQ(refined.delta_owned_visits, 64u);
    EXPECT_EQ(refined.full_owned_scans_after_bootstrap, 1u);
    EXPECT_TRUE(halo.required_refinements().empty());

    // Brute-force oracle for the 4x4x4 lattice.
    std::size_t expected_unique = 0, expected_logical = 0;
    for (std::uint64_t z = 0; z != 4; ++z)
        for (std::uint64_t y = 0; y != 4; ++y)
            for (std::uint64_t x = 0; x != 4; ++x)
                for (const auto& offset : D::cartesian_contact_offsets())
                {
                    const auto nz = std::int64_t(z) + offset[2], ny = std::int64_t(y) + offset[1],
                               nx = std::int64_t(x) + offset[0];
                    if (nx < 0 || ny < 0 || nz < 0 || nx >= 4 || ny >= 4 || nz >= 4)
                        continue;
                    const std::array<std::uint64_t, 3> next{std::uint64_t(nx), std::uint64_t(ny), std::uint64_t(nz)};
                    const std::array<std::uint64_t, 3> here{x, y, z};
                    if (!(here < next))
                        continue;
                    ++expected_unique;
                    std::size_t shared = 1;
                    for (const auto delta : offset)
                        shared *= 2 - std::size_t(delta < 0 ? -delta : delta);
                    expected_logical += shared;
                }
    EXPECT_EQ(refined.unique_cell_pairs, expected_unique);
    EXPECT_EQ(refined.logical_contact_pairs, original_statistics.logical_contact_pairs + expected_logical);
    EXPECT_EQ(refined.work, original_statistics.work + 2 * expected_unique);

    // A fresh level-one session on the same leaves is the accepted-state oracle.
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
    Halo fresh(*kernel, comm, options, {7, 11, 13, 1});
    fresh.bootstrap_canonical(level1, lookup1);
    ASSERT_FALSE(fresh.uses_cartesian_contact());
    ASSERT_TRUE(halo.uses_cartesian_contact());
    EXPECT_TRUE(fresh.required_refinements().empty());
    ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", "1", 1);
    const auto& fresh_stats = fresh.statistics();
    EXPECT_EQ(fresh_stats.unique_cell_pairs, refined.unique_cell_pairs);
    EXPECT_EQ(
        fresh_stats.logical_contact_pairs + original_statistics.logical_contact_pairs, refined.logical_contact_pairs);
    EXPECT_EQ(fresh_stats.work + original_statistics.work, refined.work);
    auto export_rows = [](const Halo& source)
    {
        std::map<AMR::AdaptiveCellKey, std::vector<std::vector<char>>> rows;
        source.for_each_incidence_row(
            [&](const D::ContactIncidenceRow& row)
            {
                EXPECT_EQ(row.points.size(), 8u);
                rows.emplace(row.key, encoded_row_points(row));
            });
        return rows;
    };
    const auto refined_rows = export_rows(halo);
    const auto fresh_rows = export_rows(fresh);
    EXPECT_EQ(refined_rows.size(), 64u);
    EXPECT_EQ(refined_rows, fresh_rows);

    // A local refinement mixes levels inside one owner and is rejected without
    // disturbing the accepted state of the candidate it was applied to.
    auto candidate = halo.fork_candidate({7, 11, 13, 2}, 0);
    EXPECT_EQ(export_rows(*candidate), refined_rows);
    EXPECT_EQ(candidate->statistics().owned_records, 64u);

    // Coarsening the whole frame returns to the level-zero accepted state.
    std::vector<D::ContactRetiredLeaf> coarsen_retired;
    std::vector<AMR::AdaptiveCellKey> coarsen_inserted;
    for (const auto& leaf : level1)
        coarsen_retired.push_back({leaf.key, leaf.revision});
    for (const auto& leaf : level0)
        coarsen_inserted.push_back(leaf.key);
    const auto before_coarsen = halo.statistics();
    halo.apply_delta(coarsen_retired, coarsen_inserted, {7, 11, 13, 4}, root_lookup(level0));
    EXPECT_EQ(halo.statistics().owned_records, level0.size());
    EXPECT_EQ(halo.statistics().unique_cell_pairs, 28u);
    EXPECT_EQ(halo.statistics().logical_contact_pairs, before_coarsen.logical_contact_pairs + 76u);
    EXPECT_EQ(export_rows(halo).size(), 8u);
    ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
}
} // namespace
