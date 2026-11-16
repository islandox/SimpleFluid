/** Actual Cartesian source certificates against literal lattice and explicit-path oracles. */
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/mesh/AMR/CartesianAuthorityCredits.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace
{
namespace AMR = SimpleFluid::Meshes::AMR;
namespace D = AMR::adaptive_detail;
namespace P = SimpleFluid::PLIC;
constexpr auto ordinary = SimpleFluid::ArithmeticMode::Ordinary;
using Key = AMR::AdaptiveCellKey;
using Point = P::PointIdentity<ordinary>;
using PointPairs = std::vector<std::pair<Point, Point>>; // physical, normalized, in publication order
using Marks = std::vector<std::pair<Key, AMR::AdaptiveSplit>>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

class FloatingEnvironment
{
public:
    FloatingEnvironment()
    {
        if (std::fegetenv(&saved_) || std::fesetround(FE_TONEAREST))
            throw std::runtime_error("Cannot establish Cartesian test arithmetic environment.");
    }
    ~FloatingEnvironment() { std::fesetenv(&saved_); }

private:
    std::fenv_t saved_{};
};

/** All geometric answers are the original factory's. Only the private source
 * capability is deliberately absent, providing an independent explicit route. */
class ExplicitPath final : public AMR::AdaptiveProviderKernel
{
public:
    explicit ExplicitPath(const AMR::AdaptiveProviderKernel& inner) : inner_(inner) {}
    AMR::AdaptiveBytes descriptor() const override { return inner_.descriptor(); }
    P::ConvexGeometryPtr<ordinary> geometry(const AMR::AdaptiveLeaf& leaf) const override
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
    Point contact_point(const AMR::AdaptiveLeaf& leaf, const Point& point) const override
    {
        return inner_.contact_point(leaf, point);
    }

private:
    const AMR::AdaptiveProviderKernel& inner_;
};

D::ContactHaloContext context(std::uint64_t transaction, std::uint64_t round = 0)
{
    return {7, 11, 13, round, transaction};
}
AMR::AMRGrid grid()
{
    AMR::AMRGrid result;
    result.block_cells = {2, 2, 2};
    result.max_level = 2;
    return result;
}
struct Owned
{
    AMR::AdaptiveLeaf leaf;
    int owner;
};
std::vector<Owned> initial(const AMR::AMRGrid& config)
{
    std::vector<Owned> result;
    for (std::uint64_t z = 0; z < 2; ++z)
        for (std::uint64_t y = 0; y < 2; ++y)
            for (std::uint64_t x = 0; x < 2; ++x)
                result.push_back({AMR::cartesian_adaptive_leaf(
                                      config, {config.region_id, AMR::CartesianCellAncestry{0, {x, y, z}}}, {'d'}),
                    0});
    return result;
}
std::vector<AMR::AdaptiveLeaf> local(std::span<const Owned> leaves)
{
    std::vector<AMR::AdaptiveLeaf> result;
    const int rank = Tpetra::getDefaultComm()->getRank();
    for (const auto& item : leaves)
        if (item.owner == rank)
            result.push_back(item.leaf);
    return result;
}
D::ContactOwnedLeafLookup lookup(const std::vector<AMR::AdaptiveLeaf>& leaves)
{
    return [&leaves](const Key& key) -> const AMR::AdaptiveLeaf&
    {
        const auto found = std::ranges::find(leaves, key, &AMR::AdaptiveLeaf::key);
        if (found == leaves.end())
            throw std::invalid_argument("Native fixture lookup reached a nonowner.");
        return *found;
    };
}
PointPairs points(const D::ContactPointArray& values)
{
    PointPairs result;
    for (const auto value : values)
        result.emplace_back(value.physical, value.normalized);
    return result;
}
struct Row
{
    std::uint64_t revision, generation;
    AMR::AdaptiveBytes domain;
    PointPairs points;
    bool operator==(const Row&) const = default;
};
using Rows = std::map<Key, Row>;
using PayloadKey = std::pair<std::uintptr_t, D::ContactPayloadRole>;
using PayloadInventory = std::map<PayloadKey, std::array<std::size_t, 4>>;
PayloadInventory payload_inventory(const D::DistributedAdaptiveContactHalo& halo)
{
    PayloadInventory result;
    halo.for_each_borrowed_contact_payload(
        [&](const D::ContactPayloadAllocation& allocation)
        {
            const PayloadKey key{
                reinterpret_cast<std::uintptr_t>(allocation.identity.address), allocation.identity.role};
            const std::array<std::size_t, 4> values{
                allocation.element_size, allocation.live_count, allocation.capacity, allocation.payload_bytes};
            const auto [at, inserted] = result.emplace(key, values);
            if (!inserted)
                EXPECT_EQ(at->second, values); // Aliases must describe the same physical extent.
        });
    return result;
}
Rows rows(const D::DistributedAdaptiveContactHalo& halo)
{
    Rows result;
    halo.for_each_incidence_row(
        [&](const auto& row)
        {
            EXPECT_TRUE(
                result.emplace(row.key, Row{row.revision, row.incidence_generation, row.domain, points(row.points)})
                    .second);
        });
    return result;
}
Marks marks(D::DistributedAdaptiveContactHalo& halo)
{
    Marks result;
    for (const auto& mark : halo.required_refinements())
        result.emplace_back(mark.key, mark.split);
    std::sort(result.begin(), result.end());
    return result;
}
void same_contents(const Rows& first, const Rows& second)
{
    EXPECT_EQ(first.size(), second.size());
    for (const auto& [key, value] : first)
    {
        const auto found = second.find(key);
        EXPECT_NE(found, second.end());
        if (found != second.end())
        {
            EXPECT_EQ(value.revision, found->second.revision);
            EXPECT_EQ(value.domain, found->second.domain);
            EXPECT_EQ(value.points, found->second.points);
        }
    }
}
void compare_fresh(const AMR::AdaptiveProviderKernel& kernel, std::span<const Owned> leaves,
    D::ContactHaloContext current, D::DistributedAdaptiveContactHalo& incremental)
{
    const auto owned = local(leaves);
    D::DistributedAdaptiveContactHalo fresh(kernel, Tpetra::getDefaultComm(), {}, current);
    fresh.bootstrap_canonical(owned, lookup(owned));
    same_contents(rows(incremental), rows(fresh));
    EXPECT_EQ(marks(incremental), marks(fresh));
}

using Lattice = std::array<std::uint64_t, 3>;
struct Bounds
{
    Lattice lower, upper;
};
Bounds bounds(const AMR::AdaptiveLeaf& leaf)
{
    const auto& ancestry = std::get<AMR::CartesianCellAncestry>(leaf.key.ancestry);
    const auto width = std::uint64_t{1} << (2 - ancestry.level);
    Bounds result;
    for (unsigned a = 0; a < 3; ++a)
    {
        result.lower[a] = ancestry.index[a] * width;
        result.upper[a] = (ancestry.index[a] + 1) * width;
    }
    return result;
}
/** Exhaustive integer-lattice oracle: every corner of every current leaf on a
 * cell's closed boundary must occur, including coarse face-interior points. No
 * source recipe, normalization, support overlay or line query builds this set. */
void expect_lattice(const Rows& actual, std::span<const Owned> leaves)
{
    std::set<Lattice> all_vertices;
    for (const auto& item : leaves)
    {
        const auto b = bounds(item.leaf);
        for (unsigned mask = 0; mask < 8; ++mask)
            all_vertices.insert({mask & 1 ? b.upper[0] : b.lower[0], mask & 2 ? b.upper[1] : b.lower[1],
                mask & 4 ? b.upper[2] : b.lower[2]});
    }
    std::size_t expected_rows = 0;
    for (const auto& item : leaves)
    {
        if (item.owner != Tpetra::getDefaultComm()->getRank())
            continue;
        ++expected_rows;
        const auto b = bounds(item.leaf);
        PointPairs expected;
        for (const auto& vertex : all_vertices)
        {
            bool inside = true, boundary = false;
            for (unsigned a = 0; a < 3; ++a)
            {
                inside &= vertex[a] >= b.lower[a] && vertex[a] <= b.upper[a];
                boundary |= vertex[a] == b.lower[a] || vertex[a] == b.upper[a];
            }
            if (inside && boundary)
            {
                const auto value =
                    P::details::GeometryIdentityAccess::point({vertex[0] / 8.0L, vertex[1] / 8.0L, vertex[2] / 8.0L});
                expected.emplace_back(value, value);
            }
        }
        std::sort(expected.begin(), expected.end());
        const auto found = actual.find(item.leaf.key);
        EXPECT_NE(found, actual.end());
        if (found != actual.end())
            EXPECT_EQ(found->second.points, expected);
    }
    EXPECT_EQ(actual.size(), expected_rows);
}
Marks lattice_marks(std::span<const Owned> leaves)
{
    Marks result;
    for (const auto& first : leaves)
    {
        if (first.owner != Tpetra::getDefaultComm()->getRank())
            continue;
        const auto level = std::get<AMR::CartesianCellAncestry>(first.leaf.key.ancestry).level;
        const auto a = bounds(first.leaf);
        for (const auto& second : leaves)
        {
            if (std::get<AMR::CartesianCellAncestry>(second.leaf.key.ancestry).level <= level + 1)
                continue;
            const auto b = bounds(second.leaf);
            bool contact = true;
            for (unsigned axis = 0; axis < 3; ++axis)
                contact &= a.lower[axis] <= b.upper[axis] && b.lower[axis] <= a.upper[axis];
            if (contact)
            {
                result.emplace_back(first.leaf.key, AMR::AdaptiveSplit::Isotropic);
                break;
            }
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}
void apply_edit(D::DistributedAdaptiveContactHalo& halo, std::span<const Owned> before, std::span<const Owned> after,
    D::ContactHaloContext current)
{
    const int rank = Tpetra::getDefaultComm()->getRank();
    std::vector<D::ContactRetiredLeaf> retired;
    std::vector<Key> inserted;
    for (const auto& item : before)
        if (item.owner == rank && std::ranges::none_of(after,
                                      [&](const auto& next)
                                      {
                                          return next.owner == rank && next.leaf.key == item.leaf.key &&
                                                 next.leaf.revision == item.leaf.revision;
                                      }))
            retired.push_back({item.leaf.key, item.leaf.revision});
    for (const auto& item : after)
        if (item.owner == rank && std::ranges::none_of(before,
                                      [&](const auto& old)
                                      {
                                          return old.owner == rank && old.leaf.key == item.leaf.key &&
                                                 old.leaf.revision == item.leaf.revision;
                                      }))
            inserted.push_back(item.leaf.key);
    const auto owned = local(after);
    halo.apply_delta(retired, inserted, current, lookup(owned));
    // Complete the canonical neighborhood before a subsequent edit or fork,
    // just as the production closure driver consumes each round's requirements.
    (void) marks(halo);
}
std::vector<Owned> replace(
    std::vector<Owned> leaves, const Key& parent, std::span<const AMR::AdaptiveLeaf> children, int owner)
{
    std::erase_if(leaves, [&](const auto& item) { return item.leaf.key == parent; });
    for (const auto& child : children)
        leaves.push_back({child, owner});
    return leaves;
}
void expect_source_path(const D::DistributedAdaptiveContactHalo& halo, std::size_t local_cells)
{
    const auto& stats = halo.statistics();
    EXPECT_EQ(stats.source_owner_manifests, std::size_t(Tpetra::getDefaultComm()->getSize()));
    EXPECT_EQ(stats.source_membership_cells, local_cells);
    EXPECT_EQ(stats.explicit_line_records, 0u);
    EXPECT_EQ(stats.stride_line_records, 0u);
    EXPECT_EQ(stats.retained_explicit_layer_input_slots, 0u);
    EXPECT_EQ(stats.retained_explicit_layer_output_slots, 0u);
    EXPECT_EQ(stats.retained_explicit_layer_edge_result_slots, 0u);
    EXPECT_EQ(stats.retained_expanded_completion_result_slots, 0u);
    // Independently inspect actual buffer capacities instead of trusting only
    // the cached storage observations above. The visitor is borrowed and pure.
    std::size_t explicit_layer_payload = 0;
    halo.for_each_borrowed_contact_payload(
        [&](const D::ContactPayloadAllocation& allocation)
        {
            if (allocation.identity.role == D::GeometryContactPayloadRole::IncidenceLayerInput ||
                allocation.identity.role == D::GeometryContactPayloadRole::IncidenceLayerOutput ||
                allocation.identity.role == D::GeometryContactPayloadRole::IncidenceLayerEdgeResults ||
                allocation.identity.role == D::ParallelSupportPayloadRole::PooledLineCompletionResults)
            {
                EXPECT_EQ(allocation.payload_bytes, allocation.element_size * allocation.capacity);
                explicit_layer_payload += allocation.payload_bytes;
            }
        });
    EXPECT_EQ(explicit_layer_payload, 0u);
    if (local_cells)
    {
        EXPECT_GT(stats.source_coordinate_points, 0u);
        EXPECT_EQ(stats.explicit_coordinate_points, 0u);
    }
    std::uint64_t local_lines = stats.source_line_records, global_lines = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_lines, &global_lines);
    EXPECT_GT(global_lines, 0u);
}

TEST(CartesianIncidenceSession, LatticeRefinementCoarseningAndMigrationMatchFreshWithEmptyOwners)
{
    FloatingEnvironment arithmetic;
    const auto config = grid();
    const auto kernel = AMR::cartesian_adaptive_kernel(config);
    ExplicitPath explicit_kernel(*kernel);
    const auto comm = Tpetra::getDefaultComm();
    const int remote = std::min(1, comm->getSize() - 1);
    const auto original = initial(config);
    const auto owned = local(original);
    auto accepted =
        std::make_unique<D::DistributedAdaptiveContactHalo>(*kernel, comm, AMR::AdaptiveClosureOptions{}, context(101));
    auto explicit_accepted = std::make_unique<D::DistributedAdaptiveContactHalo>(
        explicit_kernel, comm, AMR::AdaptiveClosureOptions{}, context(101));
    accepted->bootstrap_canonical(owned, lookup(owned));
    explicit_accepted->bootstrap_canonical(owned, lookup(owned));
    expect_source_path(*accepted, owned.size());
    const auto baseline = rows(*accepted);
    EXPECT_EQ(baseline, rows(*explicit_accepted));
    expect_lattice(baseline, original);
    EXPECT_TRUE(marks(*accepted).empty());
    EXPECT_EQ(marks(*explicit_accepted), marks(*accepted));
    EXPECT_EQ(accepted->statistics().cached_completion_requests, 0u);
    EXPECT_EQ(accepted->statistics().retained_completion_owners, 0u);

    auto candidate = accepted->fork_candidate(context(202));
    auto explicit_candidate = explicit_accepted->fork_candidate(context(202));
    auto sibling = accepted->fork_candidate(context(203));
    std::vector<std::pair<D::ContactPointArray, PointPairs>> escaping;
    accepted->for_each_incidence_row([&](const auto& row) { escaping.emplace_back(row.points, points(row.points)); });
    const auto first_split = kernel->split(original.front().leaf, AMR::AdaptiveSplit::Isotropic);
    const auto refined = replace(original, original.front().leaf.key, first_split.children, remote);
    apply_edit(*candidate, original, refined, context(202, 1));
    apply_edit(*explicit_candidate, original, refined, context(202, 1));
    EXPECT_EQ(rows(*candidate), rows(*explicit_candidate));
    expect_source_path(*candidate, local(refined).size());
    EXPECT_EQ(candidate->statistics().explicit_tile_ring_slots, 0u);
    expect_lattice(rows(*candidate), refined);
    EXPECT_EQ(marks(*candidate), lattice_marks(refined));
    compare_fresh(*kernel, refined, context(202, 1), *candidate);
    EXPECT_EQ(rows(*accepted), baseline);
    EXPECT_EQ(rows(*sibling), baseline);

    const auto chosen = std::ranges::find_if(first_split.children, [](const auto& leaf)
        { return std::get<AMR::CartesianCellAncestry>(leaf.key.ancestry).index == Lattice{1, 1, 1}; });
    ASSERT_NE(chosen, first_split.children.end()); // Identical literal fixture on every rank.
    const auto second_split = kernel->split(*chosen, AMR::AdaptiveSplit::Isotropic);
    const auto twice_refined = replace(refined, chosen->key, second_split.children, remote);
    apply_edit(*candidate, refined, twice_refined, context(202, 2));
    apply_edit(*explicit_candidate, refined, twice_refined, context(202, 2));
    EXPECT_EQ(rows(*candidate), rows(*explicit_candidate));
    expect_source_path(*candidate, local(twice_refined).size());
    EXPECT_EQ(candidate->statistics().explicit_tile_ring_slots, 0u);
    EXPECT_EQ(marks(*candidate), marks(*explicit_candidate));
    expect_lattice(rows(*candidate), twice_refined);
    EXPECT_EQ(marks(*candidate), lattice_marks(twice_refined));
    compare_fresh(*kernel, twice_refined, context(202, 2), *candidate);

    accepted.reset();
    explicit_accepted.reset();
    sibling.reset();
    apply_edit(*candidate, twice_refined, refined, context(202, 3));
    apply_edit(*explicit_candidate, twice_refined, refined, context(202, 3));
    EXPECT_EQ(rows(*candidate), rows(*explicit_candidate));
    expect_source_path(*candidate, local(refined).size());
    compare_fresh(*kernel, refined, context(202, 3), *candidate);
    apply_edit(*candidate, refined, original, context(202, 4));
    apply_edit(*explicit_candidate, refined, original, context(202, 4));
    EXPECT_EQ(rows(*candidate), rows(*explicit_candidate));
    expect_source_path(*candidate, local(original).size());
    expect_lattice(rows(*candidate), original);
    compare_fresh(*kernel, original, context(202, 4), *candidate);

    auto moved = original;
    for (auto& item : moved)
        item.owner = remote;
    apply_edit(*candidate, original, moved, context(202, 5));
    apply_edit(*explicit_candidate, original, moved, context(202, 5));
    EXPECT_EQ(rows(*candidate), rows(*explicit_candidate));
    expect_source_path(*candidate, local(moved).size());
    EXPECT_EQ(candidate->statistics().explicit_tile_ring_slots, 0u);
    expect_lattice(rows(*candidate), moved);
    compare_fresh(*kernel, moved, context(202, 5), *candidate);
    EXPECT_EQ(candidate->statistics().full_owned_scans_after_bootstrap, 0u);
    if (comm->getSize() > 1)
    {
        const std::uint64_t local_remote = candidate->statistics().cached_completion_remote_requests;
        std::uint64_t global_remote = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_remote, &global_remote);
        EXPECT_GT(global_remote, 0u);
    }
    candidate.reset();
    explicit_candidate.reset();
    for (const auto& [array, expected] : escaping)
        EXPECT_EQ(points(array), expected);
}

TEST(CartesianIncidenceSession, CandidateExactAdmissionAndOneByteShortPreserveSource)
{
    FloatingEnvironment arithmetic;
    const auto config = grid();
    const auto kernel = AMR::cartesian_adaptive_kernel(config);
    const auto comm = Tpetra::getDefaultComm();
    const auto owned = local(initial(config));
    D::DistributedAdaptiveContactHalo source(*kernel, comm, {}, context(301));
    source.bootstrap_canonical(owned, lookup(owned));
    EXPECT_TRUE(marks(source).empty());
    const auto expected = rows(source);
    expect_source_path(source, owned.size());
    const auto old_used = source.allocation().used;
    std::size_t required = 0, retained = 0;
    {
        auto probe = source.fork_candidate(context(302));
        // The caller's live owners coexist with every construction instant,
        // including scratch that has retired before fork returns.
        required = probe->allocation().peak;
        retained = probe->allocation().used;
    }
    const auto cap = source.allocation().limit;
    const int local_valid = required < cap;
    int all_valid = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_valid, &all_valid);
    ASSERT_EQ(all_valid, 1);
    const auto caller = cap - required;
    {
        auto exact = source.fork_candidate(context(303), caller);
        EXPECT_EQ(exact->allocation().peak, cap);
        EXPECT_EQ(exact->allocation().used, caller + retained);
        EXPECT_EQ(rows(*exact), expected);
    }
    int rejected = 0;
    try
    {
        // Only the last rank loses a byte, including an owner with zero cells.
        auto failed = source.fork_candidate(context(304), caller + (comm->getRank() + 1 == comm->getSize()));
    }
    catch (const std::invalid_argument&)
    {
        rejected = 1;
    }
    int all_rejected = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &rejected, &all_rejected);
    EXPECT_EQ(all_rejected, 1);
    EXPECT_EQ(source.allocation().used, old_used);
    EXPECT_EQ(rows(source), expected);
    auto retry = source.fork_candidate(context(305));
    EXPECT_EQ(rows(*retry), expected);
}

TEST(CartesianIncidenceSession, CoarseFineBootstrapCertifiesNonforwardLineResults)
{
    FloatingEnvironment arithmetic;
    const auto config = grid();
    const auto kernel = AMR::cartesian_adaptive_kernel(config);
    ExplicitPath explicit_kernel(*kernel);
    const auto comm = Tpetra::getDefaultComm();
    const auto original = initial(config);
    const auto split = kernel->split(original.front().leaf, AMR::AdaptiveSplit::Isotropic);
    const auto refined = replace(original, original.front().leaf.key, split.children, std::min(1, comm->getSize() - 1));
    const auto owned = local(refined);
    D::DistributedAdaptiveContactHalo source(*kernel, comm, {}, context(351));
    D::DistributedAdaptiveContactHalo baseline(explicit_kernel, comm, {}, context(351));
    source.bootstrap_canonical(owned, lookup(owned));
    baseline.bootstrap_canonical(owned, lookup(owned));
    EXPECT_EQ(rows(source), rows(baseline));
    expect_lattice(rows(source), refined);
    EXPECT_EQ(marks(source), lattice_marks(refined));
    EXPECT_EQ(marks(source), marks(baseline));
    expect_source_path(source, owned.size());
    const auto& stats = source.statistics();
    const std::array<std::uint64_t, 2> local_counts{
        stats.source_line_nonforward_records, stats.source_result_selectors};
    std::array<std::uint64_t, 2> global_counts{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 2, local_counts.data(), global_counts.data());
    EXPECT_GT(global_counts[0], 0u);
    EXPECT_GT(global_counts[1], 0u);
    const std::uint64_t saved_slots = stats.retained_certified_layer_output_slots;
    std::uint64_t global_saved_slots = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &saved_slots, &global_saved_slots);
    EXPECT_GT(global_saved_slots, 0u);
}

TEST(CartesianIncidenceSession, ThreeGenerationForkCreditsMatchActualCandidateOwnerImports)
{
    FloatingEnvironment arithmetic;
    const auto config = grid();
    const auto kernel = AMR::cartesian_adaptive_kernel(config);
    const auto comm = Tpetra::getDefaultComm();
    const auto original = initial(config);
    const auto split = kernel->split(original.front().leaf, AMR::AdaptiveSplit::Isotropic);
    const auto refined = replace(original, original.front().leaf.key, split.children, std::min(1, comm->getSize() - 1));
    const auto owned = local(refined);
    D::DistributedAdaptiveContactHalo first(*kernel, comm, {}, context(361));
    first.bootstrap_canonical(owned, lookup(owned));
    EXPECT_EQ(marks(first), lattice_marks(refined));
    const std::uint64_t saved = first.statistics().retained_saved_layers;
    std::uint64_t global_saved = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &saved, &global_saved);
    EXPECT_GT(global_saved, 0u); // This must exercise saved layer dependencies.

    auto second = first.fork_candidate(context(362));
    EXPECT_EQ(marks(*second), lattice_marks(refined));
    const auto expected_rows = rows(*second);
    const auto second_used = second->allocation().used;
    auto third = second->fork_candidate(context(363));
    EXPECT_EQ(rows(*third), expected_rows);

    // Observe C's real admitted immutable owners independently of the source's
    // overlap classification. These borrowed accessors cannot promote weak
    // owners or retain a new pin. The diagnostic maps are test-owned scratch.
    std::map<const void*, std::set<std::size_t>, std::less<>> imported;
    third->for_each_borrowed_source_admission(
        [&](const void* identity, std::size_t bytes) { imported[identity].insert(bytes); });
    std::set<const void*, std::less<>> seen;
    std::uint64_t private_tokens = 0, imported_tokens = 0;
    std::size_t independently_required_metadata = 0;
    constexpr auto token_metadata =
        sizeof(D::cartesian_detail::AuthorityOwnerCredit) + D::cartesian_detail::ArenaCredits::control_bytes;
    second->for_each_borrowed_overlap_credit(
        [&](const void* token, std::size_t metadata, bool production_subtracts)
        {
            EXPECT_TRUE(seen.insert(token).second);
            EXPECT_EQ(metadata, token_metadata);
            const auto found = imported.find(token);
            const bool actually_imported = found != imported.end() && found->second.contains(metadata);
            EXPECT_EQ(production_subtracts, actually_imported)
                << "Source token metadata can leave overlap only when that exact owner is admitted by C.";
            if (actually_imported)
                ++imported_tokens;
            else
            {
                ++private_tokens;
                independently_required_metadata += token_metadata;
            }
        });
    EXPECT_GE(third->current_fork_overlap_bytes(), independently_required_metadata);
    const std::array<std::uint64_t, 2> local_counts{private_tokens, imported_tokens};
    std::array<std::uint64_t, 2> global_counts{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 2, local_counts.data(), global_counts.data());
    EXPECT_GT(global_counts[0], 0u); // The missing B-private token category must be present.
    EXPECT_EQ(second->allocation().used, second_used);
    EXPECT_EQ(rows(*second), expected_rows);
    EXPECT_EQ(marks(*third), lattice_marks(refined));
    third.reset();
    EXPECT_EQ(second->allocation().used, second_used);
    EXPECT_EQ(rows(*second), expected_rows);
}

TEST(CartesianIncidenceSession, EditingParentAfterForkKeepsChildLayerOwnersAndCapacitiesImmutable)
{
    FloatingEnvironment arithmetic;
    const auto config = grid();
    const auto kernel = AMR::cartesian_adaptive_kernel(config);
    const auto comm = Tpetra::getDefaultComm();
    const auto original = initial(config);
    const auto parent_at = [&](Lattice index) -> const AMR::AdaptiveLeaf&
    {
        const auto found = std::ranges::find_if(original, [&](const auto& item)
            { return std::get<AMR::CartesianCellAncestry>(item.leaf.key.ancestry).index == index; });
        if (found == original.end())
            throw std::logic_error("Missing literal parent in COW fixture.");
        return found->leaf;
    };
    const auto first_split = kernel->split(parent_at({0, 0, 0}), AMR::AdaptiveSplit::Isotropic);
    const auto first_cells = replace(original, parent_at({0, 0, 0}).key, first_split.children, 0);
    const auto first_owned = local(first_cells);
    D::DistributedAdaptiveContactHalo first(*kernel, comm, {}, context(371));
    first.bootstrap_canonical(first_owned, lookup(first_owned));
    EXPECT_EQ(marks(first), lattice_marks(first_cells));
    const auto first_payload = payload_inventory(first);
    auto second = first.fork_candidate(context(372));

    // This edit creates B-owned saved layers on the coarse z=0 face of cell
    // (1,1,0). The later edit touches another edge of that very same face.
    const auto second_split = kernel->split(parent_at({1, 0, 0}), AMR::AdaptiveSplit::Isotropic);
    const auto second_cells =
        replace(first_cells, parent_at({1, 0, 0}).key, second_split.children, std::min(1, comm->getSize() - 1));
    apply_edit(*second, first_cells, second_cells, context(372, 1));
    expect_lattice(rows(*second), second_cells);
    expect_source_path(*second, local(second_cells).size());
    const auto second_payload = payload_inventory(*second);
    std::uint64_t new_layers = 0;
    for (const auto& [key, value] : second_payload)
        if (key.second == D::GeometryContactPayloadRole::IncidenceLayerObject && !first_payload.contains(key))
            ++new_layers;
    std::uint64_t global_new_layers = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &new_layers, &global_new_layers);
    EXPECT_GT(global_new_layers, 0u);

    auto third = second->fork_candidate(context(373, 1));
    EXPECT_EQ(marks(*third), lattice_marks(second_cells));
    const auto child_rows = rows(*third);
    const auto child_payload = payload_inventory(*third);
    const auto child_used = third->allocation().used;
    const auto child_cumulative = third->allocation().cumulative;
    std::uint64_t shared_new_layers = 0;
    for (const auto& [key, value] : second_payload)
        if (key.second == D::GeometryContactPayloadRole::IncidenceLayerObject && !first_payload.contains(key))
            if (const auto found = child_payload.find(key); found != child_payload.end())
            {
                EXPECT_EQ(found->second, value);
                ++shared_new_layers;
            }
    std::uint64_t global_shared_new_layers = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &shared_new_layers, &global_shared_new_layers);
    EXPECT_GT(global_shared_new_layers, 0u);
    const auto next_split = kernel->split(parent_at({0, 1, 0}), AMR::AdaptiveSplit::Isotropic);
    const auto next_cells = replace(second_cells, parent_at({0, 1, 0}).key, next_split.children, 0);
    apply_edit(*second, second_cells, next_cells, context(372, 2));
    expect_lattice(rows(*second), next_cells);
    expect_source_path(*second, local(next_cells).size());
    compare_fresh(*kernel, next_cells, context(372, 2), *second);

    EXPECT_EQ(rows(*third), child_rows);
    EXPECT_EQ(payload_inventory(*third), child_payload);
    EXPECT_EQ(third->allocation().used, child_used);
    EXPECT_EQ(third->allocation().cumulative, child_cumulative);
    expect_lattice(rows(*third), second_cells);
    compare_fresh(*kernel, second_cells, context(373, 1), *third);
    second.reset();
    EXPECT_EQ(rows(*third), child_rows);
    expect_source_path(*third, local(second_cells).size());
}

TEST(CartesianIncidenceSession, EveryPeriodicAxisCombinationPreservesPhysicalAndNormalizedRows)
{
    FloatingEnvironment arithmetic;
    const auto comm = Tpetra::getDefaultComm();
    for (bool nonbinary : {false, true})
        for (unsigned mask = 0; mask < 8; ++mask)
        {
            SCOPED_TRACE(nonbinary);
            SCOPED_TRACE(mask);
            auto config = grid();
            if (nonbinary)
            {
                config.origin = {.1, -.3, 2.7};
                config.extent = {.7, 1.1, 2.3};
            }
            for (unsigned axis = 0; axis < 3; ++axis)
                config.periodic[axis] = (mask & (1u << axis)) != 0;
            const auto kernel = AMR::cartesian_adaptive_kernel(config);
            ExplicitPath explicit_kernel(*kernel);
            const auto all = initial(config);
            const auto owned = local(all);
            D::DistributedAdaptiveContactHalo source(*kernel, comm, {}, context(401));
            D::DistributedAdaptiveContactHalo baseline(explicit_kernel, comm, {}, context(401));
            source.bootstrap_canonical(owned, lookup(owned));
            baseline.bootstrap_canonical(owned, lookup(owned));
            EXPECT_EQ(rows(source), rows(baseline));
            EXPECT_EQ(marks(source), marks(baseline));
            EXPECT_EQ(source.statistics().source_owner_manifests, std::size_t(comm->getSize()));
            EXPECT_EQ(source.statistics().source_membership_cells, owned.size());
            if (!owned.empty())
                EXPECT_GT(source.statistics().source_coordinate_points, 0u);
            if (!nonbinary)
            {
                const auto actual = rows(source);
                for (const auto& leaf : owned)
                {
                    const auto b = bounds(leaf);
                    PointPairs expected;
                    for (unsigned corner = 0; corner < 8; ++corner)
                    {
                        const Lattice physical{corner & 1 ? b.upper[0] : b.lower[0],
                            corner & 2 ? b.upper[1] : b.lower[1], corner & 4 ? b.upper[2] : b.lower[2]};
                        auto normalized = physical;
                        for (unsigned axis = 0; axis < 3; ++axis)
                            if (config.periodic[axis] && normalized[axis] == 8)
                                normalized[axis] = 0;
                        expected.emplace_back(P::details::GeometryIdentityAccess::point(
                                                  {physical[0] / 8.0L, physical[1] / 8.0L, physical[2] / 8.0L}),
                            P::details::GeometryIdentityAccess::point(
                                {normalized[0] / 8.0L, normalized[1] / 8.0L, normalized[2] / 8.0L}));
                    }
                    std::sort(expected.begin(), expected.end());
                    const auto found = actual.find(leaf.key);
                    EXPECT_NE(found, actual.end());
                    if (found != actual.end())
                        EXPECT_EQ(found->second.points, expected);
                }
            }
        }
}

TEST(CartesianIncidenceSession, PeriodicPartnerRefinementRetiresEnrichedPhysicalSeamLoops)
{
    FloatingEnvironment arithmetic;
    auto config = grid();
    config.periodic[0] = true;
    const auto kernel = AMR::cartesian_adaptive_kernel(config);
    ExplicitPath explicit_kernel(*kernel);
    const auto comm = Tpetra::getDefaultComm();
    const auto original = initial(config);
    auto refined = original;
    for (const auto& item : original)
    {
        const auto& ancestry = std::get<AMR::CartesianCellAncestry>(item.leaf.key.ancestry);
        if (ancestry.index[1] || ancestry.index[2])
            continue;
        // Refine BOTH physical partners, preserving legitimate periodic matching.
        const auto split = kernel->split(item.leaf, AMR::AdaptiveSplit::Isotropic);
        refined = replace(std::move(refined), item.leaf.key, split.children,
            ancestry.index[0] ? std::min(1, comm->getSize() - 1) : 0);
    }
    const auto owned = local(refined);
    D::DistributedAdaptiveContactHalo source(*kernel, comm, {}, context(451));
    D::DistributedAdaptiveContactHalo baseline(explicit_kernel, comm, {}, context(451));
    source.bootstrap_canonical(owned, lookup(owned));
    baseline.bootstrap_canonical(owned, lookup(owned));
    EXPECT_EQ(rows(source), rows(baseline));
    EXPECT_TRUE(marks(source).empty());
    EXPECT_EQ(marks(source), marks(baseline));
    expect_source_path(source, owned.size());

    std::set<Lattice> vertices;
    for (const auto& item : refined)
    {
        const auto b = bounds(item.leaf);
        for (unsigned mask = 0; mask < 8; ++mask)
            vertices.insert({mask & 1 ? b.upper[0] : b.lower[0], mask & 2 ? b.upper[1] : b.lower[1],
                mask & 4 ? b.upper[2] : b.lower[2]});
    }
    const auto actual = rows(source);
    std::uint64_t enriched_seams = 0;
    for (const auto& leaf : owned)
    {
        const auto b = bounds(leaf);
        std::set<std::pair<Point, Point>> expected;
        for (const auto& vertex : vertices)
            for (const auto shift : {-8, 0, 8})
            {
                const std::array<std::int64_t, 3> physical{static_cast<std::int64_t>(vertex[0]) + shift,
                    static_cast<std::int64_t>(vertex[1]), static_cast<std::int64_t>(vertex[2])};
                bool inside = true, boundary = false;
                for (unsigned axis = 0; axis < 3; ++axis)
                {
                    inside &= physical[axis] >= static_cast<std::int64_t>(b.lower[axis]) &&
                              physical[axis] <= static_cast<std::int64_t>(b.upper[axis]);
                    boundary |= physical[axis] == static_cast<std::int64_t>(b.lower[axis]) ||
                                physical[axis] == static_cast<std::int64_t>(b.upper[axis]);
                }
                if (!inside || !boundary)
                    continue;
                auto normalized = physical;
                if (normalized[0] == 8)
                    normalized[0] = 0;
                expected.emplace(P::details::GeometryIdentityAccess::point(
                                     {physical[0] / 8.0L, physical[1] / 8.0L, physical[2] / 8.0L}),
                    P::details::GeometryIdentityAccess::point(
                        {normalized[0] / 8.0L, normalized[1] / 8.0L, normalized[2] / 8.0L}));
            }
        const auto found = actual.find(leaf.key);
        EXPECT_NE(found, actual.end());
        if (found != actual.end())
        {
            EXPECT_EQ(found->second.points, (PointPairs(expected.begin(), expected.end())));
            for (long double seam : {0.L, 1.L})
            {
                const auto count =
                    std::count_if(found->second.points.begin(), found->second.points.end(), [&](const auto& point)
                        { return P::details::GeometryIdentityAccess::point(point.first)[0] == seam; });
                enriched_seams += count > 4;
            }
        }
    }
    std::uint64_t global_enriched_seams = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &enriched_seams, &global_enriched_seams);
    EXPECT_GT(global_enriched_seams, 0u); // Literal geometry forces >4 vertices on a physical periodic seam.
    compare_fresh(*kernel, refined, context(451), source);
}
} // namespace
