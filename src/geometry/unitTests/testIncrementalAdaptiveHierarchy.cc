/** Hierarchy-level factor order, migration and rejected-closure source oracles. */
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
namespace AMR = SimpleFluid::Meshes::AMR;
namespace P = SimpleFluid::PLIC;
constexpr auto ordinary = SimpleFluid::ArithmeticMode::Ordinary;
using Hierarchy = AMR::DistributedAdaptiveHierarchy;
using State = std::shared_ptr<const Hierarchy>;
using Bounds = std::array<std::array<long double, 2>, 3>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

struct Snapshot
{
    AMR::HierarchyObservation observation;
    std::vector<AMR::AdaptiveBytes> leaves, families;
    bool operator==(const Snapshot&) const = default;
};
Snapshot snapshot(const Hierarchy& state)
{
    Snapshot result{state.observation(), {}, {}};
    for (const auto& [key, leaf] : state.owned())
        result.leaves.push_back(AMR::encode_adaptive_leaf(leaf));
    for (const auto& [key, family] : state.families())
        result.families.push_back(AMR::encode_adaptive_family(family));
    return result;
}
template<class Operation> void reject_closure_collectively(Operation&& operation)
{
    int rejected = 0, guard = 0;
    try
    {
        operation();
    }
    catch (const std::invalid_argument& error)
    {
        rejected = 1;
        guard =
            std::string_view(error.what()) == "Closure would require multiple source generations or undo coarsening.";
    }
    catch (const std::exception& error)
    {
        ADD_FAILURE() << "Unexpected closure rejection type: " << error.what();
    }
    int all_rejected = 0, any_guard = 0;
    const auto comm = Tpetra::getDefaultComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &rejected, &all_rejected);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &guard, &any_guard);
    EXPECT_EQ(all_rejected, 1);
    EXPECT_EQ(any_guard, 1);
}
std::uint64_t sum(std::uint64_t local)
{
    std::uint64_t result = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local, &result);
    return result;
}
std::vector<AMR::AdaptiveMark> mark_owned(const Hierarchy& state, const AMR::AdaptiveCellKey& key)
{
    return state.owned().contains(key) ? std::vector{AMR::AdaptiveMark{key, AMR::AdaptiveSplit::Isotropic}}
                                       : std::vector<AMR::AdaptiveMark>{};
}
State adapt(const State& source, const std::vector<AMR::AdaptiveMark>& marks,
    const std::vector<AMR::AdaptiveBytes>& families = {})
{
    const auto before = snapshot(*source);
    auto result = source->apply(source->plan_adaptation(marks, families));
    EXPECT_EQ(snapshot(*source), before);
    EXPECT_TRUE(source->geometry_unchanged());
    return result;
}

AMR::AdaptiveCellKey product_key(unsigned base, unsigned axial, std::vector<std::uint64_t> base_path = {},
    std::vector<std::uint64_t> axial_path = {})
{
    return {0, AMR::ProductCellAncestry{{base, 1, std::move(base_path)}, {axial, std::move(axial_path)}}};
}
State product_source()
{
    AMR::ExtrudedAMRGrid grid;
    for (unsigned y = 0; y < 3; ++y)
        for (unsigned x = 0; x < 3; ++x)
            grid.xy_nodes.push_back({.25 * x, .25 * y, 0});
    grid.base_cells = {{0, 1, 4, 3}, {1, 2, 5, 4}, {3, 4, 7, 6}, {4, 5, 8, 7}};
    grid.z_edges = {0, .25, .5, .75, 1};
    std::vector<std::uint64_t> selected;
    if (Tpetra::getDefaultComm()->getRank() == 0)
        for (std::uint64_t cell = 0; cell < 16; ++cell)
            selected.push_back(cell);
    auto detached = [&]
    {
        AMR::ExtrudedAMR source(grid);
        return source.detach_owned(selected, {'f', 'a', 'c', 't', 'o', 'r'});
    }();
    return std::make_shared<const Hierarchy>(
        std::move(detached.owned), std::move(detached.families), std::move(detached.kernel), Tpetra::getDefaultComm());
}
std::vector<AMR::AdaptiveCellKey> product_keys(bool base_split, bool axial_split)
{
    std::vector<AMR::AdaptiveCellKey> result;
    for (unsigned z = 0; z < 4; ++z)
        for (unsigned base = 0; base < 4; ++base)
            if (base != 1 || z != 0)
                result.push_back(product_key(base, z));
    for (unsigned b = 0; b < (base_split ? 4u : 1u); ++b)
        for (unsigned z = 0; z < (axial_split ? 2u : 1u); ++z)
            result.push_back(
                product_key(1, 0, base_split ? std::vector<std::uint64_t>{b} : std::vector<std::uint64_t>{},
                    axial_split ? std::vector<std::uint64_t>{z} : std::vector<std::uint64_t>{}));
    std::sort(result.begin(), result.end());
    return result;
}
std::vector<Bounds> product_boxes(bool base_split, bool axial_split)
{
    std::vector<Bounds> result;
    for (unsigned z = 0; z < 4; ++z)
        for (unsigned y = 0; y < 2; ++y)
            for (unsigned x = 0; x < 2; ++x)
            {
                if (x == 1 && y == 0 && z == 0)
                    continue;
                result.push_back(
                    {{{.25L * x, .25L * (x + 1)}, {.25L * y, .25L * (y + 1)}, {.25L * z, .25L * (z + 1)}}});
            }
    const unsigned xy_count = base_split ? 2 : 1, z_count = axial_split ? 2 : 1;
    const long double xy_width = .25L / xy_count, z_width = .25L / z_count;
    for (unsigned z = 0; z < z_count; ++z)
        for (unsigned y = 0; y < xy_count; ++y)
            for (unsigned x = 0; x < xy_count; ++x)
                result.push_back({{{.25L + xy_width * x, .25L + xy_width * (x + 1)}, {xy_width * y, xy_width * (y + 1)},
                    {z_width * z, z_width * (z + 1)}}});
    return result;
}
void expect_product(const Hierarchy& state, bool base_split, bool axial_split)
{
    const auto expected_keys = product_keys(base_split, axial_split);
    const auto expected_boxes = product_boxes(base_split, axial_split);
    std::vector<int> local_keys(expected_keys.size()), global_keys(expected_keys.size());
    std::vector<int> local_boxes(expected_boxes.size()), global_boxes(expected_boxes.size());
    double local_volume = 0, global_volume = 0;
    for (const auto& [key, leaf] : state.owned())
    {
        const auto key_position = std::ranges::find(expected_keys, key);
        EXPECT_NE(key_position, expected_keys.end());
        if (key_position != expected_keys.end())
            ++local_keys[key_position - expected_keys.begin()];
        const auto geometry = state.geometry(leaf);
        const auto bounds = geometry->closed_contact_bounds();
        const auto box_position = std::ranges::find(expected_boxes, bounds);
        EXPECT_NE(box_position, expected_boxes.end());
        if (box_position != expected_boxes.end())
            ++local_boxes[box_position - expected_boxes.begin()];
        local_volume += geometry->moments().volume.as_double();
    }
    const auto comm = state.communicator();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(local_keys.size()), local_keys.data(), global_keys.data());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(local_boxes.size()), local_boxes.data(), global_boxes.data());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_volume, &global_volume);
    EXPECT_EQ(global_keys, std::vector<int>(expected_keys.size(), 1));
    EXPECT_EQ(global_boxes, std::vector<int>(expected_boxes.size(), 1));
    EXPECT_EQ(state.global_cells(), expected_keys.size());
    EXPECT_DOUBLE_EQ(global_volume, .25);
}
std::vector<AMR::AdaptiveMark> product_marks(const Hierarchy& state, AMR::AdaptiveSplit split)
{
    std::vector<AMR::AdaptiveMark> result;
    for (const auto& [key, leaf] : state.owned())
    {
        const auto& cell = std::get<AMR::ProductCellAncestry>(key.ancestry);
        if (cell.base.root == 1 && cell.axial.root == 0)
            result.push_back({key, split});
    }
    return result;
}
int product_owner(const AMR::AdaptiveCellKey& key, unsigned shift)
{
    const auto& cell = std::get<AMR::ProductCellAncestry>(key.ancestry);
    if (cell.base.root != 1 || cell.axial.root != 0)
        return 0;
    const auto b = cell.base.path.empty() ? 0 : cell.base.path.back();
    const auto z = cell.axial.path.empty() ? 0 : cell.axial.path.back();
    return int((b + 2 * z + shift) % Tpetra::getDefaultComm()->getSize());
}
State migrate_product(const State& source, bool base_split, bool axial_split, unsigned shift)
{
    const auto before = snapshot(*source);
    const auto keys = product_keys(base_split, axial_split);
    const auto old_owners = source->lookup_cells(keys);
    EXPECT_EQ(old_owners.size(), keys.size());
    bool ownership_changed = false;
    // The independent key oracle covers every global cell, so these directory
    // observations determine whether the requested routing changes ownership.
    for (std::size_t i = 0; i < std::min(old_owners.size(), keys.size()); ++i)
        ownership_changed |= old_owners[i].owner != product_owner(keys[i], shift);
    std::vector<int> destinations;
    for (const auto& [key, leaf] : source->owned())
        destinations.push_back(product_owner(key, shift));
    auto result = source->apply(source->plan_migration(destinations));
    EXPECT_EQ(snapshot(*source), before);
    if (ownership_changed)
        EXPECT_NE(result->leaf_layout(), source->leaf_layout());
    else
        EXPECT_EQ(result->leaf_layout(), source->leaf_layout());
    EXPECT_EQ(result->topology_generation(), source->topology_generation());
    EXPECT_EQ(result->partition_generation(), source->partition_generation() + std::uint64_t(ownership_changed));
    if (!ownership_changed)
        EXPECT_EQ(result->observation(), source->observation());
    const auto owners = result->lookup_cells(keys);
    EXPECT_EQ(owners.size(), old_owners.size());
    for (std::size_t i = 0; i < std::min(owners.size(), old_owners.size()); ++i)
    {
        EXPECT_EQ(owners[i].revision, old_owners[i].revision);
        EXPECT_EQ(owners[i].owner, product_owner(keys[i], shift));
    }
    expect_product(*result, base_split, axial_split);
    return result;
}
std::vector<AMR::AdaptiveBytes> product_families(AMR::AdaptiveSplit split, bool other_split)
{
    std::vector<AMR::AdaptiveBytes> result;
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getRank() != comm->getSize() - 1)
        return result;
    const unsigned count = !other_split ? 1 : split == AMR::AdaptiveSplit::Base ? 2 : 4;
    for (unsigned child = 0; child < count; ++child)
    {
        const auto path = other_split ? std::vector<std::uint64_t>{child} : std::vector<std::uint64_t>{};
        result.push_back(AMR::adaptive_family_key(
            split == AMR::AdaptiveSplit::Base ? product_key(1, 0, {}, path) : product_key(1, 0, path, {}), split));
    }
    return result;
}

TEST(IncrementalAdaptiveHierarchy, LeafLayoutSharesMapAndRetainsHistoricalVersion)
{
    auto source = product_source();
    const auto layout = source->leaf_layout();
    ASSERT_TRUE(layout);
    EXPECT_EQ(&layout->owned(), &source->owned());
    for (const auto& [key, leaf] : source->owned())
    {
        EXPECT_TRUE(layout->contains(key));
        EXPECT_EQ(layout->find(key), &leaf);
    }
    const auto absent = product_key(999, 0);
    EXPECT_FALSE(layout->contains(absent));
    EXPECT_EQ(layout->find(absent), nullptr);

    const auto before = snapshot(*source);
    auto draft = source->native_draft();
    EXPECT_EQ(draft.source_layout, layout);
    std::vector<const AMR::AdaptiveLeaf*> rows;
    for (const auto& [key, leaf] : source->owned())
        rows.push_back(&leaf);
    std::size_t used = 0;
    auto geometry = source->geometry_snapshot(
        rows, ordinary, AMR::GeometrySnapshotPurpose::State, used, 16 * 1024 * 1024);
    EXPECT_TRUE(geometry->matches_layout(layout));
    auto independent = product_source();
    EXPECT_NE(independent->leaf_layout(), layout);
    EXPECT_FALSE(geometry->matches_layout(independent->leaf_layout()));
    {
        const std::shared_ptr<const AMR::AdaptiveLeafLayout> alias(layout, independent->leaf_layout().get());
        EXPECT_FALSE(geometry->matches_layout(alias));
    }
    auto foreign_draft = draft;
    foreign_draft.source_layout = independent->leaf_layout();
    P::GeometryAllocationLedger ordinary_owners, bounded_owners;
    std::size_t rejected_used = 0;
    EXPECT_THROW((void) AMR::adaptive_detail::GeometrySnapshotAccess::native_draft(*source, foreign_draft, {},
                     source->observation(), ordinary, AMR::GeometrySnapshotPurpose::State, rejected_used,
                     16 * 1024 * 1024, ordinary_owners, bounded_owners),
        std::invalid_argument);
    {
        const std::shared_ptr<const AMR::AdaptiveLeafLayout> nonowning(layout.get(), [](const auto*) {});
        EXPECT_FALSE(geometry->matches_layout(nonowning));
        auto nonowning_draft = draft;
        nonowning_draft.source_layout = nonowning;
        rejected_used = 0;
        EXPECT_THROW((void) AMR::adaptive_detail::GeometrySnapshotAccess::native_draft(*source, nonowning_draft, {},
                         source->observation(), ordinary, AMR::GeometrySnapshotPurpose::State, rejected_used,
                         16 * 1024 * 1024, ordinary_owners, bounded_owners),
            std::invalid_argument);
    }
    auto native = AMR::assemble_adaptive_native(*source, source->configuration_descriptor());
    auto bound = source->with_partition(native.partition, native.mesh->owned_cell_map());
    EXPECT_EQ(bound->leaf_layout(), layout);
    EXPECT_EQ(&bound->owned(), &source->owned());
    EXPECT_TRUE(geometry->matches_layout(bound->leaf_layout()));
    std::size_t rebound_used = 0;
    EXPECT_EQ(bound->geometry_snapshot(
                  rows, ordinary, AMR::GeometrySnapshotPurpose::State, rebound_used, 16 * 1024 * 1024),
        geometry);

    auto unchanged = source->apply(source->plan_adaptation({}));
    EXPECT_EQ(unchanged->leaf_layout(), layout);
    auto refined = adapt(source, product_marks(*source, AMR::AdaptiveSplit::Base));
    EXPECT_NE(refined->leaf_layout(), layout);
    EXPECT_FALSE(geometry->matches_layout(refined->leaf_layout()));
    source.reset();
    independent.reset();
    EXPECT_EQ(&layout->owned(), &unchanged->owned());
    EXPECT_EQ(snapshot(*unchanged), before);
    unchanged.reset();
    bound.reset();
    native = {};
    draft.source_layout.reset();
    foreign_draft.source_layout.reset();
    EXPECT_TRUE(geometry->matches_layout(layout));
    std::vector<AMR::AdaptiveBytes> retained_leaves;
    for (const auto& [key, leaf] : layout->owned())
    {
        EXPECT_EQ(layout->find(key), &leaf);
        retained_leaves.push_back(AMR::encode_adaptive_leaf(leaf));
    }
    EXPECT_EQ(retained_leaves, before.leaves);
}

TEST(IncrementalAdaptiveHierarchy, GeometrySnapshotDoesNotRetainLeafLayout)
{
    std::weak_ptr<const AMR::AdaptiveLeafLayout> weak_layout;
    std::shared_ptr<const AMR::AdaptiveGeometrySnapshot> geometry;
    {
        const auto root = product_source();
        const auto source = adapt(root, product_marks(*root, AMR::AdaptiveSplit::Base));
        weak_layout = source->leaf_layout();
        std::vector<const AMR::AdaptiveLeaf*> rows;
        for (const auto& [key, leaf] : source->owned())
            rows.push_back(&leaf);
        std::size_t used = 0;
        geometry = source->geometry_snapshot(
            rows, ordinary, AMR::GeometrySnapshotPurpose::State, used, 16 * 1024 * 1024);
        EXPECT_FALSE(weak_layout.expired());
    }
    EXPECT_TRUE(weak_layout.expired());
    EXPECT_GT(geometry->retained_bytes(), 0u);
}

TEST(IncrementalAdaptiveHierarchy, BothProductFactorOrdersMigrateAndCoarsenIndependently)
{
    const auto root = product_source();
    const auto original = snapshot(*root);
    expect_product(*root, false, false);
    EXPECT_EQ(sum(root->owned().empty()), std::uint64_t(Tpetra::getDefaultComm()->getSize() - 1));
    for (const auto first : {AMR::AdaptiveSplit::Base, AMR::AdaptiveSplit::Axial})
    {
        SCOPED_TRACE(first == AMR::AdaptiveSplit::Base ? "Base then Axial" : "Axial then Base");
        const bool base_first = first == AMR::AdaptiveSplit::Base;
        const auto second = base_first ? AMR::AdaptiveSplit::Axial : AMR::AdaptiveSplit::Base;
        auto one = adapt(root, product_marks(*root, first));
        expect_product(*one, base_first, !base_first); // 19 or 17 leaves.
        one = migrate_product(one, base_first, !base_first, 1);
        auto both = adapt(one, product_marks(*one, second));
        expect_product(*both, true, true); // 23 leaves in either construction order.
        both = migrate_product(both, true, true, 0);
        // Remove the first-created factor first, requiring independent parents
        // synthesized in the other factor's new descendant space.
        auto remaining = adapt(both, {}, product_families(first, true));
        expect_product(*remaining, !base_first, base_first); // 17 or 19 leaves.
        remaining = migrate_product(remaining, !base_first, base_first, 1);
        auto restored = adapt(remaining, {}, product_families(second, false));
        expect_product(*restored, false, false);
        EXPECT_EQ(snapshot(*root), original);
        EXPECT_EQ(restored->apply(restored->plan_adaptation({}))->identity(), restored->identity());
    }
}

AMR::AdaptiveCellKey interval_key(std::uint64_t root, std::vector<std::uint64_t> path = {})
{
    return {0, AMR::ConvexCellAncestry{root, std::move(path)}};
}
AMR::AdaptiveLeaf interval(AMR::AdaptiveCellKey key, double lower, double upper, std::uint64_t revision = 0)
{
    AMR::AdaptiveLeaf result;
    result.key = std::move(key);
    result.domain = {'x'};
    result.revision = revision;
    result.geometry = P::ConvexCellGeometry<ordinary>::box({lower, 0, 0}, {upper, 1, 1});
    for (std::size_t face = 0; face < result.geometry->num_faces(); ++face)
    {
        auto support = AMR::encode_adaptive_key(result.key);
        support.push_back(char(face));
        result.surfaces.push_back({std::move(support), result.geometry->face_geometry(face), 1, true, "wall"});
    }
    return result;
}
class IntervalKernel final : public AMR::AdaptiveProviderKernel
{
    bool quarter_a_;
    mutable std::map<AMR::AdaptiveCellKey, std::uint64_t> split_calls_;

public:
    explicit IntervalKernel(bool quarter_a) : quarter_a_(quarter_a) {}
    AMR::AdaptiveBytes descriptor() const override { return {'x', char(quarter_a_)}; }
    P::ConvexGeometryPtr<ordinary> geometry(const AMR::AdaptiveLeaf& leaf) const override { return leaf.geometry; }
    std::vector<AMR::AdaptiveSurface> surfaces(const AMR::AdaptiveLeaf& leaf) const override { return leaf.surfaces; }
    AMR::AdaptiveContactCapabilities contact_capabilities(const AMR::AdaptiveLeaf&) const override
    {
        return {AMR::AdaptiveContactNormalization::PhysicalIdentity, false, false};
    }
    AMR::AdaptiveSplitResult split(const AMR::AdaptiveLeaf& parent, AMR::AdaptiveSplit split) const override
    {
        if (split != AMR::AdaptiveSplit::Isotropic)
            throw std::invalid_argument("Interval fixture supports one refinement factor.");
        ++split_calls_[parent.key];
        const auto& ancestry = std::get<AMR::ConvexCellAncestry>(parent.key.ancestry);
        const unsigned count = quarter_a_ && ancestry.root == 1 && ancestry.path.empty() ? 4 : 2;
        const auto bounds = parent.geometry->closed_contact_bounds();
        const double lower = double(bounds[0][0]), width = double(bounds[0][1] - bounds[0][0]) / count;
        AMR::AdaptiveSplitResult result;
        result.family.key = AMR::adaptive_family_key(parent.key, split);
        result.family.parent = parent;
        result.family.split = split;
        result.family.revision = parent.revision;
        for (unsigned child = 0; child < count; ++child)
        {
            auto key = parent.key;
            std::get<AMR::ConvexCellAncestry>(key.ancestry).path.push_back(child);
            auto leaf = interval(key, lower + child * width, lower + (child + 1) * width, parent.revision + 1);
            leaf.parent_family = result.family.key;
            result.family.children.push_back(key);
            result.children.push_back(std::move(leaf));
        }
        result.family.anchor = AMR::encode_adaptive_key(result.family.children.front());
        return result;
    }
    std::uint64_t split_calls(const AMR::AdaptiveCellKey& key) const
    {
        const auto found = split_calls_.find(key);
        return found == split_calls_.end() ? 0 : found->second;
    }
};
void own(std::vector<AMR::AdaptiveLeaf>& leaves, const AMR::AdaptiveLeaf& leaf, int owner)
{
    if (Tpetra::getDefaultComm()->getRank() == owner)
        leaves.push_back(leaf);
}
void expect_intervals(const Hierarchy& state, const std::vector<std::array<long double, 2>>& expected)
{
    std::vector<int> local(expected.size()), global(expected.size());
    for (const auto& [key, leaf] : state.owned())
    {
        const auto bounds = state.geometry(leaf)->closed_contact_bounds();
        EXPECT_EQ(bounds[1], (std::array<long double, 2>{0, 1}));
        EXPECT_EQ(bounds[2], (std::array<long double, 2>{0, 1}));
        const auto found = std::ranges::find(expected, bounds[0]);
        EXPECT_NE(found, expected.end());
        if (found != expected.end())
            ++local[found - expected.begin()];
    }
    Teuchos::reduceAll(*state.communicator(), Teuchos::REDUCE_SUM, int(local.size()), local.data(), global.data());
    EXPECT_EQ(global, std::vector<int>(expected.size(), 1));
    EXPECT_EQ(state.global_cells(), expected.size());
}

TEST(IncrementalAdaptiveHierarchy, ClosureCannotRefineAnOriginalSourceGenerationTwice)
{
    const auto comm = Tpetra::getDefaultComm();
    const int other = std::min(1, comm->getSize() - 1);
    auto kernel = std::make_shared<IntervalKernel>(true);
    const auto a = interval(interval_key(1), 0, 1), b = interval(interval_key(2), 1, 3);
    std::vector<AMR::AdaptiveLeaf> owned;
    own(owned, a, 0);
    own(owned, b, other);
    const auto source =
        std::make_shared<const Hierarchy>(std::move(owned), std::vector<AMR::AdaptiveFamily>{}, kernel, comm);
    const auto before = snapshot(*source);
    expect_intervals(*source, {{0, 1}, {1, 3}});
    // A's explicit four-way split creates width .25. The first closure split
    // of B creates widths 1 and 1; the nearest B child still exceeds ratio2.
    reject_closure_collectively([&] { (void) source->plan_adaptation(mark_owned(*source, a.key)); });
    EXPECT_EQ(sum(kernel->split_calls(b.key)), 1u);
    EXPECT_EQ(sum(kernel->split_calls(interval_key(2, {0}))), 0u);
    EXPECT_EQ(snapshot(*source), before);
    expect_intervals(*source, {{0, 1}, {1, 3}});
    // A separate valid plan proves the accepted source and cached closure can
    // still be used after the rejected multi-round candidate was destroyed.
    const auto retry = adapt(source, mark_owned(*source, b.key));
    expect_intervals(*retry, {{0, 1}, {1, 2}, {2, 3}});
    EXPECT_EQ(snapshot(*source), before);
}

TEST(IncrementalAdaptiveHierarchy, ClosureCannotUndoRequestedCoarseningAcrossOwners)
{
    const auto comm = Tpetra::getDefaultComm();
    const int other = std::min(1, comm->getSize() - 1);
    auto kernel = std::make_shared<IntervalKernel>(false);
    const auto parent = interval(interval_key(1), 0, 1), b = interval(interval_key(2), 1, 1.25);
    const auto split = kernel->split(parent, AMR::AdaptiveSplit::Isotropic);
    std::vector<AMR::AdaptiveLeaf> owned;
    own(owned, split.children[0], 0);
    own(owned, split.children[1], other);
    own(owned, b, other);
    std::vector<AMR::AdaptiveFamily> families;
    if (comm->getRank() == 0)
        families.push_back(split.family);
    const auto source = std::make_shared<const Hierarchy>(std::move(owned), std::move(families), kernel, comm);
    const auto before = snapshot(*source);
    expect_intervals(*source, {{0, .5}, {.5, 1}, {1, 1.25}});
    std::vector<AMR::AdaptiveBytes> coarsen;
    if (comm->getRank() == comm->getSize() - 1)
        coarsen.push_back(split.family.key);
    // Restoring A=[0,1] beside B of width.25 would require refining A again.
    // The accepted source contains A0/A1, so that would undo this coarsening.
    reject_closure_collectively([&] { (void) source->plan_adaptation({}, coarsen); });
    EXPECT_EQ(snapshot(*source), before);
    expect_intervals(*source, {{0, .5}, {.5, 1}, {1, 1.25}});
    const auto retry = adapt(source, mark_owned(*source, b.key));
    expect_intervals(*retry, {{0, .5}, {.5, .75}, {.75, 1}, {1, 1.125}, {1.125, 1.25}});
    EXPECT_EQ(snapshot(*source), before);
}
} // namespace
