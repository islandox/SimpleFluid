/** Hierarchy-level factor order, migration and rejected-closure source oracles. */
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "geometry/mesh/AMR/ProductFamilyElectionState.hh"
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
#include <optional>
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
    for (const auto& leaf : state.owned_leaves())
        result.leaves.push_back(AMR::encode_adaptive_leaf(leaf));
    for (const auto family : state.owned_families())
        result.families.push_back(AMR::encode_adaptive_family(family.materialize()));
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
    return state.owned_leaves().contains(key) ? std::vector{AMR::AdaptiveMark{key, AMR::AdaptiveSplit::Isotropic}}
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
class RevisionSensitiveProductKernel final : public AMR::AdaptiveProviderKernel
{
public:
    explicit RevisionSensitiveProductKernel(std::shared_ptr<const AMR::AdaptiveProviderKernel> product)
        : product_(std::move(product))
    {
    }

    AMR::AdaptiveBytes descriptor() const override
    {
        auto result = product_->descriptor();
        result.push_back('r');
        return result;
    }
    void validate(const AMR::AdaptiveLeaf& leaf) const override { product_->validate(leaf); }
    auto geometry(const AMR::AdaptiveLeaf& leaf) const -> P::ConvexGeometryPtr<ordinary> override
    {
        return product_->geometry(leaf);
    }
    std::vector<AMR::AdaptiveSurface> surfaces(const AMR::AdaptiveLeaf& leaf) const override
    {
        return product_->surfaces(leaf);
    }
    AMR::AdaptiveTrace traces(const AMR::AdaptiveLeaf& leaf) const override { return product_->traces(leaf); }
    std::vector<AMR::AdaptiveCellKey> matching_partners(const AMR::AdaptiveLeaf& leaf) const override
    {
        return product_->matching_partners(leaf);
    }
    AMR::AdaptiveSplit closure_split(const AMR::AdaptiveLeaf& leaf, std::size_t axis) const override
    {
        return product_->closure_split(leaf, axis);
    }
    AMR::AdaptiveContactCapabilities contact_capabilities(const AMR::AdaptiveLeaf& leaf) const override
    {
        return product_->contact_capabilities(leaf);
    }
    std::vector<AMR::AdaptiveLeaf> repack(std::span<const AMR::AdaptiveLeaf> rows) const override
    {
        return product_->repack(rows);
    }
    bool supports_leaf_local_repack() const noexcept override { return product_->supports_leaf_local_repack(); }
    std::vector<AMR::AdaptiveFamily> ancestry_families(const AMR::AdaptiveLeaf& leaf) const override
    {
        return product_->ancestry_families(leaf);
    }
    AMR::AdaptiveSplitResult split(const AMR::AdaptiveLeaf& parent, AMR::AdaptiveSplit split) const override
    {
        auto result = product_->split(parent, split);
        if (split != AMR::AdaptiveSplit::Base)
            return result;
        if (parent.revision == projected_revision)
        {
            ++projected_proof_checks;
            result.family.children.pop_back();
        }
        else
        {
            const auto anchor = AMR::decode_adaptive_key(result.family.anchor);
            for (auto& child : result.children)
                child.revision = child.key == anchor ? anchor_revision : projected_revision;
        }
        return result;
    }

    mutable std::uint64_t projected_proof_checks = 0;

private:
    static constexpr std::uint64_t anchor_revision = 12, projected_revision = 17;
    std::shared_ptr<const AMR::AdaptiveProviderKernel> product_;
};

State product_source(std::shared_ptr<const RevisionSensitiveProductKernel>* sensitive = nullptr,
    std::shared_ptr<const AMR::AdaptiveProviderKernel>* kernel_out = nullptr)
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
    if (!detached.owned.empty())
        EXPECT_TRUE(detached.kernel->product_family_proof_revision_invariant(&detached.owned.front()));
    if (sensitive)
    {
        *sensitive = std::make_shared<const RevisionSensitiveProductKernel>(std::move(detached.kernel));
        detached.kernel = *sensitive;
        if (!detached.owned.empty())
            EXPECT_FALSE(detached.kernel->product_family_proof_revision_invariant(&detached.owned.front()));
    }
    if (kernel_out)
        *kernel_out = detached.kernel;
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
    for (const auto& leaf : state.owned_leaves())
    {
        const auto key_position = std::ranges::find(expected_keys, leaf.key);
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
    // The historical full election remains an independent authority oracle.
    // Wire equality also checks anchor, both revisions and complete saved
    // parent/children data including native provider recipes and geometry.
    const auto reference = AMR::adaptive_detail::product_family_reference(state);
    std::vector<AMR::AdaptiveBytes> expected_families, actual_families;
    for (const auto& family : reference)
        expected_families.push_back(AMR::encode_adaptive_family(family));
    for (const auto family : state.owned_families())
    {
        EXPECT_EQ(family.parent.provider, AMR::AdaptiveProviderKind::Product);
        EXPECT_EQ(family.revision, state.topology_generation());
        actual_families.push_back(AMR::encode_adaptive_family(family.materialize()));
    }
    EXPECT_EQ(actual_families, expected_families);
}
std::vector<AMR::AdaptiveMark> product_marks(const Hierarchy& state, AMR::AdaptiveSplit split)
{
    std::vector<AMR::AdaptiveMark> result;
    for (const auto& leaf : state.owned_leaves())
    {
        const auto& cell = std::get<AMR::ProductCellAncestry>(leaf.key.ancestry);
        if (cell.base.root == 1 && cell.axial.root == 0)
            result.push_back({leaf.key, split});
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
    for (const auto& leaf : source->owned_leaves())
        destinations.push_back(product_owner(leaf.key, shift));
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
    std::vector<AMR::AdaptiveBytes> families;
    std::vector<int> expected_authorities;
    for (const auto family : source->owned_families())
    {
        families.push_back(family.key);
        expected_authorities.push_back(product_owner(AMR::decode_adaptive_key(family.anchor), shift));
    }
    const auto authorities = result->lookup_families(families);
    EXPECT_EQ(authorities.size(), expected_authorities.size());
    for (std::size_t i = 0; i < std::min(authorities.size(), expected_authorities.size()); ++i)
        EXPECT_EQ(authorities[i].owner, expected_authorities[i]);
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
    EXPECT_EQ(layout.get(), &source->owned_leaves());
    for (const auto& leaf : source->owned_leaves())
    {
        EXPECT_TRUE(layout->contains(leaf.key));
        EXPECT_EQ(layout->find(leaf.key), &leaf);
    }
    const auto absent = product_key(999, 0);
    EXPECT_FALSE(layout->contains(absent));
    EXPECT_EQ(layout->find(absent), nullptr);

    const auto before = snapshot(*source);
    auto draft = source->native_draft();
    EXPECT_EQ(draft.source_layout, layout);
    std::vector<const AMR::AdaptiveLeaf*> rows;
    for (const auto& leaf : source->owned_leaves())
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
    EXPECT_EQ(&bound->owned_leaves(), &source->owned_leaves());
    EXPECT_EQ(&bound->owned_families(), &source->owned_families());
    EXPECT_TRUE(geometry->matches_layout(bound->leaf_layout()));
    std::size_t rebound_used = 0;
    EXPECT_EQ(bound->geometry_snapshot(
                  rows, ordinary, AMR::GeometrySnapshotPurpose::State, rebound_used, 16 * 1024 * 1024),
        geometry);

    auto unchanged = source->apply(source->plan_adaptation({}));
    EXPECT_EQ(unchanged->leaf_layout(), layout);
    auto refined = adapt(source, product_marks(*source, AMR::AdaptiveSplit::Base));
    const auto family_key = AMR::adaptive_family_key(product_key(1, 0), AMR::AdaptiveSplit::Base);
    const auto fetched = refined->fetch_families(std::span(&family_key, 1));
    EXPECT_EQ(fetched.size(), 1u);
    if (!fetched.empty())
    {
        EXPECT_EQ(fetched.front().revision, refined->topology_generation());
        if (const auto local = refined->owned_families().find(family_key))
            EXPECT_EQ(AMR::encode_adaptive_family(local->materialize()), AMR::encode_adaptive_family(fetched.front()));
    }
    EXPECT_NE(refined->leaf_layout(), layout);
    EXPECT_FALSE(geometry->matches_layout(refined->leaf_layout()));
    source.reset();
    independent.reset();
    EXPECT_EQ(layout.get(), &unchanged->owned_leaves());
    EXPECT_EQ(snapshot(*unchanged), before);
    unchanged.reset();
    bound.reset();
    native = {};
    draft.source_layout.reset();
    foreign_draft.source_layout.reset();
    EXPECT_TRUE(geometry->matches_layout(layout));
    std::vector<AMR::AdaptiveBytes> retained_leaves;
    for (const auto& leaf : *layout)
    {
        EXPECT_EQ(layout->find(leaf.key), &leaf);
        retained_leaves.push_back(AMR::encode_adaptive_leaf(leaf));
    }
    EXPECT_EQ(retained_leaves, before.leaves);
}

TEST(IncrementalAdaptiveHierarchy, NonemptyFamilyBindingAndRankLocalPublicationRejection)
{
    namespace detail = AMR::adaptive_detail;
    using Storage = detail::AdaptiveFamilyStorage;
    const auto source = product_source();
    const auto refined = adapt(source, product_marks(*source, AMR::AdaptiveSplit::Base));
    const auto comm = refined->communicator();
    const bool tight_rank = comm->getRank() == 0;
    auto native = AMR::assemble_adaptive_native(*refined, refined->configuration_descriptor());
    const auto bound = refined->with_partition(native.partition, native.mesh->owned_cell_map());
    const auto& layout = bound->owned_families();
    EXPECT_EQ(&layout, &refined->owned_families());
    EXPECT_EQ(layout.empty(), !tight_rank);
    for (const auto family : layout)
    {
        const auto old = refined->owned_families().at(family.key);
        EXPECT_EQ(&family.parent, &old.parent);
        EXPECT_EQ(&family.children, &old.children);
        EXPECT_EQ(&family.anchor, &old.anchor);
        EXPECT_EQ(family.revision, old.revision);
    }

    const auto before = snapshot(*bound);
    const auto leaf_root = bound->leaf_layout();
    const auto layout_bytes = Storage::storage_bytes(layout);
    const auto full_cap = bound->options().byte_budget;
    std::vector<AMR::AdaptiveOwner> registrations;
    for (const auto family : layout)
        registrations.push_back({family.key, comm->getRank(), family.revision});

    // Use the real private family seal and leased directory registration on
    // this accepted hierarchy. The fixture chooses its rank-local admission
    // limits without adding a production failure injection hook.
    std::optional<AMR::AdaptiveFamily> replacement;
    std::vector<Storage::Edit> edits;
    auto seal_cap = full_cap;
    detail::collective(*comm,
        [&]
        {
            if (!tight_rank)
                return;
            if (layout.empty())
                throw std::logic_error("Budget fixture requires a saved family on rank zero.");
            const auto saved = *layout.begin();
            replacement.emplace(saved.materialize());
            replacement->anchor = AMR::encode_adaptive_key(replacement->children.back());
            edits.push_back({replacement->key, saved.revision, &*replacement});
            auto probe = std::make_shared<detail::Budget>(detail::Budget{full_cap});
            Storage::Work work;
            const auto accepted = Storage::edited(
                layout, edits, probe, bound->topology_generation(), work, [](const auto*, const auto*) {});
            seal_cap = probe->peak - 1;
        });

    auto seal_memory = std::make_shared<detail::Budget>(detail::Budget{seal_cap});
    std::unique_ptr<AMR::AdaptiveFamilyLayout> uncommitted;
    bool local_seal_rejected = false;
    Storage::Work work;
    EXPECT_THROW(detail::collective(*comm,
                     [&]
                     {
                         try
                         {
                             uncommitted = std::make_unique<AMR::AdaptiveFamilyLayout>(Storage::edited(layout, edits,
                                 seal_memory, bound->topology_generation(), work, [](const auto*, const auto*) {}));
                         }
                         catch (const std::invalid_argument&)
                         {
                             local_seal_rejected = true;
                             throw;
                         }
                     }),
        std::invalid_argument);
    EXPECT_EQ(local_seal_rejected, tight_rank);
    EXPECT_EQ(bool(uncommitted), !tight_rank);
    uncommitted.reset();
    EXPECT_EQ(seal_memory->used, 0u);
    EXPECT_EQ(&bound->owned_families(), &layout);
    EXPECT_EQ(Storage::storage_bytes(layout), layout_bytes);
    EXPECT_EQ(snapshot(*bound), before);

    const auto directory_cap = tight_rank ? std::size_t{1} : full_cap;
    EXPECT_THROW(
        (void) AMR::DistributedAncestryDirectory(registrations, comm, full_cap, directory_cap), std::invalid_argument);
    EXPECT_EQ(&bound->owned_families(), &layout);
    EXPECT_EQ(bound->leaf_layout(), leaf_root);
    EXPECT_EQ(snapshot(*bound), before);
    const auto key = AMR::adaptive_family_key(product_key(1, 0), AMR::AdaptiveSplit::Base);
    const auto owner = bound->lookup_families(std::span(&key, 1));
    EXPECT_EQ(owner.size(), 1u);
    if (!owner.empty())
    {
        EXPECT_EQ(owner.front().owner, 0);
        EXPECT_EQ(owner.front().revision, bound->topology_generation());
    }
}

TEST(IncrementalAdaptiveHierarchy, RankSelectiveProductShardAdmissionRejectsAfterOneEdit)
{
    namespace detail = AMR::adaptive_detail;
    using Election = detail::ProductFamilyElectionState;
    const auto root = product_source();
    const auto accepted = adapt(root, product_marks(*root, AMR::AdaptiveSplit::Base));
    const auto comm = accepted->communicator();
    const auto key = AMR::adaptive_family_key(product_key(1, 0), AMR::AdaptiveSplit::Base);
    const auto exact = accepted->fetch_families(std::span(&key, 1));
    const auto snapshot_before = snapshot(*accepted);
    const auto leaf_root = accepted->leaf_layout();
    const auto* family_root = &accepted->owned_families();
    const auto shard = detail::authority(key, comm->getSize());
    const bool tight_rank = comm->getRank() == shard;
    const auto cap = accepted->options().byte_budget;
    auto source_memory = std::make_shared<detail::Budget>(detail::Budget{cap});
    Election source(source_memory);
    std::optional<detail::AdaptiveFamilyProofFragment> fragment;
    std::optional<AMR::AdaptiveCellKey> anchor;
    std::uint64_t revision = 0;

    detail::collective(*comm,
        [&]
        {
            if (!tight_rank)
                return;
            if (exact.size() != 1)
                throw std::logic_error("Product shard fixture lacks one actual family.");
            fragment.emplace(detail::encode_family_proof_fragment(exact.front(), cap));
            anchor.emplace(AMR::decode_adaptive_key(exact.front().anchor));
            revision = exact.front().parent.revision;
            const auto proof = source.proof(fragment->view());
            const std::array rows{source.proposal(0, *anchor, revision, revision, proof)};
            source.set_proposals(key, rows);
        });
    const auto* accepted_proposal = source.winner(key).proposal;
    const auto initial_bytes = source_memory->used;

    const auto first_edit = [&](Election& candidate)
    {
        const auto proof = candidate.winner(key).proposal->proof;
        const std::array rows{candidate.proposal(0, *anchor, revision, revision + 1, proof)};
        candidate.set_proposals(key, rows);
        EXPECT_EQ(candidate.winner(key).proposal->proof, proof);
    };
    const auto second_edit = [&](Election& candidate)
    {
        const auto proof = candidate.proof(fragment->view());
        const std::array rows{candidate.proposal(0, *anchor, revision, revision + 2, proof)};
        candidate.set_proposals(key, rows);
    };

    auto limit = cap;
    detail::collective(*comm,
        [&]
        {
            if (!tight_rank)
                return;
            auto probe_memory = std::make_shared<detail::Budget>(detail::Budget{cap});
            Election probe(probe_memory, &source);
            first_edit(probe);
            const auto first_peak = probe_memory->peak;
            second_edit(probe);
            if (probe_memory->peak <= first_peak)
                throw std::logic_error("Product shard fixture did not increase its second-edit peak.");
            limit = probe_memory->peak - 1;
        });

    auto memory = std::make_shared<detail::Budget>(detail::Budget{limit});
    std::unique_ptr<Election> candidate;
    bool first_completed = false, second_completed = false;
    EXPECT_THROW(detail::collective(*comm,
                     [&]
                     {
                         candidate = std::make_unique<Election>(memory, &source);
                         if (tight_rank)
                         {
                             first_edit(*candidate);
                             first_completed = true;
                             second_edit(*candidate);
                             second_completed = true;
                         }
                     }),
        std::invalid_argument);
    EXPECT_EQ(first_completed, tight_rank);
    EXPECT_FALSE(second_completed);
    candidate.reset();
    EXPECT_EQ(memory->used, 0u);
    EXPECT_EQ(source_memory->used, initial_bytes);
    EXPECT_EQ(source.winner(key).proposal, accepted_proposal);
    EXPECT_EQ(accepted->leaf_layout(), leaf_root);
    EXPECT_EQ(&accepted->owned_families(), family_root);
    EXPECT_EQ(snapshot(*accepted), snapshot_before);
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
        for (const auto& leaf : source->owned_leaves())
            rows.push_back(&leaf);
        std::size_t used = 0;
        geometry = source->geometry_snapshot(
            rows, ordinary, AMR::GeometrySnapshotPurpose::State, used, 16 * 1024 * 1024);
        EXPECT_FALSE(weak_layout.expired());
    }
    EXPECT_TRUE(weak_layout.expired());
    EXPECT_GT(geometry->retained_bytes(), 0u);
}

TEST(IncrementalAdaptiveHierarchy, CustomProductRevalidatesAtTheProjectedParentRevision)
{
    std::shared_ptr<const RevisionSensitiveProductKernel> kernel;
    auto source = product_source(&kernel);
    // Keep one actual accepted family while a different local Product root
    // attempts a Base split with distinct anchor and maximum revisions.
    const auto axial = product_key(0, 0);
    std::vector<AMR::AdaptiveMark> prior;
    if (source->owned_leaves().contains(axial))
        prior.push_back({axial, AMR::AdaptiveSplit::Axial});
    auto accepted = adapt(source, prior);
    EXPECT_GT(sum(accepted->owned_families().size()), 0u);
    const auto before = snapshot(*accepted);
    const auto leaf_root = accepted->leaf_layout();
    const auto* family_root = &accepted->owned_families();

    std::vector<AMR::AdaptiveMark> mark;
    const auto target = product_key(3, 0);
    if (accepted->owned_leaves().contains(target))
        mark.push_back({target, AMR::AdaptiveSplit::Base});
    bool rejected = false;
    try
    {
        (void) accepted->plan_adaptation(mark);
    }
    catch (const std::invalid_argument& error)
    {
        rejected = true;
        if (accepted->communicator()->getRank() == 0)
            EXPECT_EQ(
                std::string_view(error.what()), "Family capsule does not name complete immediate provider children.");
    }
    EXPECT_EQ(sum(rejected), std::uint64_t(accepted->communicator()->getSize()));
    EXPECT_GT(sum(kernel->projected_proof_checks), 0u);
    EXPECT_EQ(accepted->leaf_layout(), leaf_root);
    EXPECT_EQ(&accepted->owned_families(), family_root);
    EXPECT_EQ(snapshot(*accepted), before);
}

TEST(IncrementalAdaptiveHierarchy, BothProductFactorOrdersMigrateAndCoarsenIndependently)
{
    const auto root = product_source();
    const auto original = snapshot(*root);
    expect_product(*root, false, false);
    EXPECT_EQ(sum(root->owned_leaves().empty()), std::uint64_t(Tpetra::getDefaultComm()->getSize() - 1));
    for (const auto first : {AMR::AdaptiveSplit::Base, AMR::AdaptiveSplit::Axial})
    {
        SCOPED_TRACE(first == AMR::AdaptiveSplit::Base ? "Base then Axial" : "Axial then Base");
        const bool base_first = first == AMR::AdaptiveSplit::Base;
        const auto second = base_first ? AMR::AdaptiveSplit::Axial : AMR::AdaptiveSplit::Base;
        auto one = adapt(root, product_marks(*root, first));
        expect_product(*one, base_first, !base_first); // 19 or 17 leaves.
        const auto accepted = snapshot(*one);
        const std::array absent{AMR::adaptive_family_key(product_key(99, 0), first)};
        EXPECT_THROW((void) one->plan_adaptation({}, absent), std::invalid_argument);
        EXPECT_EQ(snapshot(*one), accepted);
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
    for (const auto& leaf : state.owned_leaves())
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

TEST(IncrementalAdaptiveHierarchy, BootstrapSeedsElectedProductPartsFromPublishedFamilies)
{
    const auto comm = Tpetra::getDefaultComm();
    std::shared_ptr<const AMR::AdaptiveProviderKernel> kernel;
    const auto source = product_source(nullptr, &kernel);
    const auto adapted = adapt(source, product_marks(*source, AMR::AdaptiveSplit::Base));
    std::vector<AMR::AdaptiveLeaf> owned(adapted->owned_leaves().begin(), adapted->owned_leaves().end());
    std::vector<AMR::AdaptiveFamily> families;
    for (const auto family : adapted->owned_families())
        families.push_back(family.materialize());
    EXPECT_GT(sum(families.size()), 0u);
    // Reconstruct directly from non-empty Product families: bootstrap must seed
    // the elected-part index from its published family controls.
    const auto bootstrapped =
        std::make_shared<const Hierarchy>(std::move(owned), std::move(families), kernel, comm);
    std::uint64_t boot_product_families = 0;
    for (const auto family : bootstrapped->owned_families())
        boot_product_families += family.parent.provider == AMR::AdaptiveProviderKind::Product;
    EXPECT_GT(sum(boot_product_families), 0u);
    // An incremental Product coarsening must retire the elected part that
    // bootstrap published; it must not report an unrecorded retirement.
    const auto next = adapt(bootstrapped, {}, product_families(AMR::AdaptiveSplit::Base, false));
    // Coarsening the fixture's sole Base family restores root leaves, whose
    // empty ancestry paths have no remaining Product family to elect.
    EXPECT_EQ(sum(next->owned_families().size()), 0u);
    expect_product(*next, false, false);

    // Exercise the same bootstrap references across an elected-owner change
    // before their final retirement. Serial keeps the identity migration;
    // MPI moves the typed minimum anchor and checks its directory authority.
    const auto migrated = migrate_product(bootstrapped, true, false, 1);
    const auto retired = adapt(migrated, {}, product_families(AMR::AdaptiveSplit::Base, false));
    EXPECT_EQ(sum(retired->owned_families().size()), 0u);
    expect_product(*retired, false, false);
}
} // namespace
