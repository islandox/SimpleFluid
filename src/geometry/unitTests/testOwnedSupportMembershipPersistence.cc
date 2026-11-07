/** Keeper-backed owned membership storage and exact native inventory transitions. */
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "geometry/mesh/AMR/IncidencePhysicalOwners.hh"
#include "geometry/mesh/AMR/OwnedSupportMembership.hh"
#include <array>
#include <gtest/gtest.h>
#include <map>

namespace
{
namespace A = SimpleFluid::Meshes::AMR;
namespace D = A::adaptive_detail;
namespace P = SimpleFluid::PLIC;
constexpr auto ordinary = SimpleFluid::ArithmeticMode::Ordinary;
constexpr std::size_t ample = 16 * 1024 * 1024;
struct State
{
    std::shared_ptr<D::Budget> budget = std::make_shared<D::Budget>(D::Budget{ample});
    D::IdentityBudget identities;
    D::IncidencePools pools;
    std::shared_ptr<D::ContactPhysicalResidency> residency;
    std::shared_ptr<D::CartesianAuthorityArena> arena;
    D::OwnedSupportMembership membership;
    explicit State(bool observed = false)
        : pools(*budget, identities, budget, D::IdentityPoolStorage::Persistent),
          residency(observed ? std::make_shared<D::ContactPhysicalResidency>(budget) : nullptr),
          arena(D::CartesianGroupAuthority::make_arena(*budget, budget)),
          membership(*budget, pools, budget, residency.get())
    {
        D::IncidencePhysicalOwners::bind_policy(arena, residency);
    }
    explicit State(const State& source)
        : pools(source.pools, *budget, identities, budget),
          residency(
              source.residency ? std::make_shared<D::ContactPhysicalResidency>(*source.residency, budget) : nullptr),
          arena(D::CartesianGroupAuthority::make_arena(*budget, budget)),
          membership(source.membership, *budget, pools, arena, budget, residency.get())
    {
        D::IncidencePhysicalOwners::bind_policy(arena, residency);
    }
};

A::AdaptiveLeaf explicit_leaf(std::size_t id, A::AdaptiveBytes support)
{
    A::AdaptiveLeaf leaf;
    leaf.key = {3, A::ConvexCellAncestry{id, {2, 5}}};
    leaf.domain = {'d'};
    leaf.revision = 11;
    const auto x = support.size() > 1 ? 10.0 + 3.0 * static_cast<double>(id) : 0.0;
    leaf.geometry = P::ConvexCellGeometry<ordinary>::box({x, 0, 0}, {x + 1, 1, 1});
    A::AdaptiveSurface surface;
    surface.key = std::move(support);
    surface.geometry = leaf.geometry->face_geometry(0);
    surface.orientation = 1;
    surface.exterior = true;
    surface.boundary = "wall";
    leaf.surfaces.push_back(std::move(surface));
    return leaf;
}
A::AdaptiveBytes support(std::size_t id)
{
    return {'s', static_cast<char>(id & 255), static_cast<char>((id >> 8) & 255)};
}
using Identity = std::pair<std::uintptr_t, D::ContactPayloadRole>;
std::map<Identity, std::size_t> payload(const D::OwnedSupportMembership& membership)
{
    std::map<Identity, std::size_t> result;
    membership.visit_borrowed_payload(
        [&](const D::ContactPayloadAllocation& allocation)
        {
            if (allocation.identity.role == D::ContactPayloadRole::OwnedMembershipController)
                return;
            result.emplace(
                Identity{reinterpret_cast<std::uintptr_t>(allocation.identity.address), allocation.identity.role},
                allocation.payload_bytes);
        });
    return result;
}
std::size_t independent_maximum(const State& state)
{
    std::size_t result = 1;
    state.membership.visit_explicit_cells(
        [&](auto slot, const auto& cell)
        {
            const auto ancestry = D::key_capacity(state.pools.cells.get(slot));
            for (const auto& member : cell.members)
                result = std::max(result, ancestry + state.pools.supports.get(member.support).size());
        });
    return result;
}
} // namespace

TEST(OwnedSupportMembershipPersistence, FixedEditSharesDoubledUnrelatedRowsAndUpdatesTheMaximum)
{
    struct Work
    {
        std::size_t pages, bytes, shared;
    };
    auto measure = [](std::size_t count)
    {
        State source;
        const auto changed = source.membership.add(explicit_leaf(0, A::AdaptiveBytes(145, 'm')));
        for (std::size_t i = 1; i < count; ++i)
            (void) source.membership.add(explicit_leaf(i, support(i)));
        EXPECT_EQ(source.membership.maximum_change_capacity(), independent_maximum(source));
        const auto prior = payload(source.membership);
        const auto source_used = source.budget->used;
        const auto* clean = &source.membership.at(count - 1);
        State candidate(source);
        EXPECT_EQ(candidate.membership.persistence_work().legacy_fork_visits, 0u);
        EXPECT_EQ(candidate.membership.persistence_work().copied_pages, 0u);
        EXPECT_EQ(payload(candidate.membership), prior);
        candidate.membership.erase(changed);
        EXPECT_EQ(candidate.membership.maximum_change_capacity(), independent_maximum(candidate));
        EXPECT_LT(candidate.membership.maximum_change_capacity(), source.membership.maximum_change_capacity());
        EXPECT_EQ(&candidate.membership.at(count - 1), clean);
        EXPECT_EQ(payload(source.membership), prior);
        EXPECT_EQ(source.budget->used, source_used);
        const auto overlap = candidate.membership.inherited_overlap_metadata_bytes();
        const auto before = candidate.budget->used;
        EXPECT_GT(overlap, 0u);
        candidate.membership.finish_inherited_overlap();
        EXPECT_EQ(candidate.budget->used + overlap, before);

        Work result{};
        for (const auto& [id, bytes] : payload(candidate.membership))
        {
            if (prior.contains(id))
                result.shared += bytes;
            else
                result.bytes += bytes;
        }
        result.pages = candidate.membership.persistence_work().copied_pages;
        EXPECT_GT(result.shared, result.bytes);
        return result;
    };
    const auto small = measure(512), large = measure(1024);
    EXPECT_GT(large.shared, small.shared);
    EXPECT_LE(large.pages, small.pages + 4);
    EXPECT_LE(large.bytes, small.bytes + 4096);
}

TEST(OwnedSupportMembershipPersistence, ThreeGenerationsDoNotPinRetiredSpillsOrAncestorControllers)
{
    auto source = std::make_unique<State>();
    const auto first = source->membership.add(explicit_leaf(0, {'x'}));
    const auto second = source->membership.add(explicit_leaf(1, {'x'}));
    const auto third = source->membership.add(explicit_leaf(2, {'x'}));
    const auto support_slot = *source->pools.supports.find({'x'});
    const auto& old = *source->membership.support_cells(support_slot);
    ASSERT_TRUE(old.spill());
    const std::weak_ptr<const D::OwnedSupportMembership::SupportCells::Spill> spill(old.spill());
    const auto* clean = &source->membership.at(third);
    auto candidate = std::make_unique<State>(*source);
    candidate->membership.erase(first);
    candidate->membership.finish_inherited_overlap();
    State current(*candidate);
    current.membership.erase(second);
    current.membership.finish_inherited_overlap();
    EXPECT_FALSE(spill.expired());
    EXPECT_EQ(source->membership.support_cells(support_slot)->size(), 3u);
    EXPECT_EQ(candidate->membership.support_cells(support_slot)->size(), 2u);
    source.reset();
    candidate.reset();
    EXPECT_TRUE(spill.expired());
    EXPECT_EQ(&current.membership.at(third), clean);
    ASSERT_EQ(current.membership.support_cells(support_slot)->size(), 1u);
    EXPECT_EQ((*current.membership.support_cells(support_slot))[0], third);
    EXPECT_FALSE(current.membership.support_cells(support_slot)->spill());
}

TEST(OwnedSupportMembershipPersistence, AggregateForkNeedsExactCreditAndOneByteShortPreservesSource)
{
    State source;
    for (std::size_t i = 0; i < 128; ++i)
        (void) source.membership.add(explicit_leaf(i, support(i)));
    const auto source_used = source.budget->used;
    const auto before = payload(source.membership);
    auto fork = [&](std::optional<std::size_t> credit)
    {
        auto budget = std::make_shared<D::Budget>(D::Budget{ample});
        D::IdentityBudget identities;
        D::IncidencePools pools(source.pools, *budget, identities, budget);
        const auto pool_floor = budget->used;
        if (credit)
            budget->limit = pool_floor + *credit;
        if (credit && *credit < source.membership.retained_bytes())
        {
            EXPECT_THROW(
                (void) D::OwnedSupportMembership(source.membership, *budget, pools, {}, budget), std::invalid_argument);
            EXPECT_EQ(budget->used, pool_floor);
            return std::size_t(0);
        }
        std::size_t actual;
        {
            D::OwnedSupportMembership copy(source.membership, *budget, pools, {}, budget);
            actual = budget->used - pool_floor;
            EXPECT_EQ(payload(copy), before);
        }
        EXPECT_EQ(budget->used, pool_floor);
        return actual;
    };
    const auto exact = fork({});
    EXPECT_EQ(exact, source.membership.retained_bytes());
    EXPECT_EQ(fork(exact), exact);
    EXPECT_EQ(fork(exact - 1), 0u);
    EXPECT_EQ(source.budget->used, source_used);
    EXPECT_EQ(payload(source.membership), before);
}

TEST(OwnedSupportMembershipPersistence, AuthenticatedInventoryChecksOnlyStagedSelectors)
{
    const A::AMRGrid grid{};
    const A::CartesianCellAncestry cell{0, {0, 0, 0}}, other{0, {1, 0, 0}};
    auto native = A::cartesian_adaptive_leaf(grid, {grid.region_id, cell}, {'d'});
    const auto foreign = A::cartesian_adaptive_leaf(grid, {grid.region_id, other}, {'d'});
    const D::ContactHaloContext initial{0, 0, 0, 0, 1}, next{0, 0, 0, 1, 1};
    State source(true);
    for (std::size_t i = 0; i < 128; ++i)
        (void) source.membership.add(explicit_leaf(i, support(i)));
    const auto original = D::CartesianIncidenceSource::make(
        grid, std::span(&native, 1), native.domain, 0, 1, initial, *source.budget, source.budget);
    const auto witness = D::CartesianIncidenceSource::evaluate_cell(original, cell, *source.budget, source.budget);
    const auto owned_slot = source.membership.add(*witness);
    const std::array first{D::CartesianSourceInventory::CapturedSource{original, witness->evaluation_state()}};
    const auto accepted =
        D::CartesianSourceInventory::make(initial, 0, 1, first, *source.budget, source.budget, source.arena);
    source.membership.publish_inventory(accepted, source.arena);
    ASSERT_EQ(source.membership.explicit_cell_count(), 128u);
    ASSERT_EQ(source.membership.source_cell_count(), 1u);

    State candidate(source);
    const auto source_owner =
        D::ContactOwnerUsage::Key{accepted.get(), std::uint32_t(D::ContactPhysicalResidency::Role::SourceDependency)};
    ASSERT_TRUE(candidate.residency->usage().find(source_owner));
    EXPECT_EQ(candidate.residency->usage().find(source_owner)->references, 1u);
    candidate.membership.erase(owned_slot);
    ++native.revision;
    const auto replacement = D::CartesianIncidenceSource::make(
        grid, std::span(&native, 1), native.domain, 0, 2, next, *candidate.budget, candidate.budget);
    const auto changed =
        D::CartesianIncidenceSource::evaluate_cell(replacement, cell, *candidate.budget, candidate.budget);
    EXPECT_EQ(candidate.membership.add(*changed), owned_slot);
    const std::array retired{D::CartesianSourceInventory::Retirement{native.key, native.revision - 1}};
    const auto unregistered = D::CartesianIncidenceSource::make(
        grid, std::span(&foreign, 1), foreign.domain, 0, 2, next, *candidate.budget, candidate.budget);
    const std::array bad{D::CartesianSourceInventory::CapturedSource{unregistered, changed->evaluation_state()}};
    const auto rejected = accepted->derive(next, 2, retired, bad, *candidate.budget, candidate.budget, candidate.arena);
    EXPECT_THROW(candidate.membership.publish_inventory(rejected, candidate.arena), std::invalid_argument);
    EXPECT_EQ(candidate.membership.at(owned_slot).revision, native.revision);

    const std::array added{D::CartesianSourceInventory::CapturedSource{replacement, changed->evaluation_state()}};
    const auto derived =
        accepted->derive(next, 2, retired, added, *candidate.budget, candidate.budget, candidate.arena);
    const auto before = candidate.membership.persistence_work().inventory_validation_visits;
    candidate.membership.publish_inventory(derived, candidate.arena);
    EXPECT_EQ(candidate.membership.persistence_work().inventory_validation_visits - before, 2u);
    EXPECT_EQ(candidate.membership.explicit_cell_count(), 128u);
    EXPECT_EQ(candidate.membership.source_cell_count(), 1u);
    EXPECT_EQ(candidate.membership.leaf_revision(owned_slot), native.revision);
    EXPECT_EQ(source.membership.leaf_revision(owned_slot), native.revision - 1);
    EXPECT_TRUE(source.residency->usage().find(source_owner));
    EXPECT_FALSE(candidate.residency->usage().find(source_owner));
}


TEST(OwnedSupportMembershipPersistence, OwnedArenaKeeperOutlivesDeferredAccounting)
{
    State independent_pools;
    for (const bool populated : {false, true})
    {
        SCOPED_TRACE(populated);
        std::optional<std::size_t> final_used;
        auto memory = std::shared_ptr<D::Budget>(new D::Budget{ample},
            [&](D::Budget* value)
            {
                final_used = value->used;
                delete value;
            });
        const std::weak_ptr<D::Budget> lifetime = memory;
        auto membership = std::make_unique<D::OwnedSupportMembership>(*memory, independent_pools.pools, memory);
        if (populated)
        {
            (void) membership->add(explicit_leaf(0, {'x'}));
            (void) membership->add(explicit_leaf(1, {'x'}));
            (void) membership->add(explicit_leaf(2, {'x'}));
        }
        memory.reset();
        EXPECT_FALSE(lifetime.expired());
        membership.reset();
        EXPECT_TRUE(lifetime.expired());
        EXPECT_EQ(final_used, 0u);
    }
}


TEST(OwnedSupportMembershipPersistence, MultiSupportMutationNeedsExactCreditAndDiscardsRejectedCandidate)
{
    State source;
    const auto first = source.membership.add(explicit_leaf(0, {'x'}));
    const auto second = source.membership.add(explicit_leaf(1, {'x'}));
    auto inserted = explicit_leaf(2, {'x'});
    for (std::size_t i = 1; i < 3; ++i)
    {
        auto face = inserted.surfaces.front();
        face.key = {static_cast<char>('a' + i)};
        face.geometry = inserted.geometry->face_geometry(i);
        inserted.surfaces.push_back(std::move(face));
    }
    // Put typed identities in the accepted pools without activating membership.
    // The admission under test is the row, reverse spill and index mutation.
    const auto inserted_slot = source.pools.cells.intern(inserted.key);
    for (const auto& face : inserted.surfaces)
        (void) source.pools.supports.intern(face.key);
    const auto support_slot = *source.pools.supports.find({'x'});
    const auto retained = source.budget->used;
    const auto accepted = payload(source.membership);

    const auto run = [&](std::optional<std::size_t> cap, bool reject)
    {
        auto candidate = std::make_unique<State>(source);
        const auto memory = candidate->budget;
        const auto before = memory->used;
        memory->peak = before;
        if (cap)
            memory->limit = before + *cap;
        if (reject)
            EXPECT_THROW((void) candidate->membership.add(inserted), std::invalid_argument);
        else
        {
            EXPECT_EQ(candidate->membership.add(inserted), inserted_slot);
            const auto* reverse = candidate->membership.support_cells(support_slot);
            EXPECT_TRUE(reverse && reverse->size() == 3 && reverse->spill());
            EXPECT_EQ(candidate->membership.member_count(inserted_slot), 3u);
        }
        const auto peak = memory->peak - before;
        candidate.reset();
        EXPECT_EQ(memory->used, 0u);
        EXPECT_EQ(source.budget->used, retained);
        EXPECT_EQ(payload(source.membership), accepted);
        EXPECT_EQ(source.membership.explicit_cell_count(), 2u);
        const auto* reverse = source.membership.support_cells(support_slot);
        EXPECT_TRUE(reverse && reverse->size() == 2 && !reverse->spill());
        if (reverse && reverse->size() == 2)
        {
            EXPECT_EQ((*reverse)[0], first);
            EXPECT_EQ((*reverse)[1], second);
        }
        return peak;
    };
    const auto exact = run({}, false);
    ASSERT_GT(exact, 1u);
    EXPECT_EQ(run(exact, false), exact);
    (void) run(exact - 1, true);
}
