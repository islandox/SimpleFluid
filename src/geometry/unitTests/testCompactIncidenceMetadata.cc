/** @file testCompactIncidenceMetadata.cc
 * @brief Literal membership, spill-lifetime and compact physical-tile oracles.
 */
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIdentityBudget.hh"
#include "geometry/mesh/AMR/IncidencePools.hh"
#include "geometry/mesh/AMR/OwnedSupportMembership.hh"
#include "geometry/mesh/AMR/PooledSupportRegistry.hh"
#include "geometry/mesh/AMR/PooledTileCodec.hh"
#include "geometry/mesh/AMR/SupportSurfaceCodec.hh"
#include "geometry/mesh/AMR/TilePointContributions.hh"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace
{
namespace AMR = SimpleFluid::Meshes::AMR;
namespace D = AMR::adaptive_detail;
namespace P = SimpleFluid::PLIC;
using Access = P::details::GeometryIdentityAccess;
using Wide = std::array<long double, 3>;
constexpr auto ordinary = SimpleFluid::ArithmeticMode::Ordinary;
constexpr std::size_t ample = 32 * 1024 * 1024;
const AMR::AdaptiveBytes support{'s'}, domain{'d'};
using MemberValue = std::tuple<AMR::AdaptiveCellKey, std::uint64_t, std::uint64_t, std::uint64_t, int>;

AMR::AdaptiveCellKey key(std::uint64_t region, std::uint64_t root, std::initializer_list<std::uint64_t> path = {})
{
    return {region, AMR::ConvexCellAncestry{root, path}};
}
struct Arena
{
    D::Budget budget{ample};
    D::IdentityBudget identities;
    D::IncidencePools pools;
    Arena() : pools(budget, identities) {}
    explicit Arena(const Arena& source) : pools(source.pools, budget, identities) {}
};
struct RegistryState : Arena
{
    D::PooledSupportRegistry registry;
    RegistryState() : registry(budget, pools.supports, pools.cells) {}
    explicit RegistryState(const RegistryState& source)
        : Arena(source), registry(source.registry, budget, pools.supports, pools.cells)
    {
    }
};
struct OwnedState : Arena
{
    D::OwnedSupportMembership owned;
    OwnedState() : owned(budget, pools) {}
    explicit OwnedState(const OwnedState& source) : Arena(source), owned(source.owned, budget, pools) {}
};
struct FixtureCredit
{
    D::Budget& budget;
    std::size_t bytes;
    FixtureCredit(D::Budget& value, std::size_t amount = 512 * 1024) : budget(value), bytes(amount)
    {
        budget.add(bytes);
    }
    ~FixtureCredit() { budget.used -= bytes; }
};
std::array<D::RegistryMember, 5> members()
{
    // Arrival order, owner order, full ancestry order, and surface ordinals differ.
    return {{{key(0, 8, {2}), 91, 4, 401, 2}, {key(1, 1, {0}), 72, 1, 402, 0}, {key(0, 9, {3}), 63, 3, 403, 0},
        {key(0, 9, {1}), 54, 9, 404, 0}, {key(0, 9, {1, 0}), 45, 2, 405, 0}}};
}
MemberValue value(const D::RegistryMember& member)
{
    return {member.cell, member.revision, member.surface, member.encoded_size, member.owner};
}
std::vector<MemberValue> expected(std::initializer_list<D::RegistryMember> list)
{
    std::vector<MemberValue> result;
    for (const auto& member : list)
        result.push_back(value(member));
    return result;
}
void change(RegistryState& state, D::RegistryMember member, bool remove = false)
{
    const D::RegistryChange input{support, std::move(member), remove};
    state.registry.apply(std::span(&input, 1));
}
const D::PooledSupportRegistry::Members* group(const RegistryState& state)
{
    const auto slot = state.pools.supports.find(support);
    return slot ? state.registry.find(*slot) : nullptr;
}
std::vector<MemberValue> observe(const RegistryState& state)
{
    std::vector<MemberValue> result;
    if (const auto* entries = group(state))
        for (const auto& entry : *entries)
        {
            const auto materialized = state.registry.materialize(entry);
            EXPECT_EQ(materialized.cell, state.registry.member_cell(entry));
            result.push_back(value(materialized));
        }
    return result;
}
std::uint64_t generation(const RegistryState& state)
{
    return state.registry.generation(*state.pools.supports.find(support));
}

TEST(CompactIncidenceMetadata, SupportMembershipKeepsLiteralOrderAndGenerationAcrossInlineAndSpill)
{
    RegistryState state;
    const auto m = members();
    const std::array<std::vector<MemberValue>, 5> order{expected({m[0]}), expected({m[1], m[0]}),
        expected({m[2], m[1], m[0]}), expected({m[3], m[2], m[1], m[0]}), expected({m[3], m[4], m[2], m[1], m[0]})};
    for (std::size_t i = 0; i < m.size(); ++i)
    {
        change(state, m[i]);
        ASSERT_NE(group(state), nullptr);
        EXPECT_EQ(group(state)->size(), i + 1);
        EXPECT_EQ(bool(group(state)->spill()), i >= 2);
        EXPECT_EQ(observe(state), order[i]);
        EXPECT_EQ(generation(state), i + 1);
        EXPECT_EQ(state.registry.dirty_count(), 1u);
    }
    change(state, m[1], true);
    change(state, m[0], true);
    EXPECT_EQ(observe(state), expected({m[3], m[4], m[2]}));
    EXPECT_TRUE(group(state)->spill());
    change(state, m[4], true);
    EXPECT_EQ(observe(state), expected({m[3], m[2]}));
    EXPECT_FALSE(group(state)->spill());
    change(state, m[2], true);
    EXPECT_EQ(observe(state), expected({m[3]}));
    EXPECT_FALSE(group(state)->spill());
    change(state, m[3], true);
    EXPECT_EQ(group(state), nullptr);
    EXPECT_EQ(generation(state), 10u);
    state.registry.clear_dirty();
    EXPECT_EQ(state.registry.dirty_count(), 0u);
    change(state, m[3]);
    EXPECT_EQ(generation(state), 11u);
    EXPECT_EQ(observe(state), expected({m[3]}));
}

TEST(CompactIncidenceMetadata, SupportForkRetiresSharedSpillOnlyAfterSourceDestructionAndReap)
{
    const auto m = members();
    auto source = std::make_unique<RegistryState>();
    for (unsigned i = 0; i < 3; ++i)
        change(*source, m[i]);
    source->registry.clear_dirty();
    std::weak_ptr<const D::PooledSupportRegistry::Members::Spill> old = group(*source)->spill();
    RegistryState candidate(*source);
    EXPECT_EQ(group(candidate)->spill().get(), group(*source)->spill().get());
    change(candidate, m[0], true);
    EXPECT_EQ(observe(candidate), expected({m[2], m[1]}));
    EXPECT_FALSE(group(candidate)->spill());
    EXPECT_EQ(observe(*source), expected({m[2], m[1], m[0]}));
    EXPECT_EQ(generation(*source), 3u);
    EXPECT_EQ(generation(candidate), 4u);
    const auto pinned = candidate.budget.used;
    candidate.registry.reap_deferred();
    EXPECT_EQ(candidate.budget.used, pinned);
    EXPECT_FALSE(old.expired());
    source.reset(); // Also destroys the source pools and source Budget.
    EXPECT_TRUE(old.expired());
    const auto cumulative = candidate.budget.cumulative;
    candidate.registry.reap_deferred();
    EXPECT_LT(candidate.budget.used, pinned);
    EXPECT_EQ(candidate.budget.cumulative, cumulative);
    const auto reaped = candidate.budget.used;
    candidate.registry.reap_deferred();
    EXPECT_EQ(candidate.budget.used, reaped);
    EXPECT_EQ(observe(candidate), expected({m[2], m[1]}));
}

TEST(CompactIncidenceMetadata, SupportSpillSurvivesSourceAndFailedCandidatesDoNotAlterAcceptedMembership)
{
    const auto m = members();
    auto source = std::make_unique<RegistryState>();
    for (const auto& member : m)
        change(*source, member);
    source->registry.clear_dirty();
    const auto accepted = expected({m[3], m[4], m[2], m[1], m[0]});
    for (unsigned defect = 0; defect < 3; ++defect)
    {
        RegistryState failed(*source);
        auto invalid = m[2];
        if (defect == 1)
            ++invalid.revision;
        if (defect == 2)
            ++invalid.encoded_size;
        EXPECT_THROW(change(failed, invalid, defect != 0), std::invalid_argument);
        EXPECT_EQ(observe(*source), accepted);
        EXPECT_EQ(generation(*source), 5u);
        RegistryState retry(*source);
        change(retry, m[2], true);
        EXPECT_EQ(observe(retry), expected({m[3], m[4], m[1], m[0]}));
    }
    RegistryState survivor(*source);
    std::weak_ptr<const D::PooledSupportRegistry::Members::Spill> shared = group(*source)->spill();
    source.reset();
    EXPECT_FALSE(shared.expired());
    EXPECT_EQ(observe(survivor), accepted);
    change(survivor, m[0], true);
    EXPECT_TRUE(shared.expired());
    change(survivor, m[1], true);
    change(survivor, m[4], true);
    EXPECT_EQ(observe(survivor), expected({m[3], m[2]}));
    EXPECT_FALSE(group(survivor)->spill());
}

AMR::AdaptiveLeaf leaf(AMR::AdaptiveCellKey cell, bool six_surfaces = false)
{
    AMR::AdaptiveLeaf result;
    result.key = std::move(cell);
    result.domain = domain;
    result.revision = 77;
    result.geometry = P::ConvexCellGeometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    const std::array<std::size_t, 6> faces{5, 1, 4, 0, 3, 2};
    const std::array<AMR::AdaptiveBytes, 6> keys{AMR::AdaptiveBytes{'z'}, support, AMR::AdaptiveBytes{'k'},
        AMR::AdaptiveBytes{'b'}, AMR::AdaptiveBytes{'u'}, AMR::AdaptiveBytes{'a'}};
    for (std::size_t i = 0; i < (six_surfaces ? faces.size() : 1); ++i)
    {
        AMR::AdaptiveSurface surface;
        surface.key = six_surfaces ? keys[i] : support;
        surface.geometry = result.geometry->face_geometry(faces[i]);
        surface.orientation = 1;
        surface.exterior = true;
        surface.boundary = "wall";
        surface.native_face = 900 + faces[i];
        result.surfaces.push_back(std::move(surface));
    }
    return result;
}
std::vector<AMR::AdaptiveCellKey> reverse_keys(const OwnedState& state)
{
    std::vector<AMR::AdaptiveCellKey> result;
    const auto slot = state.pools.supports.find(support);
    if (const auto* cells = slot ? state.owned.support_cells(*slot) : nullptr)
        for (const auto cell : *cells)
            result.push_back(state.pools.cells.get(cell));
    return result;
}

TEST(CompactIncidenceMetadata, OwnedSurfaceOrdinalsRemainOriginalVectorPositionsAndWireSizesStayStable)
{
    OwnedState state;
    FixtureCredit fixture(state.budget);
    const auto input = leaf(key(3, 12, {8, 1}), true);
    std::vector<AMR::AdaptiveBytes> packets;
    for (std::size_t i = 0; i < 6; ++i)
        packets.push_back(D::encode_support_surface(input, i));
    const auto slot = state.owned.add(input);
    ASSERT_EQ(state.owned.at(slot).members.size(), 6u);
    for (std::size_t i = 0; i < 6; ++i)
        for (const bool remove : {false, true})
        {
            const auto result = state.owned.change(slot, i, 4, remove);
            EXPECT_EQ(result.support, input.surfaces[i].key);
            EXPECT_EQ(result.member.cell, input.key);
            EXPECT_EQ(result.member.revision, 77u);
            EXPECT_EQ(result.member.surface, i);
            EXPECT_EQ(result.member.owner, 4);
            EXPECT_EQ(result.member.encoded_size, packets[i].size());
            EXPECT_EQ(result.remove, remove);
            EXPECT_EQ(D::encode_support_surface(input, i), packets[i]);
        }
    EXPECT_EQ(state.owned.require(input.key, 77), slot);
    EXPECT_THROW((void) state.owned.require(input.key, 78), std::invalid_argument);
    EXPECT_THROW((void) state.owned.add(input), std::invalid_argument);
    EXPECT_THROW((void) state.owned.change(slot, 6, 4, false), std::invalid_argument);
}

TEST(CompactIncidenceMetadata, OwnedReverseInlineAndSpillForksPreserveFullKeysAndDeferredLifetime)
{
    const auto high = key(1, 1, {0}), middle = key(0, 9, {3}), low = key(0, 9, {1});
    const std::array inputs{leaf(high), leaf(middle), leaf(low)};
    auto source = std::make_unique<OwnedState>();
    std::array<std::size_t, 3> slots{};
    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        slots[i] = source->owned.add(inputs[i]);
        const auto* cells = source->owned.support_cells(*source->pools.supports.find(support));
        ASSERT_NE(cells, nullptr);
        EXPECT_EQ(cells->size(), i + 1);
        EXPECT_EQ(bool(cells->spill()), i == 2);
    }
    EXPECT_EQ(reverse_keys(*source), (std::vector{low, middle, high}));
    const auto support_slot = *source->pools.supports.find(support);
    std::weak_ptr<const D::OwnedSupportMembership::SupportCells::Spill> old =
        source->owned.support_cells(support_slot)->spill();
    OwnedState candidate(*source);
    candidate.owned.erase(slots[0]);
    EXPECT_EQ(reverse_keys(candidate), (std::vector{low, middle}));
    EXPECT_FALSE(candidate.owned.support_cells(support_slot)->spill());
    EXPECT_EQ(reverse_keys(*source), (std::vector{low, middle, high}));
    const auto pinned = candidate.budget.used;
    candidate.owned.reap_deferred();
    EXPECT_EQ(candidate.budget.used, pinned);
    source.reset();
    EXPECT_TRUE(old.expired());
    const auto cumulative = candidate.budget.cumulative;
    candidate.owned.reap_deferred();
    EXPECT_LT(candidate.budget.used, pinned);
    EXPECT_EQ(candidate.budget.cumulative, cumulative);
    EXPECT_EQ(reverse_keys(candidate), (std::vector{low, middle}));
    EXPECT_EQ(candidate.owned.change(slots[1], 0, 2, true).member.surface, 0u);
    candidate.owned.erase(slots[1]);
    candidate.owned.erase(slots[2]);
    EXPECT_EQ(candidate.owned.support_cells(support_slot), nullptr);
}

TEST(CompactIncidenceMetadata, OwnedReverseSpillRemainsUsableAfterSourceArenaDestruction)
{
    const auto high = key(1, 1, {0}), middle = key(0, 9, {3}), low = key(0, 9, {1});
    auto source = std::make_unique<OwnedState>();
    const auto high_slot = source->owned.add(leaf(high));
    (void) source->owned.add(leaf(middle));
    (void) source->owned.add(leaf(low));
    const auto support_slot = *source->pools.supports.find(support);
    std::weak_ptr<const D::OwnedSupportMembership::SupportCells::Spill> shared =
        source->owned.support_cells(support_slot)->spill();
    OwnedState survivor(*source);
    source.reset();
    EXPECT_FALSE(shared.expired());
    EXPECT_EQ(reverse_keys(survivor), (std::vector{low, middle, high}));
    EXPECT_EQ(survivor.owned.change(high_slot, 0, 2, false).member.cell, high);
    survivor.owned.erase(high_slot);
    EXPECT_TRUE(shared.expired());
    EXPECT_EQ(reverse_keys(survivor), (std::vector{low, middle}));
    EXPECT_FALSE(survivor.owned.support_cells(support_slot)->spill());
}

D::SupportFace face()
{
    const std::array<P::GeometryPoint, 4> vertices{{{0, 0, 0}, {2, 0, 0}, {2, 1, 0}, {0, 1, 0}}};
    return P::PlanarFace<ordinary>::make(vertices);
}
D::ReceivedTileRecipe authenticated_recipe(
    const D::SupportFace& geometry, int first_owner, int second_owner, int resident, bool second = true)
{
    const std::array<D::RegistryMember, 2> members{
        {{key(0, 1), 7, 3, 301, first_owner}, {key(0, 2), 9, 5, 405, second_owner}}};
    const auto image = second ? P::displacement<ordinary>({0, 0, 0}, {0, 0, 4}) : P::DisplacementIdentity<ordinary>{};
    const D::PrivateTileRecipe recipe{0, second ? std::optional<std::size_t>(1) : std::nullopt, geometry, image};
    const auto packet = D::encode_tile_recipe(support, domain, members, recipe, 19);
    return D::decode_tile_recipe(packet, D::authority(support, 3), resident, 3, ample);
}
using Contribution = std::pair<AMR::AdaptiveCellKey, Wide>;
std::vector<Contribution> contributions(const D::PooledTile& tile, int rank, Arena& arena)
{
    std::vector<Contribution> result;
    D::visit_tile_physical_points(tile, tile.intrinsic_loop, rank, arena.pools, [&](std::size_t cell, std::size_t point)
        { result.emplace_back(arena.pools.cells.get(cell), Access::point(arena.pools.points.get(point))); });
    return result;
}
std::vector<Contribution> expected_contributions(
    bool first, bool second, bool same_cell = false, long double shift = -4)
{
    const std::array<Wide, 4> vertices{{{0, 0, 0}, {2, 0, 0}, {2, 1, 0}, {0, 1, 0}}};
    std::vector<Contribution> result;
    for (auto vertex : vertices)
    {
        if (first)
            result.emplace_back(key(0, 1), vertex);
        if (second)
        {
            vertex[2] = shift;
            result.emplace_back(key(0, same_cell ? 1 : 2), vertex);
        }
    }
    return result;
}

TEST(CompactIncidenceMetadata, FrozenTilesKeepOnlyResidentCellsAndRetainCompletePhysicalFrame)
{
    for (const auto [first_owner, second_owner, resident, second] :
        {std::tuple{0, 1, 0, true}, std::tuple{0, 1, 1, true}, std::tuple{0, 0, 0, true}, std::tuple{0, 1, 0, false}})
    {
        Arena arena;
        FixtureCredit fixture(arena.budget); // Owns fixed small recipe/loop/restore test buffers.
        const auto original = face();
        const auto original_points = original->points();
        const auto original_key = P::canonical_face_key(*original);
        const auto original_bytes = original->encode_incidence();
        const auto recipe = authenticated_recipe(original, first_owner, second_owner, resident, second);
        const auto* recipe_owner = recipe.face.get();
        const auto recipe_owners = recipe.face.use_count();
        const auto tile = D::freeze_tile(recipe, resident, arena.pools);
        EXPECT_EQ(recipe.face.get(), recipe_owner);
        EXPECT_EQ(recipe.face.use_count(), recipe_owners);
        EXPECT_EQ(tile.emission, 19u);
        EXPECT_EQ(arena.pools.supports.get(tile.support), support);
        EXPECT_EQ(arena.pools.domains.get(tile.domain), domain);
        const bool local_first = first_owner == resident, local_second = second && second_owner == resident;
        EXPECT_EQ(tile.first_cell != D::PooledTile::absent_cell, local_first);
        EXPECT_EQ(tile.second_cell != D::PooledTile::absent_cell, local_second);
        EXPECT_EQ(arena.pools.cells.find(key(0, 1)).has_value(), local_first);
        EXPECT_EQ(arena.pools.cells.find(key(0, 2)).has_value(), local_second);
        EXPECT_EQ(Access::displacement(arena.pools.images.get(tile.image_slot)), (Wide{0, 0, second ? 4.L : 0.L}));
        EXPECT_EQ(arena.pools.normals.get(tile.normal_slot).values, (Wide{0, 0, 2}));
        EXPECT_EQ(contributions(tile, resident, arena), expected_contributions(local_first, local_second));
        const auto restored = D::restore_tile(tile, tile.intrinsic_loop, arena.pools);
        EXPECT_EQ(restored->points(), original_points);
        EXPECT_EQ(P::canonical_face_key(*restored), original_key);
        EXPECT_EQ(restored->encode_incidence(), original_bytes);
        EXPECT_EQ(original->points(), original_points);
        EXPECT_EQ(P::canonical_face_key(*original), original_key);
        EXPECT_EQ(original->encode_incidence(), original_bytes);
        EXPECT_EQ(recipe.first.revision, 7u);
        EXPECT_EQ(recipe.first.surface, 3u);
        if (second)
        {
            ASSERT_TRUE(recipe.second);
            EXPECT_EQ(recipe.second->revision, 9u);
            EXPECT_EQ(recipe.second->surface, 5u);
        }
    }
}

TEST(CompactIncidenceMetadata, RepeatedLocalCellRolesStillEmitTwoIndependentPhysicalContributions)
{
    Arena arena;
    FixtureCredit fixture(arena.budget);
    const auto recipe = authenticated_recipe(face(), 0, 0, 0);
    auto tile = D::freeze_tile(recipe, 0, arena.pools);
    // This is a direct role-accounting fixture, not an extension of the wire
    // codec's distinct-cell admission rule.
    tile.second_cell = tile.first_cell;
    EXPECT_EQ(contributions(tile, 0, arena), expected_contributions(true, true, true));
    tile.image_slot = arena.pools.images.intern(P::DisplacementIdentity<ordinary>{});
    EXPECT_EQ(contributions(tile, 0, arena), expected_contributions(true, true, true, 0));
}

TEST(CompactIncidenceMetadata, NormalImageAndTileSlotsRetainPrecisionAfterPoolForkAndSourceDestruction)
{
    std::unique_ptr<Arena> candidate;
    std::unique_ptr<FixtureCredit> candidate_fixture;
    const auto precise = std::nextafter(1.L, 2.L);
    auto source = std::make_unique<Arena>();
    auto source_fixture = std::make_unique<FixtureCredit>(source->budget);
    const std::array<Wide, 4> vertices{{{0, 0, 0}, {0, 0, 1}, {precise, 0, 1}, {precise, 0, 0}}};
    const auto original = P::PlanarFace<ordinary>::make_wide(vertices);
    const auto original_bytes = original->encode_incidence();
    const auto recipe = authenticated_recipe(original, 0, 1, 0);
    const auto tile = D::freeze_tile(recipe, 0, source->pools);
    const auto minus_zero = source->pools.normals.intern(D::IncidencePools::Normal({-0.L, precise, 0.L}));
    const auto plus_zero = source->pools.normals.intern(D::IncidencePools::Normal({0.L, precise, 0.L}));
    EXPECT_NE(minus_zero, plus_zero);
    const auto image = Access::displacement(Wide{precise, -0.L, 0.L});
    const auto image_slot = source->pools.images.intern(image);
    EXPECT_EQ(source->pools.images.intern(Access::displacement(Wide{precise, 0.L, -0.L})), image_slot);
    const auto* normal_address = &source->pools.normals.get(minus_zero);
    const auto* image_address = &source->pools.images.get(image_slot);
    candidate = std::make_unique<Arena>(*source);
    candidate_fixture = std::make_unique<FixtureCredit>(candidate->budget);
    // Transfer the fixed caller-owned fixture reservation before destroying
    // its original arena; pooled owners carry their own candidate admissions.
    source_fixture.reset();
    source.reset();
    EXPECT_EQ(&candidate->pools.normals.get(minus_zero), normal_address);
    EXPECT_EQ(&candidate->pools.images.get(image_slot), image_address);
    EXPECT_TRUE(std::signbit(candidate->pools.normals.get(minus_zero).values[0]));
    EXPECT_FALSE(std::signbit(candidate->pools.normals.get(plus_zero).values[0]));
    EXPECT_EQ(candidate->pools.normals.get(minus_zero).values[1], precise);
    EXPECT_EQ(Access::displacement(candidate->pools.images.get(image_slot)), (Wide{precise, 0, 0}));
    EXPECT_EQ(P::encode_identity(candidate->pools.images.get(image_slot)), P::encode_identity(image));
    const auto restored = D::restore_tile(tile, tile.intrinsic_loop, candidate->pools);
    EXPECT_EQ(restored->points(), (std::vector<Wide>(vertices.begin(), vertices.end())));
    EXPECT_EQ(restored->encode_incidence(), original_bytes);
    EXPECT_EQ(candidate->pools.normals.get(tile.normal_slot).values, (Wide{0, precise, 0}));
}
} // namespace
