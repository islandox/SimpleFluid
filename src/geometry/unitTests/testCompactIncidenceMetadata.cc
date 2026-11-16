/** @file testCompactIncidenceMetadata.cc
 * @brief Literal membership, spill-lifetime and compact physical-tile oracles.
 */
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/mesh/AMR/CartesianSourceLineRanges.hh"
#include "geometry/mesh/AMR/ContactPayloadCensus.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIdentityBudget.hh"
#include "geometry/mesh/AMR/IncidencePools.hh"
#include "geometry/mesh/AMR/OwnedSupportMembership.hh"
#include "geometry/mesh/AMR/PooledSupportRegistry.hh"
#include "geometry/mesh/AMR/PooledTileCodec.hh"
#include "geometry/mesh/AMR/SupportSurfaceCodec.hh"
#include "geometry/mesh/AMR/TilePointContributions.hh"
#include <algorithm>
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

// Independent literal of the Cartesian support wire grammar, including its framed binary domain.
AMR::AdaptiveBytes support_packet(std::array<std::uint64_t, 5> fields,
    const AMR::AdaptiveBytes& name = {'a', '\0', static_cast<char>(0x80)}, std::uint64_t region = 0x0102030405060708ULL)
{
    AMR::AdaptiveBytes result{'1', '6', 'T', 'R', 'A', 'C', 'F', 'S'};
    const auto word = [&](std::uint64_t value)
    {
        for (unsigned shift = 0; shift < 64; shift += 8)
            result.push_back(static_cast<char>((value >> shift) & 255));
    };
    word(name.size());
    result.insert(result.end(), name.begin(), name.end());
    word(region);
    for (const auto value : fields)
        word(value);
    return result;
}

TEST(CompactIncidenceMetadata, CompactSupportDirectoryMatchesFullFramingMembershipAndOrder)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{ample});
    D::IdentityBudget identities;
    D::IncidencePools pools(*arena, identities, arena, D::IdentityPoolStorage::Persistent);
    auto& directory = pools.supports;
    const auto first = support_packet({2, 3, 128, 255, 127});
    const auto never_inserted = support_packet({2, 3, 129, 255, 127});
    auto wrong_magic = first;
    wrong_magic.front() = '0';
    auto wrong_length = first;
    wrong_length[8] = 2;
    auto trailing = first;
    trailing.push_back('!');
    const std::vector<std::pair<AMR::AdaptiveBytes, bool>> cases{{{'x'}, false}, {first, true},
        {support_packet({2, 3, 127, 255, 127}), true}, {support_packet({2, 3, 255, 255, 127}), true},
        {support_packet({2, 3, 256, 255, 127}), false}, {support_packet({2, 3, 128, 256, 127}), false},
        {support_packet({2, 3, 128, 255, 256}), false}, {support_packet({0, 0, 0, 0, 0}), true},
        {support_packet({2, 31, 255, 255, 255}), true}, {support_packet({3, 3, 128, 255, 127}), false},
        {support_packet({2, 32, 128, 255, 127}), false},
        {support_packet({2, 3, 128, 255, 127}, {'a', '\0', static_cast<char>(0x81)}), false},
        {support_packet({2, 3, 128, 255, 127}, {'a', '\0', static_cast<char>(0x80)}, 17), false}, {wrong_magic, false},
        {wrong_length, false}, {trailing, false}};
    std::vector<std::size_t> slots;
    for (const auto& [wire, compact] : cases)
    {
        ASSERT_FALSE(directory.find(wire));
        const auto slot = slots.size() % 2 == 0 ? directory.intern(wire) : directory.intern_bytes(wire);
        EXPECT_EQ(directory.is_compact(slot), compact && std::numeric_limits<std::size_t>::digits > 32);
        EXPECT_EQ(directory.find(wire), slot);
        EXPECT_EQ(directory.find_bytes(std::span<const char>(wire)), slot);
        EXPECT_EQ(directory.intern(AMR::AdaptiveBytes(wire)), slot);
        const auto replay = directory.get(slot);
        EXPECT_TRUE(std::ranges::equal(replay, wire));
        EXPECT_EQ(std::find(slots.begin(), slots.end(), slot), slots.end());
        slots.push_back(slot);
    }
    const auto before_used = arena->used, before_cumulative = arena->cumulative;
    EXPECT_EQ(directory.size(), cases.size());
    EXPECT_EQ(directory.intern_bytes(cases.front().first), slots.front());
    EXPECT_EQ(directory.intern_bytes(first), slots[1]);
    EXPECT_FALSE(directory.find(never_inserted));
    // Borrow a real range-protocol view and replace only its plane word. This
    // distinguishes a decodable coordinate from an actually interned support.
    constexpr std::size_t plane_offset = 2 * 8 + 3 + 3 * 8;
    EXPECT_EQ(directory.find_borrowed(D::CartesianSupportRangeKeyView{first, plane_offset, 127}), slots[2]);
    EXPECT_FALSE(directory.find_borrowed(D::CartesianSupportRangeKeyView{first, plane_offset, 129}));
    for (std::size_t i = 0; i < slots.size(); ++i)
        for (std::size_t j = 0; j < slots.size(); ++j)
        {
            const auto a = directory.get(slots[i]), b = directory.get(slots[j]);
            EXPECT_EQ(a == b, cases[i].first == cases[j].first);
            EXPECT_EQ(a < b, cases[i].first < cases[j].first);
        }
    EXPECT_EQ(arena->used, before_used);
    EXPECT_EQ(arena->cumulative, before_cumulative);
}

TEST(CompactIncidenceMetadata, CompactSupportDirectoryForkPreservesSlotsAndRejectsOneByteShort)
{
    auto source_arena = std::make_shared<D::Budget>(D::Budget{ample});
    D::IdentityBudget original_ids;
    auto source = std::make_unique<D::IncidencePools>(
        *source_arena, original_ids, source_arena, D::IdentityPoolStorage::Persistent);
    const auto old = support_packet({0, 1, 3, 4, 5});
    const auto changed = support_packet({2, 31, 255, 250, 240});
    const AMR::AdaptiveBytes literal{'f', 'o', 'r', 'k'};
    const auto old_slot = source->supports.intern(old);
    const auto literal_slot = source->supports.intern(literal);
    if constexpr (std::numeric_limits<std::size_t>::digits <= 32)
    {
        EXPECT_FALSE(source->supports.is_compact(old_slot));
        EXPECT_EQ(source->supports.find(old), old_slot);
        return;
    }
    ASSERT_TRUE(source->supports.is_compact(old_slot));
    const auto escaped = source->supports.get(old_slot);

    const auto edit_candidate = [&](std::size_t cap, bool reject)
    {
        auto candidate_arena = std::make_shared<D::Budget>(D::Budget{cap});
        D::IdentityBudget candidate_ids;
        {
            D::IncidencePools candidate(*source, *candidate_arena, candidate_ids, candidate_arena);
            EXPECT_EQ(candidate.supports.find(literal), literal_slot);
            EXPECT_EQ(candidate.supports.find(old), old_slot);
            EXPECT_FALSE(candidate.supports.find(changed));
            const auto used = candidate_arena->used, spent = candidate_arena->cumulative;
            if (reject)
            {
                EXPECT_THROW((void) candidate.supports.intern(changed), std::invalid_argument);
                EXPECT_FALSE(candidate.supports.find(changed));
                EXPECT_EQ(candidate.supports.find(old), old_slot);
                EXPECT_EQ(candidate_arena->used, used);
                EXPECT_GE(candidate_arena->cumulative, spent);
            }
            else
            {
                const auto changed_slot = candidate.supports.intern(changed);
                EXPECT_TRUE(candidate.supports.is_compact(changed_slot));
                EXPECT_EQ(candidate.supports.find(changed), changed_slot);
                EXPECT_TRUE(std::ranges::equal(candidate.supports.get(changed_slot), changed));
                EXPECT_FALSE(source->supports.find(changed));
                const auto overlap = candidate.supports.overlap_bytes();
                const auto live = candidate_arena->used, cumulative = candidate_arena->cumulative;
                EXPECT_GT(overlap, 0u);
                candidate.finish_overlap();
                EXPECT_EQ(candidate_arena->used + overlap, live);
                EXPECT_EQ(candidate_arena->cumulative, cumulative);
                EXPECT_EQ(candidate.supports.find(old), old_slot);
            }
        }
        EXPECT_EQ(candidate_arena->used, 0u);
        return candidate_arena->peak;
    };
    const auto peak = edit_candidate(ample, false);
    ASSERT_GT(peak, 0u);
    (void) edit_candidate(peak - 1, true);
    EXPECT_EQ(source->supports.find(old), old_slot);
    EXPECT_EQ(source->supports.find(literal), literal_slot);
    EXPECT_FALSE(source->supports.find(changed));
    EXPECT_EQ(edit_candidate(peak, false), peak);
    auto survivor_arena = std::make_shared<D::Budget>(D::Budget{ample});
    D::IdentityBudget survivor_ids;
    {
        D::IncidencePools survivor(*source, *survivor_arena, survivor_ids, survivor_arena);
        source.reset();
        survivor.finish_overlap();
        EXPECT_EQ(survivor.supports.find(old), old_slot);
        EXPECT_EQ(survivor.supports.find(literal), literal_slot);
        EXPECT_TRUE(std::ranges::equal(survivor.supports.get(literal_slot), literal));
        EXPECT_FALSE(survivor.supports.find(changed));
    }
    EXPECT_EQ(survivor_arena->used, 0u);
    EXPECT_EQ(source_arena->used, 0u);
    source_arena.reset();
    EXPECT_TRUE(std::ranges::equal(escaped, old));

    auto fresh_arena = std::make_shared<D::Budget>(D::Budget{ample});
    D::IdentityBudget fresh_ids;
    D::IncidencePools fresh(*fresh_arena, fresh_ids, fresh_arena, D::IdentityPoolStorage::Persistent);
    EXPECT_FALSE(fresh.supports.find_bytes(escaped.bytes()));
    const auto fresh_slot = fresh.supports.intern(AMR::AdaptiveBytes(escaped.begin(), escaped.end()));
    EXPECT_EQ(fresh.supports.find_bytes(escaped.bytes()), fresh_slot);
    EXPECT_TRUE(std::ranges::equal(fresh.supports.get(fresh_slot), old));
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

TEST(CompactIncidenceMetadata, PersistentSupportGroupsShareCleanSpillsAcrossSparseEdits)
{
    const auto measure = [](std::size_t count)
    {
        auto source_budget = std::make_shared<D::Budget>(D::Budget{ample});
        auto next_budget = std::make_shared<D::Budget>(D::Budget{ample});
        D::IdentityBudget original_ids, next_ids;
        auto original_pools = std::make_unique<D::IncidencePools>(
            *source_budget, original_ids, source_budget, D::IdentityPoolStorage::Persistent);
        auto original = std::make_unique<D::PooledSupportRegistry>(
            *source_budget, original_pools->supports, original_pools->cells, source_budget);
        const auto support_key = [](std::size_t i) { return AMR::AdaptiveBytes{char(i >> 8), char(i & 0xff)}; };
        const auto make_member = [](std::size_t i, std::size_t side)
        { return D::RegistryMember{key(i, side), 9, 6 + side, 210, int(side)}; };
        for (std::size_t i = 0; i < count; ++i)
        {
            const std::array<D::RegistryChange, 3> changes{{{support_key(i), make_member(i, 0), false},
                {support_key(i), make_member(i, 1), false}, {support_key(i), make_member(i, 2), false}}};
            original->apply(changes);
        }
        original->clear_dirty();
        D::IncidencePools candidate_pools(*original_pools, *next_budget, next_ids, next_budget);
        const auto before_fork = next_budget->used;
        D::PooledSupportRegistry candidate(
            *original, *next_budget, candidate_pools.supports, candidate_pools.cells, next_budget);
        EXPECT_EQ(next_budget->used - before_fork, original->retained_bytes());
        const auto pages = [](const auto& registry)
        {
            std::vector<const void*> result;
            registry.visit_borrowed_payload(
                [&](const D::ContactPayloadAllocation& item)
                {
                    if (item.identity.role == D::ParallelSupportPayloadRole::SupportRegistryGroupNodes &&
                        item.kind == D::ContactPayloadKind::Object)
                        result.push_back(item.identity.address);
                });
            return result;
        };
        const auto shared_pages = pages(*original);
        EXPECT_GT(shared_pages.size(), 1u);
        EXPECT_EQ(pages(candidate), shared_pages);
        const auto changed = *candidate_pools.supports.find(support_key(0));
        const auto clean = *candidate_pools.supports.find(support_key(count - 1));
        const auto old_spill = std::weak_ptr(original->find(changed)->spill());
        const auto clean_owner = original->find(clean)->spill().get();
        EXPECT_EQ(candidate.find(clean)->spill().get(), clean_owner);
        EXPECT_EQ(candidate.dirty_count(), 0u);

        const D::RegistryChange remove{support_key(0), make_member(0, 2), true};
        const auto before = next_budget->cumulative;
        candidate.apply(std::span(&remove, 1));
        const auto dirty_work = next_budget->cumulative - before;
        EXPECT_EQ(candidate.dirty_count(), 1u);
        EXPECT_EQ(candidate.generation(changed), 4u);
        EXPECT_EQ(original->generation(changed), 3u);
        EXPECT_EQ(candidate.find(changed)->size(), 2u);
        EXPECT_FALSE(candidate.find(changed)->spill());
        EXPECT_FALSE(old_spill.expired());
        EXPECT_EQ(candidate.find(clean)->spill().get(), clean_owner);
        const auto exact = next_budget->peak;
        {
            auto short_budget = std::make_shared<D::Budget>(D::Budget{exact - 1});
            D::IdentityBudget short_ids;
            D::IncidencePools short_pools(*original_pools, *short_budget, short_ids, short_budget);
            D::PooledSupportRegistry rejected(
                *original, *short_budget, short_pools.supports, short_pools.cells, short_budget);
            const auto used = short_budget->used;
            EXPECT_THROW(rejected.apply(std::span(&remove, 1)), std::invalid_argument);
            EXPECT_EQ(short_budget->used, used);
            EXPECT_EQ(rejected.dirty_count(), 0u);
            EXPECT_EQ(rejected.generation(changed), 3u);
            EXPECT_EQ(rejected.find(changed)->spill().get(), original->find(changed)->spill().get());
            short_budget->limit = exact;
            EXPECT_NO_THROW(rejected.apply(std::span(&remove, 1)));
            EXPECT_EQ(rejected.generation(changed), 4u);
        }
        const auto old_overlap = candidate.overlap_bytes();
        EXPECT_GT(old_overlap, 0u);
        const auto retained = next_budget->used;
        candidate.finish_overlap();
        EXPECT_EQ(next_budget->used + old_overlap, retained);
        original.reset();
        original_pools.reset();
        EXPECT_TRUE(old_spill.expired());
        EXPECT_EQ(candidate.find(clean)->spill().get(), clean_owner);
        EXPECT_EQ(candidate.generation(clean), 3u);
        return dirty_work;
    };
    const auto small = measure(256);
    const auto large = measure(512);
    EXPECT_LT(large, 2 * small);
}

TEST(CompactIncidenceMetadata, PersistentSupportRegistryKeepsItsOwnBudgetThroughDestruction)
{
    Arena independent_pools;
    for (bool populated : {false, true})
    {
        SCOPED_TRACE(populated);
        std::optional<std::size_t> final_usage;
        auto keeper = std::shared_ptr<D::Budget>(new D::Budget{ample},
            [&](D::Budget* budget)
            {
                final_usage = budget->used;
                delete budget;
            });
        const std::weak_ptr<D::Budget> arena = keeper;
        auto registry = std::make_unique<D::PooledSupportRegistry>(
            *keeper, independent_pools.pools.supports, independent_pools.pools.cells, keeper);
        if (populated)
        {
            const D::RegistryChange add{support, members().front(), false};
            registry->apply(std::span(&add, 1));
        }
        keeper.reset();
        EXPECT_FALSE(arena.expired());
        EXPECT_FALSE(final_usage);
        registry.reset();
        EXPECT_TRUE(arena.expired());
        EXPECT_EQ(final_usage, std::optional<std::size_t>{0});
    }
}

TEST(CompactIncidenceMetadata, SupportSpillCensusReportsTheOwningObjectAndItsBuffer)
{
    Arena pools;
    for (bool persistent : {false, true})
    {
        SCOPED_TRACE(persistent);
        auto budget = std::make_shared<D::Budget>(D::Budget{ample});
        D::PooledSupportRegistry registry(
            *budget, pools.pools.supports, pools.pools.cells, persistent ? budget : nullptr);
        const auto input = members();
        for (std::size_t i = 0; i < 3; ++i)
        {
            const D::RegistryChange add{support, input[i], false};
            registry.apply(std::span(&add, 1));
        }
        const auto slot = pools.pools.supports.find(support);
        ASSERT_TRUE(slot);
        const auto* group = registry.find(*slot);
        ASSERT_NE(group, nullptr);
        const auto& spill = group->spill();
        ASSERT_TRUE(spill);

        std::optional<D::ContactPayloadAllocation> object, buffer;
        registry.visit_borrowed_payload(
            [&](const D::ContactPayloadAllocation& item)
            {
                if (item.identity.role == D::ParallelSupportPayloadRole::SupportMembershipSpill)
                    object = item;
                if (item.identity.role == D::ParallelSupportPayloadRole::SupportMembershipValues)
                    buffer = item;
            });
        ASSERT_TRUE(object);
        ASSERT_TRUE(buffer);
        EXPECT_EQ(object->owner, object->identity.address);
        EXPECT_EQ(buffer->owner, object->identity.address);
        EXPECT_EQ(buffer->identity.address, spill->data());
        EXPECT_EQ(buffer->payload_bytes, spill->capacity() * sizeof(D::PooledSupportMember));
        if (persistent)
        {
            const auto base = reinterpret_cast<std::uintptr_t>(object->identity.address);
            const auto vector = reinterpret_cast<std::uintptr_t>(spill.get());
            ASSERT_GT(vector, base);
            ASSERT_LE(vector - base, object->payload_bytes);
            EXPECT_LE(sizeof(*spill), object->payload_bytes - (vector - base));
        }
        else
        {
            EXPECT_EQ(object->identity.address, spill.get());
            EXPECT_EQ(object->payload_bytes, sizeof(*spill));
        }
    }
}
} // namespace
