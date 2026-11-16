/** Exact persistent line history witnesses and physical alias ownership. */
#include "cartesian_incidence_test_helpers.hh"
#include "geometry/mesh/AMR/AdaptiveSubfaceSnapshot.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIncidence.hh"
#include "geometry/mesh/AMR/PooledLineRegistry.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_DefaultSerialComm.hpp>
#include <array>
#include <gtest/gtest.h>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct PooledLineRegistryAdmissionTestAccess
{
    using Observer = std::size_t (*)(const Budget&, std::size_t);
    explicit PooledLineRegistryAdmissionTestAccess(Observer observer)
        : previous(PooledLineRegistry::exchange_replay_admission_observer(observer))
    {
    }
    ~PooledLineRegistryAdmissionTestAccess()
    {
        (void) PooledLineRegistry::exchange_replay_admission_observer(previous);
    }
    Observer previous;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail

namespace
{
using namespace cartesian_incidence_test;
constexpr auto ordinary = ArithmeticMode::Ordinary;
using Access = PLIC::details::GeometryIdentityAccess;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
PooledLineEdit populate(IncidencePools& pools, PooledLineRegistry& registry, std::size_t count)
{
    registry.begin_replay(true);
    const auto axis = PLIC::canonical_key(Access::line({0, {1, 0, 0}, {0, 0, 0}}));
    PooledLineEdit first;
    for (std::size_t line = 0; line < count; ++line)
    {
        Writer domain, key;
        domain.word(line);
        key.bytes(domain.data);
        key.bytes(axis);
        PooledLineSubscriber query;
        query.support = pools.supports.intern(domain.data);
        query.first = pools.points.intern(point(0));
        query.last = pools.points.intern(point(1));
        query.resident = query.support_authority = 0;
        query.emission = line;
        const PooledLineEdit added{pools.lines.intern(std::move(key.data)), std::move(query), false};
        registry.apply(std::span(&added, 1));
        if (!line)
            first = added;
    }
    registry.recompute([](auto, auto) {});
    registry.finish_replay(0);
    registry.finish_source_selection();
    return first;
}
struct Scene
{
    std::shared_ptr<Budget> budget = std::make_shared<Budget>(Budget{ample});
    IdentityBudget ids;
    IncidencePools pools{*budget, ids, budget};
    PooledLineRegistry registry{*budget, pools};
    PooledLineEdit first;
    explicit Scene(std::size_t count) : first(populate(pools, registry, count)) {}
};
using AllocationKey = std::pair<const void*, ContactPayloadRole>;
std::map<AllocationKey, std::size_t> index_pages(const PooledLineRegistry& registry)
{
    std::map<AllocationKey, std::size_t> result;
    registry.visit_borrowed_payload(
        [&](const ContactPayloadAllocation& allocation)
        {
            const auto role = allocation.identity.role;
            if (role == GeometryContactPayloadRole::LineRegistryNodes ||
                role == GeometryContactPayloadRole::LineHistoryRows ||
                role == GeometryContactPayloadRole::LineHistoryChildren)
                result.emplace(AllocationKey{allocation.identity.address, role}, allocation.payload_bytes);
        });
    return result;
}

struct ReplayReservation
{
    bool one_short;
    std::size_t used = 0, bytes = 0, limit = 0;
    static thread_local ReplayReservation* current;
    static std::size_t tighten(const Budget& budget, std::size_t bytes)
    {
        auto& reservation = *current;
        if (reservation.bytes)
            return budget.limit;
        reservation.used = budget.used;
        reservation.bytes = bytes;
        EXPECT_GT(bytes, 1u);
        EXPECT_LE(bytes, budget.limit - budget.used);
        reservation.limit = budget.used + bytes - std::size_t(reservation.one_short);
        return reservation.limit;
    }
};
thread_local ReplayReservation* ReplayReservation::current = nullptr;

auto canonical_subface_output(const AdaptiveSubfaceSnapshot& snapshot)
{
    const auto incident = [](const CanonicalSubfaceIncident& value)
    { return std::tuple{value.key(), value.owner(), value.revision(), value.source_surface(), value.native_face()}; };
    using Incident = decltype(incident(std::declval<const CanonicalSubfaceIncident&>()));
    using Piece = std::tuple<Incident, std::optional<Incident>, int, AdaptiveBytes, AdaptiveBytes, std::string,
        std::uint64_t, std::uint64_t, AdaptiveBytes, std::vector<AdaptiveBytes>>;
    std::map<AdaptiveBytes, Piece> output;
    snapshot.visit_pieces(
        [&](const CanonicalSubfaceView& view)
        {
            Budget budget{ample};
            IdentityBudget ids;
            auto piece = view.materialize(budget, ids);
            std::optional<Incident> second;
            if (const auto value = view.second())
                second = incident(*value);
            const auto support = view.support();
            const auto domain = view.domain();
            std::vector<AdaptiveBytes> loop;
            for (std::size_t i = 0; i < view.point_count(); ++i)
                loop.push_back(PLIC::canonical_key(view.point(i)));
            EXPECT_TRUE(output
                    .emplace(std::move(piece.identity),
                        Piece{incident(view.first()), std::move(second), view.outward_sign(),
                            AdaptiveBytes(support.begin(), support.end()), AdaptiveBytes(domain.begin(), domain.end()),
                            std::string(view.boundary()), view.generation(), view.emission(),
                            PLIC::canonical_key(view.second_image()), std::move(loop)})
                    .second);
        });
    EXPECT_EQ(output.size(), snapshot.size());
    return output;
}
} // namespace

TEST(PooledLineRegistryPersistence, NullWitnessAndFutureAliasAreExactWithBoundedPathCopies)
{
    std::array<std::size_t, 2> new_bytes{};
    for (std::size_t scale = 0; scale < new_bytes.size(); ++scale)
    {
        const std::size_t count = 128 << scale;
        Scene source(count);
        const auto old = index_pages(source.registry);
        auto arena = std::make_shared<Budget>(Budget{ample});
        IdentityBudget ids;
        IncidencePools pools(source.pools, *arena, ids, arena);
        PooledLineRegistry candidate(source.registry, *arena, pools);
        EXPECT_EQ(index_pages(candidate), old);
        auto remove = source.first;
        remove.remove = true;
        candidate.begin_replay(false);
        candidate.apply(std::span(&remove, 1));
        candidate.recompute([](auto, auto) {});
        EXPECT_THROW((void) candidate.query_result(remove.line, 0), std::invalid_argument);
        EXPECT_EQ(candidate.active_storage().retained_explicit_records, count);
        candidate.set_layer(1);
        EXPECT_EQ(candidate.query_result(remove.line, 0).result_size(), 2u);
        candidate.finish_replay(0);
        candidate.finish_source_selection();
        candidate.finish_overlap();
        EXPECT_EQ(candidate.active_storage().retained_explicit_records, count - 1);
        EXPECT_EQ(source.registry.query_result(remove.line, 0).result_size(), 2u);
        const auto updated = index_pages(candidate);
        std::size_t new_nodes = 0;
        for (const auto& [key, bytes] : updated)
            if (!old.contains(key))
            {
                new_bytes[scale] += bytes;
                new_nodes += key.second == GeometryContactPayloadRole::LineRegistryNodes;
            }
        EXPECT_LE(new_nodes, 3u);

        auto third_arena = std::make_shared<Budget>(Budget{ample});
        IdentityBudget third_ids;
        IncidencePools third_pools(pools, *third_arena, third_ids, third_arena);
        PooledLineRegistry third(candidate, *third_arena, third_pools);
        EXPECT_EQ(index_pages(third), updated);
    }
    // Exact leaf occupancy may vary by one row at a redistribution boundary,
    // while both variants copy a fixed maximum 64/32 metadata path.
    EXPECT_LE(new_bytes[1], new_bytes[0] + 2 * 64 * sizeof(std::shared_ptr<const void>));
}

TEST(PooledLineRegistryPersistence, DirtyReplayExactPeakAndPoisonedOwnerPrefixTearDown)
{
    Scene source(128);
    const auto accepted = index_pages(source.registry);
    const auto source_used = source.budget->used;
    auto remove = source.first;
    remove.remove = true;
    const auto replay = [&](std::size_t cap, bool fail)
    {
        auto arena = std::make_shared<Budget>(Budget{cap});
        IdentityBudget ids;
        IncidencePools pools(source.pools, *arena, ids, arena);
        const auto before_registry = arena->used;
        {
            PooledLineRegistry candidate(source.registry, *arena, pools);
            EXPECT_EQ(index_pages(candidate), accepted);
            candidate.begin_replay(false);
            const auto first_owner_event = candidate.owner_admission_work();
            const auto edit = [&]
            {
                candidate.apply(std::span(&remove, 1));
                candidate.recompute([](auto, auto) {});
                candidate.finish_replay(0);
                candidate.finish_source_selection();
                candidate.finish_overlap();
            };
            if (!fail)
            {
                edit();
                EXPECT_FALSE(candidate.poisoned());
                EXPECT_EQ(candidate.active_storage().retained_explicit_records, 127u);
            }
            else
            {
                EXPECT_THROW(edit(), std::invalid_argument);
                EXPECT_TRUE(candidate.poisoned());
                EXPECT_GT(candidate.owner_admission_work(), first_owner_event);
                EXPECT_THROW(candidate.require_publishable(), std::logic_error);
                EXPECT_THROW(candidate.begin_replay(false), std::logic_error);
                EXPECT_THROW((void) candidate.query_result(remove.line, 0), std::logic_error);
                auto third_arena = std::make_shared<Budget>(Budget{ample});
                IdentityBudget third_ids;
                IncidencePools third_pools(pools, *third_arena, third_ids, third_arena);
                EXPECT_THROW(PooledLineRegistry(candidate, *third_arena, third_pools), std::logic_error);
            }
        }
        EXPECT_EQ(arena->used, before_registry);
        return arena->peak;
    };
    const auto exact = replay(ample, false);
    ASSERT_GT(exact, 0u);
    EXPECT_EQ(replay(exact, false), exact);
    (void) replay(exact - 1, true);
    EXPECT_EQ(source.budget->used, source_used);
    EXPECT_EQ(index_pages(source.registry), accepted);
    EXPECT_EQ(source.registry.query_result(remove.line, 0).result_size(), 2u);
}

TEST(PooledLineRegistryPersistence, RecomputedBucketsReleaseTheirTransferredChargeOnce)
{
    auto arena = std::make_shared<Budget>(Budget{ample});
    std::size_t identity_bytes = 0;
    {
        IdentityBudget ids;
        {
            IncidencePools pools(*arena, ids, arena);
            PooledLineRegistry registry(*arena, pools);
            const auto first = populate(pools, registry, 8);
            EXPECT_EQ(registry.query_result(first.line, 0).result_size(), 2u);
        }
        identity_bytes = ids.admitted_bytes();
    }
    // The caller owns the identity admission. Destroying the registry and
    // pools retires their storage and leaves that separate charge intact.
    EXPECT_EQ(arena->used, identity_bytes);
}

TEST(PooledLineRegistryPersistence, ThreeIndependentRawBudgetGenerationsOutlivePredecessors)
{
    struct Raw
    {
        Budget budget{ample};
        IdentityBudget ids;
        IncidencePools pools;
        PooledLineRegistry registry;
        PooledLineEdit first;
        Raw() : pools(budget, ids), registry(budget, pools), first(populate(pools, registry, 4)) {}
        explicit Raw(const Raw& old)
            : pools(old.pools, budget, ids), registry(old.registry, budget, pools), first(old.first)
        {
        }
    };
    auto first = std::make_unique<Raw>();
    auto second = std::make_unique<Raw>(*first);
    first.reset();
    EXPECT_EQ(second->registry.query_result(second->first.line, 0).result_size(), 2u);
    auto third = std::make_unique<Raw>(*second);
    second.reset();
    EXPECT_EQ(third->registry.query_result(third->first.line, 0).result_size(), 2u);
    auto removed = third->first;
    removed.remove = true;
    third->registry.begin_replay(false);
    third->registry.apply(std::span(&removed, 1));
    third->registry.recompute([](auto, auto) {});
    EXPECT_THROW((void) third->registry.query_result(removed.line, 0), std::invalid_argument);
}

TEST(PooledLineRegistryPersistence, ActualNativeCertificationAdmitsItsExactFrozenReplayReservation)
{
    const auto config = grid();
    const auto kernel = cartesian_adaptive_kernel(config);
    const auto accepted_leaves = leaves(config);
    const auto comm = Teuchos::rcp(new Teuchos::SerialComm<int>);
    auto arena = std::make_shared<Budget>(Budget{ample});
    IdentityBudget ids;
    DistributedAdaptiveIncidenceSession source(*kernel, comm, {}, {}, *arena, ids, arena);
    const auto lookup = [&](const auto& key) -> const AdaptiveLeaf&
    {
        const auto found = std::find_if(
            accepted_leaves.begin(), accepted_leaves.end(), [&](const auto& leaf) { return leaf.key == key; });
        if (found == accepted_leaves.end())
            throw std::invalid_argument("Unknown accepted line fixture leaf.");
        return *found;
    };
    auto rows = source.bootstrap(accepted_leaves, lookup);
    const auto retired_key = accepted_leaves.front().key;
    const auto found =
        std::find_if(rows.begin(), rows.end(), [&](const auto& row) { return row.leaf->key == retired_key; });
    ASSERT_NE(found, rows.end());
    const std::array retired{
        IncidenceCellVersion{retired_key, found->leaf->revision, found->row->incidence_generation}};
    const auto delivery = rows.capacity() * sizeof(IncidenceRowDelivery);
    std::vector<IncidenceRowDelivery>().swap(rows);
    arena->used -= delivery;
    const auto old = source.subface_snapshot();
    ASSERT_TRUE(old);
    auto refined = accepted_leaves;
    const auto split = kernel->split(refined.front(), AdaptiveSplit::Isotropic);
    refined.erase(refined.begin());
    std::vector<AdaptiveCellKey> inserted;
    for (const auto& child : split.children)
    {
        inserted.push_back(child.key);
        refined.push_back(child);
    }
    const auto refined_lookup = [&](const auto& key) -> const AdaptiveLeaf&
    {
        const auto row =
            std::find_if(refined.begin(), refined.end(), [&](const auto& leaf) { return leaf.key == key; });
        if (row == refined.end())
            throw std::invalid_argument("Unknown refined line fixture leaf.");
        return *row;
    };
    const auto accepted_output = canonical_subface_output(*old);
    const auto apply = [&](std::size_t cap, ReplayReservation* reservation = nullptr)
    {
        auto candidate_arena = std::make_shared<Budget>(Budget{cap});
        IdentityBudget candidate_ids;
        auto candidate = source.fork(*candidate_arena, candidate_ids, candidate_arena);
        const auto delta_with_limit = [&]
        {
            struct ResetReservation
            {
                ~ResetReservation() { ReplayReservation::current = nullptr; }
            } reset;
            ReplayReservation::current = reservation;
            PooledLineRegistryAdmissionTestAccess hook(reservation ? ReplayReservation::tighten : nullptr);
            return candidate->apply_delta(retired, inserted, ContactHaloContext{0, 1, 0, 1, 1}, refined_lookup);
        };
        auto delta = delta_with_limit();
        EXPECT_GT(candidate->statistics().certified_line_witnesses, 1u);
        EXPECT_NE(candidate->subface_snapshot(), old);
        EXPECT_LE(candidate_arena->peak, cap);
        const auto peak = candidate_arena->peak;
        auto bytes = delta.changed.capacity() * sizeof(IncidenceRowDelivery) +
                     delta.retired.capacity() * sizeof(IncidenceCellVersion);
        for (const auto& item : delta.retired)
            bytes += key_capacity(item.key);
        delta = {};
        candidate_arena->used -= bytes;
        return std::pair{peak, canonical_subface_output(*candidate->subface_snapshot())};
    };
    const auto [peak, expected] = apply(ample);
    ASSERT_GT(peak, 1u);
    ASSERT_NE(expected, accepted_output);
    // The first fork can reap expired private source controls.
    const auto before = arena->used;
    // Whole-session page sizes and owner-tree paths can change under pressure.
    // A successful tighter replay must still certify and reproduce every piece.
    try
    {
        EXPECT_EQ(apply(peak - 1).second, expected);
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message(error.what());
        EXPECT_NE(message.find("budget"), std::string::npos) << message;
    }

    ReplayReservation exact{false};
    EXPECT_EQ(apply(ample, &exact).second, expected);
    ASSERT_GT(exact.bytes, 0u);
    EXPECT_EQ(exact.limit, exact.used + exact.bytes);

    ReplayReservation short_limit{true};
    try
    {
        (void) apply(ample, &short_limit);
        ADD_FAILURE() << "The mandatory frozen native replay reservation exceeded its byte limit.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_EQ(short_limit.bytes, exact.bytes);
        const auto required = "used=" + std::to_string(short_limit.used) +
                              " request=" + std::to_string(short_limit.bytes) +
                              " limit=" + std::to_string(short_limit.limit);
        EXPECT_NE(std::string(error.what()).find(required), std::string::npos) << error.what();
    }
    EXPECT_EQ(source.subface_snapshot(), old);
    EXPECT_EQ(canonical_subface_output(*old), accepted_output);
    EXPECT_EQ(arena->used, before);
}
