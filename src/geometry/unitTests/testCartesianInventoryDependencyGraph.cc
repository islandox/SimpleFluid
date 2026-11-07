/** Physical owner graph persistence across authenticated Cartesian inventory generations. */
#include "geometry/mesh/AMR/CartesianSourceInventory.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIncidence.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "geometry/mesh/AMR/IncidencePhysicalOwners.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_DefaultSerialComm.hpp>
#include <array>
#include <gtest/gtest.h>
#include <map>

namespace
{
namespace A = SimpleFluid::Meshes::AMR;
namespace D = A::adaptive_detail;
using Inventory = D::CartesianSourceInventory;
using Native = D::CartesianIncidenceSource;
using Role = D::ContactPhysicalResidency::Role;
constexpr std::size_t ample = 64 * 1024 * 1024;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
struct Arena
{
    std::shared_ptr<D::Budget> budget = std::make_shared<D::Budget>(D::Budget{ample});
    std::shared_ptr<D::ContactPhysicalResidency> residency;
    std::shared_ptr<D::CartesianAuthorityArena> authority = D::CartesianGroupAuthority::make_arena(*budget, budget);
    Arena() : residency(std::make_shared<D::ContactPhysicalResidency>(budget)) { bind(); }
    explicit Arena(const Arena& source)
        : residency(std::make_shared<D::ContactPhysicalResidency>(*source.residency, budget))
    {
        bind();
    }

private:
    void bind() { D::IncidencePhysicalOwners::bind_policy(authority, residency); }
};
A::AMRGrid grid()
{
    A::AMRGrid value;
    value.root_blocks = {384, 1, 1};
    value.block_cells = {2, 2, 2};
    value.extent = {768.0, 1.0, 1.0};
    return value;
}
A::AdaptiveCellKey key(std::size_t cell)
{
    return {grid().region_id, A::CartesianCellAncestry{0, {2 * cell, 0, 0}}};
}
Inventory::CapturedSource source(Arena& arena, std::size_t index, std::uint64_t revision, D::ContactHaloContext context)
{
    auto leaf = A::cartesian_adaptive_leaf(grid(), key(index), {'d'});
    leaf.revision = revision;
    Inventory::CapturedSource captured;
    if (std::fegetenv(&captured.environment))
        throw std::runtime_error("Cannot capture source test environment.");
    captured.source =
        Native::make(grid(), std::span(&leaf, 1), leaf.domain, 0, 1 + revision, context, *arena.budget, arena.budget);
    return captured;
}
std::shared_ptr<const Inventory> bootstrap(Arena& arena, std::size_t count)
{
    const D::ContactHaloContext context{0, 0, 0, 0, 1};
    std::vector<Inventory::CapturedSource> sources;
    sources.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
        sources.push_back(source(arena, i, 0, context));
    return Inventory::make(context, 0, 1, sources, *arena.budget, arena.budget, arena.authority);
}
std::shared_ptr<const Inventory> replace(Arena& arena, const std::shared_ptr<const Inventory>& original)
{
    auto context = original->context();
    ++context.round;
    const auto revision = original->revision(key(0));
    const std::array retired{Inventory::Retirement{key(0), revision}};
    const std::array added{source(arena, 0, revision + 1, context)};
    return original->derive(
        context, original->version() + 1, retired, added, *arena.budget, arena.budget, arena.authority);
}

struct Visits
{
    std::size_t nodes = 0, sources = 0;
};
Visits observe(D::ContactPhysicalResidency& residency, D::CartesianAuthorityArena& authority,
    const std::shared_ptr<const Inventory>& inventory, bool added)
{
    Visits visits;
    D::IncidencePhysicalOwners owners(residency, &authority, *authority.budget, added);
    if (!owners.observe(Role::SourceDependency, inventory, inventory->metadata_bytes(), inventory->allocation_budget()))
        return visits;
    inventory->visit_observed_dependencies(
        [&](const auto& owner, const D::Budget* origin, std::size_t bytes, bool)
        {
            using Element = typename std::remove_cvref_t<decltype(owner)>::element_type;
            using T = std::remove_const_t<Element>;
            if constexpr (std::is_same_v<T, Native>)
                ++visits.sources;
            else if constexpr (!std::is_same_v<T, D::cartesian_detail::AuthorityOwnerCredit>)
                ++visits.nodes;
            constexpr auto role = std::is_same_v<T, D::cartesian_detail::AuthorityOwnerCredit> ? Role::CreditMetadata
                                                                                               : Role::SourceDependency;
            return owners.observe(role, owner, bytes, origin);
        });
    return visits;
}
Visits observe(Arena& arena, const std::shared_ptr<const Inventory>& inventory, bool added)
{
    return observe(*arena.residency, *arena.authority, inventory, added);
}
std::map<const void*, std::size_t> pages(const Inventory& inventory)
{
    std::map<const void*, std::size_t> result;
    inventory.visit_borrowed_payload(
        [&](const D::ContactPayloadAllocation& allocation)
        {
            if (allocation.identity.role == D::ContactPayloadRole::CartesianInventoryDependencyNode)
                result.emplace(allocation.identity.address, allocation.payload_bytes);
        });
    return result;
}

void expect_unfunded_directory_fork(const D::IncidencePointDirectory<>& source, const Inventory& inventory)
{
    const auto graph = pages(inventory);
    ASSERT_FALSE(graph.empty());
    const auto run = [&](std::optional<std::size_t> cap, bool reject)
    {
        auto budget = std::make_shared<D::Budget>(D::Budget{ample});
        auto arena = D::CartesianGroupAuthority::make_arena(*budget, budget);
        const auto before = budget->used;
        budget->peak = before;
        if (cap)
            budget->limit = before + *cap;
        if (reject)
        {
            EXPECT_THROW((void) D::IncidencePointDirectory<>(source, *budget, budget, arena), std::invalid_argument);
            EXPECT_EQ(budget->used, before);
        }
        else
        {
            D::IncidencePointDirectory<> copy(source, *budget, budget, arena);
            EXPECT_EQ(copy.source(), source.source());
            EXPECT_EQ(copy.size(), source.size());
            std::size_t visited = 0;
            inventory.visit_physical_dependencies(
                [&](const auto& owner, const D::Budget*, std::size_t bytes, bool)
                {
                    if (graph.contains(owner.get()))
                    {
                        EXPECT_TRUE(arena->admits_owner(owner, bytes));
                        ++visited;
                    }
                });
            EXPECT_EQ(visited, graph.size());
        }
        const auto peak = budget->peak - before;
        arena.reset();
        EXPECT_EQ(budget->used, 0u);
        return peak;
    };
    const auto exact = run({}, false);
    ASSERT_GT(exact, 1u);
    EXPECT_EQ(run(exact, false), exact);
    (void) run(exact - 1, true);
}
} // namespace

TEST(CartesianInventoryDependencyGraph, FixedNativeEditSharesDoubledInventoryOwners)
{
    struct Work
    {
        std::size_t shared, pages, bytes, sources, nodes;
        std::size_t inventory_bytes, batch_slots, range_bytes, identity_bytes;
    };
    const auto measure = [](std::size_t count)
    {
        Arena accepted;
        const auto before = bootstrap(accepted, count);
        EXPECT_TRUE(before->has_observed_dependencies());
        EXPECT_EQ(observe(accepted, before, true).sources, count);
        const auto base = pages(*before);
        const auto retained = accepted.budget->used;

        Arena candidate(accepted);
        const auto next = replace(candidate, before);
        const auto added = observe(candidate, next, true);
        const auto removed = observe(candidate, before, false);
        EXPECT_EQ(next->dependency_work().inherited_admission_visits, 0u);
        EXPECT_EQ(next->cell_count(), count);
        EXPECT_EQ(next->revision(key(0)), 1u);
        EXPECT_EQ(before->revision(key(0)), 0u);
        EXPECT_EQ(pages(*before), base);
        EXPECT_EQ(accepted.budget->used, retained);
        Work work{};
        for (const auto& [identity, bytes] : pages(*next))
            if (base.contains(identity))
                work.shared += bytes;
        work.pages = next->dependency_work().allocated_pages;
        work.bytes = next->dependency_work().allocated_bytes;
        work.sources = added.sources + removed.sources;
        work.nodes = added.nodes + removed.nodes;
        work.inventory_bytes = next->metadata_bytes();
        next->visit_borrowed_payload(
            [&](const D::ContactPayloadAllocation& allocation)
            {
                if (allocation.owner != next.get())
                    return;
                using Payload = D::ContactPayloadRole;
                if (allocation.identity.role == Payload::CartesianInventoryBatches)
                    work.batch_slots = allocation.capacity;
                else if (allocation.identity.role == Payload::CartesianInventoryRanges)
                    work.range_bytes += allocation.payload_bytes;
                else if (allocation.identity.role == Payload::CartesianInventoryIdentity)
                    work.identity_bytes = allocation.payload_bytes;
            });
        EXPECT_GT(work.shared, 0u);
        EXPECT_LT(work.sources, 5 * 64u);
        return work;
    };
    const auto small = measure(192), large = measure(384);
    // Measure the separately rebuilt inventory shell without making its
    // current representation part of the persistent graph's contract.
    RecordProperty("small_inventory_bytes", std::to_string(small.inventory_bytes));
    RecordProperty("large_inventory_bytes", std::to_string(large.inventory_bytes));
    RecordProperty("small_inventory_batch_slots", std::to_string(small.batch_slots));
    RecordProperty("large_inventory_batch_slots", std::to_string(large.batch_slots));
    RecordProperty("small_inventory_range_bytes", std::to_string(small.range_bytes));
    RecordProperty("large_inventory_range_bytes", std::to_string(large.range_bytes));
    RecordProperty("small_inventory_identity_bytes", std::to_string(small.identity_bytes));
    RecordProperty("large_inventory_identity_bytes", std::to_string(large.identity_bytes));
    RecordProperty("small_dependency_owner_visits", std::to_string(small.sources + small.nodes));
    RecordProperty("large_dependency_owner_visits", std::to_string(large.sources + large.nodes));
    EXPECT_GT(large.shared, small.shared);
    EXPECT_LE(large.pages, small.pages + 2);
    EXPECT_LE(large.bytes, small.bytes + 4096);
    EXPECT_LE(large.sources, small.sources + 4);
    EXPECT_LE(large.nodes, small.nodes + 16);
}

TEST(CartesianInventoryDependencyGraph, HistoricalCapturesDoNotPinAncestorInventoryBodies)
{
    auto oldest_arena = std::make_unique<Arena>();
    auto oldest = bootstrap(*oldest_arena, 130);
    (void) observe(*oldest_arena, oldest, true);
    auto old_capture = oldest;
    EXPECT_EQ(observe(*oldest_arena, old_capture, true).sources, 0u);
    const std::weak_ptr<const Inventory> old_root(oldest);
    const std::weak_ptr<const Native> retired_source(oldest->lookup(key(0))->source());
    const auto* clean_source = oldest->lookup(key(1))->source().get();

    auto middle_arena = std::make_unique<Arena>(*oldest_arena);
    auto middle = replace(*middle_arena, oldest);
    (void) observe(*middle_arena, middle, true);
    EXPECT_EQ(observe(*middle_arena, oldest, false).sources, 0u);
    auto middle_capture = middle;
    EXPECT_EQ(observe(*middle_arena, middle_capture, true).sources, 0u);
    const std::weak_ptr<const Inventory> middle_root(middle);

    Arena current_arena(*middle_arena);
    const auto current = replace(current_arena, middle);
    (void) observe(current_arena, current, true);
    EXPECT_EQ(observe(current_arena, middle, false).sources, 0u);
    oldest.reset();
    middle.reset();
    oldest_arena.reset();
    middle_arena.reset();
    ASSERT_FALSE(old_root.expired());
    ASSERT_FALSE(middle_root.expired());
    EXPECT_EQ(old_capture->revision(key(0)), 0u);
    EXPECT_EQ(middle_capture->revision(key(0)), 1u);
    EXPECT_EQ(current->revision(key(0)), 2u);
    EXPECT_EQ(current->lookup(key(1))->source().get(), clean_source);

    EXPECT_GT(observe(current_arena, old_capture, false).sources, 0u);
    old_capture.reset();
    EXPECT_TRUE(old_root.expired());
    EXPECT_TRUE(retired_source.expired());
    EXPECT_GT(observe(current_arena, middle_capture, false).sources, 0u);
    middle_capture.reset();
    EXPECT_TRUE(middle_root.expired());
    EXPECT_EQ(current->lookup(key(1))->source().get(), clean_source);
}

TEST(CartesianInventoryDependencyGraph, ExactDerivationCreditAndOneByteShortLeaveAcceptedOwnersUnchanged)
{
    Arena accepted;
    const auto original = bootstrap(accepted, 130);
    (void) observe(accepted, original, true);
    const auto prior = pages(*original);
    const auto retained = accepted.budget->used;
    const auto run = [&](std::optional<std::size_t> cap, bool short_credit)
    {
        Arena candidate(accepted);
        auto next = original->context();
        ++next.round;
        const std::array retired{Inventory::Retirement{key(0), 0}};
        const std::array added{source(candidate, 0, 1, next)};
        const auto before = candidate.budget->used;
        candidate.budget->peak = before;
        if (cap)
            candidate.budget->limit = before + *cap;
        auto derive = [&]
        { return original->derive(next, 2, retired, added, *candidate.budget, candidate.budget, candidate.authority); };
        if (short_credit)
        {
            EXPECT_THROW((void) derive(), std::invalid_argument);
            EXPECT_EQ(candidate.budget->used, before);
        }
        else
        {
            const auto next_inventory = derive();
            EXPECT_EQ(next_inventory->revision(key(0)), 1u);
            EXPECT_EQ(next_inventory->dependency_work().inherited_admission_visits, 0u);
        }
        return candidate.budget->peak - before;
    };
    const auto exact = run({}, false);
    ASSERT_GT(exact, 1u);
    EXPECT_EQ(run(exact, false), exact);
    (void) run(exact - 1, true);
    EXPECT_EQ(accepted.budget->used, retained);
    EXPECT_EQ(pages(*original), prior);
    EXPECT_EQ(original->revision(key(0)), 0u);
}

TEST(CartesianInventoryDependencyGraph, CompatibilityImportKeepsFlatSourceOrderAndExactCredits)
{
    Arena accepted;
    const auto original = bootstrap(accepted, 4);
    (void) observe(accepted, original, true);
    auto budget = std::make_shared<D::Budget>(D::Budget{ample});
    auto authority = D::CartesianGroupAuthority::make_arena(*budget, budget);
    auto context = original->context();
    ++context.round;
    const std::array retired{Inventory::Retirement{key(0), 0}};
    const auto fallback = original->derive(context, 2, retired, {}, *budget, budget, authority);
    EXPECT_FALSE(fallback->has_observed_dependencies());
    EXPECT_EQ(fallback->dependency_work().inherited_admission_visits, 4u);
    EXPECT_EQ(fallback->cell_count(), 3u);

    std::vector<const Native*> order;
    fallback->visit_dependencies(
        [&](const auto& owner, const D::Budget*, std::size_t, bool)
        {
            using Element = typename std::remove_cvref_t<decltype(owner)>::element_type;
            if constexpr (std::is_same_v<std::remove_const_t<Element>, Native>)
                order.push_back(owner.get());
        });
    ASSERT_EQ(order.size(), 3u);
    for (std::size_t i = 0; i < order.size(); ++i)
        EXPECT_EQ(order[i], original->lookup(key(i + 1))->source().get());
}

TEST(CartesianInventoryDependencyGraph, LivePolicyDoesNotPrepayAReleasedPredecessorGrant)
{
    Arena accepted;
    const auto original = bootstrap(accepted, 130);
    (void) observe(accepted, original, true);
    Arena candidate(accepted);
    const auto next = replace(candidate, original);
    EXPECT_TRUE(next->dependencies_admitted_in(*candidate.authority));
    const auto inherited = pages(*original);
    EXPECT_GT(observe(candidate, original, false).nodes, 0u);
    candidate.residency->finish_overlap();
    ASSERT_FALSE(candidate.authority->prepaid_policy.expired());
    EXPECT_FALSE(next->dependencies_admitted_in(*candidate.authority));

    D::cartesian_detail::Allocation memory(*candidate.budget);
    std::vector<std::shared_ptr<D::cartesian_detail::AuthorityOwnerCredit>> leases;
    D::cartesian_detail::CreditConstruction credits(candidate.authority);
    next->retain_dependencies(credits, leases, memory);
    credits.committed = true;
    std::size_t visited = 0;
    next->visit_physical_dependencies(
        [&](const auto& owner, const D::Budget*, std::size_t bytes, bool)
        {
            if (inherited.contains(owner.get()))
            {
                EXPECT_TRUE(candidate.authority->admits_owner(owner, bytes));
                ++visited;
            }
        });
    EXPECT_GT(visited, 0u);
}

TEST(CartesianInventoryDependencyGraph, ActualPointSnapshotsRetainTheirOwnManifestGeneration)
{
    A::AMRGrid value;
    value.block_cells = {2, 2, 2};
    const auto kernel = A::cartesian_adaptive_kernel(value);
    const auto comm = Teuchos::rcp(new Teuchos::SerialComm<int>);
    std::vector<A::AdaptiveLeaf> leaves;
    for (unsigned i = 0; i < 8; ++i)
        leaves.push_back(A::cartesian_adaptive_leaf(
            value, {value.region_id, A::CartesianCellAncestry{0, {i & 1, (i >> 1) & 1, (i >> 2) & 1}}}, {'d'}));
    const auto changed = leaves.front().key;
    const auto lookup = [&](const auto& key) -> const A::AdaptiveLeaf&
    {
        const auto position = std::ranges::find(leaves, key, &A::AdaptiveLeaf::key);
        if (position == leaves.end())
            throw std::invalid_argument("Unknown inventory fixture leaf.");
        return *position;
    };
    const auto generation = [&](const std::vector<D::IncidenceRowDelivery>& rows)
    {
        const auto found =
            std::ranges::find_if(rows, [&](const auto& delivery) { return delivery.leaf->key == changed; });
        if (found == rows.end())
            throw std::invalid_argument("Missing changed inventory fixture row.");
        return found->row->incidence_generation;
    };
    const auto capture = [](const std::shared_ptr<const D::ContactSourcePointSnapshot>& points)
    {
        std::shared_ptr<const D::CartesianSourceManifestSet> manifests;
        if (points)
            points->visit_dependencies(
                [&](const auto& owner, const D::Budget*, std::size_t, bool)
                {
                    using T = std::remove_const_t<typename std::remove_cvref_t<decltype(owner)>::element_type>;
                    if constexpr (std::is_same_v<T, D::CartesianSourceManifestSet>)
                        manifests = owner;
                });
        std::shared_ptr<const Inventory> inventory;
        if (manifests)
            manifests->visit_dependencies(
                [&](const auto& owner, const D::Budget*, std::size_t, bool)
                {
                    using T = std::remove_const_t<typename std::remove_cvref_t<decltype(owner)>::element_type>;
                    if constexpr (std::is_same_v<T, Inventory>)
                        inventory = owner;
                });
        return std::pair{std::move(manifests), std::move(inventory)};
    };
    struct Session
    {
        std::shared_ptr<D::Budget> memory = std::make_shared<D::Budget>(D::Budget{ample});
        D::IdentityBudget identities;
        std::shared_ptr<D::ContactPhysicalResidency> residency;
        std::unique_ptr<D::DistributedAdaptiveIncidenceSession> incidence;
    };
    Session first, second, third;
    auto context = D::ContactHaloContext{0, 0, 0, 0, 15};
    first.residency = std::make_shared<D::ContactPhysicalResidency>(first.memory);
    first.incidence = std::make_unique<D::DistributedAdaptiveIncidenceSession>(*kernel, comm,
        A::AdaptiveClosureOptions{}, context, *first.memory, first.identities, first.memory, first.residency);
    auto rows = first.incidence->bootstrap(leaves, lookup);
    auto row_generation = generation(rows);
    const auto row_bytes = rows.capacity() * sizeof(D::IncidenceRowDelivery);
    decltype(rows){}.swap(rows);
    first.memory->used -= row_bytes;
    auto first_points = first.incidence->point_pool().source();
    auto [first_manifest, first_inventory] = capture(first_points);
    ASSERT_TRUE(first_manifest);
    ASSERT_TRUE(first_inventory && first_inventory->has_observed_dependencies());
    const std::weak_ptr<const Inventory> first_lifetime(first_inventory);

    auto advance = [&](Session& previous, Session& candidate)
    {
        candidate.residency = std::make_shared<D::ContactPhysicalResidency>(*previous.residency, candidate.memory);
        candidate.incidence = previous.incidence->fork(
            *candidate.memory, candidate.identities, candidate.memory, nullptr, candidate.residency);
        const std::array retired{D::IncidenceCellVersion{changed, leaves.front().revision, row_generation}};
        const std::array inserted{changed};
        ++leaves.front().revision;
        ++context.round;
        auto delta = candidate.incidence->apply_delta(retired, inserted, context, lookup);
        row_generation = generation(delta.changed);
        auto bytes = delta.changed.capacity() * sizeof(D::IncidenceRowDelivery) +
                     delta.retired.capacity() * sizeof(D::IncidenceCellVersion);
        for (const auto& row : delta.retired)
            bytes += D::key_capacity(row.key);
        delta = {};
        candidate.memory->used -= bytes;
        candidate.incidence->finish_inherited_overlap();
        candidate.residency->finish_overlap();
        return candidate.incidence->point_pool().source();
    };
    auto second_points = advance(first, second);
    auto [second_manifest, second_inventory] = capture(second_points);
    ASSERT_TRUE(second_manifest);
    ASSERT_TRUE(second_inventory && second_inventory->has_observed_dependencies());
    const std::weak_ptr<const Inventory> second_lifetime(second_inventory);
    auto third_points = advance(second, third);
    const auto [third_manifest, third_inventory] = capture(third_points);
    ASSERT_TRUE(third_manifest);
    ASSERT_TRUE(third_inventory && third_inventory->has_observed_dependencies());
    expect_unfunded_directory_fork(third.incidence->point_pool(), *third_inventory);

    first.incidence.reset();
    second.incidence.reset();
    third.incidence.reset();
    EXPECT_LT(first_manifest->generation(), second_manifest->generation());
    EXPECT_LT(second_manifest->generation(), third_manifest->generation());
    EXPECT_EQ(first_inventory->revision(changed), 0u);
    EXPECT_EQ(second_inventory->revision(changed), 1u);
    EXPECT_EQ(third_inventory->revision(changed), 2u);

    // Simulate the exact inventory grants ending while escaped snapshots and
    // manifests remain available outside the live residency.
    auto authority = D::CartesianGroupAuthority::make_arena(*third.memory, third.memory);
    D::IncidencePhysicalOwners::bind_policy(authority, third.residency);
    EXPECT_TRUE(third_manifest->local_dependencies_admitted_in(*authority));
    for (const auto& inventory : {third_inventory, second_inventory})
    {
        const D::ContactOwnerUsage::Key identity{inventory.get(), std::uint32_t(Role::SourceDependency)};
        while (third.residency->usage().find(identity))
            (void) observe(*third.residency, *authority, inventory, false);
    }
    third.residency->finish_overlap();
    ASSERT_FALSE(authority->prepaid_policy.expired());
    EXPECT_FALSE(third_manifest->local_dependencies_admitted_in(*authority));
    {
        // This is the manifest dependency adapter used by point snapshot construction.
        D::cartesian_detail::Allocation memory(*third.memory);
        std::vector<std::shared_ptr<D::cartesian_detail::AuthorityOwnerCredit>> leases;
        D::cartesian_detail::CreditConstruction credits(authority);
        third_manifest->retain_dependencies(credits, leases, memory);
        credits.committed = true;
        std::size_t foreign = 0;
        third_inventory->visit_physical_dependencies(
            [&](const auto& owner, const D::Budget* origin, std::size_t bytes, bool)
            {
                if (origin != third.memory.get())
                {
                    EXPECT_TRUE(authority->admits_owner(owner, bytes));
                    ++foreign;
                }
            });
        EXPECT_GT(foreign, 0u);
    }
    first_points.reset();
    first_manifest.reset();
    first_inventory.reset();
    second_points.reset();
    second_manifest.reset();
    second_inventory.reset();
    EXPECT_TRUE(first_lifetime.expired());
    EXPECT_TRUE(second_lifetime.expired());
    EXPECT_EQ(third_inventory->revision(changed), 2u);
}
