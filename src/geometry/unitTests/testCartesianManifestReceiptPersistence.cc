/** Actual native source receipts across serial and MPI edit generations. */
#include "geometry/mesh/AMR/CartesianSourceManifests.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIncidence.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "geometry/mesh/AMR/IncidencePhysicalOwners.hh"
#include "geometry/mesh/AMR/SupportRegistry.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <array>
#include <gtest/gtest.h>
#include <initializer_list>
#include <map>
#include <optional>
#include <set>

namespace
{
namespace A = SimpleFluid::Meshes::AMR;
namespace D = A::adaptive_detail;
using Manifest = D::CartesianSourceManifestSet;
constexpr auto cap = std::size_t(64) * 1024 * 1024;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using Identity = std::pair<std::uintptr_t, D::ContactPayloadRole>;
using Metadata = std::map<Identity, std::size_t>;
Metadata payload(const Manifest& manifests)
{
    Metadata result;
    manifests.visit_borrowed_payload(
        [&](const D::ContactPayloadAllocation& allocation)
        {
            using Role = D::GeometryContactPayloadRole;
            const auto role = allocation.identity.role;
            if (role == Role::CartesianManifestObject || role == Role::CartesianManifestSources ||
                role == Role::CartesianManifestIdentity || role == Role::CartesianManifestRanges ||
                role == Role::CartesianManifestLeases || role == Role::CartesianExplicitCellHeaders ||
                role == Role::CartesianExplicitSupportKeys)
                result.emplace(Identity{reinterpret_cast<std::uintptr_t>(allocation.identity.address), role},
                    allocation.payload_bytes);
        });
    return result;
}
std::shared_ptr<const Manifest> capture(const D::DistributedAdaptiveIncidenceSession& incidence)
{
    const auto& points = incidence.point_pool().source();
    std::shared_ptr<const Manifest> result;
    if (points)
        points->visit_dependencies(
            [&](const auto& owner, const D::Budget*, std::size_t, bool)
            {
                using T = std::remove_const_t<typename std::remove_cvref_t<decltype(owner)>::element_type>;
                if constexpr (std::is_same_v<T, Manifest>)
                    result = owner;
            });
    return result;
}
struct Session
{
    std::shared_ptr<D::Budget> memory = std::make_shared<D::Budget>(D::Budget{cap});
    D::IdentityBudget identities;
    std::shared_ptr<D::ContactPhysicalResidency> residency;
    std::unique_ptr<D::DistributedAdaptiveIncidenceSession> incidence;
};
void expect_retired_receipt_import(Session& current, const std::shared_ptr<const Manifest>& latest,
    std::initializer_list<std::shared_ptr<const Manifest>> captured)
{
    using Role = D::ContactPhysicalResidency::Role;
    auto& budget = *current.memory;
    auto policy = D::CartesianGroupAuthority::make_arena(budget, current.memory);
    D::IncidencePhysicalOwners::bind_policy(policy, current.residency);
    ASSERT_TRUE(latest->local_dependencies_admitted_in(*policy));

    std::shared_ptr<const D::CartesianSourceInventory> local;
    latest->visit_dependencies(
        [&](const auto& owner, const D::Budget*, std::size_t, bool)
        {
            using T = std::remove_const_t<typename std::remove_cvref_t<decltype(owner)>::element_type>;
            if constexpr (std::is_same_v<T, D::CartesianSourceInventory>)
                local = owner;
        });
    ASSERT_TRUE(local);
    // Keep an independent current local-inventory grant. Only receipt funding
    // ends, so a local grant cannot mask the missing peer graph admission.
    D::IncidencePhysicalOwners add(*current.residency, policy.get(), budget, true);
    ASSERT_FALSE(add.observe(Role::SourceDependency, local, local->metadata_bytes(), local->allocation_budget()));
    D::IncidencePhysicalOwners retire(*current.residency, policy.get(), budget, false);
    for (const auto& manifest : captured)
    {
        const D::ContactOwnerUsage::Key key{manifest.get(), std::uint32_t(Role::SourceDependency)};
        while (current.residency->usage().find(key))
            if (retire.observe(
                    Role::SourceDependency, manifest, manifest->metadata_bytes(), manifest->allocation_budget()))
                manifest->visit_dependencies<D::CartesianDependencyVisit::Observed>(
                    [&](const auto& owner, const D::Budget* origin, std::size_t bytes, bool)
                    {
                        using T = std::remove_const_t<typename std::remove_cvref_t<decltype(owner)>::element_type>;
                        constexpr auto role = std::is_same_v<T, D::cartesian_detail::AuthorityOwnerCredit>
                                                  ? Role::CreditMetadata
                                                  : Role::SourceDependency;
                        return retire.observe(role, owner, bytes, origin);
                    });
    }
    current.residency->finish_overlap();
    EXPECT_TRUE(local->dependencies_admitted_in(*policy));
    EXPECT_FALSE(policy->prepaid_policy.expired());
    EXPECT_FALSE(latest->local_dependencies_admitted_in(*policy));

    std::set<const void*> receipts;
    for (const auto& [key, bytes] : payload(*latest))
        receipts.insert(reinterpret_cast<const void*>(key.first));
    std::size_t unfunded = 0;
    latest->visit_dependencies<D::CartesianDependencyVisit::Physical>(
        [&](const auto& owner, const D::Budget* origin, std::size_t bytes, bool)
        {
            if (origin != &budget && receipts.contains(owner.get()) && !policy->prepaid(owner, origin, bytes, false))
                ++unfunded;
        });
    ASSERT_GT(unfunded, 0u);
    const auto run = [&](std::optional<std::size_t> cap, bool short_credit)
    {
        const auto before = budget.used, previous_peak = budget.peak, previous_limit = budget.limit;
        budget.peak = before;
        if (cap)
            budget.limit = before + *cap;
        const auto import = [&]
        {
            auto target = D::CartesianGroupAuthority::make_arena(budget, current.memory);
            D::IncidencePhysicalOwners::bind_policy(target, current.residency);
            D::cartesian_detail::Allocation memory(budget);
            std::vector<std::shared_ptr<D::cartesian_detail::AuthorityOwnerCredit>> leases;
            D::cartesian_detail::CreditConstruction credits(target);
            latest->retain_dependencies(credits, leases, memory);
            credits.committed = true;
            latest->visit_dependencies<D::CartesianDependencyVisit::Physical>(
                [&](const auto& owner, const D::Budget* origin, std::size_t bytes, bool)
                {
                    if (origin != &budget && receipts.contains(owner.get()))
                        EXPECT_TRUE(target->admits_owner(owner, bytes));
                });
        };
        if (short_credit)
            EXPECT_THROW(import(), std::invalid_argument);
        else
            EXPECT_NO_THROW(import());
        const auto peak = budget.peak - before;
        budget.limit = previous_limit;
        budget.peak = std::max(previous_peak, budget.peak);
        EXPECT_EQ(budget.used, before);
        return peak;
    };
    const auto exact = run({}, false);
    ASSERT_GT(exact, 1u);
    EXPECT_EQ(run(exact, false), exact);
    (void) run(exact - 1, true);
}
struct Work
{
    std::size_t shared = 0, fresh = 0, pages = 0, page_bytes = 0, copied_body_bytes = 0;
};

Work measure(std::size_t columns, bool verify_retirement)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto rank = comm->getRank(), ranks = comm->getSize();
    const auto require_collectively = [&](bool condition, const char* message)
    {
        const int local = condition;
        int all = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local, &all);
        if (!all)
            throw std::invalid_argument(message);
    };

    A::AMRGrid config;
    // Compact axes keep this receipt-storage fixture independent of the
    // native line coordinate byte-order radix path at128. Checkerboard owners
    // still contribute192/384 isolated ranges to actual peer receipts.
    config.root_blocks = {columns, 2, 2};
    config.block_cells = {2, 2, 2};
    config.extent = {2.0 * static_cast<double>(columns), 4.0, 4.0};
    const auto kernel = A::cartesian_adaptive_kernel(config);
    std::vector<A::AdaptiveLeaf> owned;
    for (std::uint64_t x = 0; x < 2 * columns; ++x)
        for (std::uint64_t y = 0; y < 4; ++y)
            for (std::uint64_t z = 0; z < 4; ++z)
                if (rank == int(ranks == 1 ? 0 : (x + y + z) % 2))
                    owned.push_back(A::cartesian_adaptive_leaf(
                        config, {config.region_id, A::CartesianCellAncestry{0, {x, y, z}}}, {'d'}));
    EXPECT_EQ(owned.empty(), rank >= 2);
    const A::AdaptiveCellKey changed{config.region_id, A::CartesianCellAncestry{0, {0, 0, 0}}};
    const auto lookup = [&](const A::AdaptiveCellKey& key) -> const A::AdaptiveLeaf&
    {
        const auto position = std::ranges::find(owned, key, &A::AdaptiveLeaf::key);
        if (position == owned.end())
            throw std::invalid_argument("Missing receipt fixture owner leaf.");
        return *position;
    };
    auto context = D::ContactHaloContext{0, 0, 0, 0, 27};
    Session first, second, third;
    if (verify_retirement)
        first.residency = std::make_shared<D::ContactPhysicalResidency>(first.memory);
    first.incidence = std::make_unique<D::DistributedAdaptiveIncidenceSession>(*kernel, comm,
        A::AdaptiveClosureOptions{}, context, *first.memory, first.identities, first.memory, first.residency);
    auto rows = first.incidence->bootstrap(owned, lookup);
    std::uint64_t row_generation = 0;
    const auto record_generation = [&](const auto& changed_rows)
    {
        const auto row =
            std::ranges::find_if(changed_rows, [&](const auto& delivery) { return delivery.leaf->key == changed; });
        require_collectively(rank != 0 || row != changed_rows.end(), "Missing receipt fixture revision row.");
        if (rank == 0)
            row_generation = row->row->incidence_generation;
    };
    record_generation(rows);
    auto bytes = rows.capacity() * sizeof(D::IncidenceRowDelivery);
    decltype(rows){}.swap(rows);
    first.memory->used -= bytes;
    auto oldest = capture(*first.incidence);
    require_collectively(bool(oldest), "Missing certified first owner manifest.");
    const auto before = payload(*oldest);
    if (ranks > 1)
    {
        EXPECT_EQ(oldest->source_count(), std::size_t(ranks));
        for (std::size_t i = 0; i != oldest->source_count(); ++i)
        {
            const auto& receipt = oldest->source(i);
            const auto expected = receipt.owner() < 2 ? 16 * columns : 0;
            EXPECT_EQ(receipt.cell_count(), expected);
            EXPECT_EQ(receipt.range_count(), expected);
        }
    }
    const auto revision = rank == 0 ? owned.front().revision : std::uint64_t(0);
    auto advance = [&](Session& previous, Session& next)
    {
        if (verify_retirement)
            next.residency = std::make_shared<D::ContactPhysicalResidency>(*previous.residency, next.memory);
        next.incidence = previous.incidence->fork(*next.memory, next.identities, next.memory, nullptr, next.residency);
        const std::array retired{
            D::IncidenceCellVersion{changed, rank == 0 ? owned.front().revision : 0, row_generation}};
        const std::array inserted{changed};
        if (rank == 0)
            ++owned.front().revision;
        ++context.round;
        const auto count = std::size_t(rank == 0);
        auto delta = next.incidence->apply_delta(
            std::span(retired).first(count), std::span(inserted).first(count), context, lookup);
        record_generation(delta.changed);
        auto memory = delta.changed.capacity() * sizeof(D::IncidenceRowDelivery) +
                      delta.retired.capacity() * sizeof(D::IncidenceCellVersion);
        for (const auto& row : delta.retired)
            memory += D::key_capacity(row.key);
        delta = {};
        next.memory->used -= memory;
        next.incidence->finish_inherited_overlap();
        if (next.residency)
            next.residency->finish_overlap();
        auto manifests = capture(*next.incidence);
        require_collectively(bool(manifests), "Missing certified edited owner manifest.");
        return manifests;
    };
    auto middle = advance(first, second);
    auto latest = advance(second, third);
    EXPECT_EQ(middle->receipt_work().shared_peers, std::size_t(ranks - 1));
    EXPECT_EQ(latest->receipt_work().shared_peers, std::size_t(ranks - 1));
    EXPECT_EQ(middle->owner_count(), std::size_t(ranks));
    EXPECT_EQ(latest->owner_count(), std::size_t(ranks));
    Work result;
    for (const auto& [id, size] : payload(*middle))
        if (before.contains(id))
            result.shared += size;
        else
            result.fresh += size;
    result.pages = middle->receipt_work().allocated_pages;
    result.page_bytes = middle->receipt_work().allocated_page_bytes;
    result.copied_body_bytes = middle->receipt_work().copied_body_bytes;
    EXPECT_GT(result.shared, 0u);

    first.incidence.reset();
    second.incidence.reset();
    third.incidence.reset();
    const A::AdaptiveBytes domain{'d'};
    EXPECT_TRUE(oldest->native_member({changed, revision, 0, 1, 0}, domain));
    EXPECT_TRUE(middle->native_member({changed, revision + 1, 0, 1, 0}, domain));
    EXPECT_TRUE(latest->native_member({changed, revision + 2, 0, 1, 0}, domain));
    EXPECT_FALSE(oldest->native_member({changed, revision + 2, 0, 1, 0}, domain));
    EXPECT_EQ(payload(*oldest), before);
    if (verify_retirement)
        expect_retired_receipt_import(third, latest, {oldest, middle, latest});
    const std::weak_ptr<const Manifest> first_lifetime(oldest), second_lifetime(middle);
    oldest.reset();
    middle.reset();
    EXPECT_TRUE(first_lifetime.expired());
    EXPECT_TRUE(second_lifetime.expired());
    EXPECT_TRUE(latest->native_member({changed, revision + 2, 0, 1, 0}, domain));
    return result;
}
} // namespace

TEST(CartesianManifestReceiptPersistence, FixedOwnerEditSharesDoubledUnchangedReceipts)
{
    const auto small = measure(12, true);
    const auto large = measure(24, false);
    if (Tpetra::getDefaultComm()->getSize() > 1)
        EXPECT_GT(large.shared, small.shared);
    RecordProperty("small_new_receipt_bytes", std::to_string(small.fresh));
    RecordProperty("large_new_receipt_bytes", std::to_string(large.fresh));
    RecordProperty("small_cumulative_receipt_page_bytes", std::to_string(small.page_bytes));
    RecordProperty("large_cumulative_receipt_page_bytes", std::to_string(large.page_bytes));
    EXPECT_LE(large.fresh, small.fresh + 16 * 1024);
    EXPECT_LE(large.pages, small.pages + 8);
    EXPECT_LE(large.page_bytes, small.page_bytes + 16 * 1024);
    EXPECT_LE(large.copied_body_bytes, small.copied_body_bytes + 4096);
}
