#include "cartesian_incidence_test_helpers.hh"
#include "geometry/mesh/AMR/CartesianAuthorityCredits.hh"
#include <array>
#include <gtest/gtest.h>
#include <memory>
#include <vector>
namespace
{
using namespace cartesian_incidence_test;
using namespace SimpleFluid::Meshes::AMR::adaptive_detail::cartesian_detail;
struct Payload
{
    std::array<char, 4096> bytes{};
};
TEST(CartesianAuthorityCredits, PrivateOverlapCensusIncludesHeaderNodesAndEachAliasedTokenOnce)
{
    auto origin = std::make_shared<Budget>(Budget{ample});
    constexpr auto body_bytes = sizeof(Payload) + 128;
    origin->add(body_bytes);
    std::shared_ptr<const Payload> owner(new Payload,
        [origin](const Payload* value)
        {
            delete value;
            origin->used -= body_bytes;
        });
    auto accepted = std::make_shared<Budget>(Budget{ample});
    auto arena = CartesianGroupAuthority::make_arena(*accepted, accepted);
    const auto header = accepted->used;
    EXPECT_EQ(arena->private_metadata_bytes(), header);
    Allocation first_memory(*accepted), second_memory(*accepted);
    std::vector<std::shared_ptr<AuthorityOwnerCredit>> first, second;
    {
        CreditConstruction construction(arena);
        construction.retain(owner, origin.get(), body_bytes, true, first, first_memory);
        construction.committed = true;
    }
    {
        CreditConstruction construction(arena);
        construction.retain(owner, origin.get(), body_bytes, true, second, second_memory);
        construction.committed = true;
    }
    ASSERT_EQ(arena->entries.size(), 1);
    ASSERT_EQ(first.size(), 1);
    ASSERT_EQ(second.size(), 1);
    EXPECT_EQ(first.front(), second.front());
    const auto metadata = sizeof(AuthorityOwnerCredit) + ArenaCredits::control_bytes;
    const auto expected = header + ArenaCredits::node_bytes + metadata;
    EXPECT_EQ(arena->private_metadata_bytes(), expected);
    EXPECT_TRUE(arena->owns_credit(first.front().get()));
    EXPECT_EQ(accepted->used, expected + body_bytes + first_memory.bytes + second_memory.bytes);
    // Admission includes accepted private metadata alongside a new candidate
    // arena. Exact capacity passes; one byte less rejects before allocation.
    auto candidate = std::make_shared<Budget>(Budget{ample});
    const auto candidate_header = header;
    candidate->limit = expected + candidate_header - 1;
    candidate->add(expected);
    EXPECT_THROW(CartesianGroupAuthority::make_arena(*candidate, candidate), std::exception);
    EXPECT_EQ(candidate->used, expected);
    candidate->used -= expected;
    candidate->limit = expected + candidate_header;
    candidate->add(expected);
    {
        auto child = CartesianGroupAuthority::make_arena(*candidate, candidate);
        EXPECT_EQ(candidate->used, candidate->limit);
    }
    candidate->used -= expected;
    EXPECT_EQ(candidate->used, 0);
    // Expired payload plus escaped lease still has a real token/control/node.
    owner.reset();
    EXPECT_EQ(origin->used, 0);
    arena->reap();
    EXPECT_EQ(arena->entries.size(), 1);
    EXPECT_EQ(arena->private_metadata_bytes(), expected);
    first.clear();
    second.clear();
    arena->reap();
    EXPECT_TRUE(arena->entries.empty());
    EXPECT_EQ(arena->private_metadata_bytes(), header);
    arena.reset();
    EXPECT_EQ(accepted->used, first_memory.bytes + second_memory.bytes);
}
TEST(CartesianAuthorityCredits, ObserverVisitsActualOwnersWithoutAllocatingTypedCredits)
{
    Budget budget{ample};
    Allocation memory(budget);
    std::vector<std::shared_ptr<AuthorityOwnerCredit>> leases;
    const auto owner = std::make_shared<const Payload>();
    struct Probe
    {
        const void* address;
        const Budget* origin;
        std::size_t visits = 0;
    } probe{owner.get(), &budget};
    CreditConstruction observe(budget, &probe,
        [](void* context, const std::shared_ptr<const void>& actual, const Budget* origin, std::size_t bytes,
            bool keeper, bool metadata, std::uint64_t scope)
        {
            auto& value = *static_cast<Probe*>(context);
            EXPECT_EQ(actual.get(), value.address);
            EXPECT_EQ(origin, value.origin);
            EXPECT_EQ(bytes, sizeof(Payload));
            EXPECT_TRUE(keeper);
            EXPECT_FALSE(metadata);
            EXPECT_EQ(scope, 0u);
            return value.visits++ == 0;
        });
    EXPECT_TRUE(observe.descend(owner, &budget, sizeof(Payload), true, leases, memory));
    EXPECT_FALSE(observe.descend(owner, &budget, sizeof(Payload), true, leases, memory));
    EXPECT_EQ(probe.visits, 2u);
    EXPECT_TRUE(leases.empty());
    EXPECT_EQ(budget.used, 0u);
}

TEST(CartesianAuthorityCredits, PrepaidBodyPreservesCapacityAndRechargesAfterPolicyExpires)
{
    auto source = std::make_shared<Budget>(Budget{ample});
    auto budget = std::make_shared<Budget>(Budget{ample});
    constexpr auto bytes = sizeof(Payload) + 128;
    const auto owner = std::make_shared<const Payload>();
    auto arena = CartesianGroupAuthority::make_arena(*budget, budget);
    Allocation memory(*budget);
    std::vector<std::shared_ptr<AuthorityOwnerCredit>> leases;
    struct Policy
    {
        const void* address;
        const Budget* origin;
    };
    auto policy = std::make_shared<const Policy>(Policy{owner.get(), source.get()});
    arena->set_prepaid_policy(policy,
        [](const void* context, const std::shared_ptr<const void>& actual, const Budget* origin, std::size_t size,
            bool metadata) noexcept
        {
            const auto& selected = *static_cast<const Policy*>(context);
            return actual.get() == selected.address && origin == selected.origin && size == bytes && !metadata;
        });
    budget->add(bytes);
    {
        CreditConstruction construction(arena);
        EXPECT_TRUE(construction.retain(owner, source.get(), bytes, true, leases, memory));
        construction.committed = true;
    }
    ASSERT_EQ(leases.size(), 1u);
    EXPECT_EQ(leases.front()->payload, bytes);
    EXPECT_EQ(leases.front()->charged_payload, 0u);
    EXPECT_TRUE(arena->admits_owner(owner, bytes));
    EXPECT_FALSE(arena->find_paid_token(owner, source.get(), bytes));
    policy.reset();
    budget->used -= bytes;
    const auto baseline = budget->used;
    EXPECT_FALSE(arena->admits_owner(owner, bytes));
    {
        CreditConstruction rejected(arena);
        EXPECT_TRUE(rejected.retain(owner, source.get(), bytes, true, leases, memory));
        EXPECT_EQ(leases.front()->charged_payload, bytes);
    }
    EXPECT_EQ(budget->used, baseline);
    EXPECT_EQ(leases.front()->charged_payload, 0u);
    {
        CreditConstruction retry(arena);
        EXPECT_TRUE(retry.retain(owner, source.get(), bytes, true, leases, memory));
        retry.committed = true;
    }
    EXPECT_EQ(budget->used, baseline + bytes);
    auto paid = arena->find_paid_token(owner, source.get(), bytes);
    ASSERT_TRUE(paid);
    EXPECT_EQ(paid->take_payload_charge(), bytes);
    EXPECT_EQ(budget->used, baseline + bytes);
    paid.reset();
    leases.clear();
    arena.reset();
    EXPECT_EQ(budget->used, memory.bytes + bytes);
    budget->used -= bytes;
}
TEST(CartesianAuthorityCredits, ExpiredPolicyReapsPreviouslyActiveOwnerWithoutAHint)
{
    auto origin = std::make_shared<Budget>(Budget{ample});
    auto destination = std::make_shared<Budget>(Budget{ample});
    auto arena = CartesianGroupAuthority::make_arena(*destination, destination);
    const auto header = destination->used;
    auto owner = std::make_shared<const Payload>();
    auto policy = std::make_shared<const int>(1);
    arena->set_prepaid_policy(policy, [](const void*, const std::shared_ptr<const void>&, const Budget*, std::size_t,
                                          bool) noexcept { return false; });
    Allocation memory(*destination);
    std::vector<std::shared_ptr<AuthorityOwnerCredit>> leases;
    {
        CreditConstruction credit(arena);
        credit.retain(owner, origin.get(), sizeof(Payload), true, leases, memory);
        credit.committed = true;
    }
    ASSERT_EQ(leases.size(), 1u);
    ASSERT_EQ(arena->retired_hints.size(), 1u);
    const auto hint_census = [&]
    {
        std::size_t count = 0;
        arena->visit_borrowed_payload(
            [&](const ContactPayloadAllocation& allocation)
            {
                if (allocation.identity.role != ContactPayloadRole::CartesianArenaRetirementHints)
                    return;
                ++count;
                ASSERT_FALSE(arena->retired_hints.empty());
                EXPECT_EQ(allocation.identity.address, &*arena->retired_hints.begin());
                EXPECT_EQ(allocation.owner, arena.get());
                EXPECT_EQ(allocation.kind, ContactPayloadKind::NodeValue);
                EXPECT_EQ(allocation.payload_bytes, sizeof(ArenaCredits::RetiredHints::value_type));
                EXPECT_TRUE(allocation.allocator_overhead_excluded);
                EXPECT_TRUE(allocation.node_overhead_excluded);
            });
        return count;
    };
    EXPECT_EQ(hint_census(), 1u);
    auto arena_scope = arena->scope_id;
    CreditConstruction observer(*destination, &arena_scope,
        [](void* context, const std::shared_ptr<const void>&, const Budget*, std::size_t, bool, bool metadata,
            std::uint64_t scope)
        {
            EXPECT_TRUE(metadata);
            EXPECT_EQ(scope, *static_cast<const std::uint64_t*>(context));
            return true;
        });
    EXPECT_TRUE(observer.descend(std::shared_ptr<const AuthorityOwnerCredit>(leases.front()), destination.get(),
        leases.front()->metadata, true, leases, memory));
    // Exact owner admission in the live residency cancels its retirement hint.
    const auto hinted_bytes = destination->used;
    arena->cancel_retirement(owner);
    EXPECT_EQ(hint_census(), 0u);
    EXPECT_EQ(destination->used + ArenaCredits::hint_node_bytes, hinted_bytes);
    arena->reap();
    ASSERT_EQ(arena->entries.size(), 1u);

    auto hinted_owner = std::make_shared<const Payload>();
    {
        CreditConstruction credit(arena);
        credit.retain(hinted_owner, origin.get(), sizeof(Payload), true, leases, memory);
        credit.committed = true;
    }
    ASSERT_EQ(leases.size(), 2u);
    ASSERT_EQ(hint_census(), 1u);
    hinted_owner.reset();
    leases.pop_back();
    arena->reap();
    EXPECT_EQ(hint_census(), 0u);
    ASSERT_EQ(arena->entries.size(), 1u);

    policy.reset();
    owner.reset();
    leases.clear();
    arena->reap();
    EXPECT_TRUE(arena->entries.empty());
    EXPECT_EQ(arena->private_metadata_bytes(), header);
    EXPECT_EQ(destination->used, header + memory.bytes);
}
} // namespace
