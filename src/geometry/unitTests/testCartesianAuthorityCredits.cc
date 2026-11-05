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
} // namespace
