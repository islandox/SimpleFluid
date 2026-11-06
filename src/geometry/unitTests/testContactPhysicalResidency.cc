#include "geometry/mesh/AMR/ContactPhysicalResidency.hh"
#include <gtest/gtest.h>

namespace
{
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
using Residency = D::ContactPhysicalResidency;
using Role = Residency::Role;
constexpr std::size_t ample = 1024 * 1024;
struct Owner
{
    D::ContactGroupAllocation charge;
    explicit Owner(std::shared_ptr<D::Budget> arena, std::size_t bytes) : charge(std::move(arena))
    {
        charge.add(bytes);
    }
};
} // namespace

TEST(ContactPhysicalResidency, SharedUseAndReintroductionPayOnePhysicalOverlap)
{
    const auto first_budget = std::make_shared<D::Budget>(D::Budget{ample});
    Residency first(first_budget);
    const auto owner = std::make_shared<Owner>(first_budget, 400);
    EXPECT_TRUE(first.retain(Role::TileBody, owner, 400, first_budget.get()).inserted());
    EXPECT_FALSE(first.retain(Role::TileBody, owner, 400, first_budget.get()).inserted());

    const auto budget = std::make_shared<D::Budget>(D::Budget{ample});
    Residency candidate(first, budget);
    EXPECT_EQ(budget->used, 400u + candidate.usage().storage_bytes());
    EXPECT_FALSE(candidate.release(Role::TileBody, owner).retired());
    EXPECT_TRUE(candidate.release(Role::TileBody, owner).retired());
    EXPECT_TRUE(candidate.admits(Role::TileBody, owner, first_budget.get(), 400));
    EXPECT_TRUE(candidate.retain(Role::TileBody, owner, 400, first_budget.get()).inserted());
    candidate.finish_overlap();
    EXPECT_EQ(budget->used, 400u + candidate.usage().storage_bytes());
    EXPECT_TRUE(candidate.release(Role::TileBody, owner).retired());
    EXPECT_EQ(budget->used, 0u);
    EXPECT_EQ(first.payload_bytes(), 400u);
}

TEST(ContactPhysicalResidency, PoolRecaptureTransfersOnlyLiveForeignAliases)
{
    const auto accepted_budget = std::make_shared<D::Budget>(D::Budget{ample});
    Residency accepted(accepted_budget);
    constexpr Residency::PointScope scope = 7;
    accepted.bind_funded_point_scope(scope);
    const auto old = std::make_shared<Owner>(accepted_budget, 400);
    accepted.retain(Role::PointBody, old, 400, accepted_budget.get(), nullptr, scope);

    const auto budget = std::make_shared<D::Budget>(D::Budget{ample});
    Residency candidate(accepted, budget);
    EXPECT_EQ(candidate.funded_point_scope(), scope);
    EXPECT_EQ(budget->used, candidate.usage().storage_bytes());
    budget->add(400); // Charge independently owned persistent pool pages.
    const auto local = std::make_shared<Owner>(budget, 200);
    candidate.retain(Role::PointBody, local, 200, budget.get(), nullptr, scope);
    const auto paid = candidate.detach_funded_point_scope(scope);
    ASSERT_EQ(paid, 400u);
    // The old pool disowns that transferred credit before it is retired.
    candidate.bind_funded_point_scope(scope + 1);
    candidate.finish_overlap();
    EXPECT_EQ(budget->used, 600u + candidate.usage().storage_bytes());
    candidate.release(Role::PointBody, old, scope);
    EXPECT_EQ(budget->used, 200u + candidate.usage().storage_bytes());
    candidate.release(Role::PointBody, local, scope);
    EXPECT_EQ(budget->used, 200u);
}

TEST(ContactPhysicalResidency, AggregateForkAdmitsExactLimitAndRejectsOneByteShort)
{
    const auto accepted_budget = std::make_shared<D::Budget>(D::Budget{ample});
    Residency accepted(accepted_budget);
    const auto first = std::make_shared<Owner>(accepted_budget, 311);
    const auto second = std::make_shared<Owner>(accepted_budget, 819);
    accepted.retain(Role::TileBody, first, 311, accepted_budget.get());
    accepted.retain(Role::SubfaceRing, second, 819, accepted_budget.get());
    const auto accepted_used = accepted_budget->used;
    const auto exact = accepted.payload_bytes() + accepted.usage().storage_bytes();
    const auto enough = std::make_shared<D::Budget>(D::Budget{exact});
    {
        Residency candidate(accepted, enough);
        EXPECT_EQ(enough->used, exact);
    }
    EXPECT_EQ(enough->used, 0u);
    const auto short_budget = std::make_shared<D::Budget>(D::Budget{exact - 1});
    EXPECT_THROW((void) Residency(accepted, short_budget), std::invalid_argument);
    EXPECT_EQ(short_budget->used, 0u);
    EXPECT_EQ(accepted_budget->used, accepted_used);
}

TEST(ContactPhysicalResidency, ActiveCreditRejectsSameAddressFromAnotherControl)
{
    const auto origin = std::make_shared<D::Budget>(D::Budget{ample});
    Residency source(origin);
    const auto owner = std::make_shared<Owner>(origin, 400);
    source.retain(Role::CreditMetadata, owner, 400, origin.get());
    const auto foreign_control = std::shared_ptr<const Owner>(owner.get(), [](const Owner*) {});
    ASSERT_EQ(owner.get(), foreign_control.get());
    ASSERT_TRUE(owner.owner_before(foreign_control) || foreign_control.owner_before(owner));

    const auto budget = std::make_shared<D::Budget>(D::Budget{ample});
    Residency candidate(source, budget);
    const auto before = budget->used;
    const auto spent = budget->cumulative;
    EXPECT_TRUE(candidate.admits(Role::CreditMetadata, owner, origin.get(), 400));
    EXPECT_TRUE(candidate.admits(Role::CreditMetadata, owner, 400));
    EXPECT_FALSE(candidate.admits(Role::CreditMetadata, foreign_control, origin.get(), 400));
    EXPECT_FALSE(candidate.admits(Role::CreditMetadata, foreign_control, 400));
    EXPECT_THROW(candidate.retain(Role::CreditMetadata, foreign_control, 400, origin.get()), std::invalid_argument);
    EXPECT_THROW(candidate.release(Role::CreditMetadata, foreign_control), std::invalid_argument);
    EXPECT_EQ(candidate.payload_bytes(), 400u);
    EXPECT_EQ(budget->used, before);
    EXPECT_EQ(budget->cumulative, spent);
    EXPECT_TRUE(candidate.release(Role::CreditMetadata, owner).retired());
    candidate.finish_overlap();
    EXPECT_EQ(budget->used, 0u);
}

TEST(ContactPhysicalResidency, FirstRetainRejectsMissingPointerOrControlBeforeCharging)
{
    constexpr std::size_t bytes = 400;
    const auto origin = std::make_shared<D::Budget>(D::Budget{ample});
    const auto budget = std::make_shared<D::Budget>(D::Budget{ample});
    Residency candidate(budget);
    const auto owner = std::make_shared<Owner>(origin, bytes);
    const std::shared_ptr<const Owner> empty;
    const std::shared_ptr<const Owner> without_control(empty, owner.get());
    const std::shared_ptr<const Owner> without_pointer(owner, nullptr);
    budget->add(bytes);
    auto prepaid = bytes;
    const auto before = budget->used;
    const auto spent = budget->cumulative;

    EXPECT_THROW(candidate.retain(Role::TileBody, without_control, bytes, origin.get()), std::invalid_argument);
    EXPECT_THROW(candidate.retain(Role::TileBody, without_pointer, bytes, origin.get()), std::invalid_argument);
    EXPECT_THROW(candidate.retain(Role::TileBody, without_control, bytes, origin.get(), &prepaid), std::invalid_argument);
    EXPECT_FALSE(candidate.retain(Role::TileBody, empty, 0, origin.get(), &prepaid).inserted());

    EXPECT_EQ(candidate.payload_bytes(), 0u);
    EXPECT_EQ(budget->used, before);
    EXPECT_EQ(budget->cumulative, spent);
    EXPECT_EQ(prepaid, bytes);
}

TEST(ContactPhysicalResidency, MissingControlPreservesExpiredRetiredCredit)
{
    constexpr std::size_t bytes = 400;
    const auto origin = std::make_shared<D::Budget>(D::Budget{ample});
    Residency source(origin);
    const Owner body(origin, bytes);
    // Keep the address valid while allowing the retired control to expire.
    auto owner = std::shared_ptr<const Owner>(&body, [](const Owner*) {});
    ASSERT_TRUE(source.retain(Role::TileBody, owner, bytes, origin.get()).inserted());
    const auto budget = std::make_shared<D::Budget>(D::Budget{ample});
    Residency candidate(source, budget);
    ASSERT_TRUE(candidate.release(Role::TileBody, owner).retired());
    ASSERT_TRUE(source.release(Role::TileBody, owner).retired());
    const std::weak_ptr<const Owner> former_owner = owner;
    owner.reset();
    ASSERT_TRUE(former_owner.expired());

    const std::shared_ptr<const Owner> without_control(std::shared_ptr<const Owner>{}, &body);
    const auto before = budget->used;
    const auto spent = budget->cumulative;
    const auto overlap = candidate.overlap_bytes();
    ASSERT_GE(overlap, bytes);
    EXPECT_THROW(candidate.retain(Role::TileBody, without_control, bytes, origin.get()), std::invalid_argument);
    EXPECT_EQ(candidate.payload_bytes(), 0u);
    EXPECT_EQ(candidate.overlap_bytes(), overlap);
    EXPECT_EQ(budget->used, before);
    EXPECT_EQ(budget->cumulative, spent);
}
