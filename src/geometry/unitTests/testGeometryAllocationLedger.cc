/** @file testGeometryAllocationLedger.cc
 * @brief Unique immutable-payload admission, strong pins and cumulative cap checks.
 */
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/GeometryError.hh"
#include <gtest/gtest.h>

namespace
{
using SimpleFluid::ArithmeticMode;
namespace P = SimpleFluid::PLIC;
template<ArithmeticMode Mode> using Geometry = P::ConvexCellGeometry<Mode>;
constexpr std::size_t byte_limit = 64 * 1024 * 1024;

template<ArithmeticMode Mode> void reuse_contract()
{
    const auto root = Geometry<Mode>::box({0, 0, 0}, {1, 1, 1});
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    ledger.retain(root, used, byte_limit);
    const auto original = used, owners = ledger.owner_count(), payload = ledger.payload_bytes();
    EXPECT_EQ(payload, root->storage_bytes());
    EXPECT_GT(ledger.bookkeeping_bytes(), 0u);
    EXPECT_EQ(used, payload + ledger.bookkeeping_bytes());
    ledger.retain(root, used, byte_limit);
    EXPECT_EQ(used, original);
    EXPECT_EQ(ledger.owner_count(), owners);
    const auto alias = Geometry<Mode>::from_dynamic(root->dynamic_geometry());
    ledger.retain(alias, used, byte_limit);
    EXPECT_EQ(ledger.payload_bytes(), payload + sizeof(Geometry<Mode>));
    EXPECT_EQ(ledger.owner_count(), owners + 1);
    const auto equal_values = Geometry<Mode>::box({0, 0, 0}, {1, 1, 1});
    const auto before_distinct = ledger.payload_bytes();
    ledger.retain(equal_values, used, byte_limit);
    EXPECT_GT(ledger.payload_bytes(), before_distinct);
}
TEST(GeometryAllocationLedgerTest, RepeatedAndAliasedInputsChargeOnlyNewOwners)
{
    reuse_contract<ArithmeticMode::Ordinary>();
    if (P::certification_environment())
        reuse_contract<ArithmeticMode::BoundedCertification>();
}

template<ArithmeticMode Mode> void sibling_contract()
{
    const auto root = Geometry<Mode>::box({0, 0, 0}, {1, 1, 1});
    auto split = root->clip({{1, 0, 0}, {}, .5}, false);
    ASSERT_TRUE(split.inside);
    ASSERT_TRUE(split.outside);
    const auto separate = split.inside->storage_bytes() + split.outside->storage_bytes();
    std::weak_ptr<const Geometry<Mode>> discarded = split.inside;
    {
        P::GeometryAllocationLedger ledger;
        std::size_t used = 0;
        ledger.retain(split.inside, used, byte_limit);
        ledger.retain(split.outside, used, byte_limit);
        EXPECT_LT(ledger.payload_bytes(), separate);
        const auto charged = used;
        split.inside.reset();
        EXPECT_FALSE(discarded.expired());
        EXPECT_EQ(used, charged);
        ledger.retain(split.outside, used, byte_limit);
        EXPECT_EQ(used, charged);
        const auto fresh = root->clip({{1, 0, 0}, {}, .5}, false).inside;
        ASSERT_TRUE(fresh);
        const auto previous_payload = ledger.payload_bytes();
        ledger.retain(fresh, used, byte_limit);
        EXPECT_GT(used, charged);
        EXPECT_GT(ledger.payload_bytes(), previous_payload);
        EXPECT_FALSE(discarded.expired());
    }
    EXPECT_TRUE(discarded.expired());
}
TEST(GeometryAllocationLedgerTest, SharedSiblingsAreDeduplicatedAndDiscardedOwnersStayPinned)
{
    sibling_contract<ArithmeticMode::Ordinary>();
    if (P::certification_environment())
        sibling_contract<ArithmeticMode::BoundedCertification>();
}

TEST(GeometryAllocationLedgerTest, RejectedAndDecreasedBudgetsCannotResumeAdmission)
{
    const auto root = Geometry<ArithmeticMode::Ordinary>::box({0, 0, 0}, {1, 1, 1});
    P::GeometryAllocationLedger measured;
    std::size_t required = 0;
    measured.retain(root, required, byte_limit);
    ASSERT_GT(required, 1u);
    for (const auto cap : {std::size_t(1), required - 1})
    {
        P::GeometryAllocationLedger limited;
        std::size_t used = 0;
        EXPECT_THROW(limited.retain(root, used, cap), P::GeometryError);
        const auto charged = used;
        EXPECT_LE(charged, cap);
        EXPECT_THROW(limited.retain(root, used, byte_limit), P::GeometryError);
        EXPECT_EQ(used, charged);
    }
    auto decreased = required - 1;
    EXPECT_THROW(measured.retain(root, decreased, byte_limit), P::GeometryError);
    EXPECT_EQ(decreased, required - 1);
}
} // namespace
