/** @file testGeometryAllocationLedger.cc
 * @brief Unique immutable-payload admission, strong pins and cumulative cap checks.
 */
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/ordinary/NativeSupportAccess.hh"
#include <gtest/gtest.h>
#include <optional>

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

TEST(GeometryAllocationLedgerTest, SharedSupportOwnersAreCountedAcrossDifferentStoredPlanes)
{
    constexpr auto ordinary = ArithmeticMode::Ordinary;
    const auto root = Geometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    std::vector<P::PlaneSupportIdentity<ordinary>> supports;
    std::size_t support_bytes = 0;
    for (std::size_t face = 0; face < root->num_faces(); ++face)
    {
        supports.push_back(P::planar_support<ordinary>(*root->face_geometry(face)));
        support_bytes += supports.back().payload_bytes();
    }
    const auto clone = P::details::NativeSupportAccess::cell_supports(*root, supports);
    for (std::size_t face = 0; face < root->num_faces(); ++face)
        EXPECT_EQ(P::planar_support<ordinary>(*clone->face_geometry(face)).storage_identity(),
            supports[face].storage_identity());
    P::GeometryAllocationLedger geometry_only;
    std::size_t geometry_used = 0;
    geometry_only.retain(root, geometry_used, byte_limit);
    P::GeometryAllocationLedger shared;
    std::size_t shared_used = 0;
    shared.retain(supports.front(), shared_used, byte_limit);
    shared.retain(root, shared_used, byte_limit);
    EXPECT_EQ(shared.payload_bytes(), geometry_only.payload_bytes());
    EXPECT_EQ(shared.owner_count(), geometry_only.owner_count());
    const auto independent =
        P::decode_identity<P::PlaneSupportIdentity<ordinary>>(P::encode_identity(supports.front()));
    ASSERT_EQ(independent, supports.front());
    ASSERT_NE(independent.storage_identity(), supports.front().storage_identity());
    P::GeometryAllocationLedger distinct;
    std::size_t distinct_used = 0;
    distinct.retain(independent, distinct_used, byte_limit);
    distinct.retain(root, distinct_used, byte_limit);
    EXPECT_EQ(distinct.payload_bytes(), geometry_only.payload_bytes() + independent.payload_bytes());
    EXPECT_EQ(distinct.owner_count(), geometry_only.owner_count() + 1);
    shared.retain(clone, shared_used, byte_limit);
    EXPECT_LE(shared.payload_bytes(), root->storage_bytes() + clone->storage_bytes() - support_bytes);
}
TEST(GeometryAllocationLedgerTest, IdentityPinsAndBudgetPoisoningHaveTheSameAdmissionContract)
{
    constexpr auto exact = ArithmeticMode::Exact;
    const std::array<P::GeometryPoint, 3> points{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}};
    std::optional<P::PointIdentity<exact>> identity;
    std::weak_ptr<const void> weak;
    {
        const auto face = P::PlanarFace<exact>::make(points);
        identity = face->vertex_identities().front();
        weak = identity->storage_owner();
    }
    {
        P::GeometryAllocationLedger ledger;
        std::size_t used = 0;
        ledger.retain(*identity, used, byte_limit);
        EXPECT_EQ(ledger.owner_count(), 1u);
        EXPECT_EQ(ledger.payload_bytes(), identity->payload_bytes());
        EXPECT_EQ(used, ledger.payload_bytes() + ledger.bookkeeping_bytes());
        const auto required = used;
        ledger.retain(*identity, used, byte_limit);
        EXPECT_EQ(used, required);
        for (const auto cap : {std::size_t(1), required - 1})
        {
            P::GeometryAllocationLedger limited;
            std::size_t admitted = 0;
            EXPECT_THROW(limited.retain(*identity, admitted, cap), P::GeometryError);
            EXPECT_LE(admitted, cap);
            const auto charged = admitted;
            EXPECT_THROW(limited.retain(*identity, admitted, byte_limit), P::GeometryError);
            EXPECT_EQ(admitted, charged);
        }
        auto decreased = required - 1;
        EXPECT_THROW(ledger.retain(*identity, decreased, byte_limit), P::GeometryError);
        EXPECT_EQ(decreased, required - 1);
        identity.reset();
        EXPECT_FALSE(weak.expired());
    }
    EXPECT_TRUE(weak.expired());
}
TEST(GeometryAllocationLedgerTest, IdentityModesDoNotSelectNumericalGeometryModeAndZeroNeedsNoOwner)
{
    constexpr auto ordinary = ArithmeticMode::Ordinary;
    constexpr auto exact = ArithmeticMode::Exact;
    P::GeometryAllocationLedger empty;
    std::size_t zero_used = 0;
    empty.retain(P::DisplacementIdentity<exact>{}, zero_used, 0);
    EXPECT_EQ(zero_used, 0u);
    EXPECT_EQ(empty.owner_count(), 0u);
    EXPECT_EQ(empty.bookkeeping_bytes(), 0u);
    const auto root = Geometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    const auto support = P::planar_support<ordinary>(*root->face_geometry(0));
    const std::array<P::GeometryPoint, 3> points{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}};
    const auto point = P::PlanarFace<exact>::make(points)->vertex_identities().front();
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    ledger.retain(support, used, byte_limit);
    ledger.retain(point, used, byte_limit);
    EXPECT_NO_THROW(ledger.retain(root, used, byte_limit));
    EXPECT_EQ(ledger.payload_bytes(), root->storage_bytes() + point.payload_bytes());
}

TEST(GeometryAllocationLedgerTest, DynamicGeometryUsesTheSameOwnersAndNumericalModeContract)
{
    constexpr auto ordinary = ArithmeticMode::Ordinary;
    auto root = Geometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    const auto total_payload = root->storage_bytes();
    std::weak_ptr<const P::DynamicConvexGeometry> weak = root->dynamic_geometry();
    {
        P::GeometryAllocationLedger ledger;
        std::size_t used = 0;
        ledger.retain<ordinary>(root->dynamic_geometry(), used, byte_limit);
        EXPECT_EQ(ledger.payload_bytes(), total_payload - sizeof(Geometry<ordinary>));
        const auto dynamic_owners = ledger.owner_count();
        ledger.retain(root, used, byte_limit);
        EXPECT_EQ(ledger.payload_bytes(), total_payload);
        EXPECT_EQ(ledger.owner_count(), dynamic_owners + 1);
        root.reset();
        EXPECT_FALSE(weak.expired());
        if (P::certification_environment())
        {
            const auto bounded = Geometry<ArithmeticMode::BoundedCertification>::box({0, 0, 0}, {1, 1, 1});
            const auto charged = used;
            EXPECT_THROW(
                ledger.retain<ArithmeticMode::BoundedCertification>(bounded->dynamic_geometry(), used, byte_limit),
                std::invalid_argument);
            EXPECT_EQ(used, charged);
            EXPECT_THROW(ledger.retain<ordinary>(weak.lock(), used, byte_limit), P::GeometryError);
        }
    }
    EXPECT_TRUE(weak.expired());
}
} // namespace
