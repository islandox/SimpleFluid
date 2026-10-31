/** @file testOriginalSupportAuthority.cc
 * @brief Key-only authority semantics with sealed native ownership metadata.
 */
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include "geometry/PLIC/details/OriginalSupportAuthority.hh"
#include "geometry/PLIC/details/ordinary/NativeGeometrySupport.hh"
#include "geometry/PLIC/details/ordinary/NativeSupportAccess.hh"
#include "point_export_test_helpers.hh"
#include <algorithm>
#include <gtest/gtest.h>
#include <map>
#include <set>

namespace
{
using namespace SimpleFluid;
namespace P = SimpleFluid::PLIC;
using Access = P::details::OriginalSupportAuthorityAccess;
constexpr auto ordinary = ArithmeticMode::Ordinary;
constexpr auto exact = ArithmeticMode::Exact;
constexpr std::size_t budget = 64 * 1024 * 1024;
P::PlaneAuthority original_with_padding()
{
    const std::array<P::GeometryPoint, 3> points{{{0, 0, 1000}, {1, 0, 1000}, {0, 1, 1000}}};
    return Access::issue(P::planar_support<exact>(*P::PlanarFace<exact>::make(points)));
}
TEST(OriginalSupportAuthorityTest, MutatedWordsAndPaddingCannotMintOrAuthenticateOriginalSupport)
{
    const auto original = original_with_padding();
    ASSERT_NE(Access::get(original), nullptr);
    const auto words = original.words();
    EXPECT_EQ(
        static_cast<std::uint8_t>(words.front() >> 56), static_cast<std::uint8_t>(P::AuthorityKeyTag::OriginalSupport));
    EXPECT_EQ(words.front() & P::authority_key_length_mask, P::canonical_key_size(*Access::get(original)));
    ASSERT_NE(words.front() % 8, 0u);
    P::details::PlaneAuthorityRegistry registry{original};
    const auto restored = Access::authenticate(words, registry);
    EXPECT_EQ(Access::get(restored)->storage_identity(), Access::get(original)->storage_identity());
    auto altered = words;
    altered[1] ^= 1;
    EXPECT_THROW((Access::authenticate(altered, registry)), P::GeometryError);
    auto padding = words;
    padding.back() ^= std::uint64_t(1) << (8 * (padding.front() % 8));
    EXPECT_THROW((Access::authenticate(padding, registry)), P::GeometryError);
    EXPECT_THROW((P::PlaneAuthority::opaque(words)), P::GeometryError);
    auto length = words;
    length.front() ^= 1;
    EXPECT_THROW((Access::authenticate(length, registry)), P::GeometryError);
    EXPECT_THROW((Access::authenticate(words, {})), P::GeometryError);
    EXPECT_EQ(original.words(), words);
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    ledger.retain(original, used, budget);
    const auto charged = used;
    ledger.retain(restored, used, budget);
    EXPECT_EQ(used, charged);
}
TEST(OriginalSupportAuthorityTest, EqualKeysKeepDistinctOwnersAndOpaqueTokensRemainUnchanged)
{
    const auto first = original_with_padding();
    const auto copied = P::decode_identity<P::PlaneSupportIdentity<ordinary>>(P::encode_identity(*Access::get(first)));
    const auto second = Access::issue(copied);
    EXPECT_EQ(first, second);
    EXPECT_EQ(first <=> second, std::strong_ordering::equal);
    ASSERT_NE(Access::get(first)->storage_identity(), Access::get(second)->storage_identity());
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0, expected = 0;
    std::set<const void*> owners;
    const auto inspect = [&](const auto& support)
    {
        P::visit_identity_allocations(support,
            [&](const P::GeometryIdentityAllocation& allocation)
            {
                if (owners.insert(allocation.owner.get()).second)
                    expected += allocation.payload_bytes;
                return true;
            });
    };
    inspect(*Access::get(first));
    inspect(*Access::get(second));
    ledger.retain(*Access::get(first), used, budget);
    ledger.retain(*Access::get(second), used, budget);
    EXPECT_EQ(ledger.owner_count(), owners.size());
    EXPECT_EQ(ledger.payload_bytes(), expected);
    const auto opaque = P::PlaneAuthority::opaque({7, 11, 13});
    const auto key = opaque.words();
    EXPECT_NO_THROW(Access::validate_boundary(opaque));
    EXPECT_EQ(Access::get(opaque), nullptr);
    EXPECT_EQ(Access::retained(opaque), nullptr);
    EXPECT_EQ(opaque.words(), key);
    const auto short_opaque = P::PlaneAuthority::opaque({1, 0x53464e4158495331ULL});
    EXPECT_NO_THROW(Access::validate_boundary(short_opaque));
    EXPECT_EQ(Access::get(short_opaque), nullptr);
}

TEST(OriginalSupportAuthorityTest, LargeSourceWordsPreserveNumericOrderingAndAuthenticatedOwner)
{
    using IdentityAccess = P::details::GeometryIdentityAccess;
    std::vector<P::details::OrdinaryIdentityPoint> points;
    for (unsigned i = 0; i < 4096; ++i)
        points.push_back({static_cast<long double>(i), 2, 3});
    const auto first = Access::issue(IdentityAccess::support_source(points));
    points.back()[2] = 4;
    const auto second = Access::issue(IdentityAccess::support_source(points));
    const auto first_words = first.words(), second_words = second.words();
    EXPECT_EQ(first <=> second, first_words <=> second_words);
    EXPECT_EQ(first.compare_words(first_words), std::strong_ordering::equal);
    EXPECT_EQ(first.compare_words(second_words), first_words <=> second_words);
    const P::details::PlaneAuthorityRegistry registry{first, second};
    const auto authenticated = Access::authenticate(first_words, registry);
    EXPECT_EQ(Access::get(authenticated)->storage_identity(), Access::get(first)->storage_identity());
    auto corrupted = first_words;
    corrupted.back() ^= 1;
    EXPECT_THROW((Access::authenticate(corrupted, registry)), P::GeometryError);

    auto moved = first;
    auto owner = std::move(moved);
    EXPECT_TRUE(moved.empty());
    EXPECT_EQ(moved.word_count(), 0u);
    EXPECT_TRUE(moved.words().empty());
    EXPECT_EQ(owner, first);
    auto opaque = P::PlaneAuthority::opaque({7, 11});
    owner = std::move(opaque);
    EXPECT_TRUE(opaque.empty());
    EXPECT_EQ(owner.words(), (std::vector<std::uint64_t>{7, 11}));
}

template<ArithmeticMode Mode>
P::ConvexGeometryPtr<Mode> authority_fixture(const P::ConvexCellGeometry<ordinary>& native)
{
    using Cell = P::ConvexCellGeometry<Mode>;
    std::vector<P::Point<Mode>> points;
    for (const auto& point : P::test::exported_points(native))
        points.push_back(P::source_point<Mode>(native, P::details::GeometryIdentityAccess::point(point.value)));
    auto result = Cell::make_points(points, native.faces());
    if constexpr (Mode == ArithmeticMode::BoundedCertification)
    {
        // The bounded native-input bridge deliberately imports represented
        // coordinates without Ordinary incidence certificates. Certify the
        // geometry first, then add truthful source labels to this test-only
        // clone: the labels never establish its planarity or convexity.
        const auto before = P::details::checked_geometry_storage<Mode>(
            P::DynamicGeometryAccess::implementation(*result->dynamic_geometry()));
        auto copy = std::make_shared<P::details::GeometryStorage<Mode>>(*before);
        std::map<P::details::StoredPlanePtr<Mode>, P::details::StoredPlanePtr<Mode>> replacements;
        for (std::size_t f = 0; f < copy->surfaces.size(); ++f)
        {
            const auto original = copy->surfaces[f].plane;
            auto plane = std::make_shared<P::details::StoredPlane<Mode>>(*original);
            plane->authorities.push_back(Access::issue(P::planar_support<ordinary>(*native.face_geometry(f))));
            replacements.emplace(original, plane);
            copy->surfaces[f].plane = std::move(plane);
        }
        for (auto& vertex : copy->vertices)
        {
            auto authorities = vertex.authorities.values();
            for (auto& plane : vertex.incident)
            {
                plane = replacements.at(plane);
                authorities.insert(authorities.end(), plane->authorities.begin(), plane->authorities.end());
            }
            std::sort(authorities.begin(), authorities.end());
            authorities.erase(std::unique(authorities.begin(), authorities.end()), authorities.end());
            vertex.authorities = std::move(authorities);
        }
        result = Cell::from_dynamic(P::DynamicGeometryAccess::geometry(std::move(copy)));
    }
    return result;
}
template<ArithmeticMode Mode> void authenticated_rebind()
{
    using Cell = P::ConvexCellGeometry<Mode>;
    const auto native = P::ConvexCellGeometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    const auto source = authority_fixture<Mode>(*native);
    const auto before = P::details::checked_geometry_storage<Mode>(
        P::DynamicGeometryAccess::implementation(*source->dynamic_geometry()));
    ASSERT_FALSE(before->surfaces.front().plane->authorities.empty());
    const auto& authority = before->surfaces.front().plane->authorities.front();
    ASSERT_NE(Access::get(authority), nullptr);
    const P::GeometryPacketContext context{};
    const auto bytes = source->encode(context);
    const std::array sources{source};
    const auto restored = Cell::decode(bytes, context, sources);
    const auto after = P::details::checked_geometry_storage<Mode>(
        P::DynamicGeometryAccess::implementation(*restored->dynamic_geometry()));
    const auto& rebound = after->surfaces.front().plane->authorities.front();
    ASSERT_NE(Access::get(rebound), nullptr);
    EXPECT_EQ(authority, rebound);
    EXPECT_EQ(Access::get(authority)->storage_identity(), Access::get(rebound)->storage_identity());
    P::GeometryAllocationLedger alone, preloaded;
    std::size_t one = 0, two = 0;
    alone.retain(restored, one, budget);
    preloaded.retain(*Access::get(rebound), two, budget);
    preloaded.retain(restored, two, budget);
    EXPECT_EQ(preloaded.owner_count(), alone.owner_count());
    EXPECT_EQ(preloaded.payload_bytes(), alone.payload_bytes());
}
TEST(OriginalSupportAuthorityTest, AuthenticatedImportsRebindTypedCarriersForBothFiniteModes)
{
    authenticated_rebind<ordinary>();
    if (P::certification_environment())
        authenticated_rebind<ArithmeticMode::BoundedCertification>();
}
TEST(OriginalSupportAuthorityTest, CompactNativeAncestryFitsFormerCacheLimitedAllowanceAndPreservesInput)
{
    const std::array<P::GeometryPoint, 3> points{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}};
    auto face = P::PlanarFace<ordinary>::make(points);
    for (unsigned i = 0; i < 20; ++i)
        face = face->translated(P::GeometryPoint{1, 0, 0});
    const auto support = P::planar_support<ordinary>(*face);
    const auto frozen = P::canonical_key(support), bytes = P::encode_identity(support);
    const auto limited = bytes.size() * 3;
    ASSERT_LT(support.payload_bytes(), limited);
    auto pin = support.storage_owner();
    const auto references = pin.use_count();
    const auto compact = P::decode_identity<P::PlaneSupportIdentity<ordinary>>(bytes, limited);
    EXPECT_EQ(compact, support);
    EXPECT_EQ(pin.use_count(), references);
    EXPECT_EQ(P::canonical_key(support), frozen);
    const auto restored = P::decode_identity<P::PlaneSupportIdentity<ordinary>>(bytes, budget);
    EXPECT_EQ(restored, support);
    unsigned owners = 0;
    P::visit_identity_allocations(restored,
        [&](const P::GeometryIdentityAllocation&)
        {
            ++owners;
            return true;
        });
    EXPECT_EQ(owners, 21u);
}

TEST(OriginalSupportAuthorityTest, EnclosingCellSharesOneNativeBudgetAcrossDeepSupportFields)
{
    const auto base = P::ConvexCellGeometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    std::vector<P::PlaneSupportIdentity<ordinary>> supports;
    for (std::size_t f = 0; f < base->num_faces(); ++f)
    {
        const auto face = base->face_geometry(f);
        auto support = P::planar_support<ordinary>(*face);
        const auto tangent = P::details::ordinary_native::sub(face->points()[1], face->points()[0]);
        for (unsigned depth = 0; depth < 20; ++depth)
            support = P::details::GeometryIdentityAccess::translated_support(support, tangent);
        supports.push_back(std::move(support));
    }
    const auto decorated = P::details::NativeSupportAccess::cell_supports(*base, supports);
    const auto frozen = P::canonical_cell_key(*decorated), bytes = decorated->encode_certificate();
    constexpr std::size_t limited = 32 * 1024;
    ASSERT_LT(bytes.size(), limited);
    const auto independent =
        P::decode_identity<P::PlaneSupportIdentity<ordinary>>(P::encode_identity(supports.front()), limited);
    EXPECT_EQ(independent, supports.front()); // One field fits; the record's fields do not.
    auto pin = supports.front().storage_owner();
    const auto references = pin.use_count();
    try
    {
        (void) P::ConvexCellGeometry<ordinary>::decode_certificate(bytes, limited);
        FAIL() << "Independent per-field budgets accepted an oversized native record.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("Native support import"), std::string::npos);
    }
    EXPECT_EQ(pin.use_count(), references);
    EXPECT_EQ(P::canonical_cell_key(*decorated), frozen);
    const auto restored = P::ConvexCellGeometry<ordinary>::decode_certificate(bytes, budget);
    EXPECT_EQ(P::canonical_cell_key(*restored), frozen);
}

template<ArithmeticMode Mode> void operation_rebind_budget()
{
    using Cell = P::ConvexCellGeometry<Mode>;
    const auto base = P::ConvexCellGeometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    std::vector<P::PlaneSupportIdentity<ordinary>> supports;
    for (std::size_t f = 0; f < base->num_faces(); ++f)
    {
        const auto face = base->face_geometry(f);
        auto support = P::planar_support<ordinary>(*face);
        if (f == 0)
        {
            const auto tangent = P::details::ordinary_native::sub(face->points()[1], face->points()[0]);
            for (unsigned depth = 0; depth < 80; ++depth)
                support = P::details::GeometryIdentityAccess::translated_support(support, tangent);
        }
        supports.push_back(std::move(support));
    }
    const auto native = P::details::NativeSupportAccess::cell_supports(*base, supports);
    const auto source = authority_fixture<Mode>(*native);
    const auto stored = P::details::checked_geometry_storage<Mode>(
        P::DynamicGeometryAccess::implementation(*source->dynamic_geometry()));
    ASSERT_FALSE(stored->surfaces.front().plane->authorities.empty());
    const auto* carrier = Access::get(stored->surfaces.front().plane->authorities.front());
    ASSERT_NE(carrier, nullptr);
    EXPECT_EQ(P::canonical_key(*carrier), P::canonical_key(supports.front()));
    const P::GeometryPacketContext context{};
    const auto bytes = source->encode(context);
    const std::array sources{source};
    auto pin = supports.front().storage_owner();
    const auto references = pin.use_count();
    P::GeometryLimits limits;
    limits.arithmetic_mode = Mode;
    limits.byte_budget = budget;
    std::size_t required = 0;
    {
        P::details::Operation operation(limits);
        const auto restored = Cell::decode(bytes, context, sources, limits);
        required = operation.work.bytes;
        // Every authenticated input support must already have entered the
        // same operation ledger, including every separately shared ancestor.
        for (const auto& support : supports)
            operation.work.allocations.retain(support, operation.work.bytes, limits.byte_budget);
        EXPECT_EQ(operation.work.bytes, required);
        EXPECT_EQ(restored->volume().value, source->volume().value);
    }
    EXPECT_EQ(pin.use_count(), references);
    ASSERT_GT(required, bytes.size());
    limits.byte_budget = required;
    EXPECT_NO_THROW((Cell::decode(bytes, context, sources, limits)));
    limits.byte_budget = required - 1;
    try
    {
        (void) Cell::decode(bytes, context, sources, limits);
        FAIL() << "Authenticated native carrier bypassed the operation budget.";
    }
    catch (const P::GeometryError& error)
    {
        EXPECT_EQ(error.code(), P::GeometryErrorCode::ResourceLimit);
    }
    EXPECT_EQ(pin.use_count(), references);
    EXPECT_NO_THROW((Cell::decode(bytes, context, sources)));

}
TEST(OriginalSupportAuthorityTest, AuthenticatedCarrierImportsChargeFullAncestryInsideEachFiniteOperation)
{
    operation_rebind_budget<ordinary>();
    if (P::certification_environment())
        operation_rebind_budget<ArithmeticMode::BoundedCertification>();
}
TEST(OriginalSupportAuthorityTest, ReconstructionIsPreflightedAgainstTheSameRemainingRecordBudget)
{
    using Cell = P::ConvexCellGeometry<ordinary>;
    const auto cell = Cell::box({0, 0, 0}, {1, 1, 1});
    const auto bytes = cell->encode_certificate();
    // Input ownership: eight points/provenance records, six face/support
    // slots, and twenty-four indices. The output graph needs separate arrays.
    std::size_t fields = sizeof(Cell) + sizeof(P::DynamicConvexGeometry) +
                         8 * (sizeof(P::Point<ordinary>) + sizeof(P::DynamicPointProvenance)) +
                         6 * (sizeof(std::vector<std::size_t>) + sizeof(P::PlaneSupportIdentity<ordinary>)) +
                         24 * sizeof(std::size_t);
    P::GeometryAllocationLedger parsed;
    for (std::size_t f = 0; f < 6; ++f)
    {
        const auto support = P::decode_identity<P::PlaneSupportIdentity<ordinary>>(
            P::encode_identity(P::planar_support<ordinary>(*cell->face_geometry(f))));
        parsed.retain(support, fields, budget);
    }
    const auto minimum_output =
        sizeof(P::details::GeometryStorage<ordinary>) +
        8 * (sizeof(P::details::StoredVertex<ordinary>) + sizeof(P::DynamicPoint)) +
        6 * (sizeof(P::details::StoredPlane<ordinary>) + sizeof(P::details::StoredSurface<ordinary>) +
                sizeof(std::vector<std::size_t>)) +
        24 * sizeof(std::size_t);
    const auto limited = fields + minimum_output / 2;
    ASSERT_GT(limited, fields);
    ASSERT_LT(bytes.size(), limited);
    const auto frozen = P::canonical_cell_key(*cell);
    try
    {
        (void) Cell::decode_certificate(bytes, limited);
        FAIL() << "Reconstruction ignored the remaining native record budget.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("cell reconstruction"), std::string::npos);
    }
    EXPECT_EQ(P::canonical_cell_key(*cell), frozen);
    EXPECT_NO_THROW((Cell::decode_certificate(bytes, budget)));
}

TEST(OriginalSupportAuthorityTest, NestedRecordBudgetsRestoreOuterScopeAfterSuccessAndFailure)
{
    using Cell = P::ConvexCellGeometry<ordinary>;
    const auto original = Cell::box({0, 0, 0}, {1, 1, 1});
    const auto frozen = P::canonical_cell_key(*original);
    const auto bytes = original->encode_certificate();
    const auto unchanged = bytes;
    auto* const previous = P::details::active_work;
    P::ConvexGeometryPtr<ordinary> success;
    constexpr std::size_t outer_budget = 512 * 1024;
    {
        P::GeometryLimits limits;
        limits.arithmetic_mode = ordinary;
        limits.byte_budget = outer_budget;
        P::details::Operation outer(limits);
        auto* const active = P::details::active_work;
        ASSERT_NE(active, nullptr);
        const auto limit = active->limits.byte_budget;
        EXPECT_EQ(limit, outer_budget);
        active->charge(64, 1);
        const auto before = active->bytes;
        success = Cell::decode_certificate(bytes, 128 * 1024);
        ASSERT_TRUE(success);
        EXPECT_EQ(P::canonical_cell_key(*success), frozen);
        EXPECT_EQ(P::details::active_work, active);
        EXPECT_EQ(active->limits.byte_budget, limit);
        EXPECT_GT(active->bytes, before);
        EXPECT_LE(active->bytes, limit);
    }
    EXPECT_EQ(P::details::active_work, previous);
    ASSERT_TRUE(success);
    EXPECT_EQ(P::canonical_cell_key(*success), frozen);
    {
        // Admission failure may poison the operation's owner ledger, so this
        // failure path deliberately uses a fresh outer operation.
        P::GeometryLimits limits;
        limits.arithmetic_mode = ordinary;
        limits.byte_budget = outer_budget;
        P::details::Operation outer(limits);
        auto* const active = P::details::active_work;
        ASSERT_NE(active, nullptr);
        const auto limit = active->limits.byte_budget;
        active->charge(64, 1);
        const auto before = active->bytes;
        EXPECT_THROW((Cell::decode_certificate(bytes, bytes.size())), std::invalid_argument);
        EXPECT_EQ(P::details::active_work, active);
        EXPECT_EQ(active->limits.byte_budget, limit);
        EXPECT_GE(active->bytes, before);
        EXPECT_LE(active->bytes, limit);
        EXPECT_EQ(bytes, unchanged);
    }
    EXPECT_EQ(P::details::active_work, previous);
    EXPECT_EQ(bytes, unchanged);
    EXPECT_EQ(P::canonical_cell_key(*success), frozen);
}
} // namespace
