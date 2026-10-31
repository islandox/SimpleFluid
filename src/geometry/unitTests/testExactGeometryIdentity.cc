/** @file testExactGeometryIdentity.cc
 * @brief Independent Exact native value, limb storage and codec admission checks.
 */
#include "geometry/PLIC/GeometryCodecTags.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/GeometryPacketCodec.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/GeometryIdentityBacking.hh"
#include "geometry/PLIC/details/exact/GeometryIdentity.hh"
#include <cmath>
#include <compare>
#include <gtest/gtest.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::PLIC;
namespace native = SimpleFluid::PLIC::exact_geometry_detail;
constexpr auto exact = ArithmeticMode::Exact;
using ExactPoint = PointIdentity<exact>;
using Measure = GeometryMeasure<exact>;
template<class Owner>
concept RetainsCanonicalKey = requires(Owner value) { value.key; };
static_assert(!RetainsCanonicalKey<details::GeometryIdentityBacking>);

struct PointPacket
{
    std::vector<char> body, wire;
};
PointPacket packet(const native::P& coordinates)
{
    native::WireWriter writer;
    writer.tag(GeometryTag::ExactPoint);
    for (const auto& coordinate : coordinates)
        writer.rational(coordinate);
    PointPacket result;
    result.body = std::move(writer.data);
    result.wire = encode_geometry_packet(GeometryPacketKind::PointIdentity, exact, result.body);
    return result;
}

template<class Identity> void expect_equal_roundtrip_order(const Identity& identity)
{
    const auto decoded = decode_identity<Identity>(encode_identity(identity));
    if (identity.storage_identity())
        EXPECT_NE(identity.storage_identity(), decoded.storage_identity());
    EXPECT_EQ(identity, decoded);
    EXPECT_EQ(identity <=> decoded, std::strong_ordering::equal);
    EXPECT_EQ(decoded <=> identity, std::strong_ordering::equal);
}
template<class Identity> void expect_native_less(const Identity& first, const Identity& second)
{
    EXPECT_LT(first, second);
    EXPECT_GT(second, first);
    EXPECT_NE(first, second);
    EXPECT_EQ((first <=> second) == std::strong_ordering::equal, first == second);
    expect_equal_roundtrip_order(first);
    expect_equal_roundtrip_order(second);
}

TEST(ExactGeometryIdentityTest, LargeRationalTuplesRetainDynamicLimbsAcrossCodecAndCopies)
{
    const native::I numerator = (native::I{1} << 257) + 1;
    const native::I denominator = (native::I{1} << 263) + 3;
    const auto supplied = packet({native::Q(numerator, denominator), native::Q{}, native::Q{}});
    const auto zero_packet = packet(native::P{});
    const auto point = decode_identity<ExactPoint>(supplied.wire);
    const auto zero = decode_identity<ExactPoint>(zero_packet.wire);

    // Public exact additions preserve the low integer terms independently of
    // the private bit-shift rational used to construct the incoming packet.
    const auto expected_numerator = Measure::from_value(std::ldexp(1., 257)) + Measure::from_value(1);
    const auto expected_denominator = Measure::from_value(std::ldexp(1., 263)) + Measure::from_value(3);
    const auto expected = expected_numerator.divided_by(expected_denominator);
    EXPECT_EQ(point_projection<exact>(point, {1, 0, 0}, {}).compare(expected), 0);
    EXPECT_EQ(canonical_key(point), supplied.body);
    EXPECT_EQ(encode_identity(point), supplied.wire);
    const auto roundtrip = decode_identity<ExactPoint>(encode_identity(point));
    EXPECT_EQ(canonical_key(roundtrip), supplied.body);
    EXPECT_EQ(point_projection<exact>(roundtrip, {1, 0, 0}, {}).compare(expected), 0);

    const auto& coordinates = native::authority_point(point);
    const auto& zero_coordinates = native::authority_point(zero);
    const native::Q inline_value;
    ASSERT_GT(coordinates[0].backend().num().capacity(), inline_value.backend().num().capacity());
    ASSERT_GT(coordinates[0].backend().denom().capacity(), inline_value.backend().denom().capacity());
    for (const auto& coordinate : zero_coordinates)
    {
        ASSERT_EQ(coordinate.backend().num().capacity(), inline_value.backend().num().capacity());
        ASSERT_EQ(coordinate.backend().denom().capacity(), inline_value.backend().denom().capacity());
    }
    for (unsigned axis = 1; axis < 3; ++axis)
    {
        ASSERT_EQ(coordinates[axis].backend().num().capacity(), inline_value.backend().num().capacity());
        ASSERT_EQ(coordinates[axis].backend().denom().capacity(), inline_value.backend().denom().capacity());
    }
    const auto& owner = PLIC::details::GeometryIdentityAccess::backing(point);
    const auto& zero_owner = PLIC::details::GeometryIdentityAccess::backing(zero);
    ASSERT_TRUE(owner);
    ASSERT_TRUE(zero_owner);
    const auto dynamic_bytes =
        (coordinates[0].backend().num().capacity() + coordinates[0].backend().denom().capacity()) *
        sizeof(boost::multiprecision::limb_type);
    // The control has the same root and native tuple sizes. Only the two
    // heap-backed integer buffers grow; serialized key bytes are not retained.
    ASSERT_GE(point.payload_bytes(), zero.payload_bytes());
    EXPECT_EQ(point.payload_bytes() - zero.payload_bytes(), dynamic_bytes);

    const auto copy = point;
    EXPECT_EQ(copy.storage_identity(), point.storage_identity());
    EXPECT_EQ(copy.storage_owner(), point.storage_owner());
    EXPECT_EQ(&native::authority_point(copy), &coordinates);
    const auto original_key = canonical_key(point);
    const auto original_wire = supplied.wire;
    const auto cap = supplied.wire.size();
    ASSERT_LT(cap, point.payload_bytes());
    EXPECT_THROW((void) decode_identity<ExactPoint>(supplied.wire, cap), std::invalid_argument);
    EXPECT_EQ(supplied.wire, original_wire);
    EXPECT_EQ(canonical_key(point), original_key);
    EXPECT_EQ(canonical_key(copy), original_key);
    EXPECT_EQ(encode_identity(point), original_wire);
    EXPECT_EQ(point_projection<exact>(point, {1, 0, 0}, {}).compare(expected), 0);
}

TEST(ExactGeometryIdentityTest, NativePointAndDisplacementOrderingUsesNumericTupleValues)
{
    const native::Q quarter(1, 4), third(1, 3);
    const auto point_quarter = native::make_point_identity({quarter, native::Q{9}, native::Q{}});
    const auto point_third = native::make_point_identity({third, native::Q{-9}, native::Q{}});
    // The first rational coordinate wins even when later coordinates reverse.
    expect_native_less(point_quarter, point_third);
    const auto positive_quarter = native::make_displacement_identity({quarter, native::Q{}, native::Q{}});
    const auto positive_third = native::make_displacement_identity({third, native::Q{}, native::Q{}});
    const auto negative = native::make_displacement_identity({-quarter, native::Q{}, native::Q{}});
    const DisplacementIdentity<exact> zero;
    expect_native_less(positive_quarter, positive_third);
    expect_native_less(negative, zero);
    expect_native_less(zero, positive_quarter);
}

TEST(ExactGeometryIdentityTest, NativeLineOrderingComparesDirectionBeforeOrthogonalMoment)
{
    const native::Q quarter(1, 4), third(1, 3);
    const native::P direction_quarter{native::Q{1}, quarter, native::Q{}};
    const native::P direction_third{native::Q{1}, third, native::Q{}};
    // Each normalized direction starts at one. A z-only moment is orthogonal
    // to both xy directions, so every tuple is a valid canonical line form.
    const auto first = native::make_line_identity(direction_quarter, {native::Q{}, native::Q{}, native::Q{9}});
    const auto second = native::make_line_identity(direction_third, {native::Q{}, native::Q{}, native::Q{-9}});
    expect_native_less(first, second);
    const auto moment_quarter = native::make_line_identity(direction_quarter, {native::Q{}, native::Q{}, quarter});
    const auto moment_third = native::make_line_identity(direction_quarter, {native::Q{}, native::Q{}, third});
    expect_native_less(moment_quarter, moment_third);
}

TEST(ExactGeometryIdentityTest, NativeSupportOrderingComparesNormalBeforeOffset)
{
    const native::Q quarter(1, 4), third(1, 3);
    const auto first = native::make_support_identity({native::Q{1}, quarter, native::Q{}, native::Q{9}});
    const auto second = native::make_support_identity({native::Q{1}, third, native::Q{}, native::Q{-9}});
    expect_native_less(first, second);
    const auto offset_quarter = native::make_support_identity({native::Q{1}, native::Q{}, native::Q{}, quarter});
    const auto offset_third = native::make_support_identity({native::Q{1}, native::Q{}, native::Q{}, third});
    expect_native_less(offset_quarter, offset_third);
}
} // namespace
