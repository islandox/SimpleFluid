/** @file testNativeSupportOrdering.cc
 * @brief Native field/provenance ordering keeps support identities distinct.
 */
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include <gtest/gtest.h>

namespace
{
using namespace SimpleFluid;
namespace P = SimpleFluid::PLIC;
using Access = P::details::GeometryIdentityAccess;
constexpr auto ordinary = ArithmeticMode::Ordinary;
constexpr auto exact = ArithmeticMode::Exact;
using Support = P::PlaneSupportIdentity<ordinary>;
std::vector<P::details::OrdinarySupportRecipe> recipes()
{
    return {{true, {0, 0, 0}, {}}, {true, {1, 2, 0}, {}}, {true, {3, -1, 0}, {}}, {false, {}, {0, 1, 2}}};
}
Support retained_exact(double x)
{
    const std::array<P::GeometryPoint, 3> points{{{x, 0, 0}, {x, 1, 0}, {x, 0, 1}}};
    return Access::retained_exact_support(P::planar_support<exact>(*P::PlanarFace<exact>::make(points)));
}
TEST(NativeSupportOrderingTest, VariantRanksAreStableAndCoincidentExpressionsStayDistinct)
{
    const auto axis = Access::support_axis(0, 0);
    const std::vector<Support> values{axis, Access::support_source({{0, 0, 0}, {0, 1, 0}, {1, 0, 1}}),
        Access::support_input({1, 1, 0}, {}, 0), Access::support_plane(0, {1, 0, 0}, 0),
        Access::translated_support(axis, {0, 1, 0}), retained_exact(0), Access::support_dag(recipes(), 0, 3)};
    for (std::size_t i = 0; i < values.size(); ++i)
        for (std::size_t j = 0; j < values.size(); ++j)
        {
            EXPECT_EQ(values[i] < values[j], i < j);
            EXPECT_EQ(values[i] == values[j], i == j);
        }
    EXPECT_NE(values[0], values[3]); // Axis and normalized-plane tokens are different expressions.
    EXPECT_NE(values[0], values[4]); // Tangential translation retains its original ancestry.
    EXPECT_NE(values[0], values[5]); // Retained Exact authority remains a different variant.
}
TEST(NativeSupportOrderingTest, FixedAndVariableFieldsUseNumericLexicographicOrder)
{
    EXPECT_LT(Access::support_axis(0, -1), Access::support_axis(0, 1));
    EXPECT_LT(Access::support_axis(0, 1000), Access::support_axis(1, -1000));
    EXPECT_LT(Access::support_source({{0, 0, 0}, {0, 1, 0}, {1, 0, 1}}),
        Access::support_source({{0, 0, 0}, {0, 2, 0}, {1, 0, 1}}));
    EXPECT_LT(Access::support_input({1, 1, 0}, {100, 0, 0}, 100), Access::support_input({2, 1, 0}, {-100, 0, 0}, -100));
    EXPECT_LT(Access::support_input({1, 1, 0}, {0, 1, 0}, 100), Access::support_input({1, 1, 0}, {0, 2, 0}, -100));
    EXPECT_LT(Access::support_input({1, 1, 0}, {}, -1), Access::support_input({1, 1, 0}, {}, 1));
    EXPECT_LT(Access::support_plane(1, {.5L, 1, 0}, 100), Access::support_plane(1, {.75L, 1, 0}, -100));
    EXPECT_LT(retained_exact(-.5), retained_exact(.5));
}
TEST(NativeSupportOrderingTest, TranslationComparesParentsBeforeShiftWithoutFlattening)
{
    const auto first = Access::support_axis(0, 0), second = Access::support_axis(0, 1);
    EXPECT_LT(Access::translated_support(first, {100, 0, 0}), Access::translated_support(second, {-100, 0, 0}));
    EXPECT_LT(Access::translated_support(first, {-2, 0, 0}), Access::translated_support(first, {-1, 0, 0}));
    const auto nested = Access::translated_support(Access::translated_support(first, {1, 0, 0}), {1, 0, 0});
    const auto direct = Access::translated_support(first, {2, 0, 0});
    EXPECT_NE(nested, direct);
    EXPECT_GT(nested, direct); // Its parent is a Translation, whose rank follows Axis.
    const auto restored = P::decode_identity<Support>(P::encode_identity(nested));
    EXPECT_EQ(restored, nested);
    EXPECT_EQ(restored <=> nested, std::strong_ordering::equal);
    EXPECT_EQ(P::canonical_key_hash(restored), P::canonical_key_hash(nested));
}
TEST(NativeSupportOrderingTest, DagOrderingUsesRecordsThenEndpointsAndKeepsKeyIdentity)
{
    auto changed = recipes();
    changed[2].root[0] = 4;
    const auto base = Access::support_dag(recipes(), 0, 3);
    EXPECT_LT(base, Access::support_dag(std::move(changed), 0, 3));
    EXPECT_LT(base, Access::support_dag(recipes(), 1, 3));
    const auto restored = P::decode_identity<Support>(P::encode_identity(base));
    EXPECT_EQ(restored, base);
    EXPECT_EQ(restored <=> base, std::strong_ordering::equal);
    EXPECT_EQ(P::canonical_key(restored), P::canonical_key(base));
}
} // namespace
