#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/ordinary/NativeGeometrySupport.hh"
#include "geometry/PLIC/details/ordinary/NativeSupportAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
#include <gtest/gtest.h>
#include <limits>
#include <tuple>
#include <type_traits>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::PLIC;
constexpr auto ordinary = ArithmeticMode::Ordinary;
constexpr auto exact = ArithmeticMode::Exact;
static_assert(!std::is_default_constructible_v<PointIdentity<ordinary>>);
static_assert(!std::is_default_constructible_v<LineIdentity<ordinary>>);
static_assert(!std::is_default_constructible_v<PlaneSupportIdentity<ordinary>>);
static_assert(!std::is_default_constructible_v<PointIdentity<exact>>);
static_assert(!std::is_default_constructible_v<LineIdentity<exact>>);
static_assert(!std::is_default_constructible_v<PlaneSupportIdentity<exact>>);
static_assert(std::is_default_constructible_v<DisplacementIdentity<ordinary>>);
static_assert(std::is_default_constructible_v<DisplacementIdentity<exact>>);
static_assert(!std::is_constructible_v<PointIdentity<ordinary>, std::vector<char>>);
static_assert(!std::is_convertible_v<PointIdentity<ordinary>, std::span<const char>>);
static_assert(!std::is_convertible_v<PointIdentity<ordinary>, PointIdentity<exact>>);
static_assert(!std::is_convertible_v<DisplacementIdentity<ordinary>, DisplacementIdentity<exact>>);
static_assert(!std::is_convertible_v<PointIdentity<ordinary>, LineIdentity<ordinary>>);
static_assert(!std::is_convertible_v<PointIdentity<ordinary>, PlaneSupportIdentity<ordinary>>);
static_assert(!std::is_convertible_v<PointIdentity<ordinary>, DisplacementIdentity<ordinary>>);
static_assert(!std::is_convertible_v<LineIdentity<ordinary>, PlaneSupportIdentity<ordinary>>);
static_assert(!std::is_convertible_v<PlaneSupportIdentity<ordinary>, DisplacementIdentity<ordinary>>);

template<ArithmeticMode Mode>
concept HasPointIdentity = requires { typename PointIdentity<Mode>; };
static_assert(!HasPointIdentity<ArithmeticMode::BoundedCertification>);
using IdentityTypes = std::tuple<PointIdentity<ordinary>, LineIdentity<ordinary>, PlaneSupportIdentity<ordinary>,
    DisplacementIdentity<ordinary>, PointIdentity<exact>, LineIdentity<exact>, PlaneSupportIdentity<exact>,
    DisplacementIdentity<exact>>;
template<std::size_t I, std::size_t J> constexpr bool separated_identity_types()
{
    using First = std::tuple_element_t<I, IdentityTypes>;
    using Second = std::tuple_element_t<J, IdentityTypes>;
    return I == J || (!std::is_constructible_v<First, Second> && !std::is_convertible_v<First, Second>);
}
template<std::size_t I, std::size_t... J> constexpr bool separated_identity_row(std::index_sequence<J...>)
{
    return (separated_identity_types<I, J>() && ...);
}
template<std::size_t... I> constexpr bool all_identity_types_separated(std::index_sequence<I...> indices)
{
    return (separated_identity_row<I>(indices) && ...);
}
static_assert(all_identity_types_separated(std::make_index_sequence<std::tuple_size_v<IdentityTypes>>{}));

template<ArithmeticMode Mode> void check_roundtrips()
{
    const std::array<GeometryPoint, 3> vertices{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}};
    const auto face = PlanarFace<Mode>::make(vertices);
    const auto point = face->vertex_identities().at(0);
    const auto line = face->edge_support_identities().at(0);
    const auto support = planar_support<Mode>(*face);
    EXPECT_EQ(decode_identity<PointIdentity<Mode>>(encode_identity(point)), point);
    EXPECT_EQ(decode_identity<LineIdentity<Mode>>(encode_identity(line)), line);
    EXPECT_EQ(decode_identity<PlaneSupportIdentity<Mode>>(encode_identity(support)), support);
    EXPECT_EQ(canonical_key(point), encode_identity(point));
    EXPECT_EQ(canonical_key_size(point), canonical_key(point).size());
    EXPECT_EQ(encoded_identity_size(point), encode_identity(point).size());
    EXPECT_EQ(canonical_face_key(*face), face->encode());
    EXPECT_THROW((decode_identity<PointIdentity<Mode>>(encode_identity(line))), std::invalid_argument);
    EXPECT_THROW((decode_identity<LineIdentity<Mode>>(encode_identity(support))), std::invalid_argument);
    EXPECT_THROW((decode_identity<PlaneSupportIdentity<Mode>>(encode_identity(point))), std::invalid_argument);
    auto bytes = encode_identity(point);
    bytes.push_back(0);
    EXPECT_THROW((decode_identity<PointIdentity<Mode>>(bytes)), std::invalid_argument);
}
TEST(GeometryIdentityTest, RoundtripsPreserveProviderKeys)
{
    check_roundtrips<ordinary>();
    check_roundtrips<exact>();
}
TEST(GeometryIdentityTest, ZeroDisplacementRetainsEmptyLegacyEncoding)
{
    const DisplacementIdentity<ordinary> finite;
    const DisplacementIdentity<exact> rational;
    EXPECT_TRUE(finite.is_zero());
    EXPECT_TRUE(rational.is_zero());
    EXPECT_TRUE(encode_identity(finite).empty());
    EXPECT_TRUE(encode_identity(rational).empty());
    EXPECT_EQ(displacement<ordinary>({}, {}), finite);
    EXPECT_EQ(displacement<exact>({}, {}), rational);
    EXPECT_EQ(decode_identity<DisplacementIdentity<ordinary>>({}), finite);
    EXPECT_EQ(decode_identity<DisplacementIdentity<exact>>({}), rational);
}
TEST(GeometryIdentityTest, PromotionIsExplicitAndPreservesRepresentedCoordinates)
{
    const std::array<GeometryPoint, 3> vertices{{{.25, -.5, 0}, {1, 0, 0}, {0, 1, 0}}};
    const auto finite = PlanarFace<ordinary>::make(vertices)->vertex_identities().front();
    const auto promoted = promote_identity(finite);
    EXPECT_EQ(point_display<exact>(promoted), vertices.front());
    EXPECT_THROW((decode_identity<PointIdentity<exact>>(encode_identity(finite))), std::invalid_argument);
    EXPECT_THROW((decode_identity<PointIdentity<ordinary>>(encode_identity(promoted))), std::invalid_argument);
    const auto shift = displacement<ordinary>({0, 0, 0}, {.125, -.25, .5});
    const auto exact_shift = promote_identity(shift);
    EXPECT_EQ(displacement_display<exact>(exact_shift), (GeometryPoint{.125, -.25, .5}));
    EXPECT_EQ(point_display<exact>(point_translated<exact>(promoted, exact_shift)), (GeometryPoint{.375, -.75, .5}));
    EXPECT_TRUE(promote_identity(DisplacementIdentity<ordinary>{}).is_zero());
    EXPECT_THROW((decode_identity<DisplacementIdentity<exact>>(encode_identity(shift))), std::invalid_argument);
    EXPECT_THROW(
        (decode_identity<DisplacementIdentity<ordinary>>(encode_identity(exact_shift))), std::invalid_argument);
    EXPECT_THROW((decode_identity<PointIdentity<ordinary>>(encode_identity(shift))), std::invalid_argument);
}
TEST(GeometryIdentityTest, RetainedSupportAncestrySurvivesCoincidenceAndNestedTranslation)
{
    namespace on = details::ordinary_native;
    const std::array<on::WidePoint, 3> points{{{0, 0, 0}, {1, -1, 0}, {0, 0, 1}}};
    const auto first_support = on::input_support_packet({1, 1, 0}, {}, 0);
    const auto second_support = on::input_support_packet({2, 2, 0}, {}, 0);
    ASSERT_NE(first_support, second_support);
    const auto first = details::NativeSupportAccess::face_support(points, {1, 1, 0}, first_support);
    const auto second = details::NativeSupportAccess::face_support(points, {1, 1, 0}, second_support);
    EXPECT_NE(planar_support<ordinary>(*first), planar_support<ordinary>(*second));
    const auto shift = displacement<ordinary>({}, {.25, 0, 0});
    const auto nested = first->translated(shift)->translated(shift);
    const auto direct = first->translated(displacement<ordinary>({}, {.5, 0, 0}));
    EXPECT_NE(planar_support<ordinary>(*nested), planar_support<ordinary>(*direct));
    const auto support = planar_support<ordinary>(*nested);
    EXPECT_EQ(decode_identity<PlaneSupportIdentity<ordinary>>(encode_identity(support)), support);
    EXPECT_EQ(planar_support<ordinary>(*PlanarFace<ordinary>::decode(nested->encode_incidence())), support);
}
TEST(GeometryIdentityTest, OrdinaryRetainsCheckedExactSupportAncestry)
{
    const std::array<GeometryPoint, 3> points{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}};
    const auto support = planar_support<exact>(*PlanarFace<exact>::make(points));
    const auto retained = decode_identity<PlaneSupportIdentity<ordinary>>(encode_identity(support));
    EXPECT_EQ(encode_identity(retained), encode_identity(support));
    EXPECT_THROW((decode_identity<LineIdentity<ordinary>>(encode_identity(support))), std::invalid_argument);
}

// Fixed original-key fingerprints. These expected values were captured before
// the typed migration; transport versions must never change this key schema.
struct KeyFingerprint
{
    std::size_t bytes;
    std::uint64_t hash;
};
template<ArithmeticMode Mode> void check_frozen_key_fixtures()
{
    constexpr KeyFingerprint finite[5][4] = {
        {{376, 12932898835778235381ULL}, {32, 384062650128611582ULL}, {56, 2818258960210848039ULL},
            {128, 17496457904856703044ULL}},
        {{744, 9288408108487068372ULL}, {320, 18214248780328361876ULL}, {56, 2818258960210848039ULL},
            {144, 4912124907681746310ULL}},
        {{376, 1316226271676016980ULL}, {32, 384062650128611582ULL}, {72, 9123539038469882533ULL},
            {144, 8843402562579065990ULL}},
        {{376, 12932898835778235381ULL}, {32, 384062650128611582ULL}, {72, 1104534281275781797ULL},
            {144, 9457837747480208935ULL}},
        {{376, 12932898835778235381ULL}, {32, 384062650128611582ULL}, {56, 2818258960210848039ULL},
            {128, 17496457904856703044ULL}},
    };
    constexpr KeyFingerprint rational[5][4] = {
        {{432, 8543248322533805272ULL}, {144, 16172476167893701600ULL}, {110, 1479579668188916136ULL},
            {212, 2955527840678384250ULL}},
        {{432, 16322552645970534970ULL}, {144, 12004435486334572297ULL}, {110, 1479579668188916136ULL},
            {212, 6299085199136013819ULL}},
        {{432, 16218280137788398344ULL}, {144, 16172476167893701600ULL}, {110, 5289762087456169833ULL},
            {212, 15274175975852174970ULL}},
        {{432, 8543248322533805272ULL}, {144, 16172476167893701600ULL}, {110, 11367116446296336017ULL},
            {212, 17127662036282428219ULL}},
        {{432, 8543248322533805272ULL}, {144, 16172476167893701600ULL}, {110, 1479579668188916136ULL},
            {212, 2955527840678384250ULL}},
    };
    const auto& expected = Mode == ordinary ? finite : rational;
    const std::vector<GeometryPoint> axis{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
    std::array<std::vector<GeometryPoint>, 5> loops{
        axis, std::vector<GeometryPoint>{{0, 0, 0}, {1, 0, 1}, {1, 1, 2}, {0, 1, 1}}, axis, axis, axis};
    std::reverse(loops[2].begin(), loops[2].end());
    std::rotate(loops[3].begin(), loops[3].begin() + 1, loops[3].end());
    loops[4].insert(loops[4].begin() + 1, {.5, 0, 0});
    for (std::size_t fixture = 0; fixture < loops.size(); ++fixture)
    {
        SCOPED_TRACE(fixture);
        const auto face = PlanarFace<Mode>::make(loops[fixture]);
        const std::array<std::vector<char>, 4> keys{canonical_face_key(*face),
            canonical_key(planar_support<Mode>(*face)), canonical_key(face->vertex_identities().front()),
            canonical_key(face->edge_support_identities().front())};
        for (std::size_t kind = 0; kind < keys.size(); ++kind)
        {
            SCOPED_TRACE(kind);
            EXPECT_EQ(keys[kind].size(), expected[fixture][kind].bytes);
            EXPECT_EQ(Meshes::AMR::adaptive_route_hash(keys[kind]), expected[fixture][kind].hash);
        }
    }
}
TEST(GeometryIdentityTest, OrdinaryCanonicalKeysRetainOriginalFingerprints)
{
    if constexpr (std::numeric_limits<long double>::digits != 64)
        GTEST_SKIP() << "Original Ordinary key fixtures require the 64-bit significand profile.";
    check_frozen_key_fixtures<ordinary>();
}
TEST(GeometryIdentityTest, ExactCanonicalKeysRetainOriginalFingerprints)
{
    check_frozen_key_fixtures<exact>();
}

} // namespace
