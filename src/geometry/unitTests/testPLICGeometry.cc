/** @file testPLICGeometry.cc
 * @brief Typed PLIC geometry, arithmetic separation, and checked publication contracts.
 */
#include "geometry/unitTests/point_export_test_helpers.hh"
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/PlanarFace.hh"
#include "geometry/PLIC/SupportClassification.hh"
#include "geometry/unitTests/plic_analytic_test_geometry.hh"
#include <cfenv>
#include <cmath>
#include <concepts>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace
{
using SimpleFluid::ArithmeticMode;
namespace PLIC = SimpleFluid::PLIC;
using OrdinaryValue = PLIC::GeometryValue<ArithmeticMode::Ordinary>;
using BoundedValue = PLIC::GeometryValue<ArithmeticMode::BoundedCertification>;
using OrdinaryGeometry = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>;
using BoundedGeometry = PLIC::ConvexCellGeometry<ArithmeticMode::BoundedCertification>;

struct SkewBoundaryFixture
{
    static constexpr auto mode = ArithmeticMode::Ordinary;
    using Point = PLIC::Point<mode>;
    PLIC::ConvexGeometryPtr<mode> geometry;

    SkewBoundaryFixture(SimpleFluid::PLIC::GeometryPoint origin, double scale)
    {
        std::array<SimpleFluid::PLIC::GeometryPoint, 8> vertices;
        for (std::size_t i = 0; i < vertices.size(); ++i)
            vertices[i] = {origin[0] + scale * ((i & 1 ? 1. : 0.) + (i & 2 ? .2 : 0.)),
                origin[1] + scale * (i & 2 ? 1. : 0.), origin[2] + scale * (i & 4 ? 1. : 0.)};
        geometry = OrdinaryGeometry::from_native_shell(
            vertices, {{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}});
    }

    std::vector<Point> wall_points(std::size_t face) const
    {
        std::vector<Point> points;
        for (const auto& identity : geometry->face_geometry(face)->vertex_identities())
            points.push_back(PLIC::source_point<mode>(*geometry, identity));
        return points;
    }
};

TEST(PLICGeometryTest, OriginalSupportClassifiesSkewWallTangencyAcrossScalesAndTranslations)
{
    using Fixture = SkewBoundaryFixture;
    using Class = PLIC::OrdinaryBoundaryClass;
    const std::array<double, 3> scales{1. / 1024, 1., 1024.};
    for (const double scale : scales)
        for (const SimpleFluid::PLIC::GeometryPoint origin :
            {SimpleFluid::PLIC::GeometryPoint{0, 0, 0}, SimpleFluid::PLIC::GeometryPoint{2, -4, 8}})
        {
            SCOPED_TRACE(scale);
            const Fixture fixture(origin, scale);
            const PLIC::OrdinaryBoundary wall({fixture.geometry, 0});
            const auto points = fixture.wall_points(0);
            const auto center = PLIC::point_average<Fixture::mode>(points);
            const std::array walls{PLIC::FaceConstraint<Fixture::mode>{fixture.geometry, 0}};
            const auto departure =
                PLIC::backtrace_constrained<Fixture::mode>(center, {.7 * scale, -.3 * scale, .2 * scale}, .031, walls);
            EXPECT_EQ(wall.classify(departure), Class::ConstructionContact);
            EXPECT_TRUE(PLIC::has_any_support(departure.dynamic(), wall.authorities()));
            // The wall equation is x-origin_x = slope*(y-origin_y); the
            // trajectory may round its metric coordinates but preserves incidence.
            const auto wall_vertices = fixture.geometry->face_geometry(0)->points();
            const long double slope =
                (wall_vertices[2][0] - wall_vertices[0][0]) / (wall_vertices[2][1] - wall_vertices[0][1]);
            EXPECT_NEAR(departure.value[0] - origin[0], slope * (departure.value[1] - origin[1]),
                64 * std::numeric_limits<long double>::epsilon() * (scale + std::abs(origin[0])));
            EXPECT_EQ(PLIC::original_support_authority(*fixture.geometry->face_geometry(0)), wall.authorities()[0]);
        }
}

TEST(PLICGeometryTest, OriginalSupportRetainsBothIntersectingWallConstraints)
{
    using Fixture = SkewBoundaryFixture;
    using Class = PLIC::OrdinaryBoundaryClass;
    const Fixture fixture({0, 0, 0}, 1);
    const PLIC::OrdinaryBoundary left({fixture.geometry, 0}), bottom({fixture.geometry, 4});
    const auto points = fixture.wall_points(0);
    std::vector<Fixture::Point> edge;
    for (const auto& point : points)
        if (bottom.classify(point) == Class::ConstructionContact)
            edge.push_back(point);
    ASSERT_EQ(edge.size(), 2u);
    const auto source = PLIC::point_average<Fixture::mode>(edge);
    const std::array walls{PLIC::FaceConstraint<Fixture::mode>{fixture.geometry, 0},
        PLIC::FaceConstraint<Fixture::mode>{fixture.geometry, 4}};
    const auto departure = PLIC::backtrace_constrained<Fixture::mode>(source, {.7, -.3, .2}, .031, walls);
    EXPECT_EQ(left.classify(departure), Class::ConstructionContact);
    EXPECT_EQ(bottom.classify(departure), Class::ConstructionContact);
    EXPECT_EQ(departure.value[2], 0);
}

TEST(PLICGeometryTest, OriginalSupportRejectsCrossingWrongSupportMutationAndFreeCenterDisplacement)
{
    using Fixture = SkewBoundaryFixture;
    using Class = PLIC::OrdinaryBoundaryClass;
    const Fixture fixture({0, 0, 0}, 1);
    const PLIC::OrdinaryBoundary left({fixture.geometry, 0});
    const auto source = PLIC::point_average<Fixture::mode>(fixture.wall_points(0));
    EXPECT_EQ(left.classify(Fixture::Point({-.25, .5, .5})), Class::Outside);
    EXPECT_EQ(left.classify(Fixture::Point({.5, .5, .5})), Class::Inside);
    EXPECT_EQ(left.classify(Fixture::Point({0, 0, .5})), Class::CoordinateContact);
    const auto wrong = PLIC::point_average<Fixture::mode>(fixture.wall_points(1));
    EXPECT_FALSE(PLIC::has_any_support(wrong.dynamic(), left.authorities()));
    EXPECT_EQ(left.classify(wrong), Class::Inside);
    const std::array walls{PLIC::FaceConstraint<Fixture::mode>{fixture.geometry, 0}};
    EXPECT_THROW(
        (void) PLIC::backtrace_constrained<Fixture::mode>(wrong, {0, 0, 0}, .031, walls), std::invalid_argument);
    auto mutated = source;
    mutated.value[0] -= .01L;
    EXPECT_TRUE(mutated.supports().empty());
    EXPECT_EQ(left.classify(mutated), Class::Outside);
    const auto corrected = PLIC::point_displaced<Fixture::mode>(source, Fixture::Point({-.01, 0, 0}));
    EXPECT_TRUE(corrected.supports().empty());
    EXPECT_EQ(left.classify(corrected), Class::Outside);
    const std::array simplex{source, source, source, corrected};
    EXPECT_TRUE(PLIC::common_supports<Fixture::mode>(simplex).empty());
}

TEST(PLICGeometryTest, OriginalBoundaryCoordinateFallbackPreservesFloatingPointEnvironment)
{
    const auto geometry = OrdinaryGeometry::box({0, 0, 0}, {1, 1, 1});
    const PLIC::OrdinaryBoundary boundary({geometry, 0});
    const PLIC::OrdinaryBoundary upper({geometry, 1});
    EXPECT_EQ(upper.classify(PLIC::Point<ArithmeticMode::Ordinary>({std::nextafter(1.L, 2.L), .5L, .5L})),
        PLIC::OrdinaryBoundaryClass::Outside);
    const int rounding = std::fegetround();
    const int flags = std::fetestexcept(FE_ALL_EXCEPT);
    ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);
    const auto inside = boundary.classify(PLIC::Point<ArithmeticMode::Ordinary>({.5, .5, .5}));
    const int after = std::fegetround();
    const int after_flags = std::fetestexcept(FE_ALL_EXCEPT);
    EXPECT_EQ(std::fesetround(rounding), 0);
    EXPECT_EQ(inside, PLIC::OrdinaryBoundaryClass::Inside);
    EXPECT_EQ(after, FE_DOWNWARD);
    EXPECT_EQ(after_flags, flags);
    // Scaling this smallest positive coordinate into the stored cell frame
    // loses representability. Arithmetic uncertainty cannot prove wall contact.
    EXPECT_EQ(boundary.classify(
                  PLIC::Point<ArithmeticMode::Ordinary>({std::numeric_limits<long double>::denorm_min(), .5, .5})),
        PLIC::OrdinaryBoundaryClass::Unresolved);
    EXPECT_THROW((void) PLIC::OrdinaryBoundary({{}, 0}), std::invalid_argument);
    EXPECT_THROW((void) PLIC::OrdinaryBoundary({geometry, geometry->num_faces()}), std::invalid_argument);
}

TEST(PLICGeometryTest, CoordinateWallContactDoesNotAcquireOriginalConstructionSupport)
{
    constexpr auto mode = ArithmeticMode::Ordinary;
    using Point = PLIC::Point<mode>;
    const auto geometry = OrdinaryGeometry::box({0, 0, 0}, {1, 1, 1});
    const PLIC::OrdinaryBoundary boundary({geometry, 0});
    const std::array walls{PLIC::FaceConstraint<mode>{geometry, 0}};
    const auto represented = PLIC::backtrace_constrained<mode>(Point({0, .5, .5}), {0, 1, 0}, .125, walls);
    EXPECT_TRUE(represented.supports().empty());
    EXPECT_EQ(boundary.classify(represented), PLIC::OrdinaryBoundaryClass::CoordinateContact);

    std::vector<Point> authenticated;
    for (const auto& identity : geometry->face_geometry(0)->vertex_identities())
        authenticated.push_back(PLIC::source_point<mode>(*geometry, identity));
    const auto source = PLIC::point_average<mode>(authenticated);
    const auto constructed = PLIC::backtrace_constrained<mode>(source, {0, 1, 0}, .125, walls);
    EXPECT_FALSE(constructed.supports().empty());
    EXPECT_EQ(boundary.classify(constructed), PLIC::OrdinaryBoundaryClass::ConstructionContact);
}

template<class T>
concept HasErrorField = requires(T value) { value.absolute_error; };
template<class T>
concept HasEnclosure = requires(T value) {
    value.lower();
    value.upper();
    value.relative_error_at_most(1.L, 1e-12);
};

static_assert(sizeof(OrdinaryValue) == sizeof(long double));
static_assert(!HasErrorField<OrdinaryValue>);
static_assert(!HasEnclosure<OrdinaryValue>);
static_assert(HasErrorField<BoundedValue> && HasEnclosure<BoundedValue>);
static_assert(HasErrorField<SimpleFluid::PLIC::GeometryValue<SimpleFluid::ArithmeticMode::Exact>>);
static_assert(!std::is_convertible_v<OrdinaryValue, BoundedValue>);
static_assert(!std::is_convertible_v<BoundedValue, OrdinaryValue>);
static_assert(
    !std::is_convertible_v<SimpleFluid::PLIC::GeometryValue<SimpleFluid::ArithmeticMode::Exact>, OrdinaryValue>);
static_assert(std::same_as<decltype(PLIC::GeometryMoments<ArithmeticMode::Ordinary>{}.volume), OrdinaryValue>);
static_assert(
    std::same_as<decltype(PLIC::GeometryMoments<ArithmeticMode::BoundedCertification>{}.volume), BoundedValue>);

TEST(PLICGeometryTest, OrdinaryPublicationRejectsNonfiniteAndLostNonzeroValues)
{
    EXPECT_DOUBLE_EQ(OrdinaryValue{.5L}.as_double(), .5);
    EXPECT_EQ(OrdinaryValue{.5L}.binary64_absolute_error(), 0);
    EXPECT_THROW((void) OrdinaryValue{std::numeric_limits<long double>::infinity()}.as_double(), std::invalid_argument);
    EXPECT_THROW(
        (void) OrdinaryValue{std::numeric_limits<long double>::quiet_NaN()}.as_double(), std::invalid_argument);
    const long double below_binary64 = static_cast<long double>(std::numeric_limits<double>::denorm_min()) / 2;
    if (below_binary64 > 0)
        EXPECT_THROW((void) OrdinaryValue{below_binary64}.as_double(), std::invalid_argument);
}

TEST(PLICGeometryTest, BoundedPublicationRejectsAnUncertainZero)
{
    if (!PLIC::certification_environment())
        GTEST_SKIP() << "Bounded wider floating-point environment unavailable.";
    EXPECT_THROW(
        (void) (BoundedValue{0, std::numeric_limits<double>::denorm_min()}.as_double()), std::invalid_argument);
}

TEST(PLICGeometryTest, ExactPublicationRetainsConversionUncertaintyAndNonzeroEvidence)
{
    const SimpleFluid::PLIC::GeometryValue<SimpleFluid::ArithmeticMode::Exact> enclosed{1, 1e-12L, true};
    EXPECT_DOUBLE_EQ(enclosed.as_double(), 1);
    EXPECT_GE(enclosed.binary64_absolute_error(), enclosed.absolute_error);
    const SimpleFluid::PLIC::GeometryValue<SimpleFluid::ArithmeticMode::Exact> underflowed{
        0, std::numeric_limits<long double>::denorm_min(), true};
    EXPECT_THROW((void) underflowed.as_double(), std::runtime_error);
    EXPECT_THROW((void) underflowed.binary64_absolute_error(), std::runtime_error);
    const SimpleFluid::PLIC::GeometryValue<SimpleFluid::ArithmeticMode::Exact> overflowed{
        std::numeric_limits<long double>::infinity(), 0, true};
    EXPECT_THROW((void) overflowed.as_double(), std::runtime_error);
}

TEST(PLICGeometryTest, PublicationAccountsForAdditionalBinary64Rounding)
{
    if constexpr (std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits)
        GTEST_SKIP() << "A wider long double is required to exercise additional binary64 rounding.";
    const long double halfway = 1.L + std::ldexp(1.L, -std::numeric_limits<double>::digits);
    const OrdinaryValue ordinary{halfway};
    const auto rounded = ordinary.as_double();
    const auto difference = std::abs(halfway - static_cast<long double>(rounded));
    ASSERT_GT(difference, 0);
    EXPECT_GE(ordinary.binary64_absolute_error(), difference);
    const SimpleFluid::PLIC::GeometryValue<SimpleFluid::ArithmeticMode::Exact> exact{halfway, difference, true};
    EXPECT_GE(exact.binary64_absolute_error(), 2 * difference);
}

template<class Left, class Right>
concept CanIntersect = requires(const Left& a, const Right& b) { a.intersect(b); };
static_assert(CanIntersect<OrdinaryGeometry, OrdinaryGeometry>);
static_assert(!CanIntersect<OrdinaryGeometry, BoundedGeometry>);
static_assert(!CanIntersect<BoundedGeometry, OrdinaryGeometry>);
static_assert(
    !std::is_convertible_v<PLIC::Point<ArithmeticMode::Ordinary>, PLIC::Point<ArithmeticMode::BoundedCertification>>);
static_assert(!HasErrorField<PLIC::Point<ArithmeticMode::Ordinary>>);
static_assert(HasErrorField<PLIC::Point<ArithmeticMode::BoundedCertification>>);
static_assert(std::same_as<decltype(OrdinaryGeometry::box({}, {})), PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>>);
static_assert(std::same_as<decltype(std::declval<const OrdinaryGeometry&>().clip({})),
    PLIC::ClipResult<ArithmeticMode::Ordinary>>);
static_assert(std::same_as<decltype(std::declval<const OrdinaryGeometry&>().volume()), OrdinaryValue>);
static_assert(std::same_as<decltype(std::declval<const BoundedGeometry&>().volume()), BoundedValue>);
static_assert(std::same_as<
    decltype(std::declval<const SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>&>().clip({})),
    PLIC::ClipResult<ArithmeticMode::Exact>>);
static_assert(std::same_as<
    decltype(std::declval<const SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>&>().volume()),
    SimpleFluid::PLIC::GeometryValue<SimpleFluid::ArithmeticMode::Exact>>);

TEST(PLICGeometryTest, OrdinaryClipPreservesTypedPhasesAndAnalyticMoments)
{
    PLIC::reset_geometry_statistics();
    const auto cube = OrdinaryGeometry::box({0, 0, 0}, {1, 1, 1});
    ASSERT_TRUE(cube);
    EXPECT_EQ(cube->volume().value, 1);
    const PLIC::GeometryHalfspace cut{{1, 0, 0}, {0, 0, 0}, .25};
    const auto clipped = cube->clip(cut);
    ASSERT_TRUE(clipped.inside);
    ASSERT_TRUE(clipped.outside);
    EXPECT_NEAR(clipped.inside->volume().value, .25L, 1e-15L);
    EXPECT_NEAR(clipped.outside->volume().value, .75L, 1e-15L);
    EXPECT_NEAR(clipped.inside->volume().value + clipped.outside->volume().value, cube->volume().value, 1e-15L);
    const auto moments = clipped.inside->moments({0, 0, 0});
    EXPECT_NEAR(moments.first_moment[0].value, .03125L, 1e-15L);
    EXPECT_NEAR(moments.first_moment[1].value, .125L, 1e-15L);
    EXPECT_NEAR(moments.first_moment[2].value, .125L, 1e-15L);
    const auto phases = cube->phases(cut);
    EXPECT_NEAR(phases.gas.value, .25L, 1e-15L);
    EXPECT_NEAR(phases.carrier.value, .75L, 1e-15L);
    EXPECT_EQ(PLIC::geometry_statistics().interval_operations, 0u);
}

TEST(PLICGeometryTest, TypedEntryRejectsForeignRuntimeModeAndReportsOrdinaryFailures)
{
    const auto cube = OrdinaryGeometry::box({0, 0, 0}, {1, 1, 1});
    EXPECT_THROW((void) BoundedGeometry::from_dynamic(cube->dynamic_geometry()), std::invalid_argument);
    try
    {
        (void) OrdinaryGeometry::box({0, 0, 0}, {0, 1, 1});
        FAIL() << "A zero-width box must reject.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_TRUE(std::string(error.what()).starts_with("Ordinary FP geometry: ")) << error.what();
    }
}

TEST(PLICGeometryTest, BoundedClipEnclosesIndependentAnalyticVolumes)
{
    if (!PLIC::certification_environment())
        GTEST_SKIP() << "Bounded wider floating-point environment unavailable.";
    const auto cube = BoundedGeometry::box({0, 0, 0}, {1, 1, 1});
    const auto cut = cube->clip({{1, 0, 0}, {0, 0, 0}, .25});
    ASSERT_TRUE(cut.inside);
    ASSERT_TRUE(cut.outside);
    const auto inside = cut.inside->volume();
    const auto outside = cut.outside->volume();
    EXPECT_LE(inside.lower(), .25L);
    EXPECT_GE(inside.upper(), .25L);
    EXPECT_LE(outside.lower(), .75L);
    EXPECT_GE(outside.upper(), .75L);
    EXPECT_LE(inside.lower() + outside.lower(), 1.L);
    EXPECT_GE(inside.upper() + outside.upper(), 1.L);
    EXPECT_TRUE(inside.relative_error_at_most(.25L, 1e-12));
}

TEST(PLICGeometryTest, BoundedPublicationIncludesAdditionalBinary64Rounding)
{
    if (!PLIC::certification_environment())
        GTEST_SKIP() << "Bounded wider floating-point environment unavailable.";
    if constexpr (std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits)
        GTEST_SKIP() << "A wider long double is required to exercise additional binary64 rounding.";
    const long double halfway = 1.L + std::ldexp(1.L, -std::numeric_limits<double>::digits);
    const auto difference = std::abs(halfway - static_cast<long double>(static_cast<double>(halfway)));
    ASSERT_GT(difference, 0);
    const BoundedValue value{halfway, difference};
    EXPECT_GE(value.binary64_absolute_error(), 2 * difference);
}

TEST(PLICGeometryTest, TypedPacketsRetainModeAndRejectStaleTransactionContext)
{
    const auto cube = OrdinaryGeometry::box({0, 0, 0}, {1, 1, 1});
    const PLIC::GeometryPacketContext context{{17, 23, 31, 47}};
    const std::array sources{cube};
    const auto cut = cube->clip({{1, 0, 0}, {0, 0, 0}, .25});
    ASSERT_TRUE(cut.inside);
    const auto packet = cut.inside->encode(context);
    const auto restored = OrdinaryGeometry::decode(packet, context, sources);
    ASSERT_TRUE(restored);
    EXPECT_DOUBLE_EQ(restored->volume().as_double(), .25);
    auto stale = context;
    ++stale.publication.back();
    EXPECT_THROW((void) OrdinaryGeometry::decode(packet, stale, sources), std::invalid_argument);
    EXPECT_DOUBLE_EQ(cube->volume().as_double(), 1);
}

TEST(PLICGeometryTest, TypedInputConversionHonorsConstructionCaps)
{
    using Point = PLIC::Point<ArithmeticMode::Ordinary>;
    const std::array vertices{Point({0, 0, 0}), Point({1, 0, 0}), Point({0, 1, 0}), Point({0, 0, 1})};
    const std::vector<std::vector<std::size_t>> faces{{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
    PLIC::GeometryLimits limits;
    limits.maximum_vertices = 3;
    PLIC::reset_geometry_statistics();
    EXPECT_THROW((void) OrdinaryGeometry::make_points(vertices, faces, limits), std::invalid_argument);
    EXPECT_EQ(PLIC::geometry_statistics().constructions, 0u);
    limits.maximum_vertices = 4;
    limits.byte_budget = 1;
    EXPECT_THROW((void) OrdinaryGeometry::make_points(vertices, faces, limits), std::invalid_argument);
    EXPECT_EQ(PLIC::geometry_statistics().constructions, 0u);
    EXPECT_NEAR(OrdinaryGeometry::make_points(vertices, faces)->volume().value, 1.L / 6, 1e-15L);
}
template<ArithmeticMode Mode> void fixed_mode_operations_do_not_redispatch()
{
    using Geometry = PLIC::ConvexCellGeometry<Mode>;
    using Point = PLIC::Point<Mode>;
    PLIC::reset_geometry_statistics();
    const auto source = OrdinaryGeometry::box({0, 0, 0}, {1, 1, 1});
    const auto imported = Geometry::from_geometry(*source);
    EXPECT_DOUBLE_EQ(imported->volume().as_double(), 1);
    if constexpr (Mode == ArithmeticMode::BoundedCertification)
        for (const auto& point : SimpleFluid::PLIC::test::exported_points(*imported))
            EXPECT_TRUE(point.supports().empty());
    const auto cube = plic_test::axis_box<Mode>({0, 0, 0}, {1, 1, 1});
    const PLIC::GeometryHalfspace plane{{1, 0, 0}, {0, 0, 0}, .25};
    const auto cut = cube->clip(plane);
    ASSERT_TRUE(cut.inside);
    ASSERT_TRUE(cut.outside);
    EXPECT_DOUBLE_EQ(cube->volume().as_double(), 1);
    EXPECT_NEAR(cube->phases(plane).gas.as_double(), .25, 1e-14);
    EXPECT_NEAR(cube->moments({0, 0, 0}).first_moment[0].as_double(), .5, 1e-14);
    EXPECT_NEAR(cube->centroid()[0], .5, 1e-14);
    EXPECT_GT(cube->storage_bytes(), 0u);
    EXPECT_GT(cube->axis_spans()[0].value, 0);
    EXPECT_GT(cube->support({1, 0, 0}, {0, 0, 0}).upper.value, 0);
    EXPECT_TRUE(cut.inside->bounds_prove_disjoint_interiors(*cut.outside));
    EXPECT_NEAR(cube->intersect(*cut.inside)->volume().as_double(), .25, 1e-14);
    EXPECT_NEAR(cut.inside->translated({1, 0, 0})->volume().as_double(), .25, 1e-14);
    const std::array donors{cut.inside, cut.outside};
    const PLIC::ConvexPartition<Mode> partition(donors);
    const auto pieces = partition.partition(*cube);
    ASSERT_EQ(pieces.size(), 2u);
    std::vector<PLIC::ConvexGeometryPtr<Mode>> cover;
    for (const auto& piece : pieces)
        cover.push_back(piece.geometry);
    EXPECT_NO_THROW(cube->certify_fragment_partition(cover, 0));
    cover.pop_back();
    EXPECT_THROW(cube->certify_fragment_partition(cover, 0), std::invalid_argument);
    const PLIC::GeometryPacketContext context{{11, 13, 17, 19}};
    const std::array sources{cube};
    const auto restored = Geometry::decode(cut.inside->encode(context), context, sources);
    EXPECT_NEAR(restored->volume().as_double(), .25, 1e-14);
    const auto points = SimpleFluid::PLIC::test::exported_points(*cube);
    const auto local_points = cube->local_points();
    EXPECT_EQ(points.size(), local_points.size());
    std::vector<Point> face;
    for (auto vertex : cube->faces().front())
        face.push_back(points[vertex]);
    const auto supports = PLIC::common_supports<Mode>(face);
    ASSERT_FALSE(supports.empty());
    const auto overlap = PLIC::planar_face_overlap<Mode>(face, face, supports.front());
    EXPECT_NEAR(overlap.overlap_over_first.as_double(), 1, 1e-14);
    const auto average = PLIC::point_average<Mode>(face);
    EXPECT_TRUE(PLIC::points_identical<Mode>(average, average));
    EXPECT_GT(PLIC::point_storage_bytes(average), 0u);
    const auto interpolated = PLIC::point_interpolate<Mode>(face[0], face[1], .5L);
    const auto displaced = PLIC::point_displaced<Mode>(interpolated, Point({1, 0, 0}));
    EXPECT_NEAR(displaced.value[0], interpolated.value[0] + 1, 1e-14L);
    const auto departure = PLIC::backtrace_constrained<Mode>(average, {0, 0, 0}, .1, {});
    EXPECT_TRUE(PLIC::points_identical<Mode>(departure, average));
    const std::array tetrahedron{Point({0, 0, 0}), Point({1, 0, 0}), Point({0, 1, 0}), Point({0, 0, 1})};
    EXPECT_NEAR(PLIC::orientation<Mode>(tetrahedron).as_double(), 1, 1e-14);
    EXPECT_NEAR(Geometry::tetrahedron(tetrahedron)->volume().as_double(), 1. / 6, 1e-14);
    EXPECT_EQ(PLIC::geometry_statistics().runtime_dispatches, 0u);
}

TEST(PLICGeometryTest, OrdinaryOperationsKeepTheSelectedKernelThroughNestedWork)
{
    fixed_mode_operations_do_not_redispatch<ArithmeticMode::Ordinary>();
}

TEST(PLICGeometryTest, BoundedOperationsKeepTheSelectedKernelThroughNestedWork)
{
    if (!PLIC::certification_environment())
        GTEST_SKIP() << "Bounded wider floating-point environment unavailable.";
    fixed_mode_operations_do_not_redispatch<ArithmeticMode::BoundedCertification>();
}

TEST(PLICGeometryTest, RuntimeScalarPublicationSelectionIsCounted)
{
    PLIC::reset_geometry_statistics();
    const auto cube = OrdinaryGeometry::box({0, 0, 0}, {1, 1, 1});
    EXPECT_EQ(PLIC::geometry_statistics().runtime_dispatches, 0u);
    const PLIC::DynamicGeometryValue published{cube->volume().value, 0, ArithmeticMode::Ordinary};
    EXPECT_DOUBLE_EQ(published.as_double(), 1);
    EXPECT_EQ(PLIC::geometry_statistics().runtime_dispatches, 1u);
}
} // namespace
