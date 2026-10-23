/** @file testBoundedConvexGeometry.cc
 * @brief Independent analytic/reference tests of bounded finite geometry.
 */
#include "geometry/BoundedConvexGeometry.hh"
#include "geometry/ExactPlanarFace.hh"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace
{
using namespace SimpleFluid::Meshes;
using Faces = std::vector<std::vector<std::size_t>>;
void encloses(const BoundedGeometryValue& value, long double oracle)
{
    EXPECT_LE(value.lower(), oracle);
    EXPECT_GE(value.upper(), oracle);
    EXPECT_GE(value.absolute_error, 0);
}
BoundedConvexGeometryPtr cube()
{
    return BoundedConvexGeometry::box({0, 0, 0}, {1, 1, 1});
}

TEST(BoundedConvexGeometryTest, AnalyticCubeAndObliqueTetrahedronEnclosures)
{
    ASSERT_TRUE(bounded_geometry_environment());
    const auto parent = cube();
    encloses(parent->volume(), 1);
    EXPECT_EQ(parent->volume().absolute_error, 0);
    const auto m = parent->moments();
    for (const auto& moment : m.first_moment)
        encloses(moment, .5L);
    const auto cut = parent->clip({{1, 1, 1}, {}, 1});
    ASSERT_TRUE(cut.inside);
    ASSERT_TRUE(cut.outside);
    encloses(cut.inside->volume(), 1.L / 6);
    encloses(cut.outside->volume(), 5.L / 6);
    const auto small = cut.inside->moments();
    for (const auto& moment : small.first_moment)
        encloses(moment, 1.L / 24);
    encloses(cut.section.area, std::sqrt(3.L) / 2);
    const std::array pieces{cut.inside, cut.outside};
    EXPECT_NO_THROW(parent->certify_partition(pieces, 8e-15));
    EXPECT_TRUE(cut.inside->volume().relative_error_at_most(1.L / 6, 8e-15));
}

TEST(BoundedConvexGeometryTest, TranslationAndPowerOfTwoScaleDoNotChangeLocalAccuracy)
{
    for (const int exponent : {-100, 0, 100})
    {
        const double scale = std::ldexp(1., exponent), anchor = 16 * scale;
        const auto cell =
            BoundedConvexGeometry::box({anchor, -anchor, anchor}, {anchor + scale, -anchor + scale, anchor + scale});
        const auto volume = std::ldexp(1.L, 3 * exponent);
        encloses(cell->volume(), volume);
        const auto cut = cell->clip({{1, 1, 1}, {anchor, -anchor, anchor}, scale});
        ASSERT_TRUE(cut.inside);
        encloses(cut.inside->volume(), volume / 6);
        for (const auto& moment : cut.inside->moments().first_moment)
            encloses(moment, std::ldexp(1.L / 24, 4 * exponent));
        EXPECT_TRUE(cut.inside->volume().relative_error_at_most(volume / 6, 8e-15));
        for (const auto& span : cell->axis_spans())
            encloses(span, scale);
    }
}

TEST(BoundedConvexGeometryTest, RationalSourceExportRetainsIndependentOracleInsideBounds)
{
    const auto exact = ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1})->clip({{10, 0, 0}, {}, 1}).inside;
    ASSERT_TRUE(exact);
    const auto finite = BoundedConvexGeometry::from_geometry(*exact);
    encloses(finite->volume(), .1L);
    const auto cut = finite->clip({{0, 1, 1}, {}, .5});
    ASSERT_TRUE(cut.inside);
    encloses(cut.inside->volume(), .0125L);
    const auto oracle = exact->clip({{0, 1, 1}, {}, .5}).inside->moments();
    const auto measured = cut.inside->moments();
    EXPECT_LE(std::abs(measured.volume.value - oracle.volume.value),
        measured.volume.absolute_error + oracle.volume.absolute_error);
    for (std::size_t axis = 0; axis < 3; ++axis)
        EXPECT_LE(std::abs(measured.first_moment[axis].value - oracle.first_moment[axis].value),
            measured.first_moment[axis].absolute_error + oracle.first_moment[axis].absolute_error);
}

TEST(BoundedConvexGeometryTest, ComplementaryAndReversedCutsRetainStructuralEvidence)
{
    reset_bounded_geometry_statistics();
    const auto parent = cube();
    const GeometryHalfspace plane{{3, 2, 1}, {}, 2.25};
    const auto cut = parent->clip(plane);
    const auto reverse = parent->clip({{-3, -2, -1}, {}, -2.25});
    ASSERT_TRUE(cut.inside);
    ASSERT_TRUE(reverse.outside);
    EXPECT_LE(std::abs(cut.inside->volume().value - reverse.outside->volume().value),
        cut.inside->volume().absolute_error + reverse.outside->volume().absolute_error);
    const auto nested = cut.inside->clip({{0, 1, 0}, {}, .3});
    const std::array pieces{nested.inside, nested.outside, cut.outside};
    EXPECT_NO_THROW(parent->certify_partition(pieces, 8e-15));
    const auto same = cut.inside->clip(plane);
    EXPECT_TRUE(same.inside);
    EXPECT_FALSE(same.outside);
    const auto opposite = cut.inside->clip({{-3, -2, -1}, {}, -2.25});
    EXPECT_FALSE(opposite.inside);
    EXPECT_TRUE(opposite.outside);
    const auto counters = bounded_geometry_statistics();
    EXPECT_GT(counters.structural_partition_reuses, 0U);
    EXPECT_EQ(counters.exact_predicates, 0U);
    EXPECT_EQ(counters.rational_constructions, 0U);
    EXPECT_EQ(counters.rational_measures, 0U);
    EXPECT_EQ(counters.rational_ledgers, 0U);
}

TEST(BoundedConvexGeometryTest, TangenciesPreserveFaceEdgeAndPointDimension)
{
    const auto parent = cube();
    const auto face = parent->clip({{1, 0, 0}, {}, 0});
    EXPECT_FALSE(face.inside);
    EXPECT_TRUE(face.outside);
    encloses(face.section.area, 1);
    EXPECT_EQ(face.section.vertices.size(), 4U);
    const auto edge = parent->clip({{1, 1, 0}, {}, 0});
    EXPECT_FALSE(edge.inside);
    EXPECT_EQ(edge.section.vertices.size(), 2U);
    EXPECT_EQ(edge.section.area.value, 0);
    const auto vertex = parent->clip({{1, 1, 1}, {}, 0});
    EXPECT_FALSE(vertex.inside);
    EXPECT_EQ(vertex.section.vertices.size(), 1U);
    EXPECT_EQ(vertex.section.area.value, 0);
}

TEST(BoundedConvexGeometryTest, IndependentContainedPiecesNeedNoNetVolumeAssumption)
{
    const auto parent = cube();
    const auto box = [](double lo, double hi) { return BoundedConvexGeometry::box({lo, 0, 0}, {hi, 1, 1}); };
    const std::array good{box(0, .5), box(.5, 1)};
    EXPECT_NO_THROW(parent->certify_partition(good, 8e-15));
    const std::array gap{box(0, .4), box(.6, 1)};
    EXPECT_THROW(parent->certify_partition(gap, 8e-15), std::invalid_argument);
    // The .2 overlap and .2 hole balance exactly in the bulk sum.
    const std::array balanced{box(0, .4), box(.2, .6), box(.8, 1)};
    EXPECT_THROW(parent->certify_partition(balanced, 8e-15), std::invalid_argument);
    const std::array outside{box(-.1, .4), box(.5, 1)};
    EXPECT_THROW(parent->certify_partition(outside, 8e-15), std::invalid_argument);
    const std::array duplicate{good[0], good[0]};
    EXPECT_THROW(parent->certify_partition(duplicate, 8e-15), std::invalid_argument);
}

TEST(BoundedConvexGeometryTest, IntersectionUsesActualObliqueShell)
{
    const auto tetra = cube()->clip({{1, 1, 1}, {}, 1}).inside;
    const auto inner = BoundedConvexGeometry::box({0, 0, 0}, {.5, 1, 1});
    const auto overlap = tetra->intersect(*inner);
    ASSERT_TRUE(overlap);
    encloses(overlap->volume(), 7.L / 48);
    const auto outside = BoundedConvexGeometry::box({2, 0, 0}, {3, 1, 1});
    EXPECT_FALSE(tetra->intersect(*outside));
}

TEST(BoundedConvexGeometryTest, MomentsAreOptionalAndSourceWorkIsSeparated)
{
    const auto source = ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1});
    reset_bounded_geometry_statistics();
    const auto finite = BoundedConvexGeometry::from_geometry(*source);
    (void) finite->volume();
    auto counters = bounded_geometry_statistics();
    EXPECT_EQ(counters.source_imports, 1U);
    EXPECT_EQ(counters.moment_evaluations, 0U);
    (void) finite->moments();
    (void) finite->centroid();
    const auto shifted_moments = finite->moments({.25, .25, .25});
    for (const auto& moment : shifted_moments.first_moment)
        encloses(moment, .25L);
    counters = bounded_geometry_statistics();
    EXPECT_EQ(counters.moment_evaluations, 1U);
    const auto translated = finite->translated({2, 0, 0});
    (void) translated->moments();
    EXPECT_EQ(bounded_geometry_statistics().moment_evaluations, 2U);
    EXPECT_EQ(counters.rational_measures, 0U);
}

TEST(BoundedConvexGeometryTest, ProvenZeroIsDifferentFromUncertainOrientation)
{
    std::array<BoundedPoint, 4> vertices;
    vertices[1].value = {1, 0, 0};
    vertices[2].value = {0, 1, 0};
    vertices[3].value = {0, 0, 1};
    encloses(bounded_orientation(vertices), 1);
    ASSERT_TRUE(BoundedConvexGeometry::tetrahedron(vertices));
    vertices[3] = vertices[0];
    EXPECT_EQ(bounded_orientation(vertices).absolute_error, 0);
    EXPECT_FALSE(BoundedConvexGeometry::tetrahedron(vertices));
    vertices[3].absolute_error[2] = 1e-4L;
    EXPECT_THROW((void) BoundedConvexGeometry::tetrahedron(vertices), std::invalid_argument);
}

TEST(BoundedConvexGeometryTest, TinyRepresentableVolumesRemainPositiveAndPublicationRejectsUnderflow)
{
    const auto cell = BoundedConvexGeometry::box({0, 0, 0}, {1e-200, 1e-200, 1e-200});
    EXPECT_GT(cell->volume().lower(), 0);
    EXPECT_THROW((void) cell->volume().as_double(), std::invalid_argument);
    const auto cut = cell->clip({{1, 0, 0}, {}, 5e-201});
    ASSERT_TRUE(cut.inside);
    EXPECT_GT(cut.inside->volume().lower(), 0);
}

TEST(BoundedConvexGeometryTest, RangeBudgetAndUnsupportedEnvironmentReject)
{
    auto limits = BoundedGeometryLimits{};
    limits.maximum_operations = 1;
    EXPECT_THROW((void) BoundedConvexGeometry::box({0, 0, 0}, {1, 1, 1}, limits), std::invalid_argument);
    limits = {};
    limits.byte_budget = 1;
    EXPECT_THROW((void) BoundedConvexGeometry::box({0, 0, 0}, {1, 1, 1}, limits), std::invalid_argument);
    EXPECT_THROW((void) cube()->clip({{0, 0, 0}, {}, 0}), std::invalid_argument);
    EXPECT_THROW((void) BoundedConvexGeometry::box({0, 0, 0}, {std::numeric_limits<double>::infinity(), 1, 1}),
        std::invalid_argument);
    const int original = std::fegetround();
    ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);
    EXPECT_FALSE(bounded_geometry_environment());
    EXPECT_THROW((void) cube(), std::invalid_argument);
    ASSERT_EQ(std::fesetround(original), 0);
}

TEST(BoundedConvexGeometryTest, MalformedAndUnresolvedPublicShellsCannotMintPlaneAuthority)
{
    const std::array<GeometryPoint, 4> tetra{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    EXPECT_THROW(
        (void) BoundedConvexGeometry::make(tetra, {{0, 1, 2}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}}), std::invalid_argument);
    auto points = cube()->points();
    points[6].absolute_error[0] = 1e-3L;
    EXPECT_THROW((void) BoundedConvexGeometry::make_bounded(points, cube()->faces()), std::invalid_argument);
}

TEST(BoundedConvexGeometryTest, AuthenticatedSupportProvenanceSurvivesAffineWorkAndRejectsMutation)
{
    const auto source = ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1});
    const auto finite = BoundedConvexGeometry::from_geometry(*source);
    const auto ids = source->face_geometry(0)->vertex_identities();
    std::vector<BoundedPoint> points;
    for (const auto& id : ids)
        points.push_back(bounded_source_point(*source, id));
    ASSERT_FALSE(points.front().supports().empty());
    for (const auto& p : finite->local_points())
        EXPECT_TRUE(p.supports().empty());
    const auto center = bounded_point_average(points);
    EXPECT_FALSE(center.supports().empty());
    const std::array planar{center, points[0], points[1], points[2]};
    EXPECT_EQ(bounded_orientation(planar).absolute_error, 0);
    EXPECT_EQ(bounded_orientation(planar).value, 0);
    auto changed = center;
    changed.value[0] += .1L;
    EXPECT_TRUE(changed.supports().empty());
    EXPECT_FALSE(bounded_points_identical(center, changed));
    const std::array altered{changed, points[0], points[1], points[2]};
    EXPECT_GT(std::abs(bounded_orientation(altered).value), 0);
    const std::array constraints{BoundedFaceConstraint{finite, 0}};
    const auto departure = bounded_backtrace_constrained(points[0], {1, 2, 3}, .01, constraints);
    EXPECT_EQ(departure.value[0], points[0].value[0]);
    EXPECT_FALSE(departure.supports().empty());
    const std::array wrong{BoundedFaceConstraint{finite, 1}};
    EXPECT_THROW((void) bounded_backtrace_constrained(points[0], {1, 2, 3}, .01, wrong), std::invalid_argument);
}

TEST(BoundedConvexGeometryTest, NumericalPacketsRequireMatchingPinnedSourceRegistry)
{
    const auto source = ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1});
    const auto finite = BoundedConvexGeometry::from_geometry(*source);
    const auto cut = finite->clip({{3, 2, 1}, {}, 2.25}, false);
    ASSERT_TRUE(cut.inside);
    const BoundedGeometryPacketContext context{{17, 2, 3, 11}};
    const auto packet = cut.inside->encode(context);
    const std::array registry{finite};
    const auto decoded = BoundedConvexGeometry::decode(packet, context, registry);
    EXPECT_LE(std::abs(decoded->volume().value - cut.inside->volume().value),
        decoded->volume().absolute_error + cut.inside->volume().absolute_error);
    auto stale = context;
    ++stale.publication[3];
    EXPECT_THROW((void) BoundedConvexGeometry::decode(packet, stale, registry), std::invalid_argument);
    EXPECT_THROW((void) BoundedConvexGeometry::decode(packet, context, {}), std::invalid_argument);
    const auto foreign_source = ConvexCellGeometry::box({2, 2, 2}, {3, 3, 3});
    const std::array foreign{BoundedConvexGeometry::from_geometry(*foreign_source)};
    EXPECT_THROW((void) BoundedConvexGeometry::decode(packet, context, foreign), std::invalid_argument);
    auto damaged = packet;
    damaged.pop_back();
    EXPECT_THROW((void) BoundedConvexGeometry::decode(damaged, context, registry), std::invalid_argument);
    auto limits = BoundedGeometryLimits{};
    limits.byte_budget = packet.size() - 1;
    EXPECT_THROW((void) BoundedConvexGeometry::decode(packet, context, registry, limits), std::invalid_argument);
    auto trailing = packet;
    trailing.push_back(0);
    EXPECT_THROW((void) BoundedConvexGeometry::decode(trailing, context, registry), std::invalid_argument);
}

TEST(BoundedConvexGeometryTest, NonbinaryEdgeParameterRetainsExactRepresentableIntersection)
{
    const auto parent = BoundedConvexGeometry::box({0, 0, 0}, {3, 1, 1});
    const auto cut = parent->clip({{1, 0, 0}, {}, 1}, false);
    ASSERT_TRUE(cut.inside);
    for (const auto& p : cut.inside->points())
        EXPECT_EQ(p.absolute_error[0], 0);
    const auto oblique = cut.inside->clip({{1, 0, -1}, {}, 0}, false);
    ASSERT_TRUE(oblique.inside);
    encloses(oblique.inside->volume(), .5L);
}

TEST(BoundedConvexGeometryTest, TangentSectionComposesAtomicFacesAndRetainsCollinearEdges)
{
    std::vector<GeometryPoint> vertices;
    for (unsigned i = 0; i < 8; ++i)
        vertices.push_back({double(i & 1), double((i >> 1) & 1), double((i >> 2) & 1)});
    const Faces tiled{{0, 4, 6}, {0, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}};
    const auto atomic = BoundedConvexGeometry::make(vertices, tiled);
    const auto contact = atomic->clip({{1, 0, 0}, {}, 0});
    EXPECT_FALSE(contact.inside);
    encloses(contact.section.area, 1);
    EXPECT_EQ(contact.section.vertices.size(), 4U);
    vertices.push_back({0, 0, .5});
    const Faces collinear{{0, 8, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4, 8}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}};
    const auto propagated = BoundedConvexGeometry::make(vertices, collinear);
    const auto edge = propagated->clip({{1, 1, 0}, {}, 0});
    EXPECT_FALSE(edge.inside);
    EXPECT_EQ(edge.section.vertices.size(), 3U);
    EXPECT_EQ(edge.section.area.value, 0);
    EXPECT_EQ(edge.section.area.absolute_error, 0);
    const auto split = propagated->clip({{1, -1, 0}, {}, 0});
    ASSERT_TRUE(split.inside);
    ASSERT_TRUE(split.outside);
    encloses(split.inside->volume(), .5L);
    encloses(split.outside->volume(), .5L);
    encloses(split.section.area, std::sqrt(2.L));
    const std::array pieces{split.inside, split.outside};
    EXPECT_NO_THROW(propagated->certify_partition(pieces, 8e-15));
}

TEST(BoundedConvexGeometryTest, RawShellCannotDoubleCountDisconnectedCoincidentComponents)
{
    std::vector<GeometryPoint> vertices;
    for (unsigned i = 0; i < 8; ++i)
        vertices.push_back({double(i & 1), double((i >> 1) & 1), double((i >> 2) & 1)});
    const Faces cube_faces{{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}};
    auto unused = vertices;
    unused.push_back({.5, .5, .5});
    EXPECT_THROW((void) BoundedConvexGeometry::make(unused, cube_faces), std::invalid_argument);
    auto duplicated = vertices;
    duplicated.insert(duplicated.end(), vertices.begin(), vertices.end());
    auto faces = cube_faces;
    for (auto face : cube_faces)
    {
        for (auto& index : face)
            index += 8;
        faces.push_back(std::move(face));
    }
    EXPECT_THROW((void) BoundedConvexGeometry::make(duplicated, faces), std::invalid_argument);
}

TEST(BoundedConvexGeometryTest, DecodedNonbinaryComplementsRetainOriginalPlaneIncidence)
{
    const auto source = ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1});
    const auto parent = BoundedConvexGeometry::from_geometry(*source);
    const auto split = parent->clip({{3, 0, 0}, {}, 1}, false);
    ASSERT_TRUE(split.inside);
    ASSERT_TRUE(split.outside);
    const BoundedGeometryPacketContext context{{9, 8, 7, 6}};
    const std::array registry{parent};
    const auto inside = BoundedConvexGeometry::decode(split.inside->encode(context), context, registry);
    const auto outside = BoundedConvexGeometry::decode(split.outside->encode(context), context, registry);
    EXPECT_FALSE(inside->intersect(*outside));
    const std::array pieces{inside, outside};
    EXPECT_NO_THROW(parent->certify_partition(pieces, 8e-15));
}

TEST(BoundedConvexGeometryTest, HangingSourcePointCarriesTheCompleteIncidentFanSupports)
{
    const auto coarse = ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1});
    const auto fine = ConvexCellGeometry::box({0, 0, 0}, {.5, .5, .5});
    std::vector<char> point_identity;
    for (const auto& identity : fine->face_geometry(4)->vertex_identities())
    {
        const auto p = exact_point_display(identity);
        if (p == GeometryPoint{.5, .5, 0})
            point_identity = identity;
    }
    ASSERT_FALSE(point_identity.empty());
    const auto coarse_only = bounded_source_point(*coarse, point_identity);
    const std::array sources{coarse, fine};
    const auto complete = bounded_source_point(sources, point_identity);
    EXPECT_GT(complete.supports().size(), coarse_only.supports().size());
    EXPECT_EQ(complete.value, coarse_only.value);
    EXPECT_EQ(complete.absolute_error, coarse_only.absolute_error);
    EXPECT_THROW(
        (void) bounded_source_point(std::span<const ConvexGeometryPtr>{}, point_identity), std::invalid_argument);
}

TEST(BoundedConvexGeometryTest, StructuralProofRejectsAncestorOverlapAndChargesItsWork)
{
    const auto parent = cube();
    const auto split = parent->clip({{1, 0, 0}, {}, .5}, false);
    const auto nested = split.inside->clip({{0, 1, 0}, {}, .5}, false);
    const std::array overlap{split.inside, nested.inside, nested.outside, split.outside};
    EXPECT_THROW(parent->certify_partition(overlap, 8e-15), std::invalid_argument);
    auto limits = BoundedGeometryLimits{};
    limits.maximum_pair_tests = 2;
    const auto limited = BoundedConvexGeometry::box({0, 0, 0}, {1, 1, 1}, limits);
    const auto pieces = limited->clip({{1, 0, 0}, {}, .5}, false);
    const std::array complete{pieces.inside, pieces.outside};
    EXPECT_THROW(limited->certify_partition(complete, 8e-15), std::invalid_argument);
}

TEST(BoundedConvexGeometryTest, RepeatedAuthenticatedNonbinaryWallSupportDoesNotCreateANewRank)
{
    const auto source = ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1})->clip({{3, 2, 0}, {}, 1}).inside;
    ASSERT_TRUE(source);
    std::size_t cut_face = source->num_faces();
    for (std::size_t f = 0; f < source->num_faces(); ++f)
    {
        const auto normal = source->face_geometry(f)->area_vector();
        if (normal.x > 0 && normal.y > 0)
            cut_face = f;
    }
    ASSERT_LT(cut_face, source->num_faces());
    const auto finite = BoundedConvexGeometry::from_geometry(*source);
    const auto identity = source->face_geometry(cut_face)->vertex_identities().front();
    const auto point = bounded_source_point(*source, identity);
    const std::array single{BoundedFaceConstraint{finite, cut_face}};
    const std::array repeated{single[0], single[0], single[0], single[0]};
    const auto expected = bounded_backtrace_constrained(point, {.4, .7, .2}, .03, single);
    const auto actual = bounded_backtrace_constrained(point, {.4, .7, .2}, .03, repeated);
    EXPECT_EQ(actual.value, expected.value);
    EXPECT_EQ(actual.absolute_error, expected.absolute_error);
    EXPECT_EQ(actual.supports(), expected.supports());
    const auto stationary = bounded_backtrace_constrained(point, {0, 0, 0}, .03, repeated);
    EXPECT_TRUE(bounded_points_identical(point, stationary));
    EXPECT_EQ(stationary.supports(), point.supports());
    EXPECT_EQ(stationary.value, point.value);
    EXPECT_EQ(stationary.absolute_error, point.absolute_error);
}

TEST(BoundedConvexGeometryTest, AllAuthenticatedCornerWallsRemainSatisfiedAfterProvenRankThree)
{
    const std::array<GeometryPoint, 6> vertices{{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};
    const Faces faces{{4, 0, 2}, {4, 2, 1}, {4, 1, 3}, {4, 3, 0}, {5, 2, 0}, {5, 1, 2}, {5, 3, 1}, {5, 0, 3}};
    const auto source = ConvexCellGeometry::make(vertices, faces);
    const auto finite = BoundedConvexGeometry::from_geometry(*source);
    std::vector<char> identity;
    for (const auto& point : source->face_geometry(0)->vertex_identities())
        if (exact_point_display(point) == GeometryPoint{0, 0, 1})
            identity = point;
    ASSERT_FALSE(identity.empty());
    std::vector<BoundedFaceConstraint> walls;
    for (std::size_t f = 0; f < source->num_faces(); ++f)
        if (source->face_geometry(f)->contains_exact_point(identity))
            walls.push_back({finite, f});
    ASSERT_EQ(walls.size(), 4u);
    const auto point = bounded_source_point(*source, identity);
    const auto departure = bounded_backtrace_constrained(point, {.1, .2, .3}, .01, walls);
    EXPECT_TRUE(bounded_points_identical(point, departure));
    EXPECT_EQ(departure.value, point.value);
    EXPECT_EQ(departure.absolute_error, point.absolute_error);
}

} // namespace
