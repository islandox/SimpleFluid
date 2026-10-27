/** @file testConvexCellGeometry.cc
 * @brief Independent convex clipping, represented coverage and ancestry oracles.
 */
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/GeometryInstrumentation.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/swept_volume/SweptGeometry.hh"
#include "geometry/mesh/AMR/AdaptiveOverlap.hh"
#include "utils/testing_environment.hh"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace
{
using namespace SimpleFluid::Meshes;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using Faces = std::vector<std::vector<std::size_t>>;
const Faces cube_faces{{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}};

SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Exact> cube()
{
    return SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::box({0, 0, 0}, {1, 1, 1});
}

void near_value(const SimpleFluid::PLIC::GeometryValue<SimpleFluid::ArithmeticMode::Exact>& actual,
    long double expected, long double relative = 2e-17L)
{
    EXPECT_LE(std::abs(actual.value - expected), std::abs(expected) * relative + actual.absolute_error);
    EXPECT_GE(actual.absolute_error, 0);
}

template<SimpleFluid::ArithmeticMode Mode>
::SimpleFluid::Meshes::AMR::HierarchyObservation observation(const SimpleFluid::PLIC::ConvexGeometryPtr<Mode>& geometry)
{
    std::shared_ptr<const void> identity = geometry;
    return {identity, identity, identity, 0, 0, 0, geometry.get(), nullptr};
}

::SimpleFluid::Meshes::AMR::AdaptiveCellKey key(std::initializer_list<std::uint64_t> path = {})
{
    return {7, ::SimpleFluid::Meshes::AMR::ConvexCellAncestry{19, path}};
}

TEST(ConvexCellGeometryTest, CubeObliqueTetrahedronAndIndependentComplementMoments)
{
    auto parent = cube();
    const auto bulk = parent->moments();
    near_value(bulk.volume, 1);
    for (const auto& first : bulk.first_moment)
        near_value(first, 0.5L);
    auto cut = parent->clip({{1, 1, 1}, {1, 0, 0}});
    ASSERT_TRUE(cut.inside);
    ASSERT_TRUE(cut.outside);
    near_value(cut.inside->moments().volume, 1.0L / 6);
    near_value(cut.outside->moments().volume, 5.0L / 6);
    for (std::size_t a = 0; a < 3; ++a)
    {
        near_value(cut.inside->moments().first_moment[a], 1.0L / 24);
        near_value(cut.outside->moments().first_moment[a], 11.0L / 24);
    }
    near_value(cut.section.area, std::sqrt(3.0L) / 2);
    EXPECT_EQ(cut.section.vertices.size(), 3U);
    const std::array pieces{cut.inside, cut.outside};
    EXPECT_NO_THROW(parent->certify_partition(pieces));
    const std::array<SimpleFluid::PLIC::GeometryPoint, 4> vertices{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    auto tetra = SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::make(
        vertices, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
    EXPECT_TRUE(tetra->equivalent(*cut.inside));
}

TEST(ConvexCellGeometryTest, TriangularPrismAndAffineSkewUseActualShell)
{
    const std::array<SimpleFluid::PLIC::GeometryPoint, 6> prism{
        {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 2}, {1, 0, 2}, {0, 1, 2}}};
    auto geometry = SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::make(
        prism, {{0, 2, 1}, {3, 4, 5}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}});
    auto m = geometry->moments();
    near_value(m.volume, 1);
    near_value(m.first_moment[0], 1.0L / 3);
    near_value(m.first_moment[1], 1.0L / 3);
    near_value(m.first_moment[2], 1);
    std::array<SimpleFluid::PLIC::GeometryPoint, 8> skew;
    for (std::size_t i = 0; i < 8; ++i)
    {
        const double x = i & 1, y = (i >> 1) & 1, z = (i >> 2) & 1;
        skew[i] = {x + .5 * y + .25 * z, y + .25 * z, z};
    }
    auto sheared = SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::make(skew, cube_faces);
    near_value(sheared->moments().volume, 1);
    auto cut = sheared->clip({{1, -.5, -.125}, {.25, 0, 0}});
    ASSERT_TRUE(cut.inside);
    const auto small = cut.inside->moments();
    near_value(small.volume, .25L);
    near_value(small.first_moment[0], .125L);
    near_value(small.first_moment[1], .15625L);
    near_value(small.first_moment[2], .125L);
    EXPECT_LT(small.volume.value, 1); // AABB replacement would give a different domain.
}

TEST(ConvexCellGeometryTest, ProductBulkIntersectionAndTiltedPrismCutHaveIndependentOracles)
{
    const std::array<std::array<double, 2>, 3> triangle{{{0, 0}, {2, 0}, {0, 2}}};
    auto prism =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::extruded_polygon(triangle, 0, 2);
    auto inner = SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::box(
        {.25, .25, .5}, {1.25, 1.25, 1.5});
    auto overlap = prism->intersect(*inner);
    ASSERT_TRUE(overlap);
    const auto m = overlap->moments();
    near_value(m.volume, 7.0L / 8);
    near_value(m.first_moment[0], 59.0L / 96);
    near_value(m.first_moment[1], 59.0L / 96);
    near_value(m.first_moment[2], 7.0L / 8);
    const std::array<std::array<double, 2>, 3> unit_triangle{{{0, 0}, {1, 0}, {0, 1}}};
    auto unit = SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::extruded_polygon(
        unit_triangle, 0, 1);
    auto tilted = unit->clip({{1, 0, 1}, {.5, 0, 0}});
    ASSERT_TRUE(tilted.inside);
    ASSERT_TRUE(tilted.outside);
    near_value(tilted.inside->moments().volume, 5.0L / 48);
    near_value(tilted.inside->moments().first_moment[0], 1.0L / 64);
    const std::array pieces{tilted.inside, tilted.outside};
    EXPECT_NO_THROW(unit->certify_partition(pieces));
}

TEST(ConvexCellGeometryTest, NativePrismExportCorrectsBottomCapOrientation)
{
    using namespace SimpleFluid;
    auto provider = std::make_shared<SemiStructuredXY_Z>(Arr<MeshUtils::Vec3>{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}},
        Arr<Arr<unsigned>>{{0, 1, 2}}, ArrReal{0, 1},
        Arr<SemiStructuredXY_Z::BoundaryEdge>{{0, 1, "bottom"}, {1, 2, "slope"}, {2, 0, "left"}});
    MultiRegionMesh native(std::vector<MultiRegionMesh::Region>{native_region("prism", provider)},
        std::vector<MultiRegionMesh::Interface>{});
    const auto shell =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::native_cell_shell(native, 0);
    auto rebuilt = SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::make(
        shell.vertices, shell.outward_faces);
    const std::array<std::array<double, 2>, 3> base{{{0, 0}, {1, 0}, {0, 1}}};
    EXPECT_TRUE(rebuilt->equivalent(
        *SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::extruded_polygon(base, 0, 1)));
    EXPECT_TRUE(rebuilt->equivalent(
        *SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::from_native_cell(native, 0)));
    unsigned bottom_caps = 0;
    for (const auto& face : shell.outward_faces)
        if (std::all_of(face.begin(), face.end(), [&](auto i) { return shell.vertices[i][2] == 0; }))
        {
            ++bottom_caps;
            const auto& a = shell.vertices[face[0]];
            const auto& b = shell.vertices[face[1]];
            const auto& c = shell.vertices[face[2]];
            EXPECT_LT((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]), 0);
        }
    EXPECT_EQ(bottom_caps, 1U);
}

TEST(ConvexCellGeometryTest, FaceEdgePointTangenciesKeepTheirGeometricDimension)
{
    const auto parent = cube();
    const auto face = parent->clip({{1, 0, 0}, {0, 0, 0}});
    EXPECT_FALSE(face.inside);
    ASSERT_TRUE(face.outside);
    near_value(face.section.area, 1);
    EXPECT_EQ(face.section.vertices.size(), 4U);
    near_value(face.inside_apertures[0], 1);
    const auto edge = parent->clip({{1, 1, 0}, {0, 0, 0}});
    EXPECT_FALSE(edge.inside);
    EXPECT_EQ(edge.section.area.value, 0);
    EXPECT_EQ(edge.section.vertices.size(), 2U);
    const auto vertex = parent->clip({{1, 1, 1}, {0, 0, 0}});
    EXPECT_FALSE(vertex.inside);
    EXPECT_EQ(vertex.section.area.value, 0);
    EXPECT_EQ(vertex.section.vertices.size(), 1U);
    EXPECT_THROW((void) parent->clip({{0, 0, 0}, {0, 0, 0}}), std::invalid_argument);
}

TEST(ConvexCellGeometryTest, ExactSequentialConstructionsPreserveMicroscopicPositivePieces)
{
    const auto parent = cube();
    const double tiny = 1e-200;
    const auto cut = parent->clip({{1, 0, 0}, {tiny, 0, 0}});
    ASSERT_TRUE(cut.inside);
    ASSERT_TRUE(cut.outside);
    near_value(cut.inside->moments().volume, static_cast<long double>(tiny));
    const auto moment = cut.inside->moments().first_moment[0];
    EXPECT_TRUE(moment.exact_nonzero);
    EXPECT_GT(moment.value + moment.absolute_error, 0);
    EXPECT_THROW((void) cut.inside->moments().first_moment[0].as_double(), std::runtime_error);
    const auto twice = cut.inside->clip({{0, 1, 0}, {0, .25, 0}});
    ASSERT_TRUE(twice.inside);
    near_value(twice.inside->moments().volume, static_cast<long double>(tiny) / 4);
    // Rational intersection 1/3 is retained exactly through later cuts.
    const auto third = parent->clip({{3, 0, 0}, {0, 0, 0}}); // supporting contact
    EXPECT_FALSE(third.inside);
    auto oblique = parent->clip({{3, 1, 0}, {.25, .25, 0}});
    ASSERT_TRUE(oblique.inside);
    auto a = oblique.inside->clip({{0, 0, 1}, {0, 0, .5}}).inside;
    auto b = parent->clip({{0, 0, 1}, {0, 0, .5}}).inside->clip({{3, 1, 0}, {.25, .25, 0}}).inside;
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    EXPECT_TRUE(a->equivalent(*b));
}

TEST(ConvexCellGeometryTest, Binary64MomentIntervalsRetainResolvedTinyVolume)
{
    const auto parent = cube();
    const auto cut = parent->clip({{1, 0, 0}, {1e-200, 0, 0}});
    ASSERT_TRUE(cut.inside);
    const auto native = cut.inside->moments();
    const auto binary64 = cut.inside->moments_binary64();
    EXPECT_GT(binary64.volume.as_double(), 0);
    EXPECT_EQ(binary64.first_moment[0].value, 0);
    EXPECT_TRUE(binary64.first_moment[0].exact_nonzero);
    EXPECT_EQ(
        binary64.first_moment[0].absolute_error, static_cast<long double>(std::numeric_limits<double>::denorm_min()));
    EXPECT_LE(std::abs(native.first_moment[0].value), binary64.first_moment[0].absolute_error);
    EXPECT_THROW((void) binary64.first_moment[0].as_double(), std::runtime_error);
    const std::array pieces{cut.inside, cut.outside};
    EXPECT_NO_THROW(parent->certify_partition(pieces));
}

TEST(ConvexCellGeometryTest, AnchoredScalarOffsetPreservesTranslatedMicroscopicCuts)
{
    constexpr double origin = 1e6, tiny = 1e-200;
    const auto translated = SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::box(
        {origin, origin, origin}, {origin + 1, origin + 1, origin + 1});
    EXPECT_EQ(origin + tiny, origin); // A global point on this plane cannot encode the cut.
    const auto cut = translated->clip({{1, 0, 0}, {origin, origin, origin}, tiny});
    ASSERT_TRUE(cut.inside);
    ASSERT_TRUE(cut.outside);
    const auto local = cube()->clip({{1, 0, 0}, {0, 0, 0}, tiny});
    const auto actual = cut.inside->moments(), expected = local.inside->moments();
    near_value(actual.volume, expected.volume.value);
    EXPECT_EQ(actual.origin, (SimpleFluid::PLIC::GeometryPoint{origin, origin, origin}));
    for (std::size_t a = 0; a < 3; ++a)
    {
        near_value(actual.first_moment[a], expected.first_moment[a].value);
        EXPECT_EQ(actual.first_moment[a].exact_nonzero, expected.first_moment[a].exact_nonzero);
    }
    const std::array pieces{cut.inside, cut.outside};
    EXPECT_NO_THROW(translated->certify_partition(pieces));
    const auto reverse = translated->clip({{-1, 0, 0}, {origin, origin, origin}, -tiny});
    ASSERT_TRUE(reverse.inside);
    ASSERT_TRUE(reverse.outside);
    EXPECT_TRUE(reverse.inside->equivalent(*cut.outside));
    EXPECT_TRUE(reverse.outside->equivalent(*cut.inside));
    EXPECT_THROW(
        (void) translated->clip({{1, 0, 0}, {origin, origin, origin}, std::numeric_limits<double>::infinity()}),
        std::invalid_argument);
}

TEST(ConvexCellGeometryTest, MomentOverflowRejectsRatherThanLosingExactMeasure)
{
    // Volume is about 1e300, but a local first moment is about 5e399.
    // The admitted output contract rejects overflow, independently of whether
    // this platform has an extended-range native long double.
    if (std::numeric_limits<long double>::max_exponent > 1400)
    {
        const auto large = SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::box(
            {0, 0, 0}, {1e100, 1e100, 1e100});
        EXPECT_GT(large->moments().volume.value, 0);
        EXPECT_THROW((void) large->moments_binary64(), std::runtime_error);
    }
    else
    {
        EXPECT_THROW((void) SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::box(
                         {0, 0, 0}, {1e100, 1e100, 1e100}),
            std::runtime_error);
    }
}

TEST(ConvexCellGeometryTest, ScaleTranslationAndActualGapOverlapCertificates)
{
    for (const auto [origin, size] : std::array<std::pair<double, double>, 3>{{{0, 1e-100}, {0, 1e75}, {1e12, .125}}})
    {
        auto parent = SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::box(
            {origin, origin, origin}, {origin + size, origin + size, origin + size});
        const auto span = static_cast<long double>(origin + size) - origin;
        near_value(parent->moments().volume, span * span * span, 1e-16L);
        const double mid = origin + size / 2;
        auto cut = parent->clip({{1, 0, 0}, {mid, origin, origin}});
        ASSERT_TRUE(cut.inside);
        ASSERT_TRUE(cut.outside);
        const std::array pieces{cut.inside, cut.outside};
        EXPECT_NO_THROW(parent->certify_partition(pieces));
        const auto gap = std::nextafter(mid, std::numeric_limits<double>::infinity());
        auto right = parent->clip({{-1, 0, 0}, {gap, origin, origin}}).inside;
        const std::array missing{cut.inside, right};
        EXPECT_THROW(parent->certify_partition(missing), std::invalid_argument);
        auto overlap =
            parent->clip({{-1, 0, 0}, {std::nextafter(mid, -std::numeric_limits<double>::infinity()), origin, origin}})
                .inside;
        const std::array doubled{cut.inside, overlap};
        EXPECT_THROW(parent->certify_partition(doubled), std::invalid_argument);
    }
}

TEST(ConvexCellGeometryTest, RejectsInvalidShellsButRetainsCollinearEdgesAndAtomicFaces)
{
    std::vector<SimpleFluid::PLIC::GeometryPoint> v{
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
    auto faces = cube_faces;
    std::reverse(faces[0].begin(), faces[0].end());
    EXPECT_THROW((void) SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::make(v, faces),
        std::invalid_argument);
    faces = cube_faces;
    faces.pop_back();
    EXPECT_THROW((void) SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::make(v, faces),
        std::invalid_argument);
    faces = cube_faces;
    faces[0] = {0, 6, 4, 2};
    EXPECT_THROW((void) SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::make(v, faces),
        std::invalid_argument);
    faces = cube_faces;
    v[7][2] = .75;
    EXPECT_THROW((void) SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::make(v, faces),
        std::invalid_argument);
    v[7][2] = 1;
    v.push_back({.5, 0, 0});
    faces = {{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 8, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1, 8}, {4, 5, 7, 6}};
    EXPECT_NO_THROW((void) SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::make(v, faces));
    // Coplanar atomic top triangles share one canonical internal diagonal.
    faces.back() = {4, 5, 7};
    faces.push_back({4, 7, 6});
    auto tiled = SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::make(v, faces);
    EXPECT_TRUE(tiled->equivalent(*cube()));
    EXPECT_NO_THROW((void) tiled->clip({{1, 0, 0}, {.5, 0, 0}}));
    // One coarse loop omits the fine edge point; native adaptation propagates
    // it exactly into all incident loops before the strict manifold check.
    faces[4] = {0, 2, 3, 1};
    EXPECT_THROW((void) SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::make(v, faces),
        std::invalid_argument);
    auto propagated =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::from_native_shell(v, faces);
    EXPECT_TRUE(propagated->equivalent(*cube()));
}

TEST(AdaptiveOverlapTest, TypedRefinementCoarseningAndExactSourceObservation)
{
    auto parent =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box({0, 0, 0}, {1, 1, 1});
    auto split = parent->clip({{1, 0, 0}, {.5, 0, 0}});
    const auto source = observation(parent), target = observation(split.inside);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> old{{key(), 17, 3, parent}};
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> children{
        {key({0}), 900, 3, split.inside}, {key({1}), 12, 3, split.outside}};
    const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation> relations{
        {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Refinement, 17, 900},
        {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Refinement, 17, 12}};
    auto plan = ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(source, target, old, children, relations);
    ASSERT_EQ(plan->records().size(), 2U);
    EXPECT_EQ(plan->records()[0].moments.volume.value, .5L);
    EXPECT_EQ(plan->records()[0].source_key, key());
    auto stale = source;
    ++stale.geometry;
    EXPECT_THROW(plan->validate_observations(stale, target), std::invalid_argument);
    EXPECT_NO_THROW(plan->validate_observations(source, target));
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation> reverse{
        {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Coarsening, 900, 17},
        {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Coarsening, 12, 17}};
    EXPECT_NO_THROW(
        (void) ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(target, source, children, old, reverse));
    children[1].geometry = split.inside;
    EXPECT_THROW(
        (void) ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(source, target, old, children, relations),
        std::invalid_argument);
    children[1].geometry = split.outside;
    children[1].material = 9;
    EXPECT_THROW(
        (void) ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(source, target, old, children, relations),
        std::invalid_argument);
    children[1].material = 3;
    children[1].key = key({1, 0});
    const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation> false_same{
        {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Unchanged, 17, 900},
        {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Unchanged, 17, 12}};
    EXPECT_THROW(
        (void) ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(source, target, old, children, false_same),
        std::invalid_argument);
}

TEST(AdaptiveOverlapTest, ActiveCatalogAncestryMustBeALeafAntichain)
{
    auto parent =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box({0, 0, 0}, {1, 1, 1});
    const auto split = parent->clip({{1, 0, 0}, {.5, 0, 0}});
    const auto obs = observation(parent);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> root{{key(), 0, 0, parent}};
    // These shells partition the root exactly, but their claimed active keys
    // overlap genealogically. Pure geometric coverage must not admit them.
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> invalid{
        {key({0}), 1, 0, split.inside}, {key({0, 1}), 2, 0, split.outside}};
    const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation> forward{
        {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Refinement, 0, 1},
        {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Refinement, 0, 2}};
    EXPECT_THROW((void) ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(obs, obs, root, invalid, forward),
        std::invalid_argument);
    const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation> reverse{
        {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Coarsening, 1, 0},
        {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Coarsening, 2, 0}};
    EXPECT_THROW((void) ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(obs, obs, invalid, root, reverse),
        std::invalid_argument);
    // Exercise the independent Cartesian bounded-parent walk as well.
    root[0].key = {7, ::SimpleFluid::Meshes::AMR::CartesianCellAncestry{0, {0, 0, 0}}};
    invalid[0].key = {7, ::SimpleFluid::Meshes::AMR::CartesianCellAncestry{1, {0, 0, 0}}};
    invalid[1].key = {7, ::SimpleFluid::Meshes::AMR::CartesianCellAncestry{2, {1, 0, 0}}};
    EXPECT_THROW((void) ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(obs, obs, root, invalid, forward),
        std::invalid_argument);
    root[0].key = key();
    invalid[0].key = key({0});
    invalid[1].key = key({1});
    EXPECT_NO_THROW((void) ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(obs, obs, root, invalid, forward));
}
} // namespace

namespace
{
TEST(ConvexCellGeometryTest, OptInExactStatisticsCountKernelEventsAndResetWithoutChangingEnablement)
{
    SimpleFluid::PLIC::enable_backend_statistics<SimpleFluid::ArithmeticMode::Exact>(false);
    SimpleFluid::PLIC::reset_backend_statistics<SimpleFluid::ArithmeticMode::Exact>();
    const auto ignored =
        SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::from_value(7).scaled(2);
    (void) ignored;
    EXPECT_EQ(SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>().measure_values, 0);
    EXPECT_EQ(SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>().measure_arithmetic, 0);
    struct DisableOnExit
    {
        ~DisableOnExit() { SimpleFluid::PLIC::enable_backend_statistics<SimpleFluid::ArithmeticMode::Exact>(false); }
    } disable;
    SimpleFluid::PLIC::enable_backend_statistics<SimpleFluid::ArithmeticMode::Exact>(true);
    SimpleFluid::PLIC::reset_backend_statistics<SimpleFluid::ArithmeticMode::Exact>();
    const auto a = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::from_value(1),
               b = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::from_value(3);
    const auto third = a.divided_by(b);
    EXPECT_GT(SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>().measure_values, 0);
    EXPECT_GT(SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>().measure_arithmetic, 0);
    EXPECT_GE(
        SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>().maximum_result_denominator_bits, 2);
    (void) third;
    SimpleFluid::PLIC::reset_backend_statistics<SimpleFluid::ArithmeticMode::Exact>();
    const auto arithmetic = a + b;
    (void) arithmetic;
    EXPECT_EQ(SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>().measure_arithmetic, 1);
    const std::array<SimpleFluid::PLIC::GeometryPoint, 4> vertices{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    const auto tetra = SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::make(
        vertices, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
    const auto before_tetra_moments = SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>();
    (void) tetra->moments();
    EXPECT_GT(SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>().moment_evaluations,
        before_tetra_moments.moment_evaluations);
    const auto before_tetra_volume = SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>();
    (void) SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::volume(*tetra);
    EXPECT_GT(SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>().volume_evaluations,
        before_tetra_volume.volume_evaluations);
    const auto product = SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::extruded_polygon(
        std::array<std::array<double, 2>, 3>{{{0, 0}, {1, 0}, {0, 1}}}, 0, 1);
    const auto before_product = SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>();
    (void) product->moments();
    EXPECT_GT(SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>().moment_evaluations,
        before_product.moment_evaluations);
    const auto face = SimpleFluid::PLIC::PlanarFace<SimpleFluid::ArithmeticMode::Exact>::make(
        std::array<SimpleFluid::PLIC::GeometryPoint, 3>{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}});
    const auto before_face = SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>();
    (void) face->area();
    (void) face->area_vector();
    (void) face->centroid();
    EXPECT_EQ(SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>().face_measure_evaluations,
        before_face.face_measure_evaluations + 3);
    EXPECT_TRUE(face->contains_point(face->vertex_identities().front()));
    EXPECT_GT(SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>().source_membership_predicates,
        before_face.source_membership_predicates);
    const auto counted = SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>();
    EXPECT_GT(counted.moment_evaluations, 0);
    EXPECT_GT(counted.volume_evaluations, 0);
    EXPECT_GT(counted.maximum_result_numerator_bits, 0);
    SimpleFluid::PLIC::enable_backend_statistics<SimpleFluid::ArithmeticMode::Exact>(false);
    (void) a.scaled(4);
    (void) tetra->moments();
    const auto disabled = SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>();
    EXPECT_EQ(disabled.measure_values, counted.measure_values);
    EXPECT_EQ(disabled.measure_arithmetic, counted.measure_arithmetic);
    EXPECT_EQ(disabled.moment_evaluations, counted.moment_evaluations);
    SimpleFluid::PLIC::reset_backend_statistics<SimpleFluid::ArithmeticMode::Exact>();
    EXPECT_EQ(SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>().measure_values, 0);
    EXPECT_EQ(SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>().moment_evaluations, 0);
}
} // namespace
