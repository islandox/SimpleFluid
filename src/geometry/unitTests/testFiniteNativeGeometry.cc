/** @file testFiniteNativeGeometry.cc
 * @brief Ordinary native topology and geometry without rational source preparation.
 */
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/GeometryInstrumentation.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/GeometryMeasure.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/PlaneMeasure.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/PLIC/details/ordinary/NativeGeometrySupport.hh"
#include "geometry/PLIC/details/ordinary/NativeSupportAccess.hh"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <set>

namespace
{
using SimpleFluid::ArithmeticMode;
namespace PLIC = SimpleFluid::PLIC;
constexpr auto ordinary = ArithmeticMode::Ordinary;
using Cell = PLIC::ConvexCellGeometry<ordinary>;
using Face = PLIC::PlanarFace<ordinary>;
using PointIdentity = PLIC::PointIdentity<ordinary>;

TEST(FiniteNativeGeometryTest, NativeShellAndCellPacketUseOrdinaryGeometry)
{
    const std::array<PLIC::GeometryPoint, 8> vertices{
        {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}}};
    const std::vector<std::vector<std::size_t>> faces{
        {0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}};
    const auto cell = Cell::from_native_shell(vertices, faces);
    EXPECT_EQ(cell->arithmetic_mode(), ordinary);
    EXPECT_DOUBLE_EQ(cell->volume().as_double(), 1);
    EXPECT_EQ(cell->num_faces(), 6u);
    const auto source = cell->numerical_shell();
    EXPECT_EQ(source.relative_vertices.size(), 8u);
    EXPECT_EQ(source.face_support_identities.size(), 6u);
    const auto restored = Cell::decode_certificate(cell->encode_certificate());
    EXPECT_TRUE(cell->equivalent(*restored));
    EXPECT_DOUBLE_EQ(restored->volume().as_double(), 1);
}
TEST(FiniteNativeGeometryTest, MidpointSplitRetainsVolumeAndFaceAncestry)
{
    const auto cell = Cell::box({0, 0, 0}, {1, 1, 1});
    const auto pieces = cell->split_aabb_midpoints();
    ASSERT_EQ(pieces.size(), 8u);
    long double sum = 0;
    std::vector<PLIC::ConvexGeometryPtr<ordinary>> children;
    for (const auto& piece : pieces)
    {
        ASSERT_TRUE(piece.geometry);
        EXPECT_EQ(piece.geometry->num_faces(), piece.face_origins.size());
        EXPECT_DOUBLE_EQ(piece.geometry->volume().as_double(), .125);
        EXPECT_TRUE(piece.geometry->quality_above(.01));
        sum += piece.geometry->volume().value;
        children.push_back(piece.geometry);
        const auto restored = Cell::decode_certificate(piece.geometry->encode_certificate());
        EXPECT_TRUE(piece.geometry->equivalent(*restored));
    }
    EXPECT_EQ(sum, 1);
    EXPECT_NO_THROW(cell->certify_partition(children));
}
TEST(FiniteNativeGeometryTest, PlanarOverlayAndPartitionPreserveAnalyticArea)
{
    const std::array<PLIC::GeometryPoint, 4> root{{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}};
    const std::array<PLIC::GeometryPoint, 4> left{{{0, 0, 0}, {.5, 0, 0}, {.5, 1, 0}, {0, 1, 0}}};
    const std::array<PLIC::GeometryPoint, 4> right{{{.5, 0, 0}, {1, 0, 0}, {1, 1, 0}, {.5, 1, 0}}};
    const auto a = Face::make(root), b = Face::make(left), c = Face::make(right);
    EXPECT_DOUBLE_EQ(a->area().as_double(), 1);
    EXPECT_DOUBLE_EQ(b->area().as_double(), .5);
    EXPECT_FALSE(b->intersection(*c));
    EXPECT_TRUE(a->contains(*b));
    EXPECT_TRUE(a->intersection(*b)->equivalent(*b));
    const std::array parts{b, c};
    EXPECT_NO_THROW(a->certify_partition(parts));
    const std::array removed{b};
    const auto remainder = PLIC::subtract_planar_faces<ordinary>(*a, removed);
    ASSERT_EQ(remainder.size(), 1u);
    EXPECT_TRUE(remainder.front()->equivalent(*c));
    EXPECT_EQ(PLIC::planar_support<ordinary>(*a), PLIC::planar_support<ordinary>(*b));
    EXPECT_TRUE(Face::decode(a->encode())->equivalent(*a));
}
TEST(FiniteNativeGeometryTest, PointIdentityAndEdgePropagationRetainFiniteCoordinates)
{
    const std::array<PLIC::GeometryPoint, 4> root{{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}};
    const auto face = Face::make(root);
    const std::array<PLIC::GeometryPoint, 4> half{{{0, 0, 0}, {.5, 0, 0}, {.5, 1, 0}, {0, 1, 0}}};
    const auto points = Face::make(half)->vertex_identities();
    const auto propagated = face->with_edge_vertices(points);
    EXPECT_EQ(propagated->points().size(), 6u);
    EXPECT_TRUE(propagated->equivalent(*face));
    EXPECT_EQ(Face::decode(propagated->encode_incidence())->points().size(), 6u);
    EXPECT_EQ(propagated->encode(), face->encode());
    const auto translated = face->translated({0, 0, 1});
    EXPECT_FALSE(translated->equivalent(*face));
    EXPECT_NE(PLIC::planar_support<ordinary>(*translated), PLIC::planar_support<ordinary>(*face));
    for (const auto& identity : points)
    {
        EXPECT_TRUE(face->contains_point(identity));
        const auto shifted =
            PLIC::point_translated<ordinary>(identity, PLIC::displacement<ordinary>({0, 0, 0}, {1, 0, 0}));
        const auto p = PLIC::point_display<ordinary>(identity), q = PLIC::point_display<ordinary>(shifted);
        EXPECT_EQ(q[0], p[0] + 1);
        EXPECT_EQ(q[1], p[1]);
        EXPECT_EQ(q[2], p[2]);
    }
    PLIC::EdgePointIndex<ordinary> index(propagated->vertex_identities());
    const auto ids = face->vertex_identities();
    EXPECT_EQ(index.closed_segment(ids[0], ids[1]).size(), 3u);
}

std::vector<PointIdentity> edge_query_points()
{
    const std::array<PLIC::GeometryPoint, 7> vertices{
        {{0, 0, 0}, {.25, 0, 0}, {.5, 0, 0}, {.75, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}};
    return Face::make(vertices)->vertex_identities();
}
std::vector<PointIdentity> copy_edge_query(const PLIC::OrdinaryEdgeQueryScratch& scratch)
{
    std::vector<PointIdentity> result;
    for (std::size_t i = 0; i < scratch.size(); ++i)
    {
        result.push_back(scratch.point(i));
    }
    return result;
}
TEST(FiniteNativeGeometryTest, EdgeQueryScratchReusesCapacityAndPreservesOwningResults)
{
    const auto ids = edge_query_points();
    auto input = ids;
    std::reverse(input.begin(), input.end());
    input.push_back(ids[2]);
    PLIC::EdgePointIndex<ordinary> index(input);
    PLIC::OrdinaryEdgeQueryScratch scratch;
    EXPECT_EQ(scratch.storage_bytes(), 0u);
    const auto peak = scratch.reserve_required_peak(input.size());
    scratch.reserve(input.size(), peak);
    EXPECT_EQ(scratch.capacity(), input.size());
    EXPECT_EQ(scratch.storage_bytes(), peak);
    const auto storage = scratch.storage_bytes();
    const std::vector<PointIdentity> forward(ids.begin(), ids.begin() + 5);
    const std::vector<PointIdentity> backward(forward.rbegin(), forward.rend());
    const auto owning = index.closed_segment(ids[0], ids[4]);
    EXPECT_EQ(owning, forward);
    index.closed_segment(ids[0], ids[4], scratch);
    EXPECT_EQ(copy_edge_query(scratch), forward);
    index.closed_segment(ids[4], ids[0], scratch);
    EXPECT_EQ(copy_edge_query(scratch), backward);
    EXPECT_EQ(owning, forward);
    EXPECT_EQ(scratch.capacity(), input.size());
    EXPECT_EQ(scratch.storage_bytes(), storage);
    EXPECT_EQ(scratch.growths(), 1u);
    EXPECT_EQ(scratch.queries(), 2u);
    EXPECT_EQ(scratch.scanned_points(), 2 * ids.size());
    scratch.clear();
    EXPECT_EQ(scratch.size(), 0u);
    EXPECT_EQ(scratch.storage_bytes(), storage);
    EXPECT_EQ(scratch.capacity(), input.size());
    EXPECT_EQ(scratch.queries(), 2u);
    // The scratch pins the immutable index while its borrowed hits are live.
    {
        PLIC::EdgePointIndex<ordinary> temporary(input);
        temporary.closed_segment(ids[0], ids[4], scratch);
    }
    EXPECT_EQ(copy_edge_query(scratch), forward);
    scratch.release();
    EXPECT_EQ(scratch.storage_bytes(), 0u);
    EXPECT_EQ(scratch.capacity(), 0u);
    EXPECT_EQ(scratch.size(), 0u);
    EXPECT_EQ(scratch.growths(), 1u);
    EXPECT_EQ(scratch.queries(), 3u);
    EXPECT_EQ(scratch.scanned_points(), 3 * ids.size());
    EXPECT_EQ(owning, forward);
}
TEST(FiniteNativeGeometryTest, EdgeQueryScratchAdmissionIsAtomicAndFailedQueriesRecover)
{
    const auto ids = edge_query_points();
    PLIC::EdgePointIndex<ordinary> index(ids);
    PLIC::OrdinaryEdgeQueryScratch scratch;
    const auto first_peak = scratch.reserve_required_peak(2);
    EXPECT_THROW(scratch.reserve(2, first_peak - 1), PLIC::GeometryError);
    EXPECT_EQ(scratch.storage_bytes(), 0u);
    EXPECT_EQ(scratch.growths(), 0u);
    scratch.reserve(2, first_peak);
    EXPECT_THROW(index.closed_segment(ids[0], ids[4], scratch), PLIC::GeometryError);
    EXPECT_EQ(scratch.size(), 0u);
    EXPECT_EQ(scratch.scanned_points(), 0u);
    const auto grow_peak = scratch.reserve_required_peak(ids.size());
    EXPECT_GT(grow_peak, first_peak);
    EXPECT_THROW(scratch.reserve(ids.size(), grow_peak - 1), PLIC::GeometryError);
    EXPECT_EQ(scratch.capacity(), 2u);
    EXPECT_EQ(scratch.storage_bytes(), first_peak);
    scratch.reserve(ids.size(), grow_peak);
    EXPECT_EQ(scratch.capacity(), ids.size());
    EXPECT_LT(scratch.storage_bytes(), grow_peak); // Old/new overlap was admitted.
    index.closed_segment(ids[0], ids[4], scratch);
    const auto expected = copy_edge_query(scratch);
    const auto old_storage = scratch.storage_bytes();
    const auto next_peak = scratch.reserve_required_peak(ids.size() + 1);
    EXPECT_THROW(scratch.reserve(ids.size() + 1, next_peak - 1), PLIC::GeometryError);
    EXPECT_EQ(copy_edge_query(scratch), expected);
    EXPECT_EQ(scratch.storage_bytes(), old_storage);
    EXPECT_EQ(scratch.growths(), 2u);
    EXPECT_THROW(scratch.reserve_required_peak(std::numeric_limits<std::size_t>::max()), PLIC::GeometryError);
    const std::vector<char> malformed{'x'};
    EXPECT_THROW(PLIC::decode_identity<PointIdentity>(malformed), std::invalid_argument);
    EXPECT_EQ(copy_edge_query(scratch), expected);
    index.closed_segment(ids[0], ids[4], scratch);
    EXPECT_THROW(index.closed_segment(ids[0], ids[0], scratch), std::invalid_argument);
    EXPECT_EQ(scratch.size(), 0u);
    index.closed_segment(ids[0], ids[4], scratch);
    EXPECT_EQ(copy_edge_query(scratch), expected);
    EXPECT_EQ(index.closed_segment(ids[0], ids[4]), expected);
    EXPECT_EQ(scratch.queries(), 5u);
    EXPECT_EQ(scratch.scanned_points(), 3 * ids.size());
    scratch.release();
    EXPECT_EQ(scratch.storage_bytes(), 0u);
    EXPECT_EQ(scratch.queries(), 5u);
    EXPECT_EQ(scratch.scanned_points(), 3 * ids.size());
    const auto recovered_peak = scratch.reserve_required_peak(ids.size());
    scratch.reserve(ids.size(), recovered_peak);
    index.closed_segment(ids[0], ids[4], scratch);
    EXPECT_EQ(copy_edge_query(scratch), expected);
    EXPECT_EQ(scratch.growths(), 3u);
}
TEST(FiniteNativeGeometryTest, EdgeQueryScratchMovesKeepResultPinAndDiagnosticHistory)
{
    const auto ids = edge_query_points();
    const std::vector<PointIdentity> expected(ids.begin(), ids.begin() + 5);
    const std::vector<PointIdentity> no_points;
    PLIC::EdgePointIndex<ordinary> empty_index(no_points);
    PLIC::OrdinaryEdgeQueryScratch source;
    empty_index.closed_segment(ids[0], ids[4], source);
    EXPECT_EQ(source.size(), 0u);
    EXPECT_EQ(source.storage_bytes(), 0u);
    EXPECT_EQ(source.growths(), 0u);
    EXPECT_EQ(source.queries(), 1u);
    EXPECT_EQ(source.scanned_points(), 0u);
    source.reserve(ids.size(), source.reserve_required_peak(ids.size()));
    {
        PLIC::EdgePointIndex<ordinary> index(ids);
        index.closed_segment(ids[0], ids[4], source);
    }
    const auto bytes = source.storage_bytes();
    PLIC::OrdinaryEdgeQueryScratch moved(std::move(source));
    EXPECT_EQ(source.storage_bytes(), 0u);
    EXPECT_EQ(source.size(), 0u);
    EXPECT_EQ(copy_edge_query(moved), expected);
    EXPECT_EQ(moved.storage_bytes(), bytes);
    EXPECT_EQ(moved.queries(), 2u);
    EXPECT_EQ(moved.scanned_points(), ids.size());
    EXPECT_EQ(moved.growths(), 1u);
    source.release();
    // Assignment replaces an existing buffer while retaining the source pin.
    PLIC::OrdinaryEdgeQueryScratch assigned;
    assigned.reserve(1, assigned.reserve_required_peak(1));
    assigned = std::move(moved);
    moved.release();
    EXPECT_EQ(copy_edge_query(assigned), expected);
    EXPECT_EQ(assigned.storage_bytes(), bytes);
    EXPECT_EQ(assigned.queries(), 2u);
    EXPECT_EQ(assigned.scanned_points(), ids.size());
    EXPECT_EQ(assigned.growths(), 1u);
    empty_index.closed_segment(ids[0], ids[4], assigned);
    EXPECT_EQ(assigned.size(), 0u);
    EXPECT_EQ(assigned.storage_bytes(), bytes);
    EXPECT_EQ(assigned.queries(), 3u);
    EXPECT_EQ(assigned.scanned_points(), ids.size());
    assigned.release();
    EXPECT_EQ(assigned.storage_bytes(), 0u);
    EXPECT_EQ(assigned.queries(), 3u);
    EXPECT_EQ(assigned.growths(), 1u);
    EXPECT_NO_THROW(empty_index.closed_segment(ids[0], ids[4], source));
    EXPECT_EQ(source.storage_bytes(), 0u);
}
TEST(FiniteNativeGeometryTest, EdgeQueryProjectionTiesRetainCoordinateTraversalOrder)
{
    // Multiplying adjacent represented coordinates by .75 can round to one
    // projection. Find an explicit pair within eight steps for this platform.
    std::array<long double, 9> steps{};
    steps[0] = .75L;
    for (std::size_t i = 1; i < steps.size(); ++i)
        steps[i] = std::nextafter(steps[i - 1], 0.L);
    std::size_t tied = 0;
    volatile long double direction = .75L;
    for (std::size_t i = 1; i + 1 < steps.size(); ++i)
    {
        volatile long double first_projection = steps[i] * direction;
        volatile long double second_projection = steps[i + 1] * direction;
        if (first_projection == second_projection)
        {
            tied = i;
            break;
        }
    }
    ASSERT_NE(tied, 0u);
    const auto low = steps[tied + 1], high = steps[tied];
    ASSERT_LT(low, high);
    const auto box = Cell::box({0, 0, 0}, {1, 1, 1});
    PLIC::PlanarFacePtr<ordinary> support;
    for (std::size_t i = 0; i < box->num_faces(); ++i)
    {
        const auto face = box->face_geometry(i);
        if (std::all_of(face->points().begin(), face->points().end(), [](const auto& p) { return p[2] == 1; }))
            support = face;
    }
    ASSERT_TRUE(support);
    const std::array<Face::WidePoint, 7> vertices{
        {{0, 0, 1}, {low, 0, 1}, {high, 0, 1}, {.75L, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}};
    const auto face =
        PLIC::details::NativeSupportAccess::face_support(vertices, {0, 0, 1}, PLIC::planar_support<ordinary>(*support));
    const auto ids = face->vertex_identities();
    // Higher coordinate arrives first; native coordinate order visits low first.
    const std::vector<PointIdentity> input{ids[2], ids[1], ids[3], ids[0], ids[1], ids[5]};
    const std::vector<PointIdentity> expected{ids[0], ids[1], ids[3]};
    PLIC::EdgePointIndex<ordinary> index(input);
    PLIC::OrdinaryEdgeQueryScratch scratch;
    scratch.reserve(input.size(), scratch.reserve_required_peak(input.size()));
    index.closed_segment(ids[0], ids[3], scratch);
    EXPECT_EQ(copy_edge_query(scratch), expected);
    EXPECT_EQ(index.closed_segment(ids[0], ids[3]), expected);
    EXPECT_EQ(scratch.scanned_points(), 5u);
}

TEST(FiniteNativeGeometryTest, RetainedNativeBoundaryVerticesHaveStructuralIncidence)
{
    // Captured extrusion coordinates: each repeated xy pair and common height
    // establish the side's planar rectangle independently of a fitted normal.
    // The retained-support seam reproduces native extrusion recipe ancestry;
    // different support identities deliberately require geometric containment.
    constexpr long double ax = .784053262272993323201L, ay = -.158935464636048989773L;
    constexpr long double bx = .931063248949179467218L, by = -.188735864255308138926L;
    constexpr long double height = .0666666666666666657415L;
    const std::array<Face::WidePoint, 4> side_points{
        {{ax, ay, 0}, {bx, by, 0}, {bx, by, height}, {ax, ay, height}}};
    const Face::WidePoint normal{by - ay, ax - bx, 0};
    const auto side = PLIC::details::NativeSupportAccess::face_support(
        side_points, normal, PLIC::details::ordinary_native::source_support_packet(side_points));
    const std::array<Face::WidePoint, 4> cap_points{{{ax, ay, height}, {bx, by, height},
        {.945253957014124512881L, -.0948417458144867470127L, height},
        {.796003332222420700859L, -.0798667333174625238001L, height}}};
    const auto cap = Face::make_wide(cap_points);
    ASSERT_EQ(side->points().size(), 4u);
    ASSERT_EQ(cap->points().size(), 4u);
    ASSERT_NE(PLIC::planar_support<ordinary>(*side), PLIC::planar_support<ordinary>(*cap));
    // This exact shared edge starts at the side's fourth vertex. The faces
    // meet on that edge but their remaining vertices have distinct planes.
    EXPECT_EQ(side->points()[3], cap->points()[0]);
    EXPECT_EQ(side->points()[2], cap->points()[1]);
    EXPECT_FALSE(side->contains(*cap));
    EXPECT_FALSE(cap->contains(*side));
    EXPECT_FALSE(side->equivalent(*cap));
    EXPECT_FALSE(cap->equivalent(*side));
    for (const bool reversed : {false, true})
        for (unsigned rotation = 0; rotation < side_points.size(); ++rotation)
        {
            SCOPED_TRACE(reversed);
            SCOPED_TRACE(rotation);
            auto reordered = side_points;
            std::rotate(reordered.begin(), reordered.begin() + rotation, reordered.end());
            auto reordered_normal = normal;
            if (reversed)
            {
                std::reverse(reordered.begin(), reordered.end());
                for (auto& value : reordered_normal)
                    value = -value;
            }
            const auto equivalent = PLIC::details::NativeSupportAccess::face_support(reordered, reordered_normal,
                PLIC::details::GeometryIdentityAccess::translated_support(
                    PLIC::details::ordinary_native::source_support_packet(side_points),
                    PLIC::details::ordinary_native::sub(side_points[1], side_points[0])));
            EXPECT_TRUE(side->contains(*equivalent));
            EXPECT_TRUE(equivalent->contains(*side));
            EXPECT_TRUE(side->equivalent(*equivalent));
            EXPECT_FALSE(equivalent->equivalent(*cap));
            for (const auto& point : equivalent->vertex_identities())
                EXPECT_TRUE(side->contains_point(point));
        }
    // Equality, not proximity, admits a represented vertex. A dyadic shift
    // away from the side plane must fail the ordinary geometric query.
    const auto shifted = cap->translated({0, 1. / 1024, 0});
    EXPECT_FALSE(side->contains_point(shifted->vertex_identities().front()));
}

TEST(FiniteNativeGeometryTest, SourceValidationStillRejectsOffPlaneFourthVertex)
{
    const std::array<Face::WidePoint, 4> nonplanar{{
        {.784053262272993323201L, -.158935464636048989773L, 0},
        {.931063248949179467218L, -.188735864255308138926L, 0},
        {.931063248949179467218L, -.188735864255308138926L, .0666666666666666657415L},
        {.784053262272993323201L, -.158935464636048989773L + 1.L / 1024, .0666666666666666657415L}}};
    try
    {
        (void) Face::make_wide(nonplanar);
        FAIL() << "An actually off-plane fourth vertex must not inherit structural incidence.";
    }
    catch (const PLIC::GeometryError& error)
    {
        EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::InvalidTopology);
        EXPECT_STREQ(error.what(), "Ordinary native face is nonplanar.");
    }
}

TEST(FiniteNativeGeometryTest, PartitionRejectsRealGapAndOverlapBelowMetricBudget)
{
    const auto square = [](double lo, double hi)
    {
        const std::array<PLIC::GeometryPoint, 4> p{{{lo, 0, 0}, {hi, 0, 0}, {hi, 1, 0}, {lo, 1, 0}}};
        return Face::make(p);
    };
    const auto source = square(0, 1), left = square(0, .5);
    const double gap = std::nextafter(.5, 1.);
    const std::array gap_tiles{left, square(gap, 1)};
    ASSERT_LT(gap - .5, 8e-15);
    EXPECT_THROW(source->certify_partition(gap_tiles), std::invalid_argument);
    const std::array overlapping{square(0, gap), square(.5, 1)};
    EXPECT_THROW(source->certify_partition(overlapping), std::invalid_argument);
    const std::array valid{left, square(.5, 1)};
    EXPECT_NO_THROW(source->certify_partition(valid));
}

TEST(FiniteNativeGeometryTest, PartitionFirstMomentsUseCommonLocalReferenceAndRetainCaps)
{
    const auto rectangle = [](double lo, double hi)
    {
        const std::array<PLIC::GeometryPoint, 4> p{
            {{lo, -2e9, 3e9}, {hi, -2e9, 3e9}, {hi, -2e9 + 8, 3e9}, {lo, -2e9 + 8, 3e9}}};
        return Face::make(p);
    };
    const auto source = rectangle(1e9, 1e9 + 8);
    const std::array tiles{rectangle(1e9, 1e9 + 3)->reversed(), rectangle(1e9 + 3, 1e9 + 8)};
    EXPECT_NO_THROW(source->certify_partition(tiles));
    // Equal total area does not authorize a shifted first moment or missing region.
    const std::array shifted{rectangle(1e9, 1e9 + 3), rectangle(1e9 + 2, 1e9 + 7)};
    EXPECT_THROW(source->certify_partition(shifted), std::invalid_argument);
    const std::vector<PLIC::PlanarFacePtr<ordinary>> excessive(65537, tiles.front());
    EXPECT_THROW(source->certify_partition(excessive), std::invalid_argument);
}

TEST(FiniteNativeGeometryTest, OrdinarySourcePreparationDoesNotInvokeExactBackend)
{
    struct Sampling
    {
        Sampling()
        {
            PLIC::enable_backend_statistics<ArithmeticMode::Exact>(true);
            PLIC::reset_backend_statistics<ArithmeticMode::Exact>();
        }
        ~Sampling() { PLIC::enable_backend_statistics<ArithmeticMode::Exact>(false); }
    } sampling;
    const auto cube = Cell::box({0, 0, 0}, {1, 1, 1});
    const auto source = cube->numerical_shell();
    const auto imported = Cell::from_geometry(*cube,
        []
        {
            PLIC::GeometryLimits limits;
            limits.arithmetic_mode = ordinary;
            return limits;
        }());
    EXPECT_EQ(imported->volume().value, 1);
    EXPECT_EQ(source.face_support_identities.size(), 6u);
    EXPECT_EQ(cube->split_aabb_midpoints().size(), 8u);
    const auto cut = cube->clip({{1, 0, 0}, {}, .25});
    ASSERT_TRUE(cut.inside);
    const auto restored = Cell::decode_certificate(cut.inside->encode_certificate());
    EXPECT_EQ(restored->volume().value, .25);
    const auto face = cube->face_geometry(0);
    EXPECT_TRUE(Face::decode(face->encode())->equivalent(*face));
    const auto stats = PLIC::backend_statistics<ArithmeticMode::Exact>();
    EXPECT_EQ(stats.source_coordinate_exports, 0u);
    EXPECT_EQ(stats.source_support_exports, 0u);
    EXPECT_EQ(stats.source_membership_predicates, 0u);
    EXPECT_EQ(stats.volume_evaluations, 0u);
    EXPECT_EQ(stats.moment_evaluations, 0u);
    EXPECT_EQ(stats.clip_constructions, 0u);
    EXPECT_EQ(stats.intersection_constructions, 0u);
    EXPECT_EQ(stats.face_measure_evaluations, 0u);
    EXPECT_EQ(stats.measure_values, 0u);
    EXPECT_EQ(stats.measure_arithmetic, 0u);
    EXPECT_EQ(stats.cover_certifications, 0u);
}

TEST(FiniteNativeGeometryTest, OrdinaryRetainedSupportDoesNotCertifyRoundedBoundedPlanarity)
{
    if (!PLIC::certification_environment())
        GTEST_SKIP() << "Bounded floating-point environment unavailable.";
    using BoundedCell = PLIC::ConvexCellGeometry<ArithmeticMode::BoundedCertification>;
    const auto cube = Cell::box({0, 0, 0}, {1, 1, 1});
    const auto bounded_cube = BoundedCell::from_geometry(*cube);
    EXPECT_LE(bounded_cube->volume().lower(), 1);
    EXPECT_GE(bounded_cube->volume().upper(), 1);
    const auto clipped = cube->clip({{1, 1, 1e-8}, {}, 1 + 1e-8}).inside;
    ASSERT_TRUE(clipped);
    // The cap keeps its original support for ordinary topology. Its rounded
    // coordinates do not thereby acquire a certified coplanarity proof.
    EXPECT_THROW((void) BoundedCell::from_geometry(*clipped), std::invalid_argument);
    const auto point_id = cube->face_geometry(0)->vertex_identities().front();
    const auto point = PLIC::source_point<ArithmeticMode::BoundedCertification>(*cube, point_id);
    EXPECT_TRUE(point.supports().empty());
}
TEST(FiniteNativeGeometryTest, PlaneMeasureIntegratesObliqueCubeAndComplementWithoutConstructingCuts)
{
    const auto cell = Cell::box({0, 0, 0}, {1, 1, 1});
    const PLIC::OrdinaryPlaneMeasure measure(*cell);
    PLIC::reset_geometry_statistics();
    for (const auto [offset, expected] :
        std::array<std::array<double, 2>, 4>{{{.5, 1. / 48}, {1, 1. / 6}, {1.5, .5}, {2.5, 47. / 48}}})
    {
        const auto below = measure.below({{1, 1, 1}, {}, offset}, true);
        const auto above = measure.below({{-1, -1, -1}, {}, -offset});
        EXPECT_NEAR(below.volume, expected, 4e-16L);
        EXPECT_NEAR(above.volume, 1 - static_cast<long double>(expected), 4e-16L);
        EXPECT_NEAR(below.volume + above.volume, 1, 4e-18L);
        EXPECT_EQ(below.status, PLIC::PlaneMeasureStatus::Resolved);
        EXPECT_GT(below.numerical_error_estimate, 0);
        if (offset <= 1)
            EXPECT_NEAR(below.section_area, std::sqrt(3.L) * offset * offset / 2, 4e-18L);
    }
    const auto stats = PLIC::geometry_statistics();
    EXPECT_EQ(stats.constructions, 0u);
    EXPECT_EQ(stats.intersections, 0u);
    EXPECT_EQ(stats.moment_evaluations, 0u);
}

TEST(FiniteNativeGeometryTest, StrictFaceSideUsesBothOffsetEndpointsAndRejectsMalformedInput)
{
    using Side = PLIC::WholeFacePlaneSide;
    std::array<PLIC::Point<ordinary>, 4> face;
    face[0].value = {0, 0, 0};
    face[1].value = {0, 1, 0};
    face[2].value = {0, 1, 1};
    face[3].value = {0, 0, 1};
    const PLIC::GeometryHalfspace plane{{1, 0, 0}, {}, 0};
    const auto side = [&](std::array<long double, 2> offsets)
    { return PLIC::OrdinaryPlaneMeasure::strict_face_side(face, plane, offsets); };
    EXPECT_EQ(side({1, 2}), Side::StrictNegative);
    EXPECT_EQ(side({-2, -1}), Side::StrictPositive);
    EXPECT_EQ(side({0, 1}), Side::UnknownOrContact);
    EXPECT_EQ(side({-1, 1}), Side::UnknownOrContact);
    auto crossing = face;
    crossing[2].value[0] = 2;
    EXPECT_EQ(PLIC::OrdinaryPlaneMeasure::strict_face_side(crossing, plane, {1, 1}), Side::UnknownOrContact);
    EXPECT_THROW((void) PLIC::OrdinaryPlaneMeasure::strict_face_side(std::span(face).first(2), plane, {1, 2}),
        std::invalid_argument);
    EXPECT_THROW((void) side({2, 1}), std::invalid_argument);
    EXPECT_THROW((void) side({1, std::numeric_limits<long double>::infinity()}), std::invalid_argument);
    auto invalid = face;
    invalid[3].value[2] = std::numeric_limits<long double>::quiet_NaN();
    EXPECT_THROW((void) PLIC::OrdinaryPlaneMeasure::strict_face_side(invalid, plane, {-1, 1}), std::invalid_argument);
    const PLIC::GeometryHalfspace zero_normal{{0, 0, 0}, {}, 0};
    EXPECT_THROW((void) PLIC::OrdinaryPlaneMeasure::strict_face_side(face, zero_normal, {1, 2}), std::invalid_argument);
}

TEST(FiniteNativeGeometryTest, PlaneMeasureUsesLocalScaledCoordinatesAndMeasuresTinyPhaseDirectly)
{
    for (const auto exponent : {-20, 0, 20})
    {
        const double scale = std::ldexp(1., exponent), shift = std::ldexp(scale, 20);
        const PLIC::GeometryPoint origin{shift, -shift, shift};
        const auto cell = Cell::box(origin, {shift + scale, -shift + scale, shift + scale});
        const PLIC::OrdinaryPlaneMeasure measure(*cell);
        for (const double width : {.5, 1e-9})
        {
            const auto result = measure.below({{1, 1, 1}, origin, width * scale}, true);
            const long double expected = std::pow(static_cast<long double>(width * scale), 3) / 6;
            EXPECT_NEAR(result.volume / expected, 1, 4e-16L);
            EXPECT_LT(result.numerical_error_estimate / expected, 8e-15L);
            EXPECT_EQ(result.status, PLIC::PlaneMeasureStatus::Resolved);
            const auto rescaled_normal = measure.below({{2, 2, 2}, origin, 2 * width * scale});
            EXPECT_NEAR(rescaled_normal.volume / expected, 1, 4e-16L);
        }
    }
}

TEST(FiniteNativeGeometryTest, PlaneFaceAreaCoversAnalyticPolygonAndPreservesComplement)
{
    const std::array<std::array<long double, 3>, 4> square{{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}};
    const auto triangle = PLIC::ordinary_plane_face_area(square, {{1, 1, 0}, {}, .5});
    const auto complement = PLIC::ordinary_plane_face_area(square, {{-1, -1, 0}, {}, -.5});
    EXPECT_NEAR(triangle.area, .125L, 1e-18L);
    EXPECT_NEAR(triangle.area + complement.area, 1, 1e-18L);
    EXPECT_GT(triangle.numerical_error_estimate, 0);
    const PLIC::GeometryPoint translated{1e9, -2e9, 3e9};
    const auto moved = PLIC::ordinary_plane_face_area(square, {{1, 1, 0}, translated, .5}, translated);
    EXPECT_EQ(moved.area, triangle.area);
    EXPECT_EQ(PLIC::ordinary_plane_face_area(square, {{1, 0, 0}, {}, -1}).area, 0);
    EXPECT_EQ(PLIC::ordinary_plane_face_area(square, {{1, 0, 0}, {}, 2}).area, 1);
}

} // namespace
