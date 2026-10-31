/** @file testOrdinaryAggregateReplay.cc
 * @brief Public aggregate replay of the captured face-20 coverage construction.
 */
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/SupportClassification.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "geometry/swept_volume/OrdinaryAggregate.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <optional>

namespace
{
namespace PLIC = SimpleFluid::PLIC;
namespace AMR = SimpleFluid::Meshes::AMR;
namespace Sweep = SimpleFluid::geometry::swept_volume;
constexpr auto mode = SimpleFluid::ArithmeticMode::Ordinary;
using Geometry = PLIC::ConvexCellGeometry<mode>;
using Point = PLIC::Point<mode>;
using Wide = std::array<long double, 3>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

PLIC::GeometryLimits ordinary_limits()
{
    PLIC::GeometryLimits limits;
    limits.arithmetic_mode = mode;
    return limits;
}

std::shared_ptr<const AMR::DistributedAdaptiveHierarchy> native_fifty_cells()
{
    // Same native grid and isotropic marks as skew_coarse_fine_plane. No
    // solver, phase reconstruction, or Exact reference supplies this fixture.
    AMR::ExtrudedAMRGrid grid;
    for (unsigned y = 0; y < 3; ++y)
        for (unsigned x = 0; x < 5; ++x)
            grid.xy_nodes.push_back({x / 4.0 + .2 * y / 2.0, y / 2.0, 0});
    for (unsigned y = 0; y < 2; ++y)
        for (unsigned x = 0; x < 4; ++x)
        {
            const unsigned p = 5 * y + x;
            grid.base_cells.push_back({p, p + 1, p + 6, p + 5});
        }
    for (unsigned x = 0; x < 4; ++x)
    {
        grid.boundaries.push_back({x, x + 1, "wall"});
        grid.boundaries.push_back({10 + x, 11 + x, "wall"});
    }
    for (unsigned y = 0; y < 2; ++y)
    {
        grid.boundaries.push_back({5 * y, 5 * (y + 1), "wall"});
        grid.boundaries.push_back({5 * y + 4, 5 * (y + 1) + 4, "wall"});
    }
    grid.z_edges = {0, .5, 1};
    std::array<std::uint64_t, 16> owned;
    std::iota(owned.begin(), owned.end(), 0);
    auto provider = std::make_shared<AMR::ExtrudedAMR>(grid);
    auto bootstrap = provider->detach_owned(owned, {'s', 'k', 'e', 'w', 'p', 'a', 's', 's', 'a', 'g', 'e'});
    auto raw = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, Tpetra::getDefaultComm());
    std::vector<AMR::AdaptiveMark> marks;
    for (const auto& [key, leaf] : raw->owned())
    {
        const auto center = Geometry::from_geometry(*raw->geometry(leaf), ordinary_limits())->centroid();
        const double xi = center[0] - .2 * center[1];
        if (xi > .5 && xi < .75)
            marks.push_back({key, AMR::AdaptiveSplit::Isotropic});
    }
    return raw->apply(raw->plan_adaptation(marks));
}

Point native_point(std::span<const PLIC::ConvexGeometryPtr<mode>> cells, PLIC::GeometryPoint position)
{
    std::optional<PLIC::PointIdentity<mode>> identity;
    std::vector<PLIC::ConvexGeometryPtr<mode>> fan;
    for (const auto& cell : cells)
    {
        for (std::size_t f = 0; f < cell->num_faces(); ++f)
        {
            const auto face = cell->face_geometry(f);
            const auto ids = face->vertex_identities();
            const auto& points = face->points();
            for (std::size_t i = 0; i < points.size(); ++i)
                if (PLIC::GeometryPoint{static_cast<double>(points[i][0]), static_cast<double>(points[i][1]),
                        static_cast<double>(points[i][2])} == position)
                {
                    if (!identity)
                        identity = ids.at(i);
                    else if (*identity != ids.at(i))
                        throw std::logic_error("Captured native point has inconsistent identities.");
                }
        }
    }
    if (!identity)
        throw std::logic_error("Captured native point is absent from the native hierarchy.");
    // Include coarse faces containing a hanging node even when that node is
    // not one of the coarse face's original polygon vertices.
    for (const auto& cell : cells)
        for (std::size_t f = 0; f < cell->num_faces(); ++f)
            if (cell->face_geometry(f)->contains_point(*identity))
            {
                fan.push_back(cell);
                break;
            }
    return PLIC::source_point<mode>(fan, *identity);
}

struct CapturedFace20
{
    Sweep::OrdinaryAggregateSourceBatch batch;
    std::vector<Sweep::OrdinaryAggregateDonor> donors;

    CapturedFace20()
    {
        const auto hierarchy = native_fifty_cells();
        if (hierarchy->owned().size() != 50)
            throw std::logic_error("Captured native hierarchy must contain fifty cells.");
        std::vector<PLIC::ConvexGeometryPtr<mode>> native;
        for (const auto& [key, leaf] : hierarchy->owned())
            native.push_back(hierarchy->geometry(leaf));

        // The serial hierarchy map order is the captured native GID order.
        // Roots 0,1,4,5 stay coarse; roots 2,6 are isotropically refined.
        // The retained fine donors use base-child 0 or 3 and the two central
        // axial children. These are original native cells, not rounded shells.
        constexpr std::array<std::uint64_t, 14> gids{0, 1, 2, 3, 5, 6, 17, 18, 28, 29, 30, 31, 33, 34};
        constexpr long double face_budget = 5.5555555555555555556e-17L;
        for (const auto gid : gids)
        {
            Sweep::OrdinaryAggregateDonor donor;
            donor.id = gid;
            donor.geometry = Geometry::from_geometry(*native.at(gid), ordinary_limits());
            // This regression isolates the original Coverage failure. The
            // original phase cuts happen later and cannot establish coverage.
            donor.pure_carrier = true;
            donor.bulk_budget = donor.carrier_budget = face_budget / (2 * gids.size());
            donors.push_back(std::move(donor));
        }

        constexpr std::array<PLIC::GeometryPoint, 5> positions{
            {{.25, 0, .5}, {.5, 0, .5}, {.55, .25, .5}, {.6, .5, .5}, {.35, .5, .5}}};
        // Historical binary64 controls, captured before the failed aggregate.
        constexpr std::array<PLIC::GeometryPoint, 5> velocities{
            {{.03231757636313036, -.034235044366371446, .00020134044560284232},
                {.037826356993721111, -.0077347754380158902, -.00011613331305328594},
                {.020126831830384558, -.0019448370560082701, -.00010707876784630988},
                {.001365136407667679, .0028179807764118036, -.000072073146715659762},
                {-.0039310268269660432, -.027857764196718336, -.000032582143939839351}}};
        std::array<Point, 5> current, departure;
        for (std::size_t node = 0; node < current.size(); ++node)
        {
            current[node] = native_point(native, positions[node]);
            std::vector<PLIC::FaceConstraint<mode>> walls;
            if (node == 0)
                for (const auto gid : {1u, 0u, 2u, 3u})
                    walls.push_back({Geometry::from_geometry(*native.at(gid), ordinary_limits()), 2});
            if (node == 1)
                for (const auto gid : {2u, 3u, 5u, 6u})
                    walls.push_back({Geometry::from_geometry(*native.at(gid), ordinary_limits()), 2});
            departure[node] = PLIC::backtrace_constrained<mode>(current[node], velocities[node], .002, walls);
        }
        batch.points.resize(4);
        batch.tetrahedra.resize(1);
        auto& tetra = batch.tetrahedra.front();
        tetra.sign = 1;
        tetra.vertices = {Sweep::OrdinaryAggregatePointIndex{0}, Sweep::OrdinaryAggregatePointIndex{1},
            Sweep::OrdinaryAggregatePointIndex{2}, Sweep::OrdinaryAggregatePointIndex{3}};
        batch.points[0] = PLIC::point_average<mode>(current);
        // The corrected backcenter has no original support in the capture.
        batch.points[1] = Point({0xe661cd3ebeb43883p-65L, 0x80016a2e733017e1p-65L, 0xffffdb9e8eeedf9ep-65L});
        batch.points[2] = departure[0];
        batch.points[3] = departure[4];
    }
};

// Independent scalar triple product of the original represented four points.
// No aggregate clipping, predicate implementation, or measured child is used.
long double independent_volume(std::span<const Point> points)
{
    Wide a, b, c;
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        a[axis] = points[1].value[axis] - points[0].value[axis];
        b[axis] = points[2].value[axis] - points[0].value[axis];
        c[axis] = points[3].value[axis] - points[0].value[axis];
    }
    return std::abs(a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) +
                    a[2] * (b[0] * c[1] - b[1] * c[0])) /
           6;
}

void expect_captured_source(const CapturedFace20& fixture)
{
    const std::array<Wide, 4> expected{{{0xe6666666666664cdp-65L, 0x8000000000000000p-65L, 0x8000000000000000p-64L},
        {0xe661cd3ebeb43883p-65L, 0x80016a2e733017e1p-65L, 0xffffdb9e8eeedf9ep-65L},
        {0xffef0e68880ebe89p-66L, 0x0p+0L, 0xfffff27cff5b0d57p-65L},
        {0xb3343b01ba0576e4p-65L, 0x8003a6c05f50f6e8p-64L, 0x80000117e0e469cep-64L}}};
    for (unsigned vertex = 0; vertex < expected.size(); ++vertex)
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            const auto observed = fixture.batch.points[vertex].value[axis];
            if constexpr (std::numeric_limits<long double>::digits == 64)
                EXPECT_EQ(observed, expected[vertex][axis]) << "vertex=" << vertex << ", axis=" << axis;
            else
                EXPECT_LE(std::abs(observed - expected[vertex][axis]),
                    2 * std::numeric_limits<long double>::epsilon() * std::max(1.L, std::abs(expected[vertex][axis])));
        }
    EXPECT_TRUE(fixture.batch.points[1].supports().empty());
    EXPECT_FALSE(fixture.batch.points[2].supports().empty());
    const auto wall = fixture.donors[1].geometry->face_geometry(2);
    EXPECT_TRUE(PLIC::has_support(fixture.batch.points[2].dynamic(), PLIC::original_support_authority(*wall)));
}

void expect_shared_support_across_frames(const CapturedFace20& fixture)
{
    bool found = false;
    for (std::size_t a = 0; a < fixture.donors.size(); ++a)
        for (std::size_t b = a + 1; b < fixture.donors.size(); ++b)
        {
            const auto& first = fixture.donors[a].geometry;
            const auto& second = fixture.donors[b].geometry;
            if (first->origin() == second->origin())
                continue;
            for (std::size_t fa = 0; fa < first->num_faces(); ++fa)
                for (std::size_t fb = 0; fb < second->num_faces(); ++fb)
                {
                    const auto face_a = first->face_geometry(fa), face_b = second->face_geometry(fb);
                    if (PLIC::original_support_authority(*face_a) != PLIC::original_support_authority(*face_b))
                        continue;
                    const auto normal_a = face_a->area_vector(), normal_b = face_b->area_vector();
                    const auto dot = normal_a.x * normal_b.x + normal_a.y * normal_b.y + normal_a.z * normal_b.z;
                    found |= dot < 0;
                }
        }
    EXPECT_TRUE(found) << "Captured donors must share an oppositely oriented support across distinct frames.";
}

void expect_covered_once(const Sweep::OrdinaryAggregateResult& result, const CapturedFace20& fixture)
{
    const auto expected = independent_volume(fixture.batch.points);
    ASSERT_GT(expected, 0);
    long double bulk = 0, carrier = 0, total_budget = 0;
    for (const auto& donor : fixture.donors)
        total_budget += donor.bulk_budget;
    EXPECT_LE(std::abs(result.signed_source_volume - expected), total_budget);
    for (const auto& group : result.groups)
    {
        const auto donor = std::find_if(fixture.donors.begin(), fixture.donors.end(),
            [&](const auto& candidate) { return candidate.id == group.donor; });
        ASSERT_NE(donor, fixture.donors.end());
        EXPECT_EQ(group.sign, 1);
        EXPECT_GE(group.nominal.bulk, 0);
        EXPECT_EQ(group.nominal.gas, 0);
        EXPECT_GE(group.nominal.carrier, 0);
        EXPECT_EQ(group.nominal.bulk, group.nominal.carrier);
        EXPECT_FALSE(group.gas_positive);
        EXPECT_EQ(group.approximation_bound.bulk, 0);
        EXPECT_EQ(group.approximation_bound.gas, 0);
        EXPECT_EQ(group.approximation_bound.carrier, 0);
        EXPECT_LE(group.numerical_error_estimate.bulk, donor->bulk_budget);
        EXPECT_EQ(group.numerical_error_estimate.gas, 0);
        EXPECT_LE(group.numerical_error_estimate.carrier, donor->carrier_budget);
        bulk += group.nominal.bulk;
        carrier += group.nominal.carrier;
    }
    EXPECT_LE(std::abs(bulk - expected), total_budget);
    EXPECT_EQ(carrier, bulk);
}

TEST(OrdinaryAggregateReplay, CapturedFace20OriginalSupportCoverageInBothDonorOrders)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "The captured native GID ordering is serial.";
    const auto before = PLIC::geometry_statistics();
    CapturedFace20 fixture;
    expect_captured_source(fixture);
    expect_shared_support_across_frames(fixture);
    auto batch = fixture.batch;
    const Sweep::OrdinaryAggregateSource source(std::move(batch), {}, ordinary_limits());
    for (unsigned order = 0; order < 2; ++order)
    {
        SCOPED_TRACE(order);
        expect_covered_once(source.measure(fixture.donors), fixture);
        std::reverse(fixture.donors.begin(), fixture.donors.end());
    }
    const auto after = PLIC::geometry_statistics();
    EXPECT_EQ(after.exact_predicates, before.exact_predicates);
    EXPECT_EQ(after.rational_constructions, before.rational_constructions);
    EXPECT_EQ(after.rational_measures, before.rational_measures);
    EXPECT_EQ(after.rational_ledgers, before.rational_ledgers);
}

TEST(OrdinaryAggregateReplay, CapturedFallbackUsesRemainingNestedWorkAndRestoresCaller)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "The captured native GID ordering is serial.";
    CapturedFace20 fixture;
    auto batch = fixture.batch;
    const Sweep::OrdinaryAggregateSource source(std::move(batch), {}, ordinary_limits());
    auto* const previous = PLIC::details::active_work;
    ASSERT_EQ(previous, nullptr);
    for (unsigned order = 0; order < 2; ++order)
    {
        SCOPED_TRACE(order);
        std::size_t bytes, operations, pairs;
        {
            PLIC::details::Operation caller(ordinary_limits());
            expect_covered_once(source.measure(fixture.donors), fixture);
            EXPECT_EQ(PLIC::details::active_work, &caller.work);
            bytes = caller.work.bytes;
            operations = caller.work.operations;
            pairs = caller.work.pairs;
        }
        EXPECT_EQ(PLIC::details::active_work, previous);
        ASSERT_GT(bytes, 0u);
        ASSERT_GT(operations, 0u);
        ASSERT_GT(pairs, 0u);
        constexpr std::size_t existing_bytes = 4093, existing_operations = 101, existing_pairs = 7;
        auto limits = ordinary_limits();
        limits.byte_budget = existing_bytes + bytes;
        {
            PLIC::details::Operation caller(limits);
            caller.work.bytes = existing_bytes;
            caller.work.operations = existing_operations;
            caller.work.pairs = existing_pairs;
            expect_covered_once(source.measure(fixture.donors), fixture);
            EXPECT_EQ(PLIC::details::active_work, &caller.work);
            EXPECT_EQ(caller.work.bytes, existing_bytes + bytes);
            EXPECT_EQ(caller.work.operations, existing_operations + operations);
            EXPECT_EQ(caller.work.pairs, existing_pairs + pairs);
        }
        EXPECT_EQ(PLIC::details::active_work, previous);
        --limits.byte_budget;
        {
            PLIC::details::Operation caller(limits);
            caller.work.bytes = existing_bytes;
            caller.work.operations = existing_operations;
            caller.work.pairs = existing_pairs;
            try
            {
                (void) source.measure(fixture.donors);
                FAIL() << "One byte less than the required remaining work must reject.";
            }
            catch (const PLIC::GeometryError& error)
            {
                EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::ResourceLimit) << error.what();
                EXPECT_FALSE(error.recoverable_trial());
            }
            EXPECT_EQ(PLIC::details::active_work, &caller.work);
            EXPECT_LE(caller.work.bytes, limits.byte_budget);
            EXPECT_GE(caller.work.operations, existing_operations);
            EXPECT_GE(caller.work.pairs, existing_pairs);
        }
        EXPECT_EQ(PLIC::details::active_work, previous);
        std::reverse(fixture.donors.begin(), fixture.donors.end());
    }
}
} // namespace
