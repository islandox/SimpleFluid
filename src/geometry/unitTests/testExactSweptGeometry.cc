/** @file testExactSweptGeometry.cc
 * @brief Independent signed sweeps, actual donor partitions and exact pullbacks.
 */
#include "geometry/ExactSweptGeometry.hh"
#include "utils/testing_environment.hh"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <map>
namespace
{
using namespace SimpleFluid::Meshes;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
ExactPlanarFacePtr face_at(const ConvexGeometryPtr& cell, unsigned axis, double coordinate)
{
    for (std::size_t f = 0; f < cell->num_faces(); ++f)
    {
        auto face = cell->face_geometry(f);
        const auto p = face->centroid();
        const double x = axis == 0 ? p.x : axis == 1 ? p.y : p.z;
        if (x == coordinate)
            return face;
    }
    throw std::logic_error("Missing oracle face.");
}
std::vector<ExactNodeTrajectory> traces(const ExactPlanarFacePtr& face, GeometryPoint velocity, double dt)
{
    std::vector<ExactNodeTrajectory> result;
    for (const auto& p : face->vertex_identities())
        result.push_back(ExactNodeTrajectory::backtrace(p, velocity, dt));
    return result;
}
TEST(ExactSweptGeometry, CertifiedClipsPreserveCollinearContactsSliversAndStrictRawAdmission)
{
    using M = ExactGeometryMeasure;
    const std::vector<GeometryPoint> points{
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}, {.5, 0, 0}};
    const std::vector<std::vector<std::size_t>> faces{
        {0, 4, 6, 2}, {1, 3, 7, 5}, {0, 8, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1, 8}, {4, 5, 7, 6}};
    const auto cell = ConvexCellGeometry::make(points, faces);
    const auto fragment = ExactConvexFragment::from_geometry(*cell);
    const auto half = fragment->phases({{1, 0, 0}, {0, 0, 0}, .5});
    EXPECT_EQ(half.gas.compare(M::from_value(.5)), 0);
    EXPECT_EQ(half.carrier.compare(M::from_value(.5)), 0);
    for (const auto& normal : std::array<GeometryPoint, 3>{{{1, 0, 0}, {1, 1, 0}, {1, 1, 1}}})
    {
        // Supporting face, edge and vertex contacts have exactly zero 3D
        // volume. A separate face authority retains their native incidence.
        const auto contact = fragment->phases({normal, {0, 0, 0}, 0});
        EXPECT_EQ(contact.gas.sign(), 0);
        EXPECT_EQ(contact.carrier.compare(M::from_value(1)), 0);
    }
    const double eta = 1e-310;
    const auto small_gas = fragment->phases({{1, 0, 0}, {0, 0, 0}, eta});
    const auto small_carrier = fragment->phases({{-1, 0, 0}, {0, 0, 0}, -eta});
    EXPECT_EQ(small_gas.gas.compare(M::from_value(eta)), 0);
    EXPECT_EQ(small_carrier.carrier.compare(M::from_value(eta)), 0);
    EXPECT_EQ((small_gas.gas + small_gas.carrier).compare(M::from_value(1)), 0);
    EXPECT_EQ((small_carrier.gas + small_carrier.carrier).compare(M::from_value(1)), 0);
    const auto side = face_at(cell, 0, 1);
    const auto swept = ExactSweptFace::make(*side, traces(side, {1, 0, 0}, .25), .25);
    const std::array<ExactSweepDonor, 2> donors{{{17, cell}, {29, ConvexCellGeometry::box({1, 0, 0}, {2, 1, 1})}}};
    const auto pieces = swept->partition(donors);
    ASSERT_FALSE(pieces.empty());
    M volume;
    for (const auto& piece : pieces)
    {
        EXPECT_EQ(piece.donor, 17);
        volume = volume + piece.geometry->volume().scaled(piece.sign);
        // Independently re-admit each derived shell through the unchanged
        // strict public/native and raw decoder paths, including full moments.
        const auto admitted = piece.geometry->as_geometry();
        EXPECT_EQ(M::volume(*admitted).compare(piece.geometry->volume()), 0);
        const auto wire = piece.geometry->encode();
        const auto decoded = ExactConvexFragment::decode(wire);
        EXPECT_EQ(decoded->encode(), wire);
        EXPECT_TRUE(decoded->as_geometry()->equivalent(*admitted));
    }
    EXPECT_EQ(volume.compare(M::from_value(.25)), 0);
    auto invalid_points = points;
    invalid_points.back()[1] = .125; // Raw y-min face is now nonplanar.
    EXPECT_THROW(ConvexCellGeometry::make(invalid_points, faces), std::invalid_argument);
    auto invalid_faces = faces;
    invalid_faces[2].insert(invalid_faces[2].begin() + 1, 8);
    EXPECT_THROW(ConvexCellGeometry::make(points, invalid_faces), std::invalid_argument);
    auto malformed = fragment->encode();
    ASSERT_GE(malformed.size(), 8);
    std::fill(malformed.end() - 8, malformed.end(), char(0xff));
    EXPECT_THROW(ExactConvexFragment::decode(malformed), std::invalid_argument);
    EXPECT_EQ(fragment->volume().compare(M::from_value(1)), 0);
    ExactSweepLimits exhausted;
    exhausted.byte_budget = 1;
    EXPECT_THROW(swept->partition(donors, exhausted), std::invalid_argument);
    EXPECT_NO_THROW(swept->partition(donors));
}
TEST(ExactSweptGeometry, OptionalClipCacheCanHaveZeroLeaseWithoutWeakeningGeometryProofs)
{
    const auto cell = ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1});
    const auto face = face_at(cell, 0, 1);
    const auto identity = ExactSweptFace::make(*face, traces(face, {0, 0, 0}, 1), 0.);
    ASSERT_TRUE(identity->simplices().empty());
    const std::array<ExactSweepDonor, 2> touching{{{3, cell}, {5, ConvexCellGeometry::box({1, 0, 0}, {2, 1, 1})}}};
    ExactSweepLimits limits;
    // These two donor records exhaust the original admitted payload exactly;
    // the pairwise geometry proof still has one allowed operation, but there
    // are ZERO bytes available for the optional distance cache. It must use
    // the same certified constant-workspace clip, not reject a valid contact.
    limits.byte_budget = touching.size() * sizeof(ExactSweepDonor);
    limits.maximum_pair_tests = 1;
    EXPECT_TRUE(identity->partition(touching, limits).empty());
    auto roomy = limits;
    roomy.byte_budget = 1 << 20;
    EXPECT_TRUE(identity->partition(touching, roomy).empty());
    auto overlapping = touching;
    overlapping[1].geometry = ConvexCellGeometry::box({.5, 0, 0}, {1.5, 1, 1});
    EXPECT_THROW(identity->partition(overlapping, limits), std::invalid_argument);
    limits.maximum_pair_tests = 0;
    EXPECT_THROW(identity->partition(touching, limits), std::invalid_argument);
    limits.maximum_pair_tests = 1;
    EXPECT_TRUE(identity->partition(touching, limits).empty());
}
TEST(ExactSweptGeometry, CoplanarAreaRatiosAndOrientedVelocityIntegralsRemainExact)
{
    using M = ExactGeometryMeasure;
    const auto parent =
        ExactPlanarFace::make(std::array<GeometryPoint, 4>{{{0, 0, 0}, {2, 0, 2}, {2, 2, 2}, {0, 2, 0}}});
    const auto child =
        ExactPlanarFace::make(std::array<GeometryPoint, 4>{{{0, 0, 0}, {1, 0, 1}, {1, 1, 1}, {0, 1, 0}}});
    EXPECT_EQ(child->coplanar_area_ratio(*parent).compare(M::from_value(.25)), 0);
    EXPECT_EQ(child->reversed()->coplanar_area_ratio(*parent).compare(M::from_value(.25)), 0);
    EXPECT_EQ(parent->coplanar_area_ratio(*child).compare(M::from_value(4)), 0);
    EXPECT_EQ(parent->area_vector_dot({1, 2, 3}).compare(M::from_value(8)), 0);
    EXPECT_EQ(parent->reversed()->area_vector_dot({1, 2, 3}).compare(M::from_value(-8)), 0);
    const auto different_plane = child->translated({0, 0, std::ldexp(1., -200)});
    EXPECT_THROW(different_plane->coplanar_area_ratio(*parent), std::invalid_argument);
    EXPECT_THROW(child->coplanar_area_ratio(*parent, 1), std::invalid_argument);
    EXPECT_THROW(parent->area_vector_dot({1, 2, 3}, 1), std::invalid_argument);
    EXPECT_THROW(parent->area_vector_dot({0, 0, INFINITY}), std::invalid_argument);
    const double eta = 1e-310, speed = 1e-20;
    const auto tiny =
        ExactPlanarFace::make(std::array<GeometryPoint, 4>{{{0, 0, 0}, {1, 0, 0}, {1, eta, 0}, {0, eta, 0}}});
    const auto flux = tiny->area_vector_dot({0, 0, speed});
    EXPECT_EQ(flux.compare(M::from_value(eta).scaled(speed)), 0);
    EXPECT_GT(flux.sign(), 0);
    EXPECT_TRUE(flux.enclosure().exact_nonzero);
    EXPECT_EQ(tiny->coplanar_area_ratio(*tiny).compare(M::from_value(1)), 0);
}
TEST(ExactSweptGeometry, ActualDonorGasComplementAndTinyUnexportedTailRemainIndependent)
{
    const auto cube = ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1});
    auto face = face_at(cube, 0, 1);
    const double t = .1;
    auto sweep = ExactSweptFace::make(*face, traces(face, {1, 0, 0}, t), t);
    EXPECT_EQ(sweep->volume().compare(ExactGeometryMeasure::from_value(t)), 0);
    const std::array<ExactSweepDonor, 1> donors{{{7, cube}}};
    auto pieces = sweep->partition(donors);
    const auto p = ExactSweptFace::phases(pieces, {{1, 1, 0}, {0, 0, 0}, 1});
    EXPECT_EQ(p.gas.compare(ExactGeometryMeasure::from_value(t).scaled(t).scaled(.5)), 0);
    EXPECT_EQ((p.gas + p.carrier).compare(p.bulk), 0);
    const std::array<OrientedFaceSweep, 1> faces{{{sweep, 1}}};
    auto budget = ExactDonorExport::make(*cube, faces);
    EXPECT_EQ((budget.exported_volume() + budget.retained_volume()).compare(ExactGeometryMeasure::volume(*cube)), 0);
    const auto tiny = budget.retained_phases({{1, 0, 0}, {0, 0, 0}, 1e-200});
    EXPECT_EQ(tiny.gas.compare(ExactGeometryMeasure::from_value(1e-200)), 0);
    EXPECT_EQ(tiny.gas.certified().as_double(), 1e-200);
    EXPECT_EQ((tiny.gas + tiny.carrier).compare(tiny.bulk), 0);
    const auto outside = budget.exported_phases({{1, 0, 0}, {0, 0, 0}, 1e-200});
    EXPECT_EQ(outside.gas.sign(), 0);
}
TEST(ExactSweptGeometry, CrossedNormalTracesKeepBothSignsAndClipActualDonorsBeforePhases)
{
    auto cube = ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1});
    auto face = face_at(cube, 2, 1);
    std::vector<ExactNodeTrajectory> paths;
    for (const auto& identity : face->vertex_identities())
    {
        const auto p = exact_point_display(identity);
        paths.push_back(ExactNodeTrajectory::backtrace(identity, {0, 0, p[0] < .5 ? -.2 : .2}, .25));
    }
    auto sweep = ExactSweptFace::make(*face, paths, 0.);
    unsigned positive = 0, negative = 0;
    for (const auto& piece : sweep->simplices())
        (piece.sign > 0 ? positive : negative)++;
    EXPECT_GT(positive, 0u);
    EXPECT_GT(negative, 0u);
    EXPECT_EQ(sweep->volume().sign(), 0);
    std::array<ExactSweepDonor, 2> donors{
        {{0, ConvexCellGeometry::box({-1, -1, 0}, {.5, 2, 2})}, {1, ConvexCellGeometry::box({.5, -1, 0}, {2, 2, 2})}}};
    auto pieces = sweep->partition(donors);
    const auto phase = ExactSweptFace::phases(pieces, {{1, 0, 0}, {0, 0, 0}, .5});
    EXPECT_EQ(phase.gas.compare(ExactGeometryMeasure::from_value(-.2).scaled(.25).scaled(.25)), 0);
    EXPECT_EQ((phase.gas + phase.carrier).sign(), 0);
    EXPECT_THROW(sweep->partition(std::span<const ExactSweepDonor>(donors.data(), 1)), std::invalid_argument);
}
TEST(ExactSweptGeometry, PrivateVolumeCorrectionPreservesSharedEdgeTimeTriangles)
{
    const auto cube = ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1});
    auto a = face_at(cube, 0, 1), b = face_at(cube, 1, 1);
    auto first = ExactSweptFace::make(*a, traces(a, {.2, .1, .05}, .3), .055);
    auto other = ExactSweptFace::make(*b, traces(b, {.2, .1, .05}, .3), .025);
    EXPECT_EQ(first->volume().compare(ExactGeometryMeasure::from_value(.055)), 0);
    EXPECT_NE(first->volume().compare(first->uncorrected_volume()), 0);
    using Triangle = std::array<std::vector<char>, 3>;
    std::vector<Triangle> left, right;
    for (auto t : first->edge_time_triangles())
    {
        std::sort(t.begin(), t.end());
        left.push_back(std::move(t));
    }
    for (auto t : other->edge_time_triangles())
    {
        std::sort(t.begin(), t.end());
        right.push_back(std::move(t));
    }
    unsigned matches = 0;
    for (const auto& t : left)
        matches += std::find(right.begin(), right.end(), t) != right.end();
    EXPECT_EQ(matches, 2u);
    auto reversed = a->reversed();
    auto inverse = ExactSweptFace::make(*reversed, traces(reversed, {.2, .1, .05}, .3), -.055);
    EXPECT_EQ(first->corrected_departure_center(), inverse->corrected_departure_center());
    auto broad = ConvexCellGeometry::box({-1, -1, -1}, {2, 2, 2});
    const std::array<ExactSweepDonor, 1> donors{{{0, broad}}};
    auto p = first->partition(donors), q = inverse->partition(donors);
    p.insert(p.end(), q.begin(), q.end());
    const auto phase = ExactSweptFace::phases(p, {{1, 1, 1}, {0, 0, 0}, 1.8});
    EXPECT_EQ(phase.bulk.sign(), 0);
    EXPECT_EQ(phase.gas.sign(), 0);
    EXPECT_EQ(phase.carrier.sign(), 0);
}
struct Loop
{
    std::vector<ExactSweepDonor> donors;
    std::vector<ExactSweepRecipient> targets;
};
Loop circulation(double phi)
{
    Loop result;
    const std::array<GeometryPoint, 4> lower{{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}};
    struct Shared
    {
        std::shared_ptr<const ExactSweptFace> sweep;
        ExactPlanarFacePtr original;
    };
    std::map<std::vector<char>, Shared> shared;
    for (std::size_t id = 0; id < 4; ++id)
    {
        auto lo = lower[id];
        auto cell = ConvexCellGeometry::box(lo, {lo[0] + 1, lo[1] + 1, 1});
        result.donors.push_back({id, cell});
        ExactSweepRecipient target{id, cell, {}};
        for (std::size_t f = 0; f < cell->num_faces(); ++f)
        {
            auto face = cell->face_geometry(f);
            auto key = std::min(face->encode(), face->reversed()->encode());
            auto found = shared.find(key);
            int orientation = 1;
            if (found == shared.end())
            {
                auto p = face->centroid();
                double flux = 0;
                if (id == 0 && p.x == 1)
                    flux = phi;
                else if (id == 0 && p.y == 1)
                    flux = -phi;
                else if (id == 1 && p.y == 1)
                    flux = phi;
                else if (id == 2 && p.x == 1)
                    flux = phi;
                auto sweep = ExactSweptFace::make(*face, traces(face, {0, 0, 0}, 0), flux);
                found = shared.emplace(key, Shared{sweep, face}).first;
            }
            else
                orientation = face->oppositely_oriented(*found->second.original) ? -1 : 1;
            target.outward_faces.push_back({found->second.sweep, orientation});
        }
        result.targets.push_back(std::move(target));
    }
    return result;
}
TEST(ExactSweptGeometry, ClosedCellPullbackProducesPositiveRoutedTransfersAndOwnerScopedColumnProof)
{
    const double amount = .01;
    auto loop = circulation(amount);
    auto routed = ExactTransportPartition::make(loop.donors, loop.targets);
    EXPECT_GT(routed.atoms().size(), 4u);
    for (const auto& donor : loop.donors)
    {
        ExactGeometryMeasure exported, retained;
        std::vector<ExactTransportAtom> column;
        for (const auto& atom : routed.atoms())
            if (atom.donor == donor.id)
            {
                column.push_back(atom);
                if (atom.recipient == donor.id)
                    retained = retained + atom.geometry->volume();
                else
                {
                    EXPECT_EQ(atom.recipient, (donor.id + 1) % 4);
                    exported = exported + atom.geometry->volume();
                }
            }
        EXPECT_EQ(exported.compare(ExactGeometryMeasure::from_value(amount)), 0);
        EXPECT_EQ((exported + retained).compare(ExactGeometryMeasure::volume(*donor.geometry)), 0);
        EXPECT_NO_THROW(ExactTransportPartition::certify_donor_cover(donor, column));
    }
    auto local =
        ExactTransportPartition::local(loop.donors, std::span<const ExactSweepRecipient>(loop.targets.data(), 1));
    for (const auto& atom : local.atoms())
        EXPECT_EQ(atom.recipient, 0u);
    EXPECT_THROW(ExactTransportPartition::certify_donor_cover(loop.donors[1], local.atoms()), std::invalid_argument);
}
TEST(ExactSweptGeometry, RealGapsOverlapsDuplicateExportsAndFiniteBudgetsFailAtomically)
{
    auto cube = ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1});
    auto face = face_at(cube, 0, 1);
    auto sweep = ExactSweptFace::make(*face, traces(face, {1, 0, 0}, .1), .1);
    const std::array<ExactSweepDonor, 2> overlaps{{{0, cube}, {1, cube}}};
    EXPECT_THROW(sweep->partition(overlaps), std::invalid_argument);
    const std::array<OrientedFaceSweep, 2> doubled{{{sweep, 1}, {sweep, 1}}};
    EXPECT_THROW(ExactDonorExport::make(*cube, doubled), std::invalid_argument);
    ExactSweepLimits limited;
    limited.byte_budget = 1;
    EXPECT_THROW(ExactSweptFace::make(*face, traces(face, {1, 0, 0}, .1), .1, limited), std::invalid_argument);
    limited = {};
    limited.maximum_atoms = 1;
    EXPECT_THROW(ExactDonorExport::make(*cube, std::span<const OrientedFaceSweep>(doubled.data(), 1), limited),
        std::invalid_argument);
    EXPECT_EQ(sweep->volume().compare(ExactGeometryMeasure::from_value(.1)), 0);
    auto loop = circulation(.01);
    loop.targets[0].outward_faces.clear();
    EXPECT_THROW(ExactTransportPartition::local(loop.donors, loop.targets), std::invalid_argument);
}
TEST(ExactSweptGeometry, SubnormalAggregateDoesNotRequireRepresentableIndividualSimplices)
{
    const auto cube = ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1});
    const auto face = face_at(cube, 0, 1);
    const auto tiny = std::numeric_limits<double>::denorm_min();
    auto sweep = ExactSweptFace::make(*face, traces(face, {0, 0, 0}, 0), tiny);
    ASSERT_GT(sweep->simplices().size(), 1u);
    ExactGeometryMeasure total;
    for (const auto& piece : sweep->simplices())
    {
        EXPECT_GT(piece.geometry->volume().sign(), 0);
        EXPECT_THROW(piece.geometry->volume().certified().as_double(), std::runtime_error);
        total = total + piece.geometry->volume().scaled(piece.sign);
        const auto copy = ExactConvexFragment::decode(piece.geometry->encode());
        EXPECT_EQ(copy->volume().compare(piece.geometry->volume()), 0);
    }
    EXPECT_EQ(total.certified().as_double(), tiny);
    const std::array<ExactSweepDonor, 1> donors{{{0, cube}}};
    const auto parts = sweep->partition(donors);
    const auto phase = ExactSweptFace::phases(parts, {{1, 0, 0}, {0, 0, 0}, 2});
    EXPECT_EQ(phase.gas.certified().as_double(), tiny);
    EXPECT_EQ(phase.carrier.sign(), 0);
}

TEST(ExactSweptGeometry, TranslatedShearUsesActualCellCoordinatesAndExactTinyRemainder)
{
    const GeometryPoint origin{1e6, -2e6, 3e6};
    const double shear = .25, t = .125;
    std::vector<GeometryPoint> nodes;
    for (unsigned z = 0; z < 2; ++z)
        for (unsigned y = 0; y < 2; ++y)
            for (unsigned x = 0; x < 2; ++x)
                nodes.push_back({origin[0] + x + shear * y, origin[1] + y, origin[2] + z});
    const std::vector<std::vector<std::size_t>> faces{
        {0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}};
    auto cell = ConvexCellGeometry::make(nodes, faces);
    auto face = face_at(cell, 0, origin[0] + 1 + shear * .5);
    auto sweep = ExactSweptFace::make(*face, traces(face, {1, 0, 0}, t), t);
    const std::array<ExactSweepDonor, 1> donors{{{0, cell}}};
    const auto parts = sweep->partition(donors);
    const auto oblique = ExactSweptFace::phases(parts, {{1, 1 - shear, 0}, origin, 1});
    EXPECT_EQ(oblique.gas.compare(ExactGeometryMeasure::from_value(t).scaled(t).scaled(.5)), 0);
    EXPECT_EQ((oblique.gas + oblique.carrier).compare(ExactGeometryMeasure::from_value(t)), 0);
    const std::array<OrientedFaceSweep, 1> outgoing{{{sweep, 1}}};
    const auto budget = ExactDonorExport::make(*cell, outgoing);
    const auto retained = budget.retained_phases({{1, -shear, 0}, origin, 1e-200});
    EXPECT_EQ(retained.gas.compare(ExactGeometryMeasure::from_value(1e-200)), 0);
    EXPECT_EQ(retained.gas.certified().as_double(), 1e-200);
}

TEST(ExactSweptGeometry, ExactMinorityAcceptanceAndLogRatiosDoNotRequireWideFloatingExponents)
{
    const auto target = ExactGeometryMeasure::from_value(std::numeric_limits<double>::denorm_min());
    const auto perturbation = target.scaled(1e-200);
    const auto candidate = target + perturbation;
    EXPECT_TRUE(candidate.relative_error_at_most(target, 2e-200));
    EXPECT_FALSE(candidate.relative_error_at_most(target, 5e-201));
    EXPECT_NEAR(perturbation.log_ratio(target), std::log(1e-200), 2e-12);
    EXPECT_NEAR(target.log_ratio(perturbation), -std::log(1e-200), 2e-12);
    const auto residual = perturbation.enclosure();
    EXPECT_TRUE(residual.exact_nonzero);
    if (residual.value == 0)
        EXPECT_GT(residual.absolute_error, 0);
    const auto ultra = ExactGeometryMeasure::from_value(1e-200).scaled(1e-200);
    EXPECT_NEAR(ultra.log_ratio(ExactGeometryMeasure::from_value(1)), 2 * std::log(1e-200), 2e-12);
    EXPECT_TRUE(ExactGeometryMeasure{}.relative_error_at_most(ExactGeometryMeasure{}, 0));
    EXPECT_FALSE(ultra.relative_error_at_most(ExactGeometryMeasure{}, 1));
    EXPECT_THROW(target.log_ratio(ExactGeometryMeasure{}), std::invalid_argument);
    EXPECT_THROW(candidate.relative_error_at_most(target, -1), std::invalid_argument);
}

TEST(ExactSweptGeometry, ExactQuotientsSquaresAndRetainedCapacityDoNotNarrowPhysicalResiduals)
{
    const auto small = ExactGeometryMeasure::from_value(1e-200);
    const auto square = small.squared();
    EXPECT_GT(square.sign(), 0);
    EXPECT_EQ(square.divided_by(small).compare(small), 0);
    EXPECT_THROW(square.certified().as_double(), std::runtime_error);
    const auto large = ExactGeometryMeasure::from_value(1e200);
    const auto ratio = large.divided_by(small);
    EXPECT_GT(ratio.sign(), 0);
    EXPECT_THROW(ratio.certified().as_double(), std::runtime_error);
    EXPECT_EQ(ratio.divided_by(large).compare(ExactGeometryMeasure::from_value(1).divided_by(small)), 0);
    EXPECT_EQ(ExactGeometryMeasure::from_value(-2).squared().compare(ExactGeometryMeasure::from_value(4)), 0);
    EXPECT_THROW(small.divided_by(ExactGeometryMeasure{}), std::invalid_argument);
    EXPECT_THROW(small.squared(1), std::invalid_argument);
    EXPECT_THROW(large.divided_by(small, 1), std::invalid_argument);
    const auto copy = square;
    EXPECT_EQ(copy.storage_identity(), square.storage_identity());
    EXPECT_EQ(copy.storage_bytes(), square.storage_bytes());
    EXPECT_GT(square.storage_bytes(), sizeof(square));
    auto cell = ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1});
    auto fragment = ExactConvexFragment::from_geometry(*cell);
    EXPECT_GT(fragment->storage_bytes(), sizeof(ExactConvexFragment));
    auto decoded = ExactConvexFragment::decode(fragment->encode());
    EXPECT_NE(decoded->storage_identity(), fragment->storage_identity());
    EXPECT_EQ(decoded->volume().compare(fragment->volume()), 0);
}

TEST(ExactSweptGeometry, FreshExactCompactionChargesOverlapAndPreservesSourceAndBelowRangeValues)
{
    using M = ExactGeometryMeasure;
    auto huge = M::from_value(2);
    for (unsigned i = 0; i < 13; ++i)
        huge = huge.squared();
    const auto slack = (huge + M::from_value(3)) - huge;
    const auto original_identity = slack.storage_identity();
    const auto original_bytes = slack.storage_bytes();
    const auto original_wire = slack.encode();
    EXPECT_THROW((void) slack.compacted(original_bytes), std::invalid_argument);
    EXPECT_EQ(slack.storage_identity(), original_identity);
    EXPECT_EQ(slack.storage_bytes(), original_bytes);
    EXPECT_EQ(slack.encode(), original_wire);
    const auto compact = slack.compacted(2 * original_bytes + 1024);
    EXPECT_EQ(compact.compare(M::from_value(3)), 0);
    EXPECT_NE(compact.storage_identity(), original_identity);
    EXPECT_LT(compact.storage_bytes(), original_bytes);
    EXPECT_EQ(compact.encode(), original_wire);
    EXPECT_EQ(slack.storage_identity(), original_identity);
    EXPECT_EQ(slack.storage_bytes(), original_bytes);
    const auto tiny = M::from_value(1e-200).squared();
    const auto tiny_compact = tiny.compacted(2 * tiny.storage_bytes() + 1024);
    EXPECT_EQ(tiny_compact.compare(tiny), 0);
    EXPECT_GT(tiny_compact.sign(), 0);
    EXPECT_THROW((void) tiny_compact.certified().as_double(), std::runtime_error);
}

TEST(ExactSweptGeometry, ExactWallNullspaceHandlesSlantedPlanesEdgesAndCorners)
{
    using M = ExactGeometryMeasure;
    const std::array<GeometryPoint, 3> points{{{0, 0, 0}, {1, -1, 0}, {1, 0, -1}}};
    const auto wall = ExactPlanarFace::make(points);
    const auto origin = wall->vertex_identities().front();
    const std::array<ExactPlanarFacePtr, 1> one{{wall}};
    const auto plane = ExactNodeTrajectory::backtrace_constrained(origin, {1, 0, 0}, 1, one);
    const auto third = M::from_value(1).divided_by(M::from_value(3));
    EXPECT_EQ(plane.displacement_squared().compare(third.scaled(2)), 0);
    // Translate back by the analytic rational result, not a rounded 2/3.
    const auto x = ExactNodeTrajectory::backtrace(origin, {1, 0, 0}, 1);
    EXPECT_LT(plane.displacement_squared().compare(x.displacement_squared()), 0);
    const auto other = ExactPlanarFace::make(std::array<GeometryPoint, 3>{{{0, 0, 0}, {1, -1, 0}, {0, 0, 1}}});
    const std::array<ExactPlanarFacePtr, 2> two{{wall, other}};
    const auto edge = ExactNodeTrajectory::backtrace_constrained(origin, {1, 0, 0}, 1, two);
    // Common tangent of x+y+z=0 and x+y=0 is (1,-1,0).
    EXPECT_EQ(edge.displacement_squared().compare(M::from_value(.5)), 0);
    const auto third_wall = ExactPlanarFace::make(std::array<GeometryPoint, 3>{{{0, 0, 0}, {0, 1, 0}, {0, 0, 1}}});
    const std::array<ExactPlanarFacePtr, 3> three{{wall, other, third_wall}};
    const auto corner = ExactNodeTrajectory::backtrace_constrained(origin, {1, 2, 3}, 1, three);
    EXPECT_EQ(corner.departure, corner.point);
    EXPECT_EQ(corner.displacement_squared().sign(), 0);
    const std::array<ExactPlanarFacePtr, 4> duplicate{{wall->reversed(), other, wall, other->reversed()}};
    EXPECT_EQ(ExactNodeTrajectory::backtrace_constrained(origin, {1, 0, 0}, 1, duplicate).departure, edge.departure);
    const auto almost_parallel =
        ExactPlanarFace::make(std::array<GeometryPoint, 3>{{{0, 0, 0}, {1, -1, 0}, {1 + std::ldexp(1., -40), 0, -1}}});
    const std::array<ExactPlanarFacePtr, 2> distinct{{wall, almost_parallel}};
    const auto narrow_edge = ExactNodeTrajectory::backtrace_constrained(origin, {1, 0, 0}, 1, distinct);
    EXPECT_EQ(narrow_edge.displacement_squared().compare(M::from_value(.5)), 0);
    EXPECT_EQ(narrow_edge.departure, edge.departure);
    const std::span<const ExactPlanarFacePtr> none;
    EXPECT_EQ(ExactNodeTrajectory::backtrace_constrained(origin, {1, 2, 3}, .25, none).departure,
        ExactNodeTrajectory::backtrace(origin, {1, 2, 3}, .25).departure);
}
TEST(ExactSweptGeometry, ExactWallTrajectoriesRetainSubnormalDisplacementsAndRejectInvalidProofs)
{
    using M = ExactGeometryMeasure;
    const auto wall = ExactPlanarFace::make(std::array<GeometryPoint, 3>{{{0, 0, 0}, {1, -1, 0}, {1, 0, -1}}});
    const auto origin = wall->vertex_identities().front();
    const std::array<ExactPlanarFacePtr, 1> walls{{wall}};
    const double eta = std::numeric_limits<double>::denorm_min();
    const auto tiny = ExactNodeTrajectory::backtrace_constrained(origin, {eta, 0, 0}, 1, walls);
    EXPECT_NE(tiny.departure, tiny.point);
    EXPECT_EQ(
        tiny.displacement_squared().compare(M::from_value(eta).squared().scaled(2).divided_by(M::from_value(3))), 0);
    EXPECT_TRUE(tiny.displacement_squared().enclosure().exact_nonzero);
    const GeometryPoint shift{1e6, -2e6, 3e6};
    const auto translated_wall = wall->translated(shift);
    const auto translated_origin = exact_point_translated(origin, exact_displacement({0, 0, 0}, shift));
    const std::array<ExactPlanarFacePtr, 1> translated_walls{{translated_wall}};
    const auto translated =
        ExactNodeTrajectory::backtrace_constrained(translated_origin, {eta, 0, 0}, 1, translated_walls);
    EXPECT_EQ(translated.displacement_squared().compare(tiny.displacement_squared()), 0);
    EXPECT_EQ(translated.departure, exact_point_translated(tiny.departure, exact_displacement({0, 0, 0}, shift)));
    const auto off_plane = ExactNodeTrajectory::backtrace(origin, {1e-300, 0, 0}, 1).departure;
    EXPECT_THROW(ExactNodeTrajectory::backtrace_constrained(off_plane, {1, 0, 0}, 1, walls), std::invalid_argument);
    const auto outside = ExactNodeTrajectory::backtrace(origin, {-4, 4, 0}, 1).departure;
    EXPECT_THROW(ExactNodeTrajectory::backtrace_constrained(outside, {1, 0, 0}, 1, walls), std::invalid_argument);
    EXPECT_THROW(ExactNodeTrajectory::backtrace_constrained(origin, {1, 0, 0}, -1, walls), std::invalid_argument);
    EXPECT_THROW(ExactNodeTrajectory::backtrace_constrained(origin, {INFINITY, 0, 0}, 1, walls), std::invalid_argument);
    ExactTrajectoryLimits limits;
    limits.maximum_constraints = 0;
    EXPECT_THROW(
        ExactNodeTrajectory::backtrace_constrained(origin, {1, 0, 0}, 1, walls, limits), std::invalid_argument);
    limits = {};
    limits.maximum_integer_bits = 8;
    EXPECT_THROW(
        ExactNodeTrajectory::backtrace_constrained(origin, {eta, 0, 0}, 1, walls, limits), std::invalid_argument);
    limits = {};
    limits.byte_budget = 1;
    EXPECT_THROW(
        ExactNodeTrajectory::backtrace_constrained(origin, {1, 0, 0}, 1, walls, limits), std::invalid_argument);
    limits = {};
    limits.maximum_operations = 1;
    EXPECT_THROW(
        ExactNodeTrajectory::backtrace_constrained(origin, {1, 0, 0}, 1, walls, limits), std::invalid_argument);
}
TEST(ExactSweptGeometry, SupportingPlanesPreserveContactsOverlapAndSubnormalDonorColumns)
{
    const auto cell = ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1});
    const ExactSweepDonor donor{37, cell};
    std::vector<ExactTransportAtom> octants;
    for (unsigned z = 0; z < 2; ++z)
        for (unsigned y = 0; y < 2; ++y)
            for (unsigned x = 0; x < 2; ++x)
            {
                const auto part =
                    ConvexCellGeometry::box({.5 * x, .5 * y, .5 * z}, {.5 * (x + 1), .5 * (y + 1), .5 * (z + 1)});
                octants.push_back({37, octants.size(), ExactConvexFragment::from_geometry(*part)});
            }
    // All pairs meet along exact faces, edges, vertices, or are separated.
    EXPECT_NO_THROW(ExactTransportPartition::certify_donor_cover(donor, octants));
    std::reverse(octants.begin(), octants.end());
    EXPECT_NO_THROW(ExactTransportPartition::certify_donor_cover(donor, octants));
    for (const double width : {std::ldexp(1., -40), 1e-310})
    {
        const auto tiny = ExactConvexFragment::from_geometry(*ConvexCellGeometry::box({0, 0, 0}, {width, 1, 1}));
        const auto remainder = ExactConvexFragment::from_geometry(*ConvexCellGeometry::box({width, 0, 0}, {1, 1, 1}));
        std::vector<ExactTransportAtom> column{{37, 0, tiny}, {37, 1, remainder}};
        EXPECT_NO_THROW(ExactTransportPartition::certify_donor_cover(donor, column));
        column.push_back({37, 2, tiny});
        // Any strictly positive overlap, even subnormal, still falls through
        // to exact clipping and rejects; no zero-contact tolerance exists.
        EXPECT_THROW(ExactTransportPartition::certify_donor_cover(donor, column), std::invalid_argument);
        column = {{37, 0, remainder}};
        EXPECT_THROW(ExactTransportPartition::certify_donor_cover(donor, column), std::invalid_argument);
    }
}
TEST(ExactSweptGeometry, SupportingPlanesRetainActualCirculationDonorFragmentCover)
{
    using M = ExactGeometryMeasure;
    auto loop = circulation(1. / 64);
    const auto routed = ExactTransportPartition::local(loop.donors, loop.targets);
    ASSERT_GT(routed.atoms().size(), loop.donors.size());
    for (const auto& donor : loop.donors)
    {
        std::vector<ExactTransportAtom> column;
        M volume;
        for (const auto& atom : routed.atoms())
            if (atom.donor == donor.id)
            {
                column.push_back(atom);
                volume = volume + atom.geometry->volume();
            }
        ASSERT_GT(column.size(), 1u);
        EXPECT_EQ(volume.compare(M::volume(*donor.geometry)), 0);
        // These are actual signed-sweep/recipient-proof fragments, not an
        // arbitrary spatial box catalog. The unchanged cover proof checks
        // containment, pairwise overlap and independent first moments.
        EXPECT_NO_THROW(ExactTransportPartition::certify_donor_cover(donor, column));
        std::reverse(column.begin(), column.end());
        EXPECT_NO_THROW(ExactTransportPartition::certify_donor_cover(donor, column));
    }
}

TEST(ExactSweptGeometry, SupportingPlanesHandleCollinearPrefixAndMomentBalancedOverlap)
{
    const ExactSweepDonor donor{37, ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1})};
    const std::vector<GeometryPoint> points{
        {0, 0, 0}, {.5, 0, 0}, {0, 1, 0}, {.5, 1, 0}, {0, 0, 1}, {.5, 0, 1}, {0, 1, 1}, {.5, 1, 1}, {.5, .5, 0}};
    // The separating x-max polygon begins with three collinear points. Its
    // first fan triangle is zero; its next fan triangle fixes the outward sign.
    const std::vector<std::vector<std::size_t>> faces{
        {1, 8, 3, 7, 5}, {0, 4, 6, 2}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 8, 1}, {4, 5, 7, 6}};
    const auto left = ExactConvexFragment::from_geometry(*ConvexCellGeometry::make(points, faces));
    const auto right = ExactConvexFragment::from_geometry(*ConvexCellGeometry::box({.5, 0, 0}, {1, 1, 1}));
    std::vector<ExactTransportAtom> column{{37, 0, right}, {37, 1, left}};
    EXPECT_NO_THROW(ExactTransportPartition::certify_donor_cover(donor, column));
    std::reverse(column.begin(), column.end());
    EXPECT_NO_THROW(ExactTransportPartition::certify_donor_cover(donor, column));

    // Two copies of the central half have exactly the donor's total volume
    // AND first moments: symmetry puts both centroids at (1/2,1/2,1/2).
    // The central overlap and the two omitted outer quarters cannot be caught
    // by aggregate moments. The independent positive-intersection proof must
    // reject them, including when the copies have different recipient IDs.
    const auto middle = ExactConvexFragment::from_geometry(*ConvexCellGeometry::box({.25, 0, 0}, {.75, 1, 1}));
    EXPECT_EQ((middle->volume() + middle->volume()).compare(ExactGeometryMeasure::volume(*donor.geometry)), 0);
    column = {{37, 0, middle}, {37, 1, middle}};
    EXPECT_THROW(ExactTransportPartition::certify_donor_cover(donor, column), std::invalid_argument);
}

} // namespace
