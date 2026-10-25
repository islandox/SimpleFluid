/** @file testDepartureFan.cc
 * @brief Independent positive-departure and signed-overlay fallback oracles.
 */
#include "geometry/PLIC/SweptGeometry.hh"
#include "utils/testing_environment.hh"
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <map>
namespace
{
using namespace SimpleFluid::Meshes;
using M = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
M product(const M& a, const M& b)
{
    return !a.sign() || !b.sign() ? M{} : a.divided_by(M::from_value(1).divided_by(b));
}
using V = std::array<M, 3>;
V exact_point(SimpleFluid::PLIC::GeometryPoint p)
{
    return {M::from_value(p[0]), M::from_value(p[1]), M::from_value(p[2])};
}
M determinant(const V& a, const V& b, const V& c)
{
    M q;
    for (unsigned i = 0; i < 3; ++i)
        q = q + product(a[i], product(b[(i + 1) % 3], c[(i + 2) % 3]) - product(b[(i + 2) % 3], c[(i + 1) % 3]));
    return q;
}
// Independent divergence integral on explicitly supplied DYADIC fixture
// endpoints. No rounded endpoint extraction is used by the tiny-motion oracle.
M dyadic_fixture_flux(const std::vector<SimpleFluid::PLIC::NodeTrajectory<SimpleFluid::ArithmeticMode::Exact>>& traces)
{
    std::vector<V> a, b;
    V center{}, back{};
    for (const auto& t : traces)
    {
        a.push_back(exact_point(SimpleFluid::PLIC::point_display<SimpleFluid::ArithmeticMode::Exact>(t.point)));
        b.push_back(exact_point(SimpleFluid::PLIC::point_display<SimpleFluid::ArithmeticMode::Exact>(t.departure)));
        for (unsigned k = 0; k < 3; ++k)
        {
            center[k] = center[k] + a.back()[k];
            back[k] = back[k] + b.back()[k];
        }
    }
    for (unsigned k = 0; k < 3; ++k)
    {
        center[k] = center[k].divided_by(M::from_value(double(a.size())));
        back[k] = back[k].divided_by(M::from_value(double(a.size())));
    }
    M q;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        auto j = (i + 1) % a.size();
        q = q + determinant(center, a[i], a[j]) + determinant(back, b[j], b[i]);
        if (traces[i].point < traces[j].point)
            q = q + determinant(a[i], b[i], b[j]) + determinant(a[i], b[j], a[j]);
        else
            q = q + determinant(a[j], a[i], b[i]) + determinant(a[j], b[i], b[j]);
    }
    return q.divided_by(M::from_value(6));
}
SimpleFluid::PLIC::SweepRecipient<SimpleFluid::ArithmeticMode::Exact> recipient(std::uint64_t id,
    SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Exact> native,
    const std::vector<SimpleFluid::PLIC::PlanarFacePtr<SimpleFluid::ArithmeticMode::Exact>>& faces,
    const std::function<SimpleFluid::PLIC::GeometryPoint(SimpleFluid::PLIC::GeometryPoint)>& velocity,
    const std::function<M(const SimpleFluid::PLIC::PlanarFace<SimpleFluid::ArithmeticMode::Exact>&,
        const std::vector<SimpleFluid::PLIC::NodeTrajectory<SimpleFluid::ArithmeticMode::Exact>>&)>& flux = {})
{
    SimpleFluid::PLIC::SweepRecipient<SimpleFluid::ArithmeticMode::Exact> r;
    r.id = id;
    r.geometry = native;
    for (const auto& face : faces)
    {
        std::vector<SimpleFluid::PLIC::NodeTrajectory<SimpleFluid::ArithmeticMode::Exact>> traces;
        for (const auto& point : face->vertex_identities())
            traces.push_back(SimpleFluid::PLIC::NodeTrajectory<SimpleFluid::ArithmeticMode::Exact>::backtrace(
                point, velocity(SimpleFluid::PLIC::point_display<SimpleFluid::ArithmeticMode::Exact>(point)), 1));
        const auto q = flux ? flux(*face, traces) : dyadic_fixture_flux(traces);
        auto sweep = SimpleFluid::PLIC::SweptFace<SimpleFluid::ArithmeticMode::Exact>::make(*face, traces, q);
        EXPECT_EQ(sweep->uncorrected_volume().compare(q), 0);
        r.outward_faces.push_back({sweep, 1});
    }
    return r;
}
std::vector<SimpleFluid::PLIC::PlanarFacePtr<SimpleFluid::ArithmeticMode::Exact>> faces(
    const SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Exact>& g)
{
    std::vector<SimpleFluid::PLIC::PlanarFacePtr<SimpleFluid::ArithmeticMode::Exact>> result;
    for (std::size_t f = 0; f < g->num_faces(); ++f)
        result.push_back(g->face_geometry(f));
    return result;
}
SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Exact> extruded(
    const std::vector<SimpleFluid::PLIC::GeometryPoint>& base)
{
    std::vector<SimpleFluid::PLIC::GeometryPoint> points;
    for (unsigned z = 0; z < 2; ++z)
        for (auto p : base)
        {
            p[2] = z;
            points.push_back(p);
        }
    std::vector<std::vector<std::size_t>> fs(2);
    const auto n = base.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        fs[0].push_back(n - 1 - i);
        fs[1].push_back(n + i);
        fs.push_back({i, (i + 1) % n, (i + 1) % n + n, i + n});
    }
    return SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::make(points, fs);
}
// Fully affine actual donor boxes; each skew transform has determinant1.
SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Exact> affine_box(
    SimpleFluid::PLIC::GeometryPoint lo, SimpleFluid::PLIC::GeometryPoint hi, double s = 0)
{
    std::vector<SimpleFluid::PLIC::GeometryPoint> p;
    for (unsigned z = 0; z < 2; ++z)
        for (unsigned y = 0; y < 2; ++y)
            for (unsigned x = 0; x < 2; ++x)
            {
                double yy = y ? hi[1] : lo[1];
                p.push_back({(x ? hi[0] : lo[0]) + s * yy, yy, z ? hi[2] : lo[2]});
            }
    const std::vector<std::vector<std::size_t>> f{
        {0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}};
    return SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::make(p, f);
}
std::vector<SimpleFluid::PLIC::SweepDonor<SimpleFluid::ArithmeticMode::Exact>> donor_grid(double shear = 0)
{
    std::vector<SimpleFluid::PLIC::SweepDonor<SimpleFluid::ArithmeticMode::Exact>> d;
    for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 1; ++y)
            for (int z = -1; z <= 1; ++z)
                d.push_back({std::uint64_t((x + 1) * 9 + (y + 1) * 3 + z + 1),
                    affine_box(
                        {double(x), double(y), double(z)}, {double(x + 1), double(y + 1), double(z + 1)}, shear)});
    return d;
}
M donor_volume(const SimpleFluid::PLIC::TransportPartition<SimpleFluid::ArithmeticMode::Exact>& p, std::uint64_t id)
{
    M q;
    for (const auto& a : p.atoms())
        if (a.donor == id)
            q = q + a.geometry->volume();
    return q;
}
TEST(DepartureFan, ConstantTranslationShearSkewAndCornerUseCertifiedPositiveFans)
{
    auto donors = donor_grid(.25);
    auto native = donors[13].geometry;
    const SimpleFluid::PLIC::GeometryPoint logical{.25, .125, .0625},
        u{logical[0] + .25 * logical[1], logical[1], logical[2]};
    auto translated = recipient(
        13, native, faces(native), [&](auto) { return u; },
        [&](const auto& f, const auto&) { return f.area_vector_dot(u); });
    auto result = SimpleFluid::PLIC::TransportPartition<SimpleFluid::ArithmeticMode::Exact>::local(
        donors, std::span(&translated, 1));
    EXPECT_TRUE(result.used_departure_fan());
    const auto own = product(product(M::from_value(.75), M::from_value(.875)), M::from_value(.9375));
    EXPECT_EQ(donor_volume(result, 13).compare(own), 0);
    M total;
    for (const auto& a : result.atoms())
        total = total + a.geometry->volume();
    EXPECT_EQ(total.compare(M::from_value(1)), 0);
    auto sheared = recipient(
        13, native, faces(native), [](auto p) { return SimpleFluid::PLIC::GeometryPoint{-.125 * p[1], 0, 0}; });
    auto shear_result = SimpleFluid::PLIC::TransportPartition<SimpleFluid::ArithmeticMode::Exact>::local(
        donors, std::span(&sheared, 1));
    EXPECT_TRUE(shear_result.used_departure_fan());
    EXPECT_EQ(donor_volume(shear_result, 13).compare(M::from_value(15.0 / 16)), 0);
    EXPECT_EQ(donor_volume(shear_result, 22).compare(M::from_value(1.0 / 16)), 0);
}
TEST(DepartureFan, CorrectedDepartureCentersPreserveIndependentFaceFluxWithStationaryNodes)
{
    auto donors = donor_grid();
    auto native = donors[13].geometry;
    SimpleFluid::PLIC::SweepRecipient<SimpleFluid::ArithmeticMode::Exact> r;
    r.id = 13;
    r.geometry = native;
    constexpr double amount = 1.0 / 64;
    for (auto f : faces(native))
    {
        std::vector<SimpleFluid::PLIC::NodeTrajectory<SimpleFluid::ArithmeticMode::Exact>> stationary;
        for (const auto& p : f->vertex_identities())
            stationary.push_back(
                SimpleFluid::PLIC::NodeTrajectory<SimpleFluid::ArithmeticMode::Exact>::backtrace(p, {}, 1));
        const auto q = f->area_vector_dot({amount, 0, 0});
        auto sweep = SimpleFluid::PLIC::SweptFace<SimpleFluid::ArithmeticMode::Exact>::make(*f, stationary, q);
        EXPECT_EQ(sweep->uncorrected_volume().sign(), 0);
        r.outward_faces.push_back({sweep, 1});
    }
    const auto routed =
        SimpleFluid::PLIC::TransportPartition<SimpleFluid::ArithmeticMode::Exact>::local(donors, std::span(&r, 1));
    EXPECT_TRUE(routed.used_departure_fan());
    EXPECT_EQ(donor_volume(routed, 13).compare(M::from_value(1 - amount)), 0);
    EXPECT_EQ(donor_volume(routed, 4).compare(M::from_value(amount)), 0);
}
TEST(DepartureFan, TinyThreeDimensionalMotionRetainsSubnormalCornerAtoms)
{
    auto donors = donor_grid();
    auto native = donors[13].geometry;
    const double eta = std::numeric_limits<double>::denorm_min();
    const SimpleFluid::PLIC::GeometryPoint u{eta, eta, eta};
    auto r = recipient(
        13, native, faces(native), [&](auto) { return u; },
        [&](const auto& f, const auto&) { return f.area_vector_dot(u); });
    auto result =
        SimpleFluid::PLIC::TransportPartition<SimpleFluid::ArithmeticMode::Exact>::local(donors, std::span(&r, 1));
    EXPECT_TRUE(result.used_departure_fan());
    const auto retained = M::from_value(1) - M::from_value(eta);
    EXPECT_EQ(donor_volume(result, 13).compare(product(product(retained, retained), retained)), 0);
    const auto corner = product(product(M::from_value(eta), M::from_value(eta)), M::from_value(eta));
    EXPECT_GT(corner.sign(), 0);
    EXPECT_EQ(donor_volume(result, 0).compare(corner), 0);
    M total;
    for (const auto& a : result.atoms())
        total = total + a.geometry->volume();
    EXPECT_EQ(total.compare(M::from_value(1)), 0);
}
TEST(DepartureFan, CoarseFineCanonicalSubfacesAndEdgeNodesRouteExactDonorVolumes)
{
    auto native = SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::box({0, 0, 0}, {1, 1, 1});
    std::vector<SimpleFluid::PLIC::SweepDonor<SimpleFluid::ArithmeticMode::Exact>> donors{{0, native}};
    std::vector<SimpleFluid::PLIC::PlanarFacePtr<SimpleFluid::ArithmeticMode::Exact>> boundary;
    std::vector<std::vector<char>> nodes;
    for (unsigned y = 0; y < 2; ++y)
        for (unsigned z = 0; z < 2; ++z)
        {
            auto d = SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::box(
                {-1, .5 * y, .5 * z}, {0, .5 * (y + 1), .5 * (z + 1)});
            donors.push_back({donors.size(), d});
            for (auto f : faces(d))
                if (f->centroid().x == 0)
                {
                    auto tile = f->reversed();
                    boundary.push_back(tile);
                    for (auto p : tile->vertex_identities())
                        nodes.push_back(p);
                }
        }
    for (auto f : faces(native))
        if (f->centroid().x != 0)
            boundary.push_back(f->with_edge_vertices(nodes));
    const SimpleFluid::PLIC::GeometryPoint u{.25, 0, 0};
    auto r = recipient(
        0, native, boundary, [&](auto) { return u; }, [&](const auto& f, const auto&) { return f.area_vector_dot(u); });
    auto result =
        SimpleFluid::PLIC::TransportPartition<SimpleFluid::ArithmeticMode::Exact>::local(donors, std::span(&r, 1));
    EXPECT_TRUE(result.used_departure_fan());
    EXPECT_EQ(donor_volume(result, 0).compare(M::from_value(.75)), 0);
    for (unsigned id = 1; id <= 4; ++id)
        EXPECT_EQ(donor_volume(result, id).compare(M::from_value(1.0 / 16)), 0);
}
TEST(DepartureFan, GenuineNonStarUShapeFallsBackToOriginalSignedOverlay)
{
    const std::vector<SimpleFluid::PLIC::GeometryPoint> current{
        {0, 0, 0}, {3, 0, 0}, {3, 1.5, 0}, {3, 3, 0}, {2, 3, 0}, {1, 3, 0}, {0, 3, 0}, {0, 1.5, 0}};
    const std::vector<SimpleFluid::PLIC::GeometryPoint> departure{
        {0, 0, 0}, {3, 0, 0}, {3, 4, 0}, {2, 4, 0}, {2, 1, 0}, {1, 1, 0}, {1, 4, 0}, {0, 4, 0}};
    auto native = extruded(current);
    std::map<SimpleFluid::PLIC::GeometryPoint, SimpleFluid::PLIC::GeometryPoint> velocity;
    for (unsigned z = 0; z < 2; ++z)
        for (std::size_t i = 0; i < current.size(); ++i)
        {
            auto p = current[i];
            p[2] = z;
            velocity[p] = {p[0] - departure[i][0], p[1] - departure[i][1], 0};
        }
    auto r = recipient(1, native, faces(native), [&](auto p) { return velocity.at(p); });
    const std::vector<SimpleFluid::PLIC::SweepDonor<SimpleFluid::ArithmeticMode::Exact>> donors{{1, native},
        {2, SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::box({0, 3, 0}, {3, 4, 1})}};
    auto result =
        SimpleFluid::PLIC::TransportPartition<SimpleFluid::ArithmeticMode::Exact>::local(donors, std::span(&r, 1));
    EXPECT_FALSE(result.used_departure_fan());
    // U area=3*4-1*3=9, and the x<=1 leg has area4, independently.
    const auto phase = SimpleFluid::PLIC::TransportPartition<SimpleFluid::ArithmeticMode::Exact>::phases(
        result.atoms(), {{1, 0, 0}, {0, 0, 0}, 1});
    EXPECT_EQ(phase.bulk.compare(M::from_value(9)), 0);
    EXPECT_EQ(phase.gas.compare(M::from_value(4)), 0);
    EXPECT_EQ(phase.carrier.compare(M::from_value(5)), 0);
}
TEST(DepartureFan, SourceValidationAndAttemptBudgetsRemainStrictAndRecoverable)
{
    auto native = SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::box({0, 0, 0}, {1, 1, 1});
    const std::vector<SimpleFluid::PLIC::SweepDonor<SimpleFluid::ArithmeticMode::Exact>> donors{{0, native}};
    auto r = recipient(0, native, faces(native), [](auto) { return SimpleFluid::PLIC::GeometryPoint{}; });
    SimpleFluid::PLIC::SweepLimits limits;
    limits.maximum_pair_tests = 1;
    EXPECT_THROW((void) SimpleFluid::PLIC::TransportPartition<SimpleFluid::ArithmeticMode::Exact>::local(
                     donors, std::span(&r, 1), limits),
        std::invalid_argument);
    auto recovered =
        SimpleFluid::PLIC::TransportPartition<SimpleFluid::ArithmeticMode::Exact>::local(donors, std::span(&r, 1));
    EXPECT_TRUE(recovered.used_departure_fan());
    EXPECT_EQ(recovered.atoms().size(), 24u);
    auto malformed = r;
    malformed.outward_faces[0].outward_sign = 0;
    EXPECT_THROW((void) SimpleFluid::PLIC::TransportPartition<SimpleFluid::ArithmeticMode::Exact>::local(
                     donors, std::span(&malformed, 1)),
        std::invalid_argument);
    limits = {};
    limits.byte_budget = 1;
    EXPECT_THROW((void) SimpleFluid::PLIC::TransportPartition<SimpleFluid::ArithmeticMode::Exact>::local(
                     donors, std::span(&r, 1), limits),
        std::invalid_argument);
    limits = {};
    limits.maximum_simplices = 1;
    // Each stationary original face has zero signed simplices. This limit
    // must not silently become a total departure-boundary triangle limit.
    EXPECT_TRUE(SimpleFluid::PLIC::TransportPartition<SimpleFluid::ArithmeticMode::Exact>::local(
        donors, std::span(&r, 1), limits)
            .used_departure_fan());
    limits = {};
    limits.maximum_atoms = 1;
    EXPECT_THROW((void) SimpleFluid::PLIC::TransportPartition<SimpleFluid::ArithmeticMode::Exact>::local(
                     donors, std::span(&r, 1), limits),
        std::invalid_argument);
    EXPECT_TRUE(
        SimpleFluid::PLIC::TransportPartition<SimpleFluid::ArithmeticMode::Exact>::local(donors, std::span(&r, 1))
            .used_departure_fan());
}
} // namespace
