#include "FVM/GeneralVoF.hh"
#include "FVM/details/VoFFailure.hh"
#include "geometry/PLIC/PlaneMeasure.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::VoF;
namespace AMR = SimpleFluid::Meshes::AMR;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
constexpr long double shear = .25L, speed = 1.L / 16, dt = 1.L / 64;
using XY = std::array<long double, 2>;
using Polygon = std::vector<XY>;

// Test-local monotone hull, half-plane clipping and shoelace integration. Native
// points supply the cell shape only; no production PLIC cuts/measures form the oracle.
long double cross(XY a, XY b, XY c)
{
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}
Polygon hull(Polygon p)
{
    std::sort(p.begin(), p.end());
    p.erase(std::unique(p.begin(), p.end()), p.end());
    Polygon q;
    for (const auto& a : p)
    {
        while (q.size() > 1 && cross(q[q.size() - 2], q.back(), a) <= 0)
            q.pop_back();
        q.push_back(a);
    }
    const auto lower = q.size();
    for (auto i = p.rbegin() + 1; i != p.rend(); ++i)
    {
        while (q.size() > lower && cross(q[q.size() - 2], q.back(), *i) <= 0)
            q.pop_back();
        q.push_back(*i);
    }
    q.pop_back();
    return q;
}
long double area(const Polygon& p)
{
    long double twice = 0;
    for (std::size_t i = 0; i < p.size(); ++i)
        twice += p[i][0] * p[(i + 1) % p.size()][1] - p[i][1] * p[(i + 1) % p.size()][0];
    return std::abs(twice) / 2;
}
long double clipped_area(const Polygon& p, long double time, bool gas)
{
    Polygon cut;
    const auto distance = [&](XY a)
    {
        const auto d = a[0] + 2 * speed * time * a[1] - (.5L + speed * time);
        return gas ? d : -d;
    };
    for (std::size_t i = 0; i < p.size(); ++i)
    {
        const auto a = p[i], b = p[(i + 1) % p.size()];
        const auto da = distance(a), db = distance(b);
        if (da <= 0)
            cut.push_back(a);
        if ((da < 0 && db > 0) || (da > 0 && db < 0))
        {
            const auto t = da / (da - db);
            cut.push_back({a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])});
        }
    }
    return area(cut);
}
struct ReferenceCell
{
    Polygon xy;
    long double height = 0;
    long double volume() const { return area(xy) * height; }
    long double phase(long double time, bool gas) const { return clipped_area(xy, time, gas) * height; }
};
ReferenceCell reference_cell(const PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>& shape)
{
    Polygon points;
    long double bottom = std::numeric_limits<long double>::max(), top = -bottom;
    for (const auto& p : shape->points())
    {
        points.push_back({p.value[0] - shear * p.value[1], p.value[1]});
        bottom = std::min(bottom, p.value[2]);
        top = std::max(top, p.value[2]);
    }
    return {hull(std::move(points)), top - bottom};
}
std::shared_ptr<const GeneralVoF> motion_fixture(
    GeneralVoFTransportBackend backend = GeneralVoFTransportBackend::SweptPartition)
{
    AMR::ExtrudedAMRGrid grid;
    for (unsigned y = 0; y < 3; ++y)
        for (unsigned x = 0; x < 5; ++x)
            grid.xy_nodes.push_back({x / 4. + double(shear) * y / 2., y / 2., 0});
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
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    for (std::uint64_t c = 0; c < 16; ++c)
        if (int(c % std::min(2, comm->getSize())) == comm->getRank())
            owned.push_back(c);
    AMR::ExtrudedAMR provider(grid);
    auto bootstrap = provider.detach_owned(owned, {'m', 'o', 't', 'i', 'o', 'n'});
    auto raw = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    std::vector<AMR::AdaptiveMark> marks;
    for (const auto& [key, leaf] : raw->owned())
    {
        const auto c = raw->geometry(leaf)->centroid();
        const auto x = c[0] - double(shear) * c[1];
        if (x > .5 && x < .75)
            marks.push_back({key, AMR::AdaptiveSplit::Isotropic});
    }
    const auto refined = raw->apply(raw->plan_adaptation(marks));
    std::vector<AMR::AdaptiveCellKey> order;
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : refined->owned())
    {
        const auto reference = reference_cell(refined->geometry(leaf));
        order.push_back(key);
        gas.push_back(double(reference.phase(0, true)));
        carrier.push_back(double(reference.phase(0, false)));
    }
    int local_empty = order.empty(), empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, std::max(0, comm->getSize() - 2));
    GeneralVoFOptions options;
    options.transport_backend = backend;
    options.phase_cleanup.relative_cutoff = 0;
    return std::make_shared<const GeneralVoF>(refined, std::move(order), std::move(gas), std::move(carrier), options);
}
long double streamfunction(XY p)
{
    const auto f = p[0] < .25L ? 4 * p[0] : p[0] > .75L ? 4 * (1 - p[0]) : 1;
    return speed * f * p[1] * (1 - p[1]);
}
std::vector<GeneralVoFNodeVelocity> velocities(const GeneralVoF& source)
{
    std::vector<GeneralVoFNodeVelocity> result;
    for (const auto& n : source.nodes())
    {
        const long double y = n.coordinate[1], x = n.coordinate[0] - shear * y;
        const auto f = x < .25L ? 4 * x : x > .75L ? 4 * (1 - x) : 1;
        const auto df = x < .25L ? 4 : x > .75L ? -4 : 0;
        const auto v = -speed * df * y * (1 - y), u = speed * f * (1 - 2 * y);
        result.push_back({n.gid, {double(u + shear * v), double(v), 0}});
    }
    return result;
}
std::vector<GeneralVoFFaceVolume> controls(const GeneralVoF& source)
{
    std::vector<GeneralVoFFaceVolume> result;
    for (const auto& tile : source.tiles())
    {
        long double q = 0;
        if (!tile.exterior)
        {
            const auto face = source.finite_face_points<ArithmeticMode::Ordinary>(tile.face_gid);
            Polygon xy;
            long double bottom = std::numeric_limits<long double>::max(), top = -bottom;
            for (const auto& p : face)
            {
                xy.push_back({p.value[0] - shear * p.value[1], p.value[1]});
                bottom = std::min(bottom, p.value[2]);
                top = std::max(top, p.value[2]);
            }
            const auto [a, b] = std::minmax_element(xy.begin(), xy.end());
            const auto dx = (*b)[0] - (*a)[0], dy = (*b)[1] - (*a)[1];
            const auto n = tile.geometry->area_vector();
            const auto orientation = dy * n.x + (-dx - shear * dy) * n.y;
            q = dt * (streamfunction(*b) - streamfunction(*a)) * (top - bottom);
            if (orientation < 0)
                q = -q;
        }
        result.push_back({tile.face_gid, double(q)});
    }
    return result;
}
#include "algebraic_motion_qualification.hh"
#include "algebraic_reconstruction_replay.hh"

TEST(GeneralVoFMotionAccuracy, IndependentPolygonOracleIntegratesKnownShearedCuts)
{
    const Polygon unit{{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    EXPECT_EQ(clipped_area(unit, 0, true), .5L);
    EXPECT_EQ(clipped_area(unit, 0, false), .5L);
    // Unshear a physical parallelogram before applying the physical analytic
    // plane x+(2Ut-s)y <= .5+Ut; the shear has unit determinant.
    Polygon physical{{0, 0}, {1, 0}, {1 + shear, 1}, {shear, 1}};
    for (auto& p : physical)
        p[0] -= shear * p[1];
    EXPECT_EQ(clipped_area(physical, 1.L / 8, true), .5L);
    EXPECT_EQ(clipped_area(physical, 1.L / 8, false), .5L);
    const Polygon patch{{.375L, 0}, {.625L, 0}, {.625L, .5L}, {.375L, .5L}};
    EXPECT_EQ(clipped_area(patch, 1.L / 8, true), 33.L / 512);
    EXPECT_EQ(clipped_area(patch, 1.L / 8, false), 31.L / 512);
}
// Genuine single-interface baseline, separate from the four-edge rotated
// PlanarPatch stress case. Independent polygon integration supplies every phase.
// Keep the existing complete eight-step T=1/8 horizon and all admission gates.
void single_interface_motion(GeneralVoFTransportBackend backend)
{
    if (!std::getenv("SIMPLEFLUID_RUN_ALGEBRAIC_MOTION"))
        GTEST_SKIP() << "Separate single-interface prescribed-motion qualification.";
    testing::Test::RecordProperty("required_endpoint", .125);
    testing::Test::RecordProperty("relative_shape_error_limit", .02);
    testing::Test::RecordProperty("centroid_cell_width_limit", .25);
    testing::Test::RecordProperty("fixture_scope", "separate single-interface baseline; not a PlanarPatch replacement");
    testing::Test::RecordProperty("configured_backend",
        backend == GeneralVoFTransportBackend::GeometricFace ? "GeometricFace" : "SweptPartition");
    if (Tpetra::getDefaultComm()->getRank() == 0)
        std::cout << "single_interface_setup accepted_time=0 required_endpoint=0.125" << std::endl;
    auto source = motion_fixture(backend);
    const auto comm = Tpetra::getDefaultComm();
    testing::Test::RecordProperty("byte_budget", std::to_string(source->options().byte_budget));
    testing::Test::RecordProperty("support_rings", std::to_string(source->options().support_rings));
    testing::Test::RecordProperty("clip_evaluation_cap",
        std::to_string(source->options().reconstruction.maximum_clip_evaluations));
    std::array<double, 2> local_structure{double(source->owned_order().size()), 0}, structure{};
    for (std::size_t f = 0; f < source->mesh()->num_owned_faces(); ++f)
    {
        const auto face = DefaultTpetraTypes::local_ordinal_type(f);
        if (source->mesh()->is_geometry_exterior_face(face))
            continue;
        const auto left = source->mesh()->cell_volume(source->mesh()->owner_cell(face));
        const auto right = source->mesh()->cell_volume(source->mesh()->neighbor_cell(face));
        local_structure[1] += std::max(left, right) > 1.5 * std::min(left, right);
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 2, local_structure.data(), structure.data());
    testing::Test::RecordProperty("global_native_cells", structure[0]);
    testing::Test::RecordProperty("global_coarse_fine_face_incidences", structure[1]);
    EXPECT_GT(structure[1], 0);
    std::vector<ReferenceCell> cells;
    for (const auto& key : source->owned_order())
        cells.push_back(reference_cell(source->hierarchy()->geometry(source->hierarchy()->owned().at(key))));
    // In the central plateau X'=U(1-2Y), Y'=0. The interface remains there
    // throughout T=1/8, so X+2UtY=.5+Ut is exact for prescribed transport.
    // This fixture does not claim an exact coupled pressure/flow solution.
    unsigned accepted_steps = 0;
    for (unsigned step = 1; step <= 8; ++step)
    {
        SCOPED_TRACE(step);
        if (comm->getRank() == 0)
            std::cout << "attempted_step=" << step << " accepted_time=" << double((step - 1) * dt)
                      << " required_endpoint=0.125" << std::endl;
        const auto identity = source->accepted_identity();
        const auto gas = source->gas_volumes(), carrier = source->carrier_volumes();
        const auto trial = backend == GeneralVoFTransportBackend::GeometricFace
                               ? source->plan_face_transport(controls(*source), double(dt), {1, 1000}, step)
                               : source->plan_transport(velocities(*source), controls(*source), double(dt));
        const auto prepared = backend == GeneralVoFTransportBackend::GeometricFace ? trial.prepared_transport()
                                                                                   : trial.prepare_transport();
        EXPECT_EQ(prepared->family, backend == GeneralVoFTransportBackend::GeometricFace
                                        ? GeneralVoFTransportFamily::AlgebraicFaceVolume
                                        : GeneralVoFTransportFamily::InterfaceLocalSwept);
        EXPECT_EQ(prepared->backend, backend);
        const auto next = source->apply(trial);
        std::array<double, 9> local{}, sum{};
        std::array<double, 3> local_moment{}, moment{};
        double local_width = std::numeric_limits<double>::infinity(), width = 0;
        std::array<double, 2> maximum{}, global_max{};
        for (std::size_t c = 0; c < cells.size(); ++c)
        {
            const auto g = next->phase_volumes()[c].gas, l = next->phase_volumes()[c].carrier;
            const auto v = cells[c].volume(), rg = cells[c].phase(step * dt, true);
            const auto rl = cells[c].phase(step * dt, false);
            EXPECT_GE(g, 0);
            EXPECT_GE(l, 0);
            local[0] += double(std::abs(g - rg));
            local[1] += double(std::abs(l - rl));
            local[2] += double(v);
            local[3] += g;
            local[4] += l;
            const auto center = source->mesh()->cell_centroid(int(c));
            local_moment[0] += g * center.x;
            local_moment[1] += g * center.y;
            local_moment[2] += g * center.z;
            if ((g > 0 && l > 0) || (rg > 0 && rg < v))
            {
                long double xlo = 1, xhi = 0, ylo = 1, yhi = 0;
                for (const auto& point : cells[c].xy)
                {
                    xlo = std::min(xlo, point[0]);
                    xhi = std::max(xhi, point[0]);
                    ylo = std::min(ylo, point[1]);
                    yhi = std::max(yhi, point[1]);
                }
                local_width = std::min(local_width, double(std::min({xhi - xlo, yhi - ylo, cells[c].height})));
            }
            local[5] += double(rg);
            local[6] += double(rl);
            if (cells[c].xy.front()[0] >= .5L)
                local[7] += g;
            maximum[0] = std::max(maximum[0], double(std::abs(g - rg) / v));
            maximum[1] = std::max(maximum[1], double(std::abs(l - rl) / v));
        }
        const bool algebraic = backend == GeneralVoFTransportBackend::GeometricFace;
        local[8] = double(prepared->diagnostics.geometric_faces);
        std::array<double, 10> histogram{}, global_histogram{};
        if (algebraic)
        {
            local[8] = 0;
            std::map<std::uint64_t, int> owners;
            for (const auto& tile : source->tiles())
                owners.emplace(tile.face_gid, tile.first_owner);
            for (const auto& stream : prepared->streams)
                if (owners.at(stream.face) == comm->getRank())
                {
                    ++local[8];
                    EXPECT_FALSE(stream.lower_order);
                    const auto bin = stream.theta <= 0   ? 0u
                                     : stream.theta >= 1 ? 9u
                                                         : std::min(8u, 1u + unsigned(8 * stream.theta));
                    ++histogram[bin];
                }
        }
        else
            for (unsigned i = 0; i < histogram.size(); ++i)
                histogram[i] = double(prepared->diagnostics.geometric_theta_histogram[i]);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 3, local_moment.data(), moment.data());
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_width, &width);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(local.size()), local.data(), sum.data());
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 2, maximum.data(), global_max.data());
        const std::array<double, 2> minimum{
            double(prepared->diagnostics.minimum_theta), double(prepared->diagnostics.minimum_quality_theta)};
        std::array<double, 2> global_min{};
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 2, minimum.data(), global_min.data());
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 10, histogram.data(), global_histogram.data());
        const auto& work = prepared->diagnostics;
        const std::array<double, 6> local_work{double(work.fallback_streams), double(work.fallback_volume),
            double(work.limited_streams), double(work.limited_volume), double(work.plane_measurements),
            double(work.swept_pieces)};
        std::array<double, 6> global_work{};
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 6, local_work.data(), global_work.data());
        const auto courant = double(work.maximum_outgoing_courant);
        double maximum_courant = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &courant, &maximum_courant);
        for (const auto value : sum)
            EXPECT_TRUE(std::isfinite(value));
        for (const auto value : global_max)
            EXPECT_TRUE(std::isfinite(value));
        for (const auto value : global_min)
            EXPECT_TRUE(std::isfinite(value));
        double count = 0;
        for (const auto n : global_histogram)
            count += n;
        EXPECT_EQ(count, sum[8]);
        EXPECT_GT(sum[8], 0);
        EXPECT_NEAR(sum[5], .5, 2e-15);
        EXPECT_NEAR(sum[6], .5, 2e-15);
        EXPECT_NEAR(sum[2], 1, 2e-15);
        EXPECT_NEAR(sum[3], .5, 2e-11 * .5);
        EXPECT_NEAR(sum[4], .5, 2e-11 * .5);
        EXPECT_EQ(source->accepted_identity(), identity);
        EXPECT_EQ(source->gas_volumes(), gas);
        EXPECT_EQ(source->carrier_volumes(), carrier);
        std::ostringstream report;
        report << std::setprecision(17) << "step=" << step << " time=" << double(step * dt)
               << " gas_L1=" << sum[0] / sum[2] << " carrier_L1=" << sum[1] / sum[2]
               << " gas_max_fraction_error=" << global_max[0] << " carrier_max_fraction_error=" << global_max[1]
               << " gas_relative_mass_drift=" << (sum[3] - .5) / .5
               << " carrier_relative_mass_drift=" << (sum[4] - .5) / .5 << " gas_passage=" << sum[7]
               << " minimum_theta=" << global_min[0] << " minimum_quality_theta=" << global_min[1]
               << " maximum_outgoing_courant=" << maximum_courant
               << " fallback_streams=" << global_work[0] << " fallback_volume=" << global_work[1]
               << " limited_streams=" << global_work[2] << " limited_volume=" << global_work[3]
               << " plane_measurements=" << global_work[4] << " swept_pieces=" << global_work[5]
               << " geometric_theta_histogram=";
        for (unsigned i = 0; i < global_histogram.size(); ++i)
            report << (i ? "," : "[") << global_histogram[i];
        report << "]";
        testing::Test::RecordProperty("motion_step_" + std::to_string(step), report.str());
        if (comm->getRank() == 0)
            std::cout << report.str() << std::endl;
        const long double displacement = speed * step * dt;
        // Integrating X <= .5+s-2*s*Y over the unit square gives these
        // exact physical centroids (unit-determinant x=X+shear*Y map).
        const long double reference_y = .5L - displacement / 3;
        const long double reference_x = .25L + displacement * displacement / 3 + shear * reference_y;
        const double centroid_error = std::hypot(moment[0] / sum[3] - double(reference_x),
            moment[1] / sum[3] - double(reference_y), moment[2] / sum[3] - .5);
        testing::Test::RecordProperty("centroid_error_" + std::to_string(step), centroid_error);
        testing::Test::RecordProperty("interface_cell_width_" + std::to_string(step), width);
        EXPECT_EQ(next->phase_cleanup_state().cleaned_cells, 0u);
        if (step == 8)
        {
            EXPECT_LE(sum[0] / .5, .02);
            EXPECT_LE(sum[1] / .5, .02);
            EXPECT_LE(centroid_error, .25 * width);
        }
        source = next;
        ++accepted_steps;
        testing::Test::RecordProperty("accepted_steps", accepted_steps);
        testing::Test::RecordProperty("accepted_time", double(step * dt));
        int local_failure = testing::Test::HasFailure(), global_failure = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_failure, &global_failure);
        ASSERT_EQ(global_failure, 0) << "Single-interface physical gate failed at step " << step;
    }
    EXPECT_EQ(accepted_steps, 8u);
    testing::Test::RecordProperty("accepted_steps", accepted_steps);
}
TEST(GeneralVoFMotionAccuracy, SingleInterfaceClosedShearSweptPartition)
{
    single_interface_motion(GeneralVoFTransportBackend::SweptPartition);
}
TEST(GeneralVoFMotionAccuracy, SingleInterfaceClosedShearGeometricFace)
{
    single_interface_motion(GeneralVoFTransportBackend::GeometricFace);
}
} // namespace
