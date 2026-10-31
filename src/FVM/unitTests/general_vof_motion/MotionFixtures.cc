#include "MotionFixtures.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "geometry/unitTests/point_export_test_helpers.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <utility>

namespace SimpleFluid::VoF::test::motion
{
namespace AMR = SimpleFluid::Meshes::AMR;
ReferenceCell reference_cell(const PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>& shape)
{
    Polygon points;
    long double bottom = std::numeric_limits<long double>::max(), top = -bottom;
    for (const auto& p : PLIC::test::exported_points(*shape))
    {
        points.push_back({p.value[0] - shear * p.value[1], p.value[1]});
        bottom = std::min(bottom, p.value[2]);
        top = std::max(top, p.value[2]);
    }
    return {hull(std::move(points)), top - bottom};
}
std::shared_ptr<const GeneralVoF> motion_fixture(GeneralVoFTransportBackend backend)
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
} // namespace SimpleFluid::VoF::test::motion
