/** Shared test-local native fixtures; no test cases or environment registration. */
#pragma once
#include "FVM/GeneralVoF.hh"
#include "FVM/details/BoundedPhaseArithmetic.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "utils/CompensatedSum.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using namespace SimpleFluid::Meshes::AMR;
using namespace SimpleFluid::VoF;
struct Fixture
{
    std::shared_ptr<const DistributedAdaptiveHierarchy> hierarchy;
    std::vector<AdaptiveCellKey> order;
    std::vector<double> gas, carrier;
};
Fixture setup(bool trace = false)
{
    auto comm = Tpetra::getDefaultComm();
    AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.extent = {2, 2, 2};
    auto source = std::make_shared<CartesianAMR>(grid);
    std::vector<std::uint64_t> owned;
    for (std::uint64_t gid = 0; gid < 8; ++gid)
        if (int(gid % std::min(2, comm->getSize())) == comm->getRank())
            owned.push_back(gid);
    auto boot = source->detach_owned(owned, {'g', 'e', 'n', 'e', 'r', 'a', 'l', '_', 'v', 'o', 'f'});
    Fixture f;
    f.hierarchy = std::make_shared<const DistributedAdaptiveHierarchy>(
        std::move(boot.owned), std::move(boot.families), boot.kernel, comm);
    for (const auto& [key, leaf] : f.hierarchy->owned())
    {
        f.order.push_back(key);
        auto c = f.hierarchy->geometry(leaf)->centroid();
        const double gas = trace ? 1e-310 : ((c[0] < 1 && c[1] < 1 && c[2] < 1) ? 1 : 0);
        f.gas.push_back(gas);
        f.carrier.push_back(trace ? 1 : 1 - gas);
    }
    return f;
}
std::vector<GeneralVoFNodeVelocity> nodes(const GeneralVoF& state)
{
    std::vector<GeneralVoFNodeVelocity> result;
    for (const auto& node : state.nodes())
        result.push_back({node.gid, {0, 0, 0}});
    return result;
}
std::vector<GeneralVoFFaceVolume> cycle(const GeneralVoF& state)
{
    std::vector<GeneralVoFFaceVolume> result;
    for (const auto& face : state.tiles())
    {
        const auto p = face.geometry->centroid(), area = face.geometry->area_vector();
        double phi = 0;
        if (!face.exterior && p.x == 1 && area.x != 0)
            phi = (p.y < 1 ? 1 : -1) * (area.x > 0 ? 1 : -1) / 64.;
        if (!face.exterior && p.y == 1 && area.y != 0)
            phi = (p.x > 1 ? 1 : -1) * (area.y > 0 ? 1 : -1) / 64.;
        result.push_back({face.face_gid, phi});
    }
    return result;
}
Fixture skew_setup(unsigned ownership_shift = 0)
{
    auto comm = Tpetra::getDefaultComm();
    std::vector<AdaptiveLeaf> owned;
    for (unsigned k = 0; k < 2; ++k)
        for (unsigned j = 0; j < 2; ++j)
            for (unsigned i = 0; i < 2; ++i)
            {
                const auto id = i + 2 * j + 4 * k;
                if (int((id + ownership_shift) % std::min(2, comm->getSize())) != comm->getRank())
                    continue;
                AdaptiveLeaf leaf;
                leaf.key = {7, ConvexCellAncestry{id, {}}};
                leaf.domain = {'s', 'k', 'e', 'w'};
                std::vector<GeometryPoint> points;
                for (unsigned z = 0; z < 2; ++z)
                    for (unsigned y = 0; y < 2; ++y)
                        for (unsigned x = 0; x < 2; ++x)
                        {
                            const double u = i + x, v = j + y, w = k + z;
                            points.push_back({u + .25 * v - w, v, u + .25 * v + w});
                        }
                leaf.geometry = ConvexCellGeometry::make(
                    points, {{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}});
                for (unsigned f = 0; f < 6; ++f)
                {
                    const unsigned axis = f / 2, coordinate = std::array<unsigned, 3>{i, j, k}[axis] + f % 2;
                    AdaptiveSurface side;
                    side.key = {'s', char(axis), char(coordinate)};
                    side.geometry = leaf.geometry->face_geometry(f);
                    side.orientation = f % 2 ? 1 : -1;
                    side.exterior = coordinate == 0 || coordinate == 2;
                    side.boundary = side.exterior ? "wall" : "";
                    leaf.surfaces.push_back(std::move(side));
                }
                owned.push_back(std::move(leaf));
            }
    Fixture fixture;
    fixture.hierarchy = std::make_shared<const DistributedAdaptiveHierarchy>(
        std::move(owned), std::vector<AdaptiveFamily>{}, convex_adaptive_kernel(2, 0), comm);
    for (const auto& [key, leaf] : fixture.hierarchy->owned())
    {
        fixture.order.push_back(key);
        const auto root = std::get<ConvexCellAncestry>(key.ancestry).root;
        fixture.gas.push_back(root < 4 ? 1 : 0);
        fixture.carrier.push_back(root < 4 ? 1 : 2);
    }
    return fixture;
}

void expect_finite_phase_balance(const GeneralVoF& source, const GeneralVoF& target, const GeneralVoF::Trial& trial)
{
    std::array<::SimpleFluid::detail::CompensatedSum<>, 6> sums;
    for (std::size_t i = 0; i < source.phase_volumes().size(); ++i)
    {
        const auto& old = source.phase_volumes()[i];
        const auto& next = target.phase_volumes()[i];
        const auto& bound = trial.phase_roundoff_bounds()[i];
        sums[0] += static_cast<long double>(next.gas) - old.gas;
        sums[1] += static_cast<long double>(next.carrier) - old.carrier;
        sums[2] += bound.gas;
        sums[3] += bound.carrier;
        sums[4] += static_cast<long double>(std::abs(next.gas)) + std::abs(old.gas);
        sums[5] += static_cast<long double>(std::abs(next.carrier)) + std::abs(old.carrier);
    }
    std::array<double, 6> local{}, global{};
    for (std::size_t i = 0; i < local.size(); ++i)
        local[i] = static_cast<double>(sums[i].value());
    const auto comm = source.hierarchy()->communicator();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(local.size()), local.data(), global.data());
    // Conservative finite arithmetic envelope for the six local/wire sums;
    // each phase uses its own L1 inventory and actual publication errors.
    const double operations = 64 * double(source.hierarchy()->global_cells()) + 32 * comm->getSize() + 128;
    const double u = operations * std::numeric_limits<double>::epsilon();
    ASSERT_LT(u, .25);
    for (unsigned phase = 0; phase < 2; ++phase)
    {
        const double tolerance = global[phase + 2] + u / (1 - u) * (global[phase + 4] + global[phase + 2]);
        EXPECT_LE(std::abs(global[phase]), tolerance);
    }
}
} // namespace
