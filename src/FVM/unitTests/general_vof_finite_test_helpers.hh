/** Shared test-local native fixtures; no test cases or environment registration. */
#pragma once
#include "FVM/GeneralVoF.hh"
#include "geometry/PLIC/details/PhaseArithmetic.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <map>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::VoF;
using namespace SimpleFluid::Meshes;
using namespace SimpleFluid::Meshes::AMR;
namespace fp = SimpleFluid::VoF::finite_detail;
struct Fixture
{
    std::shared_ptr<const DistributedAdaptiveHierarchy> hierarchy;
    std::vector<AdaptiveCellKey> order;
    std::vector<double> gas, carrier;
};
Fixture fixture(bool dilute = false)
{
    const auto comm = Tpetra::getDefaultComm();
    AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.extent = {2, 2, 2};
    CartesianAMR mesh(grid);
    std::vector<std::uint64_t> owned;
    for (std::uint64_t id = 0; id < 8; ++id)
        if (int(id % std::min(2, comm->getSize())) == comm->getRank()) owned.push_back(id);
    auto source = mesh.detach_owned(owned, {'f', 'i', 'n', 'i', 't', 'e'});
    Fixture result;
    result.hierarchy = std::make_shared<const DistributedAdaptiveHierarchy>(std::move(source.owned),
        std::move(source.families), source.kernel, comm);
    for (const auto& [key, leaf] : result.hierarchy->owned())
    {
        result.order.push_back(key);
        const auto center = result.hierarchy->geometry(leaf)->centroid();
        const auto gas = dilute ? 1e-6 : (center[0] < 1 && center[1] < 1 && center[2] < 1 ? 1. : 0.);
        result.gas.push_back(gas);
        result.carrier.push_back(1 - gas);
    }
    return result;
}
Fixture coarse_fine_fixture()
{
    const auto comm = Tpetra::getDefaultComm();
    AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 2, 2};
    CartesianAMR mesh(grid);
    std::vector<std::uint64_t> owned;
    for (std::uint64_t id = 0; id < 16; ++id)
        if (int(id % std::min(2, comm->getSize())) == comm->getRank()) owned.push_back(id);
    auto source = mesh.detach_owned(owned, {'q', 'u', 'o', 't', 'a', '_', 's', 'e', 'a', 'm'});
    const auto initial = std::make_shared<const DistributedAdaptiveHierarchy>(std::move(source.owned),
        std::move(source.families), source.kernel, comm);
    std::vector<AdaptiveMark> marks;
    if (initial->owned().contains(mesh.cell_key(0)))
        marks.push_back({mesh.cell_key(0), AdaptiveSplit::Isotropic});
    Fixture result;
    result.hierarchy = initial->apply(initial->plan_adaptation(marks));
    for (const auto& [key, leaf] : result.hierarchy->owned())
    {
        result.order.push_back(key);
        result.gas.push_back(0);
        result.carrier.push_back(result.hierarchy->metrics(leaf).volume);
    }
    return result;
}
std::vector<GeneralVoFNodeVelocity> stationary(const GeneralVoF& state)
{
    std::vector<GeneralVoFNodeVelocity> result;
    for (const auto& node : state.nodes()) result.push_back({node.gid, {0, 0, 0}});
    return result;
}
std::vector<GeneralVoFFaceVolume> circulation(const GeneralVoF& state, bool identity = false)
{
    std::vector<GeneralVoFFaceVolume> result;
    for (const auto& face : state.tiles())
    {
        const auto p = face.geometry->centroid(), n = face.geometry->area_vector();
        double phi = 0;
        if (!identity && !face.exterior && p.x == 1 && n.x != 0)
            phi = (p.y < 1 ? 1 : -1) * (n.x > 0 ? 1 : -1) / 64.;
        if (!identity && !face.exterior && p.y == 1 && n.y != 0)
            phi = (p.x > 1 ? 1 : -1) * (n.y > 0 ? 1 : -1) / 64.;
        result.push_back({face.face_gid, phi});
    }
    return result;
}
} // namespace
