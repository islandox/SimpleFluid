/** Small native fixture builders shared by the general and finite VoF tests. */
#pragma once
#include "FVM/GeneralVoF.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace SimpleFluid::VoF::test::small_fixtures
{
namespace AMR = SimpleFluid::Meshes::AMR;
struct Fixture
{
    std::shared_ptr<const AMR::DistributedAdaptiveHierarchy> hierarchy;
    std::vector<AMR::AdaptiveCellKey> order;
    std::vector<double> gas, carrier;
};
inline std::shared_ptr<const AMR::DistributedAdaptiveHierarchy> eight_cell_hierarchy(AMR::AdaptiveBytes domain)
{
    const auto comm = Tpetra::getDefaultComm();
    AMR::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.extent = {2, 2, 2};
    AMR::CartesianAMR mesh(grid);
    std::vector<std::uint64_t> owned;
    for (std::uint64_t id = 0; id < 8; ++id)
        if (int(id % std::min(2, comm->getSize())) == comm->getRank())
            owned.push_back(id);
    auto source = mesh.detach_owned(owned, std::move(domain));
    return std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(source.owned), std::move(source.families), source.kernel, comm);
}
inline std::vector<GeneralVoFNodeVelocity> stationary(const GeneralVoF& state)
{
    std::vector<GeneralVoFNodeVelocity> result;
    for (const auto& node : state.nodes())
        result.push_back({node.gid, {0, 0, 0}});
    return result;
}
inline std::vector<GeneralVoFFaceVolume> circulation(const GeneralVoF& state, bool identity = false)
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
} // namespace SimpleFluid::VoF::test::small_fixtures
