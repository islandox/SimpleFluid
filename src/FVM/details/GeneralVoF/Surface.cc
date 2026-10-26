/** @file Surface.cc
 * @brief Canonical explicit interfaces between opposite geometrically pure cells.
 */
#include "FVM/details/GeneralVoFInternal.hh"
#include "geometry/PLIC/PlanarGeometry.hh"

namespace SimpleFluid::VoF
{
namespace
{
int geometric_phase(const Cell& cell)
{
    if (cell.declaration == GeneralVoFPhaseGeometry::PureGas)
        return 1;
    if (cell.declaration == GeneralVoFPhaseGeometry::PureCarrier)
        return -1;
    if (cell.retained_interface)
    {
        if (cell.retained_interface->pure_gas)
            return 1;
        if (cell.retained_interface->pure_carrier)
            return -1;
    }
    if (cell.phase.carrier_volume == 0 && cell.phase.gas_volume > 0)
        return 1;
    if (cell.phase.gas_volume == 0 && cell.phase.carrier_volume > 0)
        return -1;
    return 0;
}
} // namespace
std::vector<GeneralVoFFaceInterface> GeneralVoF::face_interfaces() const
{
    std::vector<GeneralVoFFaceInterface> result;
    wire::collective(*data_->hierarchy->communicator(),
        [&]
        {
            current(*data_);
            if (arithmetic_mode() != ArithmeticMode::Ordinary ||
                data_->options.transport_backend == GeneralVoFTransportBackend::SweptPartition)
                throw std::invalid_argument(
                    "Explicit pure-cell face surfaces require an Ordinary face transport backend.");
            wire::Budget budget{data_->options.byte_budget};
            for (const auto& tile : data_->topology->surface_tiles)
            {
                if (tile.exterior || !tile.second)
                    continue;
                const auto& first = data_->cells.at(tile.first);
                const auto& second = data_->cells.at(*tile.second);
                const auto a = geometric_phase(first), b = geometric_phase(second);
                if (a * b != -1)
                    continue;
                if (!tile.second_image_identity.empty())
                    throw std::invalid_argument("Periodic face-interface surfaces are not qualified.");
                budget.add(sizeof(GeneralVoFFaceInterface));
                const auto centroid = tile.geometry->centroid();
                const auto area_vector = tile.geometry->area_vector();
                const auto area = tile.geometry->area().as_double();
                const auto magnitude = std::hypot(area_vector.x, area_vector.y, area_vector.z);
                if (!(area > 0) || !(magnitude > 0) || !std::isfinite(magnitude))
                    throw std::invalid_argument("A canonical face-interface surface has invalid physical area.");
                PLIC::GeometryPoint n{area_vector.x / magnitude, area_vector.y / magnitude, area_vector.z / magnitude};
                long double orientation = 0;
                for (unsigned axis = 0; axis < 3; ++axis)
                    orientation += n[axis] * (static_cast<long double>(second.centroid[axis]) - first.centroid[axis]);
                if (orientation == 0 || !std::isfinite(orientation))
                    throw std::invalid_argument("A canonical face-interface surface has unresolved orientation.");
                // The gas-to-liquid direction is independent of tile loop winding.
                if ((orientation > 0 ? 1 : -1) != a)
                    for (auto& component : n)
                        component = -component;
                result.push_back({tile.face_gid, first.gid, second.gid, tile.first_owner, tile.second_owner,
                    {centroid.x, centroid.y, centroid.z}, n, area});
            }
        });
    return result;
}
} // namespace SimpleFluid::VoF
