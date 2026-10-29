/** @file WorkCapture.cc
 * @brief Rank-local work scopes and retained capture byte accounting.
 */
#include "FVM/details/GeneralVoF/WorkCapture.hh"
#include "FVM/details/GeneralVoF/OwnerPlaneRequests.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <limits>
namespace SimpleFluid::VoF::general_vof_detail
{
WorkObservation::WorkObservation(GeneralVoFWorkMetrics* metrics) noexcept : metrics_(metrics)
{
    if (metrics_)
    {
        before_ = PLIC::geometry_statistics();
        started_ = std::chrono::steady_clock::now();
    }
}
WorkObservation::~WorkObservation() noexcept
{
    finish();
}
void WorkObservation::finish() noexcept
{
    if (!metrics_)
        return;
    metrics_->seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count();
    const auto after = PLIC::geometry_statistics();
    metrics_->floating_operations += after.floating_operations - before_.floating_operations;
    metrics_->volume_evaluations += after.volume_evaluations - before_.volume_evaluations;
    metrics_->intersections += after.intersections - before_.intersections;
    metrics_->constructions += after.constructions - before_.constructions;
    metrics_ = nullptr;
}
GeneralVoFWorkCell* balance_cell(GeneralVoFWorkSample* sample, const Meshes::AMR::AdaptiveCellKey& key) noexcept
{
    if (sample)
        for (auto& cell : sample->cells)
            if (cell.key == key)
                return &cell;
    return nullptr;
}
std::size_t balance_work_storage_bytes(const GeneralVoFWorkSample& sample)
{
    Meshes::AMR::adaptive_detail::Budget bytes{std::numeric_limits<std::size_t>::max()};
    bytes.add(sizeof(sample) + 2 * sizeof(void*)); // shared allocation control bookkeeping
    bytes.add(sample.cells.capacity() * sizeof(GeneralVoFWorkCell));
    bytes.add(sample.faces.capacity() * sizeof(GeneralVoFWorkFace));
    bytes.add(sample.receivers.capacity() * sizeof(GeneralVoFWorkReceiver));
    for (const auto& cell : sample.cells)
        bytes.add(owner_plane_key_storage_bytes(cell.key));
    for (const auto& face : sample.faces)
    {
        bytes.add(face.identity.capacity());
        bytes.add(owner_plane_key_storage_bytes(face.first.key));
        bytes.add(owner_plane_key_storage_bytes(face.second.key));
        bytes.add(face.donors.capacity() * sizeof(GeneralVoFWorkDonor));
        for (const auto& donor : face.donors)
            bytes.add(owner_plane_key_storage_bytes(donor.key));
    }
    for (const auto& receiver : sample.receivers)
        bytes.add(owner_plane_key_storage_bytes(receiver.donor.key));
    return bytes.used;
}
} // namespace SimpleFluid::VoF::general_vof_detail
