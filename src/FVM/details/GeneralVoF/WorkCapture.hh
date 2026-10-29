/** @file WorkCapture.hh
 * @brief Allocation-free measured scopes and optional capture storage accounting.
 */
#pragma once
#include "FVM/GeneralVoF.hh"
#include "FVM/GeneralVoFWork.hh"
#include "geometry/PLIC/ConvexGeometry.hh"
#include <chrono>
namespace SimpleFluid::VoF::general_vof_detail
{
class WorkObservation
{
public:
    SIMPLEFLUID_FVM_LOCAL explicit WorkObservation(GeneralVoFWorkMetrics*) noexcept;
    SIMPLEFLUID_FVM_LOCAL ~WorkObservation() noexcept;
    SIMPLEFLUID_FVM_LOCAL void finish() noexcept;
    WorkObservation(const WorkObservation&) = delete;
    WorkObservation& operator=(const WorkObservation&) = delete;

private:
    GeneralVoFWorkMetrics* metrics_;
    std::chrono::steady_clock::time_point started_{};
    PLIC::GeometryStatistics before_{};
};
[[nodiscard]] SIMPLEFLUID_FVM_LOCAL GeneralVoFWorkCell* balance_cell(
    GeneralVoFWorkSample*, const Meshes::AMR::AdaptiveCellKey&) noexcept;
[[nodiscard]] SIMPLEFLUID_FVM_LOCAL std::size_t balance_work_storage_bytes(const GeneralVoFWorkSample&);
} // namespace SimpleFluid::VoF::general_vof_detail
