/** @file PhasePublication.cc
 * @brief Preserve finite publication storage and phase-conservation arithmetic policies.
 */
#include "FVM/details/GeneralVoF/PhasePublication.tcc"
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
template PhaseConservationResult check_phase_conservation<PhaseConservationPolicy::Ordinary>(
    const PhaseConservationSummary&, int, double) noexcept;
template PhaseConservationResult check_phase_conservation<PhaseConservationPolicy::Bounded>(
    const PhaseConservationSummary&, int, double) noexcept;
void reset_finite_phase_publication(GeneralVoF::Data& data) noexcept
{
    data.gas.clear();
    data.carrier.clear();
    data.phases.clear();
    data.roundoff.clear();
    data.phase_history.clear();
    data.occupancy.clear();
}
void append_finite_phase_publication(GeneralVoF::Data& data, const Cell& cell)
{
    data.gas.push_back(cell.phase.gas_volume);
    data.carrier.push_back(cell.phase.carrier_volume);
    data.phases.push_back({cell.native_volume, cell.phase.gas_volume, cell.phase.carrier_volume});
    data.roundoff.push_back(cell.roundoff);
    data.phase_history.push_back(cell.finite_history);
    data.occupancy.push_back(cell.occupancy);
}
} // namespace SimpleFluid::VoF::general_vof_detail
