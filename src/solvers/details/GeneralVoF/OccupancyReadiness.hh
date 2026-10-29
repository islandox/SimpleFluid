#pragma once
#include "FVM/GeneralVoF.hh"
#include "geometry/swept_volume/details/FiniteGeometryAccess.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoFFlowWorkspace.hh"
#include "utils/CompensatedSum.hh"

namespace SimpleFluid::detail::general_vof
{
inline GeneralVoFFiniteProjectionProvenance::TransportReadiness occupancy_readiness(
    const VoF::GeneralVoF& phases, double courant, std::size_t byte_budget)
{
    GeneralVoFFiniteProjectionProvenance::TransportReadiness result{
        phases.options().reconstruction.volume_tolerance, courant};
    const auto& amounts = phases.phase_volumes();
    const Execution execution(*phases.mesh());
    execution.collective(
        [&]
        {
            constexpr auto bytes_per_cell =
                2 * sizeof(long double) + sizeof(PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>);
            if (amounts.size() > byte_budget / bytes_per_cell)
                throw std::runtime_error("Cumulative occupancy readiness exceeds its byte budget.");
            const auto geometry = phases.owned_finite_geometry<ArithmeticMode::Ordinary>();
            if (geometry.size() != amounts.size())
                throw std::runtime_error("Cumulative occupancy readiness exceeds its byte budget or geometry shape.");
            result.cumulative = true;
            result.signed_discrepancy.resize(amounts.size());
            result.volume.resize(amounts.size());
            for (std::size_t c = 0; c < amounts.size(); ++c)
            {
                result.volume[c] =
                    SimpleFluid::geometry::swept_volume::runtime_value<ArithmeticMode::Ordinary>(geometry[c]->volume())
                        .value;
                CompensatedSum<long double> sum;
                sum += amounts[c].gas;
                sum += amounts[c].carrier;
                sum += -result.volume[c];
                result.signed_discrepancy[c] = sum.value();
            }
        });
    return result;
}
} // namespace SimpleFluid::detail::general_vof
