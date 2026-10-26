/** @file LocalTransport.hh
 * @brief Authenticated controls for the Ordinary signed interface ledger.
 */
#pragma once
#include "FVM/details/GeneralVoF/LocalGeometry.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
struct LocalTransportControls
{
    std::span<const GeneralVoFFaceVolume> face_volume;
    const OrdinaryTraceMap& traces;
    const OrdinaryFluxMap& flux;
    const std::map<std::uint64_t, long double>& face_budgets;
    const PLIC::GeometryPacketContext& packet_context;
};
} // namespace SimpleFluid::VoF::general_vof_detail
