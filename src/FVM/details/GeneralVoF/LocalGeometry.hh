/** @file LocalGeometry.hh
 * @brief Canonical-owner Ordinary swept geometry restricted to the interface reach.
 */
#pragma once
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
using OrdinaryTraceMap =
    std::map<std::uint64_t, std::pair<PLIC::Point<ArithmeticMode::Ordinary>, PLIC::Point<ArithmeticMode::Ordinary>>>;
using OrdinaryFluxMap = std::map<std::uint64_t, PLIC::DynamicGeometryValue>;
struct LocalGeometricGroup
{
    GeneralVoFFiniteFaceContribution contribution;
    long double bulk_share = 0, allocation = 0, cut_closure = 0, gas_history = 0, carrier_history = 0;
    bool gas_geometry = false, carrier_geometry = false;
};
struct LocalFaceGeometry
{
    std::uint64_t face = 0;
    bool geometric = false;
    long double quality_theta = 1;
    GeneralVoFFaceChainDiagnostic chain;
    std::vector<LocalGeometricGroup> groups;
};
struct LocalGeometryResult
{
    std::vector<LocalFaceGeometry> owned_faces;
    GeneralVoFTransportDiagnostics diagnostics;
    std::shared_ptr<GeneralVoFWorkSample> balance_work;
};
SIMPLEFLUID_FVM_LOCAL LocalGeometryResult prepare_local_sweeps(const GeneralVoF::Data&, const OrdinaryTraceMap&,
    const OrdinaryFluxMap&, const std::map<std::uint64_t, long double>&, const PLIC::GeometryPacketContext&);
} // namespace SimpleFluid::VoF::general_vof_detail
