#pragma once
#include "FVM/GeneralVoF.hh"
#include "solvers/details/GeneralVoFFlowWorkspace.hh"
#include "solvers/details/GeneralVoFPhiClosure.hh"
namespace SimpleFluid::detail::general_vof
{
template<ArithmeticMode Mode>
SIMPLEFLUID_SOLVERS_LOCAL std::vector<PLIC::DynamicGeometryValue> face_rates(const MeshHandle<>& mesh,
    const VoF::GeneralVoF& phases, const GeneralVoFFiniteProjectionProvenance& projection,
    const GeneralVoFFlowWorkspace& flow);
template<ArithmeticMode Mode>
SIMPLEFLUID_SOLVERS_LOCAL std::vector<VoF::GeneralVoFFaceVolume> face_volumes(const MeshHandle<>& mesh,
    const VoF::GeneralVoF& phases, const GeneralVoFFiniteProjectionProvenance& projection,
    const GeneralVoFFlowWorkspace& flow, double dt);
template<class Topology>
SIMPLEFLUID_SOLVERS_LOCAL Topology projection_topology(const MeshHandle<>& mesh, const VoF::GeneralVoF& phases);
} // namespace SimpleFluid::detail::general_vof
