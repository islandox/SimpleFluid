#pragma once
#include "FVM/GeneralVoF.hh"
#include "solvers/details/GeneralVoFFlowWorkspace.hh"
#include <map>
namespace SimpleFluid::detail::general_vof::exact
{
using FaceMeasures = std::map<DefaultTpetraTypes::global_ordinal_type,
    std::pair<PLIC::GeometryMeasure<ArithmeticMode::Exact>, PLIC::GeometryMeasure<ArithmeticMode::Exact>>>;
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL FaceMeasures exchange_face_measures(
    const MeshHandle<>& mesh, const VoF::GeneralVoF& phases, const FaceMeasures& owned, std::size_t resource_budget);
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL std::vector<VoF::GeneralVoFFaceVolume> face_volumes(const MeshHandle<>& mesh,
    const VoF::GeneralVoF& phases, const GeneralVoFProjectionProvenance& projection, double dt,
    std::size_t byte_budget);
} // namespace SimpleFluid::detail::general_vof::exact
