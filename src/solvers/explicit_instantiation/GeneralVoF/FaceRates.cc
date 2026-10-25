#include "solvers/details/GeneralVoF/FaceRates.tcc"
namespace SimpleFluid::detail::general_vof
{
template std::vector<PLIC::DynamicGeometryValue> face_rates<ArithmeticMode::Ordinary>(const MeshHandle<>& mesh,
    const VoF::GeneralVoF& phases, const GeneralVoFFiniteProjectionProvenance& projection,
    const GeneralVoFFlowWorkspace& flow);
template std::vector<VoF::GeneralVoFFaceVolume> face_volumes<ArithmeticMode::Ordinary>(const MeshHandle<>& mesh,
    const VoF::GeneralVoF& phases, const GeneralVoFFiniteProjectionProvenance& projection,
    const GeneralVoFFlowWorkspace& flow, double dt);
template std::vector<PLIC::DynamicGeometryValue> face_rates<ArithmeticMode::BoundedCertification>(
    const MeshHandle<>& mesh, const VoF::GeneralVoF& phases, const GeneralVoFFiniteProjectionProvenance& projection,
    const GeneralVoFFlowWorkspace& flow);
template std::vector<VoF::GeneralVoFFaceVolume> face_volumes<ArithmeticMode::BoundedCertification>(
    const MeshHandle<>& mesh, const VoF::GeneralVoF& phases, const GeneralVoFFiniteProjectionProvenance& projection,
    const GeneralVoFFlowWorkspace& flow, double dt);
template GeneralVoFFiniteProjectionClosureTopology projection_topology<GeneralVoFFiniteProjectionClosureTopology>(
    const MeshHandle<>&, const VoF::GeneralVoF&);
template Exact::GeneralVoFProjectionClosureTopology projection_topology<Exact::GeneralVoFProjectionClosureTopology>(
    const MeshHandle<>&, const VoF::GeneralVoF&);
} // namespace SimpleFluid::detail::general_vof
