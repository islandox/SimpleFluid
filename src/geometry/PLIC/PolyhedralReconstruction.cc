#include "geometry/PLIC/details/FinitePolyhedralReconstruction.tcc"
namespace SimpleFluid::VoF
{
template void PolyhedralReconstruction::validate_finite<ArithmeticMode::Ordinary>(
    const PolyhedralPhaseSampleFor<ArithmeticMode::Ordinary>&, PolyhedralReconstructionOptions);
template void PolyhedralReconstruction::validate_finite<ArithmeticMode::BoundedCertification>(
    const PolyhedralPhaseSampleFor<ArithmeticMode::BoundedCertification>&, PolyhedralReconstructionOptions);
template PolyhedralResolutionAssessment PolyhedralReconstruction::assess_resolution_finite<ArithmeticMode::Ordinary>(
    const PolyhedralPhaseSampleFor<ArithmeticMode::Ordinary>&,
    std::span<const PolyhedralPhaseSampleFor<ArithmeticMode::Ordinary>>, PolyhedralReconstructionOptions);
template PolyhedralResolutionAssessment
PolyhedralReconstruction::assess_resolution_finite<ArithmeticMode::BoundedCertification>(
    const PolyhedralPhaseSampleFor<ArithmeticMode::BoundedCertification>&,
    std::span<const PolyhedralPhaseSampleFor<ArithmeticMode::BoundedCertification>>, PolyhedralReconstructionOptions);
template PolyhedralInterface PolyhedralReconstruction::solve_offset_finite<ArithmeticMode::Ordinary>(
    const PolyhedralPhaseSampleFor<ArithmeticMode::Ordinary>&, PLIC::GeometryPoint, PolyhedralReconstructionOptions);
template PolyhedralInterface PolyhedralReconstruction::solve_offset_finite<ArithmeticMode::BoundedCertification>(
    const PolyhedralPhaseSampleFor<ArithmeticMode::BoundedCertification>&, PLIC::GeometryPoint,
    PolyhedralReconstructionOptions);
template PolyhedralInterface PolyhedralReconstruction::reconstruct_finite<ArithmeticMode::Ordinary>(
    const PolyhedralPhaseSampleFor<ArithmeticMode::Ordinary>&,
    std::span<const PolyhedralPhaseSampleFor<ArithmeticMode::Ordinary>>, PolyhedralReconstructionOptions);
template PolyhedralInterface PolyhedralReconstruction::reconstruct_finite<ArithmeticMode::BoundedCertification>(
    const PolyhedralPhaseSampleFor<ArithmeticMode::BoundedCertification>&,
    std::span<const PolyhedralPhaseSampleFor<ArithmeticMode::BoundedCertification>>, PolyhedralReconstructionOptions);
} // namespace SimpleFluid::VoF
