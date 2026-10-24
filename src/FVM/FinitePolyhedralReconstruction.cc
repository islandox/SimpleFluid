#include "FVM/FinitePolyhedralReconstruction.tcc"
namespace SimpleFluid::VoF
{
void PolyhedralReconstruction::validate(
    const FinitePolyhedralPhaseSample& sample, PolyhedralReconstructionOptions options)
{
    if (sample.geometry && sample.geometry->arithmetic_mode() == ArithmeticMode::Ordinary)
        return validate_finite<ArithmeticMode::Ordinary>(sample, options);
    return validate_finite<ArithmeticMode::BoundedCertification>(sample, options);
}
PolyhedralResolutionAssessment PolyhedralReconstruction::assess_resolution(const FinitePolyhedralPhaseSample& sample,
    std::span<const FinitePolyhedralPhaseSample> stencil, PolyhedralReconstructionOptions options)
{
    if (sample.geometry && sample.geometry->arithmetic_mode() == ArithmeticMode::Ordinary)
        return assess_resolution_finite<ArithmeticMode::Ordinary>(sample, stencil, options);
    return assess_resolution_finite<ArithmeticMode::BoundedCertification>(sample, stencil, options);
}
PolyhedralInterface PolyhedralReconstruction::solve_offset(
    const FinitePolyhedralPhaseSample& sample, Meshes::GeometryPoint normal, PolyhedralReconstructionOptions options)
{
    if (sample.geometry && sample.geometry->arithmetic_mode() == ArithmeticMode::Ordinary)
        return solve_offset_finite<ArithmeticMode::Ordinary>(sample, normal, options);
    return solve_offset_finite<ArithmeticMode::BoundedCertification>(sample, normal, options);
}
PolyhedralInterface PolyhedralReconstruction::reconstruct(const FinitePolyhedralPhaseSample& sample,
    std::span<const FinitePolyhedralPhaseSample> stencil, PolyhedralReconstructionOptions options)
{
    if (sample.geometry && sample.geometry->arithmetic_mode() == ArithmeticMode::Ordinary)
        return reconstruct_finite<ArithmeticMode::Ordinary>(sample, stencil, options);
    return reconstruct_finite<ArithmeticMode::BoundedCertification>(sample, stencil, options);
}
template void PolyhedralReconstruction::validate_finite<ArithmeticMode::Ordinary>(
    const FinitePolyhedralPhaseSample& sample, PolyhedralReconstructionOptions options);
template void PolyhedralReconstruction::validate_finite<ArithmeticMode::BoundedCertification>(
    const FinitePolyhedralPhaseSample& sample, PolyhedralReconstructionOptions options);
template PolyhedralResolutionAssessment PolyhedralReconstruction::assess_resolution_finite<ArithmeticMode::Ordinary>(
    const FinitePolyhedralPhaseSample& sample, std::span<const FinitePolyhedralPhaseSample> stencil,
    PolyhedralReconstructionOptions options);
template PolyhedralResolutionAssessment
PolyhedralReconstruction::assess_resolution_finite<ArithmeticMode::BoundedCertification>(
    const FinitePolyhedralPhaseSample& sample, std::span<const FinitePolyhedralPhaseSample> stencil,
    PolyhedralReconstructionOptions options);
template PolyhedralInterface PolyhedralReconstruction::solve_offset_finite<ArithmeticMode::Ordinary>(
    const FinitePolyhedralPhaseSample& sample, Meshes::GeometryPoint normal, PolyhedralReconstructionOptions options);
template PolyhedralInterface PolyhedralReconstruction::solve_offset_finite<ArithmeticMode::BoundedCertification>(
    const FinitePolyhedralPhaseSample& sample, Meshes::GeometryPoint normal, PolyhedralReconstructionOptions options);
template PolyhedralInterface PolyhedralReconstruction::reconstruct_finite<ArithmeticMode::Ordinary>(
    const FinitePolyhedralPhaseSample& sample, std::span<const FinitePolyhedralPhaseSample> stencil,
    PolyhedralReconstructionOptions options);
template PolyhedralInterface PolyhedralReconstruction::reconstruct_finite<ArithmeticMode::BoundedCertification>(
    const FinitePolyhedralPhaseSample& sample, std::span<const FinitePolyhedralPhaseSample> stencil,
    PolyhedralReconstructionOptions options);
} // namespace SimpleFluid::VoF
