/** @file PhaseAllocation.cc
 * @brief Supported Ordinary and Bounded phase allocation instantiations.
 */
#include "FVM/details/GeneralVoF/PhaseAllocation.tcc"
namespace SimpleFluid::VoF::general_vof_detail
{
template struct PhaseSum<ArithmeticMode::Ordinary>;
template struct PhaseLabel<ArithmeticMode::Ordinary>;
template PhaseValues phase_inventory<ArithmeticMode::Ordinary>(
    const Cell&, const PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>&);
template PhaseLabel<ArithmeticMode::Ordinary> phase_label<ArithmeticMode::Ordinary>(
    const Cell&, const PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>&, const PLIC::PolyhedralInterface&, double);
template AllocationProvenance allocation_provenance<ArithmeticMode::Ordinary>(
    PhaseValue, PhaseValue, PhaseValue, PhaseValue, PhaseValue, long double);
template struct PhaseSum<ArithmeticMode::BoundedCertification>;
template struct PhaseLabel<ArithmeticMode::BoundedCertification>;
template PhaseValues phase_inventory<ArithmeticMode::BoundedCertification>(
    const Cell&, const PLIC::ConvexCellGeometry<ArithmeticMode::BoundedCertification>&);
template PhaseLabel<ArithmeticMode::BoundedCertification> phase_label<ArithmeticMode::BoundedCertification>(const Cell&,
    const PLIC::ConvexCellGeometry<ArithmeticMode::BoundedCertification>&, const PLIC::PolyhedralInterface&, double);
template AllocationProvenance allocation_provenance<ArithmeticMode::BoundedCertification>(
    PhaseValue, PhaseValue, PhaseValue, PhaseValue, PhaseValue, long double);
} // namespace SimpleFluid::VoF::general_vof_detail
