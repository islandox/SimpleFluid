#include "geometry/PLIC/details/PolyhedralReconstruction.tcc"
namespace SimpleFluid::PLIC
{
template class PolyhedralReconstruction<ArithmeticMode::Ordinary>;
template class PolyhedralReconstruction<ArithmeticMode::BoundedCertification>;
} // namespace SimpleFluid::PLIC
