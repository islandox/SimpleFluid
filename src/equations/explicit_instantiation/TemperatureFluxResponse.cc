/**
 * @file TemperatureFluxResponse.cc
 * @brief Internal default providers for the prepared temperature flux response.
 */

#include "FVM/details/PrescribedFluxResponse.tcc"
#include "geometry/SolidSubdomain.hh"

namespace SimpleFluid::FVM::detail
{
template class PrescribedFluxResponsePlan<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
template class PrescribedFluxResponsePlan<DefaultTpetraTypes, SolidSubdomain<DefaultTpetraTypes>>;
} // namespace SimpleFluid::FVM::detail
