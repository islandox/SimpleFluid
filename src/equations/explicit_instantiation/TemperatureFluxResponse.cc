/**
 * @file TemperatureFluxResponse.cc
 * @brief Internal default mapped-mesh provider for the prepared temperature flux response.
 */

#include "FVM/details/PrescribedFluxResponse.tcc"

namespace SimpleFluid::FVM::detail
{
template class PrescribedFluxResponsePlan<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid::FVM::detail
