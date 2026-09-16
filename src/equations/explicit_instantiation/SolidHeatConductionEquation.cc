/**
 * @file SolidHeatConductionEquation.cc
 * @brief Explicit instantiation of the default solid conduction equation.
 */

#include "equations/SolidHeatConductionEquation.hh"
#include "equations/SolidHeatConductionEquation.tcc"
#include "equations/TemperatureDiffusionEquation.tcc"

namespace SimpleFluid
{
template class SolidHeatConductionEquation<DefaultTpetraTypes, SolidSubdomain<DefaultTpetraTypes>>;
} // namespace SimpleFluid
