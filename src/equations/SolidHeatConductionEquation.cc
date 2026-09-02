/**
 * @file SolidHeatConductionEquation.cc
 * @brief Explicit instantiation of the default solid conduction equation.
 */

#include "SolidHeatConductionEquation.hh"
#include "SolidHeatConductionEquation.tcc"
#include "TemperatureDiffusionEquation.tcc"

namespace SimpleFluid
{
template class SolidHeatConductionEquation<DefaultTpetraTypes, SolidSubdomain<DefaultTpetraTypes>>;
} // namespace SimpleFluid
