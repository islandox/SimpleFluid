/**
 * @file SolidSubdomain.cc
 * @brief Explicit template instantiation for SolidSubdomain.
 */

#include "geometry/SolidSubdomain.hh"
#include "geometry/SolidSubdomain.tcc"

namespace SimpleFluid
{

template class SolidSubdomain<DefaultTpetraTypes>;

} // namespace SimpleFluid
