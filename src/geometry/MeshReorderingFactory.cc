/**
 * @file MeshReorderingFactory.cc
 * @brief Explicit template instantiation for MeshReorderingFactory.
 */

#include "geometry/MeshReorderingFactory.hh"
#include "geometry/MeshReorderingFactory.tcc"

namespace SimpleFluid
{

template class MeshReorderingFactory<DefaultTpetraTypes>;

} // namespace SimpleFluid
