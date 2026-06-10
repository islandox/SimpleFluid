/**
 * @file MeshHandle.cc
 * @brief Explicit template instantiation for MeshHandle.
 */

#include "MeshHandle.hh"
#include "MeshHandle.tcc"

namespace SimpleFluid
{
template class MeshHandle<DefaultTpetraTypes>;
}
