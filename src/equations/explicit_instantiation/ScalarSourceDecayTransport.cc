/**
 * @file ScalarSourceDecayTransport.cc
 * @brief  Explicit template instantiations.
 */
#include "utils/export_helper.hh"

#include "equations/ScalarSourceDecayTransport.tcc"

namespace SimpleFluid
{
template class ScalarSourceDecayTransport<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid
