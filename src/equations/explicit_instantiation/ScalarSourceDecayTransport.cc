/**
 * @file ScalarSourceDecayTransport.cc
 * @brief  Explicit template instantiations.
 */
#include "equations/ScalarSourceDecayTransport.tcc"

namespace SimpleFluid
{
template class ScalarSourceDecayTransport<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class ScalarSourceDecayTransport<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid
