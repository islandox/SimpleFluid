/** @file ReactingGasModel.cc
 * @brief Instantiates reacting gas models for supported mesh types.
 */
#include "equations/ReactingGasModel.tcc"

namespace SimpleFluid
{
template class ReactingGasModel<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class ReactingGasModel<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
}
