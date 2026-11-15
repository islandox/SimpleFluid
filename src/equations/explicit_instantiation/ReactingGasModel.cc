/** @file ReactingGasModel.cc
 * @brief Instantiates reacting gas models for supported mesh types.
 */
#include "utils/export_helper.hh"

#include "equations/ReactingGasModel.tcc"

namespace SimpleFluid
{
template class ReactingGasModel<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
}
