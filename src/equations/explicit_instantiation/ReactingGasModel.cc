/** @file ReactingGasModel.cc
 * @brief Instantiates reacting gas models for supported mesh types.
 */
#include "utils/export_helper.hh"
SIMPLEFLUID_LEGACY_API_BEGIN

#include "equations/ReactingGasModel.tcc"

namespace SimpleFluid
{
template class ReactingGasModel<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class ReactingGasModel<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
}

SIMPLEFLUID_LEGACY_API_END
