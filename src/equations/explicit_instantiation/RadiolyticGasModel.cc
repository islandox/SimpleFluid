/** @file RadiolyticGasModel.cc
 * @brief Instantiates radiolytic gas models for supported mesh types.
 */
#include "equations/RadiolyticGasModel.tcc"

namespace SimpleFluid
{
template class RadiolyticGasModel<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class RadiolyticGasModel<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
}
