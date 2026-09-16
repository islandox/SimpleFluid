#include "equations/RadiolyticGasModel.tcc"

namespace SimpleFluid
{
template class RadiolyticGasModel<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class RadiolyticGasModel<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
}
