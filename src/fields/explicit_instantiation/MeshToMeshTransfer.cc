/** @file MeshToMeshTransfer.cc
 * @brief Instantiates conservative mesh transfer for the default backend.
 */
#include "fields/MeshToMeshTransfer.hh"
#include "fields/MeshToMeshTransfer.tcc"

namespace SimpleFluid
{
template class MeshToMeshTransfer<DefaultTpetraTypes>;
}
