/** @brief Test-owned cold storage provider for the static native wrapper. */
#include "fields/FieldStored.tcc"
#include "geometry/mesh/PartitionedMeshBase.hh"
#include "geometry/mesh/UnstructuredMesh.hh"

namespace SimpleFluid
{
using TestPartitionedUnstructured = Meshes::PartitionedMesh<Meshes::UnstructuredMesh, DefaultTpetraTypes>;
template class FieldStored<ScalarCellFieldDescriptor<DefaultTpetraTypes>, DefaultTpetraTypes,
    TestPartitionedUnstructured>;
} // namespace SimpleFluid
