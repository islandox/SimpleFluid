/** @brief Test-owned provider for the custom partitioned-unstructured field backend.
 * Keep generic implementation opt-in out of both mesh tests and ordinary field consumers.
 */
#include "fields/FieldStored.tcc"
#include "geometry/mesh/PartitionedMeshBase.hh"
#include "geometry/mesh/UnstructuredMesh.hh"

namespace SimpleFluid
{
namespace
{
using ProviderMesh = Meshes::PartitionedMesh<Meshes::UnstructuredMesh, DefaultTpetraTypes>;
using Storage = ScalarCellFieldStored<DefaultTpetraTypes, ProviderMesh>;
} // namespace

template Storage::FieldStored(Storage::descriptor_type, SP<const ProviderMesh>, bool);
} // namespace SimpleFluid
