/** @file FieldStoredPartitionedCartesian.cc
 * @brief Compiled construction and bulk initialization for supported stored fields.
 */
#include "fields/FieldStored.tcc"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"

namespace SimpleFluid
{
namespace
{
using Pack = DefaultTpetraTypes;
using ProviderMesh = Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, Pack>;
using ScalarCellStorage = ScalarCellFieldStored<Pack, ProviderMesh>;
using VectorCellStorage = VectorCellFieldStored<Pack, ProviderMesh>;
using TensorCellStorage = TensorCellFieldStored<Pack, ProviderMesh>;
using ScalarFaceStorage = ScalarFaceFieldStored<Pack, ProviderMesh>;
using VectorFaceStorage = VectorFaceFieldStored<Pack, ProviderMesh>;
} // namespace

template ScalarCellStorage::FieldStored(ScalarCellStorage::descriptor_type, SP<const ProviderMesh>, bool);
template ScalarCellStorage::FieldStored(
    ScalarCellStorage::descriptor_type, SP<const ProviderMesh>, const ScalarCellStorage::value_type&);
template void ScalarCellStorage::put_value(const ScalarCellStorage::value_type&);

template VectorCellStorage::FieldStored(VectorCellStorage::descriptor_type, SP<const ProviderMesh>, bool);
template VectorCellStorage::FieldStored(
    VectorCellStorage::descriptor_type, SP<const ProviderMesh>, const VectorCellStorage::value_type&);
template void VectorCellStorage::put_value(const VectorCellStorage::value_type&);

template TensorCellStorage::FieldStored(TensorCellStorage::descriptor_type, SP<const ProviderMesh>, bool);
template TensorCellStorage::FieldStored(
    TensorCellStorage::descriptor_type, SP<const ProviderMesh>, const TensorCellStorage::value_type&);
template void TensorCellStorage::put_value(const TensorCellStorage::value_type&);

template ScalarFaceStorage::FieldStored(ScalarFaceStorage::descriptor_type, SP<const ProviderMesh>, bool);
template ScalarFaceStorage::FieldStored(
    ScalarFaceStorage::descriptor_type, SP<const ProviderMesh>, const ScalarFaceStorage::value_type&);
template void ScalarFaceStorage::put_value(const ScalarFaceStorage::value_type&);

template VectorFaceStorage::FieldStored(VectorFaceStorage::descriptor_type, SP<const ProviderMesh>, bool);
template VectorFaceStorage::FieldStored(
    VectorFaceStorage::descriptor_type, SP<const ProviderMesh>, const VectorFaceStorage::value_type&);
template void VectorFaceStorage::put_value(const VectorFaceStorage::value_type&);

} // namespace SimpleFluid
