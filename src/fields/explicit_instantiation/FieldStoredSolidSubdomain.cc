/** @file FieldStoredSolidSubdomain.cc
 * @brief Compiled construction and bulk initialization for supported stored fields.
 */
#include "fields/FieldStored.tcc"
#include "geometry/SolidSubdomain.hh"

namespace SimpleFluid
{
namespace
{
using Pack = DefaultTpetraTypes;
using ProviderMesh = SolidSubdomain<Pack>;
using ScalarCellStorage = ScalarCellFieldStored<Pack, ProviderMesh>;
using VectorCellStorage = VectorCellFieldStored<Pack, ProviderMesh>;
using ScalarFaceStorage = ScalarFaceFieldStored<Pack, ProviderMesh>;
} // namespace

template ScalarCellStorage::FieldStored(ScalarCellStorage::descriptor_type, SP<const ProviderMesh>, bool);
template ScalarCellStorage::FieldStored(
    ScalarCellStorage::descriptor_type, SP<const ProviderMesh>, const ScalarCellStorage::value_type&);
template void ScalarCellStorage::put_value(const ScalarCellStorage::value_type&);

template VectorCellStorage::FieldStored(VectorCellStorage::descriptor_type, SP<const ProviderMesh>, bool);
template VectorCellStorage::FieldStored(
    VectorCellStorage::descriptor_type, SP<const ProviderMesh>, const VectorCellStorage::value_type&);
template void VectorCellStorage::put_value(const VectorCellStorage::value_type&);

template ScalarFaceStorage::FieldStored(ScalarFaceStorage::descriptor_type, SP<const ProviderMesh>, bool);
template ScalarFaceStorage::FieldStored(
    ScalarFaceStorage::descriptor_type, SP<const ProviderMesh>, const ScalarFaceStorage::value_type&);
template void ScalarFaceStorage::put_value(const ScalarFaceStorage::value_type&);

} // namespace SimpleFluid
