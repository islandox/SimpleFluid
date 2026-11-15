/** @file VolumeContinuityModelTestProvider.cc
 * @brief Executable-local stored fields and preview for the observation mesh.
 */
#include "VolumeContinuityObservedMesh.hh"
#include "fields/FieldStored.tcc"
#include "solvers/VolumeContinuityModel.tcc"

namespace SimpleFluid
{
namespace
{
using Pack = DefaultTpetraTypes;
using ProviderMesh = test::VolumeContinuityObservedMesh;
using CellStorage = ScalarCellFieldStored<Pack, ProviderMesh>;
using FaceStorage = ScalarFaceFieldStored<Pack, ProviderMesh>;
using Model = VolumeContinuityModel<Pack, ProviderMesh>;
} // namespace

template CellStorage::FieldStored(CellStorage::descriptor_type, SP<const ProviderMesh>, bool);

template FaceStorage::FieldStored(FaceStorage::descriptor_type, SP<const ProviderMesh>, bool);
template FaceStorage::FieldStored(
    FaceStorage::descriptor_type, SP<const ProviderMesh>, const FaceStorage::value_type&);
template void FaceStorage::put_value(const FaceStorage::value_type&);

template auto Model::preview(const Model::Inputs&, std::uint64_t) const -> Model::Trial;

} // namespace SimpleFluid
