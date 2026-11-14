/** @brief Compiled cold pressure-workspace members for the runtime handle. */
#include "FVM/FieldStoredFaceFlux.tcc"

namespace SimpleFluid::FVM
{
template FieldStoredPressureWeightedFaceFluxWorkspace<DefaultTpetraTypes,
    MeshHandle<DefaultTpetraTypes>>::SharedGeometry::SharedGeometry(SP<const mesh_type> mesh,
    std::shared_ptr<const topology_type> topology, std::shared_ptr<const traverse_plan_type> gradient_traverse);

template<>
FieldStoredPressureWeightedFaceFluxWorkspace<DefaultTpetraTypes,
    MeshHandle<DefaultTpetraTypes>>::FieldStoredPressureWeightedFaceFluxWorkspace(SP<const mesh_type> mesh)
    : FieldStoredPressureWeightedFaceFluxWorkspace(mesh, std::make_shared<SharedGeometry>(mesh))
{
}

template FieldStoredPressureWeightedFaceFluxWorkspace<DefaultTpetraTypes,
    MeshHandle<DefaultTpetraTypes>>::FieldStoredPressureWeightedFaceFluxWorkspace(SP<const mesh_type> mesh,
    shared_geometry_type geometry);

template const std::vector<typename FieldStoredPressureWeightedFaceFluxWorkspace<DefaultTpetraTypes,
    MeshHandle<DefaultTpetraTypes>>::face_geometry_type>&
FieldStoredPressureWeightedFaceFluxWorkspace<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>::face_geometry() const;

template const typename FieldStoredPressureWeightedFaceFluxWorkspace<DefaultTpetraTypes,
    MeshHandle<DefaultTpetraTypes>>::FaceBatches&
FieldStoredPressureWeightedFaceFluxWorkspace<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>::face_batches() const;

template void
FieldStoredPressureWeightedFaceFluxWorkspace<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>::refresh_geometry();

template void
FieldStoredPressureWeightedFaceFluxWorkspace<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>::refresh_geometry(
    shared_geometry_type geometry);
} // namespace SimpleFluid::FVM
