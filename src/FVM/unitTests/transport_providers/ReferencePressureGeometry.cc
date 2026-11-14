/** @brief Test-owned pressure geometry providers for the reference comparison. */
#include "FVM/details/FieldStoredPressureWeightedFaceFlux.tcc"
#include "TestMeshes.hh"

namespace SimpleFluid::FVM::detail
{
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using Reference = test::transport_providers::ReferenceHandle;
template auto stored_pressure_face_geometry<Pack, Handle>(const Handle&)
    -> std::vector<StoredPressureFaceGeometry<Pack>>;
template auto stored_pressure_face_geometry<Pack, Reference>(const Reference&)
    -> std::vector<StoredPressureFaceGeometry<Pack>>;
} // namespace SimpleFluid::FVM::detail
