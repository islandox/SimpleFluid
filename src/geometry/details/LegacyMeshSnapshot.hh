/** @file LegacyMeshSnapshot.hh
 * @brief Explicit compatibility materialization for deprecated legacy mesh factories.
 */
#pragma once
#include "SimpleFluidExport.hh"
SIMPLEFLUID_LEGACY_API_BEGIN

#include "geometry/Mesh.hh"
#include "geometry/mesh/UnstructuredMeshData.hh"

namespace SimpleFluid::detail
{
/**
 * @brief Materialize legacy Mesh storage from detached records on the default communicator.
 *
 * This compatibility helper is separate from native factory construction. Its
 * concrete snapshot is assembled during construction; subsequent assemble()
 * calls are no-ops. Source geometry may reside only on rank zero. The legacy
 * partitioner and map builder retain their established numbering policy.
 *
 * Other type packs require geometry/details/LegacyMeshSnapshot.tcc or a custom
 * compiled provider. The library supplies the default type pack.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
SIMPLEFLUID_LOCAL SP<Mesh<Pack>> make_legacy_mesh_snapshot(Meshes::UnstructuredMeshData data);

extern template SP<Mesh<DefaultTpetraTypes>> make_legacy_mesh_snapshot<DefaultTpetraTypes>(
    Meshes::UnstructuredMeshData data);
} // namespace SimpleFluid::detail

SIMPLEFLUID_LEGACY_API_END
