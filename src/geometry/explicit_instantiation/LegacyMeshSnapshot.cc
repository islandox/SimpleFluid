/** @file LegacyMeshSnapshot.cc
 * @brief Default compiled provider for deprecated legacy factory snapshots.
 */
#include "SimpleFluidExport.hh"
SIMPLEFLUID_LEGACY_API_BEGIN

#include "geometry/details/LegacyMeshSnapshot.hh"
#include "geometry/details/LegacyMeshSnapshot.tcc"

namespace SimpleFluid::detail
{
template SP<Mesh<DefaultTpetraTypes>> make_legacy_mesh_snapshot<DefaultTpetraTypes>(Meshes::UnstructuredMeshData data);
} // namespace SimpleFluid::detail

SIMPLEFLUID_LEGACY_API_END
