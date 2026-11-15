/** @file LegacyMeshSnapshot.cc
 * @brief Default compiled provider for deprecated legacy factory snapshots.
 */

#include "geometry/details/LegacyMeshSnapshot.hh"
#include "geometry/details/LegacyMeshSnapshot.tcc"

namespace SimpleFluid::detail
{
template SP<Mesh<DefaultTpetraTypes>> make_legacy_mesh_snapshot<DefaultTpetraTypes>(Meshes::UnstructuredMeshData data);
} // namespace SimpleFluid::detail
