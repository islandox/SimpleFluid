/** @file MeshRegion.cc
 * @brief Compiled native static-region adapters.
 */
#include "geometry/mesh/MeshRegion.tcc"

namespace SimpleFluid::Meshes
{
template class NativeRegionProvider<OrthogonalCartesian3D>;
template class NativeRegionProvider<OrthogonalCylindrial3D>;
template class NativeRegionProvider<SemiStructuredXY_Z>;
template class NativeRegionProvider<UnstructuredMesh>;
} // namespace SimpleFluid::Meshes
