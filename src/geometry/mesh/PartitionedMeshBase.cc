/**
 * @file PartitionedMeshBase.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Explicit template instantiations for PartitionedMesh.
 * @version 0.1
 * @date 2026-07-21
 *
 * @details Builds the supported Cartesian and unstructured geometry wrappers
 * with DefaultTpetraTypes into SimpleFluidMesh. Other geometry/type-pack pairs
 * can be instantiated from PartitionedMeshBase.tcc.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "geometry/mesh/PartitionedMeshBase.tcc"

#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/UnstructuredMesh.hh"

namespace SimpleFluid::Meshes
{

/** @brief Default partitioned Cartesian mesh used by MeshHandle and fields. */
template class PartitionedMesh<OrthogonalCartesian3D, DefaultTpetraTypes>;

/** @brief Default partitioned unstructured mesh used by MeshPartitioner. */
template class PartitionedMesh<UnstructuredMesh, DefaultTpetraTypes>;

} // namespace SimpleFluid::Meshes
