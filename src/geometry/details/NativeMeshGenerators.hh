/** @file NativeMeshGenerators.hh
 * @brief Detached programmatic geometry without a mesh database backend.
 */
#pragma once

#include "geometry/BoundaryLayerMeshFactory.hh"
#include "geometry/details/NativeMeshAssembly.hh"

namespace SimpleFluid::detail
{
struct NativeMeshGeneratorOptions
{
    uint8_t domain_type = 0;
    int dimension = 3;
    Vec3D<ArrReal> box_cell_edges;
    Vec3D<ArrReal> annulus_cell_edges;
    real_t mesh_size = 1.0;
    real_t radius = 1.0;
    real_t cylinder_height = 1.0;
    real_t cylinder_circumferential_mesh_size = 1.0;
    ArrString domain_exterior_face_types;
    Arr<BoundaryLayerMeshFactory::BoundaryLayerSpec> boundary_layer_specs;
};

/** Generate BOX=0, CYLINDER=1, SPHERE=2, ANNULUS=4; no MPI calls. */
Meshes::UnstructuredMeshData generate_native_mesh(const NativeMeshGeneratorOptions& options);
} // namespace SimpleFluid::detail
