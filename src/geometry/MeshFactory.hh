/**
 * @file MeshFactory.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Factory that constructs mesh instances from database configuration.
 * @version 0.1
 * @date 2026-05-25
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#pragma once

#include "Mesh.hh"
#include "MeshHandle.hh"
#include "STKMesh.hh"
#include "dataclass/Database.hh"
#include "geometry/BoundaryLayerMeshFactory.hh"

#include <optional>

namespace SimpleFluid
{
/**
 * @brief Factory that constructs mesh instances from database configuration.
 *
 * The factory reads mesh shape, size, and boundary metadata from a
 * Database and produces a concrete Mesh implementation. In parallel,
 * programmatic global geometry is constructed only on rank zero and then
 * partitioned into rank-local owned and ghost cells.
 */
class MeshFactory
{
public:
    /**
     * @brief Supported domain types for mesh generation.
     */
    enum class DomainType : uint8_t
    {
        BOX = 0,
        CYLINDER = 1,
        SPHERE = 2,
        EXTERNAL = 3,
        ANNULUS = 4
    };

public:
    MeshFactory(SP<const Database> db);

    template <TpetraTypePack Pack = DefaultTpetraTypes>
    SP<Mesh<Pack>> build();

    template <TpetraTypePack Pack = DefaultTpetraTypes>
    SP<MeshHandle<Pack>> build_handle();

    template <TpetraTypePack Pack = DefaultTpetraTypes>
    static SP<Mesh<Pack>> build_empty_mesh();

private:
    using BoundaryLayerSpec =
        BoundaryLayerMeshFactory::BoundaryLayerSpec;

    template <TpetraTypePack Pack>
    void build_box_mesh(SP<STKMesh<Pack>>& mesh);

    template <TpetraTypePack Pack>
    void build_cylinder_mesh(SP<STKMesh<Pack>>& mesh);

    template <TpetraTypePack Pack>
    void build_annulus_mesh(SP<STKMesh<Pack>>& mesh);

    template <TpetraTypePack Pack>
    void build_sphere_mesh(SP<STKMesh<Pack>>& mesh);

    const BoundaryLayerSpec* boundary_layer_spec(
        const std::string& boundary_name) const;

    void validate_boundary_layer_names() const;

private:
    DomainType d_domain_type;
    int d_dimension;
    Vec3D<ArrReal> d_box_cell_edges;
    Vec3D<ArrReal> d_annulus_cell_edges;
    real_t d_mesh_size;
    real_t d_radius;
    real_t d_cylinder_height;
    real_t d_cylinder_circumferential_mesh_size = 0.0;
    std::string d_external_mesh_file;

    // types of exterior faces for each boundary part, used for setting BCs in assembly
    // Face orders are: - BOX: {X-, X+, Y-, Y+, Z-, Z+}
    //                   - CYLINDER: {radial, top, bottom}
    //                   - periodic ANNULUS: {rmin, rmax, zmin, zmax}
    //                   - sector ANNULUS:
    //                     {rmin, rmax, thetamin, thetamax, zmin, zmax}
    Arr<std::string> d_domain_exterior_face_types;
    Arr<BoundaryLayerSpec> d_boundary_layer_specs;
};
}
