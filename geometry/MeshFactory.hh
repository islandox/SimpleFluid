/**
 * @file MeshFactory.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief 
 * @version 0.1
 * @date 2026-05-25
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#pragma once

#include "Mesh.hh"
#include "STKMesh.hh"
#include "dataclass/Database.hh"

namespace SimpleFluid
{
/**
 * @brief Factory that constructs mesh instances from database configuration.
 *
 * The factory reads mesh shape, size, and boundary metadata from a
 * Database and produces a concrete Mesh implementation.
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
        EXTERNAL = 3
    };

public:
    MeshFactory(SP<const Database>& db);

    template <TpetraTypePack Pack = DefaultTpetraTypes>
    SP<Mesh<Pack>> build();

private:
    template <TpetraTypePack Pack>
    void build_box_mesh(SP<STKMesh<Pack>>& mesh);

private:
    DomainType d_domain_type;
    int d_dimension;
    Vec3D<ArrReal> d_box_cell_edges;
    real_t d_mesh_size;
    real_t d_radius;
    real_t d_cylinder_height;
    std::string d_external_mesh_file;

    // types of exterior faces for each boundary part, used for setting BCs in assembly
    // Face orders are: - BOX: {X-, X+, Y-, Y+, Z-, Z+}
    //                   - CYLINDER: {radial, top, bottom}
    Arr<std::string> d_domain_exterior_face_types;
};
}
