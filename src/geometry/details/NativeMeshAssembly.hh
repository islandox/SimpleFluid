/** @file NativeMeshAssembly.hh
 * @brief Detached native topology and metric assembly from ordered cells.
 */
#pragma once

#include "geometry/mesh/UnstructuredMeshData.hh"

namespace SimpleFluid::detail
{
/** Ordered HEX_8/WEDGE_6 node ordinals for native assembly and generators. */
const std::vector<std::vector<unsigned>>& native_mesh_side_node_ordinals(MeshUtils::CellType type);

/** A boundary label on a HEX_8/WEDGE_6 topology side, numbered from zero. */
struct NativeBoundarySide
{
    Meshes::UnstructuredMeshData::ID cell_id;
    unsigned side;
    std::string boundary_name;
};

/**
 * Build faces and exact existing polygon/tetrahedral metrics from ordered cells.
 * Cell node ordering follows HEX_8 or WEDGE_6; face IDs are the minimum
 * adjacent cell ID * 10 + side + 1. Boundary IDs follow first face encounter.
 */
Meshes::UnstructuredMeshData build_native_mesh_data(std::vector<Meshes::UnstructuredMeshData::Node> nodes,
    std::vector<Meshes::UnstructuredMeshData::Cell> cells, const std::vector<NativeBoundarySide>& boundaries,
    const ArrString& boundary_names);

} // namespace SimpleFluid::detail
