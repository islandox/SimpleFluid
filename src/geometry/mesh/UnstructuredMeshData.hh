/** @file UnstructuredMeshData.hh
 * @brief STK-free construction records for native unstructured geometry.
 */
#pragma once

#include "geometry/MeshUtils.hh"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace SimpleFluid::Meshes
{
/**
 * @brief Detached, movable topology and authoritative metric input.
 *
 * Connectivity refers to source IDs, never vector positions. Record order is
 * the desired compact local order; owned entities precede overlap entities.
 * An omitted algebraic ID uses the source ID. The full boundary catalog is
 * independent of the locally incident faces and survives empty partitions.
 */
struct UnstructuredMeshData
{
    using ID = std::uint64_t;
    using Vec3 = MeshUtils::Vec3;
    static constexpr ID invalid_id = std::numeric_limits<ID>::max();
    static constexpr size_t all_owned = std::numeric_limits<size_t>::max();

    struct Node
    {
        ID id = invalid_id;
        Vec3 coordinates{};
        ID algebraic_id = invalid_id;
    };
    struct Cell
    {
        ID id = invalid_id;
        MeshUtils::CellType type = MeshUtils::CellType::INVALID;
        std::vector<ID> node_ids;
        std::vector<ID> face_ids;
        Vec3 center{};
        real_t volume = 0.0;
        ID algebraic_id = invalid_id;
    };
    struct Face
    {
        ID id = invalid_id;
        std::vector<ID> node_ids;
        ID owner = invalid_id;
        ID neighbor = invalid_id;
        int boundary_id = -1;
        Vec3 center{};
        Vec3 normal_from_owner{};
        real_t area = 0.0;
        ID algebraic_id = invalid_id;
    };
    std::vector<Node> nodes;
    std::vector<Cell> cells;
    std::vector<Face> faces;
    std::unordered_map<int, std::string> boundary_names;
    size_t num_owned_cells = all_owned;
    size_t num_owned_faces = all_owned;
    size_t num_owned_nodes = all_owned;
};
} // namespace SimpleFluid::Meshes
