/**
 * @file SemiStructuredIndexer.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Indexing utilities for meshes formed by extruding a 2D topology.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "geometry/mesh/MeshIndexTypes.hh"

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>

namespace SimpleFluid::Meshes
{

/**
 * @brief Indexer for a layered extrusion of an arbitrary 2D mesh.
 *
 * The `ij` component identifies an entity in the base 2D topology and `k`
 * identifies its extrusion layer. Axial faces lie on layer interfaces and
 * therefore use the cell count of the base topology. Other faces are the
 * extrusions of base-topology faces and repeat once per cell layer.
 */
struct SemiStructuredIndexer
{
    using Ordinal = unsigned;

    /** @brief Base-topology and axial logical dimensions. */
    enum Dimension : std::uint8_t
    {
        IJ = 0,
        K = 1
    };

    /** @brief Axial-plane and extruded-side face orientations. */
    enum FaceOrientation : std::uint8_t
    {
        AXIAL = 0,
        SIDE = 1
    };

    /** @brief Base-cell ordinal and axial cell layer. */
    struct CellID
    {
        Ordinal ij = -1;
        Ordinal k = -1;

        constexpr auto operator<=>(const CellID&) const = default;
    };

    /** @brief Base-entity ordinal, axial layer, and face orientation. */
    struct FaceID
    {
        Ordinal ij = -1;
        Ordinal k = -1;
        unsigned orientation = -1;

        constexpr auto operator<=>(const FaceID&) const = default;
    };

    /** @brief Base-node ordinal and axial node layer. */
    struct NodeID
    {
        Ordinal ij = -1;
        Ordinal k = -1;

        constexpr auto operator<=>(const NodeID&) const = default;
    };

    using cell_id_t = CellID;
    using face_id_t = FaceID;
    using node_id_t = NodeID;
    using ordinal_t = size_t;

    Ordinal num_cells_per_layer = 0;
    Ordinal num_side_faces_per_layer = 0;
    Ordinal num_nodes_per_layer = 0;
    Ordinal num_layers = 0;
    Ordinal num_node_layers = 0;
    bool axial_periodic = false;

    std::array<size_t, 2> num_faces_per_orientation{};
    std::array<size_t, 2> face_offsets{};
    std::array<std::array<size_t, 2>, 2> face_strides{};

    SemiStructuredIndexer() = default;

    /**
     * @param cells_per_layer Number of cells in the base 2D topology.
     * @param side_faces_per_layer Number of non-axial faces per cell layer.
     * @param nodes_per_layer Number of nodes in the base 2D topology.
     * @param layers Number of extruded cell layers.
     * @param axial_periodic Whether the extrusion direction is periodic.
     */
    constexpr SemiStructuredIndexer(
        Ordinal cells_per_layer,
        Ordinal side_faces_per_layer,
        Ordinal nodes_per_layer,
        Ordinal layers,
        bool axial_periodic = false)
        : num_cells_per_layer(cells_per_layer),
          num_side_faces_per_layer(side_faces_per_layer),
          num_nodes_per_layer(nodes_per_layer),
          num_layers(layers),
          num_node_layers(axial_periodic ? layers : layers + 1),
          axial_periodic(axial_periodic),
          num_faces_per_orientation{
              static_cast<size_t>(cells_per_layer) * num_node_layers,
              static_cast<size_t>(side_faces_per_layer) * layers},
          face_offsets{
              0,
              num_faces_per_orientation[AXIAL]},
          face_strides{{
              {{num_node_layers, 1}},
              {{1, side_faces_per_layer}}}}
    {
    }

    constexpr size_t total_cells() const noexcept
    {
        return static_cast<size_t>(num_cells_per_layer) * num_layers;
    }

    constexpr size_t total_faces() const noexcept
    {
        return num_faces_per_orientation[AXIAL]
             + num_faces_per_orientation[SIDE];
    }

    constexpr size_t total_nodes() const noexcept
    {
        return static_cast<size_t>(num_nodes_per_layer)
             * num_node_layers;
    }

    constexpr size_t cell_ordinal(const CellID& cell_id) const noexcept
    {
        return cell_id.ij
             + static_cast<size_t>(num_cells_per_layer) * cell_id.k;
    }

    constexpr size_t face_ordinal(const FaceID& face_id) const noexcept
    {
        const auto orientation = face_id.orientation;
        return face_id.ij * face_strides[orientation][IJ]
             + face_id.k * face_strides[orientation][K]
             + face_offsets[orientation];
    }

    constexpr size_t node_ordinal(const NodeID& node_id) const noexcept
    {
        return node_id.ij
             + static_cast<size_t>(num_nodes_per_layer) * node_id.k;
    }

    constexpr size_t cell_local_id(const CellID& cell_id) const noexcept
    {
        return cell_ordinal(cell_id);
    }

    constexpr size_t face_local_id(const FaceID& face_id) const noexcept
    {
        return face_ordinal(face_id);
    }

    constexpr size_t node_local_id(const NodeID& node_id) const noexcept
    {
        return node_ordinal(node_id);
    }

    constexpr CellID cell_id(size_t local_id) const noexcept
    {
        return {
            static_cast<Ordinal>(local_id % num_cells_per_layer),
            static_cast<Ordinal>(local_id / num_cells_per_layer)};
    }

    constexpr FaceID face_id(size_t local_id) const noexcept
    {
        if (local_id < num_faces_per_orientation[AXIAL])
        {
            return {
                static_cast<Ordinal>(local_id / num_node_layers),
                static_cast<Ordinal>(local_id % num_node_layers),
                AXIAL};
        }

        local_id -= num_faces_per_orientation[AXIAL];
        return {
            static_cast<Ordinal>(local_id % num_side_faces_per_layer),
            static_cast<Ordinal>(local_id / num_side_faces_per_layer),
            SIDE};
    }

    constexpr NodeID node_id(size_t local_id) const noexcept
    {
        return {
            static_cast<Ordinal>(local_id % num_nodes_per_layer),
            static_cast<Ordinal>(local_id / num_nodes_per_layer)};
    }
};

using SemiStructuredMeshIndexTypes = MeshIndexTypes<
    SemiStructuredIndexer::CellID,
    SemiStructuredIndexer::FaceID,
    SemiStructuredIndexer::NodeID,
    size_t,
    uint64_t>;

} // namespace SimpleFluid::Meshes
