/**
 * @file OrthoMeshPartitioner.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Coordinate-slab partitioning for orthogonal three-dimensional meshes.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "geometry/mesh/OrthoMeshTopo.hh"

#include <compare>
#include <cstddef>
#include <vector>

namespace SimpleFluid::Meshes
{

/**
 * @brief Partition an orthogonal mesh into balanced coordinate slabs.
 *
 * Cells are assigned to contiguous partitions along one index coordinate.
 * The first partitions receive one extra coordinate plane when the selected
 * dimension is not evenly divisible. Ghost cells are drawn from adjacent
 * coordinate planes and wrap when the partition coordinate is periodic.
 */
class OrthoMeshPartitioner
{
public:
    using Topology = OrthoMeshTopo;
    using Indexer = Topology::Indexer;
    using Ordinal = Indexer::Ordinal;
    using CellID = Indexer::CellID;
    using FaceID = Indexer::FaceID;
    using Dimension = Indexer::Dimension;

    /** @brief Half-open owned-cell interval along the partition coordinate. */
    struct CoordinateRange
    {
        Ordinal begin = 0;
        Ordinal end = 0;

        constexpr size_t size() const noexcept
        {
            return static_cast<size_t>(end - begin);
        }

        constexpr bool contains(Ordinal coordinate) const noexcept
        {
            return coordinate >= begin && coordinate < end;
        }

        constexpr auto operator<=>(const CoordinateRange&) const = default;
    };

    OrthoMeshPartitioner(
        const Topology& topology,
        Dimension coordinate,
        size_t num_partitions,
        Ordinal ghost_layers = 1);
    OrthoMeshPartitioner(
        Topology&&,
        Dimension,
        size_t,
        Ordinal = 1) = delete;

    const Topology& topology() const noexcept { return *d_topology; }
    Dimension coordinate() const noexcept { return d_coordinate; }
    size_t num_partitions() const noexcept { return d_num_partitions; }
    Ordinal ghost_layers() const noexcept { return d_ghost_layers; }

    CoordinateRange owned_coordinate_range(size_t partition) const;

    size_t owner_partition(CellID cell_id) const;
    size_t owner_partition(FaceID face_id) const;

    bool is_owned_cell(size_t partition, CellID cell_id) const;
    bool is_ghost_cell(size_t partition, CellID cell_id) const;
    bool is_local_cell(size_t partition, CellID cell_id) const;
    bool is_owned_face(size_t partition, FaceID face_id) const;

    size_t num_owned_cells(size_t partition) const;
    size_t num_ghost_cells(size_t partition) const;
    std::vector<CellID> owned_cells(size_t partition) const;
    std::vector<CellID> ghost_cells(size_t partition) const;

private:
    Ordinal cell_coordinate(CellID cell_id) const noexcept;
    bool is_ghost_coordinate(
        size_t partition,
        Ordinal coordinate) const;
    void check_partition(size_t partition) const;
    void check_cell_id(CellID cell_id) const;
    void check_face_id(FaceID face_id) const;

    const Indexer& indexer() const noexcept
    {
        return d_topology->indexer();
    }

    const Topology* d_topology;
    Dimension d_coordinate;
    size_t d_num_partitions;
    Ordinal d_ghost_layers;
};

} // namespace SimpleFluid::Meshes
