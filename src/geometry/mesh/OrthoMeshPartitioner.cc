/**
 * @file OrthoMeshPartitioner.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Coordinate-slab partitioning for orthogonal meshes.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "geometry/mesh/OrthoMeshPartitioner.hh"

#include <algorithm>
#include <stdexcept>

namespace SimpleFluid::Meshes
{

/**
 * @brief Configure balanced slab partitioning for an orthogonal topology.
 * @param topology Topology retained by reference for subsequent queries.
 * @param coordinate Logical dimension along which to split the cells.
 * @param num_partitions Number of non-empty slabs to create.
 * @param ghost_layers Number of neighboring coordinate planes to overlap.
 * @throws std::invalid_argument If the dimension or partition count is invalid,
 *         or the topology has no cells.
 */
OrthoMeshPartitioner::OrthoMeshPartitioner(
    const Topology& topology,
    Dimension coordinate,
    size_t num_partitions,
    Ordinal ghost_layers)
    : d_topology(&topology),
      d_coordinate(coordinate),
      d_num_partitions(num_partitions),
      d_ghost_layers(ghost_layers)
{
    if (static_cast<size_t>(coordinate) >= 3)
    {
        throw std::invalid_argument(
            "Orthogonal mesh partition coordinate is invalid.");
    }
    if (num_partitions == 0)
    {
        throw std::invalid_argument(
            "Orthogonal mesh partition count must be positive.");
    }

    const auto& indexer = this->indexer();
    if (indexer.total_cells() == 0)
    {
        throw std::invalid_argument(
            "Cannot partition an orthogonal mesh with no cells.");
    }

    const auto coordinate_cells =
        indexer.num_cells_per_dim[coordinate];
    if (num_partitions > coordinate_cells)
    {
        throw std::invalid_argument(
            "Orthogonal mesh partition count exceeds the selected "
            "coordinate cell count.");
    }
}

/**
 * @brief Compute the half-open coordinate interval owned by a partition.
 * @param partition Partition index.
 * @return Balanced contiguous interval of cell coordinates.
 * @throws std::out_of_range If @p partition is invalid.
 */
auto OrthoMeshPartitioner::owned_coordinate_range(
    size_t partition) const -> CoordinateRange
{
    check_partition(partition);

    const auto coordinate_cells = static_cast<size_t>(
        indexer().num_cells_per_dim[d_coordinate]);
    const auto base_size = coordinate_cells / d_num_partitions;
    const auto remainder = coordinate_cells % d_num_partitions;
    const auto begin =
        partition * base_size + std::min(partition, remainder);
    const auto size = base_size + (partition < remainder ? 1 : 0);

    return {
        static_cast<Ordinal>(begin),
        static_cast<Ordinal>(begin + size)};
}

/**
 * @brief Locate the partition that owns a cell.
 * @param cell_id Structured cell identifier.
 * @return Owning partition index.
 * @throws std::out_of_range If @p cell_id is invalid.
 */
size_t OrthoMeshPartitioner::owner_partition(CellID cell_id) const
{
    check_cell_id(cell_id);

    const auto coordinate =
        static_cast<size_t>(cell_coordinate(cell_id));
    const auto coordinate_cells = static_cast<size_t>(
        indexer().num_cells_per_dim[d_coordinate]);
    const auto base_size = coordinate_cells / d_num_partitions;
    const auto remainder = coordinate_cells % d_num_partitions;
    const auto larger_partition_size = base_size + 1;
    const auto larger_partition_cells =
        remainder * larger_partition_size;

    if (coordinate < larger_partition_cells)
    {
        return coordinate / larger_partition_size;
    }
    return remainder
         + (coordinate - larger_partition_cells) / base_size;
}

/**
 * @brief Locate the partition that owns a face through its owner cell.
 * @param face_id Structured face identifier.
 * @return Owning partition index.
 * @throws std::out_of_range If @p face_id is invalid.
 */
size_t OrthoMeshPartitioner::owner_partition(FaceID face_id) const
{
    check_face_id(face_id);
    return owner_partition(d_topology->owner_cell(face_id));
}

/**
 * @brief Test whether a cell is owned by a partition.
 * @param partition Partition index.
 * @param cell_id Structured cell identifier.
 * @return True when the partition owns the cell.
 * @throws std::out_of_range If the partition or cell ID is invalid.
 */
bool OrthoMeshPartitioner::is_owned_cell(
    size_t partition,
    CellID cell_id) const
{
    check_partition(partition);
    return owner_partition(cell_id) == partition;
}

/**
 * @brief Test whether a cell lies in a partition's ghost slab.
 * @param partition Partition index.
 * @param cell_id Structured cell identifier.
 * @return True for a non-owned cell within the configured overlap depth.
 * @throws std::out_of_range If the partition or cell ID is invalid.
 */
bool OrthoMeshPartitioner::is_ghost_cell(
    size_t partition,
    CellID cell_id) const
{
    check_partition(partition);
    check_cell_id(cell_id);
    return !is_owned_cell(partition, cell_id)
        && is_ghost_coordinate(partition, cell_coordinate(cell_id));
}

/**
 * @brief Test whether a cell is owned or ghosted by a partition.
 * @param partition Partition index.
 * @param cell_id Structured cell identifier.
 * @return True when the cell is locally visible.
 * @throws std::out_of_range If the partition or cell ID is invalid.
 */
bool OrthoMeshPartitioner::is_local_cell(
    size_t partition,
    CellID cell_id) const
{
    return is_owned_cell(partition, cell_id)
        || is_ghost_cell(partition, cell_id);
}

/**
 * @brief Test whether a face's owner cell belongs to a partition.
 * @param partition Partition index.
 * @param face_id Structured face identifier.
 * @return True when the face is owned by the partition.
 * @throws std::out_of_range If the partition or face ID is invalid.
 */
bool OrthoMeshPartitioner::is_owned_face(
    size_t partition,
    FaceID face_id) const
{
    check_partition(partition);
    return owner_partition(face_id) == partition;
}

/**
 * @brief Count cells owned by one partition.
 * @param partition Partition index.
 * @return Number of owned cells.
 * @throws std::out_of_range If @p partition is invalid.
 */
size_t OrthoMeshPartitioner::num_owned_cells(size_t partition) const
{
    const auto range = owned_coordinate_range(partition);
    const auto& dimensions = indexer().num_cells_per_dim;
    size_t transverse_cells = 1;
    for (size_t dim = 0; dim < 3; ++dim)
    {
        if (dim != static_cast<size_t>(d_coordinate))
        {
            transverse_cells *= dimensions[dim];
        }
    }
    return range.size() * transverse_cells;
}

/**
 * @brief Count ghost cells visible to one partition.
 * @param partition Partition index.
 * @return Number of ghost cells after periodic wrapping, if applicable.
 * @throws std::out_of_range If @p partition is invalid.
 */
size_t OrthoMeshPartitioner::num_ghost_cells(size_t partition) const
{
    check_partition(partition);

    size_t ghost_coordinates = 0;
    const auto coordinate_cells =
        indexer().num_cells_per_dim[d_coordinate];
    for (Ordinal coordinate = 0;
         coordinate < coordinate_cells;
         ++coordinate)
    {
        if (is_ghost_coordinate(partition, coordinate))
        {
            ++ghost_coordinates;
        }
    }

    const auto& dimensions = indexer().num_cells_per_dim;
    size_t transverse_cells = 1;
    for (size_t dim = 0; dim < 3; ++dim)
    {
        if (dim != static_cast<size_t>(d_coordinate))
        {
            transverse_cells *= dimensions[dim];
        }
    }
    return ghost_coordinates * transverse_cells;
}

/**
 * @brief Enumerate cells owned by one partition in mesh-local order.
 * @param partition Partition index.
 * @return Structured IDs of owned cells.
 * @throws std::out_of_range If @p partition is invalid.
 */
std::vector<OrthoMeshPartitioner::CellID>
OrthoMeshPartitioner::owned_cells(size_t partition) const
{
    check_partition(partition);

    const auto& mesh_indexer = indexer();
    std::vector<CellID> cells;
    cells.reserve(num_owned_cells(partition));
    for (size_t local_id = 0;
         local_id < mesh_indexer.total_cells();
         ++local_id)
    {
        const auto cell_id = mesh_indexer.cell_id(local_id);
        if (is_owned_cell(partition, cell_id))
        {
            cells.push_back(cell_id);
        }
    }
    return cells;
}

/**
 * @brief Enumerate ghost cells visible to one partition in mesh-local order.
 * @param partition Partition index.
 * @return Structured IDs of ghost cells.
 * @throws std::out_of_range If @p partition is invalid.
 */
std::vector<OrthoMeshPartitioner::CellID>
OrthoMeshPartitioner::ghost_cells(size_t partition) const
{
    check_partition(partition);

    const auto& mesh_indexer = indexer();
    std::vector<CellID> cells;
    cells.reserve(num_ghost_cells(partition));
    for (size_t local_id = 0;
         local_id < mesh_indexer.total_cells();
         ++local_id)
    {
        const auto cell_id = mesh_indexer.cell_id(local_id);
        if (is_ghost_cell(partition, cell_id))
        {
            cells.push_back(cell_id);
        }
    }
    return cells;
}

/**
 * @brief Extract a cell's coordinate along the partition dimension.
 * @param cell_id Structured cell identifier.
 * @return I, J, or K coordinate selected at construction.
 */
auto OrthoMeshPartitioner::cell_coordinate(
    CellID cell_id) const noexcept -> Ordinal
{
    if (d_coordinate == Indexer::I)
    {
        return cell_id.i;
    }
    if (d_coordinate == Indexer::J)
    {
        return cell_id.j;
    }
    return cell_id.k;
}

/**
 * @brief Test whether a coordinate lies in a partition's overlap planes.
 * @param partition Partition index.
 * @param coordinate Cell coordinate along the partition dimension.
 * @return True when the coordinate is ghosted but not owned.
 * @throws std::out_of_range If @p partition is invalid.
 */
bool OrthoMeshPartitioner::is_ghost_coordinate(
    size_t partition,
    Ordinal coordinate) const
{
    const auto range = owned_coordinate_range(partition);
    if (range.contains(coordinate) || d_ghost_layers == 0)
    {
        return false;
    }

    const auto& mesh_indexer = indexer();
    const auto coordinate_cells = static_cast<size_t>(
        mesh_indexer.num_cells_per_dim[d_coordinate]);
    const auto coordinate_value = static_cast<size_t>(coordinate);
    const auto ghost_layers = std::min(
        static_cast<size_t>(d_ghost_layers),
        coordinate_cells);

    if (!mesh_indexer.periodic_dimensions[d_coordinate])
    {
        const auto begin = static_cast<size_t>(range.begin);
        const auto end = static_cast<size_t>(range.end);
        return (coordinate_value < begin
                && begin - coordinate_value <= ghost_layers)
            || (coordinate_value >= end
                && coordinate_value - end < ghost_layers);
    }

    const auto begin = static_cast<size_t>(range.begin);
    const auto last = static_cast<size_t>(range.end) - 1;
    for (size_t offset = 1; offset <= ghost_layers; ++offset)
    {
        const auto lower =
            (begin + coordinate_cells - offset) % coordinate_cells;
        const auto upper =
            (last + offset) % coordinate_cells;
        if (coordinate_value == lower || coordinate_value == upper)
        {
            return true;
        }
    }
    return false;
}

/**
 * @brief Validate a partition index.
 * @param partition Partition index to check.
 * @throws std::out_of_range If @p partition is invalid.
 */
void OrthoMeshPartitioner::check_partition(size_t partition) const
{
    if (partition >= d_num_partitions)
    {
        throw std::out_of_range(
            "Orthogonal mesh partition index is out of range.");
    }
}

/**
 * @brief Validate a structured cell identifier.
 * @param cell_id Cell identifier to check.
 * @throws std::out_of_range If any coordinate is outside the topology.
 */
void OrthoMeshPartitioner::check_cell_id(CellID cell_id) const
{
    const auto& dimensions = indexer().num_cells_per_dim;
    if (cell_id.i >= dimensions[Indexer::I]
        || cell_id.j >= dimensions[Indexer::J]
        || cell_id.k >= dimensions[Indexer::K])
    {
        throw std::out_of_range(
            "Orthogonal mesh cell ID is out of range.");
    }
}

/**
 * @brief Validate a structured face identifier.
 * @param face_id Face identifier to check.
 * @throws std::out_of_range If its orientation or coordinates are invalid.
 */
void OrthoMeshPartitioner::check_face_id(FaceID face_id) const
{
    const auto orientation = static_cast<size_t>(face_id.orientation);
    if (orientation >= 3)
    {
        throw std::out_of_range(
            "Orthogonal mesh face orientation is out of range.");
    }

    const auto& dimensions =
        indexer().num_faces_per_dim_per_orientation[
            orientation];
    if (face_id.i >= dimensions[Indexer::I]
        || face_id.j >= dimensions[Indexer::J]
        || face_id.k >= dimensions[Indexer::K])
    {
        throw std::out_of_range(
            "Orthogonal mesh face ID is out of range.");
    }
}

} // namespace SimpleFluid::Meshes
