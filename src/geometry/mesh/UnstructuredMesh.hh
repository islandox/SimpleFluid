/**
 * @file UnstructuredMesh.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief STK-free unstructured finite-volume mesh.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "geometry/mesh/MeshBase.hh"

#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace SimpleFluid::Meshes
{

using UnstructuredMeshIndexTypes = MeshIndexTypes<
    uint64_t,
    uint64_t,
    uint64_t,
    size_t,
    uint64_t>;

/**
 * @brief Unstructured mesh backed by explicit node and cell lists.
 *
 * The mesh supports the same volume element families currently handled by
 * STKMesh: hexahedra and triangular prisms. Faces are derived from cell
 * connectivity, deduplicated by node set, and tagged with optional boundary
 * batches supplied by node set.
 */
class UnstructuredMesh
    : public MeshBase<UnstructuredMesh, UnstructuredMeshIndexTypes>
{
public:
    using Base = MeshBase<UnstructuredMesh, UnstructuredMeshIndexTypes>;
    using Ordinal = uint64_t;
    using CellID = typename Base::cell_id_t;
    using FaceID = typename Base::face_id_t;
    using NodeID = typename Base::node_id_t;
    using cell_id_t = typename Base::cell_id_t;
    using face_id_t = typename Base::face_id_t;
    using node_id_t = typename Base::node_id_t;
    using Vec3 = typename Base::Vec3;
    using BoundaryFaceBatch = typename Base::BoundaryFaceBatch;
    using BoundaryBatchMap = std::unordered_map<int, BoundaryFaceBatch>;
    using BoundaryNames = std::unordered_map<int, std::string>;
    using CellType = MeshUtils::CellType;
    using FaceType = MeshUtils::FaceType;

    static constexpr Ordinal invalid_ordinal =
        std::numeric_limits<Ordinal>::max();
    static constexpr cell_id_t invalid_cell_id() noexcept
    {
        return invalid_ordinal;
    }

    /** @brief Input cell topology and ordered node connectivity. */
    struct CellDefinition
    {
        CellType type = CellType::INVALID;
        Arr<NodeID> node_ids;
    };

    /** @brief Input boundary face connectivity, ID, and display name. */
    struct BoundaryFaceDefinition
    {
        Arr<NodeID> node_ids;
        int boundary_id = invalid_boundary_id;
        std::string name;
    };

    /** @brief Identity mapping between compact IDs and local ordinals. */
    struct Indexer
    {
        using cell_id_t = CellID;
        using face_id_t = FaceID;
        using node_id_t = NodeID;
        using ordinal_t = size_t;

        Ordinal cells = 0;
        Ordinal faces = 0;
        Ordinal nodes = 0;

        constexpr CellID cell_id(size_t ordinal) const noexcept
        {
            return static_cast<CellID>(ordinal);
        }
        constexpr FaceID face_id(size_t ordinal) const noexcept
        {
            return static_cast<FaceID>(ordinal);
        }
        constexpr NodeID node_id(size_t ordinal) const noexcept
        {
            return static_cast<NodeID>(ordinal);
        }
        constexpr size_t cell_ordinal(CellID id) const noexcept
        {
            return static_cast<size_t>(id);
        }
        constexpr size_t face_ordinal(FaceID id) const noexcept
        {
            return static_cast<size_t>(id);
        }
        constexpr size_t node_ordinal(NodeID id) const noexcept
        {
            return static_cast<size_t>(id);
        }

        constexpr size_t cell_local_id(CellID id) const noexcept
        {
            return cell_ordinal(id);
        }
        constexpr size_t face_local_id(FaceID id) const noexcept
        {
            return face_ordinal(id);
        }
        constexpr size_t node_local_id(NodeID id) const noexcept
        {
            return node_ordinal(id);
        }
    };

    UnstructuredMesh(
        const Arr<Vec3>& nodes,
        const Arr<CellDefinition>& cells,
        const Arr<BoundaryFaceDefinition>& boundary_faces = {});

    UnstructuredMesh(
        const Arr<Vec3>& nodes,
        const Arr<CellDefinition>& cells,
        const Arr<BoundaryFaceDefinition>& boundary_faces,
        size_t num_owned_cells,
        size_t num_owned_faces);

    UnstructuredMesh(const UnstructuredMesh&) = default;
    UnstructuredMesh(UnstructuredMesh&&) noexcept = default;
    UnstructuredMesh& operator=(const UnstructuredMesh&) = delete;
    UnstructuredMesh& operator=(UnstructuredMesh&& other) noexcept;

    const Indexer& indexer() const noexcept { return d_indexer; }
    const Arr<Vec3>& nodes() const noexcept { return d_nodes; }
    const BoundaryNames& boundary_names() const noexcept
    {
        return d_boundary_names;
    }

    const Arr<NodeID>& cell_nodes(CellID cell_id) const;
    const Arr<NodeID>& face_nodes(FaceID face_id) const;
    CellType cell_type(CellID cell_id) const;

private:
    friend Base;

    /** @brief Derived connectivity and geometry for one cell. */
    struct CellInfo
    {
        CellType type = CellType::INVALID;
        Arr<NodeID> node_ids;
        Arr<FaceID> face_ids;
        Vec3 center;
        real_t volume = 0.0;
    };

    /** @brief Derived connectivity, adjacency, and geometry for one face. */
    struct FaceInfo
    {
        FaceType type = FaceType::INVALID;
        Arr<NodeID> node_ids;
        CellID owner = invalid_ordinal;
        CellID neighbor = invalid_ordinal;
        int boundary_id = invalid_boundary_id;
        Vec3 center;
        Vec3 normal;
        real_t area = 0.0;
    };

    void check_cell_id(CellID cell_id) const;
    void check_face_id(FaceID face_id) const;
    void check_node_id(NodeID node_id) const;

    bool is_owned_cell_impl(CellID cell_id) const noexcept
    {
        return static_cast<size_t>(cell_id) < Base::d_num_owned_cells;
    }
    bool is_owned_face_impl(FaceID face_id) const noexcept
    {
        return static_cast<size_t>(face_id) < Base::d_num_owned_faces;
    }

    real_t cell_volume_impl(CellID cell_id) const;
    Vec3 cell_centroid_impl(CellID cell_id) const;
    const Arr<FaceID>& cell_faces_impl(CellID cell_id) const;

    CellID owner_cell_impl(FaceID face_id) const;
    CellID neighbor_cell_impl(FaceID face_id) const;

    real_t face_area_impl(FaceID face_id) const;
    Vec3 face_centroid_impl(FaceID face_id) const;
    Vec3 face_normal_impl(FaceID face_id) const;
    Vec3 node_coordinates_impl(NodeID node_id) const;

    int boundary_id_impl(FaceID face_id) const;
    const std::string& boundary_batch_name_impl(int batch_id) const;
    const BoundaryFaceBatch& boundary_face_batch_impl(int batch_id) const;
    std::vector<int> boundary_batch_ids_impl() const;
    int num_boundary_batches_impl() const noexcept;
    const BoundaryBatchMap& boundary_batches_impl() const noexcept
    {
        return d_boundary_batches;
    }

    void validate_input(
        const Arr<CellDefinition>& cells,
        const Arr<BoundaryFaceDefinition>& boundary_faces) const;
    void initialize_cells(const Arr<CellDefinition>& cells);
    void build_faces(
        const Arr<BoundaryFaceDefinition>& boundary_faces);
    void compute_face_geometry();
    void update_counts();

    Arr<Vec3> d_nodes;
    Arr<CellInfo> d_cells;
    Arr<FaceInfo> d_faces;
    BoundaryNames d_boundary_names;
    BoundaryBatchMap d_boundary_batches;
    Indexer d_indexer;
};

using UnstructuredMesh3D = UnstructuredMesh;

} // namespace SimpleFluid::Meshes

#include "geometry/mesh/UnstructuredMesh.ipp"
