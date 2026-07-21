/**
 * @file MeshHandle.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Variant-backed runtime handle for all supported mesh families.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "dataclass/TpetraTypes.hh"
#include "geometry/mesh/LocalGlobalIndexer.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "geometry/mesh/OrthoMeshPartitioner.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"
#include "geometry/mesh/STKMeshAdapter.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "geometry/mesh/UnstructuredMesh.hh"
#include "io/VTUWriter.hh"
#include "utils/debug_check.hh"

#include <Teuchos_OrdinalTraits.hpp>
#include <Tpetra_Core.hpp>

#include <algorithm>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace SimpleFluid
{

namespace detail
{

/**
 * @brief Concept satisfied when a mesh type accepts a given structured cell ID.
 */
template<class Mesh, class ID>
concept mesh_has_cell_local_id = requires(const Mesh& m, ID id) {
    { m.cell_local_id(id) } -> std::convertible_to<size_t>;
};

/**
 * @brief Concept satisfied when a mesh type accepts a given structured face ID.
 */
template<class Mesh, class ID>
concept mesh_has_face_local_id = requires(const Mesh& m, ID id) {
    { m.face_local_id(id) } -> std::convertible_to<size_t>;
};

} // namespace detail

/**
 * @brief Runtime-polymorphic distributed view of supported mesh families.
 *
 * MeshHandle normalizes structured, semi-structured, and legacy STK meshes
 * into one FVM-facing API. It builds owned/overlap maps and translates compact
 * local IDs to each concrete mesh's geometry IDs.
 *
 * @tparam Pack Tpetra scalar, ordinal, communicator, and map types.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class MeshHandle
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using map_type = typename Pack::map_type;
    using mesh_index_type_pack = MeshIndexTypes<
        global_ordinal_type,
        global_ordinal_type,
        global_ordinal_type,
        local_ordinal_type,
        global_ordinal_type>;
    using indexer_type = Meshes::LocalGlobalIndexer<mesh_index_type_pack>;
    using Vec3 = MeshUtils::Vec3;

    using Cartesian = Meshes::OrthogonalCartesian3D;
    using Cylindrical = Meshes::OrthogonalCylindrial3D;
    using SemiStructured = Meshes::SemiStructuredXY_Z;
    using Unstructured = Meshes::UnstructuredMesh;
    using STKAdapter = Meshes::STKMeshAdapter<Pack>;
    using unstructured_indexer_type =
        Unstructured::local_global_indexer_t<
            local_ordinal_type,
            global_ordinal_type>;

    using CartesianPtr = SP<const Cartesian>;
    using CylindricalPtr = SP<const Cylindrical>;
    using SemiStructuredPtr = SP<const SemiStructured>;
    using UnstructuredPtr = SP<const Unstructured>;
    using STKAdapterPtr = SP<const STKAdapter>;
    using variant_type = std::variant<CartesianPtr,
                                      CylindricalPtr,
                                      SemiStructuredPtr,
                                      UnstructuredPtr,
                                      STKAdapterPtr>;

    /** @brief Locally visible faces belonging to one boundary batch. */
    struct BoundaryFaceBatch
    {
        int id = -1;
        std::vector<local_ordinal_type> face_lids;
    };

    /** @brief Partition selection and ghost depth for structured meshes. */
    struct DistributionOptions
    {
        size_t ghost_layers = 1;
        std::optional<size_t> partition;
        std::optional<size_t> partitions;
    };

    static constexpr int invalid_boundary_id = -1;

    /** @brief Build a distributed handle for a Cartesian mesh. */
    explicit MeshHandle(CartesianPtr mesh,
                        DistributionOptions options = {});

    /** @brief Build a distributed handle for a cylindrical mesh. */
    explicit MeshHandle(CylindricalPtr mesh,
                        DistributionOptions options = {});

    /** @brief Build a handle for a semi-structured mesh. */
    explicit MeshHandle(SemiStructuredPtr mesh);

    /** @brief Build a serial unstructured mesh handle. */
    explicit MeshHandle(UnstructuredPtr mesh);

    /** @brief Build a handle from a previously partitioned mesh. */
    MeshHandle(
        UnstructuredPtr mesh,
        const unstructured_indexer_type& indexer);

    template<class Partitioned>
        requires Meshes::PartitionedMeshClass<
            std::remove_const_t<Partitioned>>
    explicit MeshHandle(const SP<Partitioned>& partitioned)
    {
        using partitioned_type = std::remove_const_t<Partitioned>;
        static_assert(std::is_same_v<
            typename partitioned_type::mesh_type,
            Unstructured>);
        static_assert(std::is_same_v<
            typename partitioned_type::tpetra_type_pack,
            Pack>);
        if (!partitioned)
        {
            throw std::invalid_argument(
                "MeshHandle requires a non-null partitioned mesh.");
        }
        d_mesh = UnstructuredPtr(partitioned->mesh_ptr());
        initialize_unstructured(
            std::get<UnstructuredPtr>(d_mesh),
            partitioned->indexer());
    }

    /** @brief Build a handle around a legacy STK adapter. */
    explicit MeshHandle(STKAdapterPtr mesh);

    /** @brief Adapt a legacy distributed mesh and build a runtime handle. */
    explicit MeshHandle(SP<const SimpleFluid::Mesh<Pack>> mesh);

    const variant_type& variant() const noexcept { return d_mesh; }
    const indexer_type& indexer() const noexcept { return d_indexer; }

    /**
     * @brief Invoke a visitor with the concrete mesh object.
     *
     * Pointer alternatives are dereferenced before dispatch.
     */
    template<class Visitor>
    decltype(auto) visit(Visitor&& visitor) const
    {
        return std::visit(
            [&](const auto& mesh) -> decltype(auto)
            {
                return std::forward<Visitor>(visitor)(*mesh);
            },
            d_mesh);
    }

    bool is_stk() const noexcept
    {
        return std::holds_alternative<STKAdapterPtr>(d_mesh);
    }

    SP<const SimpleFluid::Mesh<Pack>> legacy_mesh() const noexcept
    {
        if (const auto* adapter = std::get_if<STKAdapterPtr>(&d_mesh))
        {
            return (*adapter)->mesh_ptr();
        }
        return {};
    }

    size_t spatial_dimension() const noexcept { return 3; }
    size_t num_owned_cells() const noexcept
    {
        return d_indexer.num_owned_cells();
    }
    size_t num_local_cells() const noexcept
    {
        return d_indexer.num_local_cells();
    }
    size_t num_cells() const noexcept { return num_local_cells(); }
    size_t num_owned_faces() const noexcept
    {
        return d_indexer.num_owned_faces();
    }
    size_t num_faces() const noexcept
    {
        return d_indexer.num_local_faces();
    }

    bool is_owned_cell(local_ordinal_type cell_lid) const
    {
        check_cell(cell_lid);
        return d_indexer.is_owned_cell(cell_lid);
    }

    bool is_owned_face(local_ordinal_type face_lid) const
    {
        check_face(face_lid);
        return d_indexer.is_owned_face(face_lid);
    }

    global_ordinal_type cell_global_id(local_ordinal_type cell_lid) const
    {
        check_cell(cell_lid);
        if (const auto legacy = legacy_mesh())
        {
            return legacy->mesh_gid_to_tpetra_gid(
                legacy->cell_global_id(checked_local(static_cast<size_t>(
                    geometry_cell_lid(cell_lid)))));
        }
        return d_indexer.cell_global_id(cell_lid);
    }

    global_ordinal_type face_global_id(local_ordinal_type face_lid) const
    {
        check_face(face_lid);
        if (const auto legacy = legacy_mesh())
        {
            return legacy->face_global_id(
                checked_local(static_cast<size_t>(
                    geometry_face_lid(face_lid))));
        }
        return d_indexer.face_global_id(face_lid);
    }

    real_t cell_volume(local_ordinal_type cell_lid) const;
    Vec3 cell_centroid(local_ordinal_type cell_lid) const;
    std::span<const local_ordinal_type> faces(
        local_ordinal_type cell_lid) const;
    local_ordinal_type owner_cell(local_ordinal_type face_lid) const;
    local_ordinal_type neighbor_cell(local_ordinal_type face_lid) const;
    local_ordinal_type opposite_cell(local_ordinal_type face_lid,
                                     local_ordinal_type cell_lid) const;
    local_ordinal_type opposite_or_periodic_neighbor_cell(
        local_ordinal_type face_lid,
        local_ordinal_type cell_lid) const;
    real_t face_area(local_ordinal_type face_lid) const;
    Vec3 face_centroid(local_ordinal_type face_lid) const;
    Vec3 face_normal(local_ordinal_type face_lid) const;
    Vec3 face_area_vector(local_ordinal_type face_lid) const;
    Vec3 face_normal_outward(local_ordinal_type face_lid,
                             local_ordinal_type cell_lid) const;
    Vec3 face_area_vector_outward(local_ordinal_type face_lid,
                                  local_ordinal_type cell_lid) const;
    real_t face_cell_center_distance(local_ordinal_type face_lid) const;
    Vec3 cell_center_vector(local_ordinal_type face_lid,
                            local_ordinal_type cell_lid) const;
    real_t cell_to_face_distance(local_ordinal_type face_lid,
                                 local_ordinal_type cell_lid) const;
    bool is_exterior_face(local_ordinal_type face_lid) const;
    bool is_interior_face(local_ordinal_type face_lid) const;
    int boundary_id(local_ordinal_type face_lid) const;
    bool is_boundary_face(local_ordinal_type face_lid) const;

    const std::string& boundary_batch_name(int batch_id) const
    {
        const auto iter = d_boundary_names.find(batch_id);
        if (iter == d_boundary_names.end())
        {
            throw std::out_of_range("Unknown boundary batch ID.");
        }
        return iter->second;
    }

    const BoundaryFaceBatch& boundary_face_batch(int batch_id) const
    {
        return d_boundary_batches.at(batch_id);
    }

    const std::unordered_map<int, BoundaryFaceBatch>&
    boundary_batches() const noexcept
    {
        return d_boundary_batches;
    }

    Teuchos::RCP<const map_type> owned_cell_map() const
    {
        return d_owned_cell_map;
    }

    Teuchos::RCP<const map_type> overlap_cell_map() const
    {
        return d_overlap_cell_map;
    }

    Teuchos::RCP<const map_type> owned_face_map() const
    {
        return d_owned_face_map;
    }

    Teuchos::RCP<const map_type> overlap_face_map() const
    {
        return d_overlap_face_map;
    }

    Teuchos::RCP<const map_type> boundary_face_map() const
    {
        return d_boundary_face_map;
    }

    static constexpr local_ordinal_type invalid_local_id() noexcept
    {
        return indexer_type::invalid_local_id();
    }

    template<class StoredField>
    void sync_periodic_boundaries(StoredField& field) const
    {
        field.sync_ghosts();
    }

    /**
     * @brief Export owned cells to a rank-specific VTU file.
     *
     * Legacy STK meshes delegate to their native exporter; other mesh families
     * are converted through VTUWriter.
     */
    void export_vtu(const std::string& filename) const;

private:
    template<class Pointer>
    static Pointer require_mesh(Pointer mesh)
    {
        if (!mesh)
        {
            throw std::invalid_argument(
                "MeshHandle requires a non-null mesh.");
        }
        return mesh;
    }

    static local_ordinal_type checked_local(size_t value)
    {
        if (value > static_cast<size_t>(
                std::numeric_limits<local_ordinal_type>::max()))
        {
            throw std::overflow_error(
                "MeshHandle local ordinal overflow.");
        }
        return static_cast<local_ordinal_type>(value);
    }

    static std::vector<global_ordinal_type> checked_global_ids(
        std::vector<size_t> ids)
    {
        std::vector<global_ordinal_type> result;
        result.reserve(ids.size());
        for (const auto id : ids)
        {
            if constexpr (
                std::numeric_limits<global_ordinal_type>::digits
                < std::numeric_limits<size_t>::digits)
            {
                if (id > static_cast<size_t>(
                        std::numeric_limits<global_ordinal_type>::max()))
                {
                    throw std::overflow_error(
                        "MeshHandle global ordinal overflow.");
                }
            }
            result.push_back(static_cast<global_ordinal_type>(id));
        }
        return result;
    }

    std::string local_output_filename(
        const std::string& filename) const;

    void add_geometry_cell_data(VTUWriter& writer) const;

    template<class MeshType>
    VTUWriter::VectorData collect_vtu_points(
        const MeshType& mesh) const;

    void write_vtu(
        const std::string& filename,
        VTUWriter::VectorData points,
        VTUWriter::Int64Data connectivity,
        VTUWriter::Int64Data offsets,
        VTUWriter::UInt8Data cell_types) const;

    template<class MeshType>
    void export_orthogonal_vtu(
        const MeshType& mesh,
        const std::string& filename) const;

    void export_semi_structured_vtu(
        const SemiStructured& mesh,
        const std::string& filename) const;

    void export_unstructured_vtu(
        const Unstructured& mesh,
        const std::string& filename) const;

    global_ordinal_type geometry_cell_lid(
        local_ordinal_type local_id) const
    {
        check_cell(local_id);
        if (std::holds_alternative<UnstructuredPtr>(d_mesh))
        {
            return static_cast<global_ordinal_type>(local_id);
        }
        return d_indexer.cell_global_id(local_id);
    }

    global_ordinal_type geometry_face_lid(
        local_ordinal_type local_id) const
    {
        check_face(local_id);
        if (std::holds_alternative<UnstructuredPtr>(d_mesh))
        {
            return static_cast<global_ordinal_type>(local_id);
        }
        return d_indexer.face_global_id(local_id);
    }

    template<class Function>
    decltype(auto) visit_geometry_cell(
        local_ordinal_type cell_lid,
        Function&& function) const;

    template<class Function>
    decltype(auto) visit_geometry_face(
        local_ordinal_type face_lid,
        Function&& function) const;

    local_ordinal_type adjacent_cell(local_ordinal_type face_lid,
                                     bool owner) const;

    /**
     * @brief Convert an orthogonal (i,j,k) cell ID to the field local ordinal.
     * @throws std::invalid_argument if the mesh does not use
     *         OrthogonalIndexer::CellID.
     */
    local_ordinal_type cell_local_id(
        Meshes::OrthogonalIndexer::CellID id) const
    {
        return geometry_to_local_cell(
            visit_indexed_cell(id));
    }

    /**
     * @brief Convert an orthogonal (i,j,k,orientation) face ID to the field
     *        local ordinal.
     * @throws std::invalid_argument if the mesh does not use
     *         OrthogonalIndexer::FaceID.
     */
    local_ordinal_type face_local_id(
        Meshes::OrthogonalIndexer::FaceID id) const
    {
        return geometry_to_local_face(
            visit_indexed_face(id));
    }

    /**
     * @brief Convert a semi-structured (ij,k) cell ID to the field local
     *        ordinal.
     * @throws std::invalid_argument if the mesh does not use
     *         SemiStructuredIndexer::CellID.
     */
    local_ordinal_type cell_local_id(
        Meshes::SemiStructuredIndexer::CellID id) const
    {
        return geometry_to_local_cell(
            visit_indexed_cell(id));
    }

    /**
     * @brief Convert a semi-structured (ij,k,orientation) face ID to the
     *        field local ordinal.
     * @throws std::invalid_argument if the mesh does not use
     *         SemiStructuredIndexer::FaceID.
     */
    local_ordinal_type face_local_id(
        Meshes::SemiStructuredIndexer::FaceID id) const
    {
        return geometry_to_local_face(
            visit_indexed_face(id));
    }

    template<class MeshType>
    void initialize_orthogonal(SP<const MeshType> mesh,
                               DistributionOptions options);

    void initialize_semi_structured(SemiStructuredPtr mesh);

    void initialize_unstructured(UnstructuredPtr mesh);

    void initialize_unstructured(
        UnstructuredPtr mesh,
        const unstructured_indexer_type& indexer);

    void initialize_stk(STKAdapterPtr adapter);

    template<class MeshType>
    void initialize_serial(const MeshType& mesh);

    void initialize_cells(std::vector<size_t> owned,
                          std::vector<size_t> ghost);

    void initialize_faces(std::vector<size_t> owned,
                          std::vector<size_t> overlap);

    void initialize_indexer(indexer_type indexer);

    void initialize_cell_faces();

    template<class MeshType>
    void initialize_boundary_batches(const MeshType& mesh);

    template<class CommPtr, class Range>
    Teuchos::RCP<const map_type> make_map(
        const CommPtr& comm,
        const Range& ids) const
    {
        std::vector<global_ordinal_type> gids;
        gids.reserve(ids.size());
        for (const auto id : ids)
        {
            gids.push_back(static_cast<global_ordinal_type>(id));
        }
        const auto invalid_size =
            Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
        return Teuchos::rcp(new map_type(
            invalid_size,
            gids.data(),
            checked_local(gids.size()),
            global_ordinal_type{},
            comm));
    }

    template<class CommPtr>
    void create_maps(const CommPtr& comm)
    {
        std::vector<global_ordinal_type> boundary_faces;
        for (const auto& [batch_id, batch] : d_boundary_batches)
        {
            (void)batch_id;
            for (const auto face_lid : batch.face_lids)
            {
                if (is_owned_face(face_lid))
                {
                    boundary_faces.push_back(
                        d_indexer.face_global_id(face_lid));
                }
            }
        }

        d_owned_cell_map = make_map(
            comm, d_indexer.owned_cell_global_ids());
        d_overlap_cell_map = make_map(
            comm, d_indexer.cell_global_ids());
        d_owned_face_map = make_map(
            comm, d_indexer.owned_face_global_ids());
        d_overlap_face_map = make_map(
            comm, d_indexer.face_global_ids());
        d_boundary_face_map = make_map(comm, boundary_faces);
    }

    void check_cell(local_ordinal_type cell_lid) const
    {
        CHECK_BOUNDS(cell_lid, 0, num_local_cells());
    }

    void check_face(local_ordinal_type face_lid) const
    {
        CHECK_BOUNDS(face_lid, 0, num_faces());
    }

    /**
     * @brief Dispatch a structured cell ID to the concrete mesh's
     *        cell_local_id() and return the geometry local ID.
     * @throws std::invalid_argument if the mesh type does not accept the ID.
     */
    template<class CellID>
    size_t visit_indexed_cell(CellID id) const
    {
        return visit(
            [&](const auto& mesh) -> size_t
            {
                using mesh_type = std::decay_t<decltype(mesh)>;
                if constexpr (detail::mesh_has_cell_local_id<
                                  mesh_type, CellID>)
                {
                    return static_cast<size_t>(
                        mesh.cell_local_id(id));
                }
                throw std::invalid_argument(
                    "MeshHandle received a structured cell ID "
                    "but the active mesh type does not support it.");
            });
    }

    /**
     * @brief Dispatch a structured face ID to the concrete mesh's
     *        face_local_id() and return the geometry local ID.
     * @throws std::invalid_argument if the mesh type does not accept the ID.
     */
    template<class FaceID>
    size_t visit_indexed_face(FaceID id) const
    {
        return visit(
            [&](const auto& mesh) -> size_t
            {
                using mesh_type = std::decay_t<decltype(mesh)>;
                if constexpr (detail::mesh_has_face_local_id<
                                  mesh_type, FaceID>)
                {
                    return static_cast<size_t>(
                        mesh.face_local_id(id));
                }
                throw std::invalid_argument(
                    "MeshHandle received a structured face ID "
                    "but the active mesh type does not support it.");
            });
    }

    /**
     * @brief Look up the field local ordinal for a geometry cell index.
     * @returns The local ordinal, or invalid_local_id() if not locally
     *          available.
     */
    local_ordinal_type geometry_to_local_cell(
        size_t geometry_lid) const noexcept
    {
        if (std::holds_alternative<UnstructuredPtr>(d_mesh))
        {
            if (geometry_lid >= d_indexer.num_local_cells()
                || geometry_lid > static_cast<size_t>(
                    std::numeric_limits<local_ordinal_type>::max()))
            {
                return invalid_local_id();
            }
            return static_cast<local_ordinal_type>(geometry_lid);
        }
        return d_indexer.cell_local_id(
            static_cast<global_ordinal_type>(geometry_lid));
    }

    /**
     * @brief Look up the field local ordinal for a geometry face index.
     * @returns The local ordinal, or invalid_local_id() if not locally
     *          available.
     */
    local_ordinal_type geometry_to_local_face(
        size_t geometry_lid) const noexcept
    {
        if (std::holds_alternative<UnstructuredPtr>(d_mesh))
        {
            if (geometry_lid >= d_indexer.num_local_faces()
                || geometry_lid > static_cast<size_t>(
                    std::numeric_limits<local_ordinal_type>::max()))
            {
                return invalid_local_id();
            }
            return static_cast<local_ordinal_type>(geometry_lid);
        }
        return d_indexer.face_local_id(
            static_cast<global_ordinal_type>(geometry_lid));
    }

    variant_type d_mesh;
    indexer_type d_indexer;
    std::vector<size_t> d_cell_face_offsets;
    std::vector<local_ordinal_type> d_cell_face_lids;
    std::unordered_map<int, std::string> d_boundary_names;
    std::unordered_map<int, BoundaryFaceBatch> d_boundary_batches;
    Teuchos::RCP<const map_type> d_owned_cell_map;
    Teuchos::RCP<const map_type> d_overlap_cell_map;
    Teuchos::RCP<const map_type> d_owned_face_map;
    Teuchos::RCP<const map_type> d_overlap_face_map;
    Teuchos::RCP<const map_type> d_boundary_face_map;
};

} // namespace SimpleFluid

//----------------------------- inline functions ---------------------------------//
#include "geometry/MeshHandle.ipp"
