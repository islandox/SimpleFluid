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

#include "SimpleFluidExport.hh"
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
#include <concepts>
#include <cstdint>
#include <limits>
#include <memory>
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

template<TpetraTypePack Pack>
class MeshReorderingFactory;

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
 * MeshHandle normalizes structured, semi-structured, unstructured, and legacy
 * STK meshes into one FVM-facing API. Cartesian and cylindrical meshes are
 * distributed automatically. Unstructured meshes require explicit
 * partitioning before multi-rank construction, while SemiStructuredXY_Z is
 * currently serial-only and rejects construction on a multi-rank communicator.
 * The handle builds owned/overlap maps and translates compact local IDs to each
 * concrete mesh's geometry IDs. Construction from a mutable mesh retains that
 * exact object for controlled mutable visitation while the established
 * variant() and visit() observer surface remains deeply const.
 *
 * @tparam Pack Tpetra scalar, ordinal, communicator, and map types.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class SIMPLEFLUID_PUBLIC_TYPE MeshHandle
{
    friend class MeshReorderingFactory<Pack>;

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
    using MutableCartesianPtr = SP<Cartesian>;
    using MutableCylindricalPtr = SP<Cylindrical>;
    using MutableSemiStructuredPtr = SP<SemiStructured>;
    using MutableUnstructuredPtr = SP<Unstructured>;
    using MutableSTKAdapterPtr = SP<STKAdapter>;
    using variant_type = std::variant<CartesianPtr,
                                      CylindricalPtr,
                                      SemiStructuredPtr,
                                      UnstructuredPtr,
                                      STKAdapterPtr>;
    using mutable_variant_type = std::variant<MutableCartesianPtr,
                                              MutableCylindricalPtr,
                                              MutableSemiStructuredPtr,
                                              MutableUnstructuredPtr,
                                              MutableSTKAdapterPtr>;

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

    /** @brief Retain mutable ownership of a distributed Cartesian mesh. */
    explicit MeshHandle(MutableCartesianPtr mesh,
                        DistributionOptions options = {});

    /** @brief Retain mutable ownership supplied through a Cartesian subtype. */
    template<class Derived>
        requires (!std::is_const_v<Derived>
                  && !std::same_as<Derived, Cartesian>
                  && std::derived_from<Derived, Cartesian>)
    explicit MeshHandle(
        SP<Derived> mesh,
        DistributionOptions options = {})
        : MeshHandle(
              std::static_pointer_cast<Cartesian>(std::move(mesh)),
              options)
    {
    }

    /** @brief Build a distributed handle for a cylindrical mesh. */
    explicit MeshHandle(CylindricalPtr mesh,
                        DistributionOptions options = {});

    /** @brief Retain mutable ownership of a distributed cylindrical mesh. */
    explicit MeshHandle(MutableCylindricalPtr mesh,
                        DistributionOptions options = {});

    /** @brief Retain mutable ownership supplied through a cylindrical subtype. */
    template<class Derived>
        requires (!std::is_const_v<Derived>
                  && !std::same_as<Derived, Cylindrical>
                  && std::derived_from<Derived, Cylindrical>)
    explicit MeshHandle(
        SP<Derived> mesh,
        DistributionOptions options = {})
        : MeshHandle(
              std::static_pointer_cast<Cylindrical>(std::move(mesh)),
              options)
    {
    }

    /**
     * @brief Build a serial-only handle for a semi-structured mesh.
     * @throws std::runtime_error On a communicator containing multiple ranks.
     */
    explicit MeshHandle(SemiStructuredPtr mesh);

    /** @brief Retain mutable ownership of a serial semi-structured mesh. */
    explicit MeshHandle(MutableSemiStructuredPtr mesh);

    /** @brief Retain mutable ownership supplied through a semi-structured subtype. */
    template<class Derived>
        requires (!std::is_const_v<Derived>
                  && !std::same_as<Derived, SemiStructured>
                  && std::derived_from<Derived, SemiStructured>)
    explicit MeshHandle(SP<Derived> mesh)
        : MeshHandle(std::static_pointer_cast<SemiStructured>(
              std::move(mesh)))
    {
    }

    /** @brief Build a serial unstructured mesh handle. */
    explicit MeshHandle(UnstructuredPtr mesh);

    /** @brief Retain mutable ownership of a serial unstructured mesh. */
    explicit MeshHandle(MutableUnstructuredPtr mesh);

    /** @brief Retain mutable ownership supplied through an unstructured subtype. */
    template<class Derived>
        requires (!std::is_const_v<Derived>
                  && !std::same_as<Derived, Unstructured>
                  && std::derived_from<Derived, Unstructured>)
    explicit MeshHandle(SP<Derived> mesh)
        : MeshHandle(std::static_pointer_cast<Unstructured>(
              std::move(mesh)))
    {
    }

    /** @brief Build a handle from a previously partitioned mesh. */
    MeshHandle(
        UnstructuredPtr mesh,
        const unstructured_indexer_type& indexer);

    /** @brief Retain mutable ownership of an indexed unstructured mesh. */
    MeshHandle(
        MutableUnstructuredPtr mesh,
        const unstructured_indexer_type& indexer);

    /**
     * @brief Adapt an existing unstructured partition without repartitioning.
     */
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
        if constexpr (!std::is_const_v<Partitioned>
                      && requires { partitioned->mutable_mesh_ptr(); })
        {
            if (auto mutable_mesh = partitioned->mutable_mesh_ptr())
            {
                const auto& read_only_mesh =
                    std::get<UnstructuredPtr>(d_mesh);
                const bool same_owner =
                    !mutable_mesh.owner_before(read_only_mesh)
                    && !read_only_mesh.owner_before(mutable_mesh);
                if (mutable_mesh.get() != read_only_mesh.get()
                    || !same_owner)
                {
                    throw std::invalid_argument(
                        "MeshHandle partition views must reference the same "
                        "geometry object.");
                }
                d_mutable_mesh.emplace(
                    MutableUnstructuredPtr(std::move(mutable_mesh)));
            }
        }
        initialize_unstructured(
            std::get<UnstructuredPtr>(d_mesh),
            partitioned->indexer(),
            partitioned->owned_cell_map()->getComm());
    }

    /** @brief Build a handle around a legacy STK adapter. */
    explicit MeshHandle(STKAdapterPtr mesh);

    /** @brief Retain mutable ownership of a legacy STK adapter. */
    explicit MeshHandle(MutableSTKAdapterPtr mesh);

    /** @brief Retain mutable ownership supplied through an STK-adapter subtype. */
    template<class Derived>
        requires (!std::is_const_v<Derived>
                  && !std::same_as<Derived, STKAdapter>
                  && std::derived_from<Derived, STKAdapter>)
    explicit MeshHandle(SP<Derived> mesh)
        : MeshHandle(std::static_pointer_cast<STKAdapter>(
              std::move(mesh)))
    {
    }

    /** @brief Adapt a legacy distributed mesh and build a runtime handle. */
    explicit MeshHandle(SP<const SimpleFluid::Mesh<Pack>> mesh);

    /** @brief Adapt and retain mutable ownership of a legacy distributed mesh. */
    explicit MeshHandle(SP<SimpleFluid::Mesh<Pack>> mesh);

    /** @brief Adapt mutable ownership supplied through a legacy-mesh subtype. */
    template<class Derived>
        requires (!std::is_const_v<Derived>
                  && !std::same_as<Derived, SimpleFluid::Mesh<Pack>>
                  && std::derived_from<Derived, SimpleFluid::Mesh<Pack>>)
    explicit MeshHandle(SP<Derived> mesh)
        : MeshHandle(std::static_pointer_cast<SimpleFluid::Mesh<Pack>>(
              std::move(mesh)))
    {
    }

    const variant_type& variant() const noexcept { return d_mesh; }

    /**
     * @brief Identity of the concrete geometry shared by this handle.
     *
     * Separately constructed alias handles around the same native mesh report
     * the same identity.  This lets fixed-topology motion and ALE consumers
     * reject a copied or otherwise unrelated geometry even when its topology
     * happens to be equivalent.
     */
    [[nodiscard]] const void* geometry_identity() const noexcept
    {
        return std::visit(
            [](const auto& mesh) noexcept -> const void*
            {
                return mesh.get();
            },
            d_mesh);
    }

    /**
     * @brief Return the local/global indexer.
     *
     * Legacy meshes use their existing ordinal-indexed storage directly.
     * Calling this compatibility accessor materializes the otherwise-unused
     * legacy lookup tables on demand.
     */
    const indexer_type& indexer() const
    {
        if (is_stk())
        {
            materialize_legacy_indexer();
        }
        return d_indexer;
    }

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

    /**
     * @brief Invoke a visitor with mutable access to the concrete geometry.
     *
     * Mutable access is available only when the handle was built from a
     * mutable mesh pointer. The const visit() overload remains a read-only
     * observer for every handle.
     *
     * @warning This low-level ownership seam does not publish a geometry epoch.
     *          Direct geometry changes through it after fields or geometry
     *          caches have been constructed are unsupported. Use a controlled
     *          mesh-motion transaction instead.
     *
     * @throws std::logic_error If the handle was built from a const mesh.
     */
    template<class Visitor>
    decltype(auto) visit_mutable(Visitor&& visitor)
    {
        if (!d_mutable_mesh)
        {
            throw std::logic_error(
                "MeshHandle does not retain mutable geometry ownership.");
        }
        return std::visit(
            [&](auto& mesh) -> decltype(auto)
            {
                return std::forward<Visitor>(visitor)(*mesh);
            },
            *d_mutable_mesh);
    }

    /** @brief True when visit_mutable() can access the concrete geometry. */
    bool has_mutable_geometry() const noexcept
    {
        return d_mutable_mesh.has_value();
    }

    /**
     * @brief Monotone revision of the fixed-topology geometry.
     *
     * Topology, global IDs, maps, and partitioning are not part of this
     * revision. Geometry-dependent caches may retain their graph structure,
     * but must refresh numeric data whenever this value changes.
     */
    std::uint64_t geometry_epoch() const noexcept
    {
        return std::visit(
            [](const auto& mesh) noexcept -> std::uint64_t
            {
                if constexpr (requires { mesh->geometry_epoch(); })
                {
                    return mesh->geometry_epoch();
                }
                else
                {
                    return 0;
                }
            },
            d_mesh);
    }

    bool is_stk() const noexcept
    {
        return std::holds_alternative<STKAdapterPtr>(d_mesh);
    }

    /** @brief True after a factory has changed this handle's local cell order. */
    bool has_reordered_cells() const noexcept
    {
        return d_cells_reordered;
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
        if (const auto legacy = legacy_mesh())
        {
            return legacy->num_owned_cells();
        }
        return d_indexer.num_owned_cells();
    }
    size_t num_local_cells() const noexcept
    {
        if (const auto legacy = legacy_mesh())
        {
            return legacy->num_local_cells();
        }
        return d_indexer.num_local_cells();
    }
    size_t num_cells() const noexcept { return num_local_cells(); }
    size_t num_owned_faces() const noexcept
    {
        if (const auto legacy = legacy_mesh())
        {
            const auto map = legacy->owned_face_map();
            return map.is_null() ? 0 : map->getLocalNumElements();
        }
        return d_indexer.num_owned_faces();
    }
    size_t num_faces() const noexcept
    {
        if (const auto legacy = legacy_mesh())
        {
            return legacy->num_faces();
        }
        return d_indexer.num_local_faces();
    }

    bool is_owned_cell(local_ordinal_type cell_lid) const
    {
        check_cell(cell_lid);
        if (const auto legacy = legacy_mesh())
        {
            return legacy->is_owned_cell(cell_lid);
        }
        return d_indexer.is_owned_cell(cell_lid);
    }

    bool is_owned_face(local_ordinal_type face_lid) const
    {
        check_face(face_lid);
        if (const auto legacy = legacy_mesh())
        {
            return legacy->is_owned_face(
                checked_local(static_cast<size_t>(
                    geometry_face_lid(face_lid))));
        }
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

    /**
     * @brief Return the partition-independent geometry identifier of a cell.
     *
     * Unlike cell_global_id(), this identifier is not the contiguous Tpetra
     * map ID and remains attached to the same geometric cell after
     * repartitioning.
     */
    global_ordinal_type cell_geometry_global_id(
        local_ordinal_type cell_lid) const
    {
        check_cell(cell_lid);
        if (const auto legacy = legacy_mesh())
        {
            return legacy->cell_global_id(
                checked_local(static_cast<size_t>(
                    geometry_cell_lid(cell_lid))));
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
    /** @brief True if the underlying geometry has no cell across a face. */
    bool is_geometry_exterior_face(local_ordinal_type face_lid) const;
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

    /**
     * @brief Build reusable VTU topology for the owned cells.
     *
     * The returned topology is independent of field values and can therefore
     * be cached by transient solvers across output steps.
     */
    VTUWriter::TopologyHandle vtu_topology() const;

private:
    template<class Pointer>
    SIMPLEFLUID_LOCAL static Pointer require_mesh(Pointer mesh)
    {
        if (!mesh)
        {
            throw std::invalid_argument(
                "MeshHandle requires a non-null mesh.");
        }
        return mesh;
    }

    SIMPLEFLUID_LOCAL static local_ordinal_type checked_local(size_t value)
    {
        if (value > static_cast<size_t>(
                std::numeric_limits<local_ordinal_type>::max()))
        {
            throw std::overflow_error(
                "MeshHandle local ordinal overflow.");
        }
        return static_cast<local_ordinal_type>(value);
    }

    SIMPLEFLUID_LOCAL static std::vector<global_ordinal_type> checked_global_ids(
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

    SIMPLEFLUID_LOCAL std::string local_output_filename(
        const std::string& filename) const;

    SIMPLEFLUID_LOCAL void add_geometry_cell_data(VTUWriter& writer) const;

    template<class MeshType>
    SIMPLEFLUID_LOCAL VTUWriter::VectorData collect_vtu_points(
        const MeshType& mesh) const;

    SIMPLEFLUID_LOCAL void write_vtu(
        const std::string& filename,
        VTUWriter::TopologyHandle topology) const;

    SIMPLEFLUID_LOCAL VTUWriter::TopologyHandle legacy_vtu_topology(
        const STKAdapter& mesh) const;

    template<class MeshType>
    SIMPLEFLUID_LOCAL VTUWriter::TopologyHandle orthogonal_vtu_topology(
        const MeshType& mesh) const;

    SIMPLEFLUID_LOCAL VTUWriter::TopologyHandle semi_structured_vtu_topology(
        const SemiStructured& mesh) const;

    SIMPLEFLUID_LOCAL VTUWriter::TopologyHandle unstructured_vtu_topology(
        const Unstructured& mesh) const;

    SIMPLEFLUID_LOCAL global_ordinal_type geometry_cell_lid(
        local_ordinal_type local_id) const
    {
        check_cell(local_id);
        if (!d_cell_geometry_lids.empty())
        {
            return d_cell_geometry_lids[static_cast<size_t>(local_id)];
        }
        if (std::holds_alternative<UnstructuredPtr>(d_mesh)
            || is_stk())
        {
            return static_cast<global_ordinal_type>(local_id);
        }
        return d_indexer.cell_global_id(local_id);
    }

    SIMPLEFLUID_LOCAL global_ordinal_type geometry_face_lid(
        local_ordinal_type local_id) const
    {
        check_face(local_id);
        if (is_stk())
        {
            const auto local = static_cast<size_t>(local_id);
            return d_legacy_face_geometry_lids.empty()
                ? static_cast<global_ordinal_type>(local)
                : static_cast<global_ordinal_type>(
                      d_legacy_face_geometry_lids[local]);
        }
        if (std::holds_alternative<UnstructuredPtr>(d_mesh))
        {
            return static_cast<global_ordinal_type>(local_id);
        }
        return d_indexer.face_global_id(local_id);
    }

    template<class Function>
    SIMPLEFLUID_LOCAL decltype(auto) visit_geometry_cell(
        local_ordinal_type cell_lid,
        Function&& function) const;

    template<class Function>
    SIMPLEFLUID_LOCAL decltype(auto) visit_geometry_face(
        local_ordinal_type face_lid,
        Function&& function) const;

    SIMPLEFLUID_LOCAL local_ordinal_type adjacent_cell(
        local_ordinal_type face_lid, bool owner) const;

    /**
     * @brief Convert an orthogonal (i,j,k) cell ID to the field local ordinal.
     * @throws std::invalid_argument if the mesh does not use
     *         OrthogonalIndexer::CellID.
     */
    SIMPLEFLUID_LOCAL local_ordinal_type cell_local_id(
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
    SIMPLEFLUID_LOCAL local_ordinal_type face_local_id(
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
    SIMPLEFLUID_LOCAL local_ordinal_type cell_local_id(
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
    SIMPLEFLUID_LOCAL local_ordinal_type face_local_id(
        Meshes::SemiStructuredIndexer::FaceID id) const
    {
        return geometry_to_local_face(
            visit_indexed_face(id));
    }

    template<class MeshType>
    SIMPLEFLUID_LOCAL void initialize_orthogonal(
        SP<const MeshType> mesh, DistributionOptions options);

    SIMPLEFLUID_LOCAL void initialize_semi_structured(SemiStructuredPtr mesh);

    SIMPLEFLUID_LOCAL void initialize_unstructured(UnstructuredPtr mesh);

    SIMPLEFLUID_LOCAL void initialize_unstructured(
        UnstructuredPtr mesh,
        const unstructured_indexer_type& indexer,
        Teuchos::RCP<const typename Pack::comm_type> comm);

    SIMPLEFLUID_LOCAL void initialize_stk(STKAdapterPtr adapter);

    template<class MeshType>
    SIMPLEFLUID_LOCAL void initialize_serial(const MeshType& mesh);

    SIMPLEFLUID_LOCAL void initialize_cells(
        std::vector<size_t> owned, std::vector<size_t> ghost);

    SIMPLEFLUID_LOCAL void initialize_faces(
        std::vector<size_t> owned, std::vector<size_t> overlap);

    SIMPLEFLUID_LOCAL void initialize_indexer(indexer_type indexer);

    SIMPLEFLUID_LOCAL void initialize_cell_faces();

    SIMPLEFLUID_LOCAL void materialize_legacy_indexer() const;

    template<class MeshType>
    SIMPLEFLUID_LOCAL void initialize_boundary_batches(const MeshType& mesh);

    template<class CommPtr, class Range>
    SIMPLEFLUID_LOCAL Teuchos::RCP<const map_type> make_map(
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
    SIMPLEFLUID_LOCAL void create_maps(const CommPtr& comm)
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

    SIMPLEFLUID_LOCAL void check_cell(local_ordinal_type cell_lid) const
    {
        CHECK_BOUNDS(cell_lid, 0, num_local_cells());
    }

    SIMPLEFLUID_LOCAL void check_face(local_ordinal_type face_lid) const
    {
        CHECK_BOUNDS(face_lid, 0, num_faces());
    }

    /**
     * @brief Dispatch a structured cell ID to the concrete mesh's
     *        cell_local_id() and return the geometry local ID.
     * @throws std::invalid_argument if the mesh type does not accept the ID.
     */
    template<class CellID>
    SIMPLEFLUID_LOCAL size_t visit_indexed_cell(CellID id) const
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
    SIMPLEFLUID_LOCAL size_t visit_indexed_face(FaceID id) const
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
    SIMPLEFLUID_LOCAL local_ordinal_type geometry_to_local_cell(
        size_t geometry_lid) const noexcept
    {
        if (!d_cell_local_lids_by_geometry.empty())
        {
            return geometry_lid < d_cell_local_lids_by_geometry.size()
                ? d_cell_local_lids_by_geometry[geometry_lid]
                : invalid_local_id();
        }
        if (std::holds_alternative<UnstructuredPtr>(d_mesh)
            || is_stk())
        {
            if (geometry_lid >= num_local_cells()
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
    SIMPLEFLUID_LOCAL local_ordinal_type geometry_to_local_face(
        size_t geometry_lid) const noexcept
    {
        if (is_stk())
        {
            if (geometry_lid >= num_faces()
                || geometry_lid > static_cast<size_t>(
                    std::numeric_limits<local_ordinal_type>::max()))
            {
                return invalid_local_id();
            }
            return d_legacy_face_local_lids.empty()
                ? static_cast<local_ordinal_type>(geometry_lid)
                : d_legacy_face_local_lids[geometry_lid];
        }
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
    std::optional<mutable_variant_type> d_mutable_mesh;
    mutable indexer_type d_indexer;
    mutable bool d_legacy_indexer_materialized = false;
    bool d_cells_reordered = false;
    std::vector<global_ordinal_type> d_cell_geometry_lids;
    std::vector<local_ordinal_type> d_cell_local_lids_by_geometry;
    std::vector<local_ordinal_type> d_legacy_face_geometry_lids;
    std::vector<local_ordinal_type> d_legacy_face_local_lids;
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
