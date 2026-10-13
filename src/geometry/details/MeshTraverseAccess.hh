/** @file MeshTraverseAccess.hh
 * @brief Typed geometry access behind mesh traverse batches.
 */
#pragma once

#include "geometry/MeshHandle.hh"
#include "geometry/details/StructuredTraverseLayout.hh"
#include "geometry/details/LayeredTraverseLayout.hh"
#include <memory>

#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace SimpleFluid::detail
{

template<class Mesh> inline constexpr bool traverse_is_handle = false;
template<TpetraTypePack Pack>
inline constexpr bool traverse_is_handle<MeshHandle<Pack>> = true;

template<class Mesh> inline constexpr bool traverse_is_stk_adapter = false;
template<TpetraTypePack Pack>
inline constexpr bool traverse_is_stk_adapter<Meshes::STKMeshAdapter<Pack>> = true;

/**
 * @brief Borrow a concrete provider while preserving public field ordinals.
 *
 * Runtime MeshHandle dispatch occurs before construction. Composite providers
 * retain their canonical, transformed query path, including periodic images
 * and coarse/fine logical incidence. No geometry snapshot outlives execution.
 * All input ordinals refer to the public mesh's owned/overlap ordering.
 */
template<class PublicMesh, class GeometryMesh>
class TraverseAccess
{
public:
    using mesh_type = PublicMesh;
    using local_ordinal_type = typename PublicMesh::local_ordinal_type;
    using scalar_type = typename PublicMesh::scalar_type;
    using geometry_type = GeometryMesh;
    using Vec3 = typename GeometryMesh::Vec3;

    TraverseAccess(const PublicMesh& mesh, const GeometryMesh& geometry)
        : d_mesh(mesh), d_geometry(geometry),
          d_num_cells(mesh.num_local_cells()), d_num_faces(mesh.num_faces()) {}

    static constexpr local_ordinal_type invalid_local_id() noexcept
    {
        if constexpr (requires { PublicMesh::invalid_local_id(); })
            return PublicMesh::invalid_local_id();
        else if constexpr (requires { PublicMesh::invalid_local_id; })
            return PublicMesh::invalid_local_id;
        else
            return invalid_id<local_ordinal_type>();
    }

    size_t num_cells() const noexcept { return d_num_cells; }
    size_t num_faces() const noexcept { return d_num_faces; }
    const mesh_type& mesh() const noexcept { return d_mesh; }

    /** Canonical orthogonal ordinals admit an allocation-free range plan. */
    std::optional<StructuredTraverseLayout> structured_layout() const
    {
        if constexpr (has_structured_traverse_layout<GeometryMesh>)
        {
            if constexpr (traverse_is_handle<PublicMesh>)
            {
                // Reordering and partition maps need the generic field-local
                // schedules even when their geometry provider is structured.
                if (!d_mesh.d_serial_identity || !d_mesh.d_cell_geometry_lids.empty())
                    return std::nullopt;
            }
            else if constexpr (!std::is_same_v<PublicMesh, GeometryMesh>)
                return std::nullopt;
            const auto& indexer = d_geometry.indexer();
            if (indexer.num_cells_per_dim[0] && indexer.num_cells_per_dim[1] && indexer.num_cells_per_dim[2]
                && d_num_cells == indexer.total_cells() && d_num_faces == indexer.total_faces()
                && d_mesh.num_owned_cells() == d_num_cells
                && d_mesh.num_owned_faces() == d_num_faces)
                return StructuredTraverseLayout(indexer);
        }
        return std::nullopt;
    }

    std::shared_ptr<const LayeredTraverseLayout> layered_layout() const
    {
        if constexpr (has_layered_traverse_layout<GeometryMesh>)
        {
            if constexpr (traverse_is_handle<PublicMesh>)
            {
                if (!d_mesh.d_serial_identity || !d_mesh.d_cell_geometry_lids.empty()) return {};
            }
            else if constexpr (!std::is_same_v<PublicMesh, GeometryMesh>) return {};
            const auto& indexer = d_geometry.indexer();
            if (indexer.num_cells_per_layer && indexer.num_layers
                && d_num_cells == indexer.total_cells() && d_num_faces == indexer.total_faces()
                && d_mesh.num_owned_cells() == d_num_cells && d_mesh.num_owned_faces() == d_num_faces)
                return std::make_shared<const LayeredTraverseLayout>(d_geometry.topology());
        }
        return {};
    }

    bool is_owned_cell(size_t cell) const
    {
        if constexpr (traverse_is_handle<PublicMesh>
                      || requires { typename PublicMesh::partitioned_mesh_base_tag; })
            return d_mesh.is_owned_cell(local(cell));
        else
            return d_geometry.is_owned_cell(cell_id(cell));
    }

    bool is_owned_face(size_t face) const
    {
        if constexpr (traverse_is_handle<PublicMesh>
                      || requires { typename PublicMesh::partitioned_mesh_base_tag; })
            return d_mesh.is_owned_face(local(face));
        else
            return d_geometry.is_owned_face(face_id(face));
    }

    real_t cell_volume(size_t cell) const
    { return d_geometry.cell_volume(cell_id(cell)); }
    Vec3 cell_centroid(size_t cell) const
    { return d_geometry.cell_centroid(cell_id(cell)); }
    real_t face_area(size_t face) const
    { return d_geometry.face_area(face_id(face)); }
    Vec3 face_centroid(size_t face) const
    { return d_geometry.face_centroid(face_id(face)); }
    Vec3 face_normal(size_t face) const
    { return d_geometry.face_normal(face_id(face)); }
    Vec3 face_area_vector(size_t face) const
    {
        if constexpr (std::is_same_v<PublicMesh, GeometryMesh>
                      && requires { d_geometry.face_area_vector(face_id(face)); })
            return d_geometry.face_area_vector(face_id(face));
        else
            return face_normal(face) * face_area(face);
    }
    int boundary_id(size_t face) const
    { return d_geometry.boundary_id(face_id(face)); }

    local_ordinal_type owner_cell(size_t face) const
    { return field_cell(d_geometry.owner_cell(face_id(face))); }
    local_ordinal_type neighbor_cell(size_t face) const
    { return field_cell(d_geometry.neighbor_cell(face_id(face))); }

    /** @brief Physical exterior classification, independent of overlap depth. */
    bool is_exterior_face(size_t face) const
    { return invalid_cell(d_geometry.neighbor_cell(face_id(face))); }

    /** @brief Classify through complete provider incidence, including hidden faces. */
    bool is_exterior_cell(size_t cell) const
    {
        for (const auto face : d_geometry.faces(cell_id(cell)))
            if (invalid_cell(d_geometry.neighbor_cell(face))) return true;
        return false;
    }

    /** @brief Visit visible field face ordinals in native incidence order. */
    template<class Visitor> void face_ids(size_t cell, Visitor&& visitor) const
    {
        for (const auto face : d_geometry.faces(cell_id(cell)))
        {
            const auto local_face = field_face(face);
            if (local_face != invalid_local_id()) visitor(local_face);
        }
    }

    Vec3 face_center_vector(size_t face, size_t cell) const
    {
        if constexpr (requires { d_geometry.face_center_vector(face_id(face), cell_id(cell)); })
            return d_geometry.face_center_vector(face_id(face), cell_id(cell));
        else
            return face_centroid(face) - cell_centroid(cell);
    }

    Vec3 cell_center_vector(size_t face, size_t cell) const
    {
        if constexpr (requires { typename PublicMesh::partitioned_mesh_base_tag; })
            return d_mesh.cell_center_vector(local(face), local(cell));
        else
            return d_geometry.cell_center_vector(face_id(face), cell_id(cell));
    }

    real_t cell_to_face_distance(size_t face, size_t cell) const
    {
        if constexpr (traverse_is_handle<PublicMesh>)
        {
            // Match MeshHandle's public distance arithmetic, including the
            // wrapped legacy and Cartesian periodic-face distances.
            if constexpr (traverse_is_stk_adapter<GeometryMesh>)
                return d_geometry.mesh().cell_to_face_distance(face_id(face), cell_id(cell));
            else if constexpr (std::is_same_v<GeometryMesh, Meshes::OrthogonalCartesian3D>)
                return d_geometry.cell_to_face_distance(face_id(face), cell_id(cell));
            else
                return face_center_vector(face, cell).norm();
        }
        else if constexpr (requires { typename PublicMesh::partitioned_mesh_base_tag; })
            return (face_centroid(face) - cell_centroid(cell)).norm();
        else
            return d_geometry.cell_to_face_distance(face_id(face), cell_id(cell));
    }

    real_t face_cell_center_distance(size_t face) const
    {
        if constexpr (std::is_same_v<PublicMesh, GeometryMesh>
                      && requires { d_geometry.face_cell_center_distance(face_id(face)); })
            return d_geometry.face_cell_center_distance(face_id(face));
        const auto owner = owner_cell(face);
        const auto neighbor = neighbor_cell(face);
        return neighbor == invalid_local_id() ? 0.0
            : cell_center_vector(face, static_cast<size_t>(owner)).norm();
    }

    Vec3 face_normal_outward(size_t face, size_t cell) const
    {
        if constexpr (std::is_same_v<PublicMesh, GeometryMesh>
                      && requires { d_geometry.face_normal_outward(face_id(face), cell_id(cell)); })
            return d_geometry.face_normal_outward(face_id(face), cell_id(cell));
        const auto id = local(cell);
        if (id == owner_cell(face)) return face_normal(face);
        if (id == neighbor_cell(face)) return face_normal(face) * -1.0;
        throw std::invalid_argument("Cell is not adjacent to requested traverse face.");
    }

    Vec3 face_area_vector_outward(size_t face, size_t cell) const
    {
        if constexpr (std::is_same_v<PublicMesh, GeometryMesh>
                      && requires { d_geometry.face_area_vector_outward(face_id(face), cell_id(cell)); })
            return d_geometry.face_area_vector_outward(face_id(face), cell_id(cell));
        else
            return face_normal_outward(face, cell) * face_area(face);
    }

protected:
    const GeometryMesh& geometry_provider() const noexcept { return d_geometry; }

private:
    static local_ordinal_type local(size_t value)
    {
        if (value > static_cast<size_t>(std::numeric_limits<local_ordinal_type>::max()))
            throw std::overflow_error("Mesh traverse local ordinal is not representable.");
        return static_cast<local_ordinal_type>(value);
    }

    size_t geometry_cell(size_t cell) const
    {
        if (cell >= d_num_cells)
            throw std::out_of_range("Mesh traverse cell ordinal is out of range.");
        if constexpr (traverse_is_handle<PublicMesh>)
        {
            if (!d_mesh.d_cell_geometry_lids.empty())
                return static_cast<size_t>(d_mesh.d_cell_geometry_lids[cell]);
            if constexpr (std::is_same_v<GeometryMesh, Meshes::UnstructuredMesh>
                          || traverse_is_stk_adapter<GeometryMesh>)
                return cell;
            if (d_mesh.d_serial_identity) return cell;
            return static_cast<size_t>(d_mesh.d_map_indexing
                ? d_mesh.d_overlap_cell_map->getGlobalElement(local(cell))
                : d_mesh.d_indexer.cell_global_id(local(cell)));
        }
        else
            return cell;
    }

    size_t geometry_face(size_t face) const
    {
        if (face >= d_num_faces)
            throw std::out_of_range("Mesh traverse face ordinal is out of range.");
        if constexpr (traverse_is_handle<PublicMesh>)
        {
            if constexpr (traverse_is_stk_adapter<GeometryMesh>)
                return d_mesh.d_legacy_face_geometry_lids.empty() ? face
                    : static_cast<size_t>(d_mesh.d_legacy_face_geometry_lids[face]);
            if constexpr (std::is_same_v<GeometryMesh, Meshes::UnstructuredMesh>)
                return face;
            if (d_mesh.d_serial_identity) return face;
            return static_cast<size_t>(d_mesh.d_map_indexing
                ? d_mesh.d_overlap_face_map->getGlobalElement(local(face))
                : d_mesh.d_indexer.face_global_id(local(face)));
        }
        else
            return face;
    }

    auto cell_id(size_t cell) const
    {
        if constexpr (requires { d_geometry.cell_id(geometry_cell(cell)); })
            return d_geometry.cell_id(geometry_cell(cell));
        else
            return local(cell);
    }

    auto face_id(size_t face) const
    {
        if constexpr (requires { d_geometry.face_id(geometry_face(face)); })
            return d_geometry.face_id(geometry_face(face));
        else
            return local(face);
    }

    template<class Cell> static bool invalid_cell(Cell cell)
    {
        if constexpr (requires { GeometryMesh::invalid_cell_id(); })
            return cell == GeometryMesh::invalid_cell_id();
        else
            return cell == invalid_id<Cell>();
    }

    template<class Cell> local_ordinal_type field_cell(Cell cell) const
    {
        if (invalid_cell(cell)) return invalid_local_id();
        const auto ordinal = [&]
        {
            if constexpr (requires { d_geometry.cell_local_id(cell); })
                return static_cast<size_t>(d_geometry.cell_local_id(cell));
            else
                return static_cast<size_t>(cell);
        }();
        if constexpr (traverse_is_handle<PublicMesh>)
        {
            if (!d_mesh.d_cell_local_lids_by_geometry.empty())
                return ordinal < d_mesh.d_cell_local_lids_by_geometry.size()
                    ? d_mesh.d_cell_local_lids_by_geometry[ordinal] : invalid_local_id();
            if constexpr (std::is_same_v<GeometryMesh, Meshes::UnstructuredMesh>
                          || traverse_is_stk_adapter<GeometryMesh>)
                return ordinal < num_cells() ? local(ordinal) : invalid_local_id();
            if (d_mesh.d_serial_identity)
                return ordinal < num_cells() ? local(ordinal) : invalid_local_id();
            using global_ordinal_type = typename PublicMesh::global_ordinal_type;
            return d_mesh.d_map_indexing
                ? d_mesh.d_overlap_cell_map->getLocalElement(static_cast<global_ordinal_type>(ordinal))
                : d_mesh.d_indexer.cell_local_id(static_cast<global_ordinal_type>(ordinal));
        }
        else
            return ordinal < num_cells() ? local(ordinal) : invalid_local_id();
    }

    template<class Face> local_ordinal_type field_face(Face face) const
    {
        const auto ordinal = [&]
        {
            if constexpr (requires { d_geometry.face_local_id(face); })
                return static_cast<size_t>(d_geometry.face_local_id(face));
            else
                return static_cast<size_t>(face);
        }();
        if constexpr (traverse_is_handle<PublicMesh>)
        {
            if constexpr (traverse_is_stk_adapter<GeometryMesh>)
            {
                if (ordinal >= num_faces()) return invalid_local_id();
                return d_mesh.d_legacy_face_local_lids.empty() ? local(ordinal)
                    : d_mesh.d_legacy_face_local_lids[ordinal];
            }
            if constexpr (std::is_same_v<GeometryMesh, Meshes::UnstructuredMesh>)
                return ordinal < num_faces() ? local(ordinal) : invalid_local_id();
            if (d_mesh.d_serial_identity)
                return ordinal < num_faces() ? local(ordinal) : invalid_local_id();
            using global_ordinal_type = typename PublicMesh::global_ordinal_type;
            return d_mesh.d_map_indexing
                ? d_mesh.d_overlap_face_map->getLocalElement(static_cast<global_ordinal_type>(ordinal))
                : d_mesh.d_indexer.face_local_id(static_cast<global_ordinal_type>(ordinal));
        }
        else
            return ordinal < num_faces() ? local(ordinal) : invalid_local_id();
    }

    const PublicMesh& d_mesh;
    const GeometryMesh& d_geometry;
    size_t d_num_cells, d_num_faces;
};

/**
 * Preserve the selected plan layout in nested numerical callback types.
 * This keeps the generic schedule path out of structured incidence kernels.
 * Construction requires the plan's full, canonical structured-layout proof.
 * The enclosing execution pins the provider and its fixed topology; checked
 * public ordinals therefore map directly through its orthogonal indexer.
 */
template<class Access>
class StructuredTraverseAccess : public Access
{
public:
    using mesh_type = typename Access::mesh_type;
    using geometry_type = typename Access::geometry_type;
    using local_ordinal_type = typename Access::local_ordinal_type;
    using Vec3 = typename Access::Vec3;

    static_assert(has_structured_traverse_layout<geometry_type>);
    static_assert(traverse_is_handle<mesh_type> || std::is_same_v<mesh_type, geometry_type>);

    explicit StructuredTraverseAccess(const Access& access) : Access(access) {}

    local_ordinal_type owner_cell(size_t face) const
    {
        const auto& geometry = this->geometry_provider();
        return canonical_field_cell(geometry.topology().owner_cell(canonical_face_id(face)));
    }

    local_ordinal_type neighbor_cell(size_t face) const
    {
        const auto& geometry = this->geometry_provider();
        return canonical_field_cell(geometry.topology().neighbor_cell(canonical_face_id(face)));
    }

    Vec3 cell_center_vector(size_t face, size_t cell) const
    {
        // Keep provider arithmetic, adjacency checks, and Cartesian periodic
        // image displacements exactly as in the ordinary geometry query.
        return this->geometry_provider().cell_center_vector(
            canonical_face_id(face), canonical_cell_id(cell));
    }

private:
    typename geometry_type::cell_id_t canonical_cell_id(size_t cell) const
    {
        if (cell >= this->num_cells())
            throw std::out_of_range("Mesh traverse cell ordinal is out of range.");
        return this->geometry_provider().indexer().cell_id(cell);
    }

    typename geometry_type::face_id_t canonical_face_id(size_t face) const
    {
        if (face >= this->num_faces())
            throw std::out_of_range("Mesh traverse face ordinal is out of range.");
        return this->geometry_provider().indexer().face_id(face);
    }

    local_ordinal_type canonical_field_cell(typename geometry_type::cell_id_t cell) const
    {
        if (cell == geometry_type::invalid_cell_id()) return Access::invalid_local_id();
        // A valid face's topology query returns a valid canonical cell or the
        // sentinel above. Plan construction already checked ordinal capacity,
        // so neither field-map lookup nor another geometry bounds check is
        // needed to convert this topology-produced identifier.
        return static_cast<local_ordinal_type>(this->geometry_provider().indexer().cell_ordinal(cell));
    }
};

template<class Access> inline constexpr bool is_structured_traverse_access = false;
template<class Access>
inline constexpr bool is_structured_traverse_access<StructuredTraverseAccess<Access>> = true;

/** Distinct tag: polygonal prisms never enter the orthogonal six-face kernel. */
template<class Access>
class LayeredTraverseAccess : public Access
{
public:
    using mesh_type = typename Access::mesh_type;
    using geometry_type = typename Access::geometry_type;
    using local_ordinal_type = typename Access::local_ordinal_type;
    using Vec3 = typename Access::Vec3;

    static_assert(has_layered_traverse_layout<geometry_type>);
    static_assert(traverse_is_handle<mesh_type> || std::is_same_v<mesh_type, geometry_type>);

    explicit LayeredTraverseAccess(const Access& access) : Access(access) {}

    local_ordinal_type owner_cell(size_t face) const
    {
        const auto& geometry = this->geometry_provider();
        return canonical_field_cell(geometry.topology().owner_cell(canonical_face_id(face)));
    }

    local_ordinal_type neighbor_cell(size_t face) const
    {
        const auto& geometry = this->geometry_provider();
        return canonical_field_cell(geometry.topology().neighbor_cell(canonical_face_id(face)));
    }

    Vec3 cell_center_vector(size_t face, size_t cell) const
    {
        // Keep provider arithmetic, adjacency checks, and layered periodic
        // image displacements exactly as in the ordinary geometry query.
        return this->geometry_provider().cell_center_vector(
            canonical_face_id(face), canonical_cell_id(cell));
    }

    Vec3 resolved_cell_center_vector(typename geometry_type::cell_id_t cell,
        typename geometry_type::cell_id_t other) const
    {
        return this->geometry_provider().resolved_cell_center_vector(cell, other);
    }

    auto resolve_incidence(const LayeredTraverseLayout::Incidences& faces, size_t slot) const
    {
        // Borrow topology through the active geometry access, never the retained layout.
        return faces.resolved(slot, this->geometry_provider().topology());
    }

private:
    typename geometry_type::cell_id_t canonical_cell_id(size_t cell) const
    {
        if (cell >= this->num_cells())
            throw std::out_of_range("Mesh traverse cell ordinal is out of range.");
        return this->geometry_provider().indexer().cell_id(cell);
    }

    typename geometry_type::face_id_t canonical_face_id(size_t face) const
    {
        if (face >= this->num_faces())
            throw std::out_of_range("Mesh traverse face ordinal is out of range.");
        return this->geometry_provider().indexer().face_id(face);
    }

    local_ordinal_type canonical_field_cell(typename geometry_type::cell_id_t cell) const
    {
        if (cell == geometry_type::invalid_cell_id()) return Access::invalid_local_id();
        // A valid face's topology query returns a valid canonical cell or the
        // sentinel above. Plan construction already checked ordinal capacity,
        // so neither field-map lookup nor another geometry bounds check is
        // needed to convert this topology-produced identifier.
        return static_cast<local_ordinal_type>(this->geometry_provider().indexer().cell_ordinal(cell));
    }
};

template<class Access> inline constexpr bool is_layered_traverse_access = false;
template<class Access> inline constexpr bool is_layered_traverse_access<LayeredTraverseAccess<Access>> = true;

/** @brief Dispatch a runtime mesh once, outside all batch numerical loops. */
template<class Mesh, class Visitor>
decltype(auto) with_traverse_access(const Mesh& mesh, Visitor&& visitor)
{
    const auto invoke = [&](const auto& geometry) -> decltype(auto)
    {
        // Native value meshes also reject supported mutation during callbacks.
        // Composite execution additionally validates and pins each provider.
        [[maybe_unused]] const auto execution = [&]
        {
            if constexpr (requires { geometry.acquire_execution_view(); })
                return geometry.acquire_execution_view();
            else if constexpr (requires { geometry.acquire_geometry_read(); })
                return geometry.acquire_geometry_read();
            else
            {
                struct LegacyExecution {};
                return LegacyExecution{};
            }
        }();
        const TraverseAccess<Mesh, std::remove_cvref_t<decltype(geometry)>> access(mesh, geometry);
        return std::forward<Visitor>(visitor)(access);
    };
    if constexpr (traverse_is_handle<Mesh>)
        return mesh.visit(invoke);
    else if constexpr (requires { typename Mesh::partitioned_mesh_base_tag; })
        return invoke(mesh.mesh());
    else
        return invoke(mesh);
}

} // namespace SimpleFluid::detail
