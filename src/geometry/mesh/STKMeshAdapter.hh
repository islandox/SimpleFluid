/**
 * @file STKMeshAdapter.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief FVM-facing adapter for the legacy distributed STK mesh.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "geometry/Mesh.hh"

#include <concepts>
#include <cstddef>
#include <memory>
#include <utility>

namespace SimpleFluid::Meshes
{

/**
 * @brief Adapt the legacy distributed Mesh API to the mesh concept.
 *
 * The adapter shares ownership of the underlying mesh and preserves its local
 * ordinal IDs while presenting the API expected by MeshHandle.
 *
 * @tparam Pack Tpetra type pack used by the wrapped legacy mesh.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class STKMeshAdapter
{
public:
    using mesh_type = SimpleFluid::Mesh<Pack>;
    using cell_id_t = typename Pack::local_ordinal_type;
    using face_id_t = typename Pack::local_ordinal_type;
    using node_id_t = typename Pack::global_ordinal_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using scalar_type = typename Pack::scalar_type;
    using Vec3 = typename mesh_type::Vec3;

    /**
     * @brief Wrap an assembled legacy mesh.
     * @throws std::invalid_argument If @p mesh is null.
     */
    explicit STKMeshAdapter(SP<const mesh_type> mesh)
        : d_mesh(std::move(mesh))
    {
        if (!d_mesh)
        {
            throw std::invalid_argument(
                "STKMeshAdapter requires a non-null mesh.");
        }
    }

    /** @brief Wrap a mutable assembled legacy mesh without copying it. */
    explicit STKMeshAdapter(SP<mesh_type> mesh)
        : d_mesh(mesh),
          d_mutable_mesh(std::move(mesh))
    {
        if (!d_mesh)
        {
            throw std::invalid_argument(
                "STKMeshAdapter requires a non-null mesh.");
        }
    }

    /** @brief Preserve null-pointer validation without cv-overload ambiguity. */
    explicit STKMeshAdapter(std::nullptr_t)
        : STKMeshAdapter(SP<const mesh_type>{})
    {
    }

    /** @brief Wrap mutable ownership supplied through a legacy-mesh subtype. */
    template<class Derived>
        requires (!std::is_const_v<Derived>
                  && !std::same_as<Derived, mesh_type>
                  && std::derived_from<Derived, mesh_type>)
    explicit STKMeshAdapter(SP<Derived> mesh)
        : STKMeshAdapter(std::static_pointer_cast<mesh_type>(
              std::move(mesh)))
    {
    }

    const mesh_type& mesh() const noexcept { return *d_mesh; }
    SP<const mesh_type> mesh_ptr() const noexcept { return d_mesh; }
    bool has_mutable_mesh() const noexcept
    {
        return static_cast<bool>(d_mutable_mesh);
    }
    mesh_type& mutable_mesh()
    {
        if (!d_mutable_mesh)
        {
            throw std::logic_error(
                "STKMeshAdapter does not retain mutable mesh ownership.");
        }
        return *d_mutable_mesh;
    }
    SP<mesh_type> mutable_mesh_ptr() noexcept { return d_mutable_mesh; }

    size_t spatial_dimension() const noexcept
    {
        return d_mesh->spatial_dimension();
    }

    size_t num_local_cells() const noexcept
    {
        return d_mesh->num_local_cells();
    }

    size_t num_owned_cells() const noexcept
    {
        return d_mesh->num_owned_cells();
    }

    size_t num_cells() const noexcept { return num_local_cells(); }
    size_t num_faces() const noexcept { return d_mesh->num_faces(); }

    size_t num_owned_faces() const noexcept
    {
        const auto map = d_mesh->owned_face_map();
        return map.is_null() ? 0 : map->getLocalNumElements();
    }

    cell_id_t cell_id(size_t local_id) const
    {
        return static_cast<cell_id_t>(local_id);
    }

    face_id_t face_id(size_t local_id) const
    {
        return static_cast<face_id_t>(local_id);
    }

    size_t cell_local_id(cell_id_t id) const
    {
        return static_cast<size_t>(id);
    }

    size_t face_local_id(face_id_t id) const
    {
        return static_cast<size_t>(id);
    }

    bool is_owned_cell(cell_id_t id) const
    {
        return d_mesh->is_owned_cell(id);
    }

    bool is_owned_face(face_id_t id) const
    {
        return d_mesh->is_owned_face(id);
    }

    scalar_type cell_volume(cell_id_t id) const
    {
        return d_mesh->cell_volume(id);
    }

    Vec3 cell_centroid(cell_id_t id) const
    {
        return d_mesh->cell_centroid(id);
    }

    auto faces(cell_id_t id) const { return d_mesh->faces(id); }
    cell_id_t owner_cell(face_id_t id) const
    {
        return d_mesh->owner_cell(id);
    }

    cell_id_t neighbor_cell(face_id_t id) const
    {
        return d_mesh->neighbor_cell(id);
    }

    scalar_type face_area(face_id_t id) const
    {
        return d_mesh->face_area(id);
    }

    Vec3 face_centroid(face_id_t id) const
    {
        return d_mesh->face_centroid(id);
    }

    Vec3 face_normal(face_id_t id) const
    {
        return d_mesh->face_normal(id);
    }

    int boundary_id(face_id_t id) const
    {
        return d_mesh->boundary_id(id);
    }

    const std::string& boundary_batch_name(int batch_id) const
    {
        return d_mesh->boundary_batch_name(batch_id);
    }

    const auto& boundary_face_batch(int batch_id) const
    {
        return d_mesh->boundary_face_batch(batch_id);
    }

    const auto& boundary_batches() const noexcept
    {
        return d_mesh->boundary_batches();
    }

    bool is_boundary_face(face_id_t id) const
    {
        return d_mesh->is_boundary_face(id);
    }

    bool is_interior_face(face_id_t id) const
    {
        return d_mesh->is_interior_face(id);
    }

    void export_vtu(const std::string& filename) const
    {
        d_mesh->export_vtu(filename);
    }

private:
    SP<const mesh_type> d_mesh;
    SP<mesh_type> d_mutable_mesh;
};

} // namespace SimpleFluid::Meshes
