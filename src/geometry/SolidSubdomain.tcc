/**
 * @file SolidSubdomain.tcc
 * @brief Template construction for SolidSubdomain.
 */

#pragma once

#include "geometry/SolidSubdomain.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <limits>

namespace SimpleFluid
{

template<TpetraTypePack Pack>
SolidSubdomain<Pack>::SolidSubdomain(SP<mesh_type>&& parent, cell_selector_type selector, std::string interface_name)
    : SolidSubdomain(reordering_factory_type::selected_cells_first(std::move(parent), std::move(selector)),
          std::move(interface_name))
{
}

/** Build a solid view from an already-reordered parent layout. */
template<TpetraTypePack Pack>
SolidSubdomain<Pack>::SolidSubdomain(selected_cell_layout_type layout, std::string interface_name)
    : d_parent(require_parent(std::move(layout.mesh))), d_interface_boundary_name(std::move(interface_name))
{
    initialize_layout(layout.selected_owned_cells, layout.selected_ghost_cells);
    validate_owned_adjacency();
    initialize_interface_boundary();
    initialize_faces();
    initialize_cell_faces();
    initialize_boundary_batches();
    initialize_maps();
}

/** Build a solid view containing every parent-mesh cell. */
template<TpetraTypePack Pack>
SolidSubdomain<Pack>::SolidSubdomain(SP<const mesh_type> parent, std::string interface_name)
    : SolidSubdomain(all_cells_layout(std::move(parent)), std::move(interface_name))
{
}

/** Validate and retain the two selected prefixes described by the layout. */
template<TpetraTypePack Pack>
void SolidSubdomain<Pack>::initialize_layout(size_t selected_owned_cells, size_t selected_ghost_cells)
{
    const auto comm = d_parent->owned_cell_map()->getComm();
    const auto parent_owned = d_parent->num_owned_cells();
    const auto parent_local = d_parent->num_local_cells();
    const auto max_local = static_cast<size_t>(std::numeric_limits<local_ordinal_type>::max());
    typename Pack::import_type importer(d_parent->owned_cell_map(), d_parent->overlap_cell_map());
    const int local_layout_is_valid = parent_owned <= parent_local && selected_owned_cells <= parent_owned &&
                                              selected_ghost_cells <= parent_local - parent_owned &&
                                              selected_owned_cells <= max_local && selected_ghost_cells <= max_local &&
                                              selected_owned_cells <= max_local - selected_ghost_cells &&
                                              importer.isLocallyComplete()
                                          ? 1
                                          : 0;
    int every_layout_is_valid = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_layout_is_valid, &every_layout_is_valid);
    if (every_layout_is_valid == 0)
    {
        throw std::invalid_argument("SolidSubdomain selected-cell layout does not match its parent mesh.");
    }

    d_num_owned_cells = selected_owned_cells;
    d_num_ghost_cells = selected_ghost_cells;

    const auto local_selected = static_cast<unsigned long long>(selected_owned_cells);
    unsigned long long global_selected = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_selected, &global_selected);
    if (global_selected == 0)
    {
        throw std::invalid_argument("SolidSubdomain selection is globally empty.");
    }
}

/** Require every selected owned cell to see the cell across each interior face. */
template<TpetraTypePack Pack> void SolidSubdomain<Pack>::validate_owned_adjacency() const
{
    int local_missing_adjacency = 0;
    for (size_t solid_cell = 0; solid_cell < d_num_owned_cells; ++solid_cell)
    {
        const auto parent_cell = parent_cell_lid(checked_local(solid_cell));
        for (const auto parent_face : d_parent->faces(parent_cell))
        {
            const auto opposite = d_parent->opposite_or_periodic_neighbor_cell(parent_face, parent_cell);
            if (opposite == mesh_type::invalid_local_id() && !d_parent->is_geometry_exterior_face(parent_face))
            {
                local_missing_adjacency = 1;
                break;
            }
        }
        if (local_missing_adjacency != 0)
        {
            break;
        }
    }

    int any_missing_adjacency = 0;
    Teuchos::reduceAll(*d_parent->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_missing_adjacency,
        &any_missing_adjacency);
    if (any_missing_adjacency != 0)
    {
        throw std::invalid_argument(
            "SolidSubdomain requires parent overlap containing every neighbor of a selected owned cell.");
    }
}

/** Allocate one collision-free boundary ID consistently on the communicator. */
template<TpetraTypePack Pack> void SolidSubdomain<Pack>::initialize_interface_boundary()
{
    const auto comm = d_parent->owned_cell_map()->getComm();
    const int local_name_size_is_valid =
        d_interface_boundary_name.size() <= static_cast<size_t>(std::numeric_limits<int>::max()) ? 1 : 0;
    int global_name_size_is_valid = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_name_size_is_valid, &global_name_size_is_valid);
    if (global_name_size_is_valid == 0)
    {
        throw std::invalid_argument("SolidSubdomain interface name is too large for collective validation.");
    }

    const auto local_name_size = static_cast<int>(d_interface_boundary_name.size());
    int minimum_name_size = 0;
    int maximum_name_size = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_name_size, &minimum_name_size);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_name_size, &maximum_name_size);
    if (minimum_name_size != maximum_name_size)
    {
        throw std::invalid_argument("SolidSubdomain interface name must agree on every rank.");
    }
    if (minimum_name_size == 0)
    {
        throw std::invalid_argument("SolidSubdomain interface name cannot be empty.");
    }

    std::vector<int> local_name_bytes;
    local_name_bytes.reserve(d_interface_boundary_name.size());
    for (const unsigned char byte : d_interface_boundary_name)
    {
        local_name_bytes.push_back(static_cast<int>(byte));
    }
    std::vector<int> minimum_name_bytes(local_name_bytes.size());
    std::vector<int> maximum_name_bytes(local_name_bytes.size());
    if (!local_name_bytes.empty())
    {
        Teuchos::reduceAll(
            *comm, Teuchos::REDUCE_MIN, local_name_size, local_name_bytes.data(), minimum_name_bytes.data());
        Teuchos::reduceAll(
            *comm, Teuchos::REDUCE_MAX, local_name_size, local_name_bytes.data(), maximum_name_bytes.data());
    }
    if (minimum_name_bytes != maximum_name_bytes)
    {
        throw std::invalid_argument("SolidSubdomain interface name must agree on every rank.");
    }

    int local_max_id = invalid_boundary_id;
    int local_name_collision = 0;
    for (const auto& [batch_id, batch] : d_parent->boundary_batches())
    {
        static_cast<void>(batch);
        local_max_id = std::max(local_max_id, batch_id);
        local_name_collision =
            local_name_collision || d_parent->boundary_batch_name(batch_id) == d_interface_boundary_name;
    }

    int global_max_id = invalid_boundary_id;
    int any_name_collision = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_max_id, &global_max_id);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_name_collision, &any_name_collision);
    if (any_name_collision != 0)
    {
        throw std::invalid_argument("SolidSubdomain interface name collides with a parent boundary.");
    }
    if (global_max_id == std::numeric_limits<int>::max())
    {
        throw std::overflow_error("SolidSubdomain cannot allocate an interface boundary ID.");
    }
    d_interface_boundary_id = global_max_id + 1;
}

/** Classify parent faces and pack owned faces before overlap-only faces. */
template<TpetraTypePack Pack> void SolidSubdomain<Pack>::initialize_faces()
{
    std::vector<local_ordinal_type> candidate_parent_faces;
    for (size_t solid_cell = 0; solid_cell < num_local_cells(); ++solid_cell)
    {
        const auto parent_cell = parent_cell_lid(checked_local(solid_cell));
        const auto parent_faces = d_parent->faces(parent_cell);
        if (parent_faces.size() > candidate_parent_faces.max_size() - candidate_parent_faces.size())
        {
            throw std::overflow_error("SolidSubdomain face-incidence count overflows local storage.");
        }
        candidate_parent_faces.insert(candidate_parent_faces.end(), parent_faces.begin(), parent_faces.end());
    }
    std::sort(candidate_parent_faces.begin(), candidate_parent_faces.end());
    candidate_parent_faces.erase(
        std::unique(candidate_parent_faces.begin(), candidate_parent_faces.end()), candidate_parent_faces.end());

    std::vector<FaceRecord> owned_faces;
    std::vector<FaceRecord> overlap_faces;

    for (const auto parent_face_lid : candidate_parent_faces)
    {
        const auto parent_owner = d_parent->owner_cell(parent_face_lid);
        const auto parent_neighbor = d_parent->neighbor_cell(parent_face_lid);
        const bool owner_selected = parent_owner != mesh_type::invalid_local_id() && contains_parent_cell(parent_owner);
        const bool neighbor_selected =
            parent_neighbor != mesh_type::invalid_local_id() && contains_parent_cell(parent_neighbor);
        if (!owner_selected && !neighbor_selected)
        {
            throw std::logic_error("SolidSubdomain candidate face has no selected adjacent cell.");
        }

        FaceRecord record;
        record.parent_lid = parent_face_lid;
        bool owned = false;
        if (owner_selected && neighbor_selected)
        {
            record.owner = subdomain_cell_lid(parent_owner);
            record.neighbor = subdomain_cell_lid(parent_neighbor);
            owned = d_parent->is_owned_face(parent_face_lid);
        }
        else
        {
            const auto selected_parent = owner_selected ? parent_owner : parent_neighbor;
            record.owner = subdomain_cell_lid(selected_parent);
            record.reverse_normal = !owner_selected;
            owned = d_parent->is_owned_cell(selected_parent);

            const auto outside_parent = owner_selected ? parent_neighbor : parent_owner;
            if (outside_parent != mesh_type::invalid_local_id())
            {
                record.interface = true;
                record.boundary_id = d_interface_boundary_id;
                record.outside_parent_cell = outside_parent;
            }
            else
            {
                record.boundary_id = d_parent->boundary_id(parent_face_lid);
            }
        }

        (owned ? owned_faces : overlap_faces).push_back(record);
    }

    d_num_owned_faces = owned_faces.size();
    d_faces.reserve(owned_faces.size() + overlap_faces.size());
    d_faces.insert(d_faces.end(), owned_faces.begin(), owned_faces.end());
    d_faces.insert(d_faces.end(), overlap_faces.begin(), overlap_faces.end());

    for (size_t local = 0; local < d_faces.size(); ++local)
    {
        const auto face_lid = checked_local(local);
        auto& record = d_faces[local];
        if (record.interface)
        {
            d_interface_faces.push_back(InterfaceFace{.face_lid = face_lid,
                .solid_cell_lid = record.owner,
                .parent_face_lid = record.parent_lid,
                .outside_parent_cell_lid = record.outside_parent_cell,
                .outside_cell_geometry_global_id = d_parent->cell_geometry_global_id(record.outside_parent_cell)});
        }
    }
}

/** Materialize selected-cell connectivity in subdomain face ordinals. */
template<TpetraTypePack Pack> void SolidSubdomain<Pack>::initialize_cell_faces()
{
    d_cell_face_offsets.reserve(num_local_cells() + 1);
    d_cell_face_offsets.push_back(0);
    for (size_t solid_cell = 0; solid_cell < num_local_cells(); ++solid_cell)
    {
        const auto parent_cell = parent_cell_lid(checked_local(solid_cell));
        for (const auto parent_face : d_parent->faces(parent_cell))
        {
            const auto local_face = subdomain_face_lid(parent_face);
            if (local_face == invalid_local_id())
            {
                throw std::logic_error("SolidSubdomain omitted a selected cell face.");
            }
            d_cell_face_lids.push_back(local_face);
        }
        d_cell_face_offsets.push_back(d_cell_face_lids.size());
    }
}

/** Group physical and synthetic boundaries using the packed face IDs. */
template<TpetraTypePack Pack> void SolidSubdomain<Pack>::initialize_boundary_batches()
{
    for (size_t local = 0; local < d_faces.size(); ++local)
    {
        const auto& face = d_faces[local];
        if (face.boundary_id == invalid_boundary_id)
        {
            continue;
        }

        auto [iter, inserted] =
            d_boundary_batches.try_emplace(face.boundary_id, BoundaryFaceBatch{.id = face.boundary_id});
        static_cast<void>(inserted);
        iter->second.face_lids.push_back(checked_local(local));

        if (face.interface)
        {
            d_boundary_names.try_emplace(face.boundary_id, d_interface_boundary_name);
        }
        else
        {
            d_boundary_names.try_emplace(face.boundary_id, d_parent->boundary_batch_name(face.boundary_id));
        }
    }
}

/** Build subset maps using the parent's communicator and global entity IDs. */
template<TpetraTypePack Pack> void SolidSubdomain<Pack>::initialize_maps()
{
    std::vector<global_ordinal_type> overlap_cells;
    overlap_cells.reserve(num_local_cells());
    for (size_t local = 0; local < num_local_cells(); ++local)
    {
        overlap_cells.push_back(cell_global_id(checked_local(local)));
    }

    std::vector<global_ordinal_type> overlap_faces;
    std::vector<global_ordinal_type> boundary_faces;
    overlap_faces.reserve(num_faces());
    for (size_t local = 0; local < num_faces(); ++local)
    {
        const auto face_lid = checked_local(local);
        const auto gid = face_global_id(face_lid);
        overlap_faces.push_back(gid);
        if (local < num_owned_faces() && is_boundary_face(face_lid))
        {
            boundary_faces.push_back(gid);
        }
    }

    const auto cell_gids = std::span<const global_ordinal_type>(overlap_cells);
    const auto face_gids = std::span<const global_ordinal_type>(overlap_faces);
    d_owned_cell_map = make_map(cell_gids.first(num_owned_cells()));
    d_overlap_cell_map = make_map(cell_gids);
    d_owned_face_map = make_map(face_gids.first(num_owned_faces()));
    d_overlap_face_map = make_map(face_gids);
    d_boundary_face_map = make_map(std::span<const global_ordinal_type>(boundary_faces));
}

} // namespace SimpleFluid
