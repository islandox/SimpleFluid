/**
 * @file MeshReorderingFactory.tcc
 * @brief Template implementation of MeshReorderingFactory.
 */

#pragma once

#include "geometry/MeshReorderingFactory.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Access.hpp>
#include <Tpetra_CombineMode.hpp>

#include <algorithm>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

namespace SimpleFluid
{

template<TpetraTypePack Pack>
auto MeshReorderingFactory<Pack>::partitioned_xy_order(
    const Meshes::SemiStructMeshTopo& topology, std::span<const std::size_t> xy_owners) -> std::vector<std::size_t>
{
    const std::size_t count = topology.indexer().num_cells_per_layer;
    if (xy_owners.size() != count)
        throw std::invalid_argument("XY partition owners must match the base-topology cell count.");
    std::map<std::size_t, std::vector<std::size_t>> partitions;
    for (std::size_t cell = 0; cell < count; ++cell)
    {
        if (xy_owners[cell] == std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("XY partition owner is unknown.");
        partitions[xy_owners[cell]].push_back(cell);
    }
    std::vector<std::vector<std::size_t>> adjacency(count);
    for (const auto& edge : topology.side_faces())
    {
        if (edge.owner == Meshes::SemiStructMeshTopo::invalid_ordinal ||
            edge.neighbor == Meshes::SemiStructMeshTopo::invalid_ordinal)
            continue;
        const std::size_t a = edge.owner, b = edge.neighbor;
        if (a >= count || b >= count)
            throw std::invalid_argument("XY topology contains an out-of-range adjacent cell.");
        if (xy_owners[a] == xy_owners[b])
        {
            adjacency[a].push_back(b);
            adjacency[b].push_back(a);
        }
    }
    for (auto& neighbors : adjacency)
    {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }
    std::vector<std::size_t> result;
    result.reserve(count);
    std::vector<unsigned char> visited(count);
    for (const auto& [owner, cells] : partitions)
    {
        static_cast<void>(owner);
        std::set<std::pair<std::size_t, std::size_t>> pending;
        for (const auto cell : cells)
            pending.emplace(adjacency[cell].size(), cell);
        while (!pending.empty())
        {
            std::vector<std::size_t> component{pending.begin()->second};
            visited[component.front()] = 1;
            pending.erase(pending.begin());
            for (std::size_t head = 0; head < component.size(); ++head)
            {
                std::vector<std::size_t> neighbors;
                for (const auto neighbor : adjacency[component[head]])
                    if (!visited[neighbor])
                        neighbors.push_back(neighbor);
                std::sort(neighbors.begin(), neighbors.end(), [&](std::size_t a, std::size_t b)
                    { return std::pair{adjacency[a].size(), a} < std::pair{adjacency[b].size(), b}; });
                for (const auto neighbor : neighbors)
                {
                    visited[neighbor] = 1;
                    pending.erase({adjacency[neighbor].size(), neighbor});
                    component.push_back(neighbor);
                }
            }
            result.insert(result.end(), component.rbegin(), component.rend());
        }
    }
    return result;
}

template<TpetraTypePack Pack>
auto MeshReorderingFactory<Pack>::selected_cells_first(SP<mesh_type>&& parent, cell_selector_type selector)
    -> SelectedCellLayout
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using scalar_type = typename Pack::scalar_type;

    if (!parent)
    {
        throw std::invalid_argument("MeshReorderingFactory requires a non-null mesh.");
    }

    if (std::holds_alternative<typename mesh_type::MultiRegionPtr>(parent->d_mesh))
        throw std::invalid_argument("Arbitrary cell reordering of compact MultiRegionMesh is unsupported.");
    const auto comm = parent->owned_cell_map()->getComm();
    std::vector<unsigned char> selected(parent->num_local_cells(), 0);
    {
        typename Pack::vector_type owned_mask(parent->owned_cell_map(), true);
        std::exception_ptr local_error;
        if (!selector)
        {
            local_error = std::make_exception_ptr(
                std::invalid_argument("MeshReorderingFactory requires a callable cell selector."));
        }
        else
        {
            try
            {
                for (size_t owned = 0; owned < parent->num_owned_cells(); ++owned)
                {
                    const auto local = mesh_type::checked_local(owned);
                    const bool is_selected =
                        selector(parent->cell_geometry_global_id(local), parent->cell_centroid(local));
                    owned_mask.replaceLocalValue(local, is_selected ? scalar_type{1} : scalar_type{});
                }
            }
            catch (...)
            {
                local_error = std::current_exception();
            }
        }

        const int local_failed = local_error ? 1 : 0;
        int any_failed = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_failed, &any_failed);
        if (any_failed != 0)
        {
            if (local_error)
            {
                std::rethrow_exception(local_error);
            }
            throw std::runtime_error("MeshReorderingFactory cell selector failed on another rank.");
        }

        typename Pack::vector_type overlap_mask(parent->overlap_cell_map(), true);
        typename Pack::import_type importer(parent->owned_cell_map(), parent->overlap_cell_map());
        const int local_import_is_complete = importer.isLocallyComplete() ? 1 : 0;
        int every_import_is_complete = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_import_is_complete, &every_import_is_complete);
        if (every_import_is_complete == 0)
        {
            throw std::invalid_argument(
                "MeshReorderingFactory requires every overlap cell to have an owner in the mesh communicator.");
        }
        overlap_mask.doImport(owned_mask, importer, Tpetra::REPLACE);

        const auto overlap_values = overlap_mask.getLocalViewHost(Tpetra::Access::ReadOnly);
        for (size_t local = 0; local < selected.size(); ++local)
        {
            selected[local] = overlap_values(mesh_type::checked_local(local), 0) != scalar_type{}
                                  ? static_cast<unsigned char>(1)
                                  : static_cast<unsigned char>(0);
        }
    }

    const auto owned_count = parent->num_owned_cells();
    const auto selected_owned = static_cast<size_t>(std::count(
        selected.begin(), selected.begin() + static_cast<std::ptrdiff_t>(owned_count), static_cast<unsigned char>(1)));
    const auto selected_ghost = static_cast<size_t>(std::count(
        selected.begin() + static_cast<std::ptrdiff_t>(owned_count), selected.end(), static_cast<unsigned char>(1)));

    const auto selected_first = [&](size_t begin, size_t end)
    {
        bool saw_unselected = false;
        for (size_t local = begin; local < end; ++local)
        {
            if (selected[local] == 0)
            {
                saw_unselected = true;
            }
            else if (saw_unselected)
            {
                return false;
            }
        }
        return true;
    };

    const bool local_identity = selected_first(0, owned_count) && selected_first(owned_count, selected.size());
    const int local_identity_value = local_identity ? 1 : 0;
    int every_rank_identity = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_identity_value, &every_rank_identity);
    if (every_rank_identity != 0)
    {
        return {std::move(parent), selected_owned, selected_ghost};
    }

    const int local_unique_owner = parent.use_count() == 1 ? 1 : 0;
    int every_rank_has_unique_owner = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_unique_owner, &every_rank_has_unique_owner);
    if (every_rank_has_unique_owner == 0)
    {
        throw std::invalid_argument(
            "MeshReorderingFactory requires unique mesh-handle ownership before changing local cell order.");
    }

    std::vector<local_ordinal_type> reordered_to_parent;
    reordered_to_parent.reserve(selected.size());
    auto append_group = [&](size_t begin, size_t end, bool requested)
    {
        for (size_t local = begin; local < end; ++local)
        {
            if ((selected[local] != 0) == requested)
            {
                reordered_to_parent.push_back(mesh_type::checked_local(local));
            }
        }
    };
    append_group(0, owned_count, true);
    append_group(0, owned_count, false);
    append_group(owned_count, selected.size(), true);
    append_group(owned_count, selected.size(), false);

    if (parent->d_orthogonal_topology || parent->d_semi_structured_topology)
    {
        // The compact decomposition describes the original local topology order.
        // An explicit later selection composes only an opt-in local overlay;
        // algebraic GIDs remain attached to the same physical cells.
        auto& forward_overlay =
            parent->d_orthogonal_topology ? parent->d_orthogonal_cell_permutation : parent->d_semi_cell_permutation;
        auto& inverse_overlay = parent->d_orthogonal_topology ? parent->d_orthogonal_cell_inverse_permutation
                                                              : parent->d_semi_cell_inverse_permutation;
        std::vector<std::size_t> permutation(reordered_to_parent.size()), inverse(reordered_to_parent.size());
        std::vector<global_ordinal_type> owned_gids, overlap_gids;
        owned_gids.reserve(owned_count);
        overlap_gids.reserve(reordered_to_parent.size());
        bool overlay_identity = true;
        for (std::size_t local = 0; local < reordered_to_parent.size(); ++local)
        {
            const auto previous = reordered_to_parent[local];
            const auto compact = forward_overlay.empty() ? static_cast<std::size_t>(previous)
                                                         : forward_overlay[static_cast<std::size_t>(previous)];
            permutation[local] = compact;
            inverse[compact] = local;
            overlay_identity = overlay_identity && compact == local;
            const auto algebraic = parent->overlap_cell_map()->getGlobalElement(previous);
            overlap_gids.push_back(algebraic);
            if (local < owned_count)
                owned_gids.push_back(algebraic);
        }
        auto owned_map = parent->make_map(comm, owned_gids);
        auto overlap_map = parent->make_map(comm, overlap_gids);
        if (overlay_identity)
        {
            forward_overlay.clear();
            inverse_overlay.clear();
        }
        else
        {
            forward_overlay = std::move(permutation);
            inverse_overlay = std::move(inverse);
        }
        parent->d_owned_cell_map = std::move(owned_map);
        parent->d_overlap_cell_map = std::move(overlap_map);
        parent->d_serial_identity = false;
        parent->d_map_indexing = true;
        if (!local_identity)
        {
            parent->d_cells_reordered = true;
            parent->d_indexer = typename mesh_type::indexer_type{};
            parent->d_legacy_indexer_materialized = false;
            if (!parent->d_cell_face_offsets.empty())
                parent->initialize_cell_faces();
        }
        return {std::move(parent), selected_owned, selected_ghost};
    }

    // Arbitrary reordering explicitly opts into a stored permutation/indexer.
    if (parent->d_serial_identity || parent->d_map_indexing)
    {
        static_cast<void>(parent->indexer());
        parent->d_serial_identity = false;
        parent->d_map_indexing = false;
    }

    // Structured backends already route local-to-geometry lookup through the
    // existing local/global indexer. Explicit-storage backends historically
    // assumed identity cell ordinals and therefore need one composed
    // permutation in each direction.
    const bool uses_identity_geometry_ordinals =
        parent->is_stk() || std::holds_alternative<typename mesh_type::UnstructuredPtr>(parent->d_mesh);
    std::vector<global_ordinal_type> reordered_geometry_lids;
    if (!local_identity && uses_identity_geometry_ordinals)
    {
        reordered_geometry_lids.reserve(reordered_to_parent.size());
        for (const auto parent_local : reordered_to_parent)
        {
            reordered_geometry_lids.push_back(parent->geometry_cell_lid(parent_local));
        }
    }

    std::vector<global_ordinal_type> owned_ids;
    std::vector<global_ordinal_type> ghost_ids;
    if (!parent->is_stk())
    {
        owned_ids.reserve(owned_count);
        ghost_ids.reserve(selected.size() - owned_count);
        for (size_t local = 0; local < reordered_to_parent.size(); ++local)
        {
            const auto parent_local = reordered_to_parent[local];
            const auto geometry_id = parent->d_indexer.cell_global_id(parent_local);
            (local < owned_count ? owned_ids : ghost_ids).push_back(geometry_id);
        }
    }

    auto reordered = std::move(parent);
    reordered->d_cells_reordered = true;

    if (!local_identity && uses_identity_geometry_ordinals)
    {
        reordered->d_cell_geometry_lids = std::move(reordered_geometry_lids);
        reordered->d_cell_local_lids_by_geometry.assign(reordered_to_parent.size(), mesh_type::invalid_local_id());
        for (size_t local = 0; local < reordered->d_cell_geometry_lids.size(); ++local)
        {
            const auto geometry = reordered->d_cell_geometry_lids[local];
            reordered->d_cell_local_lids_by_geometry[static_cast<size_t>(geometry)] = mesh_type::checked_local(local);
        }
    }

    if (!reordered->is_stk())
    {
        reordered->d_indexer.set_cells(std::move(owned_ids), std::move(ghost_ids));
    }

    if (!local_identity && reordered->is_stk())
    {
        // A legacy indexer is an on-demand compatibility object. Invalidate a
        // copied materialization so the next request observes the new order.
        reordered->d_indexer = typename mesh_type::indexer_type{};
        reordered->d_legacy_indexer_materialized = false;
    }

    // Preserve the legacy zero-copy cell-face path whenever the face order is
    // already identity. Other backends and the rare legacy face permutation
    // need their compact connectivity rows rebuilt in the new cell order.
    if (!local_identity && !reordered->d_cell_face_offsets.empty())
    {
        reordered->initialize_cell_faces();
    }

    std::vector<global_ordinal_type> owned_gids;
    std::vector<global_ordinal_type> overlap_gids;
    owned_gids.reserve(owned_count);
    overlap_gids.reserve(selected.size());
    for (size_t local = 0; local < selected.size(); ++local)
    {
        const auto gid = reordered->cell_global_id(mesh_type::checked_local(local));
        overlap_gids.push_back(gid);
        if (local < owned_count)
        {
            owned_gids.push_back(gid);
        }
    }
    reordered->d_owned_cell_map = reordered->make_map(comm, owned_gids);
    reordered->d_overlap_cell_map = reordered->make_map(comm, overlap_gids);

    return {std::move(reordered), selected_owned, selected_ghost};
}

} // namespace SimpleFluid
