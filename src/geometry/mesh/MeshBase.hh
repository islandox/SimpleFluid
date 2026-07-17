/**
 * @file MeshBase.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief CRTP base for meshes that provide finite-volume geometry queries.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "geometry/MeshUtils.hh"
#include "MeshIndexTypes.hh"
#include "BoundaryFaceBatch.hh"
#include "LocalGlobalIndexer.hh"

#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Common finite-volume query interface for statically dispatched meshes.
 *
 * Derived meshes provide the primitive geometry and connectivity operations
 * through private `*_impl` methods. MeshBase supplies the topology predicates,
 * outward geometry, and distance queries shared by finite-volume operators.
 */
template <class DerivedMesh, MeshIndexTypePack Pack>
class MeshBase
{
public:
    using index_type_pack = Pack;
    using cell_id_t = typename Pack::cell_id_t;
    using face_id_t = typename Pack::face_id_t;
    using node_id_t = typename Pack::node_id_t;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using scalar_type = real_t;
    using Vec3 = MeshUtils::Vec3;
    using BoundaryFaceBatch = Meshes::BoundaryFaceBatch<face_id_t>;
    template<std::integral LocalOrdinal = local_ordinal_type,
             std::integral GlobalOrdinal = global_ordinal_type>
    using local_global_indexer_t = Meshes::LocalGlobalIndexer<
        typename Pack::template rebind_ordinals<
            LocalOrdinal,
            GlobalOrdinal>>;

    static constexpr int invalid_boundary_id = -1;
    static constexpr local_ordinal_type invalid_local_id =
        std::numeric_limits<local_ordinal_type>::max();

    constexpr auto spatial_dimension() const noexcept { return d_dim; }
    size_t num_local_cells() const noexcept { return d_num_local_cells; }
    size_t num_owned_cells() const noexcept { return d_num_owned_cells; }
    size_t num_cells() const noexcept { return d_num_cells; }
    size_t num_faces() const noexcept { return d_num_faces; }
    size_t num_owned_faces() const noexcept { return d_num_owned_faces; }
    size_t num_nodes() const noexcept { return d_num_nodes; }

    cell_id_t cell_id(local_ordinal_type local_id) const
    {
        if (local_id >= num_cells())
        {
            throw std::out_of_range("Cell local ID is out of range.");
        }
        return derived().indexer().cell_id(local_id);
    }

    face_id_t face_id(local_ordinal_type local_id) const
    {
        if (local_id >= num_faces())
        {
            throw std::out_of_range("Face local ID is out of range.");
        }
        return derived().indexer().face_id(local_id);
    }

    node_id_t node_id(local_ordinal_type local_id) const
    {
        if (local_id >= num_nodes())
        {
            throw std::out_of_range("Node local ID is out of range.");
        }
        return derived().indexer().node_id(local_id);
    }

    local_ordinal_type cell_local_id(cell_id_t cell_id) const
    {
        if (cell_id == derived().invalid_cell_id())
        {
            return invalid_local_id;
        }
        derived().check_cell_id(cell_id);
        return derived().indexer().cell_ordinal(cell_id);
    }

    local_ordinal_type face_local_id(face_id_t face_id) const
    {
        derived().check_face_id(face_id);
        return derived().indexer().face_ordinal(face_id);
    }

    local_ordinal_type node_local_id(node_id_t node_id) const
    {
        derived().check_node_id(node_id);
        return derived().indexer().node_ordinal(node_id);
    }

    bool is_owned_cell(cell_id_t cell_id) const
    {
        derived().check_cell_id(cell_id);
        return derived().is_owned_cell_impl(cell_id);
    }

    bool is_owned_face(face_id_t face_id) const
    {
        derived().check_face_id(face_id);
        return derived().is_owned_face_impl(face_id);
    }

    real_t cell_volume(cell_id_t cell_id) const
    {
        derived().check_cell_id(cell_id);
        return derived().cell_volume_impl(cell_id);
    }

    Vec3 cell_centroid(cell_id_t cell_id) const
    {
        derived().check_cell_id(cell_id);
        return derived().cell_centroid_impl(cell_id);
    }

    auto cell_faces(cell_id_t cell_id) const
    {
        derived().check_cell_id(cell_id);
        return derived().cell_faces_impl(cell_id);
    }

    auto faces(cell_id_t cell_id) const
    {
        return cell_faces(cell_id);
    }

    auto face_distances(cell_id_t cell_id) const
    {
        const auto cell_face_ids = faces(cell_id);
        std::vector<real_t> distances;
        distances.reserve(cell_face_ids.size());
        for (const auto face_id : cell_face_ids)
        {
            distances.push_back(cell_to_face_distance(face_id, cell_id));
        }
        return distances;
    }

    cell_id_t owner_cell(face_id_t face_id) const
    {
        derived().check_face_id(face_id);
        return derived().owner_cell_impl(face_id);
    }
    cell_id_t neighbor_cell(face_id_t face_id) const
    {
        derived().check_face_id(face_id);
        return derived().neighbor_cell_impl(face_id);
    }

    cell_id_t opposite_cell(face_id_t face_id, cell_id_t cell_id) const
    {
        derived().check_cell_id(cell_id);
        const auto owner = owner_cell(face_id);
        const auto neighbor = neighbor_cell(face_id);

        if (cell_id == owner)
        {
            return neighbor;
        }
        if (neighbor != derived().invalid_cell_id() && cell_id == neighbor)
        {
            return owner;
        }

        throw std::invalid_argument("Cell is not adjacent to requested face.");
    }

    cell_id_t opposite_or_periodic_neighbor_cell(
        face_id_t face_id,
        cell_id_t cell_id) const
    {
        return opposite_cell(face_id, cell_id);
    }

    real_t face_area(face_id_t face_id) const
    {
        derived().check_face_id(face_id);
        return derived().face_area_impl(face_id);
    }

    Vec3 face_centroid(face_id_t face_id) const
    {
        derived().check_face_id(face_id);
        return derived().face_centroid_impl(face_id);
    }

    Vec3 face_normal(face_id_t face_id) const
    {
        derived().check_face_id(face_id);
        return derived().face_normal_impl(face_id);
    }

    Vec3 face_area_vector(face_id_t face_id) const
    {
        return face_normal(face_id) * face_area(face_id);
    }

    Vec3 face_normal_outward(face_id_t face_id, cell_id_t cell_id) const
    {
        const auto owner = owner_cell(face_id);
        const auto neighbor = neighbor_cell(face_id);

        if (cell_id == owner)
        {
            return face_normal(face_id);
        }
        if (neighbor != derived().invalid_cell_id() && cell_id == neighbor)
        {
            return face_normal(face_id) * -1.0;
        }

        derived().check_cell_id(cell_id);
        throw std::invalid_argument("Cell is not adjacent to requested face.");
    }

    Vec3 face_area_vector_outward(
        face_id_t face_id,
        cell_id_t cell_id) const
    {
        return face_normal_outward(face_id, cell_id) * face_area(face_id);
    }

    real_t face_cell_center_distance(face_id_t face_id) const
    {
        const auto neighbor = neighbor_cell(face_id);
        if (neighbor == derived().invalid_cell_id())
        {
            return 0.0;
        }

        return (cell_centroid(neighbor)
              - cell_centroid(owner_cell(face_id))).norm();
    }

    Vec3 cell_center_vector(face_id_t face_id, cell_id_t cell_id) const
    {
        const auto other = opposite_cell(face_id, cell_id);
        if (other == derived().invalid_cell_id())
        {
            throw std::invalid_argument(
                "Exterior face does not have an opposite cell.");
        }
        return cell_centroid(other) - cell_centroid(cell_id);
    }

    real_t cell_to_face_distance(
        face_id_t face_id,
        cell_id_t cell_id) const
    {
        const auto owner = owner_cell(face_id);
        const auto neighbor = neighbor_cell(face_id);
        if (cell_id != owner
            && (neighbor == derived().invalid_cell_id() || cell_id != neighbor))
        {
            derived().check_cell_id(cell_id);
            throw std::invalid_argument(
                "Cell is not adjacent to requested face.");
        }

        return (face_centroid(face_id) - cell_centroid(cell_id)).norm();
    }

    bool is_exterior_face(face_id_t face_id) const
    {
        return neighbor_cell(face_id) == derived().invalid_cell_id();
    }

    bool is_interior_face(face_id_t face_id) const
    {
        return !is_exterior_face(face_id);
    }

    bool is_boundary_face(face_id_t face_id) const
    {
        return is_exterior_face(face_id)
            && boundary_id(face_id) != invalid_boundary_id;
    }

    int boundary_id(face_id_t face_id) const
    {
        derived().check_face_id(face_id);
        return derived().boundary_id_impl(face_id);
    }

    const std::string& boundary_name(face_id_t face_id) const
    {
        const auto batch_id = boundary_id(face_id);
        if (batch_id == invalid_boundary_id)
        {
            throw std::out_of_range(
                "Requested face is not a boundary face.");
        }
        return boundary_batch_name(batch_id);
    }

    const std::string& boundary_batch_name(int batch_id) const
    {
        return derived().boundary_batch_name_impl(batch_id);
    }

    auto boundary_face_batch(int batch_id) const
    {
        return derived().boundary_face_batch_impl(batch_id);
    }

    auto boundary_batch_ids() const
    {
        return derived().boundary_batch_ids_impl();
    }

    int num_boundary_batches() const noexcept
    {
        return derived().num_boundary_batches_impl();
    }

    auto boundary_batches() const
    {
        return derived().boundary_batches_impl();
    }

    Vec3 node_coordinates(node_id_t node_id) const
    {
        derived().check_node_id(node_id);
        return derived().node_coordinates_impl(node_id);
    }

    Vec3 node_coord(node_id_t node_id) const
    {
        return node_coordinates(node_id);
    }

protected:
    constexpr MeshBase() = default;

    static constexpr uint8_t d_dim = 3;
    size_t d_num_local_cells = 0;
    size_t d_num_owned_cells = 0;
    size_t d_num_cells = 0;
    size_t d_num_faces = 0;
    size_t d_num_owned_faces = 0;
    size_t d_num_nodes = 0;

private:
    const DerivedMesh& derived() const noexcept
    {
        return static_cast<const DerivedMesh&>(*this);
    }
};

template <class DerivedMesh>
concept MeshClass = requires {
    typename DerivedMesh::index_type_pack;
    typename DerivedMesh::Indexer;
} && MeshIndexTypePack<typename DerivedMesh::index_type_pack>
  && MeshIndexer<typename DerivedMesh::Indexer>
  && std::same_as<
         typename DerivedMesh::Indexer::cell_id_t,
         typename DerivedMesh::index_type_pack::cell_id_t>
  && std::same_as<
         typename DerivedMesh::Indexer::face_id_t,
         typename DerivedMesh::index_type_pack::face_id_t>
  && std::same_as<
         typename DerivedMesh::Indexer::node_id_t,
         typename DerivedMesh::index_type_pack::node_id_t>
  && std::same_as<
         typename DerivedMesh::Indexer::ordinal_t,
         typename DerivedMesh::index_type_pack::local_ordinal_type>
  && std::derived_from<
         DerivedMesh,
         MeshBase<DerivedMesh, typename DerivedMesh::index_type_pack>>;

} // namespace SimpleFluid
