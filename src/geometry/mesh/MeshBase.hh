/**
 * @file MeshBase.hh
 * @brief CRTP base for meshes that provide finite-volume geometry queries.
 */

#pragma once

#include "geometry/MeshUtils.hh"
#include "geometry/mesh/BoundaryFacePatch.hh"

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
template <class DerivedMesh, class CellID, class FaceID, class NodeID>
class MeshBase
{
public:
    using cell_id_t = CellID;
    using face_id_t = FaceID;
    using node_id_t = NodeID;
    using local_ordinal_type = size_t;
    using scalar_type = real_t;
    using Vec3 = MeshUtils::Vec3;
    using BoundaryFacePatch = Meshes::BoundaryFacePatch<face_id_t>;

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
        return derived().indexer().cell_local_id(cell_id);
    }

    local_ordinal_type face_local_id(face_id_t face_id) const
    {
        derived().check_face_id(face_id);
        return derived().indexer().face_local_id(face_id);
    }

    local_ordinal_type node_local_id(node_id_t node_id) const
    {
        derived().check_node_id(node_id);
        return derived().indexer().node_local_id(node_id);
    }

    bool is_owned_cell(cell_id_t cell_id) const
    {
        derived().check_cell_id(cell_id);
        return derived().is_owned_cell_impl(cell_id);
    }

    bool is_owned_cell(local_ordinal_type local_id) const
    {
        return is_owned_cell(cell_id(local_id));
    }

    bool is_owned_face(face_id_t face_id) const
    {
        derived().check_face_id(face_id);
        return derived().is_owned_face_impl(face_id);
    }

    bool is_owned_face(local_ordinal_type local_id) const
    {
        return is_owned_face(face_id(local_id));
    }

    real_t cell_volume(cell_id_t cell_id) const
    {
        derived().check_cell_id(cell_id);
        return derived().cell_volume_impl(cell_id);
    }

    real_t cell_volume(local_ordinal_type local_id) const
    {
        return cell_volume(cell_id(local_id));
    }

    Vec3 cell_centroid(cell_id_t cell_id) const
    {
        derived().check_cell_id(cell_id);
        return derived().cell_centroid_impl(cell_id);
    }

    Vec3 cell_centroid(local_ordinal_type local_id) const
    {
        return cell_centroid(cell_id(local_id));
    }

    auto cell_faces(cell_id_t cell_id) const
    {
        derived().check_cell_id(cell_id);
        return derived().cell_faces_impl(cell_id);
    }

    std::vector<local_ordinal_type>
    cell_faces(local_ordinal_type local_id) const
    {
        const auto structured_faces = cell_faces(cell_id(local_id));
        std::vector<local_ordinal_type> local_faces;
        local_faces.reserve(structured_faces.size());
        for (const auto structured_face : structured_faces)
        {
            local_faces.push_back(face_local_id(structured_face));
        }
        return local_faces;
    }

    auto faces(cell_id_t cell_id) const
    {
        return cell_faces(cell_id);
    }

    std::vector<local_ordinal_type> faces(local_ordinal_type local_id) const
    {
        return cell_faces(local_id);
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

    std::vector<real_t> face_distances(local_ordinal_type local_id) const
    {
        const auto cell_face_ids = faces(local_id);
        std::vector<real_t> distances;
        distances.reserve(cell_face_ids.size());
        for (const auto face_local_id : cell_face_ids)
        {
            distances.push_back(
                cell_to_face_distance(face_local_id, local_id));
        }
        return distances;
    }

    cell_id_t owner_cell(face_id_t face_id) const
    {
        derived().check_face_id(face_id);
        return derived().owner_cell_impl(face_id);
    }

    local_ordinal_type owner_cell(local_ordinal_type local_id) const
    {
        return cell_local_id(owner_cell(face_id(local_id)));
    }

    cell_id_t neighbor_cell(face_id_t face_id) const
    {
        derived().check_face_id(face_id);
        return derived().neighbor_cell_impl(face_id);
    }

    local_ordinal_type neighbor_cell(local_ordinal_type local_id) const
    {
        return cell_local_id(neighbor_cell(face_id(local_id)));
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

    local_ordinal_type opposite_cell(
        local_ordinal_type face_local_id,
        local_ordinal_type cell_local_id) const
    {
        return this->cell_local_id(
            opposite_cell(face_id(face_local_id), cell_id(cell_local_id)));
    }

    cell_id_t opposite_or_periodic_neighbor_cell(
        face_id_t face_id,
        cell_id_t cell_id) const
    {
        return opposite_cell(face_id, cell_id);
    }

    local_ordinal_type opposite_or_periodic_neighbor_cell(
        local_ordinal_type face_local_id,
        local_ordinal_type cell_local_id) const
    {
        return this->cell_local_id(opposite_or_periodic_neighbor_cell(
            face_id(face_local_id), cell_id(cell_local_id)));
    }

    real_t face_area(face_id_t face_id) const
    {
        derived().check_face_id(face_id);
        return derived().face_area_impl(face_id);
    }

    real_t face_area(local_ordinal_type local_id) const
    {
        return face_area(face_id(local_id));
    }

    Vec3 face_centroid(face_id_t face_id) const
    {
        derived().check_face_id(face_id);
        return derived().face_centroid_impl(face_id);
    }

    Vec3 face_centroid(local_ordinal_type local_id) const
    {
        return face_centroid(face_id(local_id));
    }

    Vec3 face_normal(face_id_t face_id) const
    {
        derived().check_face_id(face_id);
        return derived().face_normal_impl(face_id);
    }

    Vec3 face_normal(local_ordinal_type local_id) const
    {
        return face_normal(face_id(local_id));
    }

    Vec3 face_area_vector(face_id_t face_id) const
    {
        return face_normal(face_id) * face_area(face_id);
    }

    Vec3 face_area_vector(local_ordinal_type local_id) const
    {
        return face_area_vector(face_id(local_id));
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

    Vec3 face_normal_outward(
        local_ordinal_type face_local_id,
        local_ordinal_type cell_local_id) const
    {
        return face_normal_outward(
            face_id(face_local_id), cell_id(cell_local_id));
    }

    Vec3 face_area_vector_outward(
        face_id_t face_id,
        cell_id_t cell_id) const
    {
        return face_normal_outward(face_id, cell_id) * face_area(face_id);
    }

    Vec3 face_area_vector_outward(
        local_ordinal_type face_local_id,
        local_ordinal_type cell_local_id) const
    {
        return face_area_vector_outward(
            face_id(face_local_id), cell_id(cell_local_id));
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

    real_t face_cell_center_distance(local_ordinal_type local_id) const
    {
        return face_cell_center_distance(face_id(local_id));
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

    Vec3 cell_center_vector(
        local_ordinal_type face_local_id,
        local_ordinal_type cell_local_id) const
    {
        return cell_center_vector(
            face_id(face_local_id), cell_id(cell_local_id));
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

    real_t cell_to_face_distance(
        local_ordinal_type face_local_id,
        local_ordinal_type cell_local_id) const
    {
        return cell_to_face_distance(
            face_id(face_local_id), cell_id(cell_local_id));
    }

    bool is_exterior_face(face_id_t face_id) const
    {
        return neighbor_cell(face_id) == derived().invalid_cell_id();
    }

    bool is_exterior_face(local_ordinal_type local_id) const
    {
        return is_exterior_face(face_id(local_id));
    }

    bool is_interior_face(face_id_t face_id) const
    {
        return !is_exterior_face(face_id);
    }

    bool is_interior_face(local_ordinal_type local_id) const
    {
        return is_interior_face(face_id(local_id));
    }

    bool is_boundary_face(face_id_t face_id) const
    {
        return is_exterior_face(face_id)
            && boundary_id(face_id) != invalid_boundary_id;
    }

    bool is_boundary_face(local_ordinal_type local_id) const
    {
        return is_boundary_face(face_id(local_id));
    }

    int boundary_id(face_id_t face_id) const
    {
        derived().check_face_id(face_id);
        return derived().boundary_id_impl(face_id);
    }

    int boundary_id(local_ordinal_type local_id) const
    {
        return boundary_id(face_id(local_id));
    }

    const std::string& boundary_name(face_id_t face_id) const
    {
        const auto patch_id = boundary_id(face_id);
        if (patch_id == invalid_boundary_id)
        {
            throw std::out_of_range(
                "Requested face is not a boundary face.");
        }
        return boundary_patch_name(patch_id);
    }

    const std::string& boundary_name(local_ordinal_type local_id) const
    {
        return boundary_name(face_id(local_id));
    }

    const std::string& boundary_patch_name(int patch_id) const
    {
        return derived().boundary_patch_name_impl(patch_id);
    }

    auto boundary_face_patch(int patch_id) const
    {
        return derived().boundary_face_patch_impl(patch_id);
    }

    auto boundary_patch_ids() const
    {
        return derived().boundary_patch_ids_impl();
    }

    int num_boundary_patches() const noexcept
    {
        return derived().num_boundary_patches_impl();
    }

    auto boundary_patches() const
    {
        return derived().boundary_patches_impl();
    }

    Vec3 node_coordinates(node_id_t node_id) const
    {
        derived().check_node_id(node_id);
        return derived().node_coordinates_impl(node_id);
    }

    Vec3 node_coordinates(local_ordinal_type local_id) const
    {
        return node_coordinates(node_id(local_id));
    }

    Vec3 node_coord(node_id_t node_id) const
    {
        return node_coordinates(node_id);
    }

    Vec3 node_coord(local_ordinal_type local_id) const
    {
        return node_coordinates(local_id);
    }

protected:
    constexpr MeshBase() = default;

    const uint8_t d_dim = 3;
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
concept MeshClass = std::derived_from<DerivedMesh, 
                                      MeshBase<DerivedMesh, 
                                               typename DerivedMesh::cell_id_t, 
                                               typename DerivedMesh::face_id_t, 
                                               typename DerivedMesh::node_id_t>>;

} // namespace SimpleFluid
