/** @file MeshConcepts.hh
 * @brief Structural contract for the local-ordinal finite-volume mesh interface.
 */
#pragma once

#include "geometry/MeshUtils.hh"

#include <concepts>
#include <cstddef>
#include <ranges>

namespace SimpleFluid
{
/**
 * @brief Common const finite-volume queries on mesh-local ordinals.
 *
 * MeshHandle, PartitionedMesh, SolidSubdomain and legacy Mesh expose this
 * interface. Native geometry with family-specific entity IDs uses MeshClass
 * instead. Distributed maps and optional epochs/leases remain capabilities
 * required by individual consumers, rather than by this query contract.
 */
template<class M>
concept MeshType =
    requires {
        typename M::scalar_type;
        typename M::local_ordinal_type;
        typename M::global_ordinal_type;
    } && std::integral<typename M::local_ordinal_type> && std::integral<typename M::global_ordinal_type> &&
    requires(const M& mesh, typename M::local_ordinal_type cell, typename M::local_ordinal_type face) {
        { mesh.num_owned_cells() } -> std::convertible_to<size_t>;
        { mesh.num_local_cells() } -> std::convertible_to<size_t>;
        { mesh.num_faces() } -> std::convertible_to<size_t>;
        { mesh.faces(cell) } -> std::ranges::input_range;
        requires std::convertible_to<std::ranges::range_reference_t<decltype(mesh.faces(cell))>,
            typename M::local_ordinal_type>;
        { mesh.owner_cell(face) } -> std::convertible_to<typename M::local_ordinal_type>;
        { mesh.neighbor_cell(face) } -> std::convertible_to<typename M::local_ordinal_type>;
        { mesh.cell_volume(cell) } -> std::convertible_to<real_t>;
        { mesh.cell_centroid(cell) } -> std::convertible_to<MeshUtils::Vec3>;
        { mesh.face_area(face) } -> std::convertible_to<real_t>;
        { mesh.face_centroid(face) } -> std::convertible_to<MeshUtils::Vec3>;
        { mesh.face_area_vector(face) } -> std::convertible_to<MeshUtils::Vec3>;
    };
} // namespace SimpleFluid
