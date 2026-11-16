/** @file test_periodic_mesh_helpers.hh
 * @brief Canonical Cartesian periodic-seam queries for FVM test fixtures.
 */
#pragma once

#include "geometry/MeshHandle.hh"

#include <array>
#include <stdexcept>
#include <variant>

namespace SimpleFluid::test
{
/** @brief Identify the single logical-zero seam on a periodic Cartesian axis. */
template<TpetraTypePack Pack>
bool is_cartesian_periodic_face(const MeshHandle<Pack>& mesh, typename Pack::local_ordinal_type face)
{
    const auto* geometry = std::get_if<typename MeshHandle<Pack>::CartesianPtr>(&mesh.variant());
    if (!geometry)
        throw std::invalid_argument("Cartesian periodic test query requires a Cartesian mesh.");

    const auto& indexer = (*geometry)->indexer();
    const auto native_face = indexer.face_id(static_cast<size_t>(mesh.face_geometry_global_id(face)));
    const auto axis = static_cast<size_t>(native_face.orientation);
    return indexer.periodic_dimensions[axis] && std::array{native_face.i, native_face.j, native_face.k}[axis] == 0;
}
} // namespace SimpleFluid::test
