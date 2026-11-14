/** @file RegionProviderConcepts.hh
 * @brief Structural contracts for independently supplied region topology and geometry.
 */
#pragma once

#include "geometry/MeshUtils.hh"
#include "geometry/mesh/MeshIndexTypes.hh"

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <ranges>

namespace SimpleFluid::Meshes
{
/** @brief Ordinal layout and orientation compatibility; explicit templates require identical identity. */
struct RegionLayout
{
    enum class Family
    {
        Rectilinear,
        Extruded,
        Explicit,
        Cylindrical
    };
    Family family;
    size_t cells, faces, nodes;
    std::array<size_t, 3> extents;
    const void* template_identity = nullptr;
    std::array<bool, 3> periodic{};
    bool operator==(const RegionLayout&) const = default;
};

namespace region_provider_detail
{
template<class R>
concept RegionOrdinalRange =
    std::ranges::random_access_range<R> && std::ranges::sized_range<R> &&
    std::convertible_to<std::ranges::range_reference_t<R>, size_t> && requires(R&& range, size_t index) {
        { range.size() } -> std::convertible_to<size_t>;
        { range[index] } -> std::convertible_to<size_t>;
    };
} // namespace region_provider_detail

/** @brief Const, owner-oriented incidence in the ordinals declared by layout(). */
template<class T>
concept TopologyProvider =
    MeshIndexTypePack<typename T::index_type_pack> && requires(const T& topology, size_t cell, size_t face) {
        { topology.layout() } -> std::same_as<RegionLayout>;
        { topology.storage_identity() } -> std::same_as<const void*>;
        { topology.cell_faces(cell) } -> region_provider_detail::RegionOrdinalRange;
        { topology.owner_cell(face) } -> std::convertible_to<size_t>;
        { topology.neighbor_cell(face) } -> std::convertible_to<size_t>;
        { topology.cell_nodes(cell) } -> region_provider_detail::RegionOrdinalRange;
        { topology.face_nodes(face) } -> region_provider_detail::RegionOrdinalRange;
        { topology.cell_type(cell) } -> std::same_as<MeshUtils::CellType>;
    };

/** @brief Const physical metrics, storage identity and revision in a declared ordinal layout. */
template<class G>
concept GeometryProvider = requires(const G& geometry, size_t cell, size_t face, size_t node) {
    { geometry.layout() } -> std::same_as<RegionLayout>;
    { geometry.storage_identity() } -> std::same_as<const void*>;
    { geometry.cell_volume(cell) } -> std::convertible_to<real_t>;
    { geometry.cell_centroid(cell) } -> std::same_as<MeshUtils::Vec3>;
    { geometry.face_area(face) } -> std::convertible_to<real_t>;
    { geometry.face_centroid(face) } -> std::same_as<MeshUtils::Vec3>;
    { geometry.face_area_vector(face) } -> std::same_as<MeshUtils::Vec3>;
    { geometry.node_coordinates(node) } -> std::same_as<MeshUtils::Vec3>;
    { geometry.geometry_epoch() } -> std::convertible_to<std::uint64_t>;
};
} // namespace SimpleFluid::Meshes
