/**
 * @file MeshGeometryHandle.ipp
 * @brief Inline definitions for the borrowed geometry view over MeshHandle.
 *
 * Included by geometry/MeshHandle.ipp after the MeshHandle class definition is
 * visible. The metric formulas are moved verbatim from that file: legacy STK,
 * serial identity, orthogonal compact topology, semi-structured, map-indexing,
 * and composite region branches, including all periodic-image arithmetic.
 *
 * @version 0.1
 * @date 2026-11-11
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <variant>

namespace SimpleFluid
{

template<TpetraTypePack Pack>
inline MeshGeometryHandle<Pack>::MeshGeometryHandle(const MeshHandle<Pack>& owner) noexcept
    : d_owner(&owner), d_binding(owner.binding_ptr())
{
}

template<TpetraTypePack Pack>
inline const detail::MeshBinding<Pack>& MeshGeometryHandle<Pack>::binding() const
{
    if (!d_binding)
        throw std::logic_error("MeshGeometryHandle has no published MeshBinding.");
    d_binding->validate(d_owner->d_layout_generation);
    return *d_binding;
}

template<TpetraTypePack Pack>
inline const typename MeshGeometryHandle<Pack>::variant_type& MeshGeometryHandle<Pack>::variant() const noexcept
{
    return d_owner->d_mesh;
}

template<TpetraTypePack Pack>
template<class Visitor>
inline decltype(auto) MeshGeometryHandle<Pack>::visit(Visitor&& visitor) const
{
    return std::visit(
        [&](const auto& mesh) -> decltype(auto)
        {
            return std::forward<Visitor>(visitor)(*mesh);
        },
        d_owner->d_mesh);
}

template<TpetraTypePack Pack>
template<class Visitor>
inline decltype(auto) MeshGeometryHandle<Pack>::visit_mutable(Visitor&& visitor)
{
    if (!d_owner->d_mutable_mesh)
    {
        throw std::logic_error("MeshHandle does not retain mutable geometry ownership.");
    }
    return std::visit(
        [&](auto& mesh) -> decltype(auto)
        {
            return std::forward<Visitor>(visitor)(*mesh);
        },
        *d_owner->d_mutable_mesh);
}

template<TpetraTypePack Pack>
inline bool MeshGeometryHandle<Pack>::has_mutable_geometry() const noexcept
{
    return d_owner->d_mutable_mesh.has_value();
}

template<TpetraTypePack Pack>
inline auto MeshGeometryHandle<Pack>::acquire_execution_view() const
{
    const auto* composite = std::get_if<typename MeshHandle<Pack>::MultiRegionPtr>(&d_owner->d_mesh);
    return Meshes::MultiRegionMesh::ExecutionView(composite ? composite->get() : nullptr);
}

template<TpetraTypePack Pack>
inline const void* MeshGeometryHandle<Pack>::geometry_identity() const noexcept
{
    return std::visit([](const auto& mesh) noexcept -> const void* { return mesh.get(); }, d_owner->d_mesh);
}

template<TpetraTypePack Pack>
inline std::uint64_t MeshGeometryHandle<Pack>::geometry_epoch() const
{
    return std::visit(
        [](const auto& mesh) -> std::uint64_t
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
        d_owner->d_mesh);
}

template<TpetraTypePack Pack>
inline real_t MeshGeometryHandle<Pack>::cell_volume(local_ordinal_type cell_lid) const
{
    return d_owner->visit_geometry_cell(
        cell_lid,
        [](const auto& mesh, const auto id)
        {
            return mesh.cell_volume(id);
        });
}

template<TpetraTypePack Pack>
inline auto MeshGeometryHandle<Pack>::cell_centroid(local_ordinal_type cell_lid) const -> Vec3
{
    return d_owner->visit_geometry_cell(
        cell_lid,
        [](const auto& mesh, const auto id)
        {
            return mesh.cell_centroid(id);
        });
}

template<TpetraTypePack Pack>
inline real_t MeshGeometryHandle<Pack>::face_area(local_ordinal_type face_lid) const
{
    return d_owner->visit_geometry_face(
        face_lid,
        [](const auto& mesh, const auto id)
        {
            return mesh.face_area(id);
        });
}

template<TpetraTypePack Pack>
inline auto MeshGeometryHandle<Pack>::face_centroid(local_ordinal_type face_lid) const -> Vec3
{
    return d_owner->visit_geometry_face(
        face_lid,
        [](const auto& mesh, const auto id)
        {
            return mesh.face_centroid(id);
        });
}

template<TpetraTypePack Pack>
inline auto MeshGeometryHandle<Pack>::face_normal(local_ordinal_type face_lid) const -> Vec3
{
    return d_owner->visit_geometry_face(
        face_lid,
        [](const auto& mesh, const auto id)
        {
            return mesh.face_normal(id);
        });
}

template<TpetraTypePack Pack>
inline auto MeshGeometryHandle<Pack>::face_area_vector(local_ordinal_type face_lid) const -> Vec3
{
    return face_normal(face_lid) * face_area(face_lid);
}

template<TpetraTypePack Pack>
inline auto MeshGeometryHandle<Pack>::face_normal_outward(
    local_ordinal_type face_lid,
    local_ordinal_type cell_lid) const -> Vec3
{
    const auto owner = d_owner->adjacent_cell(face_lid, true);
    if (cell_lid == owner)
    {
        return face_normal(face_lid);
    }
    if (cell_lid == d_owner->adjacent_cell(face_lid, false))
    {
        return face_normal(face_lid) * -1.0;
    }
    throw std::invalid_argument("Cell is not adjacent to requested face.");
}

template<TpetraTypePack Pack>
inline auto MeshGeometryHandle<Pack>::face_area_vector_outward(
    local_ordinal_type face_lid,
    local_ordinal_type cell_lid) const -> Vec3
{
    return face_normal_outward(face_lid, cell_lid) * face_area(face_lid);
}

template<TpetraTypePack Pack>
inline real_t MeshGeometryHandle<Pack>::face_cell_center_distance(local_ordinal_type face_lid) const
{
    const auto neighbor = d_owner->adjacent_cell(face_lid, false);
    if (neighbor == MeshHandle<Pack>::invalid_local_id())
    {
        return 0.0;
    }
    return cell_center_vector(face_lid, d_owner->adjacent_cell(face_lid, true)).norm();
}

template<TpetraTypePack Pack>
inline auto MeshGeometryHandle<Pack>::cell_center_vector(
    local_ordinal_type face_lid,
    local_ordinal_type cell_lid) const -> Vec3
{
    if (const auto legacy = d_owner->legacy_mesh())
        return legacy->cell_center_vector(
            static_cast<local_ordinal_type>(d_owner->geometry_face_lid(face_lid)),
            static_cast<local_ordinal_type>(d_owner->geometry_cell_lid(cell_lid)));
    if (const auto* cartesian = std::get_if<typename MeshHandle<Pack>::CartesianPtr>(&d_owner->d_mesh))
        return (*cartesian)->cell_center_vector(
            (*cartesian)->face_id(d_owner->geometry_face_lid(face_lid)),
            (*cartesian)->cell_id(d_owner->geometry_cell_lid(cell_lid)));
    const auto other = d_owner->topology().opposite_cell(face_lid, cell_lid);
    if (other == MeshHandle<Pack>::invalid_local_id())
    {
        throw std::invalid_argument("Exterior face does not have an opposite cell.");
    }
    if (const auto* composite = std::get_if<typename MeshHandle<Pack>::MultiRegionPtr>(&d_owner->d_mesh))
        return (*composite)->cell_center_vector(
            d_owner->geometry_face_lid(face_lid),
            d_owner->geometry_cell_lid(cell_lid));
    return cell_centroid(other) - cell_centroid(cell_lid);
}

/** @brief Cell-to-face displacement in the incident cell's periodic image. */
template<TpetraTypePack Pack>
inline auto MeshGeometryHandle<Pack>::face_center_vector(
    local_ordinal_type face,
    local_ordinal_type cell) const -> Vec3
{
    if (const auto* composite = std::get_if<typename MeshHandle<Pack>::MultiRegionPtr>(&d_owner->d_mesh))
        return (*composite)->face_center_vector(
            d_owner->geometry_face_lid(face),
            d_owner->geometry_cell_lid(cell));
    return face_centroid(face) - cell_centroid(cell);
}

template<TpetraTypePack Pack>
inline real_t MeshGeometryHandle<Pack>::cell_to_face_distance(
    local_ordinal_type face_lid,
    local_ordinal_type cell_lid) const
{
    if (const auto legacy = d_owner->legacy_mesh())
        return legacy->cell_to_face_distance(
            static_cast<local_ordinal_type>(d_owner->geometry_face_lid(face_lid)),
            static_cast<local_ordinal_type>(d_owner->geometry_cell_lid(cell_lid)));
    if (const auto* cartesian = std::get_if<typename MeshHandle<Pack>::CartesianPtr>(&d_owner->d_mesh))
        return (*cartesian)->cell_to_face_distance(
            (*cartesian)->face_id(d_owner->geometry_face_lid(face_lid)),
            (*cartesian)->cell_id(d_owner->geometry_cell_lid(cell_lid)));
    return face_center_vector(face_lid, cell_lid).norm();
}

} // namespace SimpleFluid
