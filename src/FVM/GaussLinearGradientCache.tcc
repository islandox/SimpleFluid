/**
 * @file GaussLinearGradientCache.tcc
 * @brief  Template implementations for GaussLinearGradientCache.
 */
#pragma once

#include "FVM/GaussLinearGradientCache.hh"

#include <array>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace SimpleFluid::FVM
{

template<TpetraTypePack Pack, class MeshType>
GaussLinearGradientCache<Pack, MeshType>::GaussLinearGradientCache(SP<const mesh_type> mesh)
    : d_mesh(std::move(mesh))
{
    if (!d_mesh)
        throw std::invalid_argument("GaussLinearGradientCache requires a non-null mesh.");
    d_traverse_plan = std::make_shared<const traverse_plan_type>(
        *d_mesh, MeshTraverseEntities::AllWithCellFaces);
    d_boundary_locations = detail::boundary_face_locations(*d_mesh);
    refresh();
}

template<TpetraTypePack Pack, class MeshType>
void GaussLinearGradientCache<Pack, MeshType>::require_mesh(const mesh_type& mesh) const
{
    if (&mesh != d_mesh.get())
        throw std::invalid_argument("Gauss-linear gradient cache belongs to another mesh.");
    d_traverse_plan->require_valid();
    if (mesh_geometry_epoch(mesh) != d_geometry_epoch)
        throw std::invalid_argument("Gauss-linear gradient cache is stale for the mesh geometry epoch.");
}

template<TpetraTypePack Pack, class MeshType>
const typename GaussLinearGradientCache<Pack, MeshType>::Geometry&
GaussLinearGradientCache<Pack, MeshType>::geometry() const
{
    require_mesh(*d_mesh);
    if (d_snapshot->structured)
        std::call_once(d_snapshot->legacy_once, [&]
        {
            const auto& structured = *d_snapshot->structured;
            Geometry geometry;
            geometry.volumes = structured.volumes;
            geometry.offsets.reserve(structured.volumes.size() + 1);
            geometry.faces.reserve(structured.volumes.size() * 6);
            for (size_t owned = 0; owned < structured.volumes.size(); ++owned)
            {
                geometry.offsets.push_back(geometry.faces.size());
                const auto row = structured.rows[owned];
                if (row & 1)
                {
                    const auto& faces = structured.boundary[row >> 1];
                    geometry.faces.insert(geometry.faces.end(), faces.begin(), faces.end());
                }
                else
                {
                    const auto& metrics = structured.interior[row >> 1];
                    const auto faces = structured.layout->cell_faces(owned);
                    for (size_t slot = 0; slot < 6; ++slot)
                    {
                        FaceGeometry face;
                        face.face_lid = static_cast<local_ordinal_type>(faces[slot].face);
                        face.other_lid = metrics.other_lids[slot];
                        face.area = metrics.areas[slot];
                        face.cell_distance = metrics.cell_distances[slot];
                        face.other_distance = metrics.other_distances[slot];
                        face.interior = true;
                        geometry.faces.push_back(face);
                    }
                }
            }
            geometry.offsets.push_back(geometry.faces.size());
            d_snapshot->geometry = std::move(geometry);
            d_snapshot->legacy_materialized.store(true, std::memory_order_release);
        });
    return d_snapshot->geometry;
}

template<TpetraTypePack Pack, class MeshType>
size_t GaussLinearGradientCache<Pack, MeshType>::numerical_storage_bytes() const noexcept
{
    auto bytes = sizeof(Snapshot) + d_boundary_locations.capacity() * sizeof(boundary_location_type);
    if (const auto* geometry = structured_geometry())
        bytes += sizeof(StructuredGeometry) + geometry->volumes.capacity() * sizeof(real_t)
            + geometry->rows.capacity() * sizeof(size_t)
            + geometry->interior.capacity() * sizeof(StructuredInteriorRow)
            + geometry->boundary.capacity() * sizeof(std::array<FaceGeometry, 6>);
    else
    {
        const auto& legacy = d_snapshot->geometry;
        bytes += legacy.offsets.capacity() * sizeof(size_t) + legacy.volumes.capacity() * sizeof(real_t)
            + legacy.faces.capacity() * sizeof(FaceGeometry);
    }
    return bytes;
}

template<TpetraTypePack Pack, class MeshType>
void GaussLinearGradientCache<Pack, MeshType>::refresh()
{
    const auto& mesh = *d_mesh;
    const auto traverse = d_traverse_plan->acquire();
    const auto& locations = d_boundary_locations;
    auto snapshot = std::make_shared<Snapshot>();
    auto& geometry = snapshot->geometry;
    const auto count = mesh.num_owned_cells();
    if (d_traverse_plan->uses_structured_ranges())
    {
        snapshot->structured = std::make_unique<StructuredGeometry>();
        auto& structured = *snapshot->structured;
        structured.volumes.reserve(count);
        structured.rows.reserve(count);
        structured.interior.reserve(traverse.cells().interior().size());
        structured.boundary.reserve(traverse.cells().boundary().size());
    }
    else
    {
        geometry.offsets.reserve(count + 1);
        geometry.volumes.reserve(count);
        geometry.faces.reserve(count * 6);
    }
    traverse.cells().apply([&](const auto& cell)
    {
        using cell_type = std::remove_cvref_t<decltype(cell)>;
        const auto cell_lid = cell.id();
        const auto* previous_boundary = static_cast<const std::array<FaceGeometry, 6>*>(nullptr);
        if (snapshot->structured)
        {
            auto& structured = *snapshot->structured;
            if (!structured.layout) structured.layout = cell.geometry().structured_layout();
            structured.volumes.push_back(cell.volume());
            if constexpr (cell_type::is_interior)
            {
                structured.rows.push_back(structured.interior.size() << 1);
                structured.interior.emplace_back();
            }
            else
            {
                structured.rows.push_back((structured.boundary.size() << 1) | 1);
                structured.boundary.emplace_back();
                if (d_snapshot && d_snapshot->structured)
                    previous_boundary = &d_snapshot->structured->boundary[
                        d_snapshot->structured->rows[static_cast<size_t>(cell_lid)] >> 1];
            }
        }
        else
        {
            geometry.offsets.push_back(geometry.faces.size());
            geometry.volumes.push_back(cell.volume());
        }
        size_t slot = 0;
        traverse.cell_faces(cell_lid).all().apply(cell.geometry(), [&](const auto& incidence)
        {
            using incidence_type = std::remove_cvref_t<decltype(incidence)>;
            using access_type = std::remove_cvref_t<decltype(incidence.geometry())>;
            const auto& access = incidence.geometry();
            FaceGeometry face;
            face.face_lid = incidence.id();
            face.area = access.face_area_vector_outward(face.face_lid, cell_lid);
            face.interior = incidence_type::is_interior;
            if constexpr (incidence_type::is_interior)
            {
                face.other_lid = incidence.orientation() > 0 ? incidence.neighbor() : incidence.owner();
                if (face.other_lid == access_type::invalid_local_id())
                    throw std::invalid_argument("Gauss-linear gradient cache requires adjacent cells in the local map.");
                face.cell_distance = access.cell_to_face_distance(face.face_lid, cell_lid);
                face.other_distance = access.cell_to_face_distance(face.face_lid, face.other_lid);
            }
            else if (incidence.boundary_id() != mesh_type::invalid_boundary_id)
            {
                face.boundary = previous_boundary ? (*previous_boundary)[slot].boundary
                    : locations.at(static_cast<size_t>(face.face_lid));
                if (face.boundary.active)
                {
                    const auto direction = access.face_centroid(face.face_lid) - access.cell_centroid(cell_lid);
                    face.normal_distance = static_cast<typename mesh_type::scalar_type>(
                        direction.dot(access.face_normal_outward(face.face_lid, cell_lid)));
                }
            }
            if (snapshot->structured)
            {
                if constexpr (cell_type::is_interior)
                {
                    auto& row = snapshot->structured->interior.back();
                    row.areas[slot] = face.area;
                    row.cell_distances[slot] = face.cell_distance;
                    row.other_distances[slot] = face.other_distance;
                    row.other_lids[slot] = face.other_lid;
                }
                else
                    snapshot->structured->boundary.back()[slot] = face;
            }
            else
                geometry.faces.push_back(face);
            ++slot;
        });
    });
    if (!snapshot->structured)
    {
        geometry.offsets.push_back(geometry.faces.size());
        snapshot->legacy_materialized.store(true, std::memory_order_relaxed);
    }
    const auto geometry_epoch = mesh_geometry_epoch(mesh);
    d_snapshot = std::move(snapshot);
    d_geometry_epoch = geometry_epoch;
    // Structured boundary rows already retain the topology-only batch locations.
    // Reuse those on refresh instead of retaining one descriptor per mesh face.
    if (d_snapshot->structured)
        std::vector<boundary_location_type>().swap(d_boundary_locations);
}

} // namespace SimpleFluid::FVM
