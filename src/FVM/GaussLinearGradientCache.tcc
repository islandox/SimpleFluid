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
    if (d_snapshot->layered)
        std::call_once(d_snapshot->legacy_once, [&]
        {
            const auto& layered = *d_snapshot->layered;
            const auto& topology = *layered.topology;
            Geometry geometry;
            geometry.volumes = layered.volumes;
            geometry.offsets.reserve(layered.volumes.size() + 1);
            geometry.faces.reserve(layered.areas.size() + layered.boundary.size());
            for (size_t owned = 0; owned < layered.volumes.size(); ++owned)
            {
                geometry.offsets.push_back(geometry.faces.size());
                const auto row = topology.rows[owned];
                if (row & 1)
                {
                    for (size_t entry = topology.boundary_offsets[row >> 1];
                         entry < topology.boundary_offsets[(row >> 1) + 1]; ++entry)
                    {
                        const auto& metadata = topology.boundary[entry];
                        const auto& metrics = layered.boundary[entry];
                        geometry.faces.push_back({metadata.face_lid, metadata.other_lid,
                            metrics.area, metrics.cell_distance, metrics.other_distance,
                            metrics.normal_distance, metadata.boundary, metadata.interior});
                    }
                }
                else
                {
                    const auto faces = topology.plan->layered_layout()->cell_faces(owned);
                    const auto begin = topology.interior_offsets[row >> 1];
                    const auto end = topology.interior_offsets[(row >> 1) + 1];
                    for (size_t entry = begin; entry < end; ++entry)
                    {
                        FaceGeometry face;
                        face.face_lid = static_cast<local_ordinal_type>(faces.face_ordinal(entry - begin));
                        face.other_lid = topology.other_lids[entry];
                        face.area = layered.areas[entry];
                        face.cell_distance = layered.cell_distances[entry];
                        face.other_distance = layered.other_distances[entry];
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
    else if (const auto* geometry = layered_geometry())
        bytes += sizeof(LayeredGeometry) + geometry->topology->storage_bytes()
            + geometry->volumes.capacity() * sizeof(real_t)
            + geometry->areas.capacity() * sizeof(typename mesh_type::Vec3)
            + geometry->cell_distances.capacity() * sizeof(real_t)
            + geometry->other_distances.capacity() * sizeof(real_t)
            + geometry->boundary.capacity() * sizeof(typename LayeredGeometry::BoundaryMetric);
    else
    {
        const auto& legacy = d_snapshot->geometry;
        bytes += legacy.offsets.capacity() * sizeof(size_t) + legacy.volumes.capacity() * sizeof(real_t)
            + legacy.faces.capacity() * sizeof(FaceGeometry);
    }
    return bytes;
}

template<TpetraTypePack Pack, class MeshType>
void GaussLinearGradientCache<Pack, MeshType>::build_layered_geometry(
    LayeredGeometry& geometry, const MeshTraverseExecution<mesh_type>& traversal) const
{
    auto topology = std::make_shared<LayeredTopology>();
    topology->plan = d_traverse_plan;
    const auto& layout = *d_traverse_plan->layered_layout();
    using Selection = ::SimpleFluid::detail::StructuredTraverseSelection;
    const size_t interior_payload = layout.cell_incidence_count(Selection::Interior);
    const size_t boundary_payload = layout.cell_incidence_count(Selection::All) - interior_payload;
    topology->rows.resize(d_mesh->num_owned_cells());
    topology->interior_offsets.reserve(traversal.cells().interior().size() + 1);
    topology->boundary_offsets.reserve(traversal.cells().boundary().size() + 1);
    topology->other_lids.reserve(interior_payload);
    topology->boundary.reserve(boundary_payload);
    geometry.volumes.resize(d_mesh->num_owned_cells());
    geometry.areas.reserve(interior_payload);
    geometry.cell_distances.reserve(interior_payload);
    geometry.other_distances.reserve(interior_payload);
    geometry.boundary.reserve(boundary_payload);
    const auto build_cell = [&](const auto& cell)
    {
        constexpr bool Interior = std::remove_cvref_t<decltype(cell)>::is_interior;
        const auto cell_lid = cell.id();
        geometry.volumes[static_cast<size_t>(cell_lid)] = cell.volume();
        if constexpr (Interior)
        {
            topology->rows[static_cast<size_t>(cell.id())] = topology->interior_offsets.size() << 1;
            topology->interior_offsets.push_back(topology->other_lids.size());
        }
        else
        {
            topology->rows[static_cast<size_t>(cell.id())] = (topology->boundary_offsets.size() << 1) | size_t{1};
            topology->boundary_offsets.push_back(topology->boundary.size());
        }
        traversal.cell_faces(cell).all().apply(cell.geometry(), [&](const auto& incidence)
        {
            constexpr bool FaceInterior = std::remove_cvref_t<decltype(incidence)>::is_interior;
            typename LayeredTopology::BoundaryFace face;
            typename LayeredGeometry::BoundaryMetric metric;
            const auto& access = incidence.geometry();
            face.face_lid = incidence.id();
            face.interior = FaceInterior;
            metric.area = access.face_area_vector_outward(face.face_lid, cell_lid);
            if constexpr (FaceInterior)
            {
                using access_type = std::remove_cvref_t<decltype(incidence.geometry())>;
                face.other_lid = incidence.orientation() > 0 ? incidence.neighbor() : incidence.owner();
                if (face.other_lid == access_type::invalid_local_id())
                    throw std::invalid_argument("Gauss-linear gradient cache requires adjacent cells in the local map.");
                metric.cell_distance = access.cell_to_face_distance(face.face_lid, cell_lid);
                metric.other_distance = access.cell_to_face_distance(face.face_lid, face.other_lid);
            }
            else if (incidence.boundary_id() != mesh_type::invalid_boundary_id)
            {
                face.boundary = d_boundary_locations.at(static_cast<size_t>(face.face_lid));
                if (face.boundary.active)
                {
                    const auto direction = access.face_centroid(face.face_lid) - access.cell_centroid(cell_lid);
                    metric.normal_distance = static_cast<typename mesh_type::scalar_type>(
                        direction.dot(access.face_normal_outward(face.face_lid, cell_lid)));
                }
            }
            if constexpr (Interior)
            {
                topology->other_lids.push_back(face.other_lid);
                geometry.areas.push_back(metric.area);
                geometry.cell_distances.push_back(metric.cell_distance);
                geometry.other_distances.push_back(metric.other_distance);
            }
            else
            {
                topology->boundary.push_back(face);
                geometry.boundary.push_back(metric);
            }
        });
    };
    traversal.cells().interior().apply(build_cell);
    traversal.cells().boundary().apply(build_cell);
    topology->interior_offsets.push_back(topology->other_lids.size());
    topology->boundary_offsets.push_back(topology->boundary.size());
    geometry.topology = std::move(topology);
}

template<TpetraTypePack Pack, class MeshType>
void GaussLinearGradientCache<Pack, MeshType>::fill_layered_geometry(LayeredGeometry& geometry) const
{
    const auto& topology = *geometry.topology;
    geometry.volumes.resize(topology.rows.size());
    geometry.areas.reserve(topology.other_lids.size());
    geometry.cell_distances.reserve(topology.other_lids.size());
    geometry.other_distances.reserve(topology.other_lids.size());
    geometry.boundary.reserve(topology.boundary.size());
    ::SimpleFluid::detail::with_traverse_access(*d_mesh, [&](const auto& source)
    {
        using source_type = std::remove_cvref_t<decltype(source)>;
        if constexpr (::SimpleFluid::detail::has_layered_traverse_layout<typename source_type::geometry_type>
            && (::SimpleFluid::detail::traverse_is_handle<mesh_type>
                || std::is_same_v<mesh_type, typename source_type::geometry_type>))
        {
            const ::SimpleFluid::detail::LayeredTraverseAccess<source_type> access(source);
            const auto& layout = *topology.plan->layered_layout();
            // The retained XY selections are disjoint, so metric refill never
            // classifies a cell or rediscovers its adjacency. Each selection
            // preserves its topology row order, so append fills retained offsets
            // without value-initializing and then overwriting metric arrays.
            const auto fill_cells = [&]<bool Interior>()
            {
                constexpr auto selection = Interior
                    ? ::SimpleFluid::detail::StructuredTraverseSelection::Interior
                    : ::SimpleFluid::detail::StructuredTraverseSelection::Boundary;
                layout.visit_cells(selection, false, [&](const auto& run)
                {
                    for (size_t owned = run.begin; owned < run.end; ++owned)
                    {
                        const auto cell_lid = static_cast<local_ordinal_type>(owned);
                        geometry.volumes[owned] = access.cell_volume(owned);
                        const auto row = topology.rows[owned] >> 1;
                        if constexpr (Interior)
                        {
                            const auto faces = layout.cell_faces(owned);
                            const auto begin = topology.interior_offsets[row];
                            const auto end = topology.interior_offsets[row + 1];
                            for (size_t entry = begin; entry < end; ++entry)
                            {
                                const auto face_lid = faces.face_ordinal(entry - begin);
                                geometry.areas.push_back(access.face_area_vector_outward(face_lid, cell_lid));
                                geometry.cell_distances.push_back(access.cell_to_face_distance(face_lid, cell_lid));
                                geometry.other_distances.push_back(access.cell_to_face_distance(face_lid, topology.other_lids[entry]));
                            }
                        }
                        else
                            for (size_t entry = topology.boundary_offsets[row];
                                 entry < topology.boundary_offsets[row + 1]; ++entry)
                            {
                                const auto& metadata = topology.boundary[entry];
                                typename LayeredGeometry::BoundaryMetric face;
                                face.area = access.face_area_vector_outward(metadata.face_lid, cell_lid);
                                if (metadata.interior)
                                {
                                    face.cell_distance = access.cell_to_face_distance(metadata.face_lid, cell_lid);
                                    face.other_distance = access.cell_to_face_distance(metadata.face_lid, metadata.other_lid);
                                }
                                else if (metadata.boundary.active)
                                {
                                    const auto direction = access.face_centroid(metadata.face_lid) - access.cell_centroid(cell_lid);
                                    face.normal_distance = static_cast<typename mesh_type::scalar_type>(
                                        direction.dot(access.face_normal_outward(metadata.face_lid, cell_lid)));
                                }
                                geometry.boundary.push_back(face);
                            }
                    }
                });
            };
            fill_cells.template operator()<true>();
            fill_cells.template operator()<false>();
        }
        else
            throw std::logic_error("Layered gradient plan lost its canonical provider.");
    });
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
    if (d_traverse_plan->uses_layered_ranges())
    {
        snapshot->layered = std::make_unique<LayeredGeometry>();
        auto& layered = *snapshot->layered;
        if (d_snapshot && d_snapshot->layered)
        {
            layered.topology = d_snapshot->layered->topology;
            fill_layered_geometry(layered);
        }
        else
            build_layered_geometry(layered, traverse);
        const auto geometry_epoch = mesh_geometry_epoch(mesh);
        d_snapshot = std::move(snapshot);
        d_geometry_epoch = geometry_epoch;
        std::vector<boundary_location_type>().swap(d_boundary_locations);
        return;
    }
    if constexpr (requires { mesh.supports_region_execution(); })
    {
        if (mesh.supports_region_execution())
        {
            geometry.offsets.reserve(count + 1);
            geometry.volumes.reserve(count);
            geometry.faces.reserve(count * 6);
            // Borrow each region under the active lease and resolve its logical
            // incidences once; the snapshot retains only the existing metrics.
            mesh.visit_owned_region_cells([&](local_ordinal_type, auto canonical_cell,
                                             auto native_cell, const auto& region)
            {
                geometry.offsets.push_back(geometry.faces.size());
                geometry.volumes.push_back(region.cell_volume(native_cell));
                region.visit_cell_faces(native_cell, [&](const auto& resolved)
                {
                    FaceGeometry face;
                    face.face_lid = mesh.region_face_local_id(resolved.face);
                    // Match the retained traversal's visible-incidence filter.
                    if (face.face_lid == mesh.invalid_local_id()) return;
                    face.area = resolved.face_area_vector_outward(canonical_cell);
                    face.interior = resolved.interior();
                    if (face.interior)
                    {
                        const auto other = resolved.opposite_cell(canonical_cell);
                        face.other_lid = mesh.region_cell_local_id(other);
                        if (face.other_lid == mesh.invalid_local_id())
                            throw std::invalid_argument(
                                "Gauss-linear gradient cache requires adjacent cells in the local map.");
                        face.cell_distance = resolved.face_center_vector(canonical_cell).norm();
                        face.other_distance = resolved.face_center_vector(other).norm();
                    }
                    else
                    {
                        face.boundary = locations.at(static_cast<size_t>(face.face_lid));
                        if (face.boundary.active)
                        {
                            // Physical boundary samples use the same raw centroid
                            // subtraction and outward-normal arithmetic as before.
                            const auto direction = resolved.centroid - resolved.owner_centroid;
                            face.normal_distance = static_cast<typename mesh_type::scalar_type>(
                                direction.dot(resolved.face_normal_outward(canonical_cell)));
                        }
                    }
                    geometry.faces.push_back(face);
                });
            });
            geometry.offsets.push_back(geometry.faces.size());
            snapshot->legacy_materialized.store(true, std::memory_order_relaxed);
            const auto geometry_epoch = mesh_geometry_epoch(mesh);
            d_snapshot = std::move(snapshot);
            d_geometry_epoch = geometry_epoch;
            return;
        }
    }
    if (d_traverse_plan->uses_structured_ranges())
    {
        snapshot->structured = std::make_unique<StructuredGeometry>();
        auto& structured = *snapshot->structured;
        structured.volumes.resize(count);
        structured.rows.resize(count);
        structured.interior.reserve(traverse.cells().interior().size());
        structured.boundary.reserve(traverse.cells().boundary().size());
        const auto build_cell = [&](const auto& cell)
        {
            constexpr bool Interior = std::remove_cvref_t<decltype(cell)>::is_interior;
            const auto cell_lid = cell.id();
            const auto owned = static_cast<size_t>(cell_lid);
            const auto* previous_boundary = static_cast<const std::array<FaceGeometry, 6>*>(nullptr);
            if (!structured.layout) structured.layout = cell.geometry().structured_layout();
            structured.volumes[owned] = cell.volume();
            if constexpr (Interior)
            {
                structured.rows[owned] = structured.interior.size() << 1;
                structured.interior.emplace_back();
            }
            else
            {
                structured.rows[owned] = (structured.boundary.size() << 1) | size_t{1};
                structured.boundary.emplace_back();
                if (d_snapshot && d_snapshot->structured)
                    previous_boundary = &d_snapshot->structured->boundary[
                        d_snapshot->structured->rows[owned] >> 1];
            }
            size_t slot = 0;
            traverse.cell_faces(cell).all().apply(cell.geometry(), [&](const auto& incidence)
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
                if constexpr (Interior)
                {
                    auto& row = structured.interior.back();
                    row.areas[slot] = face.area;
                    row.cell_distances[slot] = face.cell_distance;
                    row.other_distances[slot] = face.other_distance;
                    row.other_lids[slot] = face.other_lid;
                }
                else
                    structured.boundary.back()[slot] = face;
                ++slot;
            });
        };
        traverse.cells().interior().apply(build_cell);
        traverse.cells().boundary().apply(build_cell);
        const auto geometry_epoch = mesh_geometry_epoch(mesh);
        d_snapshot = std::move(snapshot);
        d_geometry_epoch = geometry_epoch;
        // Boundary rows retain topology-only locations for metric refresh.
        std::vector<boundary_location_type>().swap(d_boundary_locations);
        return;
    }
    geometry.offsets.reserve(count + 1);
    geometry.volumes.reserve(count);
    geometry.faces.reserve(count * 6);
    traverse.cells().apply([&](const auto& cell)
    {
        const auto cell_lid = cell.id();
        geometry.offsets.push_back(geometry.faces.size());
        geometry.volumes.push_back(cell.volume());
        traverse.cell_faces(cell).all().apply(cell.geometry(), [&](const auto& incidence)
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
                face.boundary = locations.at(static_cast<size_t>(face.face_lid));
                if (face.boundary.active)
                {
                    const auto direction = access.face_centroid(face.face_lid) - access.cell_centroid(cell_lid);
                    face.normal_distance = static_cast<typename mesh_type::scalar_type>(
                        direction.dot(access.face_normal_outward(face.face_lid, cell_lid)));
                }
            }
            geometry.faces.push_back(face);
        });
    });
    geometry.offsets.push_back(geometry.faces.size());
    snapshot->legacy_materialized.store(true, std::memory_order_relaxed);
    const auto geometry_epoch = mesh_geometry_epoch(mesh);
    d_snapshot = std::move(snapshot);
    d_geometry_epoch = geometry_epoch;
}

} // namespace SimpleFluid::FVM
