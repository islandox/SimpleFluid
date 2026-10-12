/**
 * @file FVM/details/TransportAssemblyGeometry.hh
 * @brief Ordered host topology and volumes for repeated transport assembly.
 */
#pragma once

#include "FVM/details/OperatorDetails.hh"
#include "FVM/details/ResolvedTransportGeometry.hh"
#include "FVM/details/TransportValidation.hh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimpleFluid::FVM::detail
{

/** Geometry-independent runs retain the original row/incidence and boundary-slot order. */
template<class MeshType> struct TransportAssemblyBatches
{
    using local_ordinal_type = typename MeshType::local_ordinal_type;
    struct Run { size_t begin{}, end{}; unsigned category{}; };
    struct BoundaryEntry { local_ordinal_type face_lid{}, owner{}; size_t in_batch_id{}; };
    struct BoundaryBatch { int id{}; std::vector<BoundaryEntry> faces; };
    std::vector<size_t> row_offsets;
    std::vector<Run> runs;
    std::vector<BoundaryBatch> boundary;
};

/** Geometry-epoch-owned data; face entries retain the mesh's row traversal order. */
template<class MeshType> struct TransportAssemblyGeometry
{
    using local_ordinal_type = typename MeshType::local_ordinal_type;
    using plan_type = MeshTraversePlan<MeshType>;
    using batches_type = TransportAssemblyBatches<MeshType>;
    struct Face
    {
        local_ordinal_type face_lid{};
        local_ordinal_type other{};
        bool interior = false;
        bool boundary = false;
        bool owned_orientation = false;
    };

    std::vector<typename MeshType::scalar_type> volumes;
    std::vector<size_t> face_offsets;
    std::vector<Face> faces;
    std::vector<bool> physical_boundary_faces;
    std::map<std::pair<int, size_t>, size_t> boundary_indices;
    std::shared_ptr<const plan_type> traverse_plan;
    std::shared_ptr<const batches_type> batches;
    std::uint64_t epoch{};

    void require_mesh(const MeshType& mesh) const
    {
        if (!traverse_plan || &traverse_plan->mesh() != &mesh || !batches)
            throw std::invalid_argument("Transport assembly geometry belongs to another mesh or is incomplete.");
        traverse_plan->require_valid();
        if (epoch != mesh_geometry_epoch(mesh))
            throw std::invalid_argument("Transport assembly geometry is stale for the mesh geometry epoch.");
    }
};

/** A face category and orientation selected once before a contiguous run. */
template<class Local, bool Interior, bool Boundary, bool Owned>
struct TransportAssemblyFace
{
    Local face_lid{}, other{};
    static constexpr bool interior = Interior;
    static constexpr bool boundary = Boundary;
    static constexpr bool owned_orientation = Owned;
};

template<class MeshType>
std::shared_ptr<const TransportAssemblyBatches<MeshType>> make_transport_assembly_batches(
    const MeshType& mesh, const TransportAssemblyGeometry<MeshType>& geometry)
{
    using batches_type = TransportAssemblyBatches<MeshType>;
    auto result = std::make_shared<batches_type>();
    result->row_offsets.reserve(geometry.volumes.size() + 1);
    for (size_t cell = 0; cell < geometry.volumes.size(); ++cell)
    {
        result->row_offsets.push_back(result->runs.size());
        const auto first = geometry.face_offsets[cell];
        for (size_t index = first; index < geometry.face_offsets[cell + 1]; ++index)
        {
            const auto& face = geometry.faces[index];
            const unsigned category = (face.interior ? 0U : (face.boundary ? 2U : 4U))
                                      + (face.owned_orientation ? 0U : 1U);
            if (index != first && result->runs.back().category == category)
                result->runs.back().end = index + 1;
            else result->runs.push_back({index, index + 1, category});
        }
    }
    result->row_offsets.push_back(result->runs.size());
    // These descriptors retain the legacy separate boundary sweep's order,
    // including original in-batch positions after periodic/ownership filtering.
    const auto append_boundary = [&](int id, const auto& faces)
    {
        typename batches_type::BoundaryBatch output{id, {}};
        size_t slot = 0;
        for (const auto face : faces)
        {
            if (mesh.is_owned_face(face) && mesh.is_boundary_face(face))
                output.faces.push_back({static_cast<typename MeshType::local_ordinal_type>(packed_face_local_id(mesh, face)),
                    packed_cell_local_id(mesh, mesh.owner_cell(face)), slot});
            ++slot;
        }
        if (!output.faces.empty()) result->boundary.push_back(std::move(output));
    };
    if constexpr (std::ranges::range<decltype(mesh.boundary_face_batch(0))>)
        for (int id : mesh.boundary_batch_ids()) append_boundary(id, mesh.boundary_face_batch(id));
    else
        for (const auto& [id, batch] : mesh.boundary_batches()) append_boundary(id, batch.face_lids);
    return result;
}

template<class MeshType>
TransportAssemblyGeometry<MeshType> transport_assembly_geometry(const MeshType& mesh,
    std::shared_ptr<const MeshTraversePlan<MeshType>> plan,
    const TransportAssemblyGeometry<MeshType>* previous = nullptr)
{
    if (!plan || &plan->mesh() != &mesh || plan->entities() != MeshTraverseEntities::AllWithCellFaces)
        throw std::invalid_argument("Transport assembly requires cell-face traversal for the same mesh.");
    const auto execution = plan->acquire();
    using geometry_type = TransportAssemblyGeometry<MeshType>;
    using local_ordinal_type = typename MeshType::local_ordinal_type;
    geometry_type result;
    result.traverse_plan = std::move(plan);
    if (previous)
    {
        if (previous->traverse_plan != result.traverse_plan || !previous->batches)
            throw std::invalid_argument("Transport assembly refresh requires the same retained topology.");
        result.face_offsets = previous->face_offsets;
        result.faces = previous->faces;
        result.physical_boundary_faces = previous->physical_boundary_faces;
        result.boundary_indices = previous->boundary_indices;
        result.batches = previous->batches;
        result.volumes.reserve(mesh.num_owned_cells());
        execution.cells().all().apply([&](const auto& cell) { result.volumes.push_back(cell.volume()); });
        result.epoch = mesh_geometry_epoch(mesh);
        return result;
    }
    result.volumes.reserve(mesh.num_owned_cells());
    result.face_offsets.reserve(mesh.num_owned_cells() + 1);
    result.face_offsets.push_back(0);
    result.faces.reserve(mesh.num_owned_cells() * 6);
    result.physical_boundary_faces = physical_boundary_face_mask(mesh);
    const auto locations = boundary_face_locations(mesh);
    for (size_t face = 0; face < locations.size(); ++face)
    {
        const auto& location = locations[face];
        if (location.active && result.physical_boundary_faces[face])
        {
            result.boundary_indices.emplace(std::pair{location.batch_id, location.in_batch_id}, face);
        }
    }
    if constexpr (requires { mesh.supports_region_execution(); })
    {
        if (mesh.supports_region_execution())
        {
            mesh.visit_owned_region_cells([&](local_ordinal_type, auto canonical_cell,
                                             auto native_cell, const auto& region)
            {
                result.volumes.push_back(region.cell_volume(native_cell));
                region.visit_cell_faces(native_cell, [&](const auto& resolved)
                {
                    typename geometry_type::Face face;
                    face.face_lid = mesh.region_face_local_id(resolved.face);
                    face.interior = resolved.interior();
                    face.owned_orientation = resolved.owner == canonical_cell;
                    if (face.face_lid == mesh.invalid_local_id())
                        throw std::logic_error("Region transport requires the incident face in the existing halo.");
                    face.boundary = result.physical_boundary_faces[face.face_lid];
                    if (face.interior)
                    {
                        face.other = mesh.region_cell_local_id(resolved.opposite_cell(canonical_cell));
                        if (face.other == mesh.invalid_local_id())
                            throw std::logic_error("Region transport requires the adjacent cell in the existing halo.");
                    }
                    result.faces.push_back(face);
                });
                result.face_offsets.push_back(result.faces.size());
            });
            result.batches = make_transport_assembly_batches(mesh, result);
            result.epoch = mesh_geometry_epoch(mesh);
            return result;
        }
    }
    execution.cells().all().apply([&](const auto& cell)
    {
        result.volumes.push_back(cell.volume());
        execution.cell_faces(cell.id()).all().apply(cell.geometry(), [&](const auto& incidence)
        {
            using incidence_type = std::remove_cvref_t<decltype(incidence)>;
            using access_type = std::remove_cvref_t<decltype(incidence.geometry())>;
            typename geometry_type::Face face;
            face.face_lid = incidence.id();
            face.interior = incidence_type::is_interior;
            face.owned_orientation = incidence.orientation() > 0;
            if constexpr (incidence_type::is_interior)
            {
                face.other = face.owned_orientation ? incidence.neighbor() : incidence.owner();
                if (face.other == access_type::invalid_local_id())
                    throw std::invalid_argument("Transport assembly requires adjacent cells in the local map.");
            }
            else face.boundary = incidence.boundary_id() != MeshType::invalid_boundary_id;
            result.faces.push_back(face);
        });
        result.face_offsets.push_back(result.faces.size());
    });
    result.batches = make_transport_assembly_batches(mesh, result);
    result.epoch = mesh_geometry_epoch(mesh);
    return result;
}

template<class MeshType>
TransportAssemblyGeometry<MeshType> transport_assembly_geometry(const MeshType& mesh)
{
    auto plan = std::make_shared<const MeshTraversePlan<MeshType>>(mesh, MeshTraverseEntities::AllWithCellFaces);
    return transport_assembly_geometry(mesh, std::move(plan));
}

/** Select or build locally, then coordinate failure before later assembly collectives. */
template<class MeshType, class GetCached>
const TransportAssemblyGeometry<MeshType>& select_transport_assembly_geometry_impl(
    const MeshType& mesh, TransportAssemblyGeometry<MeshType>& local, GetCached&& get_cached)
{
    const TransportAssemblyGeometry<MeshType>* selected = nullptr;
    std::exception_ptr error;
    try
    {
        selected = get_cached();
        if (!selected)
        {
            local = transport_assembly_geometry(mesh);
            selected = &local;
        }
        selected->require_mesh(mesh);
    }
    catch (...) { error = std::current_exception(); }
    const auto state = reduce_transport_validation_state(mesh, std::array<int, 1>{error ? 1 : 0});
    if (state[0])
    {
        if (error) std::rethrow_exception(error);
        throw std::runtime_error("Transport assembly geometry failed on another rank.");
    }
    return *selected;
}

template<class MeshType>
const TransportAssemblyGeometry<MeshType>& select_transport_assembly_geometry(
    const MeshType& mesh, TransportAssemblyGeometry<MeshType>& local)
{
    return select_transport_assembly_geometry_impl(mesh, local,
        []() -> const TransportAssemblyGeometry<MeshType>* { return nullptr; });
}

template<class MeshType, class Cache>
const TransportAssemblyGeometry<MeshType>& select_transport_assembly_geometry(
    const MeshType& mesh, TransportAssemblyGeometry<MeshType>& local, const Cache* cache)
{
    return select_transport_assembly_geometry_impl(mesh, local, [&]() -> const TransportAssemblyGeometry<MeshType>*
    {
        if (!cache) return nullptr;
        cache->require_mesh(mesh);
        return &cache->assembly_geometry();
    });
}

/** Dispatch once per retained run; the numerical loop contains no category tests. */
template<class MeshType, class VisitRun>
void visit_transport_face_runs(const TransportAssemblyGeometry<MeshType>& geometry, size_t cell, VisitRun&& visit)
{
    const auto& batches = *geometry.batches;
    for (size_t index = batches.row_offsets[cell]; index < batches.row_offsets[cell + 1]; ++index)
    {
        const auto& run = batches.runs[index];
        switch (run.category)
        {
        case 0: visit.template operator()<true, false, true>(run); break;
        case 1: visit.template operator()<true, false, false>(run); break;
        case 2: visit.template operator()<false, true, true>(run); break;
        case 3: visit.template operator()<false, true, false>(run); break;
        case 4: visit.template operator()<false, false, true>(run); break;
        case 5: visit.template operator()<false, false, false>(run); break;
        default: throw std::logic_error("Invalid transport face batch category.");
        }
    }
}

/**
 * @brief Ordered typed transport batches with borrowed, current-epoch metrics.
 *
 * Composite providers still resolve native incidence once per row. One reused
 * row-sized scratch buffer holds resolved snapshots while its homogeneous runs
 * execute; there is no persistent global resolved-face/metric table. Generic
 * providers dispatch once outside the row traversal through typed mesh access.
 */
template<class MeshType, class Visitor>
void visit_transport_assembly_rows(const MeshType& mesh,
    const TransportAssemblyGeometry<MeshType>& geometry, Visitor&& visitor)
{
    using local_ordinal_type = typename MeshType::local_ordinal_type;
    geometry.require_mesh(mesh);
    const auto execution = geometry.traverse_plan->acquire();
    if constexpr (requires { mesh.supports_region_execution(); })
    {
        if (mesh.supports_region_execution())
        {
            using resolved_type = Meshes::MultiRegionMesh::ResolvedFace;
            std::vector<resolved_type> resolved_faces;
            resolved_faces.reserve(6);
            mesh.visit_owned_region_cells([&](local_ordinal_type cell, auto canonical_cell,
                                             auto native_cell, const auto& region)
            {
                visitor(cell, geometry.volumes[cell], [&](auto&& visit_face)
                {
                    const auto begin = geometry.face_offsets[cell];
                    const auto end = geometry.face_offsets[cell + 1];
                    resolved_faces.clear();
                    region.visit_cell_faces(native_cell, [&](const auto& resolved) { resolved_faces.push_back(resolved); });
                    if (resolved_faces.size() != end - begin)
                        throw std::logic_error("Region transport geometry lost its cached incidence order.");
                    visit_transport_face_runs(geometry, static_cast<size_t>(cell),
                        [&]<bool Interior, bool Boundary, bool Owned>(const auto& run)
                    {
                        for (size_t index = run.begin; index < run.end; ++index)
                        {
                            const auto& face = geometry.faces[index];
                            const auto& resolved = resolved_faces[index - begin];
                            const ResolvedTransportGeometry<MeshType, resolved_type> metrics(resolved, cell, canonical_cell);
                            visit_face(TransportAssemblyFace<local_ordinal_type, Interior, Boundary, Owned>{
                                face.face_lid, face.other}, metrics);
                        }
                    });
                });
            });
            return;
        }
    }
    execution.cells().all().apply([&](const auto& cell)
    {
        const auto id = cell.id();
        visitor(id, geometry.volumes[static_cast<size_t>(id)], [&](auto&& visit_face)
        {
            visit_transport_face_runs(geometry, static_cast<size_t>(id),
                [&]<bool Interior, bool Boundary, bool Owned>(const auto& run)
            {
                for (size_t index = run.begin; index < run.end; ++index)
                {
                    const auto& face = geometry.faces[index];
                    visit_face(TransportAssemblyFace<local_ordinal_type, Interior, Boundary, Owned>{
                        face.face_lid, face.other}, cell.geometry());
                }
            });
        });
    });
}

/** Legacy post-row boundary execution keeps the original batch/slot order. */
template<class MeshType, class Visitor>
void visit_transport_boundary_batches(const MeshType& mesh,
    const TransportAssemblyGeometry<MeshType>& geometry, Visitor&& visitor)
{
    geometry.require_mesh(mesh);
    const auto execution = geometry.traverse_plan->acquire();
    ::SimpleFluid::detail::with_traverse_access(mesh, [&](const auto& access)
    {
        for (const auto& batch : geometry.batches->boundary)
            visitor(batch.id, std::span<const typename TransportAssemblyBatches<MeshType>::BoundaryEntry>(batch.faces), access);
    });
}

} // namespace SimpleFluid::FVM::detail
