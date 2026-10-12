/** @file PressureFaceBatches.hh
 * @brief Retained pressure-flux topology and directly paired numeric metrics.
 */
#pragma once

#include "FVM/details/FieldStoredPressureWeightedFaceFlux.hh"
#include "geometry/MeshTraverse.hh"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace SimpleFluid::FVM::detail
{

/** Native providers need a read lease in addition to the composite boundary. */
template<class MeshType>
auto acquire_pressure_native_geometry(const MeshType& mesh)
    -> std::optional<GeometryExecutionGuard::ReadLease>
{
    if constexpr (requires { mesh.variant(); })
        return mesh.visit([](const auto& native) { return acquire_pressure_native_geometry(native); });
    else if constexpr (requires { mesh.parent_mesh(); })
        return acquire_pressure_native_geometry(mesh.parent_mesh());
    else if constexpr (requires { typename MeshType::partitioned_mesh_base_tag; })
        return acquire_pressure_native_geometry(mesh.mesh());
    else if constexpr (requires { mesh.acquire_execution_view(); })
        return {}; // The composite execution owns its provider leases.
    else if constexpr (requires { mesh.acquire_geometry_read(); })
        return mesh.acquire_geometry_read();
    else
        return {};
}

template<TpetraTypePack Pack> struct InteriorPressureFaceTopology
{
    using local_type = typename Pack::local_ordinal_type;
    local_type face{}, owner{}, neighbor{};
    bool owned_owner = false;
};

template<TpetraTypePack Pack> struct BoundaryPressureFaceTopology
{
    using local_type = typename Pack::local_ordinal_type;
    local_type face{}, owner{};
    size_t in_batch_id{};
    bool owned_owner = false;
};

template<TpetraTypePack Pack> struct ZeroPressureFaceTopology
{
    using local_type = typename Pack::local_ordinal_type;
    local_type face{}, owner{};
    int batch_id{};
    size_t in_batch_id{};
};

struct PressureFaceBoundaryGroup
{
    int id{};
    std::string name;
    size_t begin{}, count{}, required_values{};
};

/** Group offsets and field IDs remain unchanged across geometry epochs. */
template<TpetraTypePack Pack, class MeshType> struct PressureFaceTopology
{
    using interior_type = InteriorPressureFaceTopology<Pack>;
    using boundary_type = BoundaryPressureFaceTopology<Pack>;
    using zero_type = ZeroPressureFaceTopology<Pack>;
    using boundary_group_type = PressureFaceBoundaryGroup;

    std::array<const void*, 5> binding{};
    std::array<size_t, 4> counts{};
    std::vector<interior_type> interior;
    std::vector<boundary_type> boundary;
    std::vector<zero_type> zero_flux_faces;
    std::vector<boundary_group_type> boundary_groups;

    size_t num_faces() const noexcept { return interior.size() + boundary.size() + zero_flux_faces.size(); }

    void require_valid(const MeshType& mesh) const
    {
        if (!d_valid.load(std::memory_order_relaxed)
            || binding != ::SimpleFluid::detail::mesh_traverse_binding(mesh)
            || counts != ::SimpleFluid::detail::mesh_traverse_counts(mesh))
            throw std::invalid_argument("Pressure face topology is stale; construct a new workspace after topology, map, or mesh replacement.");
    }

    // As with mesh mutation, callers externally synchronize this write with
    // execution and compatibility-view acquisition on all sharing workspaces.
    void invalidate() const noexcept
    {
        d_valid.store(false, std::memory_order_relaxed);
        if (d_legacy_traverse) d_legacy_traverse->invalidate();
    }

    /** Retain legacy plan identity without allocating it on the numerical path. */
    const MeshTraversePlan<MeshType>& legacy_traverse(const MeshType& mesh) const
    {
        require_valid(mesh);
        std::call_once(d_legacy_traverse_once, [&]
        { d_legacy_traverse = std::make_unique<MeshTraversePlan<MeshType>>(mesh, MeshTraverseEntities::Faces); });
        return *d_legacy_traverse;
    }

private:
    mutable std::atomic<bool> d_valid{true};
    mutable std::once_flag d_legacy_traverse_once;
    mutable std::unique_ptr<MeshTraversePlan<MeshType>> d_legacy_traverse;
};

/** Classify once, then fill final arrays in stable field order within groups. */
template<TpetraTypePack Pack, class MeshType>
auto make_pressure_face_topology(const MeshType& mesh,
    const std::vector<BoundaryFaceLocation<MeshType>>& locations)
    -> std::shared_ptr<const PressureFaceTopology<Pack, MeshType>>
{
    using local_type = typename Pack::local_ordinal_type;
    auto result = std::make_shared<PressureFaceTopology<Pack, MeshType>>();
    result->binding = ::SimpleFluid::detail::mesh_traverse_binding(mesh);
    result->counts = ::SimpleFluid::detail::mesh_traverse_counts(mesh);
    const size_t face_count = result->counts[1];
    if (locations.size() != face_count)
        throw std::invalid_argument("Pressure face boundary locations have the wrong size.");
    if ((face_count && face_count - 1 > static_cast<size_t>(std::numeric_limits<local_type>::max()))
        || (result->counts[0] && result->counts[0] - 1 > static_cast<size_t>(std::numeric_limits<local_type>::max())))
        throw std::overflow_error("Pressure face topology local ordinal overflow.");

    enum class Kind : unsigned char { Skip, Interior, Boundary, Zero };
    std::vector<Kind> kinds(face_count, Kind::Skip);
    std::map<int, size_t> boundary_counts;
    size_t interior_count = 0, zero_count = 0;
    ::SimpleFluid::detail::with_traverse_access(mesh, [&](const auto& access)
    {
        for (size_t face = 0; face < face_count; ++face)
        {
            if (!access.is_owned_face(face)) continue;
            const auto owner = access.owner_cell(face);
            if (owner == access.invalid_local_id() || static_cast<size_t>(owner) >= access.num_cells())
                throw std::invalid_argument("Pressure face requires a locally available owner cell.");
            if (!access.is_exterior_face(face))
            {
                const auto neighbor = access.neighbor_cell(face);
                if (neighbor == access.invalid_local_id() || static_cast<size_t>(neighbor) >= access.num_cells())
                    throw std::invalid_argument("Interior pressure face requires a locally available neighbor cell.");
                kinds[face] = Kind::Interior;
                ++interior_count;
            }
            else
            {
                const auto& location = locations[face];
                const auto batch = access.boundary_id(face);
                if (location.active && location.batch_id != batch)
                    throw std::invalid_argument("Pressure face boundary slot belongs to another batch.");
                const bool active = location.active
                    && mesh.is_boundary_face(query_face_id(mesh, static_cast<local_type>(face)));
                if (active)
                {
                    if (batch == MeshType::invalid_boundary_id)
                        throw std::invalid_argument("Active pressure face has no boundary identifier.");
                    kinds[face] = Kind::Boundary;
                    ++boundary_counts[batch];
                }
                else
                {
                    kinds[face] = Kind::Zero;
                    ++zero_count;
                }
            }
        }

        result->interior.resize(interior_count);
        result->zero_flux_faces.resize(zero_count);
        std::map<int, size_t> group_indices;
        size_t boundary_count = 0;
        for (const auto& [id, count] : boundary_counts)
        {
            group_indices[id] = result->boundary_groups.size();
            result->boundary_groups.push_back({id, mesh.boundary_batch_name(id), boundary_count, count, 0});
            boundary_count += count;
        }
        result->boundary.resize(boundary_count);
        std::vector<size_t> next;
        next.reserve(result->boundary_groups.size());
        for (const auto& group : result->boundary_groups) next.push_back(group.begin);
        size_t interior_next = 0, zero_next = 0;
        for (size_t face = 0; face < face_count; ++face)
        {
            const auto kind = kinds[face];
            if (kind == Kind::Skip) continue;
            const auto id = static_cast<local_type>(face);
            const auto owner = access.owner_cell(face);
            const bool owned_owner = static_cast<size_t>(owner) < result->counts[2];
            if (kind == Kind::Interior)
                result->interior[interior_next++] = {id, owner, access.neighbor_cell(face), owned_owner};
            else if (kind == Kind::Boundary)
            {
                const auto& location = locations[face];
                const auto index = group_indices.at(location.batch_id);
                auto& group = result->boundary_groups[index];
                group.required_values = std::max(group.required_values, location.in_batch_id + 1);
                result->boundary[next[index]++] = {id, owner, location.in_batch_id, owned_owner};
            }
            else
                result->zero_flux_faces[zero_next++] = {id, owner, access.boundary_id(face), locations[face].in_batch_id};
        }
    });
    return result;
}

template<TpetraTypePack Pack> struct InteriorPressureFaceMetrics
{
    using scalar_type = typename Pack::scalar_type;
    using vec_type = vec3<scalar_type>;
    vec_type normal{}, area_vector{};
    scalar_type area{}, owner_weight{}, neighbor_weight{}, center_projection{}, distance_squared{};
};

template<TpetraTypePack Pack> struct BoundaryPressureFaceMetrics
{
    using scalar_type = typename Pack::scalar_type;
    using vec_type = vec3<scalar_type>;
    vec_type normal{}, owner_normal{}, area_vector{};
    scalar_type area{}, boundary_diffusion{};
};

template<TpetraTypePack Pack> struct ZeroPressureFaceMetrics
{
    using scalar_type = typename Pack::scalar_type;
    vec3<scalar_type> normal{};
    scalar_type area{};
};

template<TpetraTypePack Pack> struct PressureFaceMetrics
{
    std::vector<InteriorPressureFaceMetrics<Pack>> interior;
    std::vector<BoundaryPressureFaceMetrics<Pack>> boundary;
    std::vector<ZeroPressureFaceMetrics<Pack>> zero_flux_faces;
};

/** Preserve resolved provider arithmetic without repeating topology grouping. */
template<class MeshType, class Topology, class Visitor>
void with_pressure_face_geometry(const MeshType& mesh, const Topology& face, Visitor&& visitor)
{
    if constexpr (requires { mesh.supports_region_execution(); })
    {
        if (mesh.supports_region_execution())
        {
            const auto resolved = mesh.resolve_region_face(face.face);
            const ResolvedTransportGeometry<MeshType, std::remove_cvref_t<decltype(resolved)>>
                metrics(resolved, face.owner, resolved.owner);
            visitor(metrics);
            return;
        }
    }
    visitor(mesh);
}

/** Fill final category arrays directly; caller retains one geometry lease. */
template<TpetraTypePack Pack, class MeshType>
auto pressure_face_metrics(const MeshType& mesh, const PressureFaceTopology<Pack, MeshType>& topology)
    -> PressureFaceMetrics<Pack>
{
    using scalar_type = typename Pack::scalar_type;
    PressureFaceMetrics<Pack> result;
    result.interior.resize(topology.interior.size());
    result.boundary.resize(topology.boundary.size());
    result.zero_flux_faces.resize(topology.zero_flux_faces.size());
    for (size_t i = 0; i < topology.interior.size(); ++i)
    {
        const auto& face = topology.interior[i];
        auto& entry = result.interior[i];
        with_pressure_face_geometry(mesh, face, [&](const auto& metrics)
        {
            const auto face_id = query_face_id(metrics, face.face);
            const auto owner_id = query_cell_id(metrics, face.owner);
            entry.normal = metrics.face_normal(face_id);
            entry.area = static_cast<scalar_type>(metrics.face_area(face_id));
            entry.area_vector = metrics.face_area_vector_outward(face_id, owner_id);
            const auto weights = stored_interior_face_linear_weights(metrics, face.face, face.owner, face.neighbor);
            entry.owner_weight = weights.first;
            entry.neighbor_weight = weights.second;
            const auto center_delta = metrics.cell_center_vector(face_id, owner_id);
            entry.distance_squared = center_delta.dot(center_delta);
            entry.center_projection = entry.area_vector.dot(center_delta);
        });
    }
    for (size_t i = 0; i < topology.boundary.size(); ++i)
    {
        const auto& face = topology.boundary[i];
        auto& entry = result.boundary[i];
        with_pressure_face_geometry(mesh, face, [&](const auto& metrics)
        {
            const auto face_id = query_face_id(metrics, face.face);
            const auto owner_id = query_cell_id(metrics, face.owner);
            entry.normal = metrics.face_normal(face_id);
            entry.owner_normal = metrics.face_normal_outward(face_id, owner_id);
            entry.area = static_cast<scalar_type>(metrics.face_area(face_id));
            entry.area_vector = metrics.face_area_vector_outward(face_id, owner_id);
            if (face.owned_owner)
                entry.boundary_diffusion = boundary_diffusion_coefficient(metrics, face.face, face.owner, scalar_type{1});
        });
    }
    for (size_t i = 0; i < topology.zero_flux_faces.size(); ++i)
    {
        const auto& face = topology.zero_flux_faces[i];
        auto& entry = result.zero_flux_faces[i];
        with_pressure_face_geometry(mesh, face, [&](const auto& metrics)
        {
            const auto face_id = query_face_id(metrics, face.face);
            entry.normal = metrics.face_normal(face_id);
            entry.area = static_cast<scalar_type>(metrics.face_area(face_id));
        });
    }
    return result;
}

} // namespace SimpleFluid::FVM::detail
