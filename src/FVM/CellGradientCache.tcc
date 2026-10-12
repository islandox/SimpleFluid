/**
 * @file CellGradientCache.tcc
 * @brief  Compiled template implementations.
 */
#pragma once

#include "FVM/CellOperators.hh"

namespace SimpleFluid::FVM
{

template<TpetraTypePack Pack, class MeshType>
auto CellGradientCache<Pack, MeshType>::build_geometry(
    const mesh_type& mesh,
    const std::vector<boundary_location_type>& boundary_locations,
    const traverse_plan_type& plan, bool include_boundary_samples,
    const boundary_direction_provider_type& boundary_direction) -> std::vector<CellGeometry>
{
    const auto traversal = plan.acquire();
    const auto build = [&]<bool IncludeBoundary>()
    {
        std::vector<CellGeometry> geometry(mesh.num_owned_cells());
        std::vector<detail::GradientGeometrySample<mesh_type>> interior_samples, boundary_samples;
        interior_samples.reserve(6);
        if constexpr (IncludeBoundary) boundary_samples.reserve(6);
        const auto build_cell = [&](const auto& cell)
        {
            const auto cell_lid = cell.id();
            std::array<std::array<real_t, 3>, 3> normal{};
            interior_samples.clear();
            boundary_samples.clear();
            detail::visit_traverse_gradient_samples<mesh_type, IncludeBoundary>(
                traversal.cell_faces(cell_lid), cell.geometry(), [&](auto sample)
            {
                if constexpr (!decltype(sample)::is_interior)
                {
                    if (!boundary_locations.at(sample.face_lid).active) return;
                    if (boundary_direction)
                        sample.direction = boundary_direction(sample.face_lid, cell_lid);
                    boundary_samples.push_back(sample);
                }
                else
                    interior_samples.push_back(sample);
                // Accumulate in original mixed incidence order, before grouping
                // independent inverse applications by output sample category.
                const auto& direction = sample.direction;
                normal[0][0] += direction.x * direction.x;
                normal[0][1] += direction.x * direction.y;
                normal[0][2] += direction.x * direction.z;
                normal[1][1] += direction.y * direction.y;
                normal[1][2] += direction.y * direction.z;
                normal[2][2] += direction.z * direction.z;
            });
            if constexpr (IncludeBoundary)
                if (boundary_samples.empty()) return;
            normal[1][0] = normal[0][1];
            normal[2][0] = normal[0][2];
            normal[2][1] = normal[1][2];

            const auto inverse = inverse_columns(normal);
            auto& cell_geometry = geometry[static_cast<size_t>(cell_lid)];
            cell_geometry.interior_samples.reserve(interior_samples.size());
            cell_geometry.boundary_samples.reserve(boundary_samples.size());
            for (const auto& sample : interior_samples)
                cell_geometry.interior_samples.push_back({
                    sample.other_lid, apply_inverse(inverse, sample.direction)});
            for (const auto& sample : boundary_samples)
                cell_geometry.boundary_samples.push_back({
                    boundary_locations.at(sample.face_lid), apply_inverse(inverse, sample.direction),
                    sample.normal_distance});
        };
        if constexpr (IncludeBoundary) traversal.cells().boundary().apply(build_cell);
        else traversal.cells().all().apply(build_cell);
        return geometry;
    };
    if (include_boundary_samples) return build.template operator()<true>();
    return build.template operator()<false>();
}

template<TpetraTypePack Pack, class MeshType>
void CellGradientCache<Pack, MeshType>::refresh()
{
    const auto execution = d_traverse_plan->acquire();
    auto next = build_snapshot();
    const auto epoch = mesh_geometry_epoch(*d_mesh);
    d_geometry = std::move(next);
    d_geometry_epoch = epoch;
}

template<TpetraTypePack Pack, class MeshType>
auto CellGradientCache<Pack, MeshType>::build_snapshot() const -> std::shared_ptr<GeometrySnapshot>
{
    auto next = std::make_shared<GeometrySnapshot>();
    if (d_traverse_plan->uses_structured_ranges())
    {
        if (d_geometry)
        {
            next->locations = d_geometry->locations;
            build_structured_geometry(*next, nullptr);
        }
        else
        {
            const auto locations = detail::boundary_face_locations(*d_mesh);
            build_structured_geometry(*next, &locations);
        }
    }
    else
    {
        next->locations = d_geometry ? d_geometry->locations
            : std::make_shared<const std::vector<boundary_location_type>>(detail::boundary_face_locations(*d_mesh));
        next->interior = build_geometry(*d_mesh, *next->locations, *d_traverse_plan, false, d_boundary_direction);
        next->boundary = build_geometry(*d_mesh, *next->locations, *d_traverse_plan, true, d_boundary_direction);
        next->legacy_ready.store(true, std::memory_order_relaxed);
    }
    return next;
}

template<TpetraTypePack Pack, class MeshType>
void CellGradientCache<Pack, MeshType>::build_structured_geometry(
    GeometrySnapshot& next, const std::vector<boundary_location_type>* locations) const
{
    ::SimpleFluid::detail::with_traverse_access(*d_mesh, [&](const auto& access)
    {
        const auto layout = access.structured_layout();
        if (!layout) throw std::logic_error("Structured gradient plan lost its canonical layout.");
        next.structured = std::make_unique<StructuredGeometry>(*layout);
    });
    auto& compact = *next.structured;
    const auto count = d_mesh->num_owned_cells();
    const auto interior_count = compact.layout.cell_count(::SimpleFluid::detail::StructuredTraverseSelection::Interior);
    const auto boundary_count = count - interior_count;
    const auto boundary_faces = compact.layout.face_count(::SimpleFluid::detail::StructuredTraverseSelection::Boundary);
    compact.rows.resize(count);
    compact.interior_rows.reserve(interior_count);
    compact.boundary_rows.reserve(boundary_count);
    compact.boundary_interior_only.reserve(boundary_count * 6 - boundary_faces);
    compact.boundary_interior.reserve(boundary_count * 6 - boundary_faces);
    compact.boundary_samples.reserve(boundary_faces);
    const auto traversal = d_traverse_plan->acquire();
    const auto build = [&]<bool IncludeBoundary>()
    {
        std::array<detail::GradientGeometrySample<mesh_type>, 6> inner_samples, boundary_samples;
        const auto build_cell = [&](const auto& cell)
        {
            const auto cell_lid = cell.id();
            const auto owned = static_cast<size_t>(cell_lid);
            size_t inner = 0, boundary = 0;
            std::array<boundary_location_type, 6> selected_locations;
            std::array<std::array<real_t, 3>, 3> normal{};
            detail::visit_traverse_gradient_samples<mesh_type, IncludeBoundary>(
                traversal.cell_faces(cell_lid), cell.geometry(), [&](auto sample)
            {
                if constexpr (!decltype(sample)::is_interior)
                {
                    const auto location = [&]
                    {
                        if (locations) return locations->at(sample.face_lid);
                        const auto& previous = *d_geometry->structured;
                        const auto& row = previous.boundary_rows.at(previous.rows.at(owned) >> 1);
                        return previous.boundary_samples.at(row.boundary_begin + boundary).location;
                    }();
                    if (!location.active) return;
                    if (d_boundary_direction)
                        sample.direction = d_boundary_direction(sample.face_lid, cell_lid);
                    selected_locations[boundary] = location;
                    boundary_samples[boundary++] = sample;
                }
                else inner_samples[inner++] = sample;
                const auto& direction = sample.direction;
                normal[0][0] += direction.x * direction.x;
                normal[0][1] += direction.x * direction.y;
                normal[0][2] += direction.x * direction.z;
                normal[1][1] += direction.y * direction.y;
                normal[1][2] += direction.y * direction.z;
                normal[2][2] += direction.z * direction.z;
            });
            if constexpr (IncludeBoundary)
                if (!boundary) return;
            normal[1][0] = normal[0][1];
            normal[2][0] = normal[0][2];
            normal[2][1] = normal[1][2];
            const auto inverse = inverse_columns(normal);
            if constexpr (!IncludeBoundary && std::remove_cvref_t<decltype(cell)>::is_interior)
            {
                if (inner != 6) throw std::logic_error("Structured interior gradient requires six incidences.");
                compact.rows[owned] = compact.interior_rows.size() << 1;
                auto& row = compact.interior_rows.emplace_back();
                for (size_t slot = 0; slot < inner; ++slot)
                {
                    row.neighbors[slot] = inner_samples[slot].other_lid;
                    row.weights[slot] = apply_inverse(inverse, inner_samples[slot].direction);
                }
            }
            else
            {
                const auto row_index = [&]
                {
                    if constexpr (IncludeBoundary) return compact.rows[owned] >> 1;
                    else
                    {
                        const auto index = compact.boundary_rows.size();
                        compact.rows[owned] = (index << 1) | size_t{1};
                        compact.boundary_rows.emplace_back();
                        return index;
                    }
                }();
                auto& row = compact.boundary_rows[row_index];
                row.interior_count = static_cast<unsigned char>(inner);
                auto& entries = IncludeBoundary ? compact.boundary_interior : compact.boundary_interior_only;
                if constexpr (IncludeBoundary) row.boundary_interior_begin = entries.size();
                else row.interior_begin = entries.size();
                for (size_t slot = 0; slot < inner; ++slot)
                    entries.push_back({inner_samples[slot].other_lid, apply_inverse(inverse, inner_samples[slot].direction)});
                if constexpr (IncludeBoundary)
                {
                    row.boundary_begin = compact.boundary_samples.size();
                    row.boundary_count = static_cast<unsigned char>(boundary);
                    for (size_t slot = 0; slot < boundary; ++slot)
                        compact.boundary_samples.push_back({selected_locations[slot],
                            apply_inverse(inverse, boundary_samples[slot].direction), boundary_samples[slot].normal_distance});
                }
            }
        };
        if constexpr (IncludeBoundary) traversal.cells().boundary().apply(build_cell);
        else traversal.cells().apply(build_cell);
    };
    build.template operator()<false>();
    build.template operator()<true>();
}

template<TpetraTypePack Pack, class MeshType>
void CellGradientCache<Pack, MeshType>::materialize_geometry() const
{
    if (!d_geometry->structured) return;
    std::call_once(d_geometry->legacy_once, [&]
    {
        const auto& compact = *d_geometry->structured;
        std::vector<CellGeometry> interior(compact.rows.size()), boundary(compact.rows.size());
        compact.template visit_rows<false>([&](size_t cell, const auto& row)
        {
            interior[cell].interior_samples.reserve(row.interior_samples.size());
            for (const auto& sample : row.interior_samples) interior[cell].interior_samples.push_back(sample);
        });
        compact.template visit_rows<true>([&](size_t cell, const auto& row)
        {
            if (row.boundary_samples.empty()) return;
            boundary[cell].interior_samples.reserve(row.interior_samples.size());
            for (const auto& sample : row.interior_samples) boundary[cell].interior_samples.push_back(sample);
            boundary[cell].boundary_samples.assign(row.boundary_samples.begin(), row.boundary_samples.end());
        });
        d_geometry->interior = std::move(interior);
        d_geometry->boundary = std::move(boundary);
        d_geometry->legacy_ready.store(true, std::memory_order_release);
    });
}

template<TpetraTypePack Pack, class MeshType>
void CellGradientCache<Pack, MeshType>::materialize_locations() const
{
    if (!d_geometry->structured) return;
    std::call_once(d_geometry->locations_once, [&]
    {
        if (d_geometry->locations) return;
        const auto& compact = *d_geometry->structured;
        auto locations = std::make_shared<std::vector<boundary_location_type>>(compact.layout.num_faces());
        for (size_t cell = 0; cell < compact.rows.size(); ++cell)
        {
            if (!(compact.rows[cell] & size_t{1})) continue;
            const auto& row = compact.boundary_rows[compact.rows[cell] >> 1];
            size_t slot = 0;
            for (const auto& face : compact.layout.cell_faces(cell))
                if (!face.interior)
                    locations->at(face.face) = compact.boundary_samples.at(row.boundary_begin + slot++).location;
        }
        d_geometry->locations = std::move(locations);
    });
}

template<TpetraTypePack Pack, class MeshType>
size_t CellGradientCache<Pack, MeshType>::numerical_storage_bytes() const noexcept
{
    if (!d_geometry) return 0;
    size_t bytes = sizeof(*this) + sizeof(GeometrySnapshot);
    if (d_geometry->structured) return bytes + d_geometry->structured->storage_bytes();
    bytes += d_geometry->locations->capacity() * sizeof(boundary_location_type);
    for (const auto* rows : {&d_geometry->interior, &d_geometry->boundary})
    {
        bytes += rows->capacity() * sizeof(CellGeometry);
        for (const auto& row : *rows)
            bytes += row.interior_samples.capacity() * sizeof(InteriorSample)
                + row.boundary_samples.capacity() * sizeof(BoundarySample);
    }
    return bytes;
}

template<TpetraTypePack Pack, class MeshType>
auto CellGradientCache<Pack, MeshType>::inverse_columns(
    const std::array<std::array<real_t, 3>, 3>& normal) -> std::array<vec_type, 3>
{
    std::array<vec_type, 3> inverse{};
    for (size_t column = 0; column < inverse.size(); ++column)
    {
        auto local_normal = normal;
        vec_type direction{};
        direction.component(column) = real_t{1};
        inverse[column] =
            detail::solve_3x3(local_normal, direction);
    }
    return inverse;
}


template<TpetraTypePack Pack, class MeshType>
CellGradientCache<Pack, MeshType>::CellGradientCache(SP<const mesh_type> mesh)
    : CellGradientCache(std::move(mesh), boundary_direction_provider_type{})
{
}

template<TpetraTypePack Pack, class MeshType>
CellGradientCache<Pack, MeshType>::CellGradientCache(
    SP<const mesh_type> mesh, boundary_direction_provider_type boundary_direction)
    : CellGradientCache(std::move(mesh), std::move(boundary_direction), {})
{
}

template<TpetraTypePack Pack, class MeshType>
CellGradientCache<Pack, MeshType>::CellGradientCache(
    SP<const mesh_type> mesh, boundary_direction_provider_type boundary_direction,
    std::shared_ptr<const traverse_plan_type> traverse_plan)
    : d_mesh(require_mesh(std::move(mesh))),
      d_traverse_plan(require_traverse_plan(*d_mesh, std::move(traverse_plan))),
      d_boundary_direction(std::move(boundary_direction))
{
    refresh();
}

} // namespace SimpleFluid::FVM
