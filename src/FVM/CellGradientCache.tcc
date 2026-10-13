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
                traversal.cell_faces(cell), cell.geometry(), [&](auto sample)
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
    if (d_traverse_plan->uses_layered_ranges())
    {
        next->locations = d_geometry ? d_geometry->locations
            : std::make_shared<const std::vector<boundary_location_type>>(detail::boundary_face_locations(*d_mesh));
        build_layered_geometry(*next);
    }
    else if (d_traverse_plan->uses_structured_ranges())
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
        if constexpr (requires { d_mesh->supports_region_execution(); })
        {
            if (d_mesh->supports_region_execution())
            {
                build_region_geometry(*next);
                next->legacy_ready.store(true, std::memory_order_relaxed);
                return next;
            }
        }
        next->interior = build_geometry(*d_mesh, *next->locations, *d_traverse_plan, false, d_boundary_direction);
        next->boundary = build_geometry(*d_mesh, *next->locations, *d_traverse_plan, true, d_boundary_direction);
        next->legacy_ready.store(true, std::memory_order_relaxed);
    }
    return next;
}

template<TpetraTypePack Pack, class MeshType>
void CellGradientCache<Pack, MeshType>::build_region_geometry(GeometrySnapshot& next) const
{
    if constexpr (requires { d_mesh->supports_region_execution(); })
    {
        const auto& mesh = *d_mesh;
        next.interior.resize(mesh.num_owned_cells());
        next.boundary.resize(mesh.num_owned_cells());
        using sample_type = detail::GradientGeometrySample<mesh_type>;
        using normal_type = std::array<std::array<real_t, 3>, 3>;
        std::vector<sample_type> interior_samples, boundary_samples;
        interior_samples.reserve(6);
        boundary_samples.reserve(6);
        const auto accumulate = [](normal_type& normal, const auto& direction)
        {
            normal[0][0] += direction.x * direction.x;
            normal[0][1] += direction.x * direction.y;
            normal[0][2] += direction.x * direction.z;
            normal[1][1] += direction.y * direction.y;
            normal[1][2] += direction.y * direction.z;
            normal[2][2] += direction.z * direction.z;
        };
        const auto store = [&]<bool IncludeBoundary>(CellGeometry& row, normal_type normal)
        {
            normal[1][0] = normal[0][1];
            normal[2][0] = normal[0][2];
            normal[2][1] = normal[1][2];
            const auto inverse = inverse_columns(normal);
            row.interior_samples.reserve(interior_samples.size());
            for (const auto& sample : interior_samples)
                row.interior_samples.push_back({sample.other_lid, apply_inverse(inverse, sample.direction)});
            if constexpr (IncludeBoundary)
            {
                row.boundary_samples.reserve(boundary_samples.size());
                for (const auto& sample : boundary_samples)
                    row.boundary_samples.push_back({next.locations->at(sample.face_lid),
                        apply_inverse(inverse, sample.direction), sample.normal_distance});
            }
        };
        mesh.visit_owned_region_cells([&](local_ordinal_type cell, auto canonical_cell,
                                         auto native_cell, const auto& region)
        {
            normal_type interior_normal{}, boundary_normal{};
            interior_samples.clear();
            boundary_samples.clear();
            region.visit_cell_faces(native_cell, [&](const auto& face)
            {
                sample_type sample;
                sample.face_lid = mesh.region_face_local_id(face.face);
                // Retain the traverser's visible-incidence and halo contracts.
                if (sample.face_lid == mesh.invalid_local_id()) return;
                if (face.interior())
                {
                    sample.other_lid = mesh.region_cell_local_id(face.opposite_cell(canonical_cell));
                    if (sample.other_lid == mesh.invalid_local_id())
                        throw std::invalid_argument("Gradient traversal requires adjacent cells in the local cell map.");
                    sample.direction = face.cell_center_vector(canonical_cell);
                    interior_samples.push_back(sample);
                    accumulate(interior_normal, sample.direction);
                    if (!boundary_samples.empty()) accumulate(boundary_normal, sample.direction);
                }
                else
                {
                    if (!next.locations->at(sample.face_lid).active) return;
                    sample.direction = face.centroid - face.owner_centroid;
                    sample.normal_distance = static_cast<typename mesh_type::scalar_type>(
                        sample.direction.dot(face.face_normal_outward(canonical_cell)));
                    if (d_boundary_direction) sample.direction = d_boundary_direction(sample.face_lid, cell);
                    // Until the first active boundary contribution, both normal
                    // matrices have exactly the same ordered interior sum.
                    if (boundary_samples.empty()) boundary_normal = interior_normal;
                    boundary_samples.push_back(sample);
                    accumulate(boundary_normal, sample.direction);
                }
            });
            store.template operator()<false>(next.interior[static_cast<size_t>(cell)], interior_normal);
            if (!boundary_samples.empty())
                store.template operator()<true>(next.boundary[static_cast<size_t>(cell)], boundary_normal);
        });
    }
    else
        throw std::logic_error("Region gradient construction requires region execution support.");
}

template<TpetraTypePack Pack, class MeshType>
void CellGradientCache<Pack, MeshType>::build_layered_geometry(GeometrySnapshot& next) const
{
    next.layered = std::make_unique<LayeredGeometry>();
    auto& compact = *next.layered;
    compact.layout = d_traverse_plan->layered_layout();
    compact.rows.resize(d_mesh->num_owned_cells());
    compact.interior.reserve(2 * compact.layout->face_count(
        ::SimpleFluid::detail::StructuredTraverseSelection::Interior));
    compact.boundary.reserve(compact.layout->face_count(
        ::SimpleFluid::detail::StructuredTraverseSelection::Boundary));
    compact.boundary_interior.reserve(2 * compact.layout->face_count(
        ::SimpleFluid::detail::StructuredTraverseSelection::Interior)
        - compact.layout->cell_incidence_count(::SimpleFluid::detail::StructuredTraverseSelection::Interior));
    const auto traversal = d_traverse_plan->acquire();
    using normal_type = std::array<std::array<real_t, 3>, 3>;
    const auto accumulate = [](normal_type& normal, const auto& direction)
    {
        normal[0][0] += direction.x * direction.x;
        normal[0][1] += direction.x * direction.y;
        normal[0][2] += direction.x * direction.z;
        normal[1][1] += direction.y * direction.y;
        normal[1][2] += direction.y * direction.z;
        normal[2][2] += direction.z * direction.z;
    };
    const auto inverse = [](normal_type normal)
    {
        normal[1][0] = normal[0][1];
        normal[2][0] = normal[0][2];
        normal[2][1] = normal[1][2];
        return inverse_columns(normal);
    };
    // Reused scratch grows to the largest polygon, never one allocation per cell.
    std::vector<detail::GradientGeometrySample<mesh_type>> inner_samples, boundary_samples;
    const auto build_cell = [&](const auto& cell)
    {
        using cell_type = std::remove_cvref_t<decltype(cell)>;
        const auto cell_lid = cell.id();
        auto& row = compact.rows.at(static_cast<size_t>(cell_lid));
        inner_samples.clear();
        boundary_samples.clear();
        normal_type inner_normal{}, boundary_normal{};
        detail::visit_traverse_gradient_samples<mesh_type, true>(
            traversal.cell_faces(cell), cell.geometry(), [&](auto sample)
        {
            if constexpr (decltype(sample)::is_interior)
            {
                inner_samples.push_back(sample);
                accumulate(inner_normal, sample.direction);
                if constexpr (!cell_type::is_interior)
                    if (!boundary_samples.empty()) accumulate(boundary_normal, sample.direction);
            }
            else
            {
                if (!next.locations->at(sample.face_lid).active) return;
                if (d_boundary_direction)
                    sample.direction = d_boundary_direction(sample.face_lid, cell_lid);
                if (boundary_samples.empty()) boundary_normal = inner_normal;
                boundary_samples.push_back(sample);
                accumulate(boundary_normal, sample.direction);
            }
        });
        const auto inner_inverse = inverse(inner_normal);
        row.interior_begin = compact.interior.size();
        row.interior_count = inner_samples.size();
        for (const auto& sample : inner_samples)
            compact.interior.push_back({sample.other_lid, apply_inverse(inner_inverse, sample.direction)});
        if constexpr (!cell_type::is_interior)
        {
            if (boundary_samples.empty()) return;
            const auto boundary_inverse = inverse(boundary_normal);
            row.boundary_interior_begin = compact.boundary_interior.size();
            for (const auto& sample : inner_samples)
                compact.boundary_interior.push_back({sample.other_lid,
                    apply_inverse(boundary_inverse, sample.direction)});
            row.boundary_begin = compact.boundary.size();
            row.boundary_count = boundary_samples.size();
            for (const auto& sample : boundary_samples)
                compact.boundary.push_back({next.locations->at(sample.face_lid),
                    apply_inverse(boundary_inverse, sample.direction), sample.normal_distance});
        }
    };
    traversal.cells().interior().apply(build_cell);
    traversal.cells().boundary().apply(build_cell);
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
    using normal_type = std::array<std::array<real_t, 3>, 3>;
    const auto accumulate = [](normal_type& normal, const auto& direction)
    {
        normal[0][0] += direction.x * direction.x;
        normal[0][1] += direction.x * direction.y;
        normal[0][2] += direction.x * direction.z;
        normal[1][1] += direction.y * direction.y;
        normal[1][2] += direction.y * direction.z;
        normal[2][2] += direction.z * direction.z;
    };
    const auto inverse = [](normal_type normal)
    {
        normal[1][0] = normal[0][1];
        normal[2][0] = normal[0][2];
        normal[2][1] = normal[1][2];
        return inverse_columns(normal);
    };
    // The proven interior kernel needs only neighbor IDs and displacements.
    // Keep its scratch local so boundary samples and callbacks stay outside the
    // six-face numerical loop; weights hold directions until inversion.
    const auto build_interior = [&](const auto& cell)
    {
        const auto owned = static_cast<size_t>(cell.id());
        StructuredInteriorRow row;
        normal_type normal{};
        size_t inner = 0;
        detail::visit_traverse_gradient_samples<mesh_type, false>(
            traversal.cell_faces(cell), cell.geometry(), [&](const auto& sample)
        {
            row.neighbors[inner] = sample.other_lid;
            row.weights[inner++] = sample.direction;
            accumulate(normal, sample.direction);
        });
        if (inner != 6) throw std::logic_error("Structured interior gradient requires six incidences.");
        const auto columns = inverse(normal);
        for (auto& weight : row.weights) weight = apply_inverse(columns, weight);
        compact.rows[owned] = compact.interior_rows.size() << 1;
        compact.interior_rows.push_back(std::move(row));
    };
    traversal.cells().interior().apply(build_interior);

    std::array<detail::GradientGeometrySample<mesh_type>, 6> inner_samples, boundary_samples;
    const auto build_boundary = [&](const auto& cell)
    {
        const auto cell_lid = cell.id();
        const auto owned = static_cast<size_t>(cell_lid);
        size_t inner = 0, boundary = 0;
        std::array<boundary_location_type, 6> selected_locations;
        normal_type inner_normal{}, boundary_normal{};
        detail::visit_traverse_gradient_samples<mesh_type, true>(
            traversal.cell_faces(cell), cell.geometry(), [&](auto sample)
        {
            if constexpr (decltype(sample)::is_interior)
            {
                inner_samples[inner++] = sample;
                accumulate(inner_normal, sample.direction);
                if (boundary) accumulate(boundary_normal, sample.direction);
            }
            else
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
                if (!boundary) boundary_normal = inner_normal;
                selected_locations[boundary] = location;
                boundary_samples[boundary++] = sample;
                accumulate(boundary_normal, sample.direction);
            }
        });
        const auto inner_inverse = inverse(inner_normal);
        compact.rows[owned] = (compact.boundary_rows.size() << 1) | size_t{1};
        auto& row = compact.boundary_rows.emplace_back();
        row.interior_count = static_cast<unsigned char>(inner);
        row.interior_begin = compact.boundary_interior_only.size();
        for (size_t slot = 0; slot < inner; ++slot)
            compact.boundary_interior_only.push_back({inner_samples[slot].other_lid,
                apply_inverse(inner_inverse, inner_samples[slot].direction)});
        if (!boundary) return;
        const auto boundary_inverse = inverse(boundary_normal);
        row.boundary_interior_begin = compact.boundary_interior.size();
        for (size_t slot = 0; slot < inner; ++slot)
            compact.boundary_interior.push_back({inner_samples[slot].other_lid,
                apply_inverse(boundary_inverse, inner_samples[slot].direction)});
        row.boundary_begin = compact.boundary_samples.size();
        row.boundary_count = static_cast<unsigned char>(boundary);
        for (size_t slot = 0; slot < boundary; ++slot)
            compact.boundary_samples.push_back({selected_locations[slot],
                apply_inverse(boundary_inverse, boundary_samples[slot].direction), boundary_samples[slot].normal_distance});
    };
    traversal.cells().boundary().apply(build_boundary);
}

template<TpetraTypePack Pack, class MeshType>
void CellGradientCache<Pack, MeshType>::materialize_geometry() const
{
    if (!d_geometry->structured && !d_geometry->layered) return;
    std::call_once(d_geometry->legacy_once, [&]
    {
        const auto materialize = [&](const auto& compact)
        {
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
        };
        if (d_geometry->layered) materialize(*d_geometry->layered);
        else materialize(*d_geometry->structured);
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
    if (d_geometry->layered) return bytes + d_geometry->layered->storage_bytes()
        + d_geometry->locations->capacity() * sizeof(boundary_location_type);
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
