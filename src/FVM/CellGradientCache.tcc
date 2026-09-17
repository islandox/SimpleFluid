/** @file CellGradientCache.tcc @brief Compiled template implementations. */
#pragma once

#include "FVM/CellOperators.hh"

namespace SimpleFluid::FVM
{

template<TpetraTypePack Pack, class MeshType>
auto CellGradientCache<Pack, MeshType>::build_geometry(
    const mesh_type& mesh,
    const std::vector<boundary_location_type>& boundary_locations,
    bool include_boundary_samples, const boundary_direction_provider_type& boundary_direction) -> std::vector<CellGeometry>
{
    const auto execution = acquire_mesh_execution(mesh);
    std::vector<CellGeometry> geometry(mesh.num_owned_cells());
    std::vector<detail::GradientGeometrySample<mesh_type>> samples;
    samples.reserve(6);
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        std::array<std::array<real_t, 3>, 3> normal{};
        auto add_direction = [&](const vec_type& direction)
        {
            normal[0][0] += direction.x * direction.x;
            normal[0][1] += direction.x * direction.y;
            normal[0][2] += direction.x * direction.z;
            normal[1][1] += direction.y * direction.y;
            normal[1][2] += direction.y * direction.z;
            normal[2][2] += direction.z * direction.z;
        };

        samples.clear();
        bool has_boundary_sample = false;
        detail::visit_gradient_geometry_samples(mesh, cell_lid, [&](auto sample)
        {
            if (!sample.interior)
            {
                if (!boundary_locations.at(sample.face_lid).active) return;
                has_boundary_sample = true;
                if (boundary_direction)
                    sample.direction = boundary_direction(sample.face_lid, cell_lid);
            }
            samples.push_back(sample);
            add_direction(sample.direction);
        }, include_boundary_samples);
        if (include_boundary_samples && !has_boundary_sample) continue;
        normal[1][0] = normal[0][1];
        normal[2][0] = normal[0][2];
        normal[2][1] = normal[1][2];

        const auto inverse = inverse_columns(normal);
        auto& cell_geometry = geometry[owned];
        cell_geometry.interior_samples.reserve(samples.size());
        if (include_boundary_samples)
        {
            cell_geometry.boundary_samples.reserve(samples.size());
        }

        for (const auto& sample : samples)
        {
            if (sample.interior)
            {
                cell_geometry.interior_samples.push_back({
                    sample.other_lid,
                    apply_inverse(inverse, sample.direction)});
                continue;
            }
            cell_geometry.boundary_samples.push_back({
                boundary_locations.at(sample.face_lid),
                apply_inverse(inverse, sample.direction),
                sample.normal_distance});
        }
    }
    return geometry;
}


template<TpetraTypePack Pack, class MeshType>
void CellGradientCache<Pack, MeshType>::refresh()
{
    const auto execution = acquire_mesh_execution(*d_mesh);
    auto locations = detail::boundary_face_locations(*d_mesh);
    auto interior = build_geometry(*d_mesh, locations, false, d_boundary_direction);
    auto boundary = build_geometry(*d_mesh, locations, true, d_boundary_direction);
    d_boundary_locations = std::move(locations);
    d_interior_geometry = std::move(interior);
    d_boundary_geometry = std::move(boundary);
    d_geometry_epoch = mesh_geometry_epoch(*d_mesh);
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
    : CellGradientCache(std::move(mesh), {})
{
}

template<TpetraTypePack Pack, class MeshType>
CellGradientCache<Pack, MeshType>::CellGradientCache(
    SP<const mesh_type> mesh, boundary_direction_provider_type boundary_direction)
    : d_mesh(require_mesh(std::move(mesh))), d_boundary_direction(std::move(boundary_direction)),
      d_boundary_locations(detail::boundary_face_locations(*d_mesh)),
      d_interior_geometry(build_geometry(*d_mesh, d_boundary_locations, false, d_boundary_direction)),
      d_boundary_geometry(build_geometry(*d_mesh, d_boundary_locations, true, d_boundary_direction)),
      d_geometry_epoch(mesh_geometry_epoch(*d_mesh))
{
}

} // namespace SimpleFluid::FVM
