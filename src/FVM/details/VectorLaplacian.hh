/** @file VectorLaplacian.hh @brief Shared native/legacy host vector Laplacian. */
#pragma once

#include "FVM/TransportSystem.hh"

namespace SimpleFluid::FVM
{
/**
 * Evaluate div(grad(U)) in the stored global Cartesian component basis.
 * Inputs and supplied gradient must have synchronized overlap values. This
 * function is purely local; callers use collective validation before a halo.
 * The explicit full correction uses the same face decomposition as transport,
 * irrespective of how a transport solve splits that correction implicitly.
 * BoundaryValue supplies Dirichlet face values; BoundaryType identifies
 * homogeneous Neumann outlets. Slip and unmapped periodic faces are rejected.
 */
template<class VectorField, class TensorField, class MeshType, class BoundaryValue, class BoundaryType>
void unit_vector_laplacian(const VectorField& velocity, const TensorField& gradient,
    VectorField& result, const TransportGeometryCache<MeshType>& cache,
    BoundaryValue boundary_value, BoundaryType boundary_type)
{
    const auto& mesh = velocity.mesh();
    cache.require_mesh(mesh);
    if (&gradient.mesh() != &mesh || &result.mesh() != &mesh)
        throw std::invalid_argument("Vector Laplacian fields must use the cached mesh.");
    const auto& geometry = cache.assembly_geometry();
    const auto& locations = cache.boundary_locations();
    const auto u = velocity.local_read_view();
    const auto g = gradient.local_read_view();
    auto lap = result.owned_write_view();
    using lo = typename MeshType::local_ordinal_type;
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<lo>(owned);
        const auto volume = geometry.volumes[owned];
        if (!std::isfinite(volume) || volume <= 0)
            throw std::invalid_argument("Vector Laplacian requires finite positive volumes.");
        std::array<long double, 3> sum{};
        for (size_t slot = geometry.face_offsets[owned]; slot < geometry.face_offsets[owned + 1]; ++slot)
        {
            const auto& face = geometry.faces[slot];
            const auto f = face.face_lid;
            typename MeshType::Vec3 direction{}, difference{};
            long double coefficient{};
            if (face.interior)
            {
                direction = mesh.cell_center_vector(f, cell);
                for (size_t i = 0; i < 3; ++i)
                    difference.component(i) = u(face.other, i) - u(cell, i);
            }
            else if (face.boundary && locations[f].active)
            {
                const auto& loc = locations[f];
                const auto type = boundary_type(loc.batch_id, loc.in_batch_id);
                if (type == BoundaryConditionType::Neumann)
                    continue; // Established velocity transport accepts homogeneous Neumann only.
                if (type != BoundaryConditionType::Dirichlet && type != BoundaryConditionType::NoSlip)
                    throw std::invalid_argument("SAS vector Laplacian supports prescribed velocity/no-slip and homogeneous Neumann boundaries only.");
                direction = mesh.face_centroid(f) - mesh.cell_centroid(cell);
                const auto value = boundary_value(loc.batch_id, loc.in_batch_id);
                for (size_t i = 0; i < 3; ++i)
                    difference.component(i) = value.component(i) - u(cell, i);
            }
            else
                continue;
            const auto area = mesh.face_area_vector_outward(f, cell);
            const auto d2 = direction.dot(direction);
            const auto projection = area.dot(direction);
            if (!std::isfinite(d2) || d2 <= 0 || !std::isfinite(projection) || projection <= 0)
                throw std::invalid_argument("SAS vector Laplacian requires positive finite face projection and center distance.");
            coefficient = projection / d2;
            const auto tangent = area - direction * static_cast<typename MeshType::scalar_type>(coefficient);
            for (size_t i = 0; i < 3; ++i)
            {
                long double flux = coefficient * difference.component(i);
                for (size_t j = 0; j < 3; ++j)
                {
                    const long double grad = face.interior
                        ? (static_cast<long double>(g(cell, 3*i+j)) + g(face.other, 3*i+j)) / 2
                        : g(cell, 3*i+j);
                    flux += grad * tangent.component(j);
                }
                sum[i] += flux;
            }
        }
        for (size_t i = 0; i < 3; ++i)
        {
            const auto value = static_cast<typename MeshType::scalar_type>(sum[i] / volume);
            if (!std::isfinite(value))
                throw std::invalid_argument("SAS vector Laplacian encountered nonfinite velocity/gradient or geometry.");
            lap(cell, i) = value;
        }
    }
}
} // namespace SimpleFluid::FVM
