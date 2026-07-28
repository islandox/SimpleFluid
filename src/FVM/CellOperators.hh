/**
 * @file CellOperators.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Cell-centered finite-volume gradient and divergence operators.
 * @version 0.1
 * @date 2026-05-30
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoundaryConditions.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/TensorCellField.hh"
#include "fields/VectorCellField.hh"
#include "FVM/CellGradientScheme.hh"
#include "FVM/OperatorDetails.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid::FVM
{

/**
 * @brief Mesh-bound least-squares weights for repeated cell gradients.
 *
 * The cache owns a shared reference to the mesh and precomputes two
 * reconstruction variants: an interior-neighbor-only stencil and a
 * boundary-aware stencil. Boundary condition types and values remain dynamic;
 * only topology, directions, normal distances, and least-squares weights are
 * cached. Rebuild the cache after any mesh topology or geometry revision.
 *
 * Cached data are immutable after construction, so one cache may be read by
 * concurrent evaluations provided their fields and callbacks are independent.
 *
 * @tparam Pack Tpetra type pack used by the mesh and fields.
 */
template<TpetraTypePack Pack>
class CellGradientCache
{
public:
    using mesh_type = Mesh<Pack>;
    using local_ordinal_type = typename mesh_type::local_ordinal_type;
    using vec_type = typename mesh_type::Vec3;
    using boundary_location_type =
        detail::BoundaryFaceLocation<mesh_type>;

    /** @brief One cached interior-neighbor contribution. */
    struct InteriorSample
    {
        local_ordinal_type other_lid{};
        vec_type weight{};
    };

    /** @brief One cached boundary-face contribution. */
    struct BoundarySample
    {
        boundary_location_type location{};
        vec_type weight{};
        real_t normal_distance{};
    };

    /** @brief Face-ordered weights for one owned cell. */
    struct CellGeometry
    {
        std::vector<InteriorSample> interior_samples;
        std::vector<BoundarySample> boundary_samples;
    };

    /**
     * @brief Precompute least-squares weights for an assembled mesh.
     * @param mesh Shared mesh whose lifetime and geometry the cache retains.
     * @throws std::invalid_argument if @p mesh is null.
     */
    explicit CellGradientCache(SP<const mesh_type> mesh)
        : d_mesh(require_mesh(std::move(mesh))),
          d_boundary_locations(
              detail::boundary_face_locations(*d_mesh)),
          d_interior_geometry(
              build_geometry(
                  *d_mesh, d_boundary_locations, false)),
          d_boundary_geometry(
              build_geometry(
                  *d_mesh, d_boundary_locations, true))
    {
    }

    CellGradientCache(const CellGradientCache&) = delete;
    CellGradientCache& operator=(const CellGradientCache&) = delete;
    CellGradientCache(CellGradientCache&&) = default;
    CellGradientCache& operator=(CellGradientCache&&) = default;

    /** @brief Return the mesh retained by this cache. */
    const SP<const mesh_type>& mesh_ptr() const noexcept
    {
        return d_mesh;
    }

    /**
     * @brief Throw unless @p mesh is the exact cached mesh instance.
     */
    void require_mesh(const mesh_type& mesh) const
    {
        if (&mesh != d_mesh.get())
        {
            throw std::invalid_argument(
                "Cell-gradient cache belongs to another mesh.");
        }
    }

    /** @brief Return interior-neighbor-only reconstruction weights. */
    const std::vector<CellGeometry>&
    interior_geometry() const noexcept
    {
        return d_interior_geometry;
    }

    /** @brief Return boundary-aware reconstruction weights. */
    const std::vector<CellGeometry>&
    boundary_geometry() const noexcept
    {
        return d_boundary_geometry;
    }

    /** @brief Return the cached per-face boundary lookup. */
    const std::vector<boundary_location_type>&
    boundary_locations() const noexcept
    {
        return d_boundary_locations;
    }

private:
    static SP<const mesh_type> require_mesh(SP<const mesh_type> mesh)
    {
        if (!mesh)
        {
            throw std::invalid_argument(
                "CellGradientCache requires a non-null mesh.");
        }
        return mesh;
    }

    static std::array<vec_type, 3> inverse_columns(
        const std::array<std::array<real_t, 3>, 3>& normal)
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

    static vec_type apply_inverse(
        const std::array<vec_type, 3>& inverse,
        const vec_type& direction)
    {
        return inverse[0] * direction.x
             + inverse[1] * direction.y
             + inverse[2] * direction.z;
    }

    static std::vector<CellGeometry> build_geometry(
        const mesh_type& mesh,
        const std::vector<boundary_location_type>& boundary_locations,
        bool include_boundary_samples)
    {
        std::vector<CellGeometry> geometry(mesh.num_owned_cells());
        for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            if (include_boundary_samples)
            {
                const auto has_boundary_sample =
                    std::any_of(
                        mesh.faces(cell_lid).begin(),
                        mesh.faces(cell_lid).end(),
                        [&](const auto face_lid)
                        {
                            return mesh.is_boundary_face(face_lid)
                                && boundary_locations.at(
                                       static_cast<size_t>(face_lid))
                                       .active;
                        });
                if (!has_boundary_sample)
                {
                    continue;
                }
            }
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

            for (const auto face_lid : mesh.faces(cell_lid))
            {
                if (mesh.is_interior_face(face_lid))
                {
                    add_direction(
                        mesh.cell_center_vector(face_lid, cell_lid));
                }
                else if (include_boundary_samples
                         && mesh.is_boundary_face(face_lid))
                {
                    const auto location = boundary_locations.at(
                        static_cast<size_t>(face_lid));
                    if (location.active)
                    {
                        add_direction(
                            mesh.face_centroid(face_lid)
                            - mesh.cell_centroid(cell_lid));
                    }
                }
            }
            normal[1][0] = normal[0][1];
            normal[2][0] = normal[0][2];
            normal[2][1] = normal[1][2];

            const auto inverse = inverse_columns(normal);
            auto& cell_geometry = geometry[owned];
            cell_geometry.interior_samples.reserve(
                mesh.faces(cell_lid).size());
            if (include_boundary_samples)
            {
                cell_geometry.boundary_samples.reserve(
                    mesh.faces(cell_lid).size());
            }

            for (const auto face_lid : mesh.faces(cell_lid))
            {
                if (mesh.is_interior_face(face_lid))
                {
                    const auto direction =
                        mesh.cell_center_vector(face_lid, cell_lid);
                    cell_geometry.interior_samples.push_back({
                        mesh.opposite_or_periodic_neighbor_cell(
                            face_lid, cell_lid),
                        apply_inverse(inverse, direction)});
                    continue;
                }
                if (!include_boundary_samples
                    || !mesh.is_boundary_face(face_lid))
                {
                    continue;
                }
                const auto location = boundary_locations.at(
                    static_cast<size_t>(face_lid));
                if (!location.active)
                {
                    continue;
                }
                const auto direction =
                    mesh.face_centroid(face_lid)
                    - mesh.cell_centroid(cell_lid);
                cell_geometry.boundary_samples.push_back({
                    location,
                    apply_inverse(inverse, direction),
                    static_cast<real_t>(
                        detail::boundary_normal_distance(
                            mesh, face_lid, cell_lid))});
            }
        }
        return geometry;
    }

    SP<const mesh_type> d_mesh;
    std::vector<boundary_location_type> d_boundary_locations;
    std::vector<CellGeometry> d_interior_geometry;
    std::vector<CellGeometry> d_boundary_geometry;
};

namespace detail
{

/** @brief Validate fields and a cached reconstruction share one mesh. */
template<TpetraTypePack Pack, class GradientField>
void require_cached_gradient_mesh(
    const CellField<Pack>& field,
    const GradientField& gradients,
    const CellGradientCache<Pack>& cache)
{
    const auto& mesh = field.mesh();
    if (&gradients.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "cell_gradient requires input and output fields on one mesh.");
    }
    cache.require_mesh(mesh);
}

/** @brief Evaluate cached interior-only scalar reconstruction weights. */
template<TpetraTypePack Pack>
void cached_scalar_cell_gradient(
    const CellField<Pack>& field,
    VectorCellField<Pack>& gradients,
    const CellGradientCache<Pack>& cache)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using scalar_type = typename Pack::scalar_type;

    require_cached_gradient_mesh(field, gradients, cache);
    const auto owned_values = field.owned_read_view();
    const auto local_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    const auto& geometry = cache.interior_geometry();
    for (size_t owned = 0; owned < geometry.size(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto value_p = owned_values(cell_lid, 0);
        typename Mesh<Pack>::Vec3 gradient{};
        for (const auto& sample : geometry[owned].interior_samples)
        {
            const auto delta =
                local_values(sample.other_lid, 0) - value_p;
            gradient.x += sample.weight.x * delta;
            gradient.y += sample.weight.y * delta;
            gradient.z += sample.weight.z * delta;
        }
        gradient_values(cell_lid, 0) =
            static_cast<scalar_type>(gradient.x);
        gradient_values(cell_lid, 1) =
            static_cast<scalar_type>(gradient.y);
        gradient_values(cell_lid, 2) =
            static_cast<scalar_type>(gradient.z);
    }
}

/** @brief Evaluate cached boundary-aware scalar reconstruction weights. */
template<TpetraTypePack Pack,
         class BoundaryConditionProvider,
         class BoundaryValueProvider>
void cached_scalar_cell_gradient(
    const CellField<Pack>& field,
    BoundaryConditionProvider& boundary_condition,
    BoundaryValueProvider& boundary_value,
    VectorCellField<Pack>& gradients,
    const CellGradientCache<Pack>& cache)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using scalar_type = typename Pack::scalar_type;

    require_cached_gradient_mesh(field, gradients, cache);
    const auto owned_values = field.owned_read_view();
    const auto local_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    const auto& interior_geometry = cache.interior_geometry();
    const auto& boundary_geometry = cache.boundary_geometry();
    for (size_t owned = 0;
         owned < boundary_geometry.size();
         ++owned)
    {
        const auto& cell_geometry =
            boundary_geometry[owned].boundary_samples.empty()
          ? interior_geometry[owned]
          : boundary_geometry[owned];
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto value_p = owned_values(cell_lid, 0);
        typename Mesh<Pack>::Vec3 gradient{};
        for (const auto& sample : cell_geometry.interior_samples)
        {
            const auto delta =
                local_values(sample.other_lid, 0) - value_p;
            gradient.x += sample.weight.x * delta;
            gradient.y += sample.weight.y * delta;
            gradient.z += sample.weight.z * delta;
        }
        for (const auto& sample : cell_geometry.boundary_samples)
        {
            const auto condition = boundary_condition(
                sample.location.batch_id,
                sample.location.in_batch_id);
            scalar_type delta{};
            if (condition.type == BoundaryConditionType::Dirichlet)
            {
                delta = static_cast<scalar_type>(boundary_value(
                            sample.location.batch_id,
                            sample.location.in_batch_id))
                      - value_p;
            }
            else if (condition.type == BoundaryConditionType::Neumann)
            {
                delta =
                    static_cast<scalar_type>(condition.value)
                  * static_cast<scalar_type>(sample.normal_distance);
            }
            else
            {
                throw std::invalid_argument(
                    "cell_gradient supports only Dirichlet and Neumann "
                    "scalar boundary conditions.");
            }
            gradient.x += sample.weight.x * delta;
            gradient.y += sample.weight.y * delta;
            gradient.z += sample.weight.z * delta;
        }
        gradient_values(cell_lid, 0) =
            static_cast<scalar_type>(gradient.x);
        gradient_values(cell_lid, 1) =
            static_cast<scalar_type>(gradient.y);
        gradient_values(cell_lid, 2) =
            static_cast<scalar_type>(gradient.z);
    }
}

/** @brief Evaluate cached interior-only vector reconstruction weights. */
template<TpetraTypePack Pack>
void cached_vector_cell_gradient(
    const VectorCellField<Pack>& field,
    TensorCellField<Pack>& gradients,
    const CellGradientCache<Pack>& cache)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using scalar_type = typename Pack::scalar_type;

    const auto& mesh = field.mesh();
    if (&gradients.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "cell_gradient requires input and output fields on one mesh.");
    }
    cache.require_mesh(mesh);

    const auto owned_values = field.owned_read_view();
    const auto local_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    const auto& geometry = cache.interior_geometry();
    for (size_t owned = 0; owned < geometry.size(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        std::array<typename Mesh<Pack>::Vec3, 3> gradient{};
        for (const auto& sample : geometry[owned].interior_samples)
        {
            for (size_t component = 0;
                 component < VectorCellField<Pack>::num_components;
                 ++component)
            {
                const auto delta =
                    local_values(sample.other_lid, component)
                  - owned_values(cell_lid, component);
                gradient[component].x += sample.weight.x * delta;
                gradient[component].y += sample.weight.y * delta;
                gradient[component].z += sample.weight.z * delta;
            }
        }
        for (size_t component = 0;
             component < VectorCellField<Pack>::num_components;
             ++component)
        {
            gradient_values(cell_lid, component * 3) =
                static_cast<scalar_type>(gradient[component].x);
            gradient_values(cell_lid, component * 3 + 1) =
                static_cast<scalar_type>(gradient[component].y);
            gradient_values(cell_lid, component * 3 + 2) =
                static_cast<scalar_type>(gradient[component].z);
        }
    }
}

/** @brief Evaluate cached boundary-aware vector reconstruction weights. */
template<TpetraTypePack Pack, class BoundaryValueProvider>
void cached_vector_cell_gradient(
    const VectorCellField<Pack>& field,
    BoundaryValueProvider& boundary_value,
    TensorCellField<Pack>& gradients,
    const CellGradientCache<Pack>& cache)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using scalar_type = typename Pack::scalar_type;

    const auto& mesh = field.mesh();
    if (&gradients.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "cell_gradient requires input and output fields on one mesh.");
    }
    cache.require_mesh(mesh);

    const auto owned_values = field.owned_read_view();
    const auto local_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    const auto& interior_geometry = cache.interior_geometry();
    const auto& boundary_geometry = cache.boundary_geometry();
    for (size_t owned = 0;
         owned < boundary_geometry.size();
         ++owned)
    {
        const auto& cell_geometry =
            boundary_geometry[owned].boundary_samples.empty()
          ? interior_geometry[owned]
          : boundary_geometry[owned];
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        std::array<typename Mesh<Pack>::Vec3, 3> gradient{};
        for (const auto& sample : cell_geometry.interior_samples)
        {
            for (size_t component = 0;
                 component < VectorCellField<Pack>::num_components;
                 ++component)
            {
                const auto delta =
                    local_values(sample.other_lid, component)
                  - owned_values(cell_lid, component);
                gradient[component].x += sample.weight.x * delta;
                gradient[component].y += sample.weight.y * delta;
                gradient[component].z += sample.weight.z * delta;
            }
        }
        for (const auto& sample : cell_geometry.boundary_samples)
        {
            const auto value = static_cast<
                typename VectorCellField<Pack>::vec_type>(
                    boundary_value(
                        sample.location.batch_id,
                        sample.location.in_batch_id));
            for (size_t component = 0;
                 component < VectorCellField<Pack>::num_components;
                 ++component)
            {
                const auto delta =
                    static_cast<scalar_type>(
                        value.component(component))
                  - owned_values(cell_lid, component);
                gradient[component].x += sample.weight.x * delta;
                gradient[component].y += sample.weight.y * delta;
                gradient[component].z += sample.weight.z * delta;
            }
        }
        for (size_t component = 0;
             component < VectorCellField<Pack>::num_components;
             ++component)
        {
            gradient_values(cell_lid, component * 3) =
                static_cast<scalar_type>(gradient[component].x);
            gradient_values(cell_lid, component * 3 + 1) =
                static_cast<scalar_type>(gradient[component].y);
            gradient_values(cell_lid, component * 3 + 2) =
                static_cast<scalar_type>(gradient[component].z);
        }
    }
}

/**
 * @brief Reconstruct scalar gradients with optional boundary samples.
 * @tparam Pack Tpetra type pack used by the fields and mesh.
 * @param field Scalar field to differentiate.
 * @param boundary_condition Optional boundary-condition provider.
 * @param boundary_value Optional boundary-value provider.
 * @param[out] gradients Reconstructed cell gradients.
 * @param cached_boundary_locations Optional precomputed boundary-face lookup.
 * @throws std::invalid_argument if fields or cached locations use another mesh.
 */
template<TpetraTypePack Pack>
void scalar_cell_gradient(
    const CellField<Pack>& field,
    const std::function<BoundaryCondition(int, size_t)>*
        boundary_condition,
    const std::function<typename Pack::scalar_type(int, size_t)>*
        boundary_value,
    VectorCellField<Pack>& gradients,
    const std::vector<BoundaryFaceLocation<Mesh<Pack>>>*
        cached_boundary_locations = nullptr)
{
    using mesh_type = Mesh<Pack>;
    using local_ordinal_type = typename mesh_type::local_ordinal_type;

    const auto& mesh = field.mesh();
    if (&gradients.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "cell_gradient requires input and output fields on one mesh.");
    }

    std::vector<BoundaryFaceLocation<mesh_type>> local_boundary_locations;
    auto boundary_locations = cached_boundary_locations;
    if (boundary_condition != nullptr && boundary_locations == nullptr)
    {
        local_boundary_locations = boundary_face_locations(mesh);
        boundary_locations = &local_boundary_locations;
    }
    if (boundary_locations != nullptr
        && boundary_locations->size() != mesh.num_faces())
    {
        throw std::invalid_argument(
            "cell_gradient received boundary locations for another mesh.");
    }
    const auto owned_values = field.owned_read_view();
    const auto local_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto phi_p = owned_values(cell_lid, 0);

        std::array<std::array<real_t, 3>, 3> normal{};
        typename mesh_type::Vec3 rhs{};

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            typename mesh_type::Vec3 d{};
            typename Pack::scalar_type phi_delta{};
            if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(
                        face_lid, cell_lid);
                d = mesh.cell_center_vector(face_lid, cell_lid);
                phi_delta = local_values(other, 0) - phi_p;
            }
            else
            {
                if (boundary_condition == nullptr
                    || boundary_value == nullptr
                    || !mesh.is_boundary_face(face_lid)
                    || static_cast<size_t>(face_lid)
                       >= boundary_locations->size())
                {
                    continue;
                }
                const auto location =
                    (*boundary_locations)[static_cast<size_t>(face_lid)];
                if (!location.active)
                {
                    continue;
                }
                const auto condition = (*boundary_condition)(
                    location.batch_id, location.in_batch_id);
                d = mesh.face_centroid(face_lid)
                  - mesh.cell_centroid(cell_lid);
                if (condition.type == BoundaryConditionType::Dirichlet)
                {
                    phi_delta = (*boundary_value)(
                                    location.batch_id,
                                    location.in_batch_id)
                              - phi_p;
                }
                else if (condition.type == BoundaryConditionType::Neumann)
                {
                    phi_delta =
                        condition.value
                      * boundary_normal_distance(
                            mesh, face_lid, cell_lid);
                }
                else
                {
                    throw std::invalid_argument(
                        "cell_gradient supports only Dirichlet and Neumann "
                        "scalar boundary conditions.");
                }
            }

            normal[0][0] += d.x * d.x;
            normal[0][1] += d.x * d.y;
            normal[0][2] += d.x * d.z;
            normal[1][1] += d.y * d.y;
            normal[1][2] += d.y * d.z;
            normal[2][2] += d.z * d.z;

            rhs.x += d.x * phi_delta;
            rhs.y += d.y * phi_delta;
            rhs.z += d.z * phi_delta;
        }

        normal[1][0] = normal[0][1];
        normal[2][0] = normal[0][2];
        normal[2][1] = normal[1][2];
        const auto gradient = solve_3x3(normal, rhs);
        gradient_values(cell_lid, 0) = gradient.x;
        gradient_values(cell_lid, 1) = gradient.y;
        gradient_values(cell_lid, 2) = gradient.z;
    }
}

/** Preserve the cached map-based reconstruction path used by face fluxes. */
template<TpetraTypePack Pack>
void scalar_cell_gradient(
    const CellField<Pack>& field,
    const BoundaryConditionMap* boundary_conditions,
    VectorCellField<Pack>& gradients,
    const std::vector<BoundaryFaceLocation<Mesh<Pack>>>*
        cached_boundary_locations = nullptr)
{
    if (boundary_conditions == nullptr)
    {
        scalar_cell_gradient(
            field, nullptr, nullptr, gradients,
            cached_boundary_locations);
        return;
    }

    std::function<BoundaryCondition(int, size_t)> boundary_condition =
        [&](int batch_id, size_t)
    {
        const auto& name = field.mesh().boundary_batch_name(batch_id);
        const auto iter = boundary_conditions->find(name);
        return iter == boundary_conditions->end()
             ? BoundaryCondition{}
             : iter->second;
    };
    std::function<typename Pack::scalar_type(int, size_t)> boundary_value =
        [&](int batch_id, size_t in_batch_id)
    {
        return boundary_condition(batch_id, in_batch_id).value;
    };
    scalar_cell_gradient(
        field, &boundary_condition, &boundary_value, gradients,
        cached_boundary_locations);
}

} // namespace detail

/**
 * @brief Compute a least-squares cell-centered gradient for every owned
 *        cell.
 *
 * @tparam Pack The Tpetra type pack.
 * @param field Scalar cell field whose gradient is computed.
 * @param[out] gradients Vector cell field to receive the gradient at
 *        each owned cell.
 */
template<TpetraTypePack Pack>
void cell_gradient(const CellField<Pack>& field,
                   VectorCellField<Pack>& gradients)
{
    detail::scalar_cell_gradient(field, nullptr, nullptr, gradients);
}

/**
 * @brief Compute an interior-only scalar gradient from cached mesh weights.
 *
 * This overload performs no reconstruction-system solves or dynamic
 * allocations. The caller owns the cache so its construction can be hoisted
 * to the same lifetime as the equation or solver workspace.
 *
 * @param field Scalar cell field whose gradient is computed.
 * @param[out] gradients Vector cell field receiving the gradient.
 * @param cache Mesh-bound least-squares weights.
 */
template<TpetraTypePack Pack>
void cell_gradient(
    const CellField<Pack>& field,
    VectorCellField<Pack>& gradients,
    const CellGradientCache<Pack>& cache)
{
    detail::cached_scalar_cell_gradient(
        field, gradients, cache);
}

/**
 * @brief Compute a least-squares scalar gradient including boundary data.
 *
 * Dirichlet values are sampled at boundary-face centroids. Neumann values
 * are interpreted as outward normal derivatives. Missing batch names,
 * including every batch in an empty map, explicitly default to homogeneous
 * Neumann conditions. Use the overload without a boundary map to omit
 * boundary samples from the reconstruction.
 *
 * @tparam Pack The Tpetra type pack.
 * @param field Scalar cell field whose gradient is computed.
 * @param boundary_conditions Boundary conditions keyed by mesh batch name.
 * @param[out] gradients Vector cell field receiving the reconstructed gradient.
 */
template<TpetraTypePack Pack>
void cell_gradient(
    const CellField<Pack>& field,
    const BoundaryConditionMap& boundary_conditions,
    VectorCellField<Pack>& gradients)
{
    std::function<BoundaryCondition(int, size_t)> boundary_condition =
        [&](int batch_id, size_t)
    {
        const auto& name = field.mesh().boundary_batch_name(batch_id);
        const auto iter = boundary_conditions.find(name);
        return iter == boundary_conditions.end()
             ? BoundaryCondition{}
             : iter->second;
    };
    std::function<typename Pack::scalar_type(int, size_t)> boundary_value =
        [&](int batch_id, size_t in_batch_id)
    {
        return boundary_condition(batch_id, in_batch_id).value;
    };
    detail::scalar_cell_gradient(
        field,
        &boundary_condition,
        &boundary_value,
        gradients);
}

/**
 * @brief Compute a boundary-aware scalar gradient from cached mesh weights.
 *
 * Boundary condition values remain dynamic and are evaluated on each call;
 * only mesh-dependent reconstruction weights are reused.
 *
 * @param field Scalar cell field whose gradient is computed.
 * @param boundary_conditions Boundary conditions keyed by mesh batch name.
 * @param[out] gradients Vector cell field receiving the gradient.
 * @param cache Mesh-bound least-squares weights.
 */
template<TpetraTypePack Pack>
void cell_gradient(
    const CellField<Pack>& field,
    const BoundaryConditionMap& boundary_conditions,
    VectorCellField<Pack>& gradients,
    const CellGradientCache<Pack>& cache)
{
    auto boundary_condition =
        [&](int batch_id, size_t)
    {
        const auto& name =
            field.mesh().boundary_batch_name(batch_id);
        const auto iter = boundary_conditions.find(name);
        return iter == boundary_conditions.end()
             ? BoundaryCondition{}
             : iter->second;
    };
    auto boundary_value =
        [&](int batch_id, size_t in_batch_id)
    {
        return boundary_condition(
            batch_id, in_batch_id).value;
    };
    detail::cached_scalar_cell_gradient(
        field,
        boundary_condition,
        boundary_value,
        gradients,
        cache);
}

/**
 * @brief Compute a scalar gradient using dynamic per-face boundary data.
 *
 * Dirichlet values are supplied independently from the condition object so
 * wall treatments can update face values without rebuilding a boundary map.
 * Neumann derivatives continue to use BoundaryCondition::value.
 */
template<TpetraTypePack Pack,
         class BoundaryConditionProvider,
         class BoundaryValueProvider>
    requires requires(BoundaryConditionProvider condition,
                      BoundaryValueProvider value,
                      int batch_id,
                      size_t in_batch_id)
    {
        { condition(batch_id, in_batch_id) }
            -> std::convertible_to<BoundaryCondition>;
        { value(batch_id, in_batch_id) }
            -> std::convertible_to<typename Pack::scalar_type>;
    }
void cell_gradient(const CellField<Pack>& field,
                   BoundaryConditionProvider boundary_condition,
                   BoundaryValueProvider boundary_value,
                   VectorCellField<Pack>& gradients)
{
    std::function<BoundaryCondition(int, size_t)> condition =
        std::move(boundary_condition);
    std::function<typename Pack::scalar_type(int, size_t)> value =
        std::move(boundary_value);
    detail::scalar_cell_gradient(
        field, &condition, &value, gradients);
}

/**
 * @brief Compute a dynamic-boundary scalar gradient from cached weights.
 *
 * @param field Scalar field whose gradient is reconstructed.
 * @param boundary_condition Dynamic boundary-condition provider.
 * @param boundary_value Dynamic Dirichlet-value provider.
 * @param[out] gradients Reconstructed owned-cell gradients.
 * @param cache Mesh-bound least-squares weights.
 */
template<TpetraTypePack Pack,
         class BoundaryConditionProvider,
         class BoundaryValueProvider>
    requires requires(BoundaryConditionProvider condition,
                      BoundaryValueProvider value,
                      int batch_id,
                      size_t in_batch_id)
    {
        { condition(batch_id, in_batch_id) }
            -> std::convertible_to<BoundaryCondition>;
        { value(batch_id, in_batch_id) }
            -> std::convertible_to<typename Pack::scalar_type>;
    }
void cell_gradient(
    const CellField<Pack>& field,
    BoundaryConditionProvider boundary_condition,
    BoundaryValueProvider boundary_value,
    VectorCellField<Pack>& gradients,
    const CellGradientCache<Pack>& cache)
{
    detail::cached_scalar_cell_gradient(
        field,
        boundary_condition,
        boundary_value,
        gradients,
        cache);
}

/**
 * @brief Compute least-squares cell-centered gradients for each component
 *        of a vector field.
 *
 * The tensor rows are component-major: gradients.value(cell)[0] is the
 * gradient of the x component, gradients.value(cell)[1] of y, and
 * gradients.value(cell)[2] of z.
 *
 * @tparam Pack The Tpetra type pack.
 * @param field Vector cell field whose component gradients are computed.
 * @param[out] gradients Tensor cell field to receive one 3x3 gradient per
 *        owned cell.
 */
template<TpetraTypePack Pack>
void cell_gradient(const VectorCellField<Pack>& field,
                   TensorCellField<Pack>& gradients)
{
    using mesh_type = Mesh<Pack>;
    using local_ordinal_type = typename mesh_type::local_ordinal_type;
    using tensor_type = typename TensorCellField<Pack>::tensor_type;

    const auto& mesh = field.mesh();
    if (&gradients.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "cell_gradient requires input and output fields on one mesh.");
    }

    const auto owned_values = field.owned_read_view();
    const auto local_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);

        std::array<std::array<real_t, 3>, 3> normal{};
        tensor_type rhs{};

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (!mesh.is_interior_face(face_lid))
            {
                continue;
            }

            const auto other =
                mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
            const auto d = mesh.cell_center_vector(face_lid, cell_lid);

            normal[0][0] += d.x * d.x;
            normal[0][1] += d.x * d.y;
            normal[0][2] += d.x * d.z;
            normal[1][1] += d.y * d.y;
            normal[1][2] += d.y * d.z;
            normal[2][2] += d.z * d.z;

            for (size_t component = 0;
                 component < VectorCellField<Pack>::num_components;
                 ++component)
            {
                const auto delta =
                    local_values(other, component)
                  - owned_values(cell_lid, component);
                rhs[component].x += d.x * delta;
                rhs[component].y += d.y * delta;
                rhs[component].z += d.z * delta;
            }
        }

        normal[1][0] = normal[0][1];
        normal[2][0] = normal[0][2];
        normal[2][1] = normal[1][2];
        tensor_type gradient{};
        for (size_t component = 0;
             component < VectorCellField<Pack>::num_components;
             ++component)
        {
            auto component_normal = normal;
            gradient[component] =
                detail::solve_3x3(component_normal, rhs[component]);
            gradient_values(cell_lid, component * 3) =
                gradient[component].x;
            gradient_values(cell_lid, component * 3 + 1) =
                gradient[component].y;
            gradient_values(cell_lid, component * 3 + 2) =
                gradient[component].z;
        }
    }
}

/**
 * @brief Compute interior-only vector gradients from cached mesh weights.
 *
 * @param field Vector field whose component gradients are reconstructed.
 * @param[out] gradients Row-major tensor field receiving the gradients.
 * @param cache Mesh-bound least-squares weights.
 */
template<TpetraTypePack Pack>
void cell_gradient(
    const VectorCellField<Pack>& field,
    TensorCellField<Pack>& gradients,
    const CellGradientCache<Pack>& cache)
{
    detail::cached_vector_cell_gradient(
        field, gradients, cache);
}

/**
 * @brief Compute least-squares vector gradients including boundary values.
 *
 * Boundary values are sampled at boundary-face centroids. The callback is
 * indexed by boundary-batch ID and the face index within that batch. Use the
 * overload without a callback to omit boundary samples from reconstruction.
 *
 * @tparam Pack The Tpetra type pack.
 * @tparam BoundaryValueProvider Callback returning a vector boundary value.
 * @param field Vector cell field whose component gradients are computed.
 * @param boundary_value Boundary-face value provider.
 * @param[out] gradients Tensor cell field receiving the reconstructed gradient.
 */
template<TpetraTypePack Pack, class BoundaryValueProvider>
    requires requires(BoundaryValueProvider provider,
                      int batch_id,
                      size_t in_batch_id)
    {
        { provider(batch_id, in_batch_id) }
            -> std::convertible_to<
                typename VectorCellField<Pack>::vec_type>;
    }
void cell_gradient(const VectorCellField<Pack>& field,
                   BoundaryValueProvider boundary_value,
                   TensorCellField<Pack>& gradients)
{
    using mesh_type = Mesh<Pack>;
    using local_ordinal_type = typename mesh_type::local_ordinal_type;
    using tensor_type = typename TensorCellField<Pack>::tensor_type;
    using vec_type = typename VectorCellField<Pack>::vec_type;

    const auto& mesh = field.mesh();
    if (&gradients.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "cell_gradient requires input and output fields on one mesh.");
    }

    const auto boundary_locations =
        detail::boundary_face_locations(mesh);
    const auto owned_values = field.owned_read_view();
    const auto local_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const vec_type value_p{
            owned_values(cell_lid, 0),
            owned_values(cell_lid, 1),
            owned_values(cell_lid, 2)};

        std::array<std::array<real_t, 3>, 3> normal{};
        tensor_type rhs{};

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            typename mesh_type::Vec3 d{};
            vec_type value_delta{};
            if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(
                        face_lid, cell_lid);
                d = mesh.cell_center_vector(face_lid, cell_lid);
                value_delta = vec_type{
                    local_values(other, 0),
                    local_values(other, 1),
                    local_values(other, 2)}
                            - value_p;
            }
            else
            {
                if (!mesh.is_boundary_face(face_lid)
                    || static_cast<size_t>(face_lid)
                       >= boundary_locations.size())
                {
                    continue;
                }
                const auto location =
                    boundary_locations[static_cast<size_t>(face_lid)];
                if (!location.active)
                {
                    continue;
                }
                d = mesh.face_centroid(face_lid)
                  - mesh.cell_centroid(cell_lid);
                value_delta =
                    static_cast<vec_type>(boundary_value(
                        location.batch_id,
                        location.in_batch_id))
                  - value_p;
            }

            normal[0][0] += d.x * d.x;
            normal[0][1] += d.x * d.y;
            normal[0][2] += d.x * d.z;
            normal[1][1] += d.y * d.y;
            normal[1][2] += d.y * d.z;
            normal[2][2] += d.z * d.z;

            for (size_t component = 0;
                 component < VectorCellField<Pack>::num_components;
                 ++component)
            {
                const auto delta = value_delta.component(component);
                rhs[component].x += d.x * delta;
                rhs[component].y += d.y * delta;
                rhs[component].z += d.z * delta;
            }
        }

        normal[1][0] = normal[0][1];
        normal[2][0] = normal[0][2];
        normal[2][1] = normal[1][2];
        tensor_type gradient{};
        for (size_t component = 0;
             component < VectorCellField<Pack>::num_components;
             ++component)
        {
            auto component_normal = normal;
            gradient[component] =
                detail::solve_3x3(component_normal, rhs[component]);
            gradient_values(cell_lid, component * 3) =
                gradient[component].x;
            gradient_values(cell_lid, component * 3 + 1) =
                gradient[component].y;
            gradient_values(cell_lid, component * 3 + 2) =
                gradient[component].z;
        }
    }
}

/**
 * @brief Compute boundary-aware vector gradients from cached mesh weights.
 *
 * @param field Vector field whose component gradients are reconstructed.
 * @param boundary_value Dynamic boundary-face value provider.
 * @param[out] gradients Row-major tensor field receiving the gradients.
 * @param cache Mesh-bound least-squares weights.
 */
template<TpetraTypePack Pack, class BoundaryValueProvider>
    requires requires(BoundaryValueProvider provider,
                      int batch_id,
                      size_t in_batch_id)
    {
        { provider(batch_id, in_batch_id) }
            -> std::convertible_to<
                typename VectorCellField<Pack>::vec_type>;
    }
void cell_gradient(
    const VectorCellField<Pack>& field,
    BoundaryValueProvider boundary_value,
    TensorCellField<Pack>& gradients,
    const CellGradientCache<Pack>& cache)
{
    detail::cached_vector_cell_gradient(
        field, boundary_value, gradients, cache);
}

/**
 * @brief Compute a Gauss-linear scalar gradient with dynamic boundary data.
 *
 * Interior face values use distance-weighted linear interpolation. Dirichlet
 * faces use their prescribed value, while Neumann faces extrapolate from the
 * owner cell using the prescribed outward-normal derivative.
 */
template<TpetraTypePack Pack,
         class BoundaryConditionProvider,
         class BoundaryValueProvider>
void gauss_linear_cell_gradient(
    const CellField<Pack>& field,
    BoundaryConditionProvider boundary_condition,
    BoundaryValueProvider boundary_value,
    VectorCellField<Pack>& gradients)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    const auto& mesh = field.mesh();
    if (&gradients.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "gauss_linear_cell_gradient requires input and output fields "
            "on one mesh.");
    }
    const auto boundary_locations =
        detail::boundary_face_locations(mesh);
    const auto local_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto cell_value = local_values(cell_lid, 0);
        typename VectorCellField<Pack>::vec_type gradient{};
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            scalar_type face_value = cell_value;
            if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(
                        face_lid, cell_lid);
                const auto cell_distance =
                    mesh.cell_to_face_distance(face_lid, cell_lid);
                const auto other_distance =
                    mesh.cell_to_face_distance(face_lid, other);
                const auto total_distance =
                    cell_distance + other_distance;
                face_value =
                    total_distance > scalar_type{}
                  ? (other_distance * cell_value
                     + cell_distance * local_values(other, 0))
                        / total_distance
                  : scalar_type{0.5}
                        * (cell_value + local_values(other, 0));
            }
            else if (mesh.is_boundary_face(face_lid))
            {
                const auto location = boundary_locations.at(
                    static_cast<size_t>(face_lid));
                if (location.active)
                {
                    const auto condition = boundary_condition(
                        location.batch_id, location.in_batch_id);
                    switch (condition.type)
                    {
                        case BoundaryConditionType::Dirichlet:
                            face_value = boundary_value(
                                location.batch_id,
                                location.in_batch_id);
                            break;
                        case BoundaryConditionType::Neumann:
                            face_value =
                                cell_value
                              + static_cast<scalar_type>(
                                    condition.value)
                              * static_cast<scalar_type>(
                                    detail::boundary_normal_distance(
                                        mesh, face_lid, cell_lid));
                            break;
                        default:
                            throw std::invalid_argument(
                                "Gauss-linear scalar gradients support only "
                                "Dirichlet and Neumann boundaries.");
                    }
                }
            }
            gradient =
                gradient
              + mesh.face_area_vector_outward(face_lid, cell_lid)
                    * face_value;
        }
        gradient =
            gradient
          / static_cast<scalar_type>(
                mesh.cell_volume(cell_lid));
        gradient_values(cell_lid, 0) = gradient.x;
        gradient_values(cell_lid, 1) = gradient.y;
        gradient_values(cell_lid, 2) = gradient.z;
    }
}

/** @brief Compute a Gauss-linear scalar gradient from a boundary map. */
template<TpetraTypePack Pack>
void gauss_linear_cell_gradient(
    const CellField<Pack>& field,
    const BoundaryConditionMap& boundary_conditions,
    VectorCellField<Pack>& gradients)
{
    auto boundary_condition = [&](int batch_id, size_t)
    {
        const auto& name =
            field.mesh().boundary_batch_name(batch_id);
        const auto iter = boundary_conditions.find(name);
        return iter == boundary_conditions.end()
             ? BoundaryCondition{}
             : iter->second;
    };
    auto boundary_value = [&](int batch_id, size_t in_batch_id)
    {
        return static_cast<typename Pack::scalar_type>(
            boundary_condition(batch_id, in_batch_id).value);
    };
    gauss_linear_cell_gradient(
        field, boundary_condition, boundary_value, gradients);
}

/** @brief Compute an interior Gauss-linear gradient with zero-normal walls. */
template<TpetraTypePack Pack>
void gauss_linear_cell_gradient(
    const CellField<Pack>& field,
    VectorCellField<Pack>& gradients)
{
    auto boundary_condition = [](int, size_t)
    {
        return BoundaryCondition{};
    };
    auto boundary_value = [](int, size_t)
    {
        return typename Pack::scalar_type{};
    };
    gauss_linear_cell_gradient(
        field, boundary_condition, boundary_value, gradients);
}

/**
 * @brief Reconstruct a boundary-aware scalar gradient with a selected scheme.
 *
 * The least-squares path reuses @p cache.  Gauss-linear reconstruction does
 * not need the cache, but accepts the same call surface so solver-level
 * gradient selection remains scoped and explicit.
 */
template<TpetraTypePack Pack>
void cell_gradient(
    const CellField<Pack>& field,
    const BoundaryConditionMap& boundary_conditions,
    VectorCellField<Pack>& gradients,
    const CellGradientCache<Pack>& cache,
    CellGradientScheme scheme)
{
    if (scheme == CellGradientScheme::GaussLinear)
    {
        gauss_linear_cell_gradient(
            field, boundary_conditions, gradients);
        return;
    }
    cell_gradient(
        field, boundary_conditions, gradients, cache);
}

/**
 * @brief Compute a Gauss-linear vector gradient using boundary face values.
 */
template<TpetraTypePack Pack, class BoundaryValueProvider>
void gauss_linear_cell_gradient(
    const VectorCellField<Pack>& field,
    BoundaryValueProvider boundary_value,
    TensorCellField<Pack>& gradients)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = typename VectorCellField<Pack>::vec_type;

    const auto& mesh = field.mesh();
    if (&gradients.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "gauss_linear_cell_gradient requires input and output fields "
            "on one mesh.");
    }
    const auto boundary_locations =
        detail::boundary_face_locations(mesh);
    const auto local_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const vec_type cell_value{
            local_values(cell_lid, 0),
            local_values(cell_lid, 1),
            local_values(cell_lid, 2)};
        typename TensorCellField<Pack>::tensor_type gradient{};
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            auto face_value = cell_value;
            if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(
                        face_lid, cell_lid);
                const auto cell_distance =
                    mesh.cell_to_face_distance(face_lid, cell_lid);
                const auto other_distance =
                    mesh.cell_to_face_distance(face_lid, other);
                const auto total_distance =
                    cell_distance + other_distance;
                const vec_type other_value{
                    local_values(other, 0),
                    local_values(other, 1),
                    local_values(other, 2)};
                face_value =
                    total_distance > scalar_type{}
                  ? (cell_value * other_distance
                     + other_value * cell_distance)
                        / total_distance
                  : (cell_value + other_value)
                        / scalar_type{2};
            }
            else if (mesh.is_boundary_face(face_lid))
            {
                const auto location = boundary_locations.at(
                    static_cast<size_t>(face_lid));
                if (location.active)
                {
                    face_value = boundary_value(
                        location.batch_id,
                        location.in_batch_id);
                }
            }
            const auto area =
                mesh.face_area_vector_outward(face_lid, cell_lid);
            gradient[0] =
                gradient[0] + area * face_value.x;
            gradient[1] =
                gradient[1] + area * face_value.y;
            gradient[2] =
                gradient[2] + area * face_value.z;
        }
        const auto inverse_volume =
            scalar_type{1}
          / static_cast<scalar_type>(
                mesh.cell_volume(cell_lid));
        for (size_t component = 0;
             component < VectorCellField<Pack>::num_components;
             ++component)
        {
            gradient[component] =
                gradient[component] * inverse_volume;
            gradient_values(cell_lid, component * 3) =
                gradient[component].x;
            gradient_values(cell_lid, component * 3 + 1) =
                gradient[component].y;
            gradient_values(cell_lid, component * 3 + 2) =
                gradient[component].z;
        }
    }
}

/**
 * @brief Compute the net flux balance (sum of signed face fluxes) for a
 *        single cell from a FaceField.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param face_fluxes FaceField of scalar fluxes.
 * @param cell_lid Local ID of the cell whose balance is computed.
 * @return Sum of outward-positive fluxes around @p cell_lid.
 */
template<TpetraTypePack Pack, class FaceValues>
typename Pack::scalar_type cell_flux_balance(
    const Mesh<Pack>& mesh,
    const FaceField<Pack>& face_fluxes,
    const FaceValues& face_values,
    typename Pack::local_ordinal_type cell_lid)
{
    typename Pack::scalar_type balance = 0.0;
    for (const auto face_lid : mesh.faces(cell_lid))
    {
        if (!face_fluxes.is_owned_face(face_lid))
        {
            continue;
        }

        const auto sign = mesh.owner_cell(face_lid) == cell_lid ? 1.0 : -1.0;
        balance += sign * face_values(face_lid, 0);
    }

    return balance;
}

/**
 * @brief Compute one cell's flux balance using a one-shot face-field view.
 */
template<TpetraTypePack Pack>
typename Pack::scalar_type cell_flux_balance(
    const Mesh<Pack>& mesh,
    const FaceField<Pack>& face_fluxes,
    typename Pack::local_ordinal_type cell_lid)
{
    const auto face_values = face_fluxes.owned_read_view();
    return cell_flux_balance(
        mesh, face_fluxes, face_values, cell_lid);
}

/**
 * @brief Compute the volume-normalized divergence at every owned cell
 *        from a FaceField of pre-computed face fluxes.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param face_fluxes FaceField of scalar fluxes.
 * @return Vector of divergence values indexed by owned-cell local ID.
 */
template<TpetraTypePack Pack>
std::vector<typename Pack::scalar_type>
cell_divergence_from_fluxes(
    const Mesh<Pack>& mesh,
    const FaceField<Pack>& face_fluxes)
{
    std::vector<typename Pack::scalar_type> divergence(mesh.num_owned_cells(), 0.0);
    const auto face_values = face_fluxes.owned_read_view();
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<typename Pack::local_ordinal_type>(owned);
        divergence[owned] =
            cell_flux_balance(
                mesh, face_fluxes, face_values, cell_lid)
            / mesh.cell_volume(cell_lid);
    }

    return divergence;
}

} // namespace SimpleFluid::FVM
