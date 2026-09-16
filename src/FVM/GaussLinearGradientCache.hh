/** @file GaussLinearGradientCache.hh @brief Reusable face-ordered Gauss geometry. */
#pragma once

#include "SimpleFluidExport.hh"

#include "FVM/details/OperatorDetails.hh"
#include "geometry/GeometryEpoch.hh"
#include "geometry/Mesh.hh"
#include "geometry/MeshHandle.hh"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::FVM
{

/**
 * @brief Immutable-between-refreshes geometry for Gauss-linear reconstruction.
 *
 * Only mesh data are retained: boundary types and values are evaluated on each
 * application. Face order and interpolation arithmetic match the uncached
 * operator. Motion invalidates the cache until refresh(); topology changes
 * require a new cache. Independent field evaluations may share a const cache.
 */
template<TpetraTypePack Pack, class MeshType = Mesh<Pack>>
class SIMPLEFLUID_FVM_EXPORT GaussLinearGradientCache
{
public:
    using mesh_type = MeshType;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using boundary_location_type = detail::BoundaryFaceLocation<mesh_type>;

    struct FaceGeometry
    {
        local_ordinal_type face_lid{};
        local_ordinal_type other_lid{};
        typename mesh_type::Vec3 area{};
        real_t cell_distance{};
        real_t other_distance{};
        real_t normal_distance{};
        boundary_location_type boundary{};
        bool interior = false;
    };

    struct Geometry
    {
        std::vector<size_t> offsets;
        std::vector<real_t> volumes;
        std::vector<FaceGeometry> faces;
    };

    explicit GaussLinearGradientCache(SP<const mesh_type> mesh);

    const SP<const mesh_type>& mesh_ptr() const noexcept { return d_mesh; }
    std::uint64_t geometry_epoch() const noexcept { return d_geometry_epoch; }

    void require_mesh(const mesh_type& mesh) const;

    const Geometry& geometry() const
    {
        require_mesh(*d_mesh);
        return d_geometry;
    }

    /** @brief Replace all cached geometry after a successful rebuild. */
    void refresh();

private:
    SP<const mesh_type> d_mesh;
    Geometry d_geometry;
    std::uint64_t d_geometry_epoch = 0;
};

extern template class GaussLinearGradientCache<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
extern template class GaussLinearGradientCache<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;

namespace detail
{

/** @brief Shared scalar kernel for stored and legacy fields. */
template<TpetraTypePack Pack, class ScalarField, class BoundaryConditionProvider,
    class BoundaryValueProvider, class VectorField, class MeshType>
void cached_gauss_linear_scalar_gradient(const ScalarField& field,
    BoundaryConditionProvider boundary_condition, BoundaryValueProvider boundary_value,
    VectorField& gradients, const GaussLinearGradientCache<Pack, MeshType>& cache)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    if (&field.mesh() != &gradients.mesh())
        throw std::invalid_argument("gauss_linear_cell_gradient requires input and output fields on one mesh.");
    const auto execution = acquire_mesh_execution(field.mesh());
    cache.require_mesh(field.mesh());
    const auto& geometry = cache.geometry();
    const auto local_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < geometry.volumes.size(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto cell_value = local_values(cell_lid, 0);
        vec3<scalar_type> gradient{};
        for (size_t entry = geometry.offsets[owned]; entry < geometry.offsets[owned + 1]; ++entry)
        {
            const auto& face = geometry.faces[entry];
            scalar_type face_value = cell_value;
            if (face.interior)
            {
                const auto total_distance = face.cell_distance + face.other_distance;
                face_value = total_distance > scalar_type{}
                    ? (face.other_distance * cell_value + face.cell_distance * local_values(face.other_lid, 0)) /
                          total_distance
                    : scalar_type{0.5} * (cell_value + local_values(face.other_lid, 0));
            }
            else if (face.boundary.active)
            {
                const auto& location = face.boundary;
                const auto condition = boundary_condition(location.batch_id, location.in_batch_id);
                if (condition.type == BoundaryConditionType::Dirichlet)
                    face_value = boundary_value(location.batch_id, location.in_batch_id);
                else if (condition.type == BoundaryConditionType::Neumann)
                    face_value = cell_value + static_cast<scalar_type>(condition.value) *
                                                 static_cast<scalar_type>(face.normal_distance);
                else
                    throw std::invalid_argument("Gauss-linear scalar gradients support only "
                                                "Dirichlet and Neumann boundaries.");
            }
            gradient = gradient + face.area * face_value;
        }
        gradient = gradient / static_cast<scalar_type>(geometry.volumes[owned]);
        gradient_values(cell_lid, 0) = gradient.x;
        gradient_values(cell_lid, 1) = gradient.y;
        gradient_values(cell_lid, 2) = gradient.z;
    }
}

/** @brief Shared vector kernel with the original component and face order. */
template<TpetraTypePack Pack, class VectorField, class BoundaryValueProvider, class TensorField, class MeshType>
void cached_gauss_linear_vector_gradient(const VectorField& field,
    BoundaryValueProvider boundary_value, TensorField& gradients,
    const GaussLinearGradientCache<Pack, MeshType>& cache)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = vec3<scalar_type>;
    if (&field.mesh() != &gradients.mesh())
        throw std::invalid_argument("gauss_linear_cell_gradient requires input and output fields on one mesh.");
    const auto execution = acquire_mesh_execution(field.mesh());
    cache.require_mesh(field.mesh());
    const auto& geometry = cache.geometry();
    const auto local_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < geometry.volumes.size(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const vec_type cell_value{local_values(cell_lid, 0), local_values(cell_lid, 1), local_values(cell_lid, 2)};
        std::array<vec_type, 3> gradient{};
        for (size_t entry = geometry.offsets[owned]; entry < geometry.offsets[owned + 1]; ++entry)
        {
            const auto& face = geometry.faces[entry];
            auto face_value = cell_value;
            if (face.interior)
            {
                const vec_type other_value{local_values(face.other_lid, 0),
                    local_values(face.other_lid, 1), local_values(face.other_lid, 2)};
                const auto total_distance = face.cell_distance + face.other_distance;
                face_value = total_distance > scalar_type{}
                    ? (cell_value * face.other_distance + other_value * face.cell_distance) / total_distance
                    : (cell_value + other_value) / scalar_type{2};
            }
            else if (face.boundary.active)
                face_value = boundary_value(face.boundary.batch_id, face.boundary.in_batch_id);
            for (size_t component = 0; component < 3; ++component)
                gradient[component] = gradient[component] + face.area * face_value.component(component);
        }
        const auto inverse_volume = scalar_type{1} / static_cast<scalar_type>(geometry.volumes[owned]);
        for (size_t component = 0; component < 3; ++component)
        {
            gradient[component] = gradient[component] * inverse_volume;
            gradient_values(cell_lid, component * 3) = gradient[component].x;
            gradient_values(cell_lid, component * 3 + 1) = gradient[component].y;
            gradient_values(cell_lid, component * 3 + 2) = gradient[component].z;
        }
    }
}

} // namespace detail
} // namespace SimpleFluid::FVM
