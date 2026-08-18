/**
 * @file FVM/details/TransportGeometryCache.hh
 * @brief Header-visible generic transport-geometry cache implementation.
 */

#pragma once

namespace SimpleFluid::FVM
{

template<class MeshType>
TransportGeometryCache<MeshType>::TransportGeometryCache(const MeshType& mesh)
    : d_mesh(&mesh), d_interior_stencils(detail::least_squares_gradient_stencils(mesh)),
      d_boundary_locations(detail::boundary_face_locations(mesh)),
      d_boundary_geometry(detail::boundary_aware_gradient_geometry(mesh, d_boundary_locations))
{
}

template<class MeshType> void TransportGeometryCache<MeshType>::require_mesh(const MeshType& mesh) const
{
    if (&mesh != d_mesh)
    {
        throw std::invalid_argument("transport geometry cache belongs to another mesh.");
    }
}

template<class MeshType>
const typename TransportGeometryCache<MeshType>::interior_stencils_type&
TransportGeometryCache<MeshType>::interior_stencils() const noexcept
{
    return d_interior_stencils;
}

template<class MeshType>
const typename TransportGeometryCache<MeshType>::boundary_locations_type&
TransportGeometryCache<MeshType>::boundary_locations() const noexcept
{
    return d_boundary_locations;
}

template<class MeshType>
const typename TransportGeometryCache<MeshType>::boundary_geometry_type&
TransportGeometryCache<MeshType>::boundary_geometry() const noexcept
{
    return d_boundary_geometry;
}

template<class MeshType>
std::vector<detail::AffineLeastSquaresGradientStencil<MeshType>>
TransportGeometryCache<MeshType>::scalar_affine_stencils(
    std::function<BoundaryCondition(int, size_t)> boundary_condition,
    std::function<typename MeshType::scalar_type(int, size_t)> boundary_value) const
{
    return detail::materialize_scalar_affine_gradient_stencils<MeshType>(
        d_boundary_geometry, std::move(boundary_condition), std::move(boundary_value));
}

template<class MeshType>
std::vector<detail::VectorAffineLeastSquaresGradientStencil<MeshType>>
TransportGeometryCache<MeshType>::vector_affine_stencils(
    std::function<typename MeshType::Vec3(int, size_t)> boundary_value) const
{
    return detail::materialize_vector_affine_gradient_stencils<MeshType>(
        d_boundary_geometry, std::move(boundary_value));
}

} // namespace SimpleFluid::FVM
