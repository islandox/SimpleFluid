/**
 * @file FVM/details/TransportGeometryCache.hh
 * @brief Header-visible generic transport-geometry cache implementation.
 */

#pragma once

namespace SimpleFluid::FVM
{

template<class MeshType>
TransportGeometryCache<MeshType>::TransportGeometryCache(const MeshType& mesh)
    : d_mesh(&mesh), d_geometry_epoch(mesh_geometry_epoch(mesh)),
      d_interior_stencils(detail::least_squares_gradient_stencils(mesh)),
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
    if (mesh_geometry_epoch(mesh) != d_geometry_epoch)
    {
        throw std::invalid_argument("transport geometry cache is stale for the mesh geometry epoch.");
    }
}

template<class MeshType> void TransportGeometryCache<MeshType>::refresh()
{
    auto interior = detail::least_squares_gradient_stencils(*d_mesh);
    auto locations = detail::boundary_face_locations(*d_mesh);
    auto boundary = detail::boundary_aware_gradient_geometry(*d_mesh, locations);
    d_interior_stencils = std::move(interior);
    d_boundary_locations = std::move(locations);
    d_boundary_geometry = std::move(boundary);
    d_geometry_epoch = mesh_geometry_epoch(*d_mesh);
}

template<class MeshType>
const typename TransportGeometryCache<MeshType>::interior_stencils_type&
TransportGeometryCache<MeshType>::interior_stencils() const
{
    require_mesh(*d_mesh);
    return d_interior_stencils;
}

template<class MeshType>
const typename TransportGeometryCache<MeshType>::boundary_locations_type&
TransportGeometryCache<MeshType>::boundary_locations() const
{
    require_mesh(*d_mesh);
    return d_boundary_locations;
}

template<class MeshType>
const typename TransportGeometryCache<MeshType>::boundary_geometry_type&
TransportGeometryCache<MeshType>::boundary_geometry() const
{
    require_mesh(*d_mesh);
    return d_boundary_geometry;
}

template<class MeshType>
std::vector<detail::AffineLeastSquaresGradientStencil<MeshType>>
TransportGeometryCache<MeshType>::scalar_affine_stencils(
    std::function<BoundaryCondition(int, size_t)> boundary_condition,
    std::function<typename MeshType::scalar_type(int, size_t)> boundary_value) const
{
    require_mesh(*d_mesh);
    return detail::materialize_scalar_affine_gradient_stencils<MeshType>(
        d_boundary_geometry, std::move(boundary_condition), std::move(boundary_value));
}

template<class MeshType>
std::vector<detail::VectorAffineLeastSquaresGradientStencil<MeshType>>
TransportGeometryCache<MeshType>::vector_affine_stencils(
    std::function<typename MeshType::Vec3(int, size_t)> boundary_value) const
{
    require_mesh(*d_mesh);
    return detail::materialize_vector_affine_gradient_stencils<MeshType>(
        d_boundary_geometry, std::move(boundary_value));
}

} // namespace SimpleFluid::FVM
