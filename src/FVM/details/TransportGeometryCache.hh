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
      d_boundary_geometry(detail::boundary_aware_gradient_geometry(mesh, d_boundary_locations)),
      d_assembly_geometry(detail::transport_assembly_geometry(mesh)),
      d_has_non_orthogonal_faces(detail::stored_transport_has_non_orthogonal_faces(mesh))
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
    const auto execution = acquire_mesh_execution(*d_mesh);
    auto interior = detail::least_squares_gradient_stencils(*d_mesh);
    auto locations = detail::boundary_face_locations(*d_mesh);
    auto boundary = detail::boundary_aware_gradient_geometry(*d_mesh, locations);
    auto assembly = detail::transport_assembly_geometry(*d_mesh);
    const auto has_non_orthogonal_faces = detail::stored_transport_has_non_orthogonal_faces(*d_mesh);
    d_interior_stencils = std::move(interior);
    d_boundary_locations = std::move(locations);
    d_boundary_geometry = std::move(boundary);
    d_assembly_geometry = std::move(assembly);
    d_has_non_orthogonal_faces = has_non_orthogonal_faces;
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
const typename TransportGeometryCache<MeshType>::assembly_geometry_type&
TransportGeometryCache<MeshType>::assembly_geometry() const
{
    require_mesh(*d_mesh);
    return d_assembly_geometry;
}

template<class MeshType>
bool TransportGeometryCache<MeshType>::has_non_orthogonal_faces() const
{
    require_mesh(*d_mesh);
    return d_has_non_orthogonal_faces;
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
