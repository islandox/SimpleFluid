/**
 * @file FVM/details/TransportGeometryCache.hh
 * @brief Header-visible generic transport-geometry cache implementation.
 */

#pragma once

namespace SimpleFluid::FVM
{

template<class MeshType>
TransportGeometryCache<MeshType>::SharedGeometry::SharedGeometry(const MeshType& source)
    : mesh(&source), identity(detail::ale_geometry_identity(source)), epoch(mesh_geometry_epoch(source)),
      interior(detail::least_squares_gradient_stencils(source)),
      locations(detail::boundary_face_locations(source)),
      boundary(detail::boundary_aware_gradient_geometry(source, locations)),
      assembly(detail::transport_assembly_geometry(source)),
      has_non_orthogonal_faces(detail::stored_transport_has_non_orthogonal_faces(source))
{
}

template<class MeshType>
TransportGeometryCache<MeshType>::TransportGeometryCache(const MeshType& mesh) : d_mesh(&mesh)
{
    refresh();
}

template<class MeshType> void TransportGeometryCache<MeshType>::require_mesh(const MeshType& mesh) const
{
    if (&mesh != d_mesh)
        throw std::invalid_argument("transport geometry cache belongs to another mesh.");
    if (mesh_geometry_epoch(mesh) != d_geometry->epoch ||
        detail::ale_geometry_identity(mesh) != d_geometry->identity)
        throw std::invalid_argument("transport geometry cache is stale for the mesh geometry epoch.");
}

/** Refresh without a caller-supplied geometry snapshot. */
template<class MeshType>
void TransportGeometryCache<MeshType>::refresh()
{
    refresh({});
}

template<class MeshType>
void TransportGeometryCache<MeshType>::refresh(shared_geometry_type geometry)
{
    const auto execution = acquire_mesh_execution(*d_mesh);
    if (geometry)
    {
        if (geometry->mesh != d_mesh || geometry->epoch != mesh_geometry_epoch(*d_mesh) ||
            geometry->identity != detail::ale_geometry_identity(*d_mesh))
            throw std::invalid_argument("Cannot share foreign or stale transport geometry.");
        d_geometry = std::move(geometry);
    }
    else if (!d_geometry || d_geometry->epoch != mesh_geometry_epoch(*d_mesh) ||
             d_geometry->identity != detail::ale_geometry_identity(*d_mesh))
        d_geometry = std::make_shared<SharedGeometry>(*d_mesh);
}

template<class MeshType>
const typename TransportGeometryCache<MeshType>::interior_stencils_type&
TransportGeometryCache<MeshType>::interior_stencils() const
{
    require_mesh(*d_mesh);
    return d_geometry->interior;
}

template<class MeshType>
const typename TransportGeometryCache<MeshType>::boundary_locations_type&
TransportGeometryCache<MeshType>::boundary_locations() const
{
    require_mesh(*d_mesh);
    return d_geometry->locations;
}

template<class MeshType>
const typename TransportGeometryCache<MeshType>::boundary_geometry_type&
TransportGeometryCache<MeshType>::boundary_geometry() const
{
    require_mesh(*d_mesh);
    return d_geometry->boundary;
}

template<class MeshType>
const typename TransportGeometryCache<MeshType>::assembly_geometry_type&
TransportGeometryCache<MeshType>::assembly_geometry() const
{
    require_mesh(*d_mesh);
    return d_geometry->assembly;
}

template<class MeshType>
bool TransportGeometryCache<MeshType>::has_non_orthogonal_faces() const
{
    require_mesh(*d_mesh);
    return d_geometry->has_non_orthogonal_faces;
}

template<class MeshType>
std::vector<detail::AffineLeastSquaresGradientStencil<MeshType>>
TransportGeometryCache<MeshType>::scalar_affine_stencils(
    std::function<BoundaryCondition(int, size_t)> boundary_condition,
    std::function<typename MeshType::scalar_type(int, size_t)> boundary_value) const
{
    require_mesh(*d_mesh);
    return detail::materialize_scalar_affine_gradient_stencils<MeshType>(
        d_geometry->boundary, std::move(boundary_condition), std::move(boundary_value));
}

template<class MeshType>
std::vector<detail::VectorAffineLeastSquaresGradientStencil<MeshType>>
TransportGeometryCache<MeshType>::vector_affine_stencils(
    std::function<typename MeshType::Vec3(int, size_t)> boundary_value) const
{
    require_mesh(*d_mesh);
    return detail::materialize_vector_affine_gradient_stencils<MeshType>(
        d_geometry->boundary, std::move(boundary_value));
}

} // namespace SimpleFluid::FVM
