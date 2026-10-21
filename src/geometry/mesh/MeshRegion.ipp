/** @file MeshRegion.ipp
 * @brief Visible generic region validation and factory definitions.
 */
#pragma once

namespace SimpleFluid::Meshes
{
template<class Native>
EntityRange<typename NativeRegionProvider<Native>::ID> NativeRegionProvider<Native>::cell_faces(size_t c) const
{
    const auto& faces = d_mesh->cell_faces(d_mesh->cell_id(c));
    return {this, c, faces.size(), [](const void* source, size_t c, size_t i) -> ID
    {
        const auto& m = *static_cast<const NativeRegionProvider*>(source)->d_mesh;
        const auto& faces = m.cell_faces(m.cell_id(c));
        return m.face_local_id(faces[i]);
    }};
}

template<TopologyProvider T, GeometryProvider G>
IsoRegion<T, G>::IsoRegion(std::string name, std::shared_ptr<const T> topology, std::shared_ptr<const G> geometry)
    : d_name(std::move(name)), d_topology(std::move(topology)), d_geometry(std::move(geometry))
{
    if (d_name.empty() || !d_topology || !d_geometry)
        throw std::invalid_argument("IsoRegion requires a name and live providers.");
    d_layout = d_topology->layout();
    if (d_layout != d_geometry->layout())
        throw std::invalid_argument("IsoRegion topology/geometry numbering or entity extents mismatch.");
    d_epoch = d_geometry->geometry_epoch();
    d_geometry_identity = d_geometry->storage_identity();
    d_topology_identity = d_topology->storage_identity();
}

template<TopologyProvider T, GeometryProvider G>
void IsoRegion<T, G>::validate_static() const
{
    if (d_geometry_identity != d_geometry->storage_identity() || d_topology_identity != d_topology->storage_identity()
        || d_epoch != d_geometry->geometry_epoch() || d_layout != d_geometry->layout()
        || d_layout != d_topology->layout())
        throw std::logic_error("Static region constituent changed; rebuild MultiRegionMesh and its fields/operators.");
}

/** @brief Retain one native object with two const, zero-copy provider observations. */
template<class Native>
auto native_region(std::string name, std::shared_ptr<Native> mesh)
{
    using N = std::remove_const_t<Native>;
    auto provider = std::make_shared<const NativeRegionProvider<N>>(std::move(mesh));
    return NativeIsoRegion<N>(std::move(name), provider, provider);
}

} // namespace SimpleFluid::Meshes
