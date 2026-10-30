#include "geometry/PLIC/GeometryIdentity.hh"

namespace SimpleFluid::PLIC::details
{
struct GeometryIdentityBacking
{
    std::vector<char> key;
    GeometryIdentityBackingPtr parent;
};
GeometryIdentityBackingPtr make_geometry_identity_backing(std::vector<char> key, GeometryIdentityBackingPtr parent)
{
    return std::make_shared<const GeometryIdentityBacking>(std::move(key), std::move(parent));
}
const std::vector<char>& geometry_identity_key(const GeometryIdentityBackingPtr& owner) noexcept
{
    static const std::vector<char> empty;
    return owner ? owner->key : empty;
}
std::size_t geometry_identity_backing_bytes() noexcept
{
    return sizeof(GeometryIdentityBacking);
}
std::size_t geometry_identity_payload_bytes(const GeometryIdentityBackingPtr& owner) noexcept
{
    // Match retained-construction accounting: backing structs and buffer
    // capacities, excluding allocator/control-block overhead and RSS.
    return owner ? sizeof(GeometryIdentityBacking) + owner->key.capacity() : 0;
}
GeometryIdentityBackingPtr geometry_identity_parent(const GeometryIdentityBackingPtr& owner) noexcept
{
    return owner ? owner->parent : GeometryIdentityBackingPtr{};
}
} // namespace SimpleFluid::PLIC::details
