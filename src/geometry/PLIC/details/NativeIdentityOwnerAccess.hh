#pragma once
#include "geometry/PLIC/details/GeometryIdentityBacking.hh"
#include "geometry/details/NativeConstructorOwnerSink.hh"
namespace SimpleFluid::PLIC::details
{
/** The shared-control path authenticates the actual backing control. The raw
 * path is only a synchronous census borrow while the caller holds real roots;
 * it validates the canonical envelope, live target, and exact target address.
 * Neither path fabricates a shared control around the borrowed address. */
class SIMPLEFLUID_LOCAL NativeIdentityOwnerAccess
{
public:
    using ParentVisit = void (*)(void*, const GeometryIdentityBackingPtr&);
    [[nodiscard]] static const NativeGeometryIdentityBacking* backing(const GeometryIdentityBacking& owner)
    {
        if (owner.disposal_next != &owner)
            return nullptr;
        const auto& backing = static_cast<const NativeGeometryIdentityBacking&>(owner);
        const auto& envelope = backing.envelope();
        if (!backing.storage().owner || !envelope || !envelope->authentic_owner(envelope) ||
            !envelope->object_alive() || envelope->observed_target_address() != &backing)
            throw std::invalid_argument("Incomplete or foreign borrowed identity allocation receipt.");
        return &backing;
    }
    [[nodiscard]] static const NativeGeometryIdentityBacking* backing(const GeometryIdentityBackingPtr& owner)
    {
        const auto* typed = owner ? backing(*owner) : nullptr;
        if (typed)
            geometry_detail::NativeConstructorOwnerSink::require_envelope(
                typed->envelope(), std::shared_ptr<const NativeGeometryIdentityBacking>(owner, typed));
        return typed;
    }
    static bool admitted(const GeometryIdentityBacking& owner) { return bool(backing(owner)); }
    static bool admitted(const GeometryIdentityBackingPtr& owner) { return bool(backing(owner)); }
    static bool visit_borrowed_children(const GeometryIdentityBacking& owner,
        const geometry_detail::NativeConstructorOwnerSink& sink, ParentVisit parent)
    {
        const auto* typed = backing(owner);
        if (!typed)
            return false;
        sink.fixed(typed->envelope());
        sink.storage(typed->storage());
        if (owner.parent)
        {
            if (!parent)
                throw std::invalid_argument("Missing native identity parent visitor.");
            parent(sink.context, owner.parent);
        }
        return true;
    }
    static bool visit_children(const GeometryIdentityBackingPtr& owner,
        const geometry_detail::NativeConstructorOwnerSink& sink, ParentVisit parent)
    {
        if (!admitted(owner))
            return false;
        return visit_borrowed_children(*owner, sink, parent);
    }
};
} // namespace SimpleFluid::PLIC::details
