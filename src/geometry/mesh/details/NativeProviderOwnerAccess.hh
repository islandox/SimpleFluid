#pragma once
#include "geometry/PLIC/details/ExtrudedConstructionOwnerAccess.hh"
#include "geometry/PLIC/details/ExtrudedConstructionAccess.hh"
#include "geometry/mesh/ExtrudedRegionProviders.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"

namespace SimpleFluid::Meshes::details
{
class NativeProviderOwnerAccess
{
public:
    using Sink = geometry_detail::NativeConstructorOwnerSink;
    using IdentityVisit = PLIC::details::ExtrudedConstructionOwnerAccess::IdentityVisit;
    [[nodiscard]] static PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> admitted_face(const ExtrudedGeometry& geometry,
        std::size_t face, const geometry_detail::NativeConstructionAdmission& admission)
    {
        return geometry.d_construction
                   ? PLIC::details::ExtrudedConstructionAccess::face(*geometry.d_construction, face, admission)
                   : PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>{};
    }
    static void base(const SemiStructMeshTopo& topology, const Sink& sink)
    {
        using Kind = geometry_detail::NativeConstructorOwnerKind;
        if (!topology.d_base->allocation_envelope || !topology.d_base->storage.owner)
            throw std::invalid_argument("Legacy topology must be detached before exact owner projection.");
        Sink::require_envelope(topology.d_base->allocation_envelope, topology.d_base);
        if (!sink.semantic(topology.d_base, Kind::TopologyBase, topology.d_base->allocation_envelope))
            return;
        sink.fixed(topology.d_base->allocation_envelope);
        sink.storage(topology.d_base->storage);
        // Each retained unordered-map allocator owns the same storage control.
        // Those are two additional physical shared-pointer edges.
        sink.storage_reference(topology.d_base->storage.owner, topology.d_base->storage.envelope);
        sink.storage_reference(topology.d_base->storage.owner, topology.d_base->storage.envelope);
    }
    static void topology(const std::shared_ptr<const ExtrudedTopology>& owner, const Sink& sink)
    {
        using Kind = geometry_detail::NativeConstructorOwnerKind;
        if (!owner)
            return;
        if (!owner->d_allocation_envelope || !owner->d_storage.owner)
            throw std::invalid_argument("Legacy extrusion topology requires admitted detach.");
        Sink::require_envelope(owner->d_allocation_envelope, owner);
        if (!sink.semantic(owner, Kind::Topology, owner->d_allocation_envelope))
            return;
        sink.fixed(owner->d_allocation_envelope);
        sink.storage(owner->d_storage);
        base(owner->d_topology, sink);
    }
    static void geometry(const std::shared_ptr<const ExtrudedGeometry>& owner, const Sink& sink, IdentityVisit identity)
    {
        using Kind = geometry_detail::NativeConstructorOwnerKind;
        if (!owner)
            return;
        if (!owner->d_allocation_envelope || !owner->d_storage.owner)
            throw std::invalid_argument("Legacy extrusion geometry requires admitted detach.");
        Sink::require_envelope(owner->d_allocation_envelope, owner);
        if (!sink.semantic(owner, Kind::Geometry, owner->d_allocation_envelope))
            return;
        sink.fixed(owner->d_allocation_envelope);
        sink.storage(owner->d_storage);
        topology(owner->d_topology, sink);
        // Preserve the caller's context: PLIC's synchronous topology callback
        // borrows a fixed stack bridge for this invocation only.
        struct Bridge
        {
            const Sink& sink;
            IdentityVisit identity;
        } bridge{sink, identity};
        auto forwarded = sink;
        forwarded.context = &bridge;
        forwarded.owner_edge = [](void* raw, const std::shared_ptr<const void>& p,
                                   geometry_detail::NativeConstructorOwnerKind kind, size_t bytes)
        {
            const auto& b = *static_cast<Bridge*>(raw);
            return b.sink.edge(p, kind, bytes);
        };
        forwarded.fixed_edge = [](void* raw, const auto& envelope) { static_cast<Bridge*>(raw)->sink.fixed(envelope); };
        forwarded.state_edge = [](void* raw, const auto& state)
        {
            const auto& b = *static_cast<Bridge*>(raw);
            if (!b.sink.state_edge)
                throw std::invalid_argument("Missing detached native state visitor.");
            b.sink.state_edge(b.sink.context, state);
        };
        forwarded.semantic_receipt = [](void* raw, const auto& owner, const auto& envelope)
        {
            const auto& s = static_cast<Bridge*>(raw)->sink;
            if (s.semantic_receipt) s.semantic_receipt(s.context, owner, envelope);
        };
        forwarded.schema_receipt = [](void* raw, const auto& schema)
        {
            const auto& s = static_cast<Bridge*>(raw)->sink;
            if (s.schema_receipt) s.schema_receipt(s.context, schema);
        };
        forwarded.storage_receipt = [](void* raw, const auto& storage)
        {
            const auto& s = static_cast<Bridge*>(raw)->sink;
            if (s.storage_receipt) s.storage_receipt(s.context, storage);
        };
        PLIC::details::ExtrudedConstructionOwnerAccess::visit(
            owner->d_construction, forwarded,
            [](void* raw, const auto& support)
            {
                const auto& b = *static_cast<Bridge*>(raw);
                if (!b.identity)
                    throw std::invalid_argument("Missing native identity graph visitor.");
                b.identity(b.sink.context, support);
            },
            [](void* raw, const auto& child)
            { NativeProviderOwnerAccess::topology(child, static_cast<Bridge*>(raw)->sink); });
    }
    static void native(const std::shared_ptr<const SemiStructuredXY_Z>& owner, const Sink& sink)
    {
        using Kind = geometry_detail::NativeConstructorOwnerKind;
        if (!owner)
            return;
        if (!owner->d_allocation_envelope || !owner->d_storage.owner)
            throw std::invalid_argument("Mutable native mesh requires admitted detached copy.");
        Sink::require_envelope(owner->d_allocation_envelope, owner);
        if (!sink.semantic(owner, Kind::NativeMesh, owner->d_allocation_envelope))
            return;
        sink.fixed(owner->d_allocation_envelope);
        sink.storage(owner->d_storage);
        base(owner->d_topology, sink);
    }
};
} // namespace SimpleFluid::Meshes::details
