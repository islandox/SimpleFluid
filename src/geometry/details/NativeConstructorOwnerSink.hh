#pragma once
#include "geometry/details/NativeConstructorStorage.hh"
#include "geometry/details/NativeSchemaOwner.hh"
namespace SimpleFluid::PLIC::details { class GeometryOwnerGraphEditor; }
namespace SimpleFluid::geometry_detail
{
enum class NativeConstructorOwnerKind : unsigned char
{
    Topology,
    TopologyBase,
    Geometry,
    NativeMesh,
    Construction,
    ConstructionImpl,
    ConstructionBase,
    ConstructionAxis,
    ConstructionValues,
    Storage
};
/** Borrowed synchronous traversal callbacks; no callback/context is retained by
 * a provider. edge returns the first/last transition requiring its children. */
struct NativeConstructorOwnerSink
{
    void* context = nullptr;
    bool (*owner_edge)(void*, const std::shared_ptr<const void>&, NativeConstructorOwnerKind, size_t) = nullptr;
    void (*fixed_edge)(void*, const std::shared_ptr<const NativeAllocationEnvelope>&) = nullptr;
    void (*state_edge)(void*, const std::shared_ptr<const NativeAllocationState>&) = nullptr;
    void (*semantic_receipt)(void*, const std::shared_ptr<const void>&,
        const std::shared_ptr<const NativeAllocationEnvelope>&) = nullptr;
    void (*storage_receipt)(void*, const NativeConstructorStorageOwner&) = nullptr;
    void (*schema_receipt)(void*, const std::shared_ptr<const NativeSchemaOwner>&) = nullptr;
    // Installed only by the closed financial descriptor adapter. Concrete
    // financial holders can preserve existing typed identity/geometry closure.
    PLIC::details::GeometryOwnerGraphEditor* financial_graph = nullptr;
    bool financial_retain = true;
    void schema(const std::shared_ptr<const NativeSchemaOwner>& owner) const
    {
        if (!owner) return;
        require_envelope(owner->allocation(), owner);
        if (schema_receipt) schema_receipt(context, owner);
        if (!semantic(owner, NativeConstructorOwnerKind::Storage, owner->allocation())) return;
        fixed(owner->allocation());
        if (!state_edge) throw std::invalid_argument("Native schema visitor lacks State projection.");
        state_edge(context, owner->allocation()->state());
        if (owner->capacity_bytes())
            edge(std::shared_ptr<const void>(owner, owner->span().data()), NativeConstructorOwnerKind::Storage,
                owner->capacity_bytes());
    }
    /** Receipt authorization precedes the first/last semantic edge. It creates
     * no graph edge and never authorizes an arbitrary byte count. */
    bool semantic(const std::shared_ptr<const void>& owner, NativeConstructorOwnerKind kind,
        const std::shared_ptr<const NativeAllocationEnvelope>& envelope) const
    {
        require_envelope(envelope, owner);
        if (semantic_receipt)
            semantic_receipt(context, owner, envelope);
        return edge(owner, kind, 0);
    }
    bool edge(const std::shared_ptr<const void>& owner, NativeConstructorOwnerKind kind, size_t bytes) const
    {
        if (!owner || !owner_edge)
            throw std::invalid_argument("Incomplete native owner visitor.");
        return owner_edge(context, owner, kind, bytes);
    }
    template<class T>
    static void require_envelope(
        const std::shared_ptr<const NativeAllocationEnvelope>& envelope, const std::shared_ptr<T>& owner)
    {
        if (!envelope || !envelope->authentic_owner(envelope) || !envelope->matches(owner))
            throw std::invalid_argument("Incomplete or foreign native allocation envelope/control.");
    }
    void fixed(const std::shared_ptr<const NativeAllocationEnvelope>& envelope) const
    {
        if (!envelope || !envelope->authentic_owner(envelope) || !fixed_edge)
            throw std::invalid_argument("Native owner lacks exact allocation envelope.");
        fixed_edge(context, envelope);
    }
    void storage(const NativeConstructorStorageOwner& value) const
    {
        if (!value.owner)
        {
            if (value.envelope) throw std::invalid_argument("Native storage has an envelope without its semantic owner.");
            return;
        }
        require_envelope(value.envelope, value.owner);
        if (storage_receipt)
            storage_receipt(context, value);
        fixed(value.envelope); // Parallel shared field in the containing owner.
        storage_reference(value.owner, value.envelope);
    }
    void storage_reference(const std::shared_ptr<NativeConstructorStorage>& owner,
        const std::shared_ptr<const NativeAllocationEnvelope>& envelope) const
    {
        if (owner)
        {
            require_envelope(envelope, owner);
            if (storage_receipt)
                storage_receipt(context, NativeConstructorStorageOwner{owner, envelope});
        }
        if (owner && edge(owner, NativeConstructorOwnerKind::Storage, owner->exclusive_bytes()))
        {
            if (!state_edge)
                throw std::invalid_argument("Native owner visitor lacks detached state projection.");
            state_edge(context, owner->admission().state());
        }
    }
};
} // namespace SimpleFluid::geometry_detail
