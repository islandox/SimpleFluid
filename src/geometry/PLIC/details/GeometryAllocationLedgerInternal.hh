/** @file GeometryAllocationLedgerInternal.hh
 * @brief Private immutable-owner admission layout and mode-typed graph walk.
 */
#pragma once
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include "geometry/PLIC/details/OriginalSupportAuthority.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace SimpleFluid::PLIC
{
namespace SIMPLEFLUID_LOCAL allocation_detail
{
enum class Kind
{
    TypedWrapper,
    DynamicWrapper,
    Geometry,
    Topology,
    ExportCache,
    DynamicPointExport,
    TypedPointExport,
    CompactPointBatch,
    Plane,
    PointSnapshot,
    NativeFactBlock,
    Moment,
    Ancestry,
    Split,
    Certificate,
    Recipe,
    Clip,
    IdentityPayload,
    AuthorityPayload,
    CartesianSource,
    CellCatalog,
    NativeSourceBody,
    ExternalOwner,
    PlanarWrapper = 100,
    PlanarPayload = 101
};
struct Item
{
    Kind kind;
    std::shared_ptr<const void> owner;
};
struct Key
{
    Kind kind;
    const void* address;
};
struct Less
{
    bool operator()(Key a, Key b) const noexcept
    {
        return a.kind != b.kind ? a.kind < b.kind : std::less<const void*>{}(a.address, b.address);
    }
};
[[noreturn]] inline void exhausted(const char* message)
{
    throw GeometryError(GeometryErrorCode::ResourceLimit, message);
}
inline std::size_t add(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        exhausted("Geometry allocation accounting overflows.");
    return a + b;
}
inline std::size_t product(std::size_t a, std::size_t b)
{
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        exhausted("Geometry allocation accounting overflows.");
    return a * b;
}
inline std::size_t authorities(const std::vector<PlaneAuthority>& values)
{
    return product(values.capacity(), sizeof(PlaneAuthority));
}
template<ArithmeticMode Mode> struct NativeBindingPrepaid
{
    const void* typed = nullptr;
    const void* dynamic = nullptr;
    const void* geometry = nullptr;
    const details::FrozenTopology<Mode>* topology = nullptr;
    bool contains(Kind kind, const void* owner) const noexcept
    {
        if (kind == Kind::TypedWrapper)
            return owner == typed;
        if (kind == Kind::DynamicWrapper)
            return owner == dynamic;
        if (kind == Kind::Geometry)
            return owner == geometry;
        if (kind == Kind::Topology)
            return owner == topology;
        if (kind == Kind::Plane)
            return std::any_of(topology->planes.begin(), topology->planes.end(),
                [&](const auto& plane) { return plane.get() == owner; });
        return false;
    }
};
} // namespace SIMPLEFLUID_LOCAL allocation_detail

struct GeometryAllocationLedger::Impl
{
    struct Entry
    {
        std::shared_ptr<const void> pin;
        std::size_t payload_bytes = 0;
    };
    std::map<allocation_detail::Key, Entry, allocation_detail::Less> owners;
    std::vector<allocation_detail::Item> pending;
    std::optional<ArithmeticMode> mode;
    std::size_t payload = 0, bookkeeping = 0, last_counter = 0;
    std::size_t scratch_limit = std::numeric_limits<std::size_t>::max();

    void account(std::size_t bytes, bool metadata, std::size_t& used, std::size_t limit)
    {
        using namespace allocation_detail;
        if (metadata && bytes > scratch_limit - std::min(bookkeeping, scratch_limit))
            exhausted("Geometry allocation ledger scratch byte budget is exhausted.");
        if (bytes > limit - std::min(used, limit))
            exhausted("Geometry allocation ledger byte budget is exhausted.");
        used += bytes;
        if (metadata)
            bookkeeping = add(bookkeeping, bytes);
        else
            payload = add(payload, bytes);
        last_counter = used;
    }
    template<class T>
    void push(
        allocation_detail::Kind kind, const std::shared_ptr<const T>& pointer, std::size_t& used, std::size_t limit)
    {
        if (!pointer)
            return;
        if (owners.contains({kind, pointer.get()}))
            return;
        if (pending.size() == pending.capacity())
        {
            const auto old = pending.capacity();
            if (old > std::numeric_limits<std::size_t>::max() / 2)
                allocation_detail::exhausted("Geometry allocation ledger scratch capacity overflows.");
            const auto next = old ? 2 * old : 1;
            account(allocation_detail::product(next, sizeof(allocation_detail::Item)), true, used, limit);
            pending.reserve(next);
        }
        pending.push_back({kind, pointer});
    }
    bool admit(const allocation_detail::Item& item, std::size_t bytes, std::size_t& used, std::size_t limit,
        bool prepaid = false)
    {
        using namespace allocation_detail;
        const Key key{item.kind, item.owner.get()};
        if (const auto found = owners.find(key); found != owners.end())
        {
            if (found->second.payload_bytes != bytes)
                throw GeometryError(GeometryErrorCode::InvalidInput, "Immutable allocation payload size changed.");
            return false;
        }
        // Charge before publishing the seen identity. A failed call cannot
        // make an uncharged allocation disappear from a later admission.
        if (prepaid)
            payload = add(payload, bytes); // Allocation was charged before construction; retain its strong pin below.
        else
            account(bytes, false, used, limit);
        account(sizeof(typename decltype(owners)::value_type) + 4 * sizeof(void*), true, used, limit);
        owners.emplace(key, Entry{item.owner, bytes});
        return true;
    }

    template<GeometryIdentityKind Kind, ArithmeticMode Mode>
    void retain_identity(const GeometryIdentity<Kind, Mode>& identity, std::size_t& used, std::size_t limit)
    {
        visit_identity_allocations(identity,
            [&](const GeometryIdentityAllocation& allocation)
            {
                return admit({allocation_detail::Kind::IdentityPayload, allocation.owner}, allocation.payload_bytes,
                    used, limit);
            });
    }

    void retain_authorities(const std::vector<PlaneAuthority>& authorities, std::size_t& used, std::size_t limit)
    {
        for (const auto& authority : authorities)
            retain_authority(authority, used, limit);
    }

    void retain_authority(const PlaneAuthority& authority, std::size_t& used, std::size_t limit)
    {
        if (const auto* support = details::OriginalSupportAuthorityAccess::retained(authority))
            retain_identity(*support, used, limit);
        else if (const auto owner = authority.opaque_storage_owner())
            admit({allocation_detail::Kind::AuthorityPayload, owner}, authority.opaque_payload_bytes(), used, limit);
    }

    template<ArithmeticMode Mode>
    void walk(
        std::size_t& used, std::size_t limit, const allocation_detail::NativeBindingPrepaid<Mode>* prepaid = nullptr)
    {
        using namespace allocation_detail;
        using namespace details;
        const auto add_bytes = [](std::size_t& bytes, std::size_t count, std::size_t size)
        { bytes = add(bytes, product(count, size)); };
        const auto admit_owner = [&](const Item& item, std::size_t bytes, std::size_t& counter, std::size_t cap)
        { return admit(item, bytes, counter, cap, prepaid && prepaid->contains(item.kind, item.owner.get())); };
        while (!pending.empty())
        {
            const auto item = std::move(pending.back());
            pending.pop_back();
            if (owners.contains({item.kind, item.owner.get()}))
                continue;
            switch (item.kind)
            {
                case Kind::TypedWrapper: {
                    const auto geometry = std::static_pointer_cast<const ConvexCellGeometry<Mode>>(item.owner);
                    require_mode(geometry->arithmetic_mode(), Mode);
                    if (admit_owner(item, sizeof(ConvexCellGeometry<Mode>), used, limit))
                        push(Kind::DynamicWrapper, geometry->dynamic_geometry(), used, limit);
                    break;
                }
                case Kind::DynamicWrapper: {
                    const auto geometry = std::static_pointer_cast<const DynamicConvexGeometry>(item.owner);
                    require_mode(geometry->arithmetic_mode(), Mode);
                    if (admit_owner(item, sizeof(DynamicConvexGeometry), used, limit))
                        push(Kind::Geometry, DynamicGeometryAccess::implementation(*geometry), used, limit);
                    break;
                }
                case Kind::Geometry: {
                    const auto base = std::static_pointer_cast<const DynamicConvexGeometry::Impl>(item.owner);
                    const auto geometry = checked_geometry_storage<Mode>(base);
                    std::size_t bytes = sizeof(GeometryStorage<Mode>);
                    add_bytes(bytes, geometry->vertices.owned_capacity(), sizeof(FrozenVertex<Mode>));
                    if (!admit_owner(item, bytes, used, limit))
                        break;
                    push(Kind::Topology, geometry->topology, used, limit);
                    if (const auto* owner = std::get_if<std::shared_ptr<const DynamicConvexGeometry::Impl>>(
                            &geometry->vertex_owner))
                        push(Kind::Geometry, *owner, used, limit);
                    else if (const auto* owner = std::get_if<std::shared_ptr<const NativeGeometryBody>>(
                                 &geometry->vertex_owner))
                        push(Kind::NativeSourceBody, *owner, used, limit);
                    push(Kind::PointSnapshot, std::shared_ptr<const PointSnapshotContextBase>(geometry->point_context),
                        used, limit);
                    push(Kind::ExportCache, std::shared_ptr<const PointExportCache>(geometry->point_export_cache), used,
                        limit);
                    push(Kind::Geometry, geometry->export_source, used, limit);
                    push(Kind::Recipe, geometry->native_export_source, used, limit);
                    push(Kind::Moment, std::shared_ptr<const StoredMomentCache<Mode>>(geometry->moment_cache), used,
                        limit);
                    push(Kind::Ancestry, geometry->ancestry, used, limit);
                    push(Kind::Certificate, geometry->partition_certificate, used, limit);
                    push(Kind::Recipe, geometry->construction, used, limit);
                    break;
                }
                case Kind::NativeSourceBody: {
                    require_mode(Mode, ArithmeticMode::Ordinary);
                    const auto body = std::static_pointer_cast<const NativeGeometryBody>(item.owner);
                    auto bytes = sizeof(NativeGeometryBody);
                    add_bytes(bytes, body->vertices.capacity(), sizeof(FrozenVertex<ArithmeticMode::Ordinary>));
                    if (admit_owner(item, bytes, used, limit))
                    {
                        push(Kind::Topology, body->topology, used, limit);
                        push(Kind::PointSnapshot, body->point_context, used, limit);
                        push(Kind::ExportCache, std::shared_ptr<const PointExportCache>(body->export_cache), used, limit);
                        push(Kind::Ancestry, body->ancestry, used, limit);
                    }
                    break;
                }
                case Kind::Topology: {
                    const auto topology = std::static_pointer_cast<const FrozenTopology<Mode>>(item.owner);
                    if (admit_owner(item, topology->bytes(), used, limit))
                        for (const auto& plane : topology->planes)
                            push(Kind::Plane, plane, used, limit);
                    break;
                }
                case Kind::ExportCache: {
                    const auto cache = std::static_pointer_cast<const PointExportCache>(item.owner);
                    auto bytes = sizeof(PointExportCache);
                    add_bytes(bytes, cache->vertex_provenance.capacity(), sizeof(DynamicStoredPointRef));
                    // Fixed weak slots and roots never retain their publication owners.
                    (void) admit_owner(item, bytes, used, limit);
                    break;
                }
                case Kind::DynamicPointExport: {
                    const auto payload = std::static_pointer_cast<const DynamicPointExportPayload>(item.owner);
                    require_mode(payload->mode, Mode);
                    auto bytes = sizeof(DynamicPointExportPayload);
                    add_bytes(bytes, payload->points.capacity(), sizeof(DynamicPoint));
                    if (admit_owner(item, bytes, used, limit))
                        push(Kind::PointSnapshot, payload->point_context, used, limit);
                    break;
                }
                case Kind::TypedPointExport: {
                    const auto payload = std::static_pointer_cast<const PointExportPayload<Mode>>(item.owner);
                    auto bytes = sizeof(PointExportPayload<Mode>);
                    add_bytes(bytes, payload->points.capacity(), sizeof(Point<Mode>));
                    if (admit_owner(item, bytes, used, limit))
                        push(Kind::PointSnapshot,
                            std::shared_ptr<const PointSnapshotContextBase>(payload->point_context), used, limit);
                    break;
                }
                case Kind::CompactPointBatch: {
                    const auto payload = std::static_pointer_cast<const PointSnapshotPayload<Mode>>(item.owner);
                    std::size_t bytes = sizeof(PointSnapshotPayload<Mode>) + 2 * sizeof(void*);
                    add_bytes(bytes, payload->refs.capacity(), sizeof(StoredPointRef<Mode>));
                    if (admit_owner(item, bytes, used, limit))
                        push(Kind::PointSnapshot, std::shared_ptr<const PointSnapshotContextBase>(payload->context),
                            used, limit);
                    break;
                }
                case Kind::Plane: {
                    const auto plane = std::static_pointer_cast<const StoredPlane<Mode>>(item.owner);
                    auto bytes = sizeof(StoredPlane<Mode>);
                    add_bytes(bytes, plane->source_vertices.capacity(), sizeof(StoredPointRef<Mode>));
                    // Weak source references do not retain or pin their pointees.
                    if (admit_owner(item, bytes, used, limit))
                    {
                        if (plane->point_context)
                            push(Kind::PointSnapshot, plane->point_context->shared_from_this(), used, limit);
                        if constexpr (Mode == ArithmeticMode::Ordinary)
                            if (plane->native_support)
                                retain_identity(*plane->native_support, used, limit);
                    }
                    break;
                }
                case Kind::PointSnapshot: {
                    const auto base = std::static_pointer_cast<const PointSnapshotContextBase>(item.owner);
                    require_mode(base->mode, Mode);
                    const auto context = std::static_pointer_cast<const PointSnapshotContext<Mode>>(base);
                    const auto bytes = context->exclusive_storage_bytes();
                    if (admit_owner(item, bytes, used, limit))
                    {
                        retain_authorities(context->authority_table, used, limit);
                        for (const auto& parent : context->dependencies)
                            push(Kind::PointSnapshot, parent, used, limit);
                        for (const auto& block : context->native_fact_blocks)
                            push(Kind::NativeFactBlock, block, used, limit);
                    }
                    break;
                }
                case Kind::NativeFactBlock: {
                    const auto block = std::static_pointer_cast<const NativePointFactBlock<Mode>>(item.owner);
                    std::size_t bytes = sizeof(NativePointFactBlock<Mode>) + 2 * sizeof(void*);
                    add_bytes(bytes, block->points.capacity(), sizeof(PointCoordinates<Mode>));
                    add_bytes(bytes, block->witnesses.capacity(), sizeof(PointErrorWitness));
                    (void) admit_owner(item, bytes, used, limit);
                    break;
                }
                case Kind::Moment:
                    (void) admit_owner(item, sizeof(StoredMomentCache<Mode>), used, limit);
                    break;
                case Kind::Ancestry: {
                    const auto ancestry = std::static_pointer_cast<const Ancestry>(item.owner);
                    if (admit_owner(item, sizeof(Ancestry), used, limit))
                    {
                        push(Kind::Ancestry, ancestry->parent, used, limit);
                        push(Kind::Split, ancestry->split, used, limit);
                    }
                    break;
                }
                case Kind::Split: {
                    const auto split = std::static_pointer_cast<const Split>(item.owner);
                    if (admit_owner(item, sizeof(Split), used, limit))
                        push(Kind::Ancestry, split->parent, used, limit);
                    break;
                }
                case Kind::Certificate: {
                    const auto certificate = std::static_pointer_cast<const PartitionCertificate>(item.owner);
                    auto bytes = sizeof(PartitionCertificate);
                    add_bytes(bytes, certificate->members.capacity(), sizeof(std::weak_ptr<const Ancestry>));
                    if (admit_owner(item, bytes, used, limit))
                        push(Kind::Ancestry, certificate->parent, used, limit);
                    break;
                }
                case Kind::Recipe: {
                    const auto recipe = std::static_pointer_cast<const GeometryConstruction>(item.owner);
                    auto bytes = sizeof(GeometryConstruction);
                    if (recipe->kind() == GeometryConstruction::Kind::Rebase)
                        add_bytes(bytes, 1, sizeof(GeometryConstruction::Rebase));
                    if (recipe->kind() == GeometryConstruction::Kind::Source && recipe->source().native())
                        add_bytes(bytes, 1, sizeof(NativeGeometryConstruction));
                    if (admit_owner(item, bytes, used, limit))
                    {
                        if (recipe->kind() == GeometryConstruction::Kind::Source)
                        {
                            if (const auto* native = recipe->source().native())
                            {
                                require_mode(Mode, ArithmeticMode::Ordinary);
                                push(Kind::CartesianSource, native->source, used, limit);
                                push(Kind::NativeSourceBody, native->body, used, limit);
                                push(Kind::Moment, std::shared_ptr<const StoredMomentCache<ArithmeticMode::Ordinary>>(
                                    native->moments), used, limit);
                            }
                            else
                                push(Kind::Geometry, recipe->source().geometry(), used, limit);
                        }
                        else if (recipe->kind() == GeometryConstruction::Kind::Clip)
                            push(Kind::Clip, recipe->clip().clip, used, limit);
                        else
                            push(Kind::Recipe, recipe->rebase().parent, used, limit);
                    }
                    break;
                }
                case Kind::CellCatalog:
                case Kind::IdentityPayload:
                case Kind::AuthorityPayload:
                case Kind::ExternalOwner:
                case Kind::PlanarWrapper:
                case Kind::PlanarPayload:
                    throw GeometryError(GeometryErrorCode::InvalidTopology,
                        "Specialized immutable allocations must enter through their owner visitor.");
                case Kind::CartesianSource: {
                    require_mode(Mode, ArithmeticMode::Ordinary);
                    const auto source = std::static_pointer_cast<const CartesianCellSource>(item.owner);
                    const auto source_bytes = NativeCellGeometrySourceAccess::exclusive_bytes(*source);
                    if (admit_owner(item, source_bytes[0] + source_bytes[1] + source_bytes[2], used, limit))
                    {
                        const auto base_bytes = NativeCellGeometrySourceAccess::base_bytes(*source);
                        admit({Kind::ExternalOwner, NativeCellGeometrySourceAccess::base_owner(*source)},
                            base_bytes[0] + base_bytes[1] + base_bytes[2], used, limit);
                        for (const auto& support : NativeCellGeometrySourceAccess::retained_supports(*source))
                            retain_identity(support, used, limit);
                    }
                    break;
                }
                case Kind::Clip: {
                    const auto clip = std::static_pointer_cast<const GeometryClipConstruction>(item.owner);
                    require_mode(clip->mode, Mode);
                    auto bytes = sizeof(GeometryClipConstruction);
                    add_bytes(bytes, clip->retained_points.capacity(), sizeof(DynamicStoredPointRef));
                    if (admit_owner(item, bytes, used, limit))
                    {
                        push(Kind::Recipe, clip->parent, used, limit);
                        push(Kind::Recipe, clip->native_plane_source, used, limit);
                        push(Kind::Plane, std::static_pointer_cast<const StoredPlane<Mode>>(clip->plane), used, limit);
                        push(Kind::PointSnapshot, clip->point_context, used, limit);
                    }
                    break;
                }
            }
        }
    }
};
} // namespace SimpleFluid::PLIC
