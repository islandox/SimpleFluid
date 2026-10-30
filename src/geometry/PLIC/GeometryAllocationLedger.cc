/** @file GeometryAllocationLedger.cc
 * @brief Alias-aware immutable geometry ownership traversal and cumulative admission.
 */
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"

namespace SimpleFluid::PLIC
{
namespace allocation_detail SIMPLEFLUID_LOCAL
{
enum class Kind
{
    TypedWrapper,
    DynamicWrapper,
    Geometry,
    Plane,
    Point,
    Moment,
    Ancestry,
    Split,
    Certificate,
    Recipe,
    Clip,
    IdentityPayload
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
[[noreturn]] void exhausted(const char* message)
{
    throw GeometryError(GeometryErrorCode::ResourceLimit, message);
}
std::size_t add(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        exhausted("Geometry allocation accounting overflows.");
    return a + b;
}
std::size_t product(std::size_t a, std::size_t b)
{
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        exhausted("Geometry allocation accounting overflows.");
    return a * b;
}
std::size_t authorities(const std::vector<PlaneAuthority>& values)
{
    auto bytes = product(values.capacity(), sizeof(PlaneAuthority));
    for (const auto& value : values)
        bytes = add(bytes, product(value.key.capacity(), sizeof(std::uint64_t)));
    return bytes;
}
} // namespace allocation_detail SIMPLEFLUID_LOCAL

struct GeometryAllocationLedger::Impl
{
    struct Entry
    {
        std::shared_ptr<const void> pin;
        std::size_t bytes;
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
    bool admit(const allocation_detail::Item& item, std::size_t bytes, std::size_t& used, std::size_t limit)
    {
        using namespace allocation_detail;
        const Key key{item.kind, item.owner.get()};
        if (owners.contains(key))
            return false;
        // Charge before publishing the seen identity. A failed call cannot
        // make an uncharged allocation disappear from a later admission.
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

    template<ArithmeticMode Mode> void walk(std::size_t& used, std::size_t limit)
    {
        using namespace allocation_detail;
        using namespace details;
        const auto add_bytes = [](std::size_t& bytes, std::size_t count, std::size_t size)
        { bytes = add(bytes, product(count, size)); };
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
                    if (admit(item, sizeof(ConvexCellGeometry<Mode>), used, limit))
                        push(Kind::DynamicWrapper, geometry->dynamic_geometry(), used, limit);
                    break;
                }
                case Kind::DynamicWrapper: {
                    const auto geometry = std::static_pointer_cast<const DynamicConvexGeometry>(item.owner);
                    require_mode(geometry->arithmetic_mode(), Mode);
                    if (admit(item, sizeof(DynamicConvexGeometry), used, limit))
                        push(Kind::Geometry, DynamicGeometryAccess::implementation(*geometry), used, limit);
                    break;
                }
                case Kind::Geometry: {
                    const auto base = std::static_pointer_cast<const DynamicConvexGeometry::Impl>(item.owner);
                    const auto geometry = checked_geometry_storage<Mode>(base);
                    std::size_t bytes = sizeof(GeometryStorage<Mode>);
                    add_bytes(bytes, geometry->vertices.capacity(), sizeof(StoredVertex<Mode>));
                    add_bytes(bytes, geometry->faces.capacity(), sizeof(Face));
                    add_bytes(bytes, geometry->surfaces.capacity(), sizeof(StoredSurface<Mode>));
                    add_bytes(bytes, geometry->displayed.capacity(), sizeof(DynamicPoint));
                    for (const auto& vertex : geometry->vertices)
                    {
                        add_bytes(bytes, vertex.incident.capacity(), sizeof(StoredPlanePtr<Mode>));
                        bytes = add(bytes, authorities(vertex.authorities));
                    }
                    for (const auto& face : geometry->faces)
                        add_bytes(bytes, face.capacity(), sizeof(std::size_t));
                    if (!admit(item, bytes, used, limit))
                        break;
                    for (const auto& vertex : geometry->vertices)
                    {
                        push(Kind::Point, vertex.source_identity, used, limit);
                        for (const auto& plane : vertex.incident)
                            push(Kind::Plane, plane, used, limit);
                    }
                    for (const auto& surface : geometry->surfaces)
                        push(Kind::Plane, surface.plane, used, limit);
                    for (const auto& point : geometry->displayed)
                        push(Kind::Point, DynamicGeometryAccess::identity(point), used, limit);
                    push(Kind::Moment, std::shared_ptr<const StoredMomentCache<Mode>>(geometry->moment_cache), used,
                        limit);
                    push(Kind::Ancestry, geometry->ancestry, used, limit);
                    push(Kind::Certificate, geometry->partition_certificate, used, limit);
                    push(Kind::Recipe, geometry->construction, used, limit);
                    break;
                }
                case Kind::Plane: {
                    const auto plane = std::static_pointer_cast<const StoredPlane<Mode>>(item.owner);
                    auto bytes = add(sizeof(StoredPlane<Mode>), authorities(plane->authorities));
                    add_bytes(
                        bytes, plane->source_vertices.capacity(), sizeof(std::weak_ptr<const DynamicPointProvenance>));
                    // Weak source references do not retain or pin their pointees.
                    if (admit(item, bytes, used, limit))
                        if constexpr (Mode == ArithmeticMode::Ordinary)
                            if (plane->native_support)
                                retain_identity(*plane->native_support, used, limit);
                    break;
                }
                case Kind::Point: {
                    const auto point = std::static_pointer_cast<const DynamicPointProvenance>(item.owner);
                    auto bytes = add(sizeof(DynamicPointProvenance), authorities(point->supports));
                    add_bytes(bytes, point->source_edges.capacity(), sizeof(point->source_edges.front()));
                    if (admit(item, bytes, used, limit))
                    {
                        push(Kind::Point, point->source_origin, used, limit);
                        for (const auto& edge : point->source_edges)
                        {
                            push(Kind::Point, edge[0], used, limit);
                            push(Kind::Point, edge[1], used, limit);
                        }
                    }
                    break;
                }
                case Kind::Moment:
                    (void) admit(item, sizeof(StoredMomentCache<Mode>), used, limit);
                    break;
                case Kind::Ancestry: {
                    const auto ancestry = std::static_pointer_cast<const Ancestry>(item.owner);
                    if (admit(item, sizeof(Ancestry), used, limit))
                    {
                        push(Kind::Ancestry, ancestry->parent, used, limit);
                        push(Kind::Split, ancestry->split, used, limit);
                    }
                    break;
                }
                case Kind::Split: {
                    const auto split = std::static_pointer_cast<const Split>(item.owner);
                    if (admit(item, sizeof(Split), used, limit))
                        push(Kind::Ancestry, split->parent, used, limit);
                    break;
                }
                case Kind::Certificate: {
                    const auto certificate = std::static_pointer_cast<const PartitionCertificate>(item.owner);
                    auto bytes = sizeof(PartitionCertificate);
                    add_bytes(bytes, certificate->members.capacity(), sizeof(std::weak_ptr<const Ancestry>));
                    if (admit(item, bytes, used, limit))
                        push(Kind::Ancestry, certificate->parent, used, limit);
                    break;
                }
                case Kind::Recipe: {
                    const auto recipe = std::static_pointer_cast<const GeometryConstruction>(item.owner);
                    auto bytes = sizeof(GeometryConstruction);
                    add_bytes(bytes, recipe->retained_points.capacity(),
                        sizeof(std::shared_ptr<const DynamicPointProvenance>));
                    if (admit(item, bytes, used, limit))
                    {
                        push(Kind::Geometry, recipe->source, used, limit);
                        push(Kind::Recipe, recipe->parent, used, limit);
                        push(Kind::Clip, recipe->clip, used, limit);
                        for (const auto& point : recipe->retained_points)
                            push(Kind::Point, point, used, limit);
                    }
                    break;
                }
                case Kind::IdentityPayload:
                    throw GeometryError(GeometryErrorCode::InvalidTopology,
                        "Identity allocations must enter through their owner visitor.");
                case Kind::Clip: {
                    const auto clip = std::static_pointer_cast<const GeometryClipConstruction>(item.owner);
                    require_mode(clip->mode, Mode);
                    auto bytes = sizeof(GeometryClipConstruction);
                    add_bytes(
                        bytes, clip->retained_points.capacity(), sizeof(std::shared_ptr<const DynamicPointProvenance>));
                    if (admit(item, bytes, used, limit))
                    {
                        push(Kind::Recipe, clip->parent, used, limit);
                        push(Kind::Plane, std::static_pointer_cast<const StoredPlane<Mode>>(clip->plane), used, limit);
                        for (const auto& point : clip->retained_points)
                            push(Kind::Point, point, used, limit);
                    }
                    break;
                }
            }
        }
    }
};

GeometryAllocationLedger::GeometryAllocationLedger() noexcept = default;
GeometryAllocationLedger::~GeometryAllocationLedger() = default;
GeometryAllocationLedger::GeometryAllocationLedger(GeometryAllocationLedger&&) noexcept = default;
GeometryAllocationLedger& GeometryAllocationLedger::operator=(GeometryAllocationLedger&&) noexcept = default;
std::size_t GeometryAllocationLedger::owner_count() const noexcept
{
    return impl_ ? impl_->owners.size() : 0;
}
std::size_t GeometryAllocationLedger::payload_bytes() const noexcept
{
    return impl_ ? impl_->payload : 0;
}
std::size_t GeometryAllocationLedger::bookkeeping_bytes() const noexcept
{
    return impl_ ? impl_->bookkeeping : 0;
}

void GeometryAllocationLedger::begin_admission(std::size_t& used, std::size_t limit, bool allocate_state)
{
    if (failed_)
        allocation_detail::exhausted("Geometry allocation ledger cannot resume rejected admission.");
    if (impl_ && used < impl_->last_counter)
        allocation_detail::exhausted("Geometry allocation cumulative byte count decreased.");
    if (used > limit)
        allocation_detail::exhausted("Geometry allocation byte budget is already exhausted.");
    if (!impl_ && allocate_state)
    {
        if (sizeof(Impl) > limit - used)
            allocation_detail::exhausted("Geometry allocation ledger byte budget is exhausted.");
        used += sizeof(Impl);
        impl_ = std::make_unique<Impl>();
        impl_->bookkeeping = sizeof(Impl);
        impl_->last_counter = used;
    }
}

template<ArithmeticMode Mode>
void GeometryAllocationLedger::retain(const ConvexGeometryPtr<Mode>& geometry, std::size_t& used, std::size_t limit)
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);
    try
    {
        if (!geometry)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Null geometry allocation admission.");
        details::require_mode(geometry->arithmetic_mode(), Mode);
        begin_admission(used, limit);
        if (impl_->mode)
            details::require_mode(*impl_->mode, Mode);
        else
            impl_->mode = Mode;
        impl_->push(allocation_detail::Kind::TypedWrapper, geometry, used, limit);
        impl_->walk<Mode>(used, limit);
        impl_->last_counter = used;
    }
    catch (...)
    {
        failed_ = true;
        throw;
    }
}

template<ArithmeticMode Mode>
void GeometryAllocationLedger::retain(const DynamicConvexGeometryPtr& geometry, std::size_t& used, std::size_t limit)
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);
    try
    {
        if (!geometry)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Null dynamic geometry allocation admission.");
        details::require_mode(geometry->arithmetic_mode(), Mode);
        begin_admission(used, limit);
        if (impl_->mode)
            details::require_mode(*impl_->mode, Mode);
        else
            impl_->mode = Mode;
        impl_->push(allocation_detail::Kind::DynamicWrapper, geometry, used, limit);
        impl_->walk<Mode>(used, limit);
        impl_->last_counter = used;
    }
    catch (...)
    {
        failed_ = true;
        throw;
    }
}

template<GeometryIdentityKind Kind, ArithmeticMode Mode>
    requires(GeometryIdentity<Kind, Mode>::shared_storage)
void GeometryAllocationLedger::retain(
    const GeometryIdentity<Kind, Mode>& identity, std::size_t& used, std::size_t limit)
{
    try
    {
        begin_admission(used, limit, identity.storage_identity() != nullptr);
        if (!impl_)
            return;
        impl_->retain_identity(identity, used, limit);
        impl_->last_counter = used;
    }
    catch (...)
    {
        failed_ = true;
        throw;
    }
}

template void GeometryAllocationLedger::retain<ArithmeticMode::Ordinary>(
    const ConvexGeometryPtr<ArithmeticMode::Ordinary>&, std::size_t&, std::size_t);
template void GeometryAllocationLedger::retain<ArithmeticMode::BoundedCertification>(
    const ConvexGeometryPtr<ArithmeticMode::BoundedCertification>&, std::size_t&, std::size_t);

template void GeometryAllocationLedger::retain<ArithmeticMode::Ordinary>(
    const DynamicConvexGeometryPtr&, std::size_t&, std::size_t);
template void GeometryAllocationLedger::retain<ArithmeticMode::BoundedCertification>(
    const DynamicConvexGeometryPtr&, std::size_t&, std::size_t);
template void GeometryAllocationLedger::retain<GeometryIdentityKind::PlaneSupport, ArithmeticMode::Ordinary>(
    const PlaneSupportIdentity<ArithmeticMode::Ordinary>&, std::size_t&, std::size_t);
template void GeometryAllocationLedger::retain<GeometryIdentityKind::Point, ArithmeticMode::Exact>(
    const PointIdentity<ArithmeticMode::Exact>&, std::size_t&, std::size_t);
template void GeometryAllocationLedger::retain<GeometryIdentityKind::Line, ArithmeticMode::Exact>(
    const LineIdentity<ArithmeticMode::Exact>&, std::size_t&, std::size_t);
template void GeometryAllocationLedger::retain<GeometryIdentityKind::PlaneSupport, ArithmeticMode::Exact>(
    const PlaneSupportIdentity<ArithmeticMode::Exact>&, std::size_t&, std::size_t);
template void GeometryAllocationLedger::retain<GeometryIdentityKind::Displacement, ArithmeticMode::Exact>(
    const DisplacementIdentity<ArithmeticMode::Exact>&, std::size_t&, std::size_t);

namespace details
{
template<ArithmeticMode Mode>
std::size_t geometry_allocation_footprint(const std::shared_ptr<const DynamicConvexGeometry::Impl>& geometry)
{
    require_mode(geometry->limits.arithmetic_mode, Mode);
    GeometryAllocationLedger::Impl accounting;
    accounting.mode = Mode;
    accounting.scratch_limit = geometry->limits.byte_budget;
    std::size_t used = 0;
    accounting.account(sizeof(accounting), true, used, std::numeric_limits<std::size_t>::max());
    accounting.push(allocation_detail::Kind::Geometry, geometry, used, std::numeric_limits<std::size_t>::max());
    accounting.walk<Mode>(used, std::numeric_limits<std::size_t>::max());
    return accounting.payload;
}
template std::size_t geometry_allocation_footprint<ArithmeticMode::Ordinary>(
    const std::shared_ptr<const DynamicConvexGeometry::Impl>&);
template std::size_t geometry_allocation_footprint<ArithmeticMode::BoundedCertification>(
    const std::shared_ptr<const DynamicConvexGeometry::Impl>&);
} // namespace details
} // namespace SimpleFluid::PLIC
