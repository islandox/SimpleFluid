/** Private snapshot allocation below the existing cumulative geometry ledger. */
#pragma once
#include "geometry/details/AllocationDomain.hh"
#include <map>
#include <type_traits>

namespace SimpleFluid::PLIC::details
{
template<class T>
using GeometrySnapshotAllocator =
    geometry_detail::FixedDomainAllocator<T, geometry_detail::AllocationDomain::GeometrySnapshot>;
template<class T> using GeometrySnapshotVector = std::vector<T, GeometrySnapshotAllocator<T>>;
template<class Key, class Value, class Less = std::less<Key>>
using GeometrySnapshotMap = std::map<Key, Value, Less, GeometrySnapshotAllocator<std::pair<const Key, Value>>>;
template<class T, class... Args> SIMPLEFLUID_LOCAL std::shared_ptr<T> geometry_snapshot_shared(Args&&... args)
{
    return std::allocate_shared<T>(GeometrySnapshotAllocator<std::remove_cv_t<T>>{}, std::forward<Args>(args)...);
}
/** Concrete public buffers retain their allocator type. Scope only a fresh
 * buffer request; relocation of project elements stays outside the scope. */
template<class T, class Allocator>
SIMPLEFLUID_LOCAL void geometry_snapshot_reserve(std::vector<T, Allocator>& values, std::size_t count)
{
#if SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT
    if (!values.capacity())
    {
        const geometry_detail::AllocationDomainScope domain(geometry_detail::AllocationDomain::GeometrySnapshot);
        return values.reserve(count);
    }
#endif
    values.reserve(count);
}
static_assert(std::is_empty_v<GeometrySnapshotAllocator<std::byte>>);
static_assert(sizeof(GeometrySnapshotVector<std::byte>) == sizeof(std::vector<std::byte>));
static_assert(alignof(GeometrySnapshotVector<std::byte>) == alignof(std::vector<std::byte>));
} // namespace SimpleFluid::PLIC::details
