/** Private contact containers. Heap attribution is separate from arena admission. */
#pragma once
#include "geometry/details/AllocationDomain.hh"
#include <cstddef>
#include <map>
#include <memory>
#include <new>
#include <set>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
template<class T>
using ContactAllocator = geometry_detail::DomainAllocator<T, geometry_detail::AllocationDomain::ContactHalo>;
template<class T> using ContactVector = std::vector<T, ContactAllocator<T>>;
template<class T>
using ContactFixedAllocator = geometry_detail::FixedDomainAllocator<T, geometry_detail::AllocationDomain::ContactHalo>;
template<class T> using ContactFixedVector = std::vector<T, ContactFixedAllocator<T>>;
template<class Key, class Value, class Less = std::less<Key>>
using ContactMap = std::map<Key, Value, Less, ContactAllocator<std::pair<const Key, Value>>>;
template<class Key, class Less = std::less<Key>>
using ContactSet = std::set<Key, Less, ContactAllocator<Key>>;

#if SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT
template<class T> struct SIMPLEFLUID_LOCAL ContactArrayDelete
{
    std::size_t count = 0;
    void operator()(T* values) const noexcept
    {
        std::destroy_n(values, count);
        geometry_detail::domain_deallocate(geometry_detail::AllocationDomain::ContactHalo, values, count);
    }
};
template<class T> using ContactArray = std::unique_ptr<T[], ContactArrayDelete<T>>;
/** The enclosing funded store already admits this exact capacity. */
template<class T> SIMPLEFLUID_LOCAL ContactArray<T> contact_array(std::size_t count)
{
    if (!count)
        return {};
    auto* values = geometry_detail::domain_allocate<T>(geometry_detail::AllocationDomain::ContactHalo, count);
    try
    {
        std::uninitialized_value_construct_n(values, count);
    }
    catch (...)
    {
        geometry_detail::domain_deallocate(geometry_detail::AllocationDomain::ContactHalo, values, count);
        throw;
    }
    return ContactArray<T>(values, ContactArrayDelete<T>{count});
}
#else
template<class T> using ContactArray = std::unique_ptr<T[]>;
template<class T> SIMPLEFLUID_LOCAL ContactArray<T> contact_array(std::size_t count)
{
    return count ? std::make_unique<T[]>(count) : nullptr;
}
#endif

/** Construct outside the allocator's scope, including for private constructors.
 * Keep the allocating state through both the body and shared-control lifetimes. */
template<class T> struct SIMPLEFLUID_LOCAL ContactObjectDelete
{
    [[no_unique_address]] ContactAllocator<T> memory;
    void operator()(T* value) noexcept
    {
        std::destroy_at(value);
        memory.deallocate(value, 1);
    }
};
template<class T, class Construct> SIMPLEFLUID_LOCAL std::shared_ptr<T> contact_shared_object(Construct&& construct)
{
    ContactAllocator<T> memory;
    auto* body = memory.allocate(1);
    try
    {
        construct(body);
    }
    catch (...)
    {
        memory.deallocate(body, 1);
        throw;
    }
    ContactObjectDelete<T> destroy{memory};
#if defined(__GLIBCXX__)
    static_assert(sizeof(std::_Sp_counted_deleter<T*, decltype(destroy), ContactAllocator<T>,
                      __gnu_cxx::__default_lock_policy>) <= 128,
        "Contact private control exceeds its unchanged allowance.");
#endif
    return std::shared_ptr<T>(body, std::move(destroy), memory);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
