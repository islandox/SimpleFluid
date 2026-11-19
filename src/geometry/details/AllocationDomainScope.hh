#pragma once
#include "utils/export_helper.hh"
#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>

namespace SimpleFluid::geometry_detail
{
enum class AllocationDomain : std::uint8_t
{
    Unattributed,
    LeafPages,
    FamilyStorage,
    ContactHalo,
    GeometrySnapshot,
    OwnerUnion,
    NativePublication,
    ExternalThirdParty,
    Count
};

#if defined(__ELF__) && (defined(__GNUC__) || defined(__clang__))
extern "C" __attribute__((weak, visibility("default"))) AllocationDomain* simplefluid_test_allocation_domain() noexcept;
#endif

SIMPLEFLUID_LOCAL inline AllocationDomain& allocation_domain_slot() noexcept
{
#if defined(__ELF__) && (defined(__GNUC__) || defined(__clang__))
    if (simplefluid_test_allocation_domain)
        return *simplefluid_test_allocation_domain();
#endif
    static thread_local AllocationDomain local = AllocationDomain::Unattributed;
    return local;
}

class SIMPLEFLUID_LOCAL AllocationDomainScope
{
public:
    explicit AllocationDomainScope(AllocationDomain domain) noexcept
        : slot_(allocation_domain_slot()), previous_(std::exchange(slot_, domain))
    {
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }
    ~AllocationDomainScope()
    {
        std::atomic_signal_fence(std::memory_order_seq_cst);
        slot_ = previous_;
    }
    AllocationDomainScope(const AllocationDomainScope&) = delete;
    AllocationDomainScope& operator=(const AllocationDomainScope&) = delete;

private:
    AllocationDomain& slot_;
    AllocationDomain previous_;
};

template<class T> SIMPLEFLUID_LOCAL T* domain_allocate(AllocationDomain domain, std::size_t count)
{
    const AllocationDomainScope scope(domain);
    return std::allocator<T>{}.allocate(count);
}

template<class T>
SIMPLEFLUID_LOCAL void domain_deallocate(AllocationDomain domain, T* address, std::size_t count) noexcept
{
    const AllocationDomainScope scope(domain);
    std::allocator<T>{}.deallocate(address, count);
}
} // namespace SimpleFluid::geometry_detail
