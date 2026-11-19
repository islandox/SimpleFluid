#pragma once
#include "utils/export_helper.hh"
#include <cstdint>
#include <memory>
#if SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT
#include <atomic>
#include <utility>
#endif

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

#if SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT
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
#else
class SIMPLEFLUID_LOCAL AllocationDomainScope
{
public:
    AllocationDomainScope() = default;
    explicit constexpr AllocationDomainScope(AllocationDomain) noexcept {}
};
#endif

template<class T> SIMPLEFLUID_LOCAL T* domain_allocate([[maybe_unused]] AllocationDomain domain, std::size_t count)
{
#if SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT
    const AllocationDomainScope scope(domain);
#endif
    return std::allocator<T>{}.allocate(count);
}

template<class T>
SIMPLEFLUID_LOCAL void domain_deallocate(
    [[maybe_unused]] AllocationDomain domain, T* address, std::size_t count) noexcept
{
#if SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT
    const AllocationDomainScope scope(domain);
#endif
    std::allocator<T>{}.deallocate(address, count);
}
} // namespace SimpleFluid::geometry_detail
