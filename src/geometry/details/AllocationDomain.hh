#pragma once
#include "geometry/details/AllocationDomainScope.hh"
#include <type_traits>
#include <utility>
#include <vector>
#if SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT
#include <algorithm>
#include <limits>
#include <mutex>
#include <new>
#endif

namespace SimpleFluid::geometry_detail
{
#if SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT
struct SIMPLEFLUID_LOCAL AllocationStatistics
{
    std::size_t live = 0, peak = 0, allocations = 0, deallocations = 0;
};

class SIMPLEFLUID_LOCAL AllocationAccount
{
    template<class, AllocationDomain> friend class DomainAllocator;

public:
    constexpr explicit AllocationAccount(std::size_t limit = std::numeric_limits<std::size_t>::max()) noexcept
        : limit_(limit)
    {
    }
    [[nodiscard]] AllocationStatistics statistics() const noexcept
    {
        const std::lock_guard lock(mutex_);
        return totals_;
    }

private:
    void reserve(std::size_t bytes)
    {
        const std::lock_guard lock(mutex_);
        if (bytes > limit_ - totals_.live - pending_)
            throw std::bad_alloc();
        pending_ += bytes;
    }
    void finish(std::size_t bytes, bool success) noexcept
    {
        const std::lock_guard lock(mutex_);
        pending_ -= bytes;
        if (success)
        {
            totals_.live += bytes;
            totals_.peak = std::max(totals_.live, totals_.peak);
            ++totals_.allocations;
        }
    }
    void release(std::size_t bytes) noexcept
    {
        const std::lock_guard lock(mutex_);
        totals_.live -= bytes;
        ++totals_.deallocations;
    }

    mutable std::mutex mutex_;
    const std::size_t limit_;
    std::size_t pending_ = 0;
    AllocationStatistics totals_;
};

template<AllocationDomain Domain> SIMPLEFLUID_LOCAL inline constinit AllocationAccount domain_account;

template<class T, AllocationDomain Domain> class SIMPLEFLUID_LOCAL FixedDomainAllocator
{
public:
    using value_type = T;
    using is_always_equal = std::true_type;
    template<class U> struct rebind
    {
        using other = FixedDomainAllocator<U, Domain>;
    };
    FixedDomainAllocator() noexcept = default;
    template<class U> FixedDomainAllocator(const FixedDomainAllocator<U, Domain>&) noexcept {}
    [[nodiscard]] T* allocate(std::size_t count) { return domain_allocate<T>(Domain, count); }
    void deallocate(T* address, std::size_t count) noexcept { domain_deallocate(Domain, address, count); }
    template<class U> bool operator==(const FixedDomainAllocator<U, Domain>&) const noexcept { return true; }
};

template<class T, AllocationDomain Domain> class SIMPLEFLUID_LOCAL DomainAllocator
{
    template<class, AllocationDomain> friend class DomainAllocator;

public:
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;
    template<class U> struct rebind
    {
        using other = DomainAllocator<U, Domain>;
    };

    DomainAllocator() noexcept = default;
    explicit DomainAllocator(AllocationAccount& account) noexcept : account_(&account) {}
    template<class U> DomainAllocator(const DomainAllocator<U, Domain>& other) noexcept : account_(other.account_) {}

    [[nodiscard]] T* allocate(std::size_t count)
    {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_array_new_length();
        const auto bytes = count * sizeof(T);
        account_->reserve(bytes);
        try
        {
            auto* address = domain_allocate<T>(Domain, count);
            account_->finish(bytes, true);
            return address;
        }
        catch (...)
        {
            account_->finish(bytes, false);
            throw;
        }
    }
    void deallocate(T* address, std::size_t count) noexcept
    {
        domain_deallocate(Domain, address, count);
        account_->release(count * sizeof(T));
    }
    template<class U> [[nodiscard]] bool operator==(const DomainAllocator<U, Domain>& other) const noexcept
    {
        return account_ == other.account_;
    }

private:
    AllocationAccount* account_ = &domain_account<Domain>;
};
#else
template<class T, AllocationDomain> using DomainAllocator = std::allocator<T>;
template<class T, AllocationDomain> using FixedDomainAllocator = std::allocator<T>;
#endif

template<class T, AllocationDomain Domain> using DomainVector = std::vector<T, DomainAllocator<T, Domain>>;

#if !SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT
static_assert(std::is_empty_v<AllocationDomainScope> && std::is_trivial_v<AllocationDomainScope>);
static_assert(sizeof(DomainVector<int, AllocationDomain::LeafPages>) == sizeof(std::vector<int>));
#endif

template<AllocationDomain Domain, class T, class... Args>
SIMPLEFLUID_LOCAL std::shared_ptr<T> domain_shared(Args&&... args)
{
    return std::allocate_shared<T>(DomainAllocator<std::remove_cv_t<T>, Domain>{}, std::forward<Args>(args)...);
}
} // namespace SimpleFluid::geometry_detail
