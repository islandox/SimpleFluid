/** @file NativeConstructionCounter.hh
 * @brief Self-accounted detached scalar counter creation with no stack capture.
 */
#pragma once
#include "geometry/details/NativeAllocationExtent.hh"
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class NativePublicationOperation;
class NativeSourceOwnerStore;
}
namespace SimpleFluid::geometry_detail
{
enum class NativeMirrorExtentKind { Actual, ProfiledUpper };
class NativeMirrorOwnerProof
{
public:
    NativeMirrorOwnerProof() noexcept = default;
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] NativeMirrorExtentKind kind() const noexcept { return kind_; }
    [[nodiscard]] bool publication_requests() const noexcept { return publication_requests_; }
    [[nodiscard]] bool matches(const std::shared_ptr<void>& owner) const noexcept
    {
        return owner && owner.get() == address_ && bytes_ && !control_.expired() &&
               !control_.owner_before(owner) && !owner.owner_before(control_);
    }
private:
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativePublicationOperation;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativeSourceOwnerStore;
    NativeMirrorOwnerProof(const std::shared_ptr<void>& owner, std::size_t bytes,
        NativeMirrorExtentKind kind = NativeMirrorExtentKind::Actual, bool publication_requests = false)
        : address_(owner.get()), control_(owner), bytes_(bytes), kind_(kind),
          publication_requests_(publication_requests) {}
    const void* address_ = nullptr;
    std::weak_ptr<const void> control_;
    std::size_t bytes_ = 0;
    NativeMirrorExtentKind kind_ = NativeMirrorExtentKind::Actual;
    bool publication_requests_ = false;
};
/** Optional mirror into an already owned detached source/halo scalar arena.
 * Callbacks never allocate, throw, or reenter NativeAllocationState. Prefix
 * bytes are already represented in that arena and are never mirrored twice. */
struct SIMPLEFLUID_LOCAL NativeAllocationMirror
{
    std::shared_ptr<void> owner;
    bool (*try_precharge)(void*, std::size_t) noexcept = nullptr;
    void (*retire_callback)(void*, std::size_t) noexcept = nullptr;
    NativeMirrorOwnerProof owner_proof;
    [[nodiscard]] bool acquire(std::size_t bytes) const noexcept
    { return !owner || (try_precharge && retire_callback && try_precharge(owner.get(), bytes)); }
    void retire(std::size_t bytes) const noexcept
    { if (owner) retire_callback(owner.get(), bytes); }
};
class NativeConstructionCounterAccess;
class NativeConstructionAdmission;
template<class Counter> class NativeOwnedCounter
{
public:
    [[nodiscard]] const std::shared_ptr<Counter>& owner() const noexcept { return owner_; }
    [[nodiscard]] const NativeCounterAllocationEnvelope& allocations() const noexcept { return allocations_; }

private:
    friend class NativeConstructionCounterAccess;
    friend class NativeConstructionAdmission;
    NativeOwnedCounter(std::shared_ptr<Counter> owner, NativeCounterAllocationEnvelope allocations, std::size_t prefix, NativeAllocationMirror mirror)
        : owner_(std::move(owner)), allocations_(allocations), prefix_(prefix), mirror_(std::move(mirror))
    {
    }
    std::shared_ptr<Counter> owner_;
    NativeCounterAllocationEnvelope allocations_;
    std::size_t prefix_ = 0;
    NativeAllocationMirror mirror_;
};
namespace native_counter_detail
{
template<class Counter> struct Delete
{
    NativeAllocationMirror mirror;
    void operator()(Counter* counter) noexcept
    {
        std::destroy_at(counter);
        std::allocator<Counter>{}.deallocate(counter, 1);
        mirror.retire(sizeof(Counter));
    }
};
/** The destination is the owned heap Counter itself. Allocation records into
 * it before publication; deallocate never dereferences it after destruction.
 * No retained pointer names a caller stack counter, Data or cache controller.
 */
template<class T, class Counter> class ControlAllocator
{
    template<class, class> friend class ControlAllocator;

public:
    using value_type = T;
    explicit ControlAllocator(Counter* counter, NativeAllocationMirror mirror) noexcept
        : counter_(counter), mirror_(std::move(mirror)) {}
    template<class U> ControlAllocator(const ControlAllocator<U, Counter>& other) noexcept : counter_(other.counter_), mirror_(other.mirror_) {}
    T* allocate(std::size_t count)
    {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();
        const auto bytes = count * sizeof(T);
        if (!mirror_.acquire(bytes))
            throw std::invalid_argument("Native counter control exceeds source arena headroom.");
        try { counter_->add(bytes); }
        catch (...) { mirror_.retire(bytes); throw; }
        T* allocation = nullptr;
        try
        {
            allocation = std::allocator<T>{}.allocate(count);
        }
        catch (...)
        {
            counter_->used -= bytes;
            mirror_.retire(bytes);
            throw;
        }
        if (counter_->native_allocation.control.address)
        {
            std::allocator<T>{}.deallocate(allocation, count);
            counter_->used -= bytes;
            mirror_.retire(bytes);
            throw std::invalid_argument("Detached counter requires a single qualified control allocation.");
        }
        counter_->native_allocation.control = {allocation, bytes};
        return allocation;
    }
    void deallocate(T* allocation, std::size_t count) noexcept
    {
        std::allocator<T>{}.deallocate(allocation, count);
        mirror_.retire(count * sizeof(T));
    }
    template<class U> bool operator==(const ControlAllocator<U, Counter>& other) const noexcept
    {
        return counter_ == other.counter_;
    }

private:
    Counter* counter_;
    NativeAllocationMirror mirror_;
};
} // namespace native_counter_detail

class SIMPLEFLUID_LOCAL NativeConstructionCounterAccess
{
public:
    template<class Counter>
    [[nodiscard]] static NativeOwnedCounter<Counter> create(std::size_t limit, std::size_t accepted_prefix = 0, NativeAllocationMirror mirror = {})
    {
        static_assert(std::is_trivially_copyable_v<Counter> && std::is_trivially_destructible_v<Counter> &&
                          std::is_standard_layout_v<Counter>,
            "The retained counter must be detached scalar storage.");
        static_assert(std::is_same_v<decltype(Counter::native_allocation), NativeCounterAllocationEnvelope>);
        if (accepted_prefix > limit)
            throw std::invalid_argument("Detached native counter prefix exceeds its cap.");
        // The first object allocation is checked using a stack VALUE. No
        // allocator/control keeps a pointer or reference to that stack value.
        Counter initial{limit, accepted_prefix};
        initial.add(sizeof(Counter));
        if (!mirror.acquire(sizeof(Counter)))
            throw std::invalid_argument("Native counter object exceeds source arena headroom.");
        Counter* raw = nullptr;
        try { raw = std::allocator<Counter>{}.allocate(1); }
        catch (...) { mirror.retire(sizeof(Counter)); throw; }
        std::construct_at(raw, initial);
        raw->native_allocation.object = {raw, sizeof(Counter)};
        std::shared_ptr<Counter> owner(raw, native_counter_detail::Delete<Counter>{mirror},
            native_counter_detail::ControlAllocator<Counter, Counter>(raw, mirror));
        const auto envelope = owner->native_allocation;
        return NativeOwnedCounter<Counter>(std::move(owner), envelope, accepted_prefix, std::move(mirror));
    }
};
} // namespace SimpleFluid::geometry_detail
