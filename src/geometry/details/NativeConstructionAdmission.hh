/** @file NativeConstructionAdmission.hh
 * @brief Owned scalar admission state below native Mesh and PLIC constructors.
 */
#pragma once
#include "geometry/details/NativeConstructionCounter.hh"
#include "utils/export_helper.hh"
#include <algorithm>
#include <cstddef>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class NativePublicationOperation;
class NativeSourceOwnerStore;
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
namespace SimpleFluid::geometry_detail
{
class NativeConstructionAdmission;
class NativePhaseReservation;
class NativeRetainedExtent;
/** Scalar measurements distinguish actual requests from admitted upper extents. */
struct NativeAdmissionStatistics
{
    std::size_t actual_live = 0, actual_peak = 0, actual_cumulative = 0;
    std::size_t imported_live = 0, profiled_live = 0, reserved_live = 0, admission_peak = 0;
    std::size_t profiled_cumulative_upper = 0, ledger_work_bytes = 0;
    // Reported subextent of the imported prefix, never an additional live charge.
    std::size_t imported_profiled_upper = 0, external_mirror_profiled_upper = 0;
};
/** Only the current source/operation store can issue physical prefix credit. */
class NativePrefixCredit
{
public:
    NativePrefixCredit(NativePrefixCredit&& other) noexcept : bytes_(std::exchange(other.bytes_, 0)) {}
    NativePrefixCredit& operator=(NativePrefixCredit&&) = delete;
    NativePrefixCredit(const NativePrefixCredit&) = delete;
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
    friend class NativeConstructionAdmission;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativePublicationOperation;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativeSourceOwnerStore;
    explicit NativePrefixCredit(std::size_t bytes) noexcept : bytes_(bytes) {}
    std::size_t bytes_;
};
/** Caller-derived complete upper extent, kept separate from actual allocation bytes. */
struct NativeProfileBound
{
    std::size_t bytes;
};
struct SIMPLEFLUID_LOCAL NativeAllocationCounters
{
    std::size_t limit, used, peak, cumulative;
};
/** The erased address aliases counter_owner.get(). It never borrows a caller
 * stack object. A native allocation keeps this detached scalar owner alive.
 */
struct SIMPLEFLUID_LOCAL NativeAllocationState
{
private:
    class ConstructionKey
    {
        friend class NativeConstructionAdmission;
        ConstructionKey() = default;
    public:
        ConstructionKey(const ConstructionKey&) = default;
    };
    friend class NativeConstructionAdmission;
public:
    explicit NativeAllocationState(ConstructionKey) {}
    std::weak_ptr<const NativeAllocationState> self;
    std::shared_ptr<const void> counter_owner;
    void* counter = nullptr;
    std::size_t* peak_counter = nullptr;
    std::size_t* used_counter = nullptr; // aliases the retained counter_owner, never caller stack
    void (*charge_callback)(void*, std::size_t) = nullptr;
    void (*release_callback)(void*, std::size_t) noexcept = nullptr;
    NativeAllocationCounters (*observe_callback)(const void*) noexcept = nullptr;
    std::size_t allocation_admission_bytes = 0;
    NativeAllocationExtent allocation;
    NativeCounterAllocationEnvelope counter_allocations;
    // Published immutable owners can retire on different threads. Their
    // detached counter updates are serialized; no source controller is held.
    mutable std::mutex mutex;
    mutable NativeAdmissionStatistics statistics;
    NativeAllocationMirror mirror;
    // Prefix/profile bookkeeping only changes live admission, never allocator work.
    void reserve(std::size_t bytes, bool imported) const
    {
        bool denied = false;
        {
            const std::lock_guard lock(mutex);
            const auto value = observe_callback(counter);
            denied =
                value.used > value.limit || bytes > value.limit - value.used ||
                (!imported && bytes > std::numeric_limits<std::size_t>::max() - statistics.profiled_cumulative_upper);
            if (!denied && !imported) denied = !mirror.acquire(bytes);
            if (!denied)
            {
                *used_counter += bytes;
                *peak_counter = std::max(*peak_counter, *used_counter);
                if (imported)
                    statistics.imported_live += bytes;
                else
                {
                    statistics.reserved_live += bytes;
                    statistics.profiled_cumulative_upper += bytes;
                }
                statistics.admission_peak = std::max(statistics.admission_peak, *used_counter);
            }
        }
        if (denied)
            throw std::invalid_argument("Native prefix/profile budget is exhausted.");
    }
    void retire_prefix(std::size_t bytes) const noexcept
    {
        const std::lock_guard lock(mutex);
        if (bytes > statistics.imported_live)
            std::terminate();
        statistics.imported_live -= bytes;
        release_callback(counter, bytes);
    }
    void finish_profile(std::size_t bytes, bool retained) const noexcept
    {
        const std::lock_guard lock(mutex);
        auto& total = retained ? statistics.profiled_live : statistics.reserved_live;
        if (bytes > total)
            std::terminate();
        total -= bytes;
        release_callback(counter, bytes);
        mirror.retire(bytes);
    }
    void retain_profile(std::size_t bytes) const
    {
        bool denied = false;
        {
            const std::lock_guard lock(mutex);
            denied = bytes > statistics.reserved_live;
            if (!denied)
            {
                statistics.reserved_live -= bytes;
                statistics.profiled_live += bytes;
            }
        }
        if (denied)
            throw std::invalid_argument("Native retained profile exceeds its phase reservation.");
    }
    [[nodiscard]] NativeAdmissionStatistics measurements() const noexcept
    {
        const std::lock_guard lock(mutex);
        return statistics;
    }
    void record_ledger_work(std::size_t bytes) const
    {
        bool denied = false;
        {
            const std::lock_guard lock(mutex);
            denied = bytes > std::numeric_limits<std::size_t>::max() - statistics.ledger_work_bytes;
            if (!denied)
                statistics.ledger_work_bytes += bytes;
        }
        if (denied)
            throw std::invalid_argument("Native ledger work count overflows.");
    }
    void add(std::size_t bytes) const
    {
        bool denied = false;
        {
            const std::lock_guard lock(mutex);
            const auto value = observe_callback(counter);
            denied = value.used > value.limit || bytes > value.limit - value.used ||
                     bytes > std::numeric_limits<std::size_t>::max() - value.cumulative;
            if (!denied) denied = !mirror.acquire(bytes);
            if (!denied)
            {
                try { charge_callback(counter, bytes); }
                catch (...) { mirror.retire(bytes); throw; }
                statistics.actual_live += bytes;
                statistics.actual_cumulative += bytes;
                statistics.actual_peak = std::max(statistics.actual_peak, statistics.actual_live);
                statistics.admission_peak = std::max(statistics.admission_peak, *used_counter);
            }
        }
        // Diagnostic exception construction happens after releasing the lock.
        if (denied)
            throw std::invalid_argument("Native allocation budget is exhausted.");
    }
    void retire(std::size_t bytes) const noexcept
    {
        const std::lock_guard lock(mutex);
        if (bytes > statistics.actual_live)
            std::terminate();
        statistics.actual_live -= bytes;
        release_callback(counter, bytes);
        mirror.retire(bytes);
    }
    [[nodiscard]] NativeAllocationCounters snapshot() const noexcept
    {
        const std::lock_guard lock(mutex);
        return observe_callback(counter);
    }
};
namespace native_construction_detail
{
/** Used only to create the independent State itself. Its allocator control
 * keeps the same owned scalar counter, with no State self-cycle.
 */
template<class T, class Counter> struct BootstrapAllocator
{
    template<class, class> friend struct BootstrapAllocator;
    using value_type = T;
    std::shared_ptr<Counter> counter;
    NativeAllocationMirror mirror;
    explicit BootstrapAllocator(std::shared_ptr<Counter> value, NativeAllocationMirror external)
        : counter(std::move(value)), mirror(std::move(external)) {}
    template<class U> BootstrapAllocator(const BootstrapAllocator<U, Counter>& other) noexcept : counter(other.counter), mirror(other.mirror)
    {
    }
    T* allocate(std::size_t count)
    {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();
        const auto bytes = count * sizeof(T);
        if (!mirror.acquire(bytes))
            throw std::invalid_argument("Native State bootstrap exceeds source arena headroom.");
        try { counter->add(bytes); }
        catch (...) { mirror.retire(bytes); throw; }
        try
        {
            auto* address = std::allocator<T>{}.allocate(count);
            if constexpr (requires { counter->native_allocation.most_recent_allocation; })
                counter->native_allocation.most_recent_allocation = {address, bytes};
            return address;
        }
        catch (...)
        {
            counter->used -= bytes;
            mirror.retire(bytes);
            throw;
        }
    }
    void deallocate(T* value, std::size_t count) noexcept
    {
        std::allocator<T>{}.deallocate(value, count);
        const auto bytes = count * sizeof(T);
        if (counter->used < bytes)
            std::terminate();
        counter->used -= bytes;
        mirror.retire(bytes);
    }
    template<class U> bool operator==(const BootstrapAllocator<U, Counter>& other) const noexcept
    {
        return counter.get() == other.counter.get() && !counter.owner_before(other.counter) &&
               !other.counter.owner_before(counter);
    }
};
} // namespace native_construction_detail

/** Rebound allocate_shared/map/hash calls charge their actual count and type,
 * including controls, nodes, buckets and over-aligned allocations.
 */
template<class T> class SIMPLEFLUID_LOCAL NativeConstructionAllocator
{
    template<class> friend class NativeConstructionAllocator;

public:
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;
    explicit NativeConstructionAllocator(std::shared_ptr<const NativeAllocationState> state) noexcept
        : state_(std::move(state))
    {
    }
    template<class U>
    NativeConstructionAllocator(const NativeConstructionAllocator<U>& other) noexcept : state_(other.state_)
    {
    }
    [[nodiscard]] T* allocate(std::size_t count)
    {
        if (!state_ || count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();
        if (!count)
            return nullptr;
        const auto bytes = count * sizeof(T);
        state_->add(bytes);
        try
        {
            return std::allocator<T>{}.allocate(count);
        }
        catch (...)
        {
            state_->retire(bytes);
            throw;
        }
    }
    void deallocate(T* address, std::size_t count) noexcept
    {
        if (!count)
        {
            if (address)
                std::terminate();
            return;
        }
        std::allocator<T>{}.deallocate(address, count);
        state_->retire(count * sizeof(T));
    }
    template<class U> [[nodiscard]] bool operator==(const NativeConstructionAllocator<U>& other) const noexcept
    {
        return state_.get() == other.state_.get() && !state_.owner_before(other.state_) &&
               !other.state_.owner_before(state_);
    }

    [[nodiscard]] std::shared_ptr<const NativeAllocationState> state() const noexcept { return state_; }

private:
    std::shared_ptr<const NativeAllocationState> state_;
};
/** Explicit default-or-native allocator for geometry administration only.
 * This allocator borrows State to avoid a keeper cycle in ledger metadata.
 * The containing holder must own the authentic State and destroy every native
 * container before releasing that keeper. Moving a container out separately
 * does not transfer this lifetime obligation. Native producer aliases
 * above/below own their State and remain strict. */
template<class T> class SIMPLEFLUID_LOCAL NativeAdministrationAllocator
{
    template<class> friend class NativeAdministrationAllocator;

public:
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;
    NativeAdministrationAllocator() noexcept = default;
    explicit NativeAdministrationAllocator(std::shared_ptr<const NativeAllocationState> state) noexcept
        : state_(state.get())
    {
    }
    template<class U>
    NativeAdministrationAllocator(const NativeAdministrationAllocator<U>& other) noexcept : state_(other.state_)
    {
    }
    [[nodiscard]] T* allocate(std::size_t count)
    {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();
        if (!count)
            return nullptr;
        const auto bytes = count * sizeof(T);
        if (state_)
            state_->add(bytes);
        try
        {
            return std::allocator<T>{}.allocate(count);
        }
        catch (...)
        {
            if (state_)
                state_->retire(bytes);
            throw;
        }
    }
    void deallocate(T* address, std::size_t count) noexcept
    {
        if (!count)
        {
            if (address)
                std::terminate();
            return;
        }
        std::allocator<T>{}.deallocate(address, count);
        if (state_)
            state_->retire(count * sizeof(T));
    }
    template<class U> [[nodiscard]] bool operator==(const NativeAdministrationAllocator<U>& other) const noexcept
    {
        return state_ == other.state_;
    }

    [[nodiscard]] std::shared_ptr<const NativeAllocationState> state() const noexcept { return state_ ? state_->self.lock() : std::shared_ptr<const NativeAllocationState>{}; }

private:
    const NativeAllocationState* state_ = nullptr;
};
template<class T> using NativeAdministrationVector = std::vector<T, NativeAdministrationAllocator<T>>;
template<class Key, class Value, class Less = std::less<Key>>
using NativeAdministrationMap = std::map<Key, Value, Less, NativeAdministrationAllocator<std::pair<const Key, Value>>>;
template<class T> using NativeConstructorVector = std::vector<T, NativeConstructionAllocator<T>>;
template<class Key, class Value, class Less = std::less<Key>>
using NativeConstructorMap = std::map<Key, Value, Less, NativeConstructionAllocator<std::pair<const Key, Value>>>;
template<class Key, class Value, class Hash = std::hash<Key>, class Equal = std::equal_to<Key>>
using NativeConstructorHash =
    std::unordered_map<Key, Value, Hash, Equal, NativeConstructionAllocator<std::pair<const Key, Value>>>;

template<class T> struct SIMPLEFLUID_LOCAL NativeConstructionDelete
{
    NativeConstructionAllocator<T> allocator;
    void operator()(T* value) noexcept
    {
        if (value)
        {
            std::destroy_at(value);
            allocator.deallocate(value, 1);
        }
    }
};

/** One state is shared across the entire active construction transaction.
 * Counter is an already-owned detached scalar budget with add()/used/limit/
 * peak/cumulative and a NativeCounterAllocationEnvelope. Create one domain per
 * counter and copy that domain; a counter is never rebound to a second state. The caller admits creation of that
 * counter itself. Child constructor allocations must explicitly use this same admission; shared() does not cover
 * uninstrumented child containers.
 */
class SIMPLEFLUID_LOCAL NativeConstructionAdmission
{
public:
    template<class Counter> explicit NativeConstructionAdmission(std::shared_ptr<Counter> counter,
        NativeAllocationMirror mirror = {})
    {
        static_assert(std::is_trivially_copyable_v<Counter> && std::is_trivially_destructible_v<Counter> &&
                          std::is_standard_layout_v<Counter>,
            "Native admission retains only detached scalar counter storage.");
        static_assert(std::is_same_v<decltype(Counter::native_allocation), NativeCounterAllocationEnvelope>);
        const std::shared_ptr<Counter> empty;
        if (!counter || (!counter.owner_before(empty) && !empty.owner_before(counter)) ||
            counter->used > counter->limit)
            throw std::invalid_argument("Native construction requires an owned detached scalar counter.");
        if (counter->native_allocation.admission_state_created)
            throw std::invalid_argument("Copy the existing native admission domain instead of rebinding its counter.");
        const auto before = counter->used;
        state_ = std::allocate_shared<NativeAllocationState>(
            native_construction_detail::BootstrapAllocator<NativeAllocationState, Counter>(counter, mirror),
            NativeAllocationState::ConstructionKey{});
        state_->self = state_;
        state_->mirror = std::move(mirror);
        state_->counter_owner = counter;
        state_->counter = counter.get();
        state_->used_counter = &counter->used;
        state_->peak_counter = &counter->peak;
        state_->charge_callback = [](void* raw, std::size_t bytes)
        {
            auto& value = *static_cast<Counter*>(raw);
            if (value.used > value.limit)
                throw std::invalid_argument("Native allocation counter exceeds its active limit.");
            value.add(bytes);
        };
        state_->release_callback = [](void* raw, std::size_t bytes) noexcept
        {
            auto& value = *static_cast<Counter*>(raw);
            if (value.used < bytes)
                std::terminate();
            value.used -= bytes;
        };
        state_->observe_callback = [](const void* raw) noexcept
        {
            const auto& value = *static_cast<const Counter*>(raw);
            return NativeAllocationCounters{value.limit, value.used, value.peak, value.cumulative};
        };
        state_->allocation_admission_bytes = counter->used - before;
        if constexpr (requires { counter->native_allocation.most_recent_allocation; })
            state_->allocation = counter->native_allocation.most_recent_allocation;
        if (state_->mirror.owner && state_->mirror.owner_proof.kind() == NativeMirrorExtentKind::ProfiledUpper)
            state_->statistics.external_mirror_profiled_upper = state_->mirror.owner_proof.bytes();
        state_->statistics.actual_live = counter->used;
        state_->statistics.actual_peak = counter->used;
        state_->statistics.actual_cumulative = counter->cumulative;
        state_->statistics.admission_peak = counter->peak;
        counter->native_allocation.admission_state_created = true;
    }
    template<class Counter>
    explicit NativeConstructionAdmission(const NativeOwnedCounter<Counter>& counter)
        : NativeConstructionAdmission(counter.owner(), counter.mirror_)
    {
        state_->counter_allocations = counter.allocations();
        if (state_->statistics.actual_live < counter.prefix_)
            std::terminate();
        state_->statistics.actual_live -= counter.prefix_;
        state_->statistics.actual_peak = state_->statistics.actual_live;
        state_->statistics.imported_live = counter.prefix_;
        state_->statistics.imported_profiled_upper = std::min(counter.prefix_, state_->statistics.external_mirror_profiled_upper);
    }
    /** Reuse the one authentic detached State without rebinding its Counter. */
    [[nodiscard]] static NativeConstructionAdmission from_state(std::shared_ptr<const NativeAllocationState> state)
    {
        if (!state || state->self.lock().get() != state.get() || state->self.owner_before(state) ||
            state.owner_before(state->self) || !state->counter_owner || state->counter != state->counter_owner.get() ||
            !state->used_counter || !state->peak_counter || !state->charge_callback || !state->release_callback ||
            !state->observe_callback || !state->allocation.address ||
            state->allocation.bytes != state->allocation_admission_bytes ||
            state->counter_allocations.object.address != state->counter || !state->counter_allocations.object.bytes ||
            !state->counter_allocations.control.address || !state->counter_allocations.control.bytes)
            throw std::invalid_argument("Native admission requires its authentic complete allocation State.");
        return NativeConstructionAdmission(StateTag{}, std::const_pointer_cast<NativeAllocationState>(std::move(state)));
    }
    [[nodiscard]] NativeAllocationCounters counters() const noexcept { return state_->snapshot(); }
    [[nodiscard]] NativeAdmissionStatistics statistics() const noexcept { return state_->measurements(); }
    void record_ledger_work(std::size_t bytes) const { state_->record_ledger_work(bytes); }
    void import_prefix(NativePrefixCredit&& credit) const
    {
        state_->reserve(credit.bytes_, true);
        credit.bytes_ = 0;
    }
    // Operation cleanup must release its source/root/ledger pins before this call.
    void release_prefix(std::size_t bytes) const noexcept { state_->retire_prefix(bytes); }
    [[nodiscard]] NativePhaseReservation reserve_profile(NativeProfileBound) const;
    void charge(std::size_t bytes) const { state_->add(bytes); }
    void release(std::size_t bytes) const noexcept { state_->retire(bytes); }
    [[nodiscard]] std::shared_ptr<const NativeAllocationState> state() const noexcept { return state_; }
    template<class T> [[nodiscard]] NativeConstructionAllocator<T> allocator() const noexcept
    {
        return NativeConstructionAllocator<T>(state_);
    }
    template<class T, class... Args> [[nodiscard]] std::shared_ptr<T> shared(Args&&... args) const
    {
        return std::allocate_shared<T>(allocator<T>(), std::forward<Args>(args)...);
    }
    template<class T, class... Args>
    [[nodiscard]] std::unique_ptr<T, NativeConstructionDelete<T>> unique(Args&&... args) const
    {
        auto memory = allocator<T>();
        auto* address = memory.allocate(1);
        try
        {
            std::construct_at(address, std::forward<Args>(args)...);
        }
        catch (...)
        {
            memory.deallocate(address, 1);
            throw;
        }
        return {address, NativeConstructionDelete<T>{std::move(memory)}};
    }

private:
    struct StateTag {};
    NativeConstructionAdmission(StateTag, std::shared_ptr<NativeAllocationState> state) noexcept : state_(std::move(state)) {}
    std::shared_ptr<NativeAllocationState> state_;
};
/** Scoped opaque-library upper extent. Keep this owner before its actual
 * library objects so destruction releases buffers before retiring the bound. */
class SIMPLEFLUID_LOCAL NativeRetainedExtent
{
public:
    NativeRetainedExtent(NativeRetainedExtent&& other) noexcept
        : state_(std::move(other.state_)), bytes_(std::exchange(other.bytes_, 0))
    {
    }
    NativeRetainedExtent& operator=(NativeRetainedExtent&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            state_ = std::move(other.state_);
            bytes_ = std::exchange(other.bytes_, 0);
        }
        return *this;
    }
    NativeRetainedExtent(const NativeRetainedExtent&) = delete;
    ~NativeRetainedExtent() { reset(); }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] const std::shared_ptr<const NativeAllocationState>& state() const noexcept { return state_; }
    void reset() noexcept
    {
        if (bytes_)
            state_->finish_profile(std::exchange(bytes_, 0), true);
        state_.reset();
    }

private:
    friend class NativePhaseReservation;
    NativeRetainedExtent(std::shared_ptr<const NativeAllocationState> state, std::size_t bytes) noexcept
        : state_(std::move(state)), bytes_(bytes)
    {
    }
    std::shared_ptr<const NativeAllocationState> state_;
    std::size_t bytes_;
};
class SIMPLEFLUID_LOCAL NativePhaseReservation
{
public:
    NativePhaseReservation(NativePhaseReservation&& other) noexcept
        : state_(std::move(other.state_)), bytes_(std::exchange(other.bytes_, 0))
    {
    }
    NativePhaseReservation& operator=(NativePhaseReservation&&) = delete;
    NativePhaseReservation(const NativePhaseReservation&) = delete;
    ~NativePhaseReservation() { finish_scratch(); }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] NativeRetainedExtent retain_profile(std::size_t bytes)
    {
        if (bytes > bytes_)
            throw std::invalid_argument("Native retained extent exceeds its own phase reservation.");
        state_->retain_profile(bytes);
        bytes_ -= bytes;
        return NativeRetainedExtent(state_, bytes);
    }
    void finish_scratch() noexcept
    {
        if (bytes_)
            state_->finish_profile(std::exchange(bytes_, 0), false);
    }

private:
    friend class NativeConstructionAdmission;
    NativePhaseReservation(std::shared_ptr<const NativeAllocationState> state, std::size_t bytes)
        : state_(std::move(state)), bytes_(bytes)
    {
        state_->reserve(bytes, false);
    }
    std::shared_ptr<const NativeAllocationState> state_;
    std::size_t bytes_;
};
inline NativePhaseReservation NativeConstructionAdmission::reserve_profile(NativeProfileBound bound) const
{
    return NativePhaseReservation(state_, bound.bytes);
}
} // namespace SimpleFluid::geometry_detail
