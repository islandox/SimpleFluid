/** @file pipeline_allocation_trace.hh
 * @brief Requested C++ allocation interposition with allocation-time domains.
 */
/** Test-only requested C++ allocation trace. No tracing callback allocates.
 * C/MPI/MPFR malloc traffic and allocator RSS remain outside this instrument. */
#pragma once
#if SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT
#include "geometry/details/AllocationDomainScope.hh"
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <utility>

namespace pipeline_trace
{
#if SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT
using SimpleFluid::geometry_detail::AllocationDomain;
inline constexpr std::size_t domains = static_cast<unsigned>(AllocationDomain::Count);
#endif
enum class Phase : unsigned
{
    Candidate,
    Transport,
    Directory,
    NativePublication,
    NativeCellRoutes,
    Count
};
inline constexpr std::size_t phases = static_cast<unsigned>(Phase::Count);
struct Totals
{
    std::size_t allocations = 0, bytes = 0, live = 0, peak = 0, deallocations = 0;
};
struct Entry
{
    const void* address = nullptr;
    std::size_t bytes = 0;
    std::uint64_t serial = 0, epoch = 0;
    Phase phase = Phase::Candidate;
    unsigned membership = 0;
    bool live = false, used = false;
#if SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT
    AllocationDomain domain = AllocationDomain::Unattributed;
#endif
};
struct Summary
{
    std::array<Totals, phases> phase{}, retained_phase{};
#if SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT
    std::array<Totals, domains> domain{};
    std::size_t unattributed_live_allocations = 0, unattributed_live_bytes = 0;
#endif
    std::size_t live_allocations = 0;
    std::size_t all_live = 0, all_peak = 0, incremental_peak = 0, retained_allocations = 0, retained_bytes = 0;
    std::size_t retained_streaming_bytes = 0, shared_allocations = 0, shared_bytes = 0;
};
inline constexpr std::size_t slots = 1u << 20;
inline std::array<Entry, slots> table;
inline std::array<Entry*, slots> ordered;
inline std::size_t ordered_size = 0, active_slots = slots;
inline std::atomic_flag lock = ATOMIC_FLAG_INIT;
inline std::atomic<bool> recording = false, overflow = false;
inline std::uint64_t serial = 0, epoch = 0;
inline std::array<Totals, phases> totals{};
#if SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT
inline std::array<Totals, domains> domain_totals{};
#endif
inline std::size_t all_live = 0, all_peak = 0, baseline_live = 0;
inline thread_local Phase phase = Phase::Candidate;
inline thread_local bool paused = false;
inline bool capturing() noexcept { return !paused && recording.load(std::memory_order_relaxed); }
struct Guard
{
    Guard() noexcept
    {
        while (lock.test_and_set(std::memory_order_acquire))
        {
        }
    }
    ~Guard() { lock.clear(std::memory_order_release); }
};
inline std::size_t bucket(const void* value) noexcept
{
    auto bits = reinterpret_cast<std::uintptr_t>(value);
    bits ^= bits >> 17;
    bits *= std::uintptr_t(0x9e3779b97f4a7c15ULL);
    return bits & (active_slots - 1);
}
inline void allocated(const void* value, std::size_t bytes) noexcept
{
    if (!capturing())
        return;
    const Guard guard;
    auto position = bucket(value);
    for (std::size_t visit = 0; visit != active_slots; ++visit, position = (position + 1) & (active_slots - 1))
    {
        auto& entry = table[position];
        if (entry.live)
            continue;
        entry = {value, bytes, ++serial, epoch, phase, 0, true, true};
        const auto record = [bytes](Totals& count)
        {
            ++count.allocations;
            count.bytes += bytes;
            count.live += bytes;
            count.peak = std::max(count.peak, count.live);
        };
        record(totals[static_cast<unsigned>(phase)]);
#if SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT
        entry.domain = SimpleFluid::geometry_detail::allocation_domain_slot();
        record(domain_totals[static_cast<unsigned>(entry.domain)]);
#endif
        all_live += bytes;
        all_peak = all_peak > all_live ? all_peak : all_live;
        return;
    }
    overflow.store(true, std::memory_order_relaxed);
}
inline void freed(const void* value) noexcept
{
    if (!value)
        return;
    const Guard guard;
    auto position = bucket(value);
    for (std::size_t visit = 0; visit != active_slots; ++visit, position = (position + 1) & (active_slots - 1))
    {
        auto& entry = table[position];
        if (!entry.used)
            return;
        if (entry.live && entry.address == value)
        {
            auto& count = totals[static_cast<unsigned>(entry.phase)];
            count.live -= entry.bytes;
            ++count.deallocations;
#if SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT
            auto& domain = domain_totals[static_cast<unsigned>(entry.domain)];
            domain.live -= entry.bytes;
            ++domain.deallocations;
#endif
            all_live -= entry.bytes;
            entry.live = false;
            return;
        }
    }
}
struct Pause
{
    bool previous = std::exchange(paused, true);
    Pause() = default;
    Pause(const Pause&) = delete;
    Pause& operator=(const Pause&) = delete;
    ~Pause() { paused = previous; }
};
inline void begin(std::size_t capacity = slots)
{
    const Guard guard;
    for (auto& entry : table)
        entry = {};
    active_slots = capacity;
    ordered_size = 0;
    serial = epoch = 0;
    totals = {};
#if SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT
    domain_totals = {};
#endif
    all_live = all_peak = baseline_live = 0;
    overflow.store(false);
    paused = false;
    recording.store(true);
}
inline void next_epoch()
{
    const Guard guard;
    ++epoch;
    const auto reset_epoch = [](auto& channels)
    {
        for (auto& count : channels)
        {
            count.allocations = count.bytes = count.deallocations = 0;
            count.peak = count.live;
        }
    };
    reset_epoch(totals);
#if SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT
    reset_epoch(domain_totals);
#endif
    all_peak = baseline_live = all_live;
}
inline Summary summary()
{
    const Guard guard;
    Summary result;
    result.phase = totals;
#if SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT
    result.domain = domain_totals;
#endif
    result.all_live = all_live;
    result.all_peak = all_peak;
    result.incremental_peak = all_peak - baseline_live;
    for (const auto& entry : table)
    {
        if (!entry.live)
            continue;
        ++result.live_allocations;
#if SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT
        if (entry.domain == AllocationDomain::Unattributed)
        {
            ++result.unattributed_live_allocations;
            result.unattributed_live_bytes += entry.bytes;
        }
#endif
        if (entry.epoch == epoch)
        {
            ++result.retained_allocations;
            result.retained_bytes += entry.bytes;
            auto& retained = result.retained_phase[static_cast<unsigned>(entry.phase)];
            ++retained.allocations;
            retained.bytes += entry.bytes;
            if (entry.phase == Phase::Directory || entry.phase == Phase::Transport)
                result.retained_streaming_bytes += entry.bytes;
        }
        else
        {
            ++result.shared_allocations;
            result.shared_bytes += entry.bytes;
        }
    }
    return result;
}
/** Mark a physical graph owner by its enclosing real allocation. Shared-control
 * prefixes and embedded/aliased subobjects therefore cannot inflate the union. */
inline void prepare_census()
{
    const Guard guard;
    ordered_size = 0;
    for (auto& entry : table)
        if (entry.live)
            ordered[ordered_size++] = &entry;
    std::sort(ordered.begin(), ordered.begin() + ordered_size, [](const Entry* a, const Entry* b)
        { return reinterpret_cast<std::uintptr_t>(a->address) < reinterpret_cast<std::uintptr_t>(b->address); });
}
inline const Entry* mark(const void* object, unsigned membership)
{
    const Guard guard;
    const auto address = reinterpret_cast<std::uintptr_t>(object);
    const auto upper = std::upper_bound(ordered.begin(), ordered.begin() + ordered_size, address,
        [](std::uintptr_t value, const Entry* entry)
        { return value < reinterpret_cast<std::uintptr_t>(entry->address); });
    if (upper == ordered.begin())
        return nullptr;
    auto* entry = *std::prev(upper);
    const auto first = reinterpret_cast<std::uintptr_t>(entry->address);
    if (address - first >= entry->bytes)
        return nullptr;
    entry->membership |= membership;
    return entry;
}
/** Only for tests of an explicit owner's address inventory, not domain attribution. */
inline std::pair<std::size_t, std::size_t> unmarked_graph_extents()
{
    const Guard guard;
    std::pair<std::size_t, std::size_t> result{};
    for (const auto& entry : table)
        if (entry.live && !entry.membership)
        {
            ++result.first;
            result.second += entry.bytes;
        }
    return result;
}
inline std::array<std::size_t, 8> physical_unions()
{
    const Guard guard;
    std::array<std::size_t, 8> bytes{};
    for (const auto& entry : table)
        if (entry.live && entry.membership)
            bytes[entry.membership] += entry.bytes;
    return bytes;
}
} // namespace pipeline_trace
