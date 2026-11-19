/** Test-only C++ requested-allocation probe. No probe callback allocates. */
#pragma once
#include "pipeline_allocation_trace.hh"
#include <new>
#include <optional>
namespace native_constructor_test
{
inline thread_local std::optional<size_t> fail_after;
inline thread_local const size_t* admitted_used = nullptr;
inline thread_local size_t admitted_prefix = 0, uncovered = 0;
struct AllocationRequestObserver
{
    void* context = nullptr;
    void (*visit)(void*, std::size_t) noexcept = nullptr;
};
inline thread_local AllocationRequestObserver allocation_request_observer;
inline void before_allocate(size_t bytes)
{
    if (allocation_request_observer.visit)
        allocation_request_observer.visit(allocation_request_observer.context, bytes);
    if (fail_after)
    {
        if (*fail_after == 0)
        {
            fail_after.reset();
            throw std::bad_alloc();
        }
        --*fail_after;
    }
    if (admitted_used && pipeline_trace::capturing() &&
        (*admitted_used < admitted_prefix || *admitted_used - admitted_prefix < pipeline_trace::all_live + bytes))
        ++uncovered;
}
struct Failure
{
    std::optional<size_t> previous;
    explicit Failure(size_t ordinal) : previous(std::exchange(fail_after, ordinal)) {}
    ~Failure() { fail_after = previous; }
};
struct Capture
{
    size_t* used;
    bool active = true;
    explicit Capture(size_t& count) : used(&count)
    {
        admitted_used = &count;
        admitted_prefix = count;
        uncovered = 0;
        pipeline_trace::begin();
    }
    Capture(const Capture&) = delete;
    Capture& operator=(const Capture&) = delete;
    pipeline_trace::Summary finish()
    {
        pipeline_trace::recording.store(false);
        admitted_used = nullptr;
        active = false;
        return pipeline_trace::summary();
    }
    ~Capture()
    {
        if (active)
        {
            pipeline_trace::recording.store(false);
            admitted_used = nullptr;
        }
    }
};
} // namespace native_constructor_test
