/** @file pipeline_allocation_operators.inc
 * @brief All replaceable global C++ allocation and deallocation forms.
 */
#include "native_constructor_allocation_test.hh"
#include "pipeline_allocation_trace.hh"
#include "pipeline_allocation_domain.inc"
#include <cstdlib>
#include <new>

void* operator new(std::size_t bytes)
{
    native_constructor_test::before_allocate(bytes);
    if (auto* value = std::malloc(bytes ? bytes : 1))
    {
        pipeline_trace::allocated(value, bytes);
        return value;
    }
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes)
{
    return ::operator new(bytes);
}
void* operator new(std::size_t bytes, std::align_val_t alignment)
{
    native_constructor_test::before_allocate(bytes);
    void* value = nullptr;
    if (posix_memalign(&value, static_cast<std::size_t>(alignment), bytes ? bytes : 1))
        throw std::bad_alloc();
    pipeline_trace::allocated(value, bytes);
    return value;
}
void* operator new[](std::size_t bytes, std::align_val_t alignment)
{
    return ::operator new(bytes, alignment);
}
void* operator new(std::size_t bytes, const std::nothrow_t&) noexcept
{
    try
    {
        return ::operator new(bytes);
    }
    catch (...)
    {
        return nullptr;
    }
}
void* operator new[](std::size_t bytes, const std::nothrow_t&) noexcept
{
    try
    {
        return ::operator new[](bytes);
    }
    catch (...)
    {
        return nullptr;
    }
}
void* operator new(std::size_t bytes, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    try
    {
        return ::operator new(bytes, alignment);
    }
    catch (...)
    {
        return nullptr;
    }
}
void* operator new[](std::size_t bytes, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    try
    {
        return ::operator new[](bytes, alignment);
    }
    catch (...)
    {
        return nullptr;
    }
}
void operator delete(void* value) noexcept
{
    pipeline_trace::freed(value);
    std::free(value);
}
void operator delete[](void* value) noexcept
{
    ::operator delete(value);
}
void operator delete(void* value, std::size_t) noexcept
{
    ::operator delete(value);
}
void operator delete[](void* value, std::size_t) noexcept
{
    ::operator delete(value);
}
void operator delete(void* value, std::align_val_t) noexcept
{
    ::operator delete(value);
}
void operator delete[](void* value, std::align_val_t) noexcept
{
    ::operator delete(value);
}
void operator delete(void* value, std::size_t, std::align_val_t) noexcept
{
    ::operator delete(value);
}
void operator delete[](void* value, std::size_t, std::align_val_t) noexcept
{
    ::operator delete(value);
}
void operator delete(void* value, const std::nothrow_t&) noexcept
{
    ::operator delete(value);
}
void operator delete[](void* value, const std::nothrow_t&) noexcept
{
    ::operator delete(value);
}
void operator delete(void* value, std::align_val_t, const std::nothrow_t&) noexcept
{
    ::operator delete(value);
}
void operator delete[](void* value, std::align_val_t, const std::nothrow_t&) noexcept
{
    ::operator delete(value);
}
