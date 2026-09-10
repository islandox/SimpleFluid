/** @file GeometryExecutionGuard.hh
 * @brief Scoped read access for fixed-topology geometry execution.
 */
#pragma once

#include <atomic>
#include <stdexcept>
#include <utility>

namespace SimpleFluid
{
/**
 * @brief Prevent supported mutation and value replacement during a bulk read.
 *
 * This must be the first base of a mutable value mesh: its assignment/move
 * checks run before any topology or coordinate member is changed. Concurrent
 * readers are supported; as with mesh motion, writers require external
 * synchronization. A lease borrows its object and must not outlive it.
 */
class GeometryExecutionGuard
{
public:
    class ReadLease
    {
    public:
        explicit ReadLease(const GeometryExecutionGuard& source) : d_source(&source)
        { d_source->d_readers.fetch_add(1, std::memory_order_relaxed); }
        ReadLease(const ReadLease&) = delete;
        ReadLease& operator=(const ReadLease&) = delete;
        ReadLease(ReadLease&& other) noexcept : d_source(std::exchange(other.d_source, nullptr)) {}
        ReadLease& operator=(ReadLease&&) = delete;
        ~ReadLease()
        { if (d_source) d_source->d_readers.fetch_sub(1, std::memory_order_relaxed); }
    private:
        const GeometryExecutionGuard* d_source;
    };

    [[nodiscard]] ReadLease acquire_geometry_read() const { return ReadLease(*this); }
    void require_geometry_writable() const
    {
        if (d_readers.load(std::memory_order_relaxed))
            throw std::logic_error("Geometry is held by a region execution view.");
    }

protected:
    GeometryExecutionGuard() = default;
    GeometryExecutionGuard(const GeometryExecutionGuard&) noexcept {}
    GeometryExecutionGuard(GeometryExecutionGuard&& source) { source.require_geometry_writable(); }
    GeometryExecutionGuard& operator=(const GeometryExecutionGuard&)
    {
        require_geometry_writable();
        return *this;
    }
    GeometryExecutionGuard& operator=(GeometryExecutionGuard&& source)
    {
        require_geometry_writable();
        if (this != &source) source.require_geometry_writable();
        return *this;
    }

private:
    mutable std::atomic<size_t> d_readers{0};
};

/** @brief Enter a mesh's bulk execution boundary, or a no-op for legacy meshes. */
template<class Mesh> [[nodiscard]] auto acquire_mesh_execution(const Mesh& mesh)
{
    if constexpr (requires { mesh.acquire_execution_view(); })
        return mesh.acquire_execution_view();
    else
    {
        struct UnchangedNativeExecution {};
        return UnchangedNativeExecution{};
    }
}
} // namespace SimpleFluid
