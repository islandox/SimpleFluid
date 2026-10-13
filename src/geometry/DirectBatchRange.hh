/** @file DirectBatchRange.hh
 * @brief Paired contiguous topology and metric ranges for numerical kernels.
 */
#pragma once

#include <cassert>
#include <cstddef>
#include <span>

namespace SimpleFluid
{
/**
 * @brief Borrow equally ordered immutable topology and geometry arrays.
 *
 * The owner must outlive the range and all its uses. Binding checks belong to
 * the owning execution scope; this range performs no mesh queries, geometry
 * acquisition, field synchronization, or per-entity selection. Grouped positions
 * index only these arrays, never the mesh's field or boundary-value storage.
 */
template<class Topology, class Metrics> class DirectBatchRange
{
public:
    DirectBatchRange() = default;

    /** Validate the pairing once, before entering the numerical operation. */
    static DirectBatchRange checked(std::span<const Topology> topology, std::span<const Metrics> metrics);

    size_t size() const noexcept { return d_size; }
    bool empty() const noexcept { return d_size == 0; }
    std::span<const Topology> topology() const noexcept { return {d_topology, d_size}; }
    std::span<const Metrics> metrics() const noexcept { return {d_metrics, d_size}; }

    /** @pre begin/count designate a valid subrange of this already checked view. */
    DirectBatchRange subrange(size_t begin, size_t count) const noexcept
    {
        assert(begin <= d_size && count <= d_size - begin);
        return {begin == 0 ? d_topology : d_topology + begin, begin == 0 ? d_metrics : d_metrics + begin, count};
    }

    template<class Operation> void apply(Operation&& operation) const;

private:
    DirectBatchRange(const Topology* topology, const Metrics* metrics, size_t count) noexcept;

    const Topology* d_topology = nullptr;
    const Metrics* d_metrics = nullptr;
    size_t d_size = 0;
};
} // namespace SimpleFluid

#include "geometry/DirectBatchRange.ipp"
