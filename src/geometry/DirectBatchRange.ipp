/** @file DirectBatchRange.ipp
 * @brief Inline accessors and numerical loops for DirectBatchRange.
 */
#pragma once

#include "geometry/DirectBatchRange.hh"

#include <cassert>
#include <functional>
#include <stdexcept>

namespace SimpleFluid
{

template<class Topology, class Metrics>
template<class Operation>
inline void DirectBatchRange<Topology, Metrics>::apply(Operation&& operation) const
{
    const auto* topology = d_topology;
    const auto* metrics = d_metrics;
    const auto count = d_size;
    for (size_t i = 0; i < count; ++i)
        std::invoke(operation, topology[i], metrics[i]);
}

template<class Topology, class Metrics>
inline DirectBatchRange<Topology, Metrics>::DirectBatchRange(
    const Topology* topology, const Metrics* metrics, size_t count) noexcept
    : d_topology(topology), d_metrics(metrics), d_size(count)
{
}

template<class Topology, class Metrics>
inline DirectBatchRange<Topology, Metrics> DirectBatchRange<Topology, Metrics>::checked(
    std::span<const Topology> topology, std::span<const Metrics> metrics)
{
    if (topology.size() != metrics.size())
        throw std::invalid_argument("Direct batch topology and metrics have different lengths.");
    return {topology.data(), metrics.data(), topology.size()};
}

} // namespace SimpleFluid
