#pragma once
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/swept_volume/OrdinaryAggregate.hh"
#include <span>
#include <string_view>

namespace SimpleFluid::geometry::swept_volume::details
{
/** Private, synchronous diagnostics for a same-image probe. Every view is
 * borrowed for the callback only; observers must not retain provenance pins.
 * Callbacks may inspect borrowed records only: no geometry work, publication,
 * serialization or recursive aggregate invocation is permitted.
 * No observer is installed in ordinary production execution. */
struct OrdinaryAggregateRetentionObservation
{
    std::string_view stage;
    std::span<const PLIC::Point<ArithmeticMode::Ordinary>> source_points;
    std::span<const OrdinaryAggregateDonor> donors;
    const PLIC::GeometryAllocationLedger& ledger;
    const OrdinaryAggregateStatistics& statistics;
};
using OrdinaryAggregateRetentionCallback = void (*)(const OrdinaryAggregateRetentionObservation&, void*);
class SIMPLEFLUID_LOCAL ScopedOrdinaryAggregateRetentionObserver
{
public:
    ScopedOrdinaryAggregateRetentionObserver(OrdinaryAggregateRetentionCallback, void*) noexcept;
    ~ScopedOrdinaryAggregateRetentionObserver();
    ScopedOrdinaryAggregateRetentionObserver(const ScopedOrdinaryAggregateRetentionObserver&) = delete;
    ScopedOrdinaryAggregateRetentionObserver& operator=(const ScopedOrdinaryAggregateRetentionObserver&) = delete;

private:
    OrdinaryAggregateRetentionCallback previous_callback_;
    void* previous_state_;
};
} // namespace SimpleFluid::geometry::swept_volume::details
