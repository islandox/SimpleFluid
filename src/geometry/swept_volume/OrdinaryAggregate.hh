/** @file OrdinaryAggregate.hh
 * @brief Ordinary signed-chain measures without measured computational children.
 */
#pragma once
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/swept_volume/SweptVolumeDeclarations.hh"
#include <compare>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace SimpleFluid::geometry::swept_volume
{
/** Identity of one slot in an immutable source point table. Coordinate equality
 * does not merge slots or authenticate source-face incidence. */
struct OrdinaryAggregatePointIndex
{
    std::size_t value = 0;
    auto operator<=>(const OrdinaryAggregatePointIndex&) const = default;
};
struct OrdinaryAggregateTetra
{
    std::array<OrdinaryAggregatePointIndex, 4> vertices;
    int sign = 1;
};
struct OrdinaryAggregateSourceBatch
{
    std::vector<PLIC::Point<ArithmeticMode::Ordinary>> points;
    std::vector<OrdinaryAggregateTetra> tetrahedra;
};
struct OrdinaryAggregateDonor
{
    std::uint64_t id = 0;
    PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry;
    /** An absent plane requires exactly one pure declaration. */
    std::optional<PLIC::GeometryHalfspace> gas;
    bool pure_gas = false, pure_carrier = false;
    /** Independent physical scales, supplied by the accounting owner. */
    long double bulk_budget = 0, gas_budget = 0, carrier_budget = 0;
};
struct OrdinaryAggregateAmounts
{
    long double bulk = 0, gas = 0, carrier = 0;
};
struct OrdinaryAggregateGroup
{
    std::uint64_t donor = 0;
    int sign = 1;
    OrdinaryAggregateAmounts nominal;
    /** Numerical quality indicators, never certified bounds. */
    OrdinaryAggregateAmounts numerical_error_estimate;
    /** Conservative explicitly approximated contribution radius; separate from roundoff. */
    OrdinaryAggregateAmounts approximation_bound;
    /** Strict interiors observed structurally; zero publication must retain these obligations. */
    bool bulk_positive = false, gas_positive = false, carrier_positive = false;
};
enum class OrdinaryAggregateStage
{
    Source,
    DonorValidation,
    Coverage,
    Metric
};
struct OrdinaryAggregateStatistics
{
    /** cancelled_boundaries counts authenticated source-face partner pairs;
     * each donor/sign/phase channel removes only pairs whose two faces survive. */
    /** witnesses counts requested construction admissions, including empty
     * complementary children; it is not a live-object count. */
    std::size_t source_pieces = 0, witnesses = 0, cancelled_boundaries = 0, retries = 0;
    /** Cumulative construction charges, including destroyed temporary geometry;
     * these counters do not describe simultaneous live memory or RSS. */
    std::size_t storage_bytes = 0, pair_tests = 0, operations = 0;
    std::array<std::size_t, 4> stage_charged_bytes{};
    /** Requested bytes of the charge rejected by the cumulative allowance. */
    std::size_t failed_charge_bytes = 0;
    std::size_t vertices_classified = 0, vertices_copied = 0;
    /** Admitted charges for copied vertex, error, and support arrays. */
    std::size_t copied_vertex_bytes = 0;
    /** Vertex counts summed over clipping observations, including repeated visits. */
    std::size_t active_vertices_seen = 0, stored_vertices_seen = 0;
    std::size_t peak_active_vertices = 0, peak_stored_vertices = 0, peak_remainders = 0;
    /** High-water marks for tracked Poly objects and their owned vertex, error,
     * support and face capacities, observed at allocation mutations.
     * Excludes allocator metadata, maps, source/support caches, predicate scratch,
     * and total RSS; this is not a whole-memory measurement. */
    std::size_t peak_live_witnesses = 0, peak_live_witness_bytes = 0;
};
struct OrdinaryAggregateResult
{
    std::vector<OrdinaryAggregateGroup> groups;
    long double signed_source_volume = 0;
    OrdinaryAggregateStatistics statistics;
    /** Result object and owned group capacity, included in statistics.storage_bytes.
     * Excludes the source and temporary construction charges in that counter. */
    [[nodiscard]] std::size_t storage_bytes() const noexcept;
};
/** Optional bounded progress snapshot for an exception's accounting owner.
 * Never retained by the source; absent identities are not invented on failure. */
struct OrdinaryAggregateContext
{
    using Stage = OrdinaryAggregateStage;
    static constexpr auto absent = std::numeric_limits<std::uint64_t>::max();
    Stage stage = Stage::Source;
    std::uint64_t source = absent, donor = absent, cut = absent;
    unsigned retry = 0;
    std::size_t bytes = 0, pairs = 0, operations = 0;
    OrdinaryAggregateStatistics statistics;
};
/** Immutable recipes and authenticated source incidence. Only final aggregates
 * are measured. Each source-tetra coverage proof and each donor/sign physical
 * aggregate permits one reconstruction retry. Unresolved quality then rejects;
 * a zero approximation bound never authorizes discarding a witness.
 * No surface area or curvature accuracy contract is provided by this API. */
class OrdinaryAggregateSource
{
public:
    struct Impl;
    /** Validates and allocates metadata before consuming the batch. On every
     * constructor failure, both caller vectors and their elements are unchanged.
     * Success transfers the buffers; callers must relinquish mutable aliases. */
    explicit OrdinaryAggregateSource(OrdinaryAggregateSourceBatch&&, SweepLimits = {},
        PLIC::GeometryLimits = {.arithmetic_mode = ArithmeticMode::Ordinary}, OrdinaryAggregateContext* = nullptr);
    /** Borrowed immutable views remain valid while this source owns its batch,
     * including after a measurement failure. */
    [[nodiscard]] std::span<const PLIC::Point<ArithmeticMode::Ordinary>> points() const& noexcept;
    [[nodiscard]] std::span<const OrdinaryAggregateTetra> tetrahedra() const& noexcept;
    std::span<const PLIC::Point<ArithmeticMode::Ordinary>> points() const&& = delete;
    std::span<const OrdinaryAggregateTetra> tetrahedra() const&& = delete;
    [[nodiscard]] OrdinaryAggregateResult measure(
        std::span<const OrdinaryAggregateDonor>, OrdinaryAggregateContext* = nullptr) const;
    [[nodiscard]] std::size_t storage_bytes() const noexcept;

private:
    std::shared_ptr<const Impl> impl_;
};
} // namespace SimpleFluid::geometry::swept_volume
