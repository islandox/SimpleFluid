/** @file OrdinaryAggregate.hh
 * @brief Ordinary signed-chain measures without measured computational children.
 */
#pragma once
#include "geometry/PLIC/ConvexGeometry.hh"
#include <limits>
#include <memory>
#include <optional>

namespace SimpleFluid::PLIC
{
struct OrdinaryAggregateTetra
{
    std::array<Point<ArithmeticMode::Ordinary>, 4> vertices;
    /** Identities belong to this source batch; equal coordinates do not issue an identity. */
    std::array<std::uint64_t, 4> vertex_ids{};
    int sign = 1;
};
struct OrdinaryAggregateDonor
{
    std::uint64_t id = 0;
    ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry;
    /** An absent plane requires exactly one pure declaration. */
    std::optional<GeometryHalfspace> gas;
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
struct OrdinaryAggregateStatistics
{
    /** cancelled_boundaries counts authenticated source-face partner pairs;
     * each donor/sign/phase channel removes only pairs whose two faces survive. */
    std::size_t source_pieces = 0, witnesses = 0, cancelled_boundaries = 0, retries = 0;
    std::size_t storage_bytes = 0, pair_tests = 0, operations = 0;
};
struct OrdinaryAggregateResult
{
    std::vector<OrdinaryAggregateGroup> groups;
    long double signed_source_volume = 0;
    OrdinaryAggregateStatistics statistics;
};
/** Optional bounded progress snapshot for an exception's accounting owner.
 * Never retained by the source; absent identities are not invented on failure. */
struct OrdinaryAggregateContext
{
    enum class Stage
    {
        Source,
        DonorValidation,
        Coverage,
        Metric
    };
    static constexpr auto absent = std::numeric_limits<std::uint64_t>::max();
    Stage stage = Stage::Source;
    std::uint64_t source = absent, donor = absent, cut = absent;
    unsigned retry = 0;
    std::size_t bytes = 0, pairs = 0, operations = 0;
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
    explicit OrdinaryAggregateSource(std::span<const OrdinaryAggregateTetra>, SweepLimits = {},
        GeometryLimits = {.arithmetic_mode = ArithmeticMode::Ordinary}, OrdinaryAggregateContext* = nullptr);
    [[nodiscard]] OrdinaryAggregateResult measure(
        std::span<const OrdinaryAggregateDonor>, OrdinaryAggregateContext* = nullptr) const;
    [[nodiscard]] std::size_t storage_bytes() const noexcept;

private:
    std::shared_ptr<const Impl> impl_;
};
} // namespace SimpleFluid::PLIC
