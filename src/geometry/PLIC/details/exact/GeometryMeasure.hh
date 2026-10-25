#pragma once
#include "geometry/PLIC/CertifiedGeometryProperties.hh"
#include "geometry/PLIC/GeometryBackendTypes.hh"
#include <span>
#include <vector>
namespace SimpleFluid::PLIC
{
/** Exact rational scalar arithmetic retained independently of output rounding.
 * Geometry producers use volume units (m^3); derived inventory, momentum and
 * residual consumers retain their own physical units. This type does not check
 * dimensions. Conversion has the ConvexCellGeometry<ArithmeticMode::Exact> finite/range contract. */
template<> class GeometryMeasure<ArithmeticMode::Exact>
{
public:
    struct Impl;
    GeometryMeasure() = default;
    [[nodiscard]] static GeometryMeasure<ArithmeticMode::Exact> from_value(double);
    [[nodiscard]] static GeometryMeasure<ArithmeticMode::Exact> volume(
        const ConvexCellGeometry<ArithmeticMode::Exact>&);
    [[nodiscard]] static GeometryMeasure<ArithmeticMode::Exact> volume(
        const ConvexCellGeometry<ArithmeticMode::Ordinary>&);
    [[nodiscard]] GeometryValue<ArithmeticMode::Exact> certified() const;
    /** Diagnostic enclosure, including nonzero residuals below output range.
     * exact_nonzero survives an interval containing zero; publication through
     * as_double() still rejects disappearance of a known nonzero measure. */
    [[nodiscard]] GeometryValue<ArithmeticMode::Exact> enclosure() const;
    /** Exact relative acceptance; never forms tolerance*reference in floating
     * arithmetic. A zero reference admits only exact equality. */
    [[nodiscard]] bool relative_error_at_most(
        const GeometryMeasure<ArithmeticMode::Exact>& reference, double tolerance) const;
    /** Approximate interpolation aid only, never an acceptance predicate.
     * Positive rational operands are normalized before taking a floating log. */
    [[nodiscard]] double log_ratio(const GeometryMeasure<ArithmeticMode::Exact>& denominator) const;
    [[nodiscard]] std::vector<char> encode(std::size_t byte_budget = 1024 * 1024) const;
    [[nodiscard]] static GeometryMeasure<ArithmeticMode::Exact> decode(
        std::span<const char>, std::size_t byte_budget = 1024 * 1024);
    [[nodiscard]] int sign() const;
    [[nodiscard]] int compare(const GeometryMeasure<ArithmeticMode::Exact>&) const;
    [[nodiscard]] GeometryMeasure<ArithmeticMode::Exact> operator+(const GeometryMeasure<ArithmeticMode::Exact>&) const;
    [[nodiscard]] GeometryMeasure<ArithmeticMode::Exact> operator-(const GeometryMeasure<ArithmeticMode::Exact>&) const;
    [[nodiscard]] GeometryMeasure<ArithmeticMode::Exact> scaled(double) const;
    /** Exact arithmetic with a retained-result byte limit and a conservative
     * integer bit-count preflight; temporary allocator peak is separate. */
    [[nodiscard]] GeometryMeasure<ArithmeticMode::Exact> divided_by(
        const GeometryMeasure<ArithmeticMode::Exact>&, std::size_t byte_budget = 1024 * 1024) const;
    [[nodiscard]] GeometryMeasure<ArithmeticMode::Exact> squared(std::size_t byte_budget = 1024 * 1024) const;
    /** Fresh exact copy without arithmetic allocation slack. The explicit lease
     * covers simultaneous source and destination capacities; source is unchanged.
     * No represented-value conversion or change to other measure factories. */
    [[nodiscard]] GeometryMeasure<ArithmeticMode::Exact> compacted(std::size_t operation_byte_budget) const;
    /** Truncate toward zero to at most significant_bits binary significand bits
     * at the exact value's own exponent. Nonzero signs survive below binary64
     * range. This is a proposed representation only: callers must independently
     * admit its exact discarded difference against their original certificate.
     * The lease covers source, result and conservative integer workspace;
     * resource failures throw length_error, distinct from malformed inputs. */
    [[nodiscard]] GeometryMeasure<ArithmeticMode::Exact> rounded_dyadic(
        unsigned significant_bits, std::size_t operation_byte_budget) const;
    /** Capacity estimate, including the opaque rational allocation; excludes
     * allocator/control-block overhead. Deduplicate shared storage_identity(). */
    [[nodiscard]] std::size_t storage_bytes() const;
    [[nodiscard]] const void* storage_identity() const noexcept { return impl_.get(); }

private:
    friend class SweepAccess<ArithmeticMode::Exact>;
    friend class SweptFace<ArithmeticMode::Exact>;
    friend class DonorExport<ArithmeticMode::Exact>;
    explicit GeometryMeasure(std::shared_ptr<const Impl>);
    std::shared_ptr<const Impl> impl_;
};
} // namespace SimpleFluid::PLIC
