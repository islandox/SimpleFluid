/** @file ExactPhaseLabelMap.hh
 * @brief Bounded donor-wide complementary geometric phase label correction.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/PLIC/SweptGeometry.hh"
namespace SimpleFluid::VoF::detail
{
[[nodiscard]] SIMPLEFLUID_FVM_EXPORT SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>
exact_phase_product(const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>&,
    const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>&, std::size_t byte_budget);
/** One donor-wide complementary label map. It repairs only a separately
 * bounded geometric minority mismatch; independent old inventories and their
 * numerical occupancy defect remain unchanged. This map is linear in the raw
 * complementary pair, so a fixed donor map may be applied after exact raw
 * aggregation. Preserve mapped-zero row checks (including singular maps),
 * separate signed face groups, and original piece multiplicity in repair bounds.
 * Its correction has one fixed sign on positive pieces: create changes only
 * a or b, never both. Independent physical inventories are not relabelled. */
class SIMPLEFLUID_FVM_EXPORT ExactPhaseLabelMap
{
public:
    [[nodiscard]] static ExactPhaseLabelMap create(
        const SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>& full, bool gas_is_minority,
        const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>& authoritative_minority,
        const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>& discrepancy_bound,
        std::size_t byte_budget);
    [[nodiscard]] SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> apply(
        const SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>&) const;
    /** Explicit call-local arithmetic arena, positive and no larger than the
     * captured construction limit. Conservatively includes raw/coefficient
     * inputs and every intermediate, with an eight-times operand-capacity
     * pending-result preflight. Backend allocator/GCD temporaries and preceding
     * geometric clipping are not a process-peak promise. Omitting this argument
     * preserves the existing apply contract. Caller admits returned handles
     * while its source/raw leases remain live. */
    [[nodiscard]] SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> apply(
        const SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>&,
        std::size_t operation_byte_budget) const;
    [[nodiscard]] const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>&
    discrepancy() const noexcept
    {
        return discrepancy_;
    }
    [[nodiscard]] const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>& bound() const noexcept
    {
        return bound_;
    }
    [[nodiscard]] bool gas_is_minority() const noexcept { return gas_; }
    [[nodiscard]] std::size_t storage_bytes() const;

private:
    ExactPhaseLabelMap(bool gas, SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> a,
        SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> b,
        SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> discrepancy,
        SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> bound, std::size_t limit);
    bool gas_;
    SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> a_, b_, discrepancy_, bound_;
    std::size_t limit_;
};
} // namespace SimpleFluid::VoF::detail
