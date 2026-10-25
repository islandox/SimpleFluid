/** @file GeneralVoFProjectionArithmetic.hh
 * @brief Operation-derived arithmetic intervals for represented projection fluxes.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/PLIC/SweptGeometry.hh"
namespace SimpleFluid::detail
{
/** Arithmetic uncertainty only. Represented geometry/operator/gradient inputs
 * are exact leaves; discretization and Krylov truncation errors are excluded. */
struct GeneralVoFRoundedValue
{
    double value = 0;
    SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> lower, upper;
    std::uint64_t operations = 0;
};
/** Free inner pressure normalization. For positive maximum m, choose a
 * positive dyadic s with m < s <= 2*m at m's own exponent; zero selects one.
 * This changes neither the fixed physical reference nor an arithmetic radius.
 * Caller supplies the same16*K operation workspace used by scalar arithmetic. */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>
general_vof_pressure_normalization_scale(
    const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>& maximum, std::size_t result_cap);
/** Fixed original max-divergence threshold from the initial exact PRIMARY
 * residual. Caller provides the separate16*K operation workspace lease. */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>
general_vof_projection_divergence_limit(
    const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>& initial_maximum, double dt,
    double absolute_tolerance, double relative_tolerance, double divergence_tolerance, std::size_t result_cap);
class SIMPLEFLUID_SOLVERS_LOCAL GeneralVoFProjectionArithmetic
{
public:
    explicit GeneralVoFProjectionArithmetic(std::size_t byte_budget) : budget_(byte_budget) {}
    [[nodiscard]] GeneralVoFRoundedValue leaf(double) const;
    [[nodiscard]] GeneralVoFRoundedValue add(const GeneralVoFRoundedValue&, const GeneralVoFRoundedValue&) const;
    [[nodiscard]] GeneralVoFRoundedValue subtract(const GeneralVoFRoundedValue&, const GeneralVoFRoundedValue&) const;
    [[nodiscard]] GeneralVoFRoundedValue multiply(const GeneralVoFRoundedValue&, const GeneralVoFRoundedValue&) const;
    [[nodiscard]] GeneralVoFRoundedValue divide(const GeneralVoFRoundedValue&, const GeneralVoFRoundedValue&) const;
    [[nodiscard]] SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> radius(
        const GeneralVoFRoundedValue&) const;

private:
    std::size_t budget_;
    [[nodiscard]] GeneralVoFRoundedValue apply(
        const GeneralVoFRoundedValue&, const GeneralVoFRoundedValue&, unsigned operation) const;
};
} // namespace SimpleFluid::detail
