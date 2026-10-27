/** @file PhaseArithmetic.hh
 * @brief Finite phase publication and outward conversion-bound declarations.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/swept_volume/SweptGeometry.hh"
namespace SimpleFluid::VoF::detail
{
struct RoundedPhaseValue
{
    double value = 0, absolute_error = 0;
};
/** One-step conversion only: no source remainder is cached or carried forward. */
[[nodiscard]] SIMPLEFLUID_FVM_EXPORT RoundedPhaseValue round_phase_residual(
    const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>&);
[[nodiscard]] SIMPLEFLUID_FVM_EXPORT RoundedPhaseValue round_phase_value(
    const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>&);
/** Nonnegative exact transient bound -> finite upward binary64 enclosure. */
[[nodiscard]] SIMPLEFLUID_FVM_EXPORT double phase_upper_bound(
    const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>&);
/** Signed finite diagnostic; callers include its conversion in admitted bounds. */
[[nodiscard]] SIMPLEFLUID_FVM_EXPORT double phase_signed_value(
    const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>&);
} // namespace SimpleFluid::VoF::detail
