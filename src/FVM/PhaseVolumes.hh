/** @file PhaseVolumes.hh
 * @brief Fixed-size finite-precision accepted phase inventories and error bounds.
 */
#pragma once
namespace SimpleFluid::VoF
{
/** Independent authoritative binary64 volumes. bulk is the native cell metric;
 * neither phase is reconstructed as bulk minus the other. */
struct PhaseVolumes
{
    double bulk = 0, gas = 0, carrier = 0;
};
/** Upward-rounded absolute phase publication/conversion error bounds. These are
 * arithmetic uncertainty, not physical sources or permission to renormalize. */
struct PhaseRoundoff
{
    double gas = 0, carrier = 0;
};
/** Fixed-size finite provenance. Signed residuals are finite diagnostic values;
 * nonnegative bounds include their conversion uncertainty and round upward.
 * No rational remainder or unrounded physical inventory is retained here. */
struct PhaseError
{
    double difference = 0, admitted_bound = 0;
    double native_conversion = 0, phase_input_bound = 0;
    double fit_residual = 0, fit_bound = 0, transferred_bound = 0, fit_allocation_bound = 0;
};
} // namespace SimpleFluid::VoF
