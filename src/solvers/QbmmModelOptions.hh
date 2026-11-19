/** @file QbmmModelOptions.hh
 * @brief Physical and discretization choices for native QBMM transport.
 */
#pragma once
#include "FVM/ScalarTransportDiscretization.hh"

namespace SimpleFluid::Reacting
{
/** Fixed meshes and supported prescribed planar ALE admit shared carrier and supplied phase slip.
 * Unsupported models reject solver construction collectively, including on ranks without cells.
 * Native phase temperatures, prescribed drift and mesh motion have their own
 * configuration or state checks.
 */
struct QbmmModelOptions
{
    enum class Velocity
    {
        SharedCarrier, ///< Exactly zero drift on every physical phase.
        ZeroSlip = SharedCarrier,
        SuppliedPhaseSlip, ///< Immutable per-cell phase drift; all family atoms share its velocity, m/s.
        NodeDependentSlip,
        IndependentPhaseMomentum
    };
    FVM::ScalarTransportDiscretization transport;
    Velocity velocity = Velocity::SharedCarrier;
    bool variable_composition = false;
    bool cross_family_coalescence = false;
    bool bulk_gas_conversion = false;
    bool capillarity = false; ///< Laplace pressure and surface-energy closure.
    bool independent_node_temperatures = false;
};
} // namespace SimpleFluid::Reacting
