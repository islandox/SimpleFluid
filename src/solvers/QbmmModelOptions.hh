/** @file QbmmModelOptions.hh
 * @brief Physical and discretization choices for native QBMM transport.
 */
#pragma once
#include "FVM/ScalarTransportDiscretization.hh"

namespace SimpleFluid::Reacting
{
/** The fixed-mesh baseline admits the defaults. Selecting an unsupported model
 * rejects solver construction collectively, including on ranks without cells.
 * Native phase temperatures, prescribed drift and mesh motion have their own
 * configuration or state checks.
 */
struct QbmmModelOptions
{
    enum class Velocity
    {
        SharedCarrier,
        SuppliedPhaseSlip,
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
