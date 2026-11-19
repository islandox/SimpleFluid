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
        NodeDependentSlip, ///< Fixed Cartesian kinetic transport with one immutable law per family.
        IndependentPhaseMomentum
    };
    enum class FamilyTransport
    {
        NativeImplicit,
        KineticEuler,
        RealizableLinearSSPRK2 ///< Reserved for Q7c. The native solver rejects this selection.
    };
    FVM::ScalarTransportDiscretization transport;
    Velocity velocity = Velocity::SharedCarrier;
    bool variable_composition = false;
    bool cross_family_coalescence = false;
    bool bulk_gas_conversion = false;
    bool capillarity = false; ///< Laplace pressure and surface-energy closure.
    bool independent_node_temperatures = false;
    FamilyTransport family_transport = FamilyTransport::NativeImplicit;
    double kinetic_cfl = 1.; ///< Maximum sum of all outgoing fractions per active atom and substep, in (0, 1].
    int maximum_kinetic_substeps = 1024;
};

[[nodiscard]] constexpr bool uses_qbmm_kinetic_transport(const QbmmModelOptions& model) noexcept
{
    return model.family_transport == QbmmModelOptions::FamilyTransport::KineticEuler;
}
} // namespace SimpleFluid::Reacting
