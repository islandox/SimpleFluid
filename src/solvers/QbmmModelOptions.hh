/** @file QbmmModelOptions.hh
 * @brief Physical and discretization choices for native QBMM transport.
 */
#pragma once
#include "FVM/ScalarTransportDiscretization.hh"

namespace SimpleFluid::Reacting
{
/** The implicit integrator admits shared carrier and supplied phase slip on fixed meshes and supported planar ALE.
 * KineticEuler admits fixed closed Cartesian meshes with node-dependent slip and optional body acceleration.
 * It also admits orthogonal native cylindrical meshes and polygonal disk or annulus extrusions with axial
 * node slip and axial body acceleration. Carrier velocities have Cartesian components in all three directions.
 * RealizableLinearSSPRK2 requires closed orthogonal meshes with shared carrier. Supplied acceleration is
 * supported on Cartesian meshes, native cylindrical meshes, and polygonal disk/annulus extrusions, with
 * the Q7a geometry constraints on prescribed planar ALE. The solver checks this collectively, including empty owners.
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
        NodeDependentSlip, ///< Fixed kinetic transport with one immutable law per family.
        IndependentPhaseMomentum
    };
    enum class FamilyTransport
    {
        NativeImplicit,
        KineticEuler,
        RealizableLinearSSPRK2 ///< Limited linear block reconstruction on closed orthogonal meshes.
    };
    enum class LinearSlopeLimiter
    {
        RealizabilityOnly = 0,
        TvdMc = 1 ///< Family-shared TVD-MC cap and conservative return of tiny outgoing packets.
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
    double linear_cfl = 0.5; ///< Maximum sum of outgoing carrier fractions per donor and stage, in (0, 0.5].
    int maximum_linear_substeps = 1024;
    LinearSlopeLimiter linear_slope_limiter = LinearSlopeLimiter::RealizabilityOnly;
};

inline constexpr int qbmm_linear_transport_revision = 1; ///< Legacy linear family restart revision.
inline constexpr int qbmm_linear_tvd_mc_revision = 2;     ///< Tagged linear family restart with TVD-MC.
inline constexpr int qbmm_planar_linear_transport_revision = 1; ///< Tagged moving-grid linear family restart.

[[nodiscard]] constexpr bool uses_qbmm_kinetic_transport(const QbmmModelOptions& model) noexcept
{
    return model.family_transport == QbmmModelOptions::FamilyTransport::KineticEuler;
}

[[nodiscard]] constexpr bool uses_qbmm_linear_transport(const QbmmModelOptions& model) noexcept
{
    return model.family_transport == QbmmModelOptions::FamilyTransport::RealizableLinearSSPRK2;
}

[[nodiscard]] constexpr bool uses_qbmm_explicit_transport(const QbmmModelOptions& model) noexcept
{
    return uses_qbmm_kinetic_transport(model) || uses_qbmm_linear_transport(model);
}
} // namespace SimpleFluid::Reacting
