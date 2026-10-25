/** @file Diagnostics.hh
 * @brief Physical inventory and velocity diagnostics independent of solver publication.
 */
#pragma once
#include "FVM/GeneralVoF.hh"
#include "SimpleFluidExport.hh"
#include "solvers/VoFSolver.hh"
#include <span>

namespace SimpleFluid::detail::general_vof
{
template<ArithmeticMode Mode> struct SIMPLEFLUID_SOLVERS_LOCAL Diagnostics
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);
    /** Collective. Replaces physical inventory/velocity fields and retains the
     * supplied clock, projection, and other operation diagnostics. */
    [[nodiscard]] static VoFStepDiagnostics evaluate(const MeshHandle<>&, const VoF::GeneralVoF&, const VoFMaterial&,
        std::span<const long double> mass, std::span<const VoF::Vector> velocity, VoFStepDiagnostics diagnostics);
};
template<> struct SIMPLEFLUID_SOLVERS_LOCAL Diagnostics<ArithmeticMode::Exact>
{
    [[nodiscard]] static VoFStepDiagnostics evaluate(const MeshHandle<>&, const VoF::GeneralVoF&, const VoFMaterial&,
        std::span<const PLIC::GeometryMeasure<ArithmeticMode::Exact>> mass, std::span<const VoF::Vector> velocity,
        std::span<const double> density, std::size_t byte_budget, VoFStepDiagnostics diagnostics);
};
extern template struct Diagnostics<ArithmeticMode::Ordinary>;
extern template struct Diagnostics<ArithmeticMode::BoundedCertification>;
} // namespace SimpleFluid::detail::general_vof
