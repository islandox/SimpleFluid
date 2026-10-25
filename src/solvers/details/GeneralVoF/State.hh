/** @file State.hh
 * @brief Private accepted and candidate records for general VoF flow.
 */
#pragma once
#include "solvers/GeneralVoFSolver.hh"
#include "solvers/details/GeneralVoFFlowWorkspace.hh"
#include <array>
#include <memory>
#include <vector>

namespace SimpleFluid::detail::general_vof
{
struct SIMPLEFLUID_SOLVERS_LOCAL State
{
    std::shared_ptr<const VoF::GeneralVoF> phases;
    std::vector<double> fraction, density, face_velocity, pressure, curvature;
    std::vector<VoF::Vector> velocity;
    // Derived from the source-selected authority. Finite mode always
    // rebuilds from represented phases; experimental exact mode opts into
    // its separately retained authority, never a solver-side reinterpretation.
    std::vector<SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>> physical_mass;
    std::vector<long double> finite_mass, finite_mass_error;
    detail::GeneralVoFFiniteProjectionProvenance finite_projection;
    detail::GeneralVoFProjectionProvenance projection_arithmetic;
    VoFStepDiagnostics diagnostics;
    std::array<GeneralVoFMomentumDiagnostics, 3> momentum_transport;
};
} // namespace SimpleFluid::detail::general_vof
