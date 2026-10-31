/** @file StateTrace.hh
 * @brief Borrowed finite advancement diagnostics for optional rank-local traces.
 */
#pragma once
#include "solvers/GeneralVoFSolver.hh"
#include <span>

namespace SimpleFluid::detail
{
class ConservativeVoFTransport;
struct GeneralVoFFiniteProjectionProvenance;
namespace general_vof
{
struct StateTraceView
{
    const VoF::GeneralVoF* phases;
    std::span<const VoF::Vector> velocity;
    std::span<const double> face_velocity, pressure;
    std::span<const long double> finite_mass;
    const GeneralVoFFiniteProjectionProvenance& finite_projection;
    const VoFStepDiagnostics& diagnostics;
    std::span<const GeneralVoFMomentumDiagnostics, 3> momentum_transport;
};
/** No collectives; derived diagnostics are evaluated only when flow tracing is enabled. */
SIMPLEFLUID_SOLVERS_LOCAL void trace_finite_state(const MeshHandle<>& mesh, const char* stage, double accepted_time,
    double dt, std::span<const VoF::Vector> accepted_velocity, const ConservativeVoFTransport& transport,
    const StateTraceView& state);
} // namespace general_vof
} // namespace SimpleFluid::detail
