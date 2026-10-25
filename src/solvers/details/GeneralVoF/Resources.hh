/** @file Resources.hh
 * @brief Mesh-bound operators and immutable controls, without accepted-state ownership.
 */
#pragma once
#include "solvers/GeneralVoFSolver.hh"
#include "solvers/details/ConservativeVoFTransport.hh"
#include "solvers/details/GeneralVoFFlowWorkspace.hh"
namespace SimpleFluid::detail::general_vof
{
struct SIMPLEFLUID_SOLVERS_LOCAL Resources
{
    std::shared_ptr<const MeshHandle<>> mesh;
    ArithmeticMode arithmetic_mode;
    VoFMaterial material;
    GeneralVoFSolverOptions options;
    GeneralVoFFlowWorkspace flow;
    ConservativeVoFTransport transport;
    std::vector<double> volume, thickness;
    Resources(std::shared_ptr<const VoF::GeneralVoF>, VoFMaterial, GeneralVoFSolverOptions);
};
} // namespace SimpleFluid::detail::general_vof
