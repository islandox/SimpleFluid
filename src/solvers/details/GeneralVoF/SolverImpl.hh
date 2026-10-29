/** @file SolverImpl.hh
 * @brief Operation dispatch and sole accepted-state publication owner.
 */
#pragma once
#include "solvers/GeneralVoFSolver.hh"
#include "solvers/details/GeneralVoF/Resources.hh"
#include "solvers/details/GeneralVoF/State.hh"
#include <utility>
namespace SimpleFluid
{
struct GeneralVoFSolver::Impl
{
    using State = detail::general_vof::State;
    using Resources = detail::general_vof::Resources;
    std::unique_ptr<Resources> resources;
    std::shared_ptr<const State> accepted;
    Impl(std::shared_ptr<const VoF::GeneralVoF>, VoFMaterial, GeneralVoFSolverOptions);
    void initialize(const std::vector<VoF::Vector>& owned_velocity, const std::vector<double>& face_velocity);
    [[nodiscard]] VoFStepDiagnostics step(double dt);
    [[nodiscard]] double stable_timestep() const;
    [[nodiscard]] GeneralVoFRebalanceResult rebalance(const VoF::GeneralVoFBalanceOptions&);
    void remap(bool migration, std::span<const int> destinations, std::span<const Meshes::AMR::AdaptiveMark> refine,
        std::span<const Meshes::AMR::AdaptiveBytes> coarsen);
};
} // namespace SimpleFluid
