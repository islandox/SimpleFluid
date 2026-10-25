/** @file SolverImpl.cc
 * @brief Shared solver transaction owner and compiled numerical operations.
 */
#include "solvers/details/GeneralVoF/SolverImpl.hh"
#include "FVM/ExactFaceTransfer.hh"
#include "FVM/ExactPhaseTransfer.hh"
#include "FVM/details/FiniteGeometryAccess.hh"
#include "geometry/PLIC/PlanarFace.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/PhaseArithmetic.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "parallel/SparseOwnerExchange.hh"
#include "solvers/GeneralVoFSolver.hh"
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include "solvers/details/ConservativeVoFTransport.hh"
#include "solvers/details/GeneralVoF/AdaptedFaceTransfer.hh"
#include "solvers/details/GeneralVoF/Curvature.hh"
#include "solvers/details/GeneralVoF/Diagnostics.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoF/FaceRates.hh"
#include "solvers/details/GeneralVoF/NodeVelocity.hh"
#include "solvers/details/GeneralVoF/OwnedFaceTransfer.hh"
#include "solvers/details/GeneralVoF/PhaseStorage.hh"
#include "solvers/details/GeneralVoF/exact/Reduction.hh"
#include "solvers/details/GeneralVoFCurvature.hh"
#include "solvers/details/GeneralVoFExactStorage.hh"
#include "solvers/details/GeneralVoFFlowWorkspace.hh"
#include "solvers/details/GeneralVoFPhiClosure.hh"
#include "utils/CompensatedSum.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

// Compiled providers follow their shared declarations and support.
// clang-format off
#include "solvers/details/GeneralVoF/Support.tcc"
#include "solvers/details/GeneralVoF/exact/Operations.tcc"
#include "solvers/details/GeneralVoF/Implementation.tcc"
// clang-format on

namespace SimpleFluid
{
void GeneralVoFSolver::Impl::require(bool condition, const char* message) const
{
    detail::general_vof::Execution(*mesh).require(condition, message);
}

double GeneralVoFSolver::Impl::maximum(double value) const
{
    return detail::general_vof::Execution(*mesh).maximum(value);
}

GeneralVoFSolver::Impl::Impl(std::shared_ptr<const VoF::GeneralVoF> phases, VoFMaterial m, GeneralVoFSolverOptions o)
    : mesh(validated_phase_mesh(phases)), arithmetic_mode_(phases->arithmetic_mode()), material(m), options(o),
      flow(mesh, arithmetic_mode_, finite_spans(phases)), transport(mesh)
{
    if (arithmetic_mode_ == ArithmeticMode::BoundedCertification)
        accepted = construct_mode<ArithmeticMode::BoundedCertification>(std::move(phases), m, o);
    else
        accepted = construct_mode<ArithmeticMode::Ordinary>(std::move(phases), m, o);
}

std::vector<double> GeneralVoFSolver::Impl::import(
    const std::vector<double>& owned, const Teuchos::RCP<const Pack::map_type>& target) const
{
    return detail::general_vof::Execution(*mesh).import_owned(owned, target);
}

void GeneralVoFSolver::Impl::surface_curvature(State& state) const
{
    if (arithmetic_mode_ == ArithmeticMode::Exact)
        return surface_curvature_mode<ArithmeticMode::Exact>(state);
    if (arithmetic_mode_ == ArithmeticMode::Ordinary)
        return surface_curvature_mode<ArithmeticMode::Ordinary>(state);
    return surface_curvature_mode<ArithmeticMode::BoundedCertification>(state);
}

std::vector<VoF::GeneralVoFNodeVelocity> GeneralVoFSolver::Impl::node_velocity(const State& state) const
{
    return detail::general_vof::node_velocity(*mesh, *state.phases, state.velocity, options.byte_budget);
}

double GeneralVoFSolver::Impl::stable_timestep(
    const State& state, const std::vector<VoF::GeneralVoFNodeVelocity>& nodes) const
{
    if (arithmetic_mode_ == ArithmeticMode::BoundedCertification)
        return stable_timestep_mode<ArithmeticMode::BoundedCertification>(state, nodes);
    return stable_timestep_mode<ArithmeticMode::Ordinary>(state, nodes);
}

void GeneralVoFSolver::Impl::initialize(
    const std::vector<Vector>& owned_velocity, const std::vector<double>& face_velocity)
{
    auto next = arithmetic_mode_ == ArithmeticMode::BoundedCertification
                    ? prepare_initialize_mode<ArithmeticMode::BoundedCertification>(owned_velocity, face_velocity)
                    : prepare_initialize_mode<ArithmeticMode::Ordinary>(owned_velocity, face_velocity);
    accepted = std::move(next);
}

void GeneralVoFSolver::Impl::phase_storage(State& state) const
{
    if (arithmetic_mode_ == ArithmeticMode::BoundedCertification)
        return phase_storage_mode<ArithmeticMode::BoundedCertification>(state);
    return phase_storage_mode<ArithmeticMode::Ordinary>(state);
}

void GeneralVoFSolver::Impl::budget(State& state) const
{
    if (arithmetic_mode_ == ArithmeticMode::BoundedCertification)
        return budget_mode<ArithmeticMode::BoundedCertification>(state);
    return budget_mode<ArithmeticMode::Ordinary>(state);
}

VoFStepDiagnostics GeneralVoFSolver::Impl::step(double dt)
{
    auto next = arithmetic_mode_ == ArithmeticMode::BoundedCertification
                    ? prepare_step_mode<ArithmeticMode::BoundedCertification>(dt)
                    : prepare_step_mode<ArithmeticMode::Ordinary>(dt);
    accepted = std::move(next);
    return accepted->diagnostics;
}

std::unique_ptr<GeneralVoFSolver::Impl> GeneralVoFSolver::Impl::remap(bool migration, std::span<const int> destinations,
    std::span<const Meshes::AMR::AdaptiveMark> refine, std::span<const Meshes::AMR::AdaptiveBytes> coarsen) const
{
    auto prepared =
        arithmetic_mode_ == ArithmeticMode::BoundedCertification
            ? prepare_remap_mode<ArithmeticMode::BoundedCertification>(migration, destinations, refine, coarsen)
            : prepare_remap_mode<ArithmeticMode::Ordinary>(migration, destinations, refine, coarsen);
    prepared.resources->accepted = std::move(prepared.state);
    return std::move(prepared.resources);
}
} // namespace SimpleFluid
