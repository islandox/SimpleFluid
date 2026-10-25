/** @file SolverImpl.cc
 * @brief Compiled general VoF SolverImpl operation.
 */
#include "solvers/details/GeneralVoF/SolverImpl.hh"
#include "solvers/details/GeneralVoF/Advance.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoF/Initialization.hh"
#include "solvers/details/GeneralVoF/NodeVelocity.hh"
#include "solvers/details/GeneralVoF/Remap.hh"
#include <stdexcept>

namespace SimpleFluid
{
using Pack = DefaultTpetraTypes;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;
using Vector = VoF::Vector;
using namespace detail::general_vof;
GeneralVoFSolver::Impl::Impl(
    std::shared_ptr<const VoF::GeneralVoF> phases, VoFMaterial material, GeneralVoFSolverOptions options)
    : resources(std::make_unique<Resources>(phases, material, options))
{
    auto state = std::make_shared<State>();
    state->phases = std::move(phases);
    accepted = std::move(state);
}
void GeneralVoFSolver::Impl::initialize(
    const std::vector<VoF::Vector>& owned_velocity, const std::vector<double>& face_velocity)
{
    std::shared_ptr<State> next;
    switch (resources->arithmetic_mode)
    {
        case ArithmeticMode::Ordinary:
            next = Initialization<ArithmeticMode::Ordinary>::prepare(
                *resources, accepted->phases, owned_velocity, face_velocity);
            break;
        case ArithmeticMode::BoundedCertification:
            next = Initialization<ArithmeticMode::BoundedCertification>::prepare(
                *resources, accepted->phases, owned_velocity, face_velocity);
            break;
        case ArithmeticMode::Exact:
            next = Initialization<ArithmeticMode::Exact>::prepare(
                *resources, accepted->phases, owned_velocity, face_velocity);
            break;
        default:
            throw std::invalid_argument("Unknown general VoF arithmetic mode.");
    }
    accepted = std::move(next);
}
VoFStepDiagnostics GeneralVoFSolver::Impl::step(double dt)
{
    const Execution execution(*resources->mesh);
    execution.require(accepted->phases->arithmetic_mode() == resources->arithmetic_mode,
        "General VoF accepted state changed arithmetic mode.");
    std::shared_ptr<State> next;
    switch (resources->arithmetic_mode)
    {
        case ArithmeticMode::Ordinary:
            next = Advance<ArithmeticMode::Ordinary>::prepare(*resources, *accepted, dt);
            break;
        case ArithmeticMode::BoundedCertification:
            next = Advance<ArithmeticMode::BoundedCertification>::prepare(*resources, *accepted, dt);
            break;
        case ArithmeticMode::Exact:
            next = Advance<ArithmeticMode::Exact>::prepare(*resources, *accepted, dt);
            break;
        default:
            throw std::invalid_argument("Unknown general VoF arithmetic mode.");
    }
    accepted = std::move(next);
    return accepted->diagnostics;
}
double GeneralVoFSolver::Impl::stable_timestep() const
{
    const auto nodes =
        node_velocity(*resources->mesh, *accepted->phases, accepted->velocity, resources->options.byte_budget);
    switch (resources->arithmetic_mode)
    {
        case ArithmeticMode::Ordinary:
            return Advance<ArithmeticMode::Ordinary>::stable_timestep(*resources, *accepted, nodes);
        case ArithmeticMode::BoundedCertification:
            return Advance<ArithmeticMode::BoundedCertification>::stable_timestep(*resources, *accepted, nodes);
        case ArithmeticMode::Exact:
            return Advance<ArithmeticMode::Exact>::stable_timestep(*resources, *accepted, nodes);
    }
    throw std::invalid_argument("Unknown general VoF arithmetic mode.");
}
void GeneralVoFSolver::Impl::remap(bool migration, std::span<const int> destinations,
    std::span<const Meshes::AMR::AdaptiveMark> refine, std::span<const Meshes::AMR::AdaptiveBytes> coarsen)
{
    PreparedRemap prepared;
    switch (resources->arithmetic_mode)
    {
        case ArithmeticMode::Ordinary:
            prepared = Remap<ArithmeticMode::Ordinary>::prepare(
                *resources, *accepted, migration, destinations, refine, coarsen);
            break;
        case ArithmeticMode::BoundedCertification:
            prepared = Remap<ArithmeticMode::BoundedCertification>::prepare(
                *resources, *accepted, migration, destinations, refine, coarsen);
            break;
        case ArithmeticMode::Exact:
            prepared =
                Remap<ArithmeticMode::Exact>::prepare(*resources, *accepted, migration, destinations, refine, coarsen);
            break;
        default:
            throw std::invalid_argument("Unknown general VoF arithmetic mode.");
    }
    // Preparation has completed every collective/source validation. Publication
    // performs no allocation; the detached records retain the old resources/state.
    std::shared_ptr<const State> next = std::move(prepared.state);
    resources.swap(prepared.resources);
    accepted.swap(next);
}

} // namespace SimpleFluid
