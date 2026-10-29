/** @file SolverImpl.cc
 * @brief Compiled general VoF SolverImpl operation.
 */
#include "solvers/details/GeneralVoF/SolverImpl.hh"
#include "solvers/details/GeneralVoF/Advance.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoF/Initialization.hh"
#include "solvers/details/GeneralVoF/NodeVelocity.hh"
#include "solvers/details/GeneralVoF/OwnedMigration.hh"
#include "solvers/details/GeneralVoF/Remap.hh"
#include <bit>
#include <chrono>
#include <cmath>
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
    const bool face_backend =
        accepted->phases->options().transport_backend != VoF::GeneralVoFTransportBackend::SweptPartition;
    const bool homogeneous = !face_backend && accepted->phases->homogeneous_transport_eligible();
    const auto nodes = face_backend || homogeneous ? std::vector<VoF::GeneralVoFNodeVelocity>{}
                                                   : node_velocity(*resources->mesh, *accepted->phases,
                                                         accepted->velocity, resources->options.byte_budget);
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
GeneralVoFRebalanceResult GeneralVoFSolver::Impl::rebalance(const VoF::GeneralVoFBalanceOptions& options)
{
    const auto pinned = accepted;
    const Execution execution(*resources->mesh);
    const auto start = std::chrono::steady_clock::now();
    execution.require(resources->arithmetic_mode == ArithmeticMode::Ordinary && bool(pinned->prepared_transport),
        "Live rebalance requires an accepted Ordinary step with captured prepared work.");
    const auto ledger = pinned->prepared_transport;
    execution.collective(
        [&]
        {
            const auto source_time = std::bit_cast<double>(ledger->time_stage);
            if (!std::isfinite(source_time) || !std::isfinite(ledger->timestep) || !(ledger->timestep > 0) ||
                std::bit_cast<std::uint64_t>(source_time + ledger->timestep) !=
                    std::bit_cast<std::uint64_t>(pinned->diagnostics.time) ||
                ledger->density !=
                    std::array<double, 2>{resources->material.gas_density, resources->material.liquid_density} ||
                !ledger->balance_work || !pinned->phases->accepted_unchanged())
                throw std::runtime_error("Live rebalance has stale clock, density, geometry or incomplete capture.");
        });
    pinned->phases->validate_prepared_transport_candidate(*ledger);
    GeneralVoFRebalanceResult result;
    result.proposal = VoF::plan_partition_proposal(
        pinned->phases, *ledger, VoF::GeneralVoFBalanceBinding::AcceptedSuccessor, options);
    result.proposal_seconds =
        execution.maximum(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    execution.require(accepted == pinned && pinned->prepared_transport == ledger &&
                          result.proposal.current_phase_identity == pinned->phases->accepted_identity() &&
                          result.proposal.current_phase_control == pinned->phases->accepted_control(),
        "Live rebalance accepted state changed during proposal.");
    if (!result.proposal.accepted)
        return result;
    const auto migration_start = std::chrono::steady_clock::now();
    const auto hierarchy = pinned->phases->hierarchy();
    const auto plan = hierarchy->plan_migration(result.proposal.native_plan);
    result.local_native_application_bytes = plan.actual_migration_application_bytes();
    Teuchos::reduceAll(*hierarchy->communicator(), Teuchos::REDUCE_SUM, 1, &result.local_native_application_bytes,
        &result.global_native_application_bytes);
    execution.require(result.local_native_application_bytes == result.proposal.local_native_application_bytes &&
                          result.global_native_application_bytes == result.proposal.global_native_application_bytes,
        "Live rebalance native application differs from qualified bytes.");
    auto prepared = prepare_owned_migration(*resources, *pinned, plan);
    pinned->phases->validate_prepared_transport_candidate(*ledger);
    execution.require(accepted == pinned && pinned->prepared_transport == ledger,
        "Live rebalance accepted state changed before publication.");
    result.migration_seconds =
        execution.maximum(std::chrono::duration<double>(std::chrono::steady_clock::now() - migration_start).count());
    std::shared_ptr<const State> next = std::move(prepared.state);
    resources.swap(prepared.resources);
    accepted.swap(next);
    result.applied = true;
    return result;
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
    if (prepared.unchanged)
        return;
    // Preparation has completed every collective/source validation. Publication
    // performs no allocation; the detached records retain the old resources/state.
    std::shared_ptr<const State> next = std::move(prepared.state);
    resources.swap(prepared.resources);
    accepted.swap(next);
}

} // namespace SimpleFluid
