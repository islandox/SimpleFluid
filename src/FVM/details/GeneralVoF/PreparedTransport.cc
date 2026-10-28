/** @file PreparedTransport.cc
 * @brief Detached finite phase-ledger preparation for both transport families.
 */
#include "FVM/details/GeneralVoFInternal.hh"
#include "geometry/PLIC/GeometryError.hh"
namespace SimpleFluid::VoF
{
namespace
{
template<class Reserve> void reserve_key_storage(const GeneralVoF::Key& key, const Reserve& reserve)
{
    std::visit(
        [&](const auto& ancestry)
        {
            using Ancestry = std::decay_t<decltype(ancestry)>;
            if constexpr (std::is_same_v<Ancestry, AMR::ConvexCellAncestry>)
                reserve(ancestry.path.capacity(), sizeof(std::uint64_t));
            else if constexpr (std::is_same_v<Ancestry, AMR::ProductCellAncestry>)
            {
                reserve(ancestry.base.path.capacity(), sizeof(std::uint64_t));
                reserve(ancestry.axial.path.capacity(), sizeof(std::uint64_t));
            }
        },
        key.ancestry);
}
template<ArithmeticMode Mode>
void capacities(const GeneralVoF::Data& source, const GeneralVoF::Data& candidate, GeneralVoFPreparedTransport& result)
{
    using A = fp::Arithmetic<Mode>;
    const auto mass = [&](const PhaseVolumes& p)
    {
        return A::add(A::multiply(A::make_value(p.gas, 0), A::make_value(result.density[0], 0)),
            A::multiply(A::make_value(p.carrier, 0), A::make_value(result.density[1], 0)));
    };
    result.old_mass.reserve(source.phases.size());
    result.new_mass.reserve(source.phases.size());
    result.old_mass_error.reserve(source.phases.size());
    result.new_mass_error.reserve(source.phases.size());
    result.raw_new_mass.reserve(source.phases.size());
    result.raw_new_mass_error.reserve(source.phases.size());
    result.cleanup_mass.assign(source.phases.size(), 0);
    std::map<std::uint64_t, const GeneralVoFPhaseCleanupDelta*> cleanup;
    for (const auto& delta : result.phase_cleanup)
        if (!cleanup.emplace(delta.gid, &delta).second)
            throw std::invalid_argument("Prepared VoF cleanup has duplicate owned cells.");
    for (std::size_t c = 0; c < source.phases.size(); ++c)
    {
        const auto old = mass(source.phases[c]), next = mass(candidate.phases[c]);
        auto raw = next;
        const auto gid = source.topology->native.mesh->owned_cell_map()->getGlobalElement(c);
        if (const auto found = cleanup.find(gid); found != cleanup.end())
        {
            const auto& delta = *found->second;
            if (delta.key != source.topology->order[c] || delta.after[0] != candidate.phases[c].gas ||
                delta.after[1] != candidate.phases[c].carrier ||
                delta.volume[0] != static_cast<long double>(delta.after[0]) - delta.before[0] ||
                delta.volume[1] != static_cast<long double>(delta.after[1]) - delta.before[1])
                throw std::invalid_argument("Prepared VoF cleanup differs from its published phases.");
            raw = mass(PhaseVolumes{candidate.phases[c].bulk, delta.before[0], delta.before[1]});
            result.cleanup_mass[c] =
                A::add(A::multiply(A::make_value(delta.volume[0], 0), A::make_value(result.density[0], 0)),
                    A::multiply(A::make_value(delta.volume[1], 0), A::make_value(result.density[1], 0)))
                    .value;
            cleanup.erase(found);
        }
        result.old_mass.push_back(old.value);
        result.new_mass.push_back(next.value);
        result.old_mass_error.push_back(old.absolute_error);
        result.new_mass_error.push_back(next.absolute_error);
        result.raw_new_mass.push_back(raw.value);
        result.raw_new_mass_error.push_back(raw.absolute_error);
    }
    if (!cleanup.empty())
        throw std::invalid_argument("Prepared VoF cleanup contains foreign owned cells.");
}
} // namespace
void GeneralVoFPreparedTransport::validate_byte_budget(std::size_t remaining) const
{
    namespace failure = ::SimpleFluid::detail::vof_failure;
    failure::Scope context("phase_preparation_budget");
    context.record.byte_budget = remaining;
    context.record.mark(failure::Field::ByteBudget);
    const auto reserve = [&](std::size_t count, std::size_t width)
    {
        context.record.bytes_used = context.record.byte_budget - remaining;
        context.record.mark(failure::Field::BytesUsed);
        if (count > remaining / width)
            throw PLIC::GeometryError(
                PLIC::GeometryErrorCode::ResourceLimit, "Prepared VoF transport exceeds its retained byte budget.");
        remaining -= count * width;
    };
    reserve(1, sizeof(*this));
    reserve(streams.capacity(), sizeof(GeneralVoFDirectedStream));
    reserve(signed_face_contributions.capacity(), sizeof(GeneralVoFFiniteFaceContribution));
    reserve(positive_transfers.capacity(), sizeof(GeneralVoFFiniteTransfer));
    reserve(donor_budgets.capacity(), sizeof(GeneralVoFFiniteDonorBudget));
    reserve(signed_receipts.capacity(), sizeof(GeneralVoFSignedReceipt));
    reserve(signed_donor_budgets.capacity(), sizeof(GeneralVoFSignedDonorBudget));
    reserve(numerical_support_transfers.capacity(), sizeof(GeneralVoFFiniteTransfer));
    reserve(numerical_support_budgets.capacity(), sizeof(GeneralVoFFiniteDonorBudget));
    reserve(face_chain_diagnostics.capacity(), sizeof(GeneralVoFFaceChainDiagnostic));
    reserve(cell_closure_diagnostics.capacity(), sizeof(GeneralVoFCellClosureDiagnostic));
    for (const auto& closure : cell_closure_diagnostics)
        reserve_key_storage(closure.key, reserve);
    reserve(phase_cleanup.capacity(), sizeof(GeneralVoFPhaseCleanupDelta));
    for (const auto& delta : phase_cleanup)
        reserve_key_storage(delta.key, reserve);
    reserve(cleanup_momentum.capacity(), sizeof(std::array<long double, 3>));
    reserve(node_trajectories.capacity(), sizeof(GeneralVoFFiniteNodeTrajectory));
    for (const auto& trace : node_trajectories)
    {
        reserve(PLIC::dynamic_point_storage_bytes(trace.point) - sizeof(PLIC::DynamicPoint), 1);
        reserve(PLIC::dynamic_point_storage_bytes(trace.departure) - sizeof(PLIC::DynamicPoint), 1);
    }
    for (const auto* history : {&phase_roundoff, &old_phase_history, &new_phase_history})
        reserve(history->capacity(), sizeof(PhaseRoundoff));
    for (const auto* mass :
        {&old_mass, &new_mass, &old_mass_error, &new_mass_error, &raw_new_mass, &raw_new_mass_error, &cleanup_mass})
        reserve(mass->capacity(), sizeof(long double));
}
void GeneralVoF::Trial::validate_prepared_transport(
    const GeneralVoFPreparedTransport& prepared, std::array<double, 2> density, std::uint64_t time_stage) const
{
    if (!source_)
        throw std::invalid_argument("General VoF trial has no source.");
    agree_transaction(source_->hierarchy->communicator(), transaction_);
    GeneralVoF(source_).validate_prepared_transport_source(prepared);
    wire::collective(*source_->hierarchy->communicator(),
        [&]
        {
            if (prepared.candidate_identity != candidate_ || prepared.transaction != transaction_ ||
                prepared.density != density || prepared.time_stage != time_stage || prepared.timestep != timestep_ ||
                prepared.old_mass.size() != source_->phases.size() ||
                prepared.new_mass.size() != source_->phases.size() ||
                prepared.old_mass_error.size() != source_->phases.size() ||
                prepared.new_mass_error.size() != source_->phases.size() ||
                prepared.raw_new_mass.size() != source_->phases.size() ||
                prepared.raw_new_mass_error.size() != source_->phases.size() ||
                prepared.cleanup_mass.size() != source_->phases.size() ||
                prepared.family != (interface_local_ ? GeneralVoFTransportFamily::InterfaceLocalSwept
                                       : homogeneous_algebraic_ || source_->options.transport_backend !=
                                                                       GeneralVoFTransportBackend::SweptPartition
                                           ? GeneralVoFTransportFamily::AlgebraicFaceVolume
                                           : GeneralVoFTransportFamily::GeometricSweptTetrahedra))
                throw std::invalid_argument("Prepared VoF transport trial or caller context differs.");
        });
}
void GeneralVoF::validate_prepared_transport_source(const GeneralVoFPreparedTransport& prepared) const
{
    wire::collective(*data_->hierarchy->communicator(),
        [&]
        {
            current(*data_);
            if (prepared.source_identity != data_ || prepared.source_control != data_->control ||
                prepared.mesh_identity != data_->hierarchy || prepared.topology_identity != data_->topology ||
                prepared.mesh_generation != data_->hierarchy->partition_generation() ||
                prepared.mesh_topology_generation != data_->hierarchy->topology_generation() ||
                prepared.arithmetic_mode != data_->options.arithmetic_mode ||
                prepared.backend != data_->options.transport_backend || prepared.authenticated_source_ != data_ ||
                prepared.authenticated_candidate_ != prepared.candidate_identity ||
                prepared.authenticated_transaction_ != prepared.transaction ||
                prepared.authenticated_source_control_ != prepared.source_control ||
                prepared.authenticated_time_stage_ != prepared.time_stage ||
                prepared.authenticated_timestep_ != prepared.timestep ||
                prepared.authenticated_density_ != prepared.density ||
                prepared.authenticated_mesh_generation_ != prepared.mesh_generation ||
                prepared.authenticated_mesh_topology_generation_ != prepared.mesh_topology_generation ||
                prepared.authenticated_family_ != prepared.family ||
                (prepared.family == GeneralVoFTransportFamily::AlgebraicFaceVolume &&
                    prepared.arithmetic_mode != ArithmeticMode::Ordinary) ||
                (prepared.family == GeneralVoFTransportFamily::InterfaceLocalSwept &&
                    prepared.arithmetic_mode != ArithmeticMode::Ordinary) ||
                ((prepared.family == GeneralVoFTransportFamily::GeometricSweptTetrahedra ||
                     prepared.family == GeneralVoFTransportFamily::InterfaceLocalSwept) &&
                    prepared.backend != GeneralVoFTransportBackend::SweptPartition))
                throw std::invalid_argument("Prepared VoF transport belongs to another source or mesh publication.");
        });
    agree_transaction(data_->hierarchy->communicator(), data_->control);
}
void GeneralVoF::Trial::validate_prepared_transport_mode(
    ArithmeticMode mode, const std::shared_ptr<const GeneralVoFPreparedTransport>* retained) const
{
    if (!source_)
        throw std::invalid_argument("General VoF trial has no source.");
    wire::collective(*source_->hierarchy->communicator(),
        [&]
        {
            current(*source_);
            if ((retained && !*retained) || mode == ArithmeticMode::Exact || mode != source_->options.arithmetic_mode)
                throw std::invalid_argument("Prepared VoF trial view is empty or its arithmetic mode differs.");
        });
}
std::shared_ptr<const GeneralVoFPreparedTransport> GeneralVoF::Trial::prepare_transport_checked(
    ArithmeticMode mode, std::array<double, 2> density, std::uint64_t time_stage) const
{
    validate_prepared_transport_mode(mode);
    const auto prepared = prepare_transport(density, time_stage);
    validate_prepared_transport(*prepared, density, time_stage);
    return prepared;
}
std::shared_ptr<const GeneralVoFPreparedTransport> GeneralVoF::Trial::prepare_transport(
    std::array<double, 2> density, std::uint64_t time_stage) const
{
    if (!source_)
        throw std::invalid_argument("General VoF trial has no source.");
    const auto comm = source_->hierarchy->communicator();
    namespace failure = ::SimpleFluid::detail::vof_failure;
    const auto safe = [&](auto&& action)
    {
        wire::collective(*comm,
            [&]
            {
                failure::Scope context("phase_preparation");
                context.record.arithmetic_mode = source_->options.arithmetic_mode;
                context.record.mark(failure::Field::Arithmetic);
                context.record.byte_budget = source_->options.byte_budget;
                context.record.mark(failure::Field::ByteBudget);
                action();
            });
    };
    agree_transaction(comm, transaction_);
    safe(
        [&]
        {
            current(*source_);
            if (!candidate_ || source_->options.arithmetic_mode == ArithmeticMode::Exact ||
                candidate_->options.arithmetic_mode != source_->options.arithmetic_mode || !std::isfinite(timestep_) ||
                timestep_ < 0 || !std::isfinite(density[0]) || density[0] <= 0 || !std::isfinite(density[1]) ||
                density[1] <= 0 || candidate_->phases.size() != source_->phases.size())
                throw std::invalid_argument(
                    "Prepared VoF transport requires a valid finite trial and positive densities.");
            if (prepared_ && (prepared_->density != density || prepared_->time_stage != time_stage ||
                                 prepared_->source_identity != source_ || prepared_->transaction != transaction_ ||
                                 prepared_->timestep != timestep_))
                throw std::invalid_argument("Prepared VoF transport density, stage or trial identity differs.");
        });
    const std::array<double, 3> controls{timestep_, density[0], density[1]};
    std::array<double, 3> low{}, high{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 3, controls.data(), low.data());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 3, controls.data(), high.data());
    std::uint64_t first_stage = 0, last_stage = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &time_stage, &first_stage);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &time_stage, &last_stage);
    safe(
        [&]
        {
            if (low != high || first_stage != last_stage)
                throw std::invalid_argument("Prepared VoF transport controls differ across ranks.");
        });
    if (prepared_ && prepared_->mesh_identity)
    {
        validate_prepared_transport(*prepared_, density, time_stage);
        return prepared_;
    }
    std::shared_ptr<GeneralVoFPreparedTransport> result;
    safe(
        [&]
        {
            std::size_t remaining = source_->options.byte_budget;
            const auto reserve = [&](std::size_t count, std::size_t width)
            {
                if (count > remaining / width)
                    throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit,
                        "Prepared VoF transport exceeds its retained byte budget.");
                remaining -= count * width;
            };
            reserve(1, sizeof(GeneralVoFPreparedTransport));
            reserve(prepared_ ? prepared_->streams.size() : unbound_streams_.size(), sizeof(GeneralVoFDirectedStream));
            reserve(finite_faces_.size(), sizeof(GeneralVoFFiniteFaceContribution));
            reserve(finite_transfers_.size(), sizeof(GeneralVoFFiniteTransfer));
            reserve(finite_budgets_.size(), sizeof(GeneralVoFFiniteDonorBudget));
            reserve(signed_receipts_.size(), sizeof(GeneralVoFSignedReceipt));
            reserve(signed_budgets_.size(), sizeof(GeneralVoFSignedDonorBudget));
            reserve(numerical_support_transfers_.size(), sizeof(GeneralVoFFiniteTransfer));
            reserve(numerical_support_budgets_.size(), sizeof(GeneralVoFFiniteDonorBudget));
            reserve(finite_trajectories_.size(), sizeof(GeneralVoFFiniteNodeTrajectory));
            reserve(finite_face_chain_.size(), sizeof(GeneralVoFFaceChainDiagnostic));
            reserve(finite_cell_closure_.size(), sizeof(GeneralVoFCellClosureDiagnostic));
            for (const auto& closure : finite_cell_closure_)
                reserve_key_storage(closure.key, reserve);
            reserve(roundoff_.size() + source_->phase_history.size() + candidate_->phase_history.size(),
                sizeof(PhaseRoundoff));
            reserve(source_->phases.size(), 7 * sizeof(long double));
            if (candidate_ != source_)
            {
                reserve(candidate_->latest_phase_cleanup.size(), sizeof(GeneralVoFPhaseCleanupDelta));
                for (const auto& delta : candidate_->latest_phase_cleanup)
                    reserve_key_storage(delta.key, reserve);
            }
            result = std::make_shared<GeneralVoFPreparedTransport>();
            result->source_identity = source_;
            result->candidate_identity = candidate_;
            result->mesh_identity = source_->hierarchy;
            result->topology_identity = source_->topology;
            result->source_control = source_->control;
            result->transaction = transaction_;
            result->mesh_generation = source_->hierarchy->partition_generation();
            result->mesh_topology_generation = source_->hierarchy->topology_generation();
            result->time_stage = time_stage;
            result->timestep = timestep_;
            result->density = density;
            result->arithmetic_mode = source_->options.arithmetic_mode;
            result->backend = source_->options.transport_backend;
            result->family = interface_local_ ? GeneralVoFTransportFamily::InterfaceLocalSwept
                             : homogeneous_algebraic_ || result->backend != GeneralVoFTransportBackend::SweptPartition
                                 ? GeneralVoFTransportFamily::AlgebraicFaceVolume
                                 : GeneralVoFTransportFamily::GeometricSweptTetrahedra;
            result->authenticated_source_ = source_;
            result->authenticated_candidate_ = candidate_;
            result->authenticated_transaction_ = transaction_;
            result->authenticated_source_control_ = result->source_control;
            result->authenticated_time_stage_ = time_stage;
            result->authenticated_timestep_ = timestep_;
            result->authenticated_density_ = density;
            result->authenticated_mesh_generation_ = result->mesh_generation;
            result->authenticated_mesh_topology_generation_ = result->mesh_topology_generation;
            result->authenticated_family_ = result->family;
            if (prepared_)
            {
                result->streams = prepared_->streams;
                result->diagnostics = prepared_->diagnostics;
            }
            else if (homogeneous_algebraic_ || interface_local_)
            {
                result->streams = unbound_streams_;
                result->diagnostics = unbound_diagnostics_;
                for (auto& stream : result->streams)
                    stream.mass = density[0] * stream.gas + density[1] * stream.liquid;
            }
            result->signed_face_contributions = finite_faces_;
            result->positive_transfers = finite_transfers_;
            result->signed_receipts = signed_receipts_;
            result->signed_donor_budgets = signed_budgets_;
            result->numerical_support_transfers = numerical_support_transfers_;
            result->numerical_support_budgets = numerical_support_budgets_;
            result->donor_budgets = finite_budgets_;
            result->node_trajectories = finite_trajectories_;
            result->face_chain_diagnostics = finite_face_chain_;
            result->cell_closure_diagnostics = finite_cell_closure_;
            result->phase_roundoff = roundoff_;
            result->old_phase_history = source_->phase_history;
            result->new_phase_history = candidate_->phase_history;
            // A no-op trial can retain the source publication, whose latest
            // cleanup belongs to an earlier step rather than this transport.
            if (candidate_ != source_)
                result->phase_cleanup = candidate_->latest_phase_cleanup;
            result->old_phase_cleanup = source_->phase_cleanup;
            result->new_phase_cleanup = candidate_->phase_cleanup;
            if (result->arithmetic_mode == ArithmeticMode::Ordinary)
                capacities<ArithmeticMode::Ordinary>(*source_, *candidate_, *result);
            else
                capacities<ArithmeticMode::BoundedCertification>(*source_, *candidate_, *result);
            result->validate_byte_budget(source_->options.byte_budget);
            current(*source_);
        });
    return result;
}
} // namespace SimpleFluid::VoF
