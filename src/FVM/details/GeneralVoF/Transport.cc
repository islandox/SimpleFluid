/** @file Transport.cc
 * @brief Operation dispatch and trial publication validation.
 */
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF
{
std::shared_ptr<const GeneralVoFTransportControls> GeneralVoF::make_transport_controls(ArithmeticMode mode,
    std::span<const GeneralVoFNodeVelocity> nodes, std::span<const GeneralVoFFaceVolume> faces, double timestep) const
{
    std::shared_ptr<GeneralVoFTransportControls> result;
    wire::collective(*data_->hierarchy->communicator(),
        [&]
        {
            namespace failure = ::SimpleFluid::detail::vof_failure;
            failure::Scope context("transport_controls");
            context.record.arithmetic_mode = mode;
            context.record.mark(failure::Field::Arithmetic);
            context.record.byte_budget = data_->options.byte_budget;
            context.record.mark(failure::Field::ByteBudget);
            current(*data_);
            if (mode != arithmetic_mode() || !std::isfinite(timestep) || timestep < 0)
                throw std::invalid_argument("General VoF control arithmetic mode or timestep differs.");
            wire::Budget budget{data_->options.byte_budget};
            budget.add(sizeof(GeneralVoFTransportControls));
            const auto reserve = [&](std::size_t count, std::size_t width)
            {
                if (count > (budget.limit - budget.used) / width)
                    throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit,
                        "General VoF control view exceeds its retained byte budget.");
                budget.add(count * width);
            };
            reserve(nodes.size(), sizeof(GeneralVoFNodeVelocity));
            reserve(faces.size(), sizeof(GeneralVoFFaceVolume));
            std::set<const void*> exact_payloads;
            for (const auto& face : faces)
            {
                if (mode == ArithmeticMode::Exact)
                {
                    if (face.bounded_integrated_volume)
                        throw std::invalid_argument("Exact VoF cannot adopt a finite-policy face control.");
                    if (face.exact_integrated_volume)
                    {
                        const auto identity = face.exact_integrated_volume->storage_identity();
                        if (identity && !exact_payloads.contains(identity))
                        {
                            reserve(1, sizeof(decltype(exact_payloads)::value_type) + 4 * sizeof(void*));
                            budget.add(
                                face.exact_integrated_volume->storage_bytes() - sizeof(*face.exact_integrated_volume));
                            exact_payloads.insert(identity);
                        }
                    }
                }
                else
                {
                    if (face.exact_integrated_volume)
                        throw std::invalid_argument("Finite VoF cannot adopt an exact-policy face control.");
                    if (face.bounded_integrated_volume)
                    {
                        fp::validate(*face.bounded_integrated_volume);
                        if (face.bounded_integrated_volume->arithmetic_mode != mode)
                            throw std::invalid_argument("General VoF face control arithmetic mode differs.");
                    }
                }
            }
            result = std::make_shared<GeneralVoFTransportControls>();
            result->source_ = data_;
            result->source_control_ = data_->control;
            result->mode_ = mode;
            result->timestep_ = timestep;
            result->nodes_.assign(nodes.begin(), nodes.end());
            result->faces_.assign(faces.begin(), faces.end());
            // Account actual allocation, including any vector over-allocation.
            reserve(result->nodes_.capacity() - nodes.size(), sizeof(GeneralVoFNodeVelocity));
            reserve(result->faces_.capacity() - faces.size(), sizeof(GeneralVoFFaceVolume));
        });
    agree_transaction(data_->hierarchy->communicator(), data_->control);
    return result;
}
void GeneralVoF::validate_transport_controls(
    const std::shared_ptr<const GeneralVoFTransportControls>& controls, ArithmeticMode mode, bool face_only) const
{
    wire::collective(*data_->hierarchy->communicator(),
        [&]
        {
            current(*data_);
            if (!controls || mode != arithmetic_mode() || controls->mode_ != mode || controls->source_ != data_ ||
                controls->source_control_ != data_->control || (face_only && !controls->nodes_.empty()))
                throw std::invalid_argument(
                    "General VoF control view has incompatible nodes, arithmetic mode or source publication.");
        });
    agree_transaction(data_->hierarchy->communicator(), data_->control);
}
GeneralVoF::Trial GeneralVoF::plan_transport(std::span<const GeneralVoFNodeVelocity> velocity,
    std::span<const GeneralVoFFaceVolume> volumes, double timestep) const
{
    const auto controls = make_transport_controls(arithmetic_mode(), velocity, volumes, timestep);
    return plan_transport_impl(controls->nodes(), controls->faces(), timestep);
}
GeneralVoF::Trial GeneralVoF::plan_transport_impl(std::span<const GeneralVoFNodeVelocity> velocity,
    std::span<const GeneralVoFFaceVolume> volumes, double timestep) const
{
    wire::collective(*data_->hierarchy->communicator(),
        [&]
        {
            if (data_->options.transport_backend != GeneralVoFTransportBackend::SweptPartition)
                throw std::invalid_argument(
                    "Face backend requires plan_face_transport with its explicit constant normal velocity model.");
        });
    if (inventory_policy() != PhaseInventoryPolicy::Finite)
        return Exact::GeneralVoF::plan_transport(*this, velocity, volumes, timestep);
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return plan_transport_finite<ArithmeticMode::Ordinary>(velocity, volumes, timestep);
    return plan_transport_finite<ArithmeticMode::BoundedCertification>(velocity, volumes, timestep);
}
bool GeneralVoF::homogeneous_transport_eligible() const
{
    const auto comm = data_->hierarchy->communicator();
    int local_eligible = 1, eligible = 0;
    wire::collective(*comm,
        [&]
        {
            current(*data_);
            local_eligible = data_->options.arithmetic_mode == ArithmeticMode::Ordinary &&
                             data_->options.transport_backend == GeneralVoFTransportBackend::SweptPartition;
            // Declarations describe geometry. Physical trace composition remains
            // independent and cannot disqualify an otherwise pure declaration.
            for (const auto& [key, cell] : data_->cells)
            {
                local_eligible &= cell.declaration == GeneralVoFPhaseGeometry::PureGas ||
                                  cell.declaration == GeneralVoFPhaseGeometry::PureCarrier;
                if (cell.retained_interface && !cell.retained_interface->pure_gas &&
                    !cell.retained_interface->pure_carrier)
                    local_eligible = 0;
            }
            for (const auto& tile : data_->topology->tiles)
                if (!tile.second_image_identity.empty() ||
                    (tile.second &&
                        data_->cells.at(tile.first).declaration != data_->cells.at(*tile.second).declaration))
                    local_eligible = 0;
        });
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_eligible, &eligible);
    return eligible;
}
GeneralVoF::Trial GeneralVoF::plan_homogeneous_transport(
    std::span<const GeneralVoFFaceVolume> volumes, double timestep) const
{
    if (!homogeneous_transport_eligible())
        throw std::invalid_argument(
            "Homogeneous VoF transport requires an interface-free Ordinary SweptPartition source.");
    return plan_face_transport_impl(volumes, timestep, {1, 1}, 0, true);
}
void GeneralVoF::Trial::validate_retained_algebraic_payload() const
{
    const auto limit = source_->options.byte_budget;
    wire::collective(*source_->hierarchy->communicator(),
        [&]
        {
            wire::Budget budget{limit};
            budget.add(sizeof(Trial));
            const auto retain = [&](const auto& records)
            {
                using Record = typename std::remove_reference_t<decltype(records)>::value_type;
                if (records.capacity() > (limit - budget.used) / sizeof(Record))
                    throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit,
                        "Finite face-ledger trial exceeds its retained byte budget.");
                budget.add(records.capacity() * sizeof(Record));
            };
            retain(unbound_streams_);
            retain(finite_faces_);
            retain(finite_transfers_);
            retain(finite_budgets_);
            retain(finite_trajectories_);
            retain(roundoff_);
            retain(signed_receipts_);
            retain(signed_budgets_);
            retain(numerical_support_transfers_);
            retain(numerical_support_budgets_);
            retain(finite_face_chain_);
            retain(finite_cell_closure_);
            for (const auto& closure : finite_cell_closure_)
                budget.add(AMR::encode_adaptive_key(closure.key).size());
            for (const auto& trace : finite_trajectories_)
                budget.add(PLIC::dynamic_point_storage_bytes(trace.point) +
                           PLIC::dynamic_point_storage_bytes(trace.departure) - 2 * sizeof(PLIC::DynamicPoint));
        });
}
} // namespace SimpleFluid::VoF
