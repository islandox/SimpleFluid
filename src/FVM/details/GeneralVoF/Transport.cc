/** @file Transport.cc
 * @brief Operation dispatch and trial publication validation.
 */
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF
{
GeneralVoF::Trial GeneralVoF::plan_transport(std::span<const GeneralVoFNodeVelocity> velocity,
    std::span<const GeneralVoFFaceVolume> volumes, double timestep) const
{
    if (data_->options.transport_backend != GeneralVoFTransportBackend::SweptPartition)
        throw std::invalid_argument(
            "Face backend requires plan_face_transport with its explicit constant normal velocity model.");
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
