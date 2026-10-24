/** @file GeneralVoFExactTransfer.cc
 * @brief Explicit exact capsule validation, export and adoption.
 */
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF
{
void GeneralVoF::validate_transfer_target(const ExactPhaseTransferResult& result) const
{
    if (arithmetic_mode() != ArithmeticMode::Exact)
        throw std::invalid_argument("Exact transfer API requires explicit Exact arithmetic mode.");
    Exact::GeneralVoF::validate_transfer_target(*this, result);
}
ExactPhaseTransferSource GeneralVoF::export_transfer_source(bool prepare_geometry) const
{
    if (arithmetic_mode() != ArithmeticMode::Exact)
        throw std::invalid_argument("Exact transfer API requires explicit Exact arithmetic mode.");
    return Exact::GeneralVoF::export_transfer_source(*this, prepare_geometry);
}
std::shared_ptr<const GeneralVoF> GeneralVoF::adopt_transfer(const ExactPhaseTransferResult& result) const
{
    if (arithmetic_mode() != ArithmeticMode::Exact)
        throw std::invalid_argument("Exact transfer API requires explicit Exact arithmetic mode.");
    return Exact::GeneralVoF::adopt_transfer(*this, result);
}
namespace Exact
{
void GeneralVoF::validate_transfer_target(const State& self, const ExactPhaseTransferResult& result)
{
    const auto comm = result.source_hierarchy()->communicator();
    wire::collective(*comm,
        [&]
        {
            if (self.data_->hierarchy->communicator().getRawPtr() != comm.getRawPtr() ||
                result.target_hierarchy()->communicator().getRawPtr() != comm.getRawPtr())
                throw std::invalid_argument("General VoF transfer target communicator differs.");
            current(*self.data_);
            if (result.inventory_policy() != self.inventory_policy())
                throw std::invalid_argument("General VoF transfer target policy differs.");
        });
    result.validate_observations();
    agree_transaction(comm, self.data_->control);
    wire::collective(*comm,
        [&]
        {
            current(*self.data_);
            if (self.data_->adopted_source.lock() != result.source_identity() ||
                self.data_->adopted_source_control != result.source_control() ||
                self.data_->adopted_transaction != result.transaction() ||
                self.data_->adopted_target_control != self.data_->control ||
                self.data_->hierarchy->identity() != result.target_hierarchy()->identity() ||
                self.data_->hierarchy->topology_generation() != result.target_hierarchy()->topology_generation() ||
                self.data_->hierarchy->partition_generation() != result.target_hierarchy()->partition_generation() ||
                result.owned().size() != self.data_->hierarchy->owned().size())
                throw std::invalid_argument("General VoF is not the actual adopted transfer publication.");
            // The immutable hierarchy token is preserved only by with_partition;
            // its checked binding authenticates geometry/provider/domain/native GIDs.
            // Compare exact owned state directly. No fitted planes, geometry shells
            // or full source/target capsule arrays are reconstructed here.
            for (const auto& capsule : result.owned())
            {
                const auto found = self.data_->cells.find(capsule.key);
                if (found == self.data_->cells.end() || !self.data_->hierarchy->owned().contains(capsule.key))
                    throw std::invalid_argument("General VoF adopted transfer ownership differs.");
                const auto& cell = found->second;
                if (capsule.policy != self.inventory_policy() ||
                    bool(capsule.experimental_exact) != bool(cell.experimental_exact))
                    throw std::invalid_argument("General VoF adopted policy/payload differs.");
                if (cell.experimental_exact)
                {
                    const auto& a = *cell.experimental_exact;
                    const auto& b = *capsule.experimental_exact;
                    for (const auto& pair : {std::pair{&a.phases.bulk, &b.phases.bulk},
                             std::pair{&a.phases.gas, &b.phases.gas}, std::pair{&a.phases.carrier, &b.phases.carrier},
                             std::pair{&a.error.difference, &b.error.difference},
                             std::pair{&a.error.admitted_bound, &b.error.admitted_bound},
                             std::pair{&a.error.native_conversion, &b.error.native_conversion},
                             std::pair{&a.error.phase_input_bound, &b.error.phase_input_bound},
                             std::pair{&a.error.fit_residual, &b.error.fit_residual},
                             std::pair{&a.error.fit_bound, &b.error.fit_bound},
                             std::pair{&a.error.transferred_bound, &b.error.transferred_bound},
                             std::pair{&a.error.fit_allocation_bound, &b.error.fit_allocation_bound}})
                        if (pair.first->compare(*pair.second))
                            throw std::invalid_argument("Experimental adopted authority/provenance differs.");
                }
                const auto declaration = capsule.geometry.pure_gas
                                             ? GeneralVoFPhaseGeometry::PureGas
                                             : (capsule.geometry.pure_carrier ? GeneralVoFPhaseGeometry::PureCarrier
                                                                              : GeneralVoFPhaseGeometry::Infer);
                if (cell.revision != capsule.revision || cell.declaration != declaration ||
                    bool(cell.retained_interface) !=
                        (capsule.geometry_ready &&
                            self.inventory_policy() == PhaseInventoryPolicy::ExperimentalExact) ||
                    cell.native_volume != capsule.phases.bulk || cell.phase.gas_volume != capsule.phases.gas ||
                    cell.phase.carrier_volume != capsule.phases.carrier || cell.roundoff.gas != capsule.roundoff.gas ||
                    cell.roundoff.carrier != capsule.roundoff.carrier ||
                    cell.finite_history.gas != capsule.accumulated_roundoff.gas ||
                    cell.finite_history.carrier != capsule.accumulated_roundoff.carrier)
                    throw std::invalid_argument("General VoF adopted transfer finite phases differ.");
                const auto& e = cell.occupancy;
                const auto& p = capsule.error;
                for (const auto& pair :
                    {std::pair{&e.difference, &p.difference}, std::pair{&e.admitted_bound, &p.admitted_bound},
                        std::pair{&e.native_conversion, &p.native_conversion},
                        std::pair{&e.phase_input_bound, &p.phase_input_bound},
                        std::pair{&e.fit_residual, &p.fit_residual}, std::pair{&e.fit_bound, &p.fit_bound},
                        std::pair{&e.transferred_bound, &p.transferred_bound},
                        std::pair{&e.fit_allocation_bound, &p.fit_allocation_bound}})
                    if (*pair.first != *pair.second)
                        throw std::invalid_argument("General VoF adopted transfer error provenance differs.");
            }
        });
}
ExactPhaseTransferSource GeneralVoF::export_transfer_source(const State& self, bool prepare_geometry)
{
    const auto comm = self.data_->hierarchy->communicator();
    std::vector<ExactPhaseTransferCapsule> capsules;
    wire::collective(*comm,
        [&]
        {
            current(*self.data_);
            wire::Budget budget{self.data_->options.byte_budget};
            const auto interfaces =
                prepare_geometry ? reconstruct_exact_local(*self.data_) : std::map<Key, PolyhedralInterface>{};
            for (const auto& key : self.data_->topology->order)
            {
                budget.add(sizeof(ExactPhaseTransferCapsule) + AMR::encode_adaptive_key(key).size());
                const auto& cell = self.data_->cells.at(key);
                ExactPhaseTransferCapsule capsule;
                capsule.key = key;
                capsule.policy = self.inventory_policy();
                capsule.experimental_exact = cell.experimental_exact;
                capsule.revision = cell.revision;
                capsule.phases = {cell.native_volume, cell.phase.gas_volume, cell.phase.carrier_volume};
                capsule.roundoff = cell.roundoff;
                capsule.accumulated_roundoff = cell.finite_history;
                const auto& e = cell.occupancy;
                capsule.error = {e.difference, e.admitted_bound, e.native_conversion, e.phase_input_bound,
                    e.fit_residual, e.fit_bound, e.transferred_bound, e.fit_allocation_bound};
                if (self.inventory_policy() == PhaseInventoryPolicy::Finite)
                {
                    // This explicitly selected reference export reconstructs
                    // diagnostic centers only; the accepted finite allowance
                    // is not replaced by a new input-admission tolerance.
                    const auto exact = phase_measures(cell);
                    const auto defect = exact.gas + exact.carrier - exact.bulk;
                    if ((defect.sign() < 0 ? defect.scaled(-1) : defect)
                            .compare(Measure::from_value(e.admitted_bound)) > 0)
                        throw std::invalid_argument(
                            "Explicit reference export exceeds accepted finite occupancy allowance.");
                    capsule.error.difference = detail::phase_signed_value(defect);
                    capsule.error.native_conversion =
                        detail::phase_signed_value(Measure::from_value(cell.native_volume) - exact.bulk);
                }
                if (prepare_geometry)
                {
                    capsule.geometry = interfaces.at(key);
                    capsule.geometry_ready = true;
                }
                else if (cell.retained_interface &&
                         self.inventory_policy() == PhaseInventoryPolicy::ExperimentalExact)
                {
                    capsule.geometry = *cell.retained_interface;
                    capsule.geometry_ready = true;
                }
                if (cell.declaration != GeneralVoFPhaseGeometry::Infer || !phase_measures(cell).gas.sign() ||
                    !phase_measures(cell).carrier.sign())
                {
                    capsule.geometry.pure_gas =
                        cell.declaration == GeneralVoFPhaseGeometry::PureGas || !phase_measures(cell).carrier.sign();
                    capsule.geometry.pure_carrier =
                        cell.declaration == GeneralVoFPhaseGeometry::PureCarrier || !phase_measures(cell).gas.sign();
                    capsule.geometry_ready = true;
                }
                if (capsule.geometry_ready)
                {
                    const bool gas = phase_measures(cell).gas.compare(phase_measures(cell).carrier) <= 0;
                    const auto target = gas ? phase_measures(cell).gas : phase_measures(cell).carrier;
                    const auto raw =
                        phases(*Meshes::ExactConvexFragment::from_geometry(*cell.phase.geometry), capsule.geometry);
                    const auto fit_diagnostic = detail::round_phase_residual(target - (gas ? raw.gas : raw.carrier));
                    capsule.error.fit_residual = fit_diagnostic.value;
                    const auto view = Measure::from_value(
                        gas ? geometric_sample(cell).gas_volume : geometric_sample(cell).carrier_volume);
                    const auto conversion = target - view;
                    auto fit_bound = view.scaled(self.data_->options.reconstruction.volume_tolerance) +
                                     (conversion.sign() < 0 ? conversion.scaled(-1) : conversion);
                    if (cell.declaration != GeneralVoFPhaseGeometry::Infer)
                    {
                        if (cell.experimental_exact)
                        {
                            const auto& exact = cell.experimental_exact->error;
                            fit_bound = exact.admitted_bound.compare(exact.fit_bound) >= 0 ? exact.admitted_bound
                                                                                           : exact.fit_bound;
                        }
                        else
                            fit_bound = finite_bound(std::max(e.admitted_bound, e.fit_bound));
                    }
                    capsule.error.fit_bound =
                        detail::phase_upper_bound(fit_bound + finite_bound(fit_diagnostic.absolute_error));
                    // Reuse the same independent admission proof as transport;
                    // merely observing the fit residual never enlarges its bound.
                    (void) detail::ExactPhaseLabelMap::create(
                        raw, gas, target, fit_bound, self.data_->options.byte_budget);
                    if (cell.experimental_exact)
                    {
                        auto prepared = std::make_shared<ExperimentalPhaseInventory>(*cell.experimental_exact);
                        prepared->error.fit_residual = target - (gas ? raw.gas : raw.carrier);
                        prepared->error.fit_bound = fit_bound;
                        capsule.experimental_exact = std::move(prepared);
                    }
                }
                capsules.push_back(std::move(capsule));
            }
        });
    std::optional<ExactPhaseTransferSource> result;
    wire::collective(*comm,
        [&]
        {
            result.emplace(ExactPhaseTransferSource(self.data_->hierarchy, self.data_, self.data_->control,
                std::move(capsules), self.inventory_policy()));
        });
    return std::move(*result);
}
std::shared_ptr<const VoF::GeneralVoF> GeneralVoF::adopt_transfer(
    const State& self, const ExactPhaseTransferResult& result)
{
    const auto comm = self.data_->hierarchy->communicator();
    result.validate_observations();
    std::vector<Key> keys;
    std::vector<double> gas, carrier;
    std::vector<GeneralVoFPhaseGeometry> declarations;
    wire::collective(*comm,
        [&]
        {
            current(*self.data_);
            if (result.inventory_policy() != self.inventory_policy())
                throw std::invalid_argument("General VoF adoption inventory policy differs.");
            if (result.source_identity().get() != self.data_.get() || result.source_control() != self.data_->control ||
                result.source_hierarchy().get() != self.data_->hierarchy.get())
                throw std::invalid_argument("General VoF exact transfer belongs to another accepted source.");
            wire::Budget budget{self.data_->options.byte_budget};
            for (const auto& capsule : result.owned())
            {
                budget.add(sizeof(Key) + 2 * sizeof(double) + sizeof(GeneralVoFPhaseGeometry) +
                           AMR::encode_adaptive_key(capsule.key).size());
                keys.push_back(capsule.key);
                gas.push_back(capsule.phases.gas);
                carrier.push_back(capsule.phases.carrier);
                declarations.push_back(capsule.geometry.pure_gas
                                           ? GeneralVoFPhaseGeometry::PureGas
                                           : (capsule.geometry.pure_carrier ? GeneralVoFPhaseGeometry::PureCarrier
                                                                            : GeneralVoFPhaseGeometry::Infer));
            }
        });
    agree_transaction(comm, result.transaction());
    return std::shared_ptr<const VoF::GeneralVoF>(new VoF::GeneralVoF(result.target_hierarchy(), std::move(keys),
        std::move(gas), std::move(carrier), self.data_->options, std::move(declarations), &result));
}
} // namespace Exact
}
