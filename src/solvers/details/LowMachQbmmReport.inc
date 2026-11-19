/** @file LowMachQbmmReport.inc
 * @brief Native endpoint diagnostics and retained fixed-mesh QBMM provenance.
 */
namespace
{
using ReportSum = detail::CompensatedSum<long double>;

enum MaterialOperand : std::size_t
{
    defect,
    old_inventory,
    new_inventory,
    gross_external,
    net_external,
    operand_count
};
using MaterialSums = std::array<ReportSum, operand_count>;

long double qbmm_check_material_budget(
    const MaterialSums& sums, double relative_tolerance, double roundoff_tolerance, const char* message)
{
    const long double before = sums[old_inventory].value(), after = sums[new_inventory].value(),
                      external = sums[gross_external].value(), residual = sums[defect].value();
    const auto physical_budget = relative_tolerance * std::max({before, after, external});
    const auto roundoff_budget = roundoff_tolerance * (before + after + external);
    check(std::isfinite(residual) && std::isfinite(physical_budget) && std::isfinite(roundoff_budget) &&
              std::abs(residual) <= physical_budget + roundoff_budget,
        message);
    return residual;
}

std::map<std::string, double> qbmm_declared_elements(const SpeciesCatalog& catalog)
{
    std::map<std::string, double> elements;
    for (const auto& species : catalog.species())
        for (const auto& entry : species.elements)
            elements.try_emplace(entry.first, 0);
    return elements;
}

Solver::QbmmBalances qbmm_balance_diagnostics(const Solver::Diagnostics& stats)
{
    Solver::QbmmBalances balances;
    balances.mass_defect_kg = stats.mass_error;
    balances.internal_energy_defect_j = stats.internal_energy_error;
    balances.pressure_work_j = stats.pressure_work;
    balances.energy_roundoff_bound_j = stats.internal_energy_roundoff_bound;
    balances.species_defects_mol = stats.species_errors;
    return balances;
}

std::optional<Solver::QbmmBalances> qbmm_archived_native_balances(const Solver::Diagnostics& stats)
{
    if (stats.correctors == 0)
        return std::nullopt;
    auto balances = qbmm_balance_diagnostics(stats);
    balances.material_basis = Solver::QbmmMaterialBalanceBasis::ArchivedNativeDefect;
    return balances;
}

double qbmm_report_quotient(double numerator, double denominator, const char* message)
{
    const auto value = static_cast<long double>(numerator) / denominator;
    const double rounded = numerator / denominator;
    check(
        std::isfinite(denominator) && denominator > 0 && std::isfinite(value) && std::isfinite(rounded) &&
            (numerator == 0 || (rounded != 0 && std::abs((rounded - value) / value) <= qbmm_reconstruction_tolerance)),
        message);
    return rounded;
}

double qbmm_report_phase_mass(const PhaseThermalState& phase)
{
    const auto mass = phase_mass(phase);
    check(std::isfinite(mass) &&
              (mass > 0 || (mass == 0 && std::ranges::all_of(phase.moles, [](double value) { return value == 0; }))),
        "QBMM report phase mass is not representable.");
    if (std::fpclassify(mass) == FP_SUBNORMAL)
    {
        ReportSum summed_mass;
        for (std::size_t s = 0; s < phase.moles.size(); ++s)
            summed_mass +=
                static_cast<long double>(phase.moles[s]) * phase.thermodynamics->catalog().species()[s].molarMass;
        const auto exact = summed_mass.value();
        check(std::abs((mass - exact) / exact) <= qbmm_reconstruction_tolerance,
            "QBMM report phase mass loses relative precision.");
    }
    return mass;
}

Solver::QbmmCellFamilyReport qbmm_cell_report(const QbmmPopulationSchema& family, const Solver::CellState& cell,
    std::size_t phase_index, double phase_mass, double bulk_volume)
{
    Solver::QbmmCellFamilyReport report;
    const auto& phase = cell.phases[phase_index];
    const auto& number = cell.phase_numbers[phase_index].value();
    const auto& shape = cell.qbmm_shape[phase_index].value();
    const QbmmCellView view{
        family.population(), family.catalog().fingerprint(), phase.moles, number, shape, family.fingerprint()};
    report.species_moles = phase.moles;
    report.moments = qbmm_cell_moments(family, view);
    report.quadrature = invert_qbmm_moments(report.moments);
    report.absolute_pressure_pa = phase.absolute_pressure;
    report.phase_internal_energy_j = phase.internal_energy;
    report.phase_mass_kg = phase_mass;
    report.bulk_mass_density_kg_per_m3 =
        qbmm_report_quotient(phase_mass, bulk_volume, "QBMM report bulk mass contribution is not representable.");
    report.bulk_concentration_mol_per_m3.reserve(phase.moles.size());
    for (double moles : phase.moles)
        report.bulk_concentration_mol_per_m3.push_back(
            qbmm_report_quotient(moles, bulk_volume, "QBMM report bulk molar concentration is not representable."));
    long double scale_power = 1;
    for (std::size_t k = 0; k < report.moments.size(); ++k)
    {
        const auto density = scale_power * report.moments[k] / bulk_volume;
        check(std::isfinite(density) && (report.moments[k] == 0 || density > 0),
            "QBMM report SI mass moment density is not representable.");
        report.mass_moment_density_si[k] = density;
        if (k + 1 < report.moments.size())
            scale_power *= family.mass_scale_kg();
    }
    std::optional<double> specific_volume;
    if (number > 0)
    {
        const double density =
            phase.thermodynamics->evaluate(phase.temperature, phase.absolute_pressure, fractions(phase)).density;
        report.phase_density_kg_per_m3 = density;
        report.temperature_K = phase.temperature;
        report.phase_volume_m3 =
            qbmm_report_quotient(phase_mass, density, "QBMM report phase volume is not representable.");
        report.phase_fraction = qbmm_report_quotient(
            report.phase_volume_m3, bulk_volume, "QBMM report phase fraction is not representable.");
        specific_volume = qbmm_report_quotient(1, density, "QBMM report phase specific volume is not representable.");
        auto& concentrations = report.phase_concentration_mol_per_m3.emplace();
        concentrations.reserve(phase.moles.size());
        for (double moles : phase.moles)
            concentrations.push_back(qbmm_report_quotient(
                moles, report.phase_volume_m3, "QBMM report phase molar concentration is not representable."));
    }
    if (report.quadrature.valid())
    {
        report.geometry = qbmm_geometry(family, view, bulk_volume, specific_volume);
        report.closure_evaluated = number > 0;
    }
    return report;
}

void qbmm_update_aggregate(Solver::QbmmFamilyReport& family, const QbmmQuadrature& quadrature)
{
    ++family.inversion_counts[static_cast<std::size_t>(quadrature.status)];
    if (quadrature.status == InversionStatus::Absent)
        return;
    const auto& conditioning = quadrature.conditioning;
    const auto add_margin = [](std::optional<long double>& current, long double margin)
    { current = current ? std::min(*current, margin) : margin; };
    add_margin(family.minimum_hankel_margin, conditioning.hankel_margin);
    add_margin(family.minimum_shifted_hankel_margin, conditioning.shifted_hankel_margin);
    for (std::size_t k = 0; k < quadrature.reconstruction.absolute.size(); ++k)
    {
        auto& maximum = family.maximum_reconstruction;
        maximum.absolute[k] = std::max(maximum.absolute[k], std::abs(quadrature.reconstruction.absolute[k]));
        maximum.relative[k] = std::max(maximum.relative[k], std::abs(quadrature.reconstruction.relative[k]));
    }
}

void qbmm_reduce_aggregate(const Teuchos::Comm<int>& comm, Solver::QbmmFamilyReport& family)
{
    if (comm.getSize() == 1)
        return;
    auto counts = family.inversion_counts;
    Teuchos::reduceAll(
        comm, Teuchos::REDUCE_SUM, static_cast<int>(counts.size()), counts.data(), family.inversion_counts.data());
    for (std::size_t i = 0; i < family.applicability_counts.size(); ++i)
    {
        const auto local_counts = family.applicability_counts[i];
        Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, static_cast<int>(local_counts.size()), local_counts.data(),
            family.applicability_counts[i].data());
        const auto local_failures = family.applicability_gate_failures[i];
        Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, 1, &local_failures, &family.applicability_gate_failures[i]);
    }
#ifdef HAVE_TEUCHOS_MPI
    if (const auto* mpi = dynamic_cast<const Teuchos::MpiComm<int>*>(&comm))
    {
        // A single maximum reduction retains the sign and precision of both Hankel extrema.
        constexpr auto negative_infinity = -std::numeric_limits<long double>::infinity();
        using Extrema = std::array<long double, 10>;
        Extrema local{-family.minimum_hankel_margin.value_or(-negative_infinity),
            -family.minimum_shifted_hankel_margin.value_or(-negative_infinity)};
        Extrema global{};
        for (std::size_t k = 0; k < family.maximum_reconstruction.absolute.size(); ++k)
        {
            local[2 + k] = family.maximum_reconstruction.absolute[k];
            local[6 + k] = family.maximum_reconstruction.relative[k];
        }
        if (MPI_Allreduce(local.data(), global.data(), static_cast<int>(local.size()), MPI_LONG_DOUBLE, MPI_MAX,
                *mpi->getRawMpiComm()) != MPI_SUCCESS)
            throw std::runtime_error("QBMM report extrema reduction failed.");
        if (global[0] != negative_infinity)
            family.minimum_hankel_margin = -global[0];
        if (global[1] != negative_infinity)
            family.minimum_shifted_hankel_margin = -global[1];
        for (std::size_t k = 0; k < family.maximum_reconstruction.absolute.size(); ++k)
        {
            family.maximum_reconstruction.absolute[k] = global[2 + k];
            family.maximum_reconstruction.relative[k] = global[6 + k];
        }
        return;
    }
#endif
    throw std::invalid_argument("Distributed QBMM reporting requires a native MPI communicator.");
}

void qbmm_report_applicability(const QbmmPopulationSchema& schema, const Solver::CellState& state,
    std::size_t phase_index, double bulk_volume, double gas_fraction, std::optional<double> duration,
    Solver::QbmmCellFamilyReport& report, Solver::QbmmFamilyReport& aggregate)
{
    if (aggregate.law_metadata.empty())
        return;
    std::optional<std::vector<double>> liquid_concentrations;
    std::optional<double> liquid_temperature;
    const bool needs_liquid = std::ranges::any_of(aggregate.law_metadata, qbmm_uses_carrier_liquid);
    const auto liquid_index = needs_liquid ? qbmm_native_law_detail::carrier_liquid_index(state.phases) : std::nullopt;
    if (liquid_index)
    {
        const auto& liquid = state.phases[*liquid_index];
        const double mass = qbmm_report_phase_mass(liquid);
        if (mass > 0)
        {
            const auto point =
                liquid.thermodynamics->evaluate(liquid.temperature, liquid.absolute_pressure, fractions(liquid));
            const double volume = qbmm_report_quotient(
                mass, point.density, "QBMM applicability carrier-liquid volume is not representable.");
            liquid_concentrations = qbmm_phase_concentrations(liquid.moles, volume);
            liquid_temperature = liquid.temperature;
        }
    }
    const auto& drift = state.drift_velocity.at(phase_index);
    const auto inputs = make_qbmm_law_inputs(schema,
        {report.quadrature, report.geometry, report.species_moles, bulk_volume, report.phase_volume_m3,
            report.absolute_pressure_pa, report.temperature_K, duration, gas_fraction, std::move(liquid_concentrations),
            liquid_temperature, {drift.component(0), drift.component(1), drift.component(2)}});
    for (std::size_t i = 0; i < aggregate.law_metadata.size(); ++i)
    {
        auto evaluation = evaluate_qbmm_applicability(aggregate.law_metadata[i], inputs);
        const bool absent =
            report.quadrature.status == InversionStatus::Absent && aggregate.law_metadata[i].requires_population;
        ++aggregate.applicability_counts[i][absent ? 2 : evaluation.applicable ? 0 : 1];
        if (!evaluation.gate_passed)
            ++aggregate.applicability_gate_failures[i];
        report.applicability.push_back(std::move(evaluation));
    }
}
} // namespace

std::shared_ptr<const LowMachMixtureSolver::QbmmIntervalReport> LowMachMixtureSolver::qbmm_interval(const Trial& trial)
{
    const auto& transport = trial.transport_snapshot();
    auto interval = std::make_shared<QbmmIntervalReport>();
    interval->preview_id = trial.qbmm_preview_id;
    interval->transported_from_revision = transport.accepted_revision;
    interval->transported_from_generation = transport.accepted_generation;
    interval->transported_from_time = transport.accepted_time;
    interval->old_geometry_epoch = transport.old_geometry_epoch;
    interval->new_geometry_epoch = transport.new_geometry_epoch;
    interval->catalog_fingerprint = transport.schema_fingerprint;
    interval->time_step = transport.time_step;
    interval->source_stage = transport.stage;
    interval->local_receipts = trial.qbmm_cells;
    interval->local_gross_events = trial.gross_events;
    return interval;
}

LowMachMixtureSolver::QbmmReport LowMachMixtureSolver::qbmm_report() const
{
    return qbmm_report_impl(nullptr);
}

LowMachMixtureSolver::QbmmReport LowMachMixtureSolver::qbmm_report(const Trial& trial) const
{
    return qbmm_report_impl(&trial);
}

LowMachMixtureSolver::QbmmReport LowMachMixtureSolver::qbmm_report_impl(const Trial* trial) const
{
    const auto& mesh = *d->mesh;
    QbmmReport report;
    auto& provenance = report.provenance;
    const auto& catalog = d->eos.front()->catalog();
    const auto& families = d->options.qbmm_families;
    collective_detail::collective_local_validation(mesh, "Native QBMM report selection",
        [&]
        {
            check(d->initialized && !families.empty() && d->epoch == mesh_geometry_epoch(mesh),
                "QBMM reports require initialized state on the configured fixed geometry.");
            if (trial)
            {
                check(trial->owner == this && !trial->ale_geometry && trial->nonce == d->trial_nonce &&
                          trial->base_revision == d->accepted.revision && trial->geometry_epoch == d->epoch &&
                          trial->logical_geometry == d->logical_geometry && trial->qbmm_balances.has_value(),
                    "QBMM report trial is foreign, stale or belongs to different geometry.");
                provenance.kind = QbmmStateKind::Trial;
                provenance.origin = QbmmStateOrigin::Corrector;
                provenance.preview_id = trial->qbmm_preview_id;
                if (trial->provider)
                    provenance.controller_configuration_signature = d->active_qbmm_controller;
                report.receipt_history = QbmmReceiptHistory::Available;
            }
            else
            {
                provenance.controller_configuration_signature = d->accepted_qbmm_controller;
                if (d->qbmm_publication)
                {
                    provenance.origin = d->qbmm_publication->origin;
                    provenance.archived_generation = d->qbmm_publication->archived_generation;
                    report.receipt_history = d->qbmm_publication->history;
                }
            }
        });
    const State& state = trial ? trial->candidate : d->accepted;
    provenance.revision = state.revision;
    provenance.generation = d->trial_nonce;
    provenance.geometry_epoch = d->epoch;
    provenance.catalog_fingerprint = catalog.fingerprint();
    provenance.time = state.time;
    provenance.thermodynamic_pressure_pa = state.thermodynamic_pressure;
    collective_detail::require_uniform_value(mesh, static_cast<int>(provenance.kind), "QBMM report kind");
    collective_detail::require_uniform_value(mesh, static_cast<int>(provenance.origin), "QBMM report origin");
    collective_detail::require_uniform_value(mesh, static_cast<int>(report.receipt_history), "QBMM report history");
    collective_detail::require_uniform_value(mesh, provenance.generation, "QBMM report live generation");
    collective_detail::require_uniform_value(
        mesh, provenance.archived_generation.value_or(0), "QBMM report archive generation");
    collective_detail::require_uniform_value(mesh, int(provenance.controller_configuration_signature.has_value()),
        "QBMM report controller configuration presence");
    collective_detail::require_uniform_value(
        mesh, provenance.controller_configuration_signature.value_or(0), "QBMM report controller configuration");
    collective_detail::require_uniform_value(mesh, provenance.preview_id.value_or(0), "QBMM report trial selection");
    collective_detail::require_uniform_value(mesh, provenance.revision, "QBMM report revision");
    collective_detail::require_uniform_value(mesh, provenance.geometry_epoch, "QBMM report geometry epoch");
    collective_detail::require_uniform_value(mesh, provenance.time, "QBMM report time");
    collective_detail::require_uniform_value(mesh, provenance.thermodynamic_pressure_pa, "QBMM report pressure");
    collective_detail::collective_local_validation(mesh, "Native QBMM endpoint report",
        [&]
        {
            const auto execution = acquire_mesh_execution(mesh);
            check(d->epoch == mesh_geometry_epoch(mesh) && state.cells.size() == mesh.num_owned_cells() &&
                      (!trial || trial->transport.new_control_volumes.size() == state.cells.size()),
                "QBMM report requires the original owned cells and geometry.");
            qbmm_detail::validate_stamps(d->options, state);
            report.species_ids.reserve(catalog.species().size());
            for (const auto& species : catalog.species())
                report.species_ids.push_back(species.id);
            report.families.resize(families.size());
            for (std::size_t a = 0; a < families.size(); ++a)
                if (families[a])
                {
                    auto& aggregate = report.families[a].emplace();
                    aggregate.population = families[a]->population();
                    aggregate.stamp = state.qbmm_families[a].value();
                    aggregate.law_metadata = families[a]->closure_versions().laws;
                    aggregate.applicability_counts.resize(aggregate.law_metadata.size());
                    aggregate.applicability_gate_failures.resize(aggregate.law_metadata.size());
                }
            report.cells.resize(state.cells.size());
            for (std::size_t c = 0; c < state.cells.size(); ++c)
            {
                auto& cell = report.cells[c];
                const auto local = static_cast<LO>(c);
                cell.geometry_cell_id = mesh.cell_geometry_global_id(local);
                cell.bulk_volume_m3 = trial ? trial->transport.new_control_volumes[c] : mesh.cell_volume(local);
                cell.mechanical_pressure_pa = state.cells[c].mechanical_pressure;
                cell.carrier_velocity = state.cells[c].velocity;
                check(std::isfinite(cell.bulk_volume_m3) && cell.bulk_volume_m3 > 0, "QBMM report needs bulk volume.");
                double mixture_mass = 0, gas_volume = 0;
                FaceSum total_mass;
                std::array<FaceSum, 3> drift_momentum, absolute_momentum;
                cell.families.resize(families.size());
                for (std::size_t a = 0; a < families.size(); ++a)
                {
                    const auto& phase = state.cells[c].phases[a];
                    const auto mass = qbmm_report_phase_mass(phase);
                    mixture_mass += mass;
                    total_mass.add_product(mass, 1.);
                    for (std::size_t axis = 0; axis < 3; ++axis)
                    {
                        const double drift = state.cells[c].drift_velocity[a].component(axis);
                        drift_momentum[axis].add_product(mass, drift);
                        // Distribute the phase mass over carrier + drift to preserve both operands.
                        absolute_momentum[axis].add_product(mass, cell.carrier_velocity.component(axis));
                        absolute_momentum[axis].add_product(mass, drift);
                    }
                    if (mass > 0 && !cell.temperature_K)
                        cell.temperature_K = phase.temperature;
                    if (families[a])
                    {
                        cell.families[a] = qbmm_cell_report(*families[a], state.cells[c], a, mass, cell.bulk_volume_m3);
                        qbmm_update_aggregate(*report.families[a], cell.families[a]->quadrature);
                        gas_volume += cell.families[a]->phase_volume_m3;
                    }
                    else if (mass > 0 && phase.thermodynamics->kind() == EOSKind::Gas)
                    {
                        const auto point = phase.thermodynamics->evaluate(
                            phase.temperature, phase.absolute_pressure, fractions(phase));
                        gas_volume += qbmm_report_quotient(
                            mass, point.density, "QBMM report gas phase volume is not representable.");
                    }
                }
                cell.mixture_density_kg_per_m3 = qbmm_report_quotient(
                    mixture_mass, cell.bulk_volume_m3, "QBMM report mixture density is not representable.");
                cell.gas_volume_fraction = qbmm_report_quotient(
                    gas_volume, cell.bulk_volume_m3, "QBMM report gas volume fraction is not representable.");
                const auto interval = trial ? nullptr : d->qbmm_publication ? d->qbmm_publication->interval : nullptr;
                const std::optional<double> duration = trial      ? std::optional<double>(trial->transport.time_step)
                                                       : interval ? std::optional<double>(interval->time_step)
                                                                  : std::nullopt;
                for (std::size_t a = 0; a < families.size(); ++a)
                    if (families[a] && !families[a]->closure_versions().laws.empty())
                        qbmm_report_applicability(*families[a], state.cells[c], a, cell.bulk_volume_m3,
                            cell.gas_volume_fraction, duration, *cell.families[a], *report.families[a]);
                check(mixture_mass > 0, "QBMM report barycentric velocity requires positive finite mixture mass.");
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    const double drift = drift_momentum[axis].rounded(total_mass);
                    const double barycentric = absolute_momentum[axis].rounded(total_mass);
                    check(std::isfinite(drift) && std::isfinite(barycentric),
                        "QBMM report drift or barycentric velocity is not finite.");
                    cell.mass_weighted_drift.component(axis) = drift;
                    cell.barycentric_velocity.component(axis) = barycentric;
                }
            }
            if (trial)
            {
                report.interval = qbmm_interval(*trial);
                report.balances = trial->qbmm_balances;
            }
            else if (d->qbmm_publication)
            {
                report.interval = d->qbmm_publication->interval;
                report.balances = d->qbmm_publication->balances;
            }
            const auto& diagnostics = trial ? trial->statistics : d->diagnostics;
            if (diagnostics.correctors > 0)
            {
                auto& convergence = report.convergence.emplace();
                convergence.correctors = diagnostics.correctors;
                convergence.source_evaluations = diagnostics.source_evaluations;
                convergence.maximum_event_residual = diagnostics.maximum_event_residual;
                convergence.maximum_fourier_gross_comparison_uncertainty_j =
                    diagnostics.maximum_fourier_gross_comparison_uncertainty_j;
                const auto version = trial || !d->qbmm_publication ? 3 : d->qbmm_publication->shape_diagnostics_version;
                if (version >= 2)
                    convergence.maximum_shape_source_residual = diagnostics.maximum_shape_source_residual;
                if (version >= 3)
                    convergence.maximum_shape_transport_residual = diagnostics.maximum_shape_transport_residual;
            }
        });
    const auto comm = mesh.owned_cell_map()->getComm();
    for (auto& aggregate : report.families)
        if (aggregate)
            qbmm_reduce_aggregate(*comm, *aggregate);
    return report;
}

void LowMachMixtureSolver::finalize_qbmm_balances(
    Trial& trial, std::optional<std::span<const SpeciesSourceLedger>> chemistry) const
{
    const auto& mesh = *d->mesh;
    const auto& catalog = d->eos.front()->catalog();
    const auto species = catalog.species().size();
    const auto cells = trial.candidate.cells.size();
    const auto comm = mesh.owned_cell_map()->getComm();
    std::vector<long double> local, global;
    collective_detail::collective_local_validation(mesh, "QBMM physical report balance operands",
        [&]
        {
            check(trial.owner == this && cells == d->accepted.cells.size() &&
                      (!chemistry || (cells == chemistry->size() && trial.qbmm_cells.size() == cells)) &&
                      (trial.sources.empty() || trial.sources.size() == cells) &&
                      species < static_cast<std::size_t>(std::numeric_limits<int>::max()) / operand_count,
                "QBMM report balance layout differs from the trial and accepted state.");
            std::vector<MaterialSums> material_sums(species + 1);
            auto& mass = material_sums.back();
            const auto add_stock = [](MaterialSums& sums, long double before, long double after)
            {
                sums[old_inventory] += before;
                sums[new_inventory] += after;
                sums[defect] += after;
                sums[defect] += -before;
            };
            const auto add_external = [&](std::size_t s, double moles, long double kilograms)
            {
                const auto add_operand = [](MaterialSums& sums, long double amount)
                {
                    sums[defect] += -amount;
                    sums[net_external] += amount;
                    sums[gross_external] += std::abs(amount);
                };
                add_operand(material_sums[s], moles);
                add_operand(mass, kilograms);
            };
            const auto add_species_packet = [&](std::size_t s, double moles)
            {
                const auto molar_mass = catalog.species()[s].molarMass;
                add_external(s, moles, static_cast<long double>(moles) * molar_mass);
            };
            for (std::size_t c = 0; c < cells; ++c)
            {
                const auto& before = d->accepted.cells[c];
                const auto& after = trial.candidate.cells[c];
                for (std::size_t a = 0; a < after.phases.size(); ++a)
                    for (std::size_t s = 0; s < species; ++s)
                    {
                        const auto n_old = before.phases[a].moles[s], n_new = after.phases[a].moles[s];
                        const auto molar_mass = catalog.species()[s].molarMass;
                        add_stock(material_sums[s], n_old, n_new);
                        add_stock(mass, static_cast<long double>(n_old) * molar_mass,
                            static_cast<long double>(n_new) * molar_mass);
                        if (!chemistry && !trial.sources.empty() && !trial.sources[c].species_moles.empty())
                            add_species_packet(s, trial.sources[c].species_moles[a][s]);
                    }
                if (chemistry)
                {
                    for (const auto& transfer : (*chemistry)[c].externalTransfers)
                        add_external(catalog.index(transfer.species), transfer.moles, transfer.mass);
                    for (const auto& receipt : trial.qbmm_cells[c])
                        if (receipt)
                            for (const auto& event : receipt->events)
                                for (const auto& item : event.transfers)
                                    if (std::holds_alternative<QbmmReservoirExchange>(item.transfer.counterpart))
                                        for (std::size_t s = 0; s < species; ++s)
                                            add_species_packet(
                                                s, item.into_family ? item.transfer.moles[s] : -item.transfer.moles[s]);
                }
            }
            local.resize(operand_count * material_sums.size());
            global.resize(local.size());
            for (std::size_t s = 0; s < material_sums.size(); ++s)
                for (std::size_t k = 0; k < operand_count; ++k)
                {
                    auto& operand = local[s * operand_count + k];
                    operand = material_sums[s][k].value();
                    check(std::isfinite(operand), "QBMM local physical mass or element balance is unrepresentable.");
                }
        });
    transport_consistency_detail::sum_qbmm_budget(*comm, local, global);
    collective_detail::collective_local_validation(mesh, "QBMM global physical report balances",
        [&]
        {
            auto balances = qbmm_balance_diagnostics(trial.statistics);
            if (chemistry)
                balances.material_basis = QbmmMaterialBalanceBasis::DeclaredReservoirs;
            balances.external_species_moles.resize(species);
            std::map<std::string, MaterialSums> elements;
            MaterialSums mass;
            for (std::size_t k = 0; k < operand_count; ++k)
                mass[k] += global[species * operand_count + k];
            for (std::size_t s = 0; s < species; ++s)
            {
                const auto operands = std::span<const long double>(global).subspan(s * operand_count, operand_count);
                const auto& definition = catalog.species()[s];
                for (const auto& [name, count] : definition.elements)
                {
                    auto& sums = elements[name];
                    for (std::size_t k = 0; k < operand_count; ++k)
                        sums[k] += operands[k] * count;
                }
                balances.external_species_moles[s] = static_cast<double>(operands[net_external]);
                check(std::isfinite(balances.external_species_moles[s]),
                    "QBMM report global external species input is unrepresentable.");
            }
            const double tolerance = d->options.conservation_relative_tolerance;
            // Canonical stocks and the source packets are binary64. Reuse the native closed transport
            // allowance for those operands and the rank reduction, independently of the physical gate.
            const double roundoff = transport_consistency_detail::qbmm_closed_budget_tolerance(comm->getSize());
            balances.mass_defect_kg = static_cast<double>(
                qbmm_check_material_budget(mass, tolerance, roundoff, "QBMM physical total mass budget failed."));
            check(
                std::isfinite(balances.mass_defect_kg), "QBMM report global physical mass defect is unrepresentable.");
            for (const auto& [name, sums] : elements)
            {
                const auto residual = static_cast<double>(
                    qbmm_check_material_budget(sums, tolerance, roundoff, "QBMM physical element budget failed."));
                check(std::isfinite(residual), "QBMM report global physical element defect is unrepresentable.");
                balances.element_defects_mol.emplace(name, residual);
            }
            trial.qbmm_balances = std::move(balances);
        });
}
