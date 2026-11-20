/** @file ReactingMixtureCoupler.cc
 * @brief Transport-first chemistry and immutable low-Mach source transactions.
 */
#include "solvers/ReactingMixtureCoupler.hh"
#include "equations/CollectiveValidation.hh"
#include "equations/QbmmLawInputs.hh"
#include "equations/details/QbmmPopulationEvents.hh"
#include "solvers/details/LowMachConditionedThermal.hh"
#include "solvers/details/LowMachFaceSourceComposition.hh"
#include "solvers/details/LowMachPopulationSource.hh"
#include "solvers/details/LowMachQbmmControllerRestart.hh"
#include "solvers/details/LowMachQbmmLawInputs.hh"
#include "solvers/details/LowMachQbmmState.hh"
#include "utils/CompensatedSum.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <locale>
#include <new>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>

namespace SimpleFluid::Reacting
{
namespace
{
using Sum = detail::CompensatedSum<long double>;
using Solver = LowMachMixtureSolver;
namespace Faces = low_mach_faces;
void require(bool valid, const char* message)
{
    if (!valid)
        throw std::invalid_argument(message);
}
ReactionLocation source_location(const Solver::SourceStageContext& context, size_t owned_cell)
{
    const auto epoch = context.transport.new_geometry_epoch;
    require(epoch == context.mesh.geometry_epoch(), "Source location belongs to stale trial geometry.");
    const auto cell = static_cast<Solver::Pack::local_ordinal_type>(owned_cell);
    const auto geometry_id = context.mesh.cell_geometry_global_id(cell);
    require(std::in_range<std::uint64_t>(geometry_id), "Canonical source cell identity is not representable.");
    const auto center = context.mesh.cell_centroid(cell);
    return {static_cast<std::uint64_t>(geometry_id), epoch, {center.x, center.y, center.z}, owned_cell};
}
SpeciesPhase compartment(const MixturePhaseBinding& binding, size_t species)
{
    return binding.species_phases.empty() ? binding.phase : binding.species_phases.at(species);
}
void uniform_text(const Solver::Mesh& mesh, const std::string& text)
{
    collective_detail::collective_local_validation(mesh, "Reacting mixture schema length", [&]
        { require(text.size() <= size_t(std::numeric_limits<int>::max()), "Reacting configuration is too large."); });
    const int size = static_cast<int>(text.size());
    collective_detail::require_uniform_value(mesh, size, "Reacting mixture configuration length");
    std::vector<int> values, lower, upper;
    collective_detail::collective_local_validation(mesh, "Controller metadata buffers",
        [&]
        {
            values.assign(text.begin(), text.end());
            lower.resize(size);
            upper.resize(size);
        });
    if (size)
    {
        const auto comm = mesh.owned_cell_map()->getComm();
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, size, values.data(), lower.data());
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, size, values.data(), upper.data());
    }
    require(lower == upper, "Reacting mixture configuration must agree on every rank.");
}
bool identical_sources(std::span<const Solver::CellSources> a, std::span<const Solver::CellSources> b)
{
    if (a.size() != b.size())
        return false;
    for (size_t c = 0; c < a.size(); ++c)
    {
        if (a[c].species_moles != b[c].species_moles || a[c].internal_energy != b[c].internal_energy ||
            a[c].population_number_changes != b[c].population_number_changes ||
            a[c].qbmm_shape_changes != b[c].qbmm_shape_changes || a[c].impulse_parts != b[c].impulse_parts)
            return false;
        for (size_t axis = 0; axis < 3; ++axis)
            if (a[c].momentum.component(axis) != b[c].momentum.component(axis))
                return false;
    }
    return true;
}
// Preallocate both object storage and its shared control block inside a local fence.
// Moved constructor inputs then enter the adapter's own communicating constructor without a rank-local allocation.
struct FaceStageStorage
{
    using Adapter = Faces::LowMachFaceTransportAdapter;
    alignas(Adapter) std::byte memory[sizeof(Adapter)];
    Adapter* value = nullptr;
    ~FaceStageStorage()
    {
        if (value)
            value->~Adapter();
    }
};

#include "solvers/details/LowMachQbmmConfiguration.inc"

struct SourceEngine
{
    std::shared_ptr<Solver> flow;
    ReactingMixtureOptions options;
    std::vector<std::shared_ptr<const Faces::LowMachFaceTransportAdapter>> face_stages;
    std::shared_ptr<const InterphaseConfiguration> thermal_configuration;
    struct QbmmStage
    {
        std::vector<std::vector<std::optional<QbmmQuadrature>>> pre_material;
        Solver::QbmmSourceCells cells;
        std::vector<std::vector<double>> number_changes;
    };
    struct Prepared
    {
        Solver::SourceStageResult result;
        std::vector<SpeciesSourceLedger> ledgers;
        std::vector<std::shared_ptr<const Faces::Receipt>> face_receipts;
        std::vector<CellSpeciesState> post_faces;
        std::vector<MixturePopulationEventReceipt> population_events;
    };
    CellSpeciesState species(const Solver::CellState& cell) const
    {
        const auto& catalog = options.mechanism->catalog();
        CellSpeciesState result(catalog);
        for (size_t p = 0; p < options.phases.size(); ++p)
        {
            const auto& binding = options.phases[p];
            if (binding.phase == SpeciesPhase::Dispersed && !binding.population.empty())
                result.dispersedPopulations.push_back(
                    {binding.population, std::vector<double>(catalog.species().size())});
            require(
                cell.phases[p].moles.size() == catalog.species().size(), "Phase composition has wrong species size.");
            for (size_t i = 0; i < catalog.species().size(); ++i)
                result.set_amount(catalog, catalog.species()[i].id, compartment(binding, i), cell.phases[p].moles[i],
                    binding.population);
        }
        result.validate(catalog);
        return result;
    }

    size_t phase_for(SpeciesId species_id, SpeciesPhase phase, const std::string& population) const
    {
        const auto i = options.mechanism->catalog().index(species_id);
        for (size_t p = 0; p < options.phases.size(); ++p)
            if (compartment(options.phases[p], i) == phase && options.phases[p].population == population)
                return p;
        throw std::invalid_argument("External transfer references an unbound physical compartment.");
    }

    bool qbmm_bound(size_t phase) const
    {
        const auto& families = flow->options().qbmm_families;
        return !families.empty() && bool(families.at(phase));
    }

    static QbmmCellView qbmm_view(const QbmmPopulationSchema& family, const CellSpeciesState& state,
        const Solver::TransportPhaseInventory& before)
    {
        const auto row =
            std::ranges::find(state.dispersedPopulations, family.population(), &SpeciesPopulationState::id);
        require(row != state.dispersedPopulations.end() && before.population_number && before.qbmm_shape,
            "Transported QBMM family is missing material, count or shape.");
        return {family.population(), state.schemaFingerprint, row->amounts, *before.population_number,
            *before.qbmm_shape, family.fingerprint()};
    }

    void prepare_chemistry(
        const Solver::SourceStageContext& context, Faces::LocalChemistryStage& stage, QbmmStage& qbmm) const
    {
        const auto& catalog = options.mechanism->catalog();
        const auto& transport = context.transport;
        const auto& trial = context.thermodynamic_trial;
        const double dt = transport.time_step;
        const auto ns = catalog.species().size(), np = options.phases.size();
        require(
            transport.cells.size() == trial.cells.size() && transport.cells.size() == flow->mesh()->num_owned_cells(),
            "Transport/source cell counts do not agree.");
        stage.sums.reset(transport.cells.size(), np, ns);
        stage.gross_events.resize(transport.cells.size());
        stage.ledgers.reserve(transport.cells.size());
        stage.post_chemistry.reserve(transport.cells.size());
        stage.post_chemistry_parts.reserve(transport.cells.size());
        stage.caloric_guesses.reserve(transport.cells.size());
        for (size_t c = 0; c < transport.cells.size(); ++c)
        {
            auto cell = trial.cells.at(c);
            require(transport.cells[c].phases.size() == np && cell.phases.size() == np,
                "Transport/source physical phase counts do not agree.");
            cell.phase_numbers.resize(np);
            const bool kinetic = uses_qbmm_kinetic_transport(flow->options().qbmm_model);
            std::vector<double> kinetic_volumes(kinetic ? np : 0);
            for (size_t p = 0; p < np; ++p)
            {
                cell.phase_numbers[p] = transport.cells[c].phases[p].population_number;
                if (qbmm_bound(p))
                    cell.qbmm_shape.at(p) = transport.cells[c].phases[p].qbmm_shape;
                auto& phase = cell.phases[p];
                phase.moles = transport.cells[c].phases[p].moles;
                Sum moles, phase_mass;
                for (size_t i = 0; i < ns; ++i)
                {
                    moles += phase.moles.at(i);
                    phase_mass += static_cast<long double>(phase.moles[i]) * catalog.species()[i].molarMass;
                }
                if (moles.value() > 0)
                {
                    // Rates see transported composition at the current thermal guess.
                    // Recompute the EOS energy; transported U is exposed separately.
                    std::vector<double> x(ns);
                    for (size_t i = 0; i < ns; ++i)
                        x[i] = double(phase.moles[i] / moles.value());
                    if (!(phase.temperature > 0))
                    {
                        if (flow->options().thermal_mode == PhaseThermalMode::SeparatePhaseTemperatures)
                        {
                            auto arriving = phase;
                            arriving.internal_energy = transport.cells[c].phases[p].internal_energy;
                            phase.temperature = recover_phase_temperatures(std::vector{arriving},
                                arriving.internal_energy, PhaseThermalMode::SeparatePhaseTemperatures)
                                                    .phases[0]
                                                    .temperature;
                        }
                        else
                            for (const auto& other : trial.cells[c].phases)
                                if (other.temperature > 0)
                                {
                                    phase.temperature = other.temperature;
                                    break;
                                }
                    }
                    const auto point = phase.thermodynamics->evaluate(phase.temperature, phase.absolute_pressure, x);
                    phase.internal_energy = double(phase_mass.value() * point.internal_energy);
                    if (kinetic)
                        kinetic_volumes[p] = static_cast<double>(phase_mass.value() / point.density);
                    require(std::isfinite(phase.internal_energy), "Rate-state phase energy overflow.");
                }
                else
                {
                    phase.temperature = 0;
                    phase.internal_energy = 0;
                }
            }
            if (kinetic)
                qbmm_native_law_detail::evaluate_kinetic_closure(flow->options(), cell,
                    transport.new_control_volumes.at(c), kinetic_volumes, source_location(context, c));
            const auto initial = species(cell);
            for (size_t p = 0; p < np; ++p)
                if (qbmm_bound(p))
                {
                    const auto& family = *flow->options().qbmm_families[p];
                    auto quadrature = invert_qbmm_moments(
                        qbmm_cell_moments(family, qbmm_view(family, initial, transport.cells[c].phases[p])));
                    require(quadrature.valid(), "Transported QBMM family has no admissible pre-material quadrature.");
                    qbmm.pre_material[c][p] = std::move(quadrature);
                }
            const auto& liquid = cell.phases.at(options.liquid_phase);
            Sum amount, mass;
            for (size_t i = 0; i < ns; ++i)
            {
                amount += liquid.moles[i];
                mass += static_cast<long double>(liquid.moles[i]) * catalog.species()[i].molarMass;
            }
            double liquid_volume = 0;
            if (amount.value() > 0)
            {
                std::vector<double> fractions(ns);
                for (size_t i = 0; i < ns; ++i)
                    fractions[i] = double(liquid.moles[i] / amount.value());
                liquid_volume = double(
                    mass.value() /
                    liquid.thermodynamics->evaluate(liquid.temperature, liquid.absolute_pressure, fractions).density);
            }
            double temperature = liquid.temperature;
            if (!(temperature > 0))
                for (const auto& phase : cell.phases)
                    if (phase.temperature > 0)
                    {
                        temperature = phase.temperature;
                        break;
                    }
            ReactionEnvironment environment{transport.accepted_time, temperature, liquid.absolute_pressure,
                liquid_volume, transport.new_control_volumes.at(c), ReactionVolumeClosure::ThermodynamicPredictor};
            environment.location = source_location(context, c);

            ReactionContext reaction{catalog, initial, environment};
            const auto input =
                options.source ? options.source({reaction, cell, c, transport.cells[c]}, dt) : ReactionSourceInput{};
            if (!flow->options().qbmm_families.empty())
                for (const auto& transfer : input.externalTransfers)
                    require(!qbmm_bound(phase_for(transfer.species, transfer.phase, transfer.population)),
                        "Chemistry reservoir transfers cannot touch a QBMM family.");
            auto chemistry =
                advance_reactions(*options.mechanism, initial, environment, dt, input, options.integration);
            // Retain actual pure-kernel state for the first face stage, not an endpoint reconstruction.
            stage.post_chemistry.push_back(chemistry.state);
            auto caloric = cell;
            for (size_t p = 0; p < np; ++p)
            {
                // Preserve an available current guess for a newly created phase whose transported stock was empty.
                // Arriving transported phases already recovered their own T from transported U above.
                if (!(caloric.phases[p].temperature > 0))
                {
                    const double guess = trial.cells[c].phases[p].temperature;
                    if (guess > 0)
                        caloric.phases[p].temperature = guess;
                    else if (flow->options().thermal_mode == PhaseThermalMode::Equilibrium)
                        caloric.phases[p].temperature = temperature;
                }
            }
            stage.caloric_guesses.push_back(std::move(caloric));
            const auto& raw = chemistry.localReceipt;
            require(raw.schemaFingerprint == catalog.fingerprint() &&
                        raw.reactionHeatContributions.size() == options.mechanism->reactions().size(),
                "Local reaction receipt does not match the immutable mechanism/schema.");
            stage.post_chemistry_parts.push_back(raw.postSourceAmounts);
            for (const auto& change : raw.chemicalChanges)
            {
                // The receipt is sparse; an unbound zero compartment never reaches phase_for.
                if (change.moles != 0)
                    stage.sums.add_material(c, phase_for(change.species, change.phase, change.population),
                        catalog.index(change.species), change.moles);
            }
            Sum declared_heat;
            for (const auto part : raw.externalHeat)
                declared_heat += part;
            for (const auto& heat : raw.reactionHeatContributions)
                for (const auto part : heat)
                    declared_heat += part;

            if (options.partition_heat)
            {
                // Declare original channels before custom double allocation. Later cancellation must not hide loss.
                for (const auto part : raw.externalHeat)
                    stage.sums.declare_energy(c, part);
                for (const auto& heat : raw.reactionHeatContributions)
                    for (const auto part : heat)
                        stage.sums.declare_energy(c, part);
                const double heat = static_cast<double>(declared_heat.value());
                require(std::isfinite(heat) && (declared_heat.value() == 0 || heat != 0),
                    "Heat allocation input is not representable.");
                const auto parts = options.partition_heat(heat, chemistry.ledger, cell);
                require(parts.size() == np, "Heat allocation must match physical phases.");
                Sum allocated, scale;
                for (size_t p = 0; p < np; ++p)
                {
                    require(std::isfinite(parts[p]), "Phase heat allocation is nonfinite.");
                    stage.sums.add_allocated_energy(c, p, parts[p]);
                    allocated += parts[p];
                    scale += std::abs(static_cast<long double>(parts[p]));
                }
                require(
                    std::abs(allocated.value() - declared_heat.value()) <=
                        64 * std::numeric_limits<double>::epsilon() * (scale.value() + std::abs(declared_heat.value())),
                    "Phase heat allocation does not conserve its original budget.");
            }
            else
            {
                for (const auto part : raw.externalHeat)
                    stage.sums.add_energy(c, options.heat_phase, part);
                for (const auto& heat : raw.reactionHeatContributions)
                    for (const auto part : heat)
                        stage.sums.add_energy(c, options.heat_phase, part);
            }
            for (const auto& transfer : chemistry.ledger.externalTransfers)
            {
                const auto p = phase_for(transfer.species, transfer.phase, transfer.population);
                stage.sums.add_material(c, p, catalog.index(transfer.species), transfer.moles);
                stage.sums.add_energy(c, p, transfer.energy);
                for (size_t axis = 0; axis < 3; ++axis)
                    stage.sums.add_momentum(c, axis, transfer.momentum[axis]);
            }
            stage.gross_events[c] = gross_events(chemistry.ledger);
            stage.ledgers.push_back(std::move(chemistry.ledger));
        }
    }

    Solver::CellState material_guess(const Solver::CellState& guess, const Solver::TransportCellInventory& transported,
        const CellSpeciesState& material) const
    {
        auto result = guess;
        const auto& catalog = options.mechanism->catalog();
        result.phase_numbers.resize(options.phases.size());
        for (size_t p = 0; p < options.phases.size(); ++p)
        {
            result.phase_numbers[p] = transported.phases[p].population_number;
            if (qbmm_bound(p))
                result.qbmm_shape.at(p) = transported.phases[p].qbmm_shape;
            auto& phase = result.phases.at(p);
            for (size_t i = 0; i < catalog.species().size(); ++i)
                phase.moles.at(i) = material.amount(
                    catalog, catalog.species()[i].id, compartment(options.phases[p], i), options.phases[p].population);
            if (std::ranges::none_of(phase.moles, [](double n) { return n > 0; }))
            {
                phase.temperature = 0;
                phase.internal_energy = 0;
            }
        }
        return result;
    }

    std::vector<double> qbmm_phase_volumes(Solver::CellState& guess) const
    {
        const auto& catalog = options.mechanism->catalog();
        const auto available =
            std::ranges::find_if(guess.phases, [](const auto& phase) { return phase.temperature > 0; });
        std::vector<double> volumes(guess.phases.size());
        for (size_t p = 0; p < guess.phases.size(); ++p)
        {
            auto& phase = guess.phases[p];
            Sum moles, mass;
            for (size_t i = 0; i < phase.moles.size(); ++i)
            {
                moles += phase.moles[i];
                mass += static_cast<long double>(phase.moles[i]) * catalog.species()[i].molarMass;
            }
            if (moles.value() == 0)
                continue;
            if (!(phase.temperature > 0))
            {
                require(available != guess.phases.end(), "Occupied QBMM rate-state phase needs a temperature guess.");
                phase.temperature = available->temperature;
            }
            auto fractions = phase.moles;
            for (auto& amount : fractions)
                amount = static_cast<double>(amount / moles.value());
            const auto point = phase.thermodynamics->evaluate(phase.temperature, phase.absolute_pressure, fractions);
            volumes[p] = static_cast<double>(mass.value() / point.density);
            phase.internal_energy = static_cast<double>(mass.value() * point.internal_energy);
            require(std::isfinite(volumes[p]) && volumes[p] > 0 && std::isfinite(phase.internal_energy),
                "QBMM rate-state phase volume or energy is not representable.");
        }
        return volumes;
    }

    void refresh_kinetic_source_closure(
        const Solver::SourceStageContext& context, size_t c, Solver::CellState& guess) const
    {
        if (uses_qbmm_kinetic_transport(flow->options().qbmm_model))
        {
            // Material stages own their original thermal operands. Evaluate slip using the EOS
            // trace of this stage's actual material and publish just the closure to its callback.
            auto rate_state = guess;
            const auto volumes = qbmm_phase_volumes(rate_state);
            qbmm_native_law_detail::evaluate_kinetic_closure(flow->options(), rate_state,
                context.transport.new_control_volumes.at(c), volumes, source_location(context, c));
            guess.qbmm_kinetic_closure = std::move(rate_state.qbmm_kinetic_closure);
            guess.qbmm_kinetic_eos_volumes = std::move(rate_state.qbmm_kinetic_eos_volumes);
        }
    }

    void add_material(size_t c, size_t p, size_t s, double amount, std::vector<Faces::MolarTerm>& terms,
        Faces::LocalSourceAccumulator& sums) const
    {
        const auto& binding = options.phases[p];
        terms.push_back(
            {options.mechanism->catalog().species()[s].id, {compartment(binding, s), binding.population}, amount});
        sums.add_material(c, p, s, amount);
    }

    FrozenMaterialSourceTail apply_qbmm(const Solver::SourceStageContext& context, size_t c,
        const Solver::CellState& caloric_guess, CellSpeciesState& material, Faces::RawAmounts& raw_material,
        Faces::LocalSourceAccumulator& sums, QbmmStage& qbmm) const
    {
        const auto& transport = context.transport;
        const auto& catalog = options.mechanism->catalog();
        const auto& families = flow->options().qbmm_families;
        const auto& transported = transport.cells[c];
        Faces::validate_raw_inventory(catalog, material, raw_material);
        FrozenMaterialSourceTail tail;
        const std::vector<MaterialTransferSlot>* slots = nullptr;
        if (options.qbmm_population && options.qbmm_population->transfers)
        {
            require(thermal_configuration && !thermal_configuration->stages.empty(),
                "QBMM paired material is missing its frozen thermal configuration.");
            tail.paired_stage = thermal_configuration->stages.size() - 1;
            slots = &thermal_configuration->stages.back().material;
            tail.paired_amounts.resize(slots->size());
        }
        Solver::CellState guess;
        std::vector<double> phase_volumes;
        std::vector<Faces::MolarTerm> terms;
        for (size_t p = 0; p < families.size(); ++p)
        {
            if (!families[p])
                continue;
            const auto& family = *families[p];
            const auto& before = transported.phases[p];
            const auto pre = qbmm_view(family, material, before);
            require(std::ranges::equal(pre.species_moles, before.moles),
                "An upstream material stage changed a QBMM family without a paired allocation.");
            for (size_t s = 0; s < before.moles.size(); ++s)
            {
                const auto slot = Faces::raw_amount_slot(
                    catalog, material, catalog.species()[s].id, {SpeciesPhase::Dispersed, family.population()});
                const auto& parts = raw_material.at(slot);
                Sum change;
                change += parts[0];
                change += parts[1];
                change += -static_cast<long double>(before.moles[s]);
                require(change.value() == 0, "An upstream raw material stage changed a QBMM family.");
            }

            QbmmEventBatch batch{family.fingerprint(), 0, transport.time_step, {}};
            if (options.qbmm_population)
            {
                const auto& provider = *options.qbmm_population;
                const auto& quadrature = *qbmm.pre_material[c][p];
                const bool occupied = quadrature.status != InversionStatus::Absent;
                const auto in_scope = [&](const QbmmLawMetadata& law)
                { return !qbmm_is_transport_law(law) && (occupied || !law.requires_population); };
                const auto& declared = family.closure_versions().laws;
                const bool scoped =
                    std::ranges::any_of(declared, [](const auto& law) { return !qbmm_is_transport_law(law); });
                const bool integrated = provider.evaluate && (!scoped || std::ranges::any_of(declared, in_scope));
                const auto selected = [&](const auto& law)
                { return in_scope(law->metadata) && family.declares_law(law->metadata); };
                if (integrated || std::ranges::any_of(provider.laws, selected))
                {
                    if (phase_volumes.empty())
                    {
                        guess = material_guess(caloric_guess, transported, material);
                        phase_volumes = qbmm_phase_volumes(guess);
                        if (uses_qbmm_kinetic_transport(flow->options().qbmm_model))
                            qbmm_native_law_detail::evaluate_kinetic_closure(flow->options(), guess,
                                transport.new_control_volumes.at(c), phase_volumes, source_location(context, c));
                    }
                    const double bulk_volume = transport.new_control_volumes.at(c);
                    std::optional<double> specific_volume, temperature;
                    if (occupied)
                    {
                        const long double mass =
                            static_cast<long double>(family.scaled_mass(pre.species_moles)) * family.mass_scale_kg();
                        specific_volume = population_source_detail::narrow(phase_volumes[p] / mass);
                        temperature = guess.phases[p].temperature;
                    }
                    double gas_volume = 0;
                    for (size_t other = 0; other < phase_volumes.size(); ++other)
                        if (guess.phases[other].thermodynamics->kind() == EOSKind::Gas)
                            gas_volume += phase_volumes[other];
                    const auto drift = qbmm_native_law_detail::mean_drift_velocity(guess, p);
                    const auto inputs = make_qbmm_law_inputs(family,
                        {quadrature, qbmm_geometry(family, pre, bulk_volume, specific_volume), pre.species_moles,
                            bulk_volume, phase_volumes[p], guess.phases[p].absolute_pressure, temperature,
                            transport.time_step, gas_volume / bulk_volume,
                            qbmm_phase_concentrations(
                                guess.phases[options.liquid_phase].moles, phase_volumes[options.liquid_phase]),
                            phase_volumes[options.liquid_phase] > 0
                                ? std::optional<double>(guess.phases[options.liquid_phase].temperature)
                                : std::nullopt,
                            {drift.component(0), drift.component(1), drift.component(2)}, source_location(context, c)});
                    for (const auto& metadata : declared)
                        if (in_scope(metadata) && metadata.gate == QbmmApplicabilityGate::RejectOutside &&
                            !evaluate_qbmm_applicability(metadata, inputs).gate_passed)
                            throw std::invalid_argument(
                                "QBMM family '" + family.population() + "' law '" + metadata.identity +
                                "' is outside its declared applicability or lacks a required input.");
                    if (integrated)
                    {
                        auto operations = provider.evaluate({family, p, c, quadrature, material, phase_volumes,
                            guess.phases, transport.stage, transport.time_step, transport.accepted_time, &inputs});
                        for (auto& operation : operations)
                            batch.events.push_back(
                                {provider.metadata.identity, provider.metadata.revision, std::move(operation)});
                    }
                    for (const auto& law : provider.laws)
                    {
                        if (!selected(law))
                            continue;
                        for (auto& operation : law->evaluate(inputs))
                            batch.events.push_back(
                                {law->metadata.identity, law->metadata.revision, std::move(operation)});
                    }
                }
            }
            auto advance = qbmm_event_detail::advance_with_raw_species(family, material, pre, batch, raw_material);
            if (!advance.endpoint)
                throw std::invalid_argument(
                    "QBMM population event rejected for " + family.population() + ". " + advance.diagnostic);
            auto& endpoint = *advance.endpoint;
            for (const auto& event : endpoint.receipts)
                for (const auto& receipt : event.transfers)
                {
                    const auto& transfer = receipt.transfer;
                    const auto* counterpart = std::get_if<QbmmPhaseExchange>(&transfer.counterpart);
                    std::vector<double> external;
                    if (!counterpart)
                        external.resize(transfer.moles.size());
                    for (size_t s = 0; s < transfer.moles.size(); ++s)
                    {
                        const auto amount = transfer.moles[s];
                        if (amount == 0)
                            continue;
                        const double family_amount = receipt.into_family ? amount : -amount;
                        add_material(c, p, s, family_amount, terms, sums);
                        if (!counterpart)
                        {
                            external[s] = family_amount;
                            continue;
                        }
                        require(slots, "QBMM paired material requires configured partial enthalpy transfer slots.");
                        const auto id = catalog.species()[s].id;
                        const auto other = phase_for(id, counterpart->phase, {});
                        require(
                            !qbmm_bound(other), "QBMM material counterparts must be unnamed physical compartments.");
                        const MaterialTransferSlot key{
                            id, receipt.into_family ? other : p, receipt.into_family ? p : other};
                        const auto i = std::ranges::find(*slots, key);
                        require(i != slots->end(), "QBMM paired material does not match a configured transfer slot.");
                        tail.paired_amounts[i - slots->begin()].push_back(amount);
                        add_material(c, other, s, -family_amount, terms, sums);
                    }
                    if (!counterpart)
                    {
                        const auto& reservoir = std::get<QbmmReservoirExchange>(transfer.counterpart);
                        tail.reservoirs.push_back({p, std::move(external), reservoir.energy_j});
                        sums.add_energy(c, p, reservoir.energy_j);
                        for (size_t axis = 0; axis < 3; ++axis)
                            sums.add_momentum(c, axis, reservoir.momentum[axis]);
                    }
                }
            qbmm.number_changes[c][p] = population_source_detail::narrow(endpoint.moment_change[0]);
            const auto final = std::ranges::find(
                endpoint.species.dispersedPopulations, family.population(), &SpeciesPopulationState::id);
            require(final != endpoint.species.dispersedPopulations.end(), "Missing constructive QBMM family endpoint.");
            qbmm.cells[c][p] =
                Solver::QbmmSourceReceipt{std::move(endpoint.receipts), endpoint.count, std::move(final->amounts),
                    endpoint.shape, family.fingerprint(), {endpoint.moment_change[2], endpoint.moment_change[3]}};
        }
        // Replay all families together against one uncollapsed predecessor. No family can borrow another's credits.
        material = Faces::apply_raw_material(catalog, material, raw_material, terms);
        for (size_t p = 0; p < families.size(); ++p)
            if (qbmm.cells[c][p])
            {
                const auto pre = qbmm_view(*families[p], material, transported.phases[p]);
                const auto& expected = qbmm.cells[c][p]->final_moles;
                for (size_t s = 0; s < expected.size(); ++s)
                    Faces::validate_source_endpoint(0, pre.species_moles[s], expected[s]);
            }
        return tail;
    }

    void prepare_material_stages(const Solver::SourceStageContext& context, std::vector<CellSpeciesState>& donors,
        std::vector<Faces::RawAmounts>& raw_donors, Faces::LocalSourceAccumulator& sums,
        std::vector<std::vector<double>>& gross, std::span<const Solver::CellState> guesses,
        PressureConditionedThermalBatch* thermal, QbmmStage& qbmm) const
    {
        const auto& catalog = options.mechanism->catalog();
        const auto np = options.phases.size(), ns = catalog.species().size();
        require(donors.size() == context.transport.cells.size() && guesses.size() == donors.size() &&
                    raw_donors.size() == donors.size(),
            "Interphase actual predecessor rows differ.");
        const bool has_qbmm = !flow->options().qbmm_families.empty();
        for (size_t c = 0; c < donors.size(); ++c)
        {
            Solver::CellState guess;
            std::vector<std::vector<std::array<long double, 2>>> phase_parts;
            std::vector<std::vector<long double>> raw_energy;
            std::vector<long double> raw_declared;
            if (thermal)
            {
                guess = material_guess(guesses[c], context.transport.cells[c], donors[c]);
                phase_parts.assign(np, std::vector<std::array<long double, 2>>(ns));
                for (size_t p = 0; p < np; ++p)
                    for (size_t s = 0; s < ns; ++s)
                        phase_parts[p][s] =
                            raw_donors[c].at(Faces::raw_amount_slot(catalog, donors[c], catalog.species()[s].id,
                                {compartment(options.phases[p], s), options.phases[p].population}));
                raw_energy = sums.raw_energy(c);
                raw_declared = sums.raw_declared(c);
            }
            FrozenInterphasePlan generic;
            if (options.interphase)
            {
                refresh_kinetic_source_closure(context, c, guess);
                auto amounts = options.interphase->integrate(
                    {catalog, donors[c], context.transport.cells[c], guess, context.transport.stage, c,
                        context.transport.time_step, context.transport.accepted_time, source_location(context, c)});
                generic = compile_interphase_plan(options.interphase->configuration, guess.phases, std::move(amounts),
                    raw_energy, raw_declared, phase_parts);
                for (size_t j = 0; j < generic.amounts.size(); ++j)
                {
                    const auto& stage = generic.configuration->stages[j];
                    if (stage.kind == InterphaseStageKind::Material)
                    {
                        std::vector<Faces::MolarTerm> terms;
                        for (size_t i = 0; i < generic.amounts[j].size(); ++i)
                        {
                            const double amount = generic.amounts[j][i];
                            const auto& slot = stage.material[i];
                            const auto species = catalog.index(slot.species);
                            add_material(c, slot.donor, species, -amount, terms, sums);
                            add_material(c, slot.recipient, species, amount, terms, sums);
                            gross[c].push_back(amount);
                        }
                        donors[c] = Faces::apply_raw_material(catalog, donors[c], raw_donors[c], terms);
                    }
                    else
                        for (double amount : generic.amounts[j])
                        {
                            gross[c].push_back(std::max(amount, 0.));
                            gross[c].push_back(std::max(-amount, 0.));
                        }
                }
                for (size_t p = 0; p < np; ++p)
                    for (size_t s = 0; s < ns; ++s)
                        require(donors[c].amount(catalog, catalog.species()[s].id, compartment(options.phases[p], s),
                                    options.phases[p].population) == generic.final_moles[p][s],
                            "Raw interphase replay differs from its frozen material plan.");
            }
            if (!has_qbmm)
            {
                thermal->cells.push_back(std::move(generic));
                continue;
            }
            auto tail = apply_qbmm(context, c, guesses[c], donors[c], raw_donors[c], sums, qbmm);
            if (!thermal)
                continue;
            if (tail.paired_stage)
            {
                auto& paired = generic.amounts.emplace_back();
                for (const auto& terms : tail.paired_amounts)
                {
                    Sum amount;
                    for (double term : terms)
                        amount += term;
                    paired.push_back(population_source_detail::narrow(amount.value()));
                }
            }
            auto plan = compile_interphase_plan(thermal->configuration, guess.phases, std::move(generic.amounts),
                std::move(raw_energy), std::move(raw_declared), phase_parts, std::move(tail));
            for (size_t p = 0; p < np; ++p)
                for (size_t s = 0; s < ns; ++s)
                    Faces::validate_source_endpoint(0,
                        donors[c].amount(catalog, catalog.species()[s].id, compartment(options.phases[p], s),
                            options.phases[p].population),
                        plan.final_moles[p][s]);
            thermal->cells.push_back(std::move(plan));
        }
    }

    Prepared prepare(const Solver::SourceStageContext& context) const
    {
        std::vector<const Faces::LowMachFaceTransportAdapter*> ordered;
        std::function<void(const Solver::SourceStageContext&, Faces::LocalChemistryStage&)> local_prepare;
        Faces::PostFaceLocalStage interphase_stage;
        std::shared_ptr<PressureConditionedThermalBatch> thermal_batch;
        QbmmStage qbmm;
        collective_detail::collective_local_validation(context.mesh, "Ordered face-stage views",
            [&]
            {
                const auto np = flow->options().qbmm_families.size(), nc = context.transport.cells.size();
                if (np)
                {
                    qbmm.pre_material.resize(nc, std::vector<std::optional<QbmmQuadrature>>(np));
                    qbmm.cells.resize(nc, std::vector<std::optional<Solver::QbmmSourceReceipt>>(np));
                    qbmm.number_changes.resize(nc, std::vector<double>(np));
                }
                local_prepare = [this, &qbmm](const Solver::SourceStageContext& current,
                                    Faces::LocalChemistryStage& local) { prepare_chemistry(current, local, qbmm); };
                if (thermal_configuration)
                {
                    thermal_batch = std::make_shared<PressureConditionedThermalBatch>();
                    thermal_batch->configuration = thermal_configuration;
                    thermal_batch->source = conditioned_thermal_detail::identity(context.transport);
                    thermal_batch->cells.reserve(nc);
                }
                if (thermal_batch || np || options.body_acceleration)
                    interphase_stage =
                        [this, thermal = thermal_batch.get(), np, &qbmm](const Solver::SourceStageContext& current,
                            std::vector<CellSpeciesState>& donors, std::vector<Faces::RawAmounts>& raw_donors,
                            Faces::LocalSourceAccumulator& sums, std::vector<std::vector<double>>& gross,
                            std::span<const Solver::CellState> guesses)
                    {
                        if (thermal || np)
                            prepare_material_stages(current, donors, raw_donors, sums, gross, guesses, thermal, qbmm);
                        if (options.body_acceleration)
                            add_body_impulse(current.transport.time_step, donors, sums, gross);
                    };
                ordered.reserve(face_stages.size());
                for (const auto& stage : face_stages)
                {
                    require(bool(stage), "Null immutable face-stage adapter.");
                    ordered.push_back(stage.get());
                }
            });
        auto composed = Faces::compose_post_chemistry_faces(
            context, options.mechanism->catalog(), ordered, local_prepare, interphase_stage);
        Prepared result;
        collective_detail::collective_local_validation(context.mesh, "Immutable composed source receipt",
            [&]
            {
                result.result = std::move(composed.result);
                result.result.gross_channel_metadata = gross_event_channels();
                result.result.thermal_receipt = std::move(thermal_batch);
                result.result.qbmm_cells = std::move(qbmm.cells);
                result.ledgers = std::move(composed.chemistry_ledgers);
                result.face_receipts = std::move(composed.faces);
                for (const auto& face : result.face_receipts)
                    if (face->fourier_thermal)
                        result.result.fourier_thermal_stages.push_back(face);
                result.post_faces = std::move(composed.post_faces);
                result.population_events.reserve(result.post_faces.size());
                const auto& roles = flow->options().phase_roles;
                const auto& population_catalog = options.mechanism->catalog();
                const bool has_qbmm = !flow->options().qbmm_families.empty();
                for (size_t c = 0; c < result.post_faces.size(); ++c)
                {
                    const auto& transported = context.transport.cells[c];
                    auto guess =
                        material_guess(context.thermodynamic_trial.cells.at(c), transported, result.post_faces[c]);
                    std::vector<int> occupied(roles.size(), 0);
                    auto& source = result.result.sources.at(c);
                    if (has_qbmm)
                        source.qbmm_shape_changes.resize(roles.size());
                    for (size_t p = 0; p < roles.size(); ++p)
                    {
                        occupied[p] = std::ranges::any_of(guess.phases[p].moles, [](double n) { return n > 0; });
                        if (qbmm_bound(p))
                        {
                            const auto& receipt = *result.result.qbmm_cells[c][p];
                            const auto& before = *transported.phases[p].qbmm_shape;
                            // Canonical endpoint differences include the Q1 reconstruction residual. Raw event
                            // deltas stay with the original receipts for the native replay and convergence checks.
                            const QbmmShapeState delta{
                                population_source_detail::narrow(
                                    static_cast<long double>(receipt.final_shape.H2) - before.H2),
                                population_source_detail::narrow(
                                    static_cast<long double>(receipt.final_shape.H3) - before.H3)};
                            Faces::validate_source_endpoint(before.H2, delta.H2, receipt.final_shape.H2);
                            Faces::validate_source_endpoint(before.H3, delta.H3, receipt.final_shape.H3);
                            source.qbmm_shape_changes[p] = delta;
                            guess.phase_numbers[p] = receipt.final_count;
                            guess.qbmm_shape[p] = receipt.final_shape;
                        }
                    }
                    // The legacy count law sees final material and the actual count inventory at this stage.
                    std::vector<BubbleGasPhysics::PopulationNumberEvent> events;
                    if (options.population_events)
                    {
                        refresh_kinetic_source_closure(context, c, guess);
                        events = options.population_events->evaluate({population_catalog, result.post_faces[c],
                            transported, guess, context.transport.stage, c, context.transport.time_step});
                        const auto& declared = options.population_events->populations;
                        for (const auto& event : events)
                        {
                            require(declared.empty() || std::ranges::find(declared, event.population) != declared.end(),
                                "Count-only population event is outside its immutable declared scope.");
                            const auto phase =
                                std::ranges::find(roles, event.population, &Solver::PhaseRoleDescriptor::identity);
                            require(phase == roles.end() || !qbmm_bound(phase - roles.begin()),
                                "Count-only population event cannot change a QBMM family.");
                        }
                    }
                    auto number =
                        population_source_detail::apply(roles, guess.phase_numbers, occupied, std::move(events));
                    number.source_stage = context.transport.stage;
                    number.schema_fingerprint = population_catalog.fingerprint();
                    if (options.population_events)
                    {
                        number.provider_identity = options.population_events->identity;
                        number.provider_revision = options.population_events->revision;
                    }
                    source.population_number_changes = number.changes;
                    for (size_t p = 0; p < roles.size(); ++p)
                    {
                        result.result.gross_events.at(c).push_back(number.births[p]);
                        result.result.gross_events.at(c).push_back(number.deaths[p]);
                        if (!qbmm_bound(p))
                            continue;
                        const auto& receipt = *result.result.qbmm_cells[c][p];
                        source.population_number_changes[p] = qbmm.number_changes[c][p];
                        Faces::validate_source_endpoint(*transported.phases[p].population_number,
                            source.population_number_changes[p], receipt.final_count);
                    }
                    result.population_events.push_back(std::move(number));
                }
                // A single signed double packet must replay the actual sequential chemistry/face endpoint.
                // In particular, a large depletion cannot erase a later finite incoming face credit.
                const auto& catalog = options.mechanism->catalog();
                require(result.post_faces.size() == context.transport.cells.size() &&
                            result.result.sources.size() == result.post_faces.size(),
                    "Final source endpoint row mismatch.");
                for (size_t c = 0; c < result.post_faces.size(); ++c)
                    for (size_t p = 0; p < options.phases.size(); ++p)
                        for (size_t s = 0; s < catalog.species().size(); ++s)
                        {
                            const auto actual = result.post_faces[c].amount(catalog, catalog.species()[s].id,
                                compartment(options.phases[p], s), options.phases[p].population);
                            Faces::validate_source_endpoint(context.transport.cells[c].phases[p].moles.at(s),
                                result.result.sources[c].species_moles.at(p).at(s), actual);
                        }
            });
        return result;
    }

    void add_body_impulse(double dt, const std::vector<CellSpeciesState>& endpoints,
        Faces::LocalSourceAccumulator& sums, std::vector<std::vector<double>>& gross) const
    {
        const auto& catalog = options.mechanism->catalog();
        for (size_t cell = 0; cell < endpoints.size(); ++cell)
        {
            Sum mass;
            for (const auto& phase : options.phases)
                for (size_t species = 0; species < catalog.species().size(); ++species)
                {
                    const auto& descriptor = catalog.species()[species];
                    const auto moles =
                        endpoints[cell].amount(catalog, descriptor.id, compartment(phase, species), phase.population);
                    mass += static_cast<long double>(moles) * descriptor.molarMass;
                }
            for (size_t axis = 0; axis < 3; ++axis)
            {
                const auto impulse = mass.value() * options.body_acceleration->component(axis) * dt;
                sums.add_body_momentum(cell, axis, impulse);
                gross[cell].push_back(population_source_detail::narrow(std::abs(impulse)));
            }
        }
    }

    void validate_face_receipts(const Prepared& prepared, const Solver::SourceStageContext& context) const
    {
        collective_detail::collective_local_validation(context.mesh, "Ordered face receipt count",
            [&]
            {
                require(prepared.face_receipts.size() == face_stages.size(), "Face receipt stage count changed.");
                for (const auto& face : prepared.face_receipts)
                    require(bool(face), "Missing immutable face receipt.");
            });
        for (size_t stage = 0; stage < face_stages.size(); ++stage)
            face_stages[stage]->validate_receipt(*prepared.face_receipts[stage], context);
    }

    std::vector<double> gross_events(const SpeciesSourceLedger& ledger) const
    {
        const auto& catalog = options.mechanism->catalog();
        const size_t nr = options.mechanism->reactions().size(), np = options.phases.size(),
                     ns = catalog.species().size();
        const size_t material = 2 * nr, energy = material + 2 * np * ns, momentum = energy + 2 * np;
        const size_t heat = momentum + 6 * np;
        std::vector<Sum> events(heat + 4);
        for (size_t r = 0; r < nr; ++r)
        {
            events[r] += ledger.forwardReactionExtents.at(r);
            events[nr + r] += ledger.reverseReactionExtents.at(r);
        }
        auto add_signed = [&](size_t offset, double value)
        { events[offset + (value < 0 ? 1 : 0)] += std::abs(static_cast<long double>(value)); };
        for (const auto& transfer : ledger.externalTransfers)
        {
            const auto p = phase_for(transfer.species, transfer.phase, transfer.population);
            add_signed(material + 2 * (p * ns + catalog.index(transfer.species)), transfer.moles);
            add_signed(energy + 2 * p, transfer.energy);
            for (size_t axis = 0; axis < 3; ++axis)
                add_signed(momentum + 6 * p + 2 * axis, transfer.momentum[axis]);
        }
        add_signed(heat, ledger.externalHeat);
        add_signed(heat + 2, ledger.reactionHeat);
        std::vector<double> result(events.size());
        for (size_t i = 0; i < events.size(); ++i)
        {
            result[i] = double(events[i].value());
            require(std::isfinite(result[i]) && result[i] >= 0, "Source event convergence packet overflow.");
        }
        return result;
    }

    std::vector<Solver::GrossChannelMetadata> gross_event_channels() const
    {
        using Stage = Solver::GrossResidualStage;
        const auto np = options.phases.size(), ns = options.mechanism->catalog().species().size();
        const auto nr = options.mechanism->reactions().size();
        std::vector<Solver::GrossChannelMetadata> channels;
        const auto add_pairs = [&](Stage stage, size_t index, size_t count)
        {
            for (size_t i = 0; i < count; ++i)
            {
                channels.push_back({stage, index, 1});
                channels.push_back({stage, index, -1});
            }
        };
        channels.insert(channels.end(), nr, {Stage::Chemistry, 0, 1}); // Forward extents.
        channels.insert(channels.end(), nr, {Stage::Chemistry, 0, -1}); // Reverse extents.
        add_pairs(Stage::Chemistry, 0, np * ns + 4 * np + 2);
        for (size_t i = 0; i < face_stages.size(); ++i)
            add_pairs(Stage::FaceTransport, i, ns + 3);
        if (options.interphase)
        {
            const auto& stages = options.interphase->configuration->stages;
            for (size_t i = 0; i < stages.size(); ++i)
            {
                const auto& stage = stages[i];
                if (stage.kind == InterphaseStageKind::Material)
                    channels.insert(channels.end(), stage.material.size(), {Stage::Interphase, i, 1});
                else
                    add_pairs(Stage::Interphase, i, stage.heat.size());
            }
        }
        if (options.body_acceleration)
            for (size_t axis = 0; axis < 3; ++axis)
            {
                const double value = options.body_acceleration->component(axis);
                channels.push_back({Stage::BodyImpulse, 0, (value > 0) - (value < 0)});
            }
        add_pairs(Stage::PopulationNumbers, 0, np);
        return channels;
    }
};
// Fence allocation before entering the communicating motion constructor.
struct PlanarMotionStorage
{
    alignas(Solver::PlanarMotion) std::byte memory[sizeof(Solver::PlanarMotion)];
    Solver::PlanarMotion* value = nullptr;
    ~PlanarMotionStorage()
    {
        if (value)
            std::destroy_at(value);
    }
};
struct SourceReceipt
{
    Solver::TransportSnapshot transport;
    SourceEngine::Prepared prepared;
};
} // namespace

struct ReactingMixtureCoupler::Trial::Data
{
    const ReactingMixtureCoupler* owner = nullptr;
    std::uint64_t revision = 0;
    Solver::Trial flow;
    Diagnostics diagnostics;
    std::vector<SpeciesSourceLedger> ledgers;
    std::vector<std::shared_ptr<const Faces::Receipt>> face_receipts;
    std::vector<MixturePopulationEventReceipt> population_events;
};

struct ReactingMixtureCoupler::Implementation
{
    std::shared_ptr<Solver> flow;
    ReactingMixtureOptions options;
    std::optional<Solver::ControllerLease> lease;
    std::shared_ptr<const SourceEngine> source_engine;
    std::shared_ptr<Solver::PlanarMotion> motion;
    Diagnostics diagnostics;
    std::uint64_t generation = 0;

    Implementation(std::shared_ptr<Solver> solver, ReactingMixtureOptions controls)
        : flow(std::move(solver)), options(std::move(controls))
    {
        const auto& mesh = *flow->mesh();
        collective_detail::require_uniform_value(mesh, static_cast<unsigned long long>(options.face_transport.size()),
            "Ordered controller face-stage count");
        collective_detail::collective_local_validation(mesh, "Reacting mixture controller setup",
            [&]
            {
                require(options.mechanism && !options.phases.empty() && options.maximum_correctors >= 2 &&
                            std::isfinite(options.source_relative_tolerance) && options.source_relative_tolerance > 0 &&
                            options.liquid_phase < options.phases.size() && options.heat_phase < options.phases.size(),
                    "Reacting mixture controller requires a mechanism, phases and valid iteration controls.");
                require((!options.source || !options.source_identity.empty()) && options.source_revision > 0,
                    "A reaction source needs an explicit identity and a positive immutable revision.");
                qbmm_detail::validate_capability(
                    flow->options(), mesh, bool(options.planar_motion), options.body_acceleration);
                require(flow->initialized(), "Initialize low-Mach flow before constructing its reaction controller.");
                if (options.planar_motion)
                {
                    const auto& law = *options.planar_motion;
                    require(law.mesh && law.mesh.get() == flow->mesh().get() && !law.identity.empty() &&
                                law.revision > 0 && law.target_surface,
                        "Prescribed planar motion requires its exact mutable mesh and immutable target law metadata.");
                }
                if (options.interphase)
                {
                    const auto& event = *options.interphase;
                    require(event.configuration && event.integrate && !event.metadata.identity.empty() &&
                                event.metadata.revision > 0 &&
                                event.metadata.schema_fingerprint == options.mechanism->catalog().fingerprint(),
                        "Interphase kinetics needs explicit immutable slots, identity/revision/schema and a pure local "
                        "law.");
                    auto configuration = std::make_shared<const InterphaseConfiguration>(*event.configuration);
                    options.interphase = std::make_shared<const MixtureInterphaseEventProvider>(
                        MixtureInterphaseEventProvider{event.metadata, std::move(configuration), event.integrate});
                    if (!flow->state().cells.empty())
                        validate_interphase_configuration(
                            *options.interphase->configuration, flow->state().cells.front().phases);
                }
                // Callback copies/immutable engine allocation are local and may fail even on an empty owner.
                const auto thermal_configuration = configure_qbmm_sources(*flow, options);
                source_engine =
                    std::make_shared<const SourceEngine>(SourceEngine{flow, options, {}, thermal_configuration});
                const auto& catalog = options.mechanism->catalog();
                const auto ns = catalog.species().size();
                require(options.phases.size() == flow->options().phase_roles.size(),
                    "Controller physical bindings must match immutable flow roles, including empty owners.");
                for (size_t p = 0; p < options.phases.size(); ++p)
                    if (flow->options().phase_roles[p].role == Solver::PhaseRole::DispersedGas)
                        require(options.phases[p].phase == SpeciesPhase::Dispersed &&
                                    options.phases[p].population == flow->options().phase_roles[p].identity,
                            "Explicit dispersed role identity must match its named physical/chemical binding.");
                if (options.population_events)
                {
                    const auto& law = *options.population_events;
                    require(!law.identity.empty() && law.revision > 0 &&
                                law.schema_fingerprint == catalog.fingerprint() && law.evaluate,
                        "Population provider needs explicit immutable identity/revision/schema and a pure local law.");
                }
                std::set<std::tuple<SpeciesId, SpeciesPhase, std::string>> mapped;
                for (const auto& phase : options.phases)
                {
                    require(static_cast<size_t>(phase.phase) < static_cast<size_t>(SpeciesPhase::Count) &&
                                (phase.species_phases.empty() || phase.species_phases.size() == ns) &&
                                (phase.population.empty() || phase.phase == SpeciesPhase::Dispersed),
                        "Invalid chemical-to-physical phase binding.");
                    for (size_t i = 0; i < ns; ++i)
                    {
                        const auto target = compartment(phase, i);
                        require(static_cast<size_t>(target) < static_cast<size_t>(SpeciesPhase::Count) &&
                                    ((phase.phase == SpeciesPhase::Liquid &&
                                         (target == SpeciesPhase::Liquid || target == SpeciesPhase::Dissolved)) ||
                                        target == phase.phase),
                            "Physical liquid may contain liquid and dissolved inventories only.");
                        require(mapped.emplace(catalog.species()[i].id, target, phase.population).second,
                            "A chemical compartment cannot belong to two physical phases.");
                    }
                }
                require(options.phases[options.liquid_phase].phase == SpeciesPhase::Liquid,
                    "The designated physical liquid must bind Liquid/Dissolved species together.");
                for (size_t p = 0; p < options.phases.size(); ++p)
                    if (p != options.liquid_phase)
                        require(options.phases[p].phase == SpeciesPhase::Resolved ||
                                    options.phases[p].phase == SpeciesPhase::Dispersed,
                            "Only one physical liquid phase is supported; dissolved species share it.");
                if (options.interphase)
                {
                    const auto& config = *options.interphase->configuration;
                    require(config.phase_enthalpy.size() == options.phases.size() &&
                                config.metadata.schema_fingerprint == catalog.fingerprint(),
                        "Interphase immutable phase/schema layout differs.");
                    for (const auto& stage : config.stages)
                        for (const auto& slot : stage.material)
                        {
                            const auto i = catalog.index(slot.species);
                            require(slot.donor < options.phases.size() && slot.recipient < options.phases.size() &&
                                        catalog.allows(slot.species, compartment(options.phases[slot.donor], i)) &&
                                        catalog.allows(slot.species, compartment(options.phases[slot.recipient], i)),
                                "Interphase species must be explicitly allowed in both bound physical compartments.");
                        }
                }
                for (const auto& reaction : options.mechanism->reactions())
                    for (const auto& term : reaction.changes)
                        require(mapped.contains({term.species, term.phase, term.population}),
                            "A reaction references an unbound physical compartment.");
                if (flow->options().thermal_mode == PhaseThermalMode::SeparatePhaseTemperatures &&
                    !options.partition_heat)
                    for (const auto& reaction : options.mechanism->reactions())
                    {
                        std::set<size_t> physical_phases;
                        for (const auto& term : reaction.changes)
                            physical_phases.insert(source_engine->phase_for(term.species, term.phase, term.population));
                        require(physical_phases.size() <= 1,
                            "Separate-temperature cross-phase chemistry needs explicit phase-energy allocation.");
                    }
                for (const auto& face : options.face_transport)
                {
                    require(!face.identity.empty() && face.revision > 0 &&
                                face.schema_fingerprint == catalog.fingerprint() &&
                                face.physical_phase < options.phases.size() && (face.diffusion || face.fourier),
                        "Invalid immutable face-stage identity/schema/physical role.");
                    if (face.diffusion)
                    {
                        const auto& law = *face.diffusion;
                        const auto& b = law.binding;
                        require(!b.identity.empty() && b.revision > 0 && b.schemaFingerprint == catalog.fingerprint() &&
                                    b.counterSpecies.has_value() && law.molar_conductance &&
                                    law.partial_molar_enthalpy && !law.primary_species.empty() &&
                                    b.energyConvention == options.mechanism->energy_convention(),
                            "Diffusion requires explicit finite counter, laws and common energy/schema metadata.");
                        require(source_engine->phase_for(*b.counterSpecies, b.counter.phase, b.counter.population) ==
                                        face.physical_phase &&
                                    catalog.allows(*b.counterSpecies, b.counter.phase),
                            "Diffusion counter must belong to its declared physical carrier.");
                        std::set<SpeciesId> primary;
                        for (const auto id : law.primary_species)
                            require(primary.insert(id).second && catalog.allows(id, b.transported.phase) &&
                                        source_engine->phase_for(id, b.transported.phase, b.transported.population) ==
                                            face.physical_phase &&
                                        id != *b.counterSpecies,
                                "Diffusion primary/counter roles must be distinct and share one physical carrier.");
                    }
                    if (face.fourier)
                        require(!face.fourier->identity.empty() && face.fourier->revision > 0 &&
                                    face.fourier->schema_fingerprint == catalog.fingerprint() &&
                                    face.fourier->conductance,
                            "Fourier contact requires explicit immutable conductance metadata.");
                }
                for (const auto& cell : flow->state().cells)
                {
                    require(cell.phases.size() == options.phases.size(), "Physical phase schema does not match flow.");
                    for (size_t p = 0; p < cell.phases.size(); ++p)
                    {
                        const auto& phase = cell.phases[p];
                        require(phase.thermodynamics &&
                                    phase.thermodynamics->catalog().fingerprint() == catalog.fingerprint() &&
                                    phase.thermodynamics->energy_convention() == options.mechanism->energy_convention(),
                            "Reaction and thermodynamic schema/energy conventions must agree.");
                        require(phase.thermodynamics->kind() ==
                                    (p == options.liquid_phase ? EOSKind::Liquid : EOSKind::Gas),
                            "Physical phase binding and EOS kind must agree.");
                    }
                    require(cell.phases[options.liquid_phase].thermodynamics->kind() == EOSKind::Liquid,
                        "The reaction liquid-volume carrier must use a liquid EOS.");
                    (void) source_engine->species(cell);
                }
            });
        std::string schema_signature;
        collective_detail::collective_local_validation(mesh, "Controller immutable metadata encoding",
            [&]
            {
                std::ostringstream signature;
                signature.imbue(std::locale::classic());
                signature << std::hexfloat << options.mechanism->catalog().fingerprint() << ':' << bool(options.source)
                          << ':' << options.source_identity.size() << ':' << options.source_identity << ':'
                          << options.source_revision << ':' << options.maximum_correctors << ':'
                          << options.source_relative_tolerance << ':' << options.liquid_phase << ':'
                          << options.heat_phase << ':' << bool(options.partition_heat) << ':'
                          << options.integration.maximumSubstep << ':' << options.integration.maximumSubsteps << ':'
                          << options.integration.maximumRetries << ':' << int(options.mechanism->energy_convention());
                signature << ':' << options.phases.size() << ':' << options.mechanism->reactions().size();
                if (options.body_acceleration)
                {
                    signature << ":body-acceleration-si-v1";
                    for (size_t axis = 0; axis < 3; ++axis)
                        signature << ':' << options.body_acceleration->component(axis);
                }
                for (const auto& phase : options.phases)
                {
                    signature << ':' << int(phase.phase) << ':' << phase.population.size() << ':' << phase.population
                              << ':' << phase.species_phases.size();
                    for (const auto kind : phase.species_phases)
                        signature << ':' << int(kind);
                }
                for (const auto& reaction : options.mechanism->reactions())
                {
                    signature << ':' << reaction.id << ':' << reaction.heatPerExtent << ':' << reaction.changes.size();
                    for (const auto& term : reaction.changes)
                        signature << ':' << term.species << ':' << int(term.phase) << ':' << term.population.size()
                                  << ':' << term.population << ':' << term.coefficient;
                }
                signature << ':' << options.face_transport.size();
                for (const auto& face : options.face_transport)
                {
                    signature << ':' << face.identity.size() << ':' << face.identity << ':' << face.revision << ':'
                              << face.schema_fingerprint << ':' << face.physical_phase << ':' << bool(face.diffusion)
                              << ':' << bool(face.fourier);
                    if (face.diffusion)
                    {
                        const auto& law = *face.diffusion;
                        const auto& b = law.binding;
                        signature << ':' << b.identity.size() << ':' << b.identity << ':' << b.revision << ':'
                                  << b.schemaFingerprint << ':' << *b.counterSpecies << ':' << int(b.energyConvention)
                                  << ':' << int(b.transported.phase) << ':' << b.transported.population.size() << ':'
                                  << b.transported.population << ':' << int(b.counter.phase) << ':'
                                  << b.counter.population.size() << ':' << b.counter.population << ':'
                                  << law.primary_species.size();
                        for (const auto id : law.primary_species)
                            signature << ':' << id;
                    }
                    if (face.fourier)
                        signature << ':' << face.fourier->identity.size() << ':' << face.fourier->identity << ':'
                                  << face.fourier->revision << ':' << face.fourier->schema_fingerprint;
                }
                signature << ":population:" << bool(options.population_events);
                if (options.population_events)
                {
                    const auto& law = *options.population_events;
                    signature << ':' << law.identity.size() << ':' << law.identity << ':' << law.revision << ':'
                              << law.schema_fingerprint;
                }
                if (options.population_events)
                {
                    signature << ":scope:" << options.population_events->populations.size();
                    for (const auto& name : options.population_events->populations)
                        signature << ':' << name.size() << ':' << name;
                }
                if (uses_qbmm_kinetic_transport(flow->options().qbmm_model))
                {
                    const auto& model = flow->options().qbmm_model;
                    signature << ":kinetic-euler-v1:" << model.kinetic_cfl << ':' << model.maximum_kinetic_substeps;
                    if (options.body_acceleration)
                        signature << ":balanced-kinetic-pressure:" << Solver::balanced_kinetic_pressure_revision;
                    for (const auto& binding : flow->options().qbmm_node_slip_laws)
                        if (binding)
                        {
                            const auto metadata = qbmm_node_slip_signature(*binding);
                            signature << ':' << metadata.size() << ':' << metadata;
                        }
                        else
                            signature << ":0:";
                }
                if (uses_qbmm_linear_transport(flow->options().qbmm_model))
                {
                    const auto& model = flow->options().qbmm_model;
                    signature << ":realizable-linear-ssprk2:" << qbmm_linear_transport_revision << ':'
                              << model.linear_cfl << ':' << model.maximum_linear_substeps;
                    if (options.body_acceleration)
                        signature << ":balanced-linear-pressure:" << Solver::balanced_linear_pressure_revision
                                  << ":forced-linear-thermal:" << Solver::forced_linear_thermal_revision;
                }
                signature << ":qbmm:" << bool(options.qbmm_population);
                if (options.qbmm_population)
                {
                    const auto& law = *options.qbmm_population;
                    signature << ':' << law.metadata.identity.size() << ':' << law.metadata.identity << ':'
                              << law.metadata.revision << ':' << law.metadata.schema_fingerprint << ':'
                              << bool(law.transfers);
                    if (law.transfers)
                        signature << conditioned_thermal_detail::signature(*law.transfers);
                    if (!law.laws.empty())
                    {
                        signature << ":constitutive:" << bool(law.evaluate) << ':' << law.laws.size();
                        for (const auto& supplied : law.laws)
                        {
                            const auto metadata = qbmm_law_signature(supplied->metadata);
                            signature << ':' << metadata.size() << ':' << metadata;
                        }
                    }
                }
                signature << ":planar:" << bool(options.planar_motion);
                if (options.planar_motion)
                {
                    const auto& law = *options.planar_motion;
                    signature << ':' << law.identity.size() << ':' << law.identity << ':' << law.revision;
                }
                signature << ":interphase:" << bool(options.interphase);
                if (options.interphase)
                {
                    const auto& law = *options.interphase;
                    signature << ':' << law.metadata.identity.size() << ':' << law.metadata.identity << ':'
                              << law.metadata.revision << ':' << law.metadata.schema_fingerprint << ':'
                              << bool(law.integrate);
                    const auto configuration = conditioned_thermal_detail::signature(*law.configuration);
                    signature << ':' << configuration.size() << ':' << configuration;
                }
                schema_signature = signature.str();
            });
        uniform_text(mesh, schema_signature);
        if (!options.face_transport.empty())
        {
            std::shared_ptr<SourceEngine> engine;
            std::shared_ptr<const SpeciesCatalog> catalog_alias;
            Faces::PhaseBindings bindings;
            collective_detail::collective_local_validation(mesh, "Immutable face-stage construction inputs",
                [&]
                {
                    engine = std::make_shared<SourceEngine>(*source_engine);
                    engine->face_stages.reserve(options.face_transport.size());
                    // Aliasing ownership retains the mechanism/catalog lifetime without copying its immutable schema.
                    catalog_alias =
                        std::shared_ptr<const SpeciesCatalog>(options.mechanism, &options.mechanism->catalog());
                    bindings.resize(options.phases.size());
                    for (size_t p = 0; p < options.phases.size(); ++p)
                        for (size_t s = 0; s < options.mechanism->catalog().species().size(); ++s)
                            bindings[p].push_back({compartment(options.phases[p], s), options.phases[p].population});
                });
            // Collective constructors run only after all phase/source/ordered-stage metadata agrees.
            for (const auto& face : options.face_transport)
            {
                std::shared_ptr<FaceStageStorage> storage;
                Faces::Options prepared_options;
                Faces::PhaseBindings prepared_bindings;
                collective_detail::collective_local_validation(mesh, "Allocate one ordered face stage",
                    [&]
                    {
                        storage = std::make_shared<FaceStageStorage>();
                        prepared_options = face;
                        prepared_bindings = bindings;
                    });
                storage->value = ::new (static_cast<void*>(storage->memory)) Faces::LowMachFaceTransportAdapter(
                    flow->mesh(), catalog_alias, std::move(prepared_options), std::move(prepared_bindings),
                    flow->options().thermal_mode);
                // Aliasing does not allocate; the reserved vector preserves this exact immutable stage association.
                engine->face_stages.emplace_back(storage, storage->value);
            }
            source_engine = std::move(engine);
        }
        if (options.planar_motion)
        {
            std::shared_ptr<PlanarMotionStorage> storage;
            collective_detail::collective_local_validation(mesh, "Allocate controller-owned planar motion",
                [&] { storage = std::make_shared<PlanarMotionStorage>(); });
            storage->value = ::new (static_cast<void*>(storage->memory))
                Solver::PlanarMotion(options.planar_motion->mesh, options.planar_motion->options);
            motion = std::shared_ptr<Solver::PlanarMotion>(storage, storage->value);
        }
        const bool legacy_compatible =
            !options.planar_motion && !options.body_acceleration &&
            (!options.qbmm_population || options.qbmm_population->laws.empty()) &&
            flow->options().qbmm_model.velocity == QbmmModelOptions::Velocity::SharedCarrier &&
            std::ranges::all_of(flow->options().qbmm_families,
                [](const auto& family) { return !family || family->closure_versions().laws.empty(); });
        const Solver::ControllerRestartDiagnostics restart{
            options.mechanism->reactions().size(), nullptr, &diagnostics};
        lease.emplace(flow->acquire_controller_for(
            schema_signature, legacy_compatible, options.body_acceleration, motion.get(), &restart));
        diagnostics.flow.family_transport = flow->options().qbmm_model.family_transport;
    }

    Diagnostics totals(const std::vector<SpeciesSourceLedger>& ledgers, std::span<const Solver::CellSources> applied,
        std::span<const MixturePopulationEventReceipt> populations, const Solver::QbmmSourceCells& qbmm,
        std::span<const PressureConditionedThermalEvaluation> thermal, int correctors,
        const Solver::Diagnostics& flow_statistics) const
    {
        const auto nr = options.mechanism->reactions().size();
        std::vector<double> local, global;
        collective_detail::collective_local_validation(*flow->mesh(), "Reacting source reduction input",
            [&]
            {
                require(ledgers.size() == applied.size(), "Reaction diagnostics must match applied owned packets.");
                require(
                    nr <= (size_t(std::numeric_limits<int>::max()) - 5) / 3, "Too many reaction diagnostic entries.");
                std::vector<Sum> sums(3 * nr + 5);
                for (const auto& ledger : ledgers)
                {
                    for (size_t r = 0; r < nr; ++r)
                    {
                        sums[r] += ledger.reactionExtents.at(r);
                        sums[nr + r] += ledger.forwardReactionExtents.at(r);
                        sums[2 * nr + r] += ledger.reverseReactionExtents.at(r);
                    }
                    sums[3 * nr] += ledger.reactionHeat;
                    sums[3 * nr + 1] += ledger.externalHeat;
                    sums[3 * nr + 2] += ledger.transferEnergy;
                    sums[3 * nr + 3] += ledger.externalMass;
                }
                for (const auto& cell : qbmm)
                    for (const auto& family : cell)
                        if (family)
                            for (const auto& event : family->events)
                                for (const auto& raw : event.transfers)
                                    if (const auto* reservoir =
                                            std::get_if<QbmmReservoirExchange>(&raw.transfer.counterpart))
                                    {
                                        sums[3 * nr + 2] += reservoir->energy_j;
                                        const auto& species = options.mechanism->catalog().species();
                                        for (size_t i = 0; i < species.size(); ++i)
                                        {
                                            const long double moles = raw.transfer.moles.at(i);
                                            sums[3 * nr + 3] +=
                                                (raw.into_family ? moles : -moles) * species[i].molarMass;
                                        }
                                    }
                // These are the final applied physical packets, after every raw face/source cancellation.
                // Keep one rank-local sum across all cells and phases rather than first rounding cell totals.
                require(!source_engine->thermal_configuration || thermal.size() == applied.size(),
                    "Final thermal diagnostic receipt row mismatch.");
                for (size_t c = 0; c < applied.size(); ++c)
                {
                    const auto& energies = source_engine->thermal_configuration ? thermal[c].applied_phase_energy
                                                                                : applied[c].internal_energy;
                    for (const auto value : energies)
                        sums[3 * nr + 4] += value;
                }
                local.resize(sums.size());
                global.resize(sums.size());
                for (size_t i = 0; i < local.size(); ++i)
                {
                    const auto value = sums[i].value();
                    local[i] = static_cast<double>(value);
                    require(std::isfinite(value) && std::isfinite(local[i]) && (value == 0 || local[i] != 0),
                        "Reaction diagnostic overflow or underflow.");
                }
            });
        Teuchos::reduceAll(*flow->mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM,
            static_cast<int>(local.size()), local.data(), global.data());
        Diagnostics result;
        collective_detail::collective_local_validation(*flow->mesh(), "Reacting source reduction report",
            [&]
            {
                for (const auto value : global)
                    require(std::isfinite(value), "Global reaction diagnostic overflow.");
                result.correctors = correctors;
                result.reaction_extents.assign(global.begin(), global.begin() + nr);
                result.forward_reaction_extents.assign(global.begin() + nr, global.begin() + 2 * nr);
                result.reverse_reaction_extents.assign(global.begin() + 2 * nr, global.begin() + 3 * nr);
                result.chemical_heat = global[3 * nr];
                result.external_heat = global[3 * nr + 1];
                result.transfer_energy = global[3 * nr + 2];
                result.external_mass = global[3 * nr + 3];
                result.net_thermal_energy = global[3 * nr + 4];
                result.flow = flow_statistics;
            });
        // All ranks, including empty owners, reduce the same immutable physical-role layout.
        std::vector<double> local_numbers, global_numbers;
        const size_t np = options.phases.size();
        collective_detail::collective_local_validation(*flow->mesh(), "Population event diagnostics",
            [&]
            {
                require(populations.size() == applied.size() && np <= size_t(std::numeric_limits<int>::max()) / 3,
                    "Population diagnostics row/phase layout differs.");
                std::vector<Sum> totals(3 * np);
                for (size_t c = 0; c < populations.size(); ++c)
                {
                    const auto& receipt = populations[c];
                    require(receipt.births.size() == np && receipt.deaths.size() == np && receipt.changes.size() == np,
                        "Population receipt diagnostic shape differs.");
                    for (size_t p = 0; p < np; ++p)
                    {
                        totals[p] += receipt.births[p];
                        totals[np + p] += receipt.deaths[p];
                        totals[2 * np + p] += applied[c].population_number_changes.at(p);
                        if (!qbmm.empty() && qbmm[c][p])
                            for (const auto& event : qbmm[c][p]->events)
                            {
                                if (event.kind == QbmmEventKind::Birth)
                                    totals[p] += event.gross_created[0];
                                if (event.kind == QbmmEventKind::Disappearance)
                                    totals[np + p] += event.gross_consumed[0];
                            }
                    }
                }
                local_numbers.resize(totals.size());
                global_numbers.resize(totals.size());
                for (size_t i = 0; i < totals.size(); ++i)
                    local_numbers[i] = population_source_detail::narrow(totals[i].value());
            });
        Teuchos::reduceAll(*flow->mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM,
            static_cast<int>(local_numbers.size()), local_numbers.data(), global_numbers.data());
        collective_detail::collective_local_validation(*flow->mesh(), "Population event global diagnostic report",
            [&]
            {
                for (double n : global_numbers)
                    require(std::isfinite(n), "Population global diagnostics overflow.");
                result.population_births.assign(global_numbers.begin(), global_numbers.begin() + np);
                result.population_deaths.assign(global_numbers.begin() + np, global_numbers.begin() + 2 * np);
                result.population_number_changes.assign(global_numbers.begin() + 2 * np, global_numbers.end());
            });
        return result;
    }
};

ReactingMixtureCoupler::ReactingMixtureCoupler(std::shared_ptr<Solver> solver, ReactingMixtureOptions options)
{
    require(bool(solver), "Reacting mixture controller requires shared solver ownership.");
    d = std::make_unique<Implementation>(std::move(solver), std::move(options));
}
ReactingMixtureCoupler::~ReactingMixtureCoupler() = default;
const std::shared_ptr<Solver>& ReactingMixtureCoupler::solver() const noexcept
{
    return d->flow;
}
const ReactingMixtureCoupler::Diagnostics& ReactingMixtureCoupler::diagnostics() const noexcept
{
    return d->diagnostics;
}

Solver::QbmmReport ReactingMixtureCoupler::qbmm_report() const
{
    return qbmm_report_impl(nullptr);
}

Solver::QbmmReport ReactingMixtureCoupler::qbmm_report(const Trial& trial) const
{
    return qbmm_report_impl(&trial);
}

Solver::QbmmReport ReactingMixtureCoupler::qbmm_report_impl(const Trial* trial) const
{
    collective_detail::collective_local_validation(*d->flow->mesh(), "Reacting QBMM report selection",
        [&]
        {
            require(!trial || (trial->d && trial->d->owner == this && trial->d->revision == d->generation),
                "Reacting QBMM report trial is empty, foreign or stale.");
        });
    return trial ? d->flow->qbmm_report(trial->d->flow) : d->flow->qbmm_report();
}

const Solver::State& ReactingMixtureCoupler::Trial::state() const
{
    require(bool(d), "Empty reacting mixture trial.");
    return d->flow.state();
}
const ReactingMixtureCoupler::Diagnostics& ReactingMixtureCoupler::Trial::diagnostics() const
{
    require(bool(d), "Empty reacting mixture trial.");
    return d->diagnostics;
}
const std::vector<SpeciesSourceLedger>& ReactingMixtureCoupler::Trial::local_sources() const
{
    require(bool(d), "Empty reacting mixture trial.");
    return d->ledgers;
}

const Solver::QbmmSourceCells& ReactingMixtureCoupler::Trial::local_qbmm_sources() const
{
    require(bool(d), "Empty reacting mixture trial.");
    return d->flow.qbmm_source_cells();
}
const std::vector<MixturePopulationEventReceipt>& ReactingMixtureCoupler::Trial::local_population_events() const
{
    require(bool(d), "Empty reacting mixture trial.");
    return d->population_events;
}

const std::shared_ptr<const PressureConditionedThermalBatch>& ReactingMixtureCoupler::Trial::interphase_receipt() const
{
    require(bool(d), "Empty reacting mixture trial.");
    return d->flow.conditioned_thermal_receipt();
}
std::span<const PressureConditionedThermalEvaluation> ReactingMixtureCoupler::Trial::final_thermal_evaluations() const
{
    require(bool(d), "Empty reacting mixture trial.");
    return d->flow.final_thermal_evaluations();
}

const Solver::TransportSnapshot& ReactingMixtureCoupler::Trial::transport_snapshot() const
{
    require(bool(d), "Empty reacting mixture trial.");
    return d->flow.transport_snapshot();
}

std::span<const Solver::PhaseFaceTransport> ReactingMixtureCoupler::Trial::owned_phase_transport() const
{
    require(bool(d), "Empty reacting mixture trial.");
    return d->flow.owned_phase_transport();
}

ReactingMixtureCoupler::Trial ReactingMixtureCoupler::preview(double dt)
{
    const auto& mesh = *d->flow->mesh();
    Solver::require_uniform_advancement(mesh, Solver::AdvancementOperation::ControllerPreview);
    collective_detail::collective_local_validation(mesh, "Reacting mixture timestep",
        [&] { require(std::isfinite(dt) && dt > 0, "Reacting mixture timestep must be positive and finite."); });
    collective_detail::require_uniform_value(mesh, dt, "Reacting mixture timestep");
    double target_surface = 0;
    if (d->motion)
    {
        collective_detail::collective_local_validation(mesh, "Prescribed planar target law",
            [&]
            {
                target_surface = d->options.planar_motion->target_surface(d->flow->state().time, dt);
                require(std::isfinite(target_surface), "Prescribed planar target must be finite.");
            });
        collective_detail::require_uniform_value(mesh, target_surface, "Prescribed planar target");
    }
    std::shared_ptr<SourceReceipt> receipt;
    std::shared_ptr<const Solver::SourceStageProvider> provider;
    collective_detail::collective_local_validation(mesh, "Allocate immutable source provider",
        [&]
        {
            receipt = std::make_shared<SourceReceipt>();
            provider = std::make_shared<const Solver::SourceStageProvider>(Solver::SourceStageProvider{
                "ReactingMixtureCoupler:" + d->options.source_identity +
                    (d->options.population_events
                            ? (":population:" + std::to_string(d->options.population_events->identity.size()) + ":" +
                                  d->options.population_events->identity + ":" +
                                  std::to_string(d->options.population_events->revision))
                            : std::string{}) +
                    (d->options.qbmm_population
                            ? (":qbmm:" + std::to_string(d->options.qbmm_population->metadata.identity.size()) + ":" +
                                  d->options.qbmm_population->metadata.identity + ":" +
                                  std::to_string(d->options.qbmm_population->metadata.revision))
                            : std::string{}) +
                    (d->options.planar_motion
                            ? (":motion:" + std::to_string(d->options.planar_motion->identity.size()) + ":" +
                                  d->options.planar_motion->identity + ":" +
                                  std::to_string(d->options.planar_motion->revision))
                            : std::string{}) +
                    (d->options.interphase
                            ? (":interphase:" + std::to_string(d->options.interphase->metadata.identity.size()) + ":" +
                                  d->options.interphase->metadata.identity + ":" +
                                  std::to_string(d->options.interphase->metadata.revision))
                            : std::string{}),
                d->options.source_revision, d->options.mechanism->catalog().fingerprint(),
                [engine = d->source_engine, receipt](const Solver::SourceStageContext& context)
                {
                    // This is a communicating stage. Its local chemistry and each face stage own their fences.
                    auto prepared = engine->prepare(context);
                    engine->validate_face_receipts(prepared, context);
                    Solver::SourceStageResult result;
                    collective_detail::collective_local_validation(context.mesh, "Retain composed source receipt",
                        [&]
                        {
                            receipt->transport = context.transport;
                            receipt->prepared = std::move(prepared);
                            result = receipt->prepared.result;
                        });
                    return result;
                },
                d->options.source_relative_tolerance, d->options.maximum_correctors, d->options.body_acceleration});
        });
    const bool has_qbmm = !d->flow->options().qbmm_families.empty();
    auto flow_trial = [&]
    {
        if (d->motion)
            return has_qbmm ? d->flow->preview_ale_with_qbmm_sources(
                                  dt, *d->motion, target_surface, std::move(provider), &*d->lease)
                            : d->flow->preview_ale_with_sources(
                                  dt, *d->motion, target_surface, std::move(provider), &*d->lease);
        return has_qbmm ? d->flow->preview_with_qbmm_sources(dt, std::move(provider), &*d->lease)
                        : d->flow->preview_with_sources(dt, std::move(provider), &*d->lease);
    }();
    collective_detail::collective_local_validation(mesh, "Reacting mixture immutable source receipt",
        [&]
        {
            require(receipt->transport == flow_trial.transport_snapshot(),
                "Reaction receipt does not match the accepted transport trial.");
            require(identical_sources(receipt->prepared.result.sources, flow_trial.integrated_sources()),
                "Reaction receipt does not match the exact applied source packets.");
            require(receipt->prepared.result.thermal_receipt.get() == flow_trial.conditioned_thermal_receipt().get(),
                "Interphase receipt does not match the exact frozen flow thermal plan.");
            if (flow_trial.conditioned_thermal_receipt())
            {
                const auto evaluations = flow_trial.final_thermal_evaluations();
                require(evaluations.size() == flow_trial.state().cells.size(), "Final thermal receipt layout differs.");
                for (size_t c = 0; c < evaluations.size(); ++c)
                    for (size_t a = 0; a < evaluations[c].phases.size(); ++a)
                    {
                        const auto& x = evaluations[c].phases[a];
                        const auto& y = flow_trial.state().cells[c].phases[a];
                        require(x.thermodynamics.get() == y.thermodynamics.get() && x.moles == y.moles &&
                                    x.internal_energy == y.internal_energy && x.temperature == y.temperature &&
                                    x.absolute_pressure == y.absolute_pressure,
                            "Final thermal evaluation does not match the exact accepted phase state.");
                    }
            }
            const auto events = flow_trial.integrated_gross_events();
            const auto& supplied = receipt->prepared.result.gross_events;
            require(events.size() == supplied.size(), "Reaction receipt gross cell layout differs.");
            const auto enabled = std::ranges::count_if(
                d->flow->options().qbmm_families, [](const auto& family) { return bool(family); });
            const auto native_width = Solver::QbmmSourceReceipt::gross_channels * static_cast<size_t>(enabled);
            for (size_t c = 0; c < events.size(); ++c)
                require(events[c].size() == supplied[c].size() + native_width &&
                            std::ranges::equal(std::span(events[c]).first(supplied[c].size()), supplied[c]),
                    "Reaction receipt does not match the exact provider-supplied gross channels.");
        });
    const Solver::SourceStageContext final_context{flow_trial.transport_snapshot(), flow_trial.state(), mesh};
    // ALE receipts were producer/phase/epoch validated inside the active source callback.
    // The exact immutable transport/source/gross equality above binds that proof to the retained trial.
    // Its live geometry is now rolled back, so only fixed-grid previews repeat live-epoch validation.
    if (!d->motion)
        d->source_engine->validate_face_receipts(receipt->prepared, final_context);
    if (!d->flow->options().qbmm_families.empty())
        d->flow->finalize_qbmm_balances(flow_trial, std::span<const SpeciesSourceLedger>(receipt->prepared.ledgers));
    auto statistics = d->totals(receipt->prepared.ledgers, flow_trial.integrated_sources(),
        receipt->prepared.population_events, flow_trial.qbmm_source_cells(), flow_trial.final_thermal_evaluations(),
        flow_trial.diagnostics().source_evaluations, flow_trial.diagnostics());
    std::shared_ptr<Trial::Data> data;
    collective_detail::collective_local_validation(mesh, "Retain immutable reacting trial",
        [&]
        {
            data = std::make_shared<Trial::Data>();
            data->owner = this;
            data->revision = d->generation;
            data->diagnostics = std::move(statistics);
            data->flow = std::move(flow_trial);
            data->ledgers = std::move(receipt->prepared.ledgers);
            data->face_receipts = std::move(receipt->prepared.face_receipts);
            data->population_events = std::move(receipt->prepared.population_events);
        });
    // The flow trial retains the immutable provider lifetime. Release this
    // temporary duplicate only after exact receipt verification and publication.
    *receipt = {};
    Trial result;
    result.d = std::move(data);
    return result;
}

void ReactingMixtureCoupler::commit(const Trial& trial)
{
    const auto& mesh = *d->flow->mesh();
    Solver::require_uniform_advancement(mesh, Solver::AdvancementOperation::ControllerCommit);
    Diagnostics candidate;
    collective_detail::collective_local_validation(mesh, "Reacting mixture commit preflight",
        [&]
        {
            require(trial.d && trial.d->owner == this && trial.d->revision == d->generation,
                "Reacting mixture trial is foreign, stale or already committed.");
            require(d->generation != std::numeric_limits<std::uint64_t>::max(),
                "Reacting mixture controller generation exhausted.");
            candidate = trial.d->diagnostics;
        });
    if (d->motion)
        d->flow->commit_ale(trial.d->flow, *d->motion, &*d->lease);
    else
    {
        const Solver::ControllerRestartDiagnostics diagnostics{d->options.mechanism->reactions().size(), &candidate};
        d->flow->commit_fixed(trial.d->flow, &*d->lease, &diagnostics);
    }
    d->diagnostics = std::move(candidate);
    ++d->generation;
}
ReactingMixtureCoupler::Diagnostics ReactingMixtureCoupler::advance(double dt)
{
    const auto trial = preview(dt);
    commit(trial);
    return d->diagnostics;
}
ReactingMixtureCoupler::Checkpoint ReactingMixtureCoupler::checkpoint() const
{
    Checkpoint result;
    result.owner = this;
    result.flow = d->motion ? d->flow->checkpoint(*d->motion) : d->flow->checkpoint();
    result.diagnostics = d->diagnostics;
    result.diagnostics.flow.body_acceleration_m_per_s2 = result.flow.diagnostics.body_acceleration_m_per_s2;
    return result;
}
void ReactingMixtureCoupler::restore(const Checkpoint& checkpoint)
{
    collective_detail::require_uniform_value(*d->flow->mesh(), d->generation, "Reacting mixture restore generation");
    Diagnostics restored;
    collective_detail::collective_local_validation(*d->flow->mesh(), "Reacting mixture checkpoint owner",
        [&]
        {
            require(checkpoint.owner == this, "Reacting mixture checkpoint belongs to another controller.");
            require(d->generation != std::numeric_limits<std::uint64_t>::max(),
                "Reacting mixture controller generation exhausted.");
            restored = checkpoint.diagnostics;
        });
    if (d->motion)
        d->flow->restore(checkpoint.flow, *d->motion, &*d->lease);
    else
        d->flow->restore(checkpoint.flow, &*d->lease);
    d->diagnostics = std::move(restored);
    ++d->generation;
}

std::vector<std::byte> ReactingMixtureCoupler::serialize_qbmm_restart() const
{
    if (!d->motion && !uses_qbmm_explicit_transport(d->flow->options().qbmm_model))
        return d->flow->serialize_qbmm_restart();
    const auto* target = d->options.planar_motion.get();
    const Solver::ControllerRestartDiagnostics diagnostics{d->options.mechanism->reactions().size(), &d->diagnostics};
    return d->flow->serialize_qbmm_restart_impl(
        d->motion.get(), target ? std::string_view(target->identity) : "", target ? target->revision : 0, &diagnostics);
}

void ReactingMixtureCoupler::restore_qbmm_restart(std::span<const std::byte> payload)
{
    const auto& mesh = *d->flow->mesh();
    collective_detail::require_uniform_value(mesh, d->generation, "Reacting mixture restart generation");
    require(d->generation != std::numeric_limits<std::uint64_t>::max(),
        "Reacting mixture controller generation exhausted.");
    Diagnostics restored;
    const auto* target = d->options.planar_motion.get();
    const Solver::ControllerRestartDiagnostics diagnostics{
        d->options.mechanism->reactions().size(), nullptr, &restored};
    d->flow->restore_qbmm_restart_impl(payload, d->motion.get(), target ? std::string_view(target->identity) : "",
        target ? target->revision : 0, &*d->lease, &diagnostics);
    d->diagnostics = std::move(restored);
    ++d->generation;
}

void ReactingMixtureCoupler::restore_qbmm_checkpoint(
    const Solver& previous, const Solver::Checkpoint& saved, const Solver::QbmmInitialization& initialization)
{
    const auto& mesh = *d->flow->mesh();
    collective_detail::require_uniform_value(mesh, d->generation, "Reacting mixture migration generation");
    Diagnostics restored;
    collective_detail::collective_local_validation(mesh, "Reacting mixture checkpoint migration",
        [&]
        {
            require(!d->motion && d->generation != std::numeric_limits<std::uint64_t>::max(),
                "Reacting mixture checkpoint migration requires fixed geometry and a live generation.");
            restored.flow = saved.diagnostics;
        });
    d->flow->restore_qbmm_checkpoint(previous, saved, initialization, &*d->lease);
    const auto& migrated = d->flow->diagnostics();
    restored.flow.family_transport = migrated.family_transport;
    restored.flow.body_acceleration_m_per_s2 = migrated.body_acceleration_m_per_s2;
    restored.flow.impulse_parts = migrated.impulse_parts;
    d->diagnostics = std::move(restored);
    ++d->generation;
}
} // namespace SimpleFluid::Reacting
