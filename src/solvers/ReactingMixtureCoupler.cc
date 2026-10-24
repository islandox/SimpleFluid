/** @file ReactingMixtureCoupler.cc
 * @brief Transport-first chemistry and immutable low-Mach source transactions.
 */
#include "solvers/ReactingMixtureCoupler.hh"
#include "equations/CollectiveValidation.hh"
#include "solvers/details/LowMachConditionedThermal.hh"
#include "solvers/details/LowMachFaceSourceComposition.hh"
#include "solvers/details/LowMachPopulationSource.hh"
#include "utils/CompensatedSum.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
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
            a[c].population_number_changes != b[c].population_number_changes)
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

struct SourceEngine
{
    std::shared_ptr<Solver> flow;
    ReactingMixtureOptions options;
    std::vector<std::shared_ptr<const Faces::LowMachFaceTransportAdapter>> face_stages;
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
    void prepare_chemistry(const Solver::SourceStageContext& context, Faces::LocalChemistryStage& stage) const
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
            for (size_t p = 0; p < np; ++p)
            {
                cell.phase_numbers[p] = transport.cells[c].phases[p].population_number;
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
                            for (const auto& other : cell.phases)
                                if (other.temperature > 0)
                                {
                                    phase.temperature = other.temperature;
                                    break;
                                }
                    }
                    const auto point = phase.thermodynamics->evaluate(phase.temperature, phase.absolute_pressure, x);
                    phase.internal_energy = double(phase_mass.value() * point.internal_energy);
                    require(std::isfinite(phase.internal_energy), "Rate-state phase energy overflow.");
                }
                else
                {
                    phase.temperature = 0;
                    phase.internal_energy = 0;
                }
            }
            const auto initial = species(cell);
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
            require(transport.new_geometry_epoch == context.mesh.geometry_epoch(),
                "Reaction source location belongs to stale trial geometry.");
            const auto local = static_cast<Solver::Pack::local_ordinal_type>(c);
            const auto geometryCell = context.mesh.cell_geometry_global_id(local);
            require(std::in_range<std::uint64_t>(geometryCell), "Canonical source cell identity is not representable.");
            const auto center = context.mesh.cell_centroid(local);
            environment.location = ReactionLocation{static_cast<std::uint64_t>(geometryCell),
                transport.new_geometry_epoch, {center.x, center.y, center.z}, c};

            ReactionContext reaction{catalog, initial, environment};
            const auto input =
                options.source ? options.source({reaction, cell, c, transport.cells[c]}, dt) : ReactionSourceInput{};
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
            declared_heat += raw.externalHeat;
            for (const auto heat : raw.reactionHeatContributions)
                declared_heat += heat;

            if (options.partition_heat)
            {
                // Declare original channels before custom double allocation. Later cancellation must not hide loss.
                stage.sums.declare_energy(c, raw.externalHeat);
                for (const auto heat : raw.reactionHeatContributions)
                    stage.sums.declare_energy(c, heat);
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
                stage.sums.add_energy(c, options.heat_phase, raw.externalHeat);
                for (const auto heat : raw.reactionHeatContributions)
                    stage.sums.add_energy(c, options.heat_phase, heat);
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

    Prepared prepare(const Solver::SourceStageContext& context) const
    {
        std::vector<const Faces::LowMachFaceTransportAdapter*> ordered;
        std::function<void(const Solver::SourceStageContext&, Faces::LocalChemistryStage&)> local_prepare;
        Faces::PostFaceLocalStage interphase_stage;
        std::shared_ptr<PressureConditionedThermalBatch> thermal_batch;
        collective_detail::collective_local_validation(context.mesh, "Ordered face-stage views",
            [&]
            {
                local_prepare = [this](const Solver::SourceStageContext& current, Faces::LocalChemistryStage& local)
                { prepare_chemistry(current, local); };
                if (options.interphase)
                {
                    thermal_batch = std::make_shared<PressureConditionedThermalBatch>();
                    thermal_batch->configuration = options.interphase->configuration;
                    thermal_batch->source = conditioned_thermal_detail::identity(context.transport);
                    thermal_batch->cells.reserve(context.transport.cells.size());
                    interphase_stage = [this, thermal_batch](const Solver::SourceStageContext& current,
                                           std::vector<CellSpeciesState>& donors,
                                           std::vector<Faces::RawAmounts>& raw_donors,
                                           Faces::LocalSourceAccumulator& sums,
                                           std::vector<std::vector<double>>& gross,
                                           std::span<const Solver::CellState> guesses)
                    {
                        const auto& catalog = options.mechanism->catalog();
                        const auto np = options.phases.size();
                        require(donors.size() == current.transport.cells.size() && guesses.size() == donors.size(),
                            "Interphase actual predecessor rows differ.");
                        for (size_t c = 0; c < donors.size(); ++c)
                        {
                            auto guess = guesses[c];
                            guess.phase_numbers.resize(np);
                            for (size_t a = 0; a < np; ++a)
                            {
                                guess.phase_numbers[a] = current.transport.cells[c].phases[a].population_number;
                                bool occupied = false;
                                for (size_t i = 0; i < catalog.species().size(); ++i)
                                {
                                    auto& n = guess.phases.at(a).moles.at(i);
                                    n = donors[c].amount(catalog, catalog.species()[i].id,
                                        compartment(options.phases[a], i), options.phases[a].population);
                                    occupied = occupied || n > 0;
                                }
                                if (!occupied)
                                {
                                    guess.phases[a].temperature = 0;
                                    guess.phases[a].internal_energy = 0;
                                }
                            }
                            require(current.transport.new_geometry_epoch == current.mesh.geometry_epoch(),
                                "Interphase rate location belongs to stale trial geometry.");
                            const auto local = static_cast<Solver::Pack::local_ordinal_type>(c);
                            const auto geometry_cell = current.mesh.cell_geometry_global_id(local);
                            require(std::in_range<std::uint64_t>(geometry_cell),
                                "Canonical interphase cell identity is not representable.");
                            const auto center = current.mesh.cell_centroid(local);
                            const ReactionLocation location{static_cast<std::uint64_t>(geometry_cell),
                                current.transport.new_geometry_epoch, {center.x, center.y, center.z}, c};
                            const auto amounts = options.interphase->integrate(
                                {catalog, donors[c], current.transport.cells[c], guess, current.transport.stage, c,
                                    current.transport.time_step, current.transport.accepted_time, location});
                            require(raw_donors.size() == donors.size(), "Raw interphase predecessor rows differ.");
                            std::vector<std::vector<std::array<long double, 2>>> phase_parts(
                                np, std::vector<std::array<long double, 2>>(catalog.species().size()));
                            for (size_t a = 0; a < np; ++a)
                                for (size_t i = 0; i < catalog.species().size(); ++i)
                                    phase_parts[a][i] = raw_donors[c].at(Faces::raw_amount_slot(catalog, donors[c],
                                        catalog.species()[i].id,
                                        {compartment(options.phases[a], i), options.phases[a].population}));
                            auto plan = compile_interphase_plan(options.interphase->configuration, guess.phases,
                                amounts, sums.raw_energy(c), sums.raw_declared(c), phase_parts);
                            for (size_t j = 0; j < plan.amounts.size(); ++j)
                            {
                                const auto& stage = plan.configuration->stages[j];
                                std::vector<Faces::MolarTerm> material_terms;
                                for (size_t i = 0; i < plan.amounts[j].size(); ++i)
                                {
                                    const double amount = plan.amounts[j][i];
                                    if (stage.kind == InterphaseStageKind::Material)
                                    {
                                        const auto& slot = stage.material[i];
                                        const auto species = catalog.index(slot.species);
                                        sums.add_material(c, slot.donor, species, -static_cast<long double>(amount));
                                        sums.add_material(c, slot.recipient, species, amount);
                                        material_terms.push_back({slot.species,
                                            {compartment(options.phases[slot.donor], species),
                                                options.phases[slot.donor].population}, -amount});
                                        material_terms.push_back({slot.species,
                                            {compartment(options.phases[slot.recipient], species),
                                                options.phases[slot.recipient].population}, amount});
                                        gross[c].push_back(amount);
                                    }
                                    else
                                    {
                                        gross[c].push_back(std::max(amount, 0.));
                                        gross[c].push_back(std::max(-amount, 0.));
                                    }
                                }
                                if (stage.kind == InterphaseStageKind::Material)
                                    donors[c] = Faces::apply_raw_material(
                                        catalog, donors[c], raw_donors[c], material_terms);
                            }
                            for (size_t a = 0; a < np; ++a)
                                for (size_t i = 0; i < catalog.species().size(); ++i)
                                    require(donors[c].amount(catalog, catalog.species()[i].id,
                                                compartment(options.phases[a], i), options.phases[a].population) ==
                                                plan.final_moles[a][i],
                                        "Raw interphase replay differs from its frozen material plan.");
                            donors[c].validate(catalog);
                            thermal_batch->cells.push_back(std::move(plan));
                        }
                    };
                }
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
                result.result.thermal_receipt = std::move(thermal_batch);
                result.ledgers = std::move(composed.chemistry_ledgers);
                result.face_receipts = std::move(composed.faces);
                result.post_faces = std::move(composed.post_faces);
                result.population_events.reserve(result.post_faces.size());
                const auto& roles = flow->options().phase_roles;
                const auto& population_catalog = options.mechanism->catalog();
                for (size_t c = 0; c < result.post_faces.size(); ++c)
                {
                    auto guess = context.thermodynamic_trial.cells.at(c);
                    guess.phase_numbers.resize(roles.size());
                    std::vector<int> occupied(roles.size(), 0);
                    for (size_t p = 0; p < roles.size(); ++p)
                    {
                        guess.phase_numbers[p] = context.transport.cells[c].phases[p].population_number;
                        auto& moles = guess.phases.at(p).moles;
                        for (size_t i = 0; i < population_catalog.species().size(); ++i)
                        {
                            moles.at(i) =
                                result.post_faces[c].amount(population_catalog, population_catalog.species()[i].id,
                                    compartment(options.phases[p], i), options.phases[p].population);
                            occupied[p] |= moles[i] > 0;
                        }
                        if (!occupied[p])
                        {
                            guess.phases[p].temperature = 0;
                            guess.phases[p].internal_energy = 0;
                        }
                    }
                    // Final post-material moles and Nadv are actual inventories. T/p remains the existing candidate
                    // guess; never manufacture a separate-phase temperature just to call an optional number law.
                    std::vector<BubbleGasPhysics::PopulationNumberEvent> events;
                    if (options.population_events)
                        events = options.population_events->evaluate(
                            {population_catalog, result.post_faces[c], context.transport.cells[c], guess,
                                context.transport.stage, c, context.transport.time_step});
                    auto number =
                        population_source_detail::apply(roles, guess.phase_numbers, occupied, std::move(events));
                    number.source_stage = context.transport.stage;
                    number.schema_fingerprint = population_catalog.fingerprint();
                    if (options.population_events)
                    {
                        number.provider_identity = options.population_events->identity;
                        number.provider_revision = options.population_events->revision;
                    }
                    // Number events change no material, thermal packet or momentum. This is the sole narrowing
                    // of their independent raw dimension; material/energy were narrowed once by the face composer.
                    result.result.sources.at(c).population_number_changes = number.changes;
                    for (size_t p = 0; p < roles.size(); ++p)
                    {
                        result.result.gross_events.at(c).push_back(number.births[p]);
                        result.result.gross_events.at(c).push_back(number.deaths[p]);
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
    Solver::ControllerLease lease;
    std::shared_ptr<const SourceEngine> source_engine;
    std::shared_ptr<Solver::PlanarMotion> motion;
    Diagnostics diagnostics;
    std::uint64_t generation = 0;

    Implementation(std::shared_ptr<Solver> solver, ReactingMixtureOptions controls)
        : flow(std::move(solver)), options(std::move(controls)), lease(flow->acquire_controller())
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
                source_engine = std::make_shared<const SourceEngine>(SourceEngine{flow, options, {}});
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
                signature << std::hexfloat << options.mechanism->catalog().fingerprint() << ':' << bool(options.source)
                          << ':' << options.source_identity.size() << ':' << options.source_identity << ':'
                          << options.source_revision << ':' << options.maximum_correctors << ':'
                          << options.source_relative_tolerance << ':' << options.liquid_phase << ':'
                          << options.heat_phase << ':' << bool(options.partition_heat) << ':'
                          << options.integration.maximumSubstep << ':' << options.integration.maximumSubsteps << ':'
                          << options.integration.maximumRetries << ':' << int(options.mechanism->energy_convention());
                signature << ':' << options.phases.size() << ':' << options.mechanism->reactions().size();
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
                    engine = std::make_shared<SourceEngine>(SourceEngine{flow, options, {}});
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
                    flow->mesh(), catalog_alias, std::move(prepared_options), std::move(prepared_bindings));
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
    }

    Diagnostics totals(const std::vector<SpeciesSourceLedger>& ledgers, std::span<const Solver::CellSources> applied,
        std::span<const MixturePopulationEventReceipt> populations,
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
                // These are the final applied physical packets, after every raw face/source cancellation.
                // Keep one rank-local sum across all cells and phases rather than first rounding cell totals.
                require(!options.interphase || thermal.size() == applied.size(),
                    "Final thermal diagnostic receipt row mismatch.");
                for (size_t c = 0; c < applied.size(); ++c)
                {
                    const auto& energies =
                        options.interphase ? thermal[c].applied_phase_energy : applied[c].internal_energy;
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
                for (const auto& receipt : populations)
                {
                    require(receipt.births.size() == np && receipt.deaths.size() == np && receipt.changes.size() == np,
                        "Population receipt diagnostic shape differs.");
                    for (size_t p = 0; p < np; ++p)
                    {
                        totals[p] += receipt.births[p];
                        totals[np + p] += receipt.deaths[p];
                        totals[2 * np + p] += receipt.changes[p];
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

ReactingMixtureCoupler::Trial ReactingMixtureCoupler::preview(double dt)
{
    const auto& mesh = *d->flow->mesh();
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
                d->options.source_relative_tolerance, d->options.maximum_correctors});
        });
    auto flow_trial =
        d->motion ? d->flow->preview_ale_with_sources(dt, *d->motion, target_surface, std::move(provider), &d->lease)
                  : d->flow->preview_with_sources(dt, std::move(provider), &d->lease);
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
            require(events.size() == receipt->prepared.result.gross_events.size() &&
                        std::equal(events.begin(), events.end(), receipt->prepared.result.gross_events.begin()),
                "Reaction receipt does not match the exact accepted gross events.");
        });
    const Solver::SourceStageContext final_context{flow_trial.transport_snapshot(), flow_trial.state(), mesh};
    // ALE receipts were producer/phase/epoch validated inside the active source callback.
    // The exact immutable transport/source/gross equality above binds that proof to the retained trial.
    // Its live geometry is now rolled back, so only fixed-grid previews repeat live-epoch validation.
    if (!d->motion)
        d->source_engine->validate_face_receipts(receipt->prepared, final_context);
    auto statistics = d->totals(receipt->prepared.ledgers, flow_trial.integrated_sources(),
        receipt->prepared.population_events, flow_trial.final_thermal_evaluations(),
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
    Diagnostics candidate;
    collective_detail::collective_local_validation(*d->flow->mesh(), "Reacting mixture commit preflight",
        [&]
        {
            require(trial.d && trial.d->owner == this && trial.d->revision == d->generation,
                "Reacting mixture trial is foreign, stale or already committed.");
            require(d->generation != std::numeric_limits<std::uint64_t>::max(),
                "Reacting mixture controller generation exhausted.");
            candidate = trial.d->diagnostics;
        });
    if (d->motion)
        d->flow->commit_ale(trial.d->flow, *d->motion, &d->lease);
    else
        d->flow->commit(trial.d->flow, &d->lease);
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
    return result;
}
void ReactingMixtureCoupler::restore(const Checkpoint& checkpoint)
{
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
        d->flow->restore(checkpoint.flow, *d->motion, &d->lease);
    else
        d->flow->restore(checkpoint.flow, &d->lease);
    d->diagnostics = std::move(restored);
    ++d->generation;
}
} // namespace SimpleFluid::Reacting
