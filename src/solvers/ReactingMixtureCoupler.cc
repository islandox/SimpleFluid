/** @file ReactingMixtureCoupler.cc
 * @brief Transport-first chemistry and immutable low-Mach source transactions.
 */
#include "solvers/ReactingMixtureCoupler.hh"
#include "equations/CollectiveValidation.hh"
#include "utils/CompensatedSum.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <tuple>

namespace SimpleFluid::Reacting
{
namespace
{
using Sum = detail::CompensatedSum<long double>;
using Solver = LowMachMixtureSolver;
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
    std::vector<int> values(text.begin(), text.end()), lower(size), upper(size);
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
        if (a[c].species_moles != b[c].species_moles || a[c].internal_energy != b[c].internal_energy)
            return false;
        for (size_t axis = 0; axis < 3; ++axis)
            if (a[c].momentum.component(axis) != b[c].momentum.component(axis))
                return false;
    }
    return true;
}
struct SourceEngine
{
    std::shared_ptr<Solver> flow;
    ReactingMixtureOptions options;
    struct Prepared
    {
        Solver::SourceStageResult result;
        std::vector<SpeciesSourceLedger> ledgers;
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
    Prepared prepare(const Solver::SourceStageContext& context) const
    {
        const auto& catalog = options.mechanism->catalog();
        const auto& transport = context.transport;
        const auto& trial = context.thermodynamic_trial;
        const double dt = transport.time_step;
        const auto ns = catalog.species().size(), np = options.phases.size();
        require(
            transport.cells.size() == trial.cells.size() && transport.cells.size() == flow->mesh()->num_owned_cells(),
            "Transport/source cell counts do not agree.");
        Prepared prepared;
        auto& sources = prepared.result.sources;
        auto& events = prepared.result.gross_events;
        sources.resize(transport.cells.size());
        events.resize(transport.cells.size());
        std::vector<SpeciesSourceLedger> ledgers;
        ledgers.reserve(transport.cells.size());
        for (size_t c = 0; c < transport.cells.size(); ++c)
        {
            auto cell = trial.cells.at(c);
            require(transport.cells[c].phases.size() == np && cell.phases.size() == np,
                "Transport/source physical phase counts do not agree.");
            for (size_t p = 0; p < np; ++p)
            {
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
            ReactionContext reaction{catalog, initial, environment};
            const auto input =
                options.source ? options.source({reaction, cell, c, transport.cells[c]}, dt) : ReactionSourceInput{};
            auto chemistry =
                advance_reactions(*options.mechanism, initial, environment, dt, input, options.integration);
            auto& output = sources[c];
            output.species_moles.assign(np, std::vector<double>(ns));
            output.internal_energy.assign(np, 0);
            for (size_t p = 0; p < np; ++p)
                for (size_t i = 0; i < ns; ++i)
                {
                    const auto phase = compartment(options.phases[p], i);
                    const auto id = catalog.species()[i].id;
                    Sum delta;
                    delta += chemistry.ledger.chemicalChange.amount(catalog, id, phase, options.phases[p].population);
                    delta += chemistry.ledger.externalChange.amount(catalog, id, phase, options.phases[p].population);
                    output.species_moles[p][i] = static_cast<double>(delta.value());
                }
            Sum declared_energy;
            declared_energy += chemistry.ledger.externalHeat;
            declared_energy += chemistry.ledger.reactionHeat;
            std::vector<Sum> phase_energy(np);
            if (options.partition_heat)
            {
                const auto heat = static_cast<double>(declared_energy.value());
                require(std::isfinite(heat), "Heat allocation input is not representable.");
                const auto heat_parts = options.partition_heat(heat, chemistry.ledger, cell);
                require(heat_parts.size() == np, "Heat allocation must match physical phases.");
                Sum heat_sum, heat_scale;
                for (size_t phase = 0; phase < np; ++phase)
                {
                    const auto part = heat_parts[phase];
                    require(std::isfinite(part), "Phase heat allocation is nonfinite.");
                    phase_energy[phase] += part;
                    heat_sum += part;
                    heat_scale += std::abs(static_cast<long double>(part));
                }
                require(std::abs(heat_sum.value() - declared_energy.value()) <=
                            64 * std::numeric_limits<double>::epsilon() *
                                (heat_scale.value() + std::abs(declared_energy.value())),
                    "Phase heat allocation does not conserve the supplied budget.");
            }
            else
            {
                // Keep independent source channels separate until material-transfer
                // energy has also been accumulated; their large terms may cancel.
                phase_energy[options.heat_phase] += chemistry.ledger.externalHeat;
                phase_energy[options.heat_phase] += chemistry.ledger.reactionHeat;
            }
            for (const auto& transfer : chemistry.ledger.externalTransfers)
            {
                phase_energy[phase_for(transfer.species, transfer.phase, transfer.population)] += transfer.energy;
                declared_energy += transfer.energy;
            }
            Sum applied_energy, applied_scale;
            for (size_t phase = 0; phase < np; ++phase)
            {
                output.internal_energy[phase] = static_cast<double>(phase_energy[phase].value());
                require(std::isfinite(output.internal_energy[phase]), "Physical phase source energy overflow.");
                applied_energy += output.internal_energy[phase];
                applied_scale += std::abs(static_cast<long double>(output.internal_energy[phase]));
            }
            require(std::abs(applied_energy.value() - declared_energy.value()) <=
                        64 * std::numeric_limits<double>::epsilon() *
                            (applied_scale.value() + std::abs(declared_energy.value())),
                "Physical phase source packets lost declared thermal energy.");
            output.momentum = Solver::Vector{chemistry.ledger.externalMomentum};
            events[c] = gross_events(chemistry.ledger);
            ledgers.push_back(std::move(chemistry.ledger));
        }
        prepared.ledgers = std::move(ledgers);
        return prepared;
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
};

struct ReactingMixtureCoupler::Implementation
{
    std::shared_ptr<Solver> flow;
    ReactingMixtureOptions options;
    Solver::ControllerLease lease;
    std::shared_ptr<const SourceEngine> source_engine;
    Diagnostics diagnostics;
    std::uint64_t generation = 0;

    Implementation(std::shared_ptr<Solver> solver, ReactingMixtureOptions controls)
        : flow(std::move(solver)), options(std::move(controls)), lease(flow->acquire_controller()),
          source_engine(std::make_shared<const SourceEngine>(SourceEngine{flow, options}))
    {
        const auto& mesh = *flow->mesh();
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
                const auto& catalog = options.mechanism->catalog();
                const auto ns = catalog.species().size();
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
        std::ostringstream signature;
        signature << std::hexfloat << options.mechanism->catalog().fingerprint() << ':' << bool(options.source) << ':'
                  << options.source_identity.size() << ':' << options.source_identity << ':' << options.source_revision
                  << ':' << options.maximum_correctors << ':' << options.source_relative_tolerance << ':'
                  << options.liquid_phase << ':' << options.heat_phase << ':' << bool(options.partition_heat) << ':'
                  << options.integration.maximumSubstep << ':' << options.integration.maximumSubsteps << ':'
                  << options.integration.maximumRetries << ':' << int(options.mechanism->energy_convention());
        signature << ':' << options.phases.size() << ':' << options.mechanism->reactions().size();
        for (const auto& phase : options.phases)
        {
            signature << ':' << int(phase.phase) << ':' << phase.population.size() << ':' << phase.population << ':'
                      << phase.species_phases.size();
            for (const auto kind : phase.species_phases)
                signature << ':' << int(kind);
        }
        for (const auto& reaction : options.mechanism->reactions())
        {
            signature << ':' << reaction.id << ':' << reaction.heatPerExtent << ':' << reaction.changes.size();
            for (const auto& term : reaction.changes)
                signature << ':' << term.species << ':' << int(term.phase) << ':' << term.population.size() << ':'
                          << term.population << ':' << term.coefficient;
        }
        uniform_text(mesh, signature.str());
    }

    Diagnostics totals(const std::vector<SpeciesSourceLedger>& ledgers, int correctors,
        const Solver::Diagnostics& flow_statistics) const
    {
        const auto nr = options.mechanism->reactions().size();
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
            sums[3 * nr + 4] += ledger.thermal_energy();
        }
        std::vector<double> local(3 * nr + 5), global(3 * nr + 5);
        collective_detail::collective_local_validation(*flow->mesh(), "Reacting source reduction input",
            [&]
            {
                require(
                    local.size() <= size_t(std::numeric_limits<int>::max()), "Too many reaction diagnostic entries.");
                for (size_t i = 0; i < local.size(); ++i)
                {
                    local[i] = static_cast<double>(sums[i].value());
                    require(std::isfinite(local[i]), "Reaction diagnostic overflow.");
                }
            });
        Teuchos::reduceAll(*flow->mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM,
            static_cast<int>(local.size()), local.data(), global.data());
        for (const auto value : global)
            require(std::isfinite(value), "Global reaction diagnostic overflow.");
        Diagnostics result;
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
    auto receipt = std::make_shared<SourceReceipt>();
    auto provider = std::make_shared<const Solver::SourceStageProvider>(
        Solver::SourceStageProvider{"ReactingMixtureCoupler:" + d->options.source_identity, d->options.source_revision,
            d->options.mechanism->catalog().fingerprint(),
            [engine = d->source_engine, receipt](const Solver::SourceStageContext& context)
            {
                SourceEngine::Prepared prepared;
                // User laws and pure chemical kernels are local. Fence their errors
                // before any later mesh exchange or global reduction in this stage.
                collective_detail::collective_local_validation(
                    context.mesh, "Transported reaction source preview", [&] { prepared = engine->prepare(context); });
                receipt->transport = context.transport;
                receipt->prepared = std::move(prepared);
                return receipt->prepared.result;
            },
            d->options.source_relative_tolerance, d->options.maximum_correctors});
    auto flow_trial = d->flow->preview_with_sources(dt, std::move(provider), &d->lease);
    collective_detail::collective_local_validation(mesh, "Reacting mixture immutable source receipt",
        [&]
        {
            require(receipt->transport == flow_trial.transport_snapshot(),
                "Reaction receipt does not match the accepted transport trial.");
            require(identical_sources(receipt->prepared.result.sources, flow_trial.integrated_sources()),
                "Reaction receipt does not match the exact applied source packets.");
            const auto events = flow_trial.integrated_gross_events();
            require(events.size() == receipt->prepared.result.gross_events.size() &&
                        std::equal(events.begin(), events.end(), receipt->prepared.result.gross_events.begin()),
                "Reaction receipt does not match the exact accepted gross events.");
        });
    auto data = std::make_shared<Trial::Data>();
    data->owner = this;
    data->revision = d->generation;
    data->diagnostics =
        d->totals(receipt->prepared.ledgers, flow_trial.diagnostics().source_evaluations, flow_trial.diagnostics());
    data->flow = std::move(flow_trial);
    data->ledgers = std::move(receipt->prepared.ledgers);
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
    result.flow = d->flow->checkpoint();
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
    d->flow->restore(checkpoint.flow, &d->lease);
    d->diagnostics = std::move(restored);
    ++d->generation;
}
} // namespace SimpleFluid::Reacting
