/** @file ReactingMixtureCoupler.cc
 * @brief Accepted-state chemistry and low-Mach source fixed-point transactions.
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
bool near(double a, double b, double tolerance)
{
    const long double scale = std::abs(static_cast<long double>(a)) + std::abs(static_cast<long double>(b));
    return std::abs(static_cast<long double>(a) - b) <= tolerance * scale;
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
bool same_sources(
    const std::vector<Solver::CellSources>& a, const std::vector<Solver::CellSources>& b, double tolerance)
{
    if (a.size() != b.size())
        return false;
    for (size_t c = 0; c < a.size(); ++c)
    {
        if (a[c].species_moles.size() != b[c].species_moles.size() ||
            a[c].internal_energy.size() != b[c].internal_energy.size())
            return false;
        for (size_t p = 0; p < a[c].species_moles.size(); ++p)
        {
            if (a[c].species_moles[p].size() != b[c].species_moles[p].size())
                return false;
            for (size_t i = 0; i < a[c].species_moles[p].size(); ++i)
                if (!near(a[c].species_moles[p][i], b[c].species_moles[p][i], tolerance))
                    return false;
            if (!near(a[c].internal_energy[p], b[c].internal_energy[p], tolerance))
                return false;
        }
        for (size_t axis = 0; axis < 3; ++axis)
            if (!near(a[c].momentum.component(axis), b[c].momentum.component(axis), tolerance))
                return false;
    }
    return true;
}
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
    Diagnostics diagnostics;
    std::uint64_t generation = 0;

    Implementation(std::shared_ptr<Solver> solver, ReactingMixtureOptions controls)
        : flow(std::move(solver)), options(std::move(controls)), lease(flow->acquire_controller())
    {
        const auto& mesh = *flow->mesh();
        collective_detail::collective_local_validation(mesh, "Reacting mixture controller setup",
            [&]
            {
                require(options.mechanism && !options.phases.empty() && options.maximum_correctors >= 2 &&
                            std::isfinite(options.source_relative_tolerance) && options.source_relative_tolerance > 0 &&
                            options.liquid_phase < options.phases.size() && options.heat_phase < options.phases.size(),
                    "Reacting mixture controller requires a mechanism, phases and valid iteration controls.");
                require(!options.source || !options.source_identity.empty(),
                    "A reaction source needs an explicit identity.");
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
                            physical_phases.insert(phase_for(term.species, term.phase, term.population));
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
                    (void) species(cell);
                }
            });
        std::ostringstream signature;
        signature << std::hexfloat << options.mechanism->catalog().fingerprint() << ':' << bool(options.source) << ':'
                  << options.source_identity.size() << ':' << options.source_identity << ':'
                  << options.maximum_correctors << ':' << options.source_relative_tolerance << ':'
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
    std::pair<std::vector<Solver::CellSources>, std::vector<SpeciesSourceLedger>> prepare(
        const Solver::State& accepted, const Solver::State& trial, double dt) const
    {
        const auto& catalog = options.mechanism->catalog();
        const auto ns = catalog.species().size(), np = options.phases.size();
        std::vector<Solver::CellSources> sources(accepted.cells.size());
        std::vector<SpeciesSourceLedger> ledgers;
        ledgers.reserve(accepted.cells.size());
        for (size_t c = 0; c < accepted.cells.size(); ++c)
        {
            const auto initial = species(accepted.cells[c]);
            const auto& cell = trial.cells.at(c);
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
            ReactionEnvironment environment{accepted.time, temperature, liquid.absolute_pressure, liquid_volume,
                flow->mesh()->cell_volume(static_cast<Solver::Pack::local_ordinal_type>(c))};
            ReactionContext context{catalog, initial, environment};
            const auto input = options.source ? options.source({context, cell, c}, dt) : ReactionSourceInput{};
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
            ledgers.push_back(std::move(chemistry.ledger));
        }
        return {std::move(sources), std::move(ledgers)};
    }

    Diagnostics totals(const std::vector<SpeciesSourceLedger>& ledgers, int correctors,
        const Solver::Diagnostics& flow_statistics) const
    {
        const auto nr = options.mechanism->reactions().size();
        std::vector<Sum> sums(nr + 5);
        for (const auto& ledger : ledgers)
        {
            for (size_t r = 0; r < nr; ++r)
                sums[r] += ledger.reactionExtents.at(r);
            sums[nr] += ledger.reactionHeat;
            sums[nr + 1] += ledger.externalHeat;
            sums[nr + 2] += ledger.transferEnergy;
            sums[nr + 3] += ledger.externalMass;
            sums[nr + 4] += ledger.thermal_energy();
        }
        std::vector<double> local(nr + 5), global(nr + 5);
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
        result.chemical_heat = global[nr];
        result.external_heat = global[nr + 1];
        result.transfer_energy = global[nr + 2];
        result.external_mass = global[nr + 3];
        result.net_thermal_energy = global[nr + 4];
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

ReactingMixtureCoupler::Trial ReactingMixtureCoupler::preview(double dt)
{
    const auto& mesh = *d->flow->mesh();
    collective_detail::collective_local_validation(mesh, "Reacting mixture timestep",
        [&] { require(std::isfinite(dt) && dt > 0, "Reacting mixture timestep must be positive and finite."); });
    collective_detail::require_uniform_value(mesh, dt, "Reacting mixture timestep");
    const auto& accepted = d->flow->state();
    auto thermal_trial = accepted;
    std::vector<Solver::CellSources> previous;
    for (int iteration = 0; iteration < d->options.maximum_correctors; ++iteration)
    {
        std::pair<std::vector<Solver::CellSources>, std::vector<SpeciesSourceLedger>> prepared;
        collective_detail::collective_local_validation(
            mesh, "Reacting mixture source preview", [&] { prepared = d->prepare(accepted, thermal_trial, dt); });
        auto flow_trial = d->flow->preview(dt, prepared.first, &d->lease);
        const int local_converged =
            iteration > 0 && same_sources(previous, prepared.first, d->options.source_relative_tolerance);
        int converged = 0;
        Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_converged, &converged);
        if (converged)
        {
            auto data = std::make_shared<Trial::Data>();
            data->owner = this;
            data->revision = d->generation;
            data->diagnostics = d->totals(prepared.second, iteration + 1, flow_trial.diagnostics());
            data->flow = std::move(flow_trial);
            data->ledgers = std::move(prepared.second);
            Trial result;
            result.d = std::move(data);
            return result;
        }
        thermal_trial = flow_trial.state();
        previous = std::move(prepared.first);
    }
    throw std::runtime_error("Reacting mixture source/thermodynamic fixed point did not converge.");
}

void ReactingMixtureCoupler::commit(const Trial& trial)
{
    Diagnostics candidate;
    collective_detail::collective_local_validation(*d->flow->mesh(), "Reacting mixture commit preflight",
        [&]
        {
            require(trial.d && trial.d->owner == this && trial.d->revision == d->generation,
                "Reacting mixture trial is foreign, stale or already committed.");
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
            restored = checkpoint.diagnostics;
        });
    d->flow->restore(checkpoint.flow, &d->lease);
    d->diagnostics = std::move(restored);
    ++d->generation;
}
} // namespace SimpleFluid::Reacting
