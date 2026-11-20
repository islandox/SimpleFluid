/** Historical dense planar-ALE comparison, reconstructed from the model's public receipts.
 * Keep the layout and arithmetic independent of the solver's convergence visitor and storage.
 */
#pragma once

#include "equations/SpeciesInventoryHistory.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace SimpleFluid::Testing::ALESpeciesDense
{
namespace R = Reacting;
using Phase = R::SpeciesPhase;

enum class SlotKind
{
    Ordinary,
    SignedNet
};
struct Entry
{
    std::string identity;
    double value;
    long double allowance = 0;
    SlotKind kind = SlotKind::Ordinary;
    bool structural_zero = false;
    bool operator==(const Entry&) const = default;
};
using State = std::vector<Entry>;

inline constexpr std::array phases{Phase::Liquid, Phase::Dissolved, Phase::Dispersed, Phase::Resolved};
inline constexpr std::array phase_names{"liquid", "dissolved", "dispersed", "resolved"};

// A forbidden phase alone is insufficient. Only a species confined to Liquid
// has the omissions specified by the ALE convergence contract.
inline bool liquid_only(const R::SpeciesDefinition& species)
{
    return species.allowedPhases.size() == 1 && species.allowedPhases.front() == Phase::Liquid;
}

inline void append_amounts(State& out, const std::string& prefix, const R::SpeciesCatalog& catalog,
    const std::vector<std::string>& bins, const R::CellSpeciesState* state)
{
    const auto& species = catalog.species();
    for (std::size_t phase = 0; phase < phases.size(); ++phase)
        for (std::size_t i = 0; i < species.size(); ++i)
            out.push_back({prefix + phase_names[phase] + "/" + species[i].name,
                state ? state->amounts.at(phase * species.size() + i) : 0., 0.L, SlotKind::Ordinary,
                phases[phase] != Phase::Liquid && liquid_only(species[i])});
    for (const auto& bin : bins)
    {
        const R::SpeciesPopulationState* population = nullptr;
        if (state)
        {
            const auto found = std::ranges::find(state->dispersedPopulations, bin, &R::SpeciesPopulationState::id);
            if (found == state->dispersedPopulations.end())
                throw std::logic_error("Missing dense oracle population " + bin);
            population = &*found;
        }
        for (std::size_t i = 0; i < species.size(); ++i)
            out.push_back({prefix + "population/" + bin + "/" + species[i].name,
                population ? population->amounts.at(i) : 0., 0.L, SlotKind::Ordinary, liquid_only(species[i])});
    }
}

inline void append_carried(State& out, const std::string& prefix, const R::SpeciesSourceLedger* ledger, Phase phase,
    std::string_view population)
{
    std::array<detail::CompensatedSum<long double>, 8> sums{};
    if (ledger)
        for (const auto& packet : ledger->externalTransfers)
        {
            if (packet.phase != phase || packet.population != population)
                continue;
            const std::array components{packet.energy, packet.momentum[0], packet.momentum[1], packet.momentum[2]};
            for (std::size_t i = 0; i < components.size(); ++i)
            {
                const auto x = static_cast<long double>(components[i]);
                sums[2 * i] += std::max(0.L, x);
                sums[2 * i + 1] += std::max(0.L, -x);
            }
        }
    constexpr std::array channels{"energy+", "energy-", "px+", "px-", "py+", "py-", "pz+", "pz-"};
    for (std::size_t i = 0; i < sums.size(); ++i)
        out.push_back({prefix + channels[i], static_cast<double>(sums[i].value())});
}

inline long double signed_allowance(const R::SpeciesCompartmentEvents& event)
{
    if (std::max({event.chemical_production, event.external_supply, event.interphase_inflow}) == 0 ||
        std::max({event.chemical_consumption, event.external_withdrawal, event.interphase_outflow}) == 0)
        return 0;
    detail::CompensatedSum<long double> sum;
    for (double gross : {event.chemical_production, event.external_supply, event.interphase_inflow,
             event.chemical_consumption, event.external_withdrawal, event.interphase_outflow})
        sum += static_cast<long double>(gross) * std::numeric_limits<double>::epsilon();
    return sum.value();
}

template<class Model, class Mesh> State capture(const Model& gas, const Mesh& mesh)
{
    State out;
    const auto& bins = gas.options().population_ids;
    const auto& catalog = gas.options().reaction_mechanism->catalog();
    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<typename Model::local_ordinal_type>(owned);
        const auto prefix = "cell[" + std::to_string(owned) + "]/";
        const auto stocks = gas.cell_species_state(cell);
        append_amounts(out, prefix + "stock/", catalog, bins, &stocks);
        for (const auto& bin : bins)
            out.push_back(
                {prefix + "number/" + bin, gas.population_number_field(bin).value(cell) * mesh.cell_volume(cell)});
        out.push_back({prefix + "thermal source", gas.thermal_source().value(cell)});
        out.push_back({prefix + "liquid mass density", gas.liquid_mass_density().value(cell)});
        out.push_back({prefix + "pure liquid density", gas.pure_liquid_density().value(cell)});
        out.push_back({prefix + "absolute pressure", gas.absolute_pressure().value(cell)});

        const auto ledger = [&](const auto& rows, const std::string& name, bool carried)
        {
            const auto path = prefix + name + "/";
            const auto* row = owned < rows.size() ? &rows[owned] : nullptr;
            append_amounts(out, path + "chemical/", catalog, bins, row ? &row->chemicalChange : nullptr);
            append_amounts(out, path + "external/", catalog, bins, row ? &row->externalChange : nullptr);
            if (carried)
            {
                out.push_back({path + "external heat", row ? row->externalHeat : 0.});
                out.push_back({path + "reaction heat", row ? row->reactionHeat : 0.});
                out.push_back({path + "transfer energy", row ? row->transferEnergy : 0.});
                for (std::size_t i = 0; i < phases.size(); ++i)
                    append_carried(out, path + phase_names[i] + "/", row, phases[i], {});
                for (const auto& bin : bins)
                    append_carried(out, path + "population/" + bin + "/", row, Phase::Dispersed, bin);
            }
        };
        ledger(gas.source_ledgers(), "source", true);
        ledger(gas.population_ledgers(), "population", true);
        ledger(gas.escape_ledgers(), "escape", false);
    }

    const auto& events = gas.species_step_events();
    for (const auto& event : events)
    {
        const auto& species = catalog.species().at(catalog.index(event.species));
        const bool omitted = liquid_only(species) && event.phase != Phase::Liquid;
        const auto path = "event/" + species.name + "/" + phase_names.at(static_cast<std::size_t>(event.phase)) + "/" +
                          (event.population.empty() ? "" : event.population + "/");
        out.push_back({path + "chemical production", event.chemical_production, 0, SlotKind::Ordinary, omitted});
        out.push_back({path + "chemical consumption", event.chemical_consumption, 0, SlotKind::Ordinary, omitted});
        out.push_back({path + "external supply", event.external_supply, 0, SlotKind::Ordinary, omitted});
        out.push_back({path + "external withdrawal", event.external_withdrawal, 0, SlotKind::Ordinary, omitted});
        out.push_back({path + "interphase in", event.interphase_inflow, 0, SlotKind::Ordinary, omitted});
        out.push_back({path + "interphase out", event.interphase_outflow, 0, SlotKind::Ordinary, omitted});
        out.push_back({path + "physical outflow", event.physical_outflow, 0, SlotKind::Ordinary, omitted});
        out.push_back({path + "signed source", event.signed_source_change, signed_allowance(event), SlotKind::SignedNet,
            omitted});
    }
    for (const auto& subset : gas.species_subset_source_changes())
    {
        long double allowance = 0;
        const bool liquid = catalog.allows(subset.species, Phase::Liquid);
        if (liquid != catalog.allows(subset.species, Phase::Dissolved))
            for (const auto& event : events)
                if (event.species == subset.species && event.phase == (liquid ? Phase::Liquid : Phase::Dissolved) &&
                    event.population.empty())
                    allowance = signed_allowance(event);
        const auto& species = catalog.species().at(catalog.index(subset.species));
        const auto path = "subset/" + species.name + "/";
        out.push_back({path + "gas", subset.gas_source_moles, 0, SlotKind::Ordinary, liquid_only(species)});
        out.push_back({path + "solution", subset.solution_source_moles, allowance, SlotKind::SignedNet});
    }
    return out;
}

struct Maximum
{
    double residual = 0;
    std::size_t slot = std::numeric_limits<std::size_t>::max();
};

inline Maximum compare(const State& before, const State& after)
{
    if (before.size() != after.size())
        throw std::logic_error("Dense ALE species oracle changed width.");
    Maximum result;
    for (std::size_t i = 0; i < before.size(); ++i)
    {
        if (before[i].identity != after[i].identity || before[i].kind != after[i].kind)
            throw std::logic_error("Dense ALE species slot changed at " + before[i].identity);
        const double a = before[i].value, b = after[i].value;
        const long double da = before[i].allowance, db = after[i].allowance;
        const double scale = std::max(std::abs(a), std::abs(b));
        if (!std::isfinite(scale) || !std::isfinite(a) || !std::isfinite(b))
            throw std::logic_error("Nonfinite dense ALE species value.");
        if (scale == 0)
            continue;
        const double residual =
            da == 0 && db == 0
                ? std::abs(b - a) / scale
                : static_cast<double>(std::max(0.L, std::abs(static_cast<long double>(b) - a) - da - db) / scale);
        if (residual > result.residual)
            result = {residual, i};
    }
    return result;
}
} // namespace SimpleFluid::Testing::ALESpeciesDense
