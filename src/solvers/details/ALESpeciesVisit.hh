#pragma once

#include "equations/ReactionMechanism.hh"
#include "solvers/details/ALESpeciesConvergence.hh"
#include "solvers/details/SpeciesSurfaceAccounting.hh"
#include "utils/CompensatedSum.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::detail::ale_species
{
class LiquidOnlySlots
{
public:
    explicit LiquidOnlySlots(const Reacting::SpeciesCatalog& catalog)
        : d_species_count(catalog.species().size()), d_catalog_fingerprint(catalog.fingerprint())
    {
        for (std::size_t i = 0; i < d_species_count; ++i)
        {
            if (is_liquid_only(catalog.species()[i]))
                ++d_count;
            else if (d_retained.empty() || d_retained.back().end != i)
                d_retained.push_back({i, i + 1});
            else
                ++d_retained.back().end;
        }
    }

    std::size_t count() const noexcept { return d_count; }

    bool liquid_only(const Reacting::SpeciesCatalog& catalog, Reacting::SpeciesId id) const
    {
        return d_count != 0 && is_liquid_only(catalog.species()[catalog.index(id)]);
    }

    template<class Event>
    bool omit_event(const Reacting::SpeciesCatalog& catalog, const Event& event) const
    {
        return event.phase != Reacting::SpeciesPhase::Liquid && liquid_only(catalog, event.species);
    }

    void require_catalog(const Reacting::SpeciesCatalog& catalog) const
    {
        if (catalog.fingerprint() != d_catalog_fingerprint)
            throw std::logic_error("ALE liquid-only slot plan has a foreign catalog.");
    }

    template<class Sink, class ValueAt>
    void visit_nonliquid(Sink& sink, const ValueAt& value_at) const
    {
        std::size_t next = 0;
        for (const auto& run : d_retained)
        {
            if (next < run.begin)
                sink.skip(run.begin - next);
            for (std::size_t i = run.begin; i < run.end; ++i)
                sink.ordinary(value_at(i));
            next = run.end;
        }
        if (next < d_species_count)
            sink.skip(d_species_count - next);
    }

private:
    static bool is_liquid_only(const Reacting::SpeciesDefinition& species)
    {
        return species.allowedPhases.size() == 1 && species.allowedPhases.front() == Reacting::SpeciesPhase::Liquid;
    }

    struct Run
    {
        std::size_t begin, end;
    };

    std::size_t d_species_count;
    std::uint64_t d_catalog_fingerprint;
    std::size_t d_count = 0;
    std::vector<Run> d_retained;
};

template<class Model, class Mesh, class Sink>
void visit_typed_gas_state(const Model& model, const Mesh& mesh, const LiquidOnlySlots& plan, Sink& sink)
{
    using scalar_type = typename Model::scalar_type;
    const auto& options = model.options();
    const auto& catalog = options.reaction_mechanism->catalog();
    plan.require_catalog(catalog);
    const auto species_count = catalog.species().size();
    const auto visit_base_amounts = [&](const auto& value_at)
    {
        for (std::size_t i = 0; i < species_count; ++i)
            sink.ordinary(value_at(i));
        for (std::size_t phase = 1; phase < static_cast<std::size_t>(Reacting::SpeciesPhase::Count); ++phase)
        {
            const auto offset = phase * species_count;
            plan.visit_nonliquid(sink, [&](std::size_t i) { return value_at(offset + i); });
        }
    };
    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<typename Model::local_ordinal_type>(owned);
        const auto species = model.cell_species_state(cell);
        visit_base_amounts([&](std::size_t i) { return species.amounts[i]; });
        for (const auto& population : species.dispersedPopulations)
            plan.visit_nonliquid(sink, [&](std::size_t i) { return population.amounts[i]; });
        for (const auto& population : options.population_ids)
            sink.ordinary(
                model.population_number_field(population).value(cell) * mesh.cell_volume(cell));
        // The subtractive material-volume diagnostic does not drive the ALE target.
        // Exact old/new material volumes and transport packets have their own physical gate.
        if (!std::isfinite(model.material_volume_source().value(cell)))
            throw std::runtime_error("ALE material-volume diagnostic is nonfinite.");
        for (const auto* field : {&model.thermal_source(), &model.liquid_mass_density(),
                 &model.pure_liquid_density(), &model.absolute_pressure()})
            sink.ordinary(field->value(cell));
        const auto append_delta = [&](const Reacting::CellSpeciesState* delta)
        {
            if (delta && (delta->amounts.size() != species.amounts.size() ||
                             delta->dispersedPopulations.size() != species.dispersedPopulations.size()))
                throw std::logic_error("ALE source delta has a foreign phase/population shape.");
            visit_base_amounts([&](std::size_t i) { return delta ? delta->amounts[i] : scalar_type{}; });
            for (std::size_t pop = 0; pop < species.dispersedPopulations.size(); ++pop)
            {
                if (delta && (delta->dispersedPopulations[pop].id != species.dispersedPopulations[pop].id ||
                                 delta->dispersedPopulations[pop].amounts.size() !=
                                     species.dispersedPopulations[pop].amounts.size()))
                    throw std::logic_error("ALE source delta has a foreign population identity.");
                plan.visit_nonliquid(sink,
                    [&](std::size_t i) { return delta ? delta->dispersedPopulations[pop].amounts[i] : scalar_type{}; });
            }
        };
        for (const auto* ledgers :
            {&model.source_ledgers(), &model.population_ledgers(), &model.escape_ledgers()})
        {
            const auto* ledger = owned < ledgers->size() ? &(*ledgers)[owned] : nullptr;
            append_delta(ledger ? &ledger->chemicalChange : nullptr);
            append_delta(ledger ? &ledger->externalChange : nullptr);
            // Escaped moles drive headspace/volume closure. Their carried h/momentum are
            // passive traces in this Boussinesq/prescribed-temperature headspace model;
            // they remain fully recorded and validated but are not nonlinear source inputs.
            if (ledgers == &model.escape_ledgers())
                continue;
            sink.ordinary(ledger ? ledger->externalHeat : scalar_type{});
            sink.ordinary(ledger ? ledger->reactionHeat : scalar_type{});
            sink.ordinary(ledger ? ledger->transferEnergy : scalar_type{});
            const auto compartments = std::size_t{4} + options.population_ids.size();
            std::vector<detail::CompensatedSum<long double>> carried(8 * compartments);
            if (ledger)
                for (const auto& event : ledger->externalTransfers)
                {
                    std::size_t compartment = static_cast<std::size_t>(event.phase);
                    if (!event.population.empty())
                    {
                        const auto found = std::ranges::find(options.population_ids, event.population);
                        if (event.phase != Reacting::SpeciesPhase::Dispersed ||
                            found == options.population_ids.end())
                            throw std::logic_error("ALE transfer budget has a foreign population.");
                        compartment = 4 + static_cast<std::size_t>(found - options.population_ids.begin());
                    }
                    if (compartment >= compartments)
                        throw std::logic_error("ALE transfer budget has an invalid phase.");
                    const auto add = [&](std::size_t offset, double value)
                    {
                        carried[8 * compartment + offset + (value < 0 ? 1 : 0)] +=
                            std::abs(static_cast<long double>(value));
                    };
                    add(0, event.energy);
                    for (std::size_t axis = 0; axis < 3; ++axis)
                        add(2 + 2 * axis, event.momentum[axis]);
                }
            for (const auto& budget : carried)
                sink.ordinary(detail::surface_species::narrow(budget.value()));
        }
    }
    for (const auto& event : model.species_step_events())
    {
        if (plan.omit_event(catalog, event))
        {
            (void) detail::ale_species::source_cancellation_allowance(event);
            sink.skip(8);
            continue;
        }
        sink.ordinary(event.chemical_production);
        sink.ordinary(event.chemical_consumption);
        sink.ordinary(event.external_supply);
        sink.ordinary(event.external_withdrawal);
        sink.ordinary(event.interphase_inflow);
        sink.ordinary(event.interphase_outflow);
        sink.ordinary(event.physical_outflow);
        // Keep the raw signed receipt. Only its own opposing source operands
        // establish arithmetic uncertainty; physical escape never contributes.
        sink.signed_net(event.signed_source_change,
            detail::ale_species::source_cancellation_allowance(event));
    }
    for (const auto& event : model.species_subset_source_changes())
    {
        if (plan.liquid_only(catalog, event.species))
            sink.skip(1);
        else
            sink.ordinary(event.gas_source_moles);
        // A sole allowed solution phase is the identical source net. General
        // multi-phase solution subsets and all gas subsets retain no allowance.
        sink.signed_net(event.solution_source_moles,
            detail::ale_species::solution_subset_cancellation_allowance(
                catalog, event.species, model.species_step_events()));
    }
}

template<class Model, class Mesh, class Sink>
void visit_typed_gas_state(const Model& model, const Mesh& mesh, Sink& sink)
{
    const LiquidOnlySlots plan(model.options().reaction_mechanism->catalog());
    visit_typed_gas_state(model, mesh, plan, sink);
}
}
