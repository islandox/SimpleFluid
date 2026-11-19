/** @file LowMachQbmmState.hh
 * @brief Private validation of native QBMM phase bindings and canonical stock.
 */
#pragma once
#include "solvers/LowMachMixtureSolver.hh"
#include "solvers/details/LowMachSourceStage.hh"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace SimpleFluid::Reacting::qbmm_detail
{
using Flow = LowMachMixtureSolver;

SIMPLEFLUID_SOLVERS_LOCAL inline void require(bool valid, const char* message)
{
    if (!valid)
        throw std::invalid_argument(message);
}

SIMPLEFLUID_SOLVERS_LOCAL inline void validate_configuration(
    const Flow::Options& options, std::span<const std::shared_ptr<const MixtureThermodynamics>> eos)
{
    const auto& families = options.qbmm_families;
    const auto& model = options.qbmm_model;
    require(model.transport.mode == FVM::AssemblyMode::Transient &&
                model.transport.time == FVM::ScalarTimeScheme::BackwardEuler &&
                model.transport.convection == FVM::ScalarConvectionScheme::Upwind,
        "Native QBMM transport requires transient backward Euler and first-order upwind.");
    require(model.velocity == QbmmModelOptions::Velocity::SharedCarrier,
        "Native QBMM requires the shared carrier velocity with zero slip.");
    require(!model.variable_composition && !model.cross_family_coalescence && !model.bulk_gas_conversion,
        "Native QBMM requires fixed family composition and within-family population events.");
    require(!model.capillarity && !model.independent_node_temperatures,
        "Native QBMM requires common pressure and temperature with zero surface tension.");
    if (families.empty())
        return;
    require(families.size() == eos.size() && options.phase_roles.size() == eos.size(),
        "QBMM family bindings must match the native EOS phase layout.");
    require(
        options.thermal_mode == PhaseThermalMode::Equilibrium, "Native QBMM requires equilibrium phase temperatures.");
    require(std::ranges::any_of(families, [](const auto& family) { return bool(family); }),
        "A native QBMM phase layout requires at least one bound family.");
    for (size_t a = 0; a < families.size(); ++a)
    {
        const auto& family = families[a];
        if (!family)
            continue;
        const auto& role = options.phase_roles[a];
        require(role.role == Flow::PhaseRole::DispersedGas && role.identity == family->population() && eos[a] &&
                    eos[a]->kind() == EOSKind::Gas &&
                    eos[a]->catalog().fingerprint() == family->catalog().fingerprint(),
            "A native QBMM family must bind to its named dispersed-gas EOS phase and ordered catalog.");
    }
}

SIMPLEFLUID_SOLVERS_LOCAL inline void validate_stamps(const Flow::Options& options, const Flow::State& state)
{
    require(state.qbmm_families.size() == options.qbmm_families.size(),
        "Native QBMM family stamps must match the configured phase layout.");
    for (size_t a = 0; a < options.qbmm_families.size(); ++a)
    {
        const auto& family = options.qbmm_families[a];
        const auto& stamp = state.qbmm_families[a];
        require(family ? stamp && *stamp == Flow::qbmm_family_stamp(*family) : !stamp,
            "Native QBMM catalog, family, or mass-scale stamp differs from the configured phase.");
    }
}

SIMPLEFLUID_SOLVERS_LOCAL inline std::vector<std::optional<InversionStatus>> validate_cell(
    const Flow::Options& options, const Flow::CellState& cell)
{
    const auto& families = options.qbmm_families;
    require(
        cell.qbmm_shape.size() == families.size(), "Native QBMM shape entries must match the configured phase layout.");
    if (families.empty())
        return {};
    require(cell.phases.size() == families.size() && cell.phase_numbers.size() == families.size() &&
                cell.drift_velocity.size() == families.size(),
        "Native QBMM cell phase, population number, or velocity layout differs.");
    std::vector<std::optional<InversionStatus>> statuses(families.size());
    for (size_t a = 0; a < families.size(); ++a)
    {
        for (size_t axis = 0; axis < 3; ++axis)
            require(
                cell.drift_velocity[a].component(axis) == 0, "Native QBMM requires exactly zero supplied phase slip.");
        const auto& family = families[a];
        const auto& shape = cell.qbmm_shape[a];
        require(bool(shape) == bool(family), "Native QBMM requires shape precisely for each bound family.");
        if (!family)
            continue;
        const auto& number = cell.phase_numbers[a];
        const auto& phase = cell.phases[a];
        require(number && phase.thermodynamics && phase.thermodynamics->kind() == EOSKind::Gas,
            "A native QBMM family requires an engaged native population number and gas EOS.");
        const auto quadrature = invert_qbmm_moments(
            qbmm_cell_moments(*family, {options.phase_roles[a].identity, phase.thermodynamics->catalog().fingerprint(),
                                           phase.moles, *number, *shape, family->fingerprint()}));
        if (!quadrature.valid())
            throw std::invalid_argument("Native QBMM family '" + family->population() + "' failed inversion (" +
                                        qbmm_inversion_status_name(quadrature.status) + ").");
        statuses[a] = quadrature.status;
    }
    return statuses;
}

SIMPLEFLUID_SOLVERS_LOCAL inline void validate_state(const Flow::Options& options, const Flow::State& state)
{
    validate_stamps(options, state);
    for (const auto& cell : state.cells)
        (void) validate_cell(options, cell);
}

/** The transport owner invokes this before publishing a snapshot or evaluating sources.
 * Kept distinct from scalar balance checks: nonnegative scalars and globally
 * conserved totals do not establish fixed composition or moment realizability.
 */
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_transport(const Flow::Options& options, const Flow::State& state)
{
    validate_state(options, state);
}

// The caller first validates the native species and count packet layouts.
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_sources(
    const Flow::Options& options, std::span<const Flow::CellSources> sources, bool coherent = false)
{
    const auto& families = options.qbmm_families;
    for (const auto& source : sources)
    {
        require(source.qbmm_shape_changes.empty() || source.qbmm_shape_changes.size() == families.size(),
            "Native QBMM shape-source layout differs from the configured family layout.");
        for (size_t a = 0; a < families.size(); ++a)
        {
            if (!source.qbmm_shape_changes.empty())
            {
                const auto& shape = source.qbmm_shape_changes[a];
                require(!shape || (families[a] && std::isfinite(shape->H2) && std::isfinite(shape->H3) &&
                                      (coherent || (shape->H2 == 0 && shape->H3 == 0))),
                    "Native QBMM shape sources require a bound phase and a coherent source receipt for changes.");
            }
            if (!families[a] || coherent)
                continue;
            require(source.species_moles.empty() ||
                        std::ranges::all_of(source.species_moles[a], [](double change) { return change == 0; }),
                "Native QBMM currently requires zero material sources on bound phases.");
            require(source.population_number_changes.empty() || source.population_number_changes[a] == 0,
                "Native QBMM currently requires zero population number sources on bound phases.");
        }
    }
}

SIMPLEFLUID_SOLVERS_LOCAL inline void validate_raw_value(double actual, long double expected, const char* message)
{
    require(std::isfinite(actual) && std::isfinite(expected) && (actual == 0) == (expected == 0) &&
                std::abs(static_cast<long double>(actual) - expected) <=
                    qbmm_reconstruction_tolerance *
                        std::max(std::abs(static_cast<long double>(actual)), std::abs(expected)),
        message);
}

/** Check one constructive Q1 endpoint against the exact native transport predecessor and packets. */
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_source_receipt(const QbmmPopulationSchema& family,
    const Flow::TransportSnapshot& transport, Flow::SourceStageResult& result, size_t c, size_t a)
{
    using Sum = SimpleFluid::detail::CompensatedSum<long double>;
    using source_stage_detail::mass;
    using source_stage_detail::number_packet;
    using source_stage_detail::shape_packet;
    using source_stage_detail::species_packet;
    using source_stage_detail::validate_source_endpoint;
    const auto& receipt = *result.qbmm_cells[c][a];
    const auto& initial = transport.cells[c].phases[a];
    const auto species = initial.moles.size();
    require(transport.qbmm_families[a] == Flow::qbmm_family_stamp(family) && initial.population_number &&
                initial.qbmm_shape && receipt.family_fingerprint == family.fingerprint() &&
                receipt.final_moles.size() == species,
        "QBMM source receipt family or inventory layout differs from the transport snapshot.");

    const auto number_change = number_packet(result.sources, c, a);
    const auto shape_change = shape_packet(result.sources, c, a);
    validate_source_endpoint(*initial.population_number, number_change, receipt.final_count);
    validate_source_endpoint(initial.qbmm_shape->H2, shape_change.H2, receipt.final_shape.H2);
    validate_source_endpoint(initial.qbmm_shape->H3, shape_change.H3, receipt.final_shape.H3);
    validate_raw_value(shape_change.H2, static_cast<long double>(receipt.final_shape.H2) - initial.qbmm_shape->H2,
        "QBMM signed H2 packet disagrees with the constructive delta.");
    validate_raw_value(shape_change.H3, static_cast<long double>(receipt.final_shape.H3) - initial.qbmm_shape->H3,
        "QBMM signed H3 packet disagrees with the constructive delta.");
    for (size_t s = 0; s < species; ++s)
        validate_source_endpoint(initial.moles[s], species_packet(result.sources, c, a, s), receipt.final_moles[s]);

    // Per-family gross layout stays fixed through rank changes, empty states and changes of event kind.
    Sum births, deaths, coalescence_pairs, breakup_parents, incoming_mass, outgoing_mass;
    Sum produced_H2, consumed_H2, produced_H3, consumed_H3;
    std::array<Sum, 8> reservoir_activity{};
    if (receipt.events.empty())
    {
        require(receipt.final_moles == initial.moles && receipt.final_count == *initial.population_number &&
                    receipt.final_shape == *initial.qbmm_shape && number_change == 0 &&
                    shape_change == QbmmShapeState{},
            "QBMM source changed a family without a constructive event receipt.");
        for (size_t s = 0; s < species; ++s)
            require(species_packet(result.sources, c, a, s) == 0,
                "QBMM source contains material without a constructive event receipt.");
    }
    else
    {
        const auto pre = invert_qbmm_moments(qbmm_cell_moments(
            family, {family.population(), transport.schema_fingerprint, initial.moles, *initial.population_number,
                        *initial.qbmm_shape, receipt.family_fingerprint}));
        require(pre.valid(), "QBMM source receipt predecessor has no supported quadrature.");
        std::array<Sum, 2> remaining;
        Sum raw_count_change, represented_count, H2, H3;
        std::vector<Sum> raw_moles(species);
        for (size_t i = 0; i < pre.active_atoms; ++i)
            remaining[i] += pre.atoms[i].count;
        for (const auto& event : receipt.events)
        {
            require(event.population == family.population() && event.family_fingerprint == family.fingerprint() &&
                        event.catalog_fingerprint == transport.schema_fingerprint && !event.provider.empty() &&
                        event.provider_revision > 0 && event.substep == 0 && event.duration == transport.time_step,
                "QBMM event provenance differs from the family and fixed source interval.");
            for (const auto& donor : event.donors)
            {
                require(donor.index < pre.active_atoms &&
                            donor.atom.mass_over_scale == pre.atoms[donor.index].mass_over_scale &&
                            std::isfinite(donor.atom.count) && donor.atom.count > 0,
                    "QBMM event donor differs from the transported quadrature.");
                remaining[donor.index] += -static_cast<long double>(donor.atom.count);
                require(remaining[donor.index].value() >= 0, "QBMM source overdraws an original donor count.");
                raw_count_change += -static_cast<long double>(donor.atom.count);
            }
            for (const auto& product : event.products)
            {
                require(std::isfinite(product.count) && product.count > 0 && std::isfinite(product.mass_over_scale) &&
                            product.mass_over_scale > 0,
                    "QBMM event product requires finite positive count and size.");
                raw_count_change += product.count;
                represented_count += product.count;
                const long double sized =
                    static_cast<long double>(product.count) * product.mass_over_scale * product.mass_over_scale;
                H2 += sized;
                H3 += sized * product.mass_over_scale;
            }
            for (const auto& allocation : event.transfers)
            {
                const auto& moles = allocation.transfer.moles;
                require(moles.size() == species, "QBMM event material catalog layout differs.");
                for (size_t s = 0; s < species; ++s)
                {
                    require(std::isfinite(moles[s]) && moles[s] >= 0, "QBMM event material transfer is invalid.");
                    raw_moles[s] += allocation.into_family ? moles[s] : -static_cast<long double>(moles[s]);
                }
                (allocation.into_family ? incoming_mass : outgoing_mass) += mass(moles, family.catalog());
                if (const auto* reservoir = std::get_if<QbmmReservoirExchange>(&allocation.transfer.counterpart))
                {
                    const std::array channels{
                        reservoir->energy_j, reservoir->momentum[0], reservoir->momentum[1], reservoir->momentum[2]};
                    for (size_t i = 0; i < channels.size(); ++i)
                        reservoir_activity[2 * i + (channels[i] < 0)] += std::abs(channels[i]);
                }
            }
            switch (event.kind)
            {
                case QbmmEventKind::Birth:
                    births += event.gross_created[0];
                    break;
                case QbmmEventKind::Disappearance:
                    deaths += event.gross_consumed[0];
                    break;
                case QbmmEventKind::Coalescence:
                    coalescence_pairs += event.gross_created[0];
                    break;
                case QbmmEventKind::Breakup:
                    breakup_parents += event.gross_consumed[0];
                    break;
                case QbmmEventKind::Growth:
                case QbmmEventKind::Shrinkage:
                    break;
            }
            produced_H2 += event.gross_created[2];
            consumed_H2 += event.gross_consumed[2];
            produced_H3 += event.gross_created[3];
            consumed_H3 += event.gross_consumed[3];
        }
        for (size_t i = 0; i < pre.active_atoms; ++i)
        {
            const long double size = pre.atoms[i].mass_over_scale;
            represented_count += remaining[i].value();
            const long double sized = remaining[i].value() * size * size;
            H2 += sized;
            H3 += sized * size;
        }
        validate_raw_value(receipt.final_shape.H2, H2.value(), "QBMM constructive H2 disagrees with raw atoms.");
        validate_raw_value(receipt.final_shape.H3, H3.value(), "QBMM constructive H3 disagrees with raw atoms.");
        if (represented_count.value() < .5L * *initial.population_number)
        {
            // Replay a large depletion from its small positive remainder, as in the Q1 kernel.
            validate_raw_value(
                receipt.final_count, represented_count.value(), "QBMM remaining count disagrees with raw products.");
            represented_count += -static_cast<long double>(*initial.population_number);
            validate_raw_value(number_change, represented_count.value(),
                "QBMM signed count packet disagrees with the retained measure.");
        }
        else
        {
            validate_raw_value(
                number_change, raw_count_change.value(), "QBMM signed count packet disagrees with raw atoms.");
            raw_count_change += *initial.population_number;
            validate_raw_value(
                receipt.final_count, raw_count_change.value(), "QBMM constructive count disagrees with raw atoms.");
        }
        for (size_t s = 0; s < species; ++s)
        {
            validate_raw_value(species_packet(result.sources, c, a, s), raw_moles[s].value(),
                "QBMM signed species packet disagrees with raw transfers.");
            raw_moles[s] += initial.moles[s];
            validate_raw_value(receipt.final_moles[s], raw_moles[s].value(),
                "QBMM constructive species disagrees with raw transfers.");
        }
    }
    const auto append_gross = [&](const Sum& sum)
    {
        const double narrowed = static_cast<double>(sum.value());
        require(narrowed >= 0, "QBMM gross activity must be nonnegative.");
        validate_raw_value(narrowed, sum.value(), "QBMM gross activity is not representable.");
        result.gross_events[c].push_back(narrowed);
    };
    for (const auto& sum : {births, deaths, coalescence_pairs, breakup_parents, incoming_mass, outgoing_mass,
             produced_H2, consumed_H2, produced_H3, consumed_H3})
        append_gross(sum);
    for (const auto& sum : reservoir_activity)
        append_gross(sum);
}

SIMPLEFLUID_SOLVERS_LOCAL inline void validate_source_receipts(
    const Flow::Options& options, const Flow::TransportSnapshot& transport, Flow::SourceStageResult& result)
{
    const auto& families = options.qbmm_families;
    const auto cells = transport.cells.size();
    require(result.qbmm_cells.size() == cells, "QBMM source receipt owned-cell layout differs.");
    require(transport.qbmm_families.size() == families.size(), "QBMM transport phase layout differs.");
    if (result.gross_events.empty())
        result.gross_events.resize(cells);
    for (size_t c = 0; c < cells; ++c)
    {
        require(result.qbmm_cells[c].size() == families.size(), "QBMM source receipt phase layout differs.");
        for (size_t a = 0; a < families.size(); ++a)
        {
            const auto& family = families[a];
            const auto& receipt = result.qbmm_cells[c][a];
            require(bool(family) == bool(receipt), "QBMM source receipts must match the bound families.");
            if (family)
                validate_source_receipt(*family, transport, result, c, a);
        }
    }
}
} // namespace SimpleFluid::Reacting::qbmm_detail
