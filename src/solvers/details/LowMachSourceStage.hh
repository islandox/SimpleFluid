/** @file LowMachSourceStage.hh
 * @brief Private local source validation and reference-safe fixed-point residuals.
 */
#pragma once
#include "solvers/LowMachMixtureSolver.hh"
#include "utils/CompensatedSum.hh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace SimpleFluid::Reacting::source_stage_detail
{
using Flow = LowMachMixtureSolver;
SIMPLEFLUID_SOLVERS_LOCAL inline void require(bool valid, const char* message)
{
    if (!valid)
        throw std::invalid_argument(message);
}
/** Use the endpoint operands only so large turnover cannot hide a lost small stock. */
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_source_endpoint(double transported, double packet, double actual)
{
    require(
        std::isfinite(transported) && transported >= 0 && std::isfinite(packet) && std::isfinite(actual) && actual >= 0,
        "Nonfinite or negative source endpoint input.");
    SimpleFluid::detail::CompensatedSum<long double> replay;
    replay += transported;
    replay += packet;
    const auto recovered = replay.value();
    const auto endpoint = static_cast<long double>(actual);
    SimpleFluid::detail::CompensatedSum<long double> error;
    error += recovered;
    error += -endpoint;
    const auto bound = 64 * std::numeric_limits<double>::epsilon() * (std::abs(recovered) + std::abs(endpoint));
    require(std::isfinite(recovered) && recovered >= 0 && (recovered == 0) == (actual == 0) &&
                std::isfinite(error.value()) && std::isfinite(bound) && std::abs(error.value()) <= bound,
        "Rounded source packet cannot replay its actual sequential endpoint.");
}
SIMPLEFLUID_SOLVERS_LOCAL inline long double mass(std::span<const double> moles, const SpeciesCatalog& catalog)
{
    long double total = 0;
    for (size_t s = 0; s < moles.size(); ++s)
        total += static_cast<long double>(moles[s]) * catalog.species()[s].molarMass;
    return total;
}
SIMPLEFLUID_SOLVERS_LOCAL inline double normalized(long double difference, long double scale, double tolerance)
{
    if (difference == 0)
        return 0;
    if (!(scale > 0) || !std::isfinite(scale) || !std::isfinite(difference))
        return std::numeric_limits<double>::infinity();
    return static_cast<double>(std::abs(difference) / scale / tolerance);
}
SIMPLEFLUID_SOLVERS_LOCAL inline double species_packet(
    std::span<const Flow::CellSources> cells, size_t c, size_t a, size_t s)
{
    return cells.empty() || cells[c].species_moles.empty() ? 0 : cells[c].species_moles[a][s];
}
SIMPLEFLUID_SOLVERS_LOCAL inline double energy_packet(std::span<const Flow::CellSources> cells, size_t c, size_t a)
{
    return cells.empty() || cells[c].internal_energy.empty() ? 0 : cells[c].internal_energy[a];
}
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_packets(
    std::span<const Flow::CellSources> sources, size_t cells, size_t phases, size_t species)
{
    require(sources.empty() || sources.size() == cells, "Source stage returned a different owned-cell layout.");
    for (const auto& source : sources)
    {
        require((source.species_moles.empty() || source.species_moles.size() == phases) &&
                    (source.internal_energy.empty() || source.internal_energy.size() == phases),
            "Source stage changed phase layout.");
        for (const auto& phase : source.species_moles)
        {
            require(phase.size() == species, "Source stage changed species layout.");
            for (const double value : phase)
                require(std::isfinite(value), "Nonfinite source-stage molar packet.");
        }
        for (const double value : source.internal_energy)
            require(std::isfinite(value), "Nonfinite source-stage energy packet.");
        for (size_t axis = 0; axis < 3; ++axis)
            require(std::isfinite(source.momentum.component(axis)), "Nonfinite source-stage impulse.");
    }
}
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_events(const std::vector<std::vector<double>>& events, size_t cells)
{
    require(events.empty() || events.size() == cells, "Source stage returned a different gross-event cell layout.");
    for (const auto& cell : events)
        for (const double value : cell)
            require(std::isfinite(value) && value >= 0, "Gross source events must be finite and nonnegative.");
}
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_event_shape(
    const std::vector<std::vector<double>>& current, const std::vector<std::vector<double>>& previous)
{
    require(current.size() == previous.size(), "Source stage changed its gross-event layout.");
    for (size_t c = 0; c < current.size(); ++c)
        require(current[c].size() == previous[c].size(), "Source stage changed its gross-event dimensions.");
}
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_fourier_resolution(
    const Flow::FourierGrossResolutions& resolution, const std::vector<std::vector<double>>& events)
{
    require(resolution.empty() || resolution.size() == events.size(), "Fourier gross resolution rows differ.");
    for (size_t c = 0; c < resolution.size(); ++c)
    {
        size_t first_available = 0;
        for (const auto& stage : resolution[c])
        {
            const auto channel = stage.positive_channel();
            require(channel >= first_available && channel < events[c].size() && events[c].size() - channel >= 2,
                "Fourier gross resolution channel pair is invalid or overlaps.");
            first_available = channel + 2;
            std::array<SimpleFluid::detail::CompensatedSum<long double>, 2> gross;
            const auto faces = stage.faces();
            for (size_t i = 0; i < faces.size(); ++i)
            {
                const auto& face = faces[i];
                require(i == 0 || faces[i - 1].geometry_identity < face.geometry_identity,
                    "Fourier resolution faces must have unique ordered canonical identities.");
                require(std::isfinite(face.energy) && std::isfinite(face.decrease) && face.decrease >= 0 &&
                            std::isfinite(face.increase) && face.increase >= 0,
                    "Fourier packet and temperature resolution must be finite, with nonnegative bounds.");
                gross[face.energy < 0 ? 1 : 0] += std::abs(static_cast<long double>(face.energy));
            }
            for (size_t sign = 0; sign < gross.size(); ++sign)
                require(static_cast<double>(gross[sign].value()) == events[c][channel + sign],
                    "Fourier face resolution packets must reproduce the exact directional gross totals.");
        }
    }
}

SIMPLEFLUID_SOLVERS_LOCAL inline double outward_temperature_resolution(long double value)
{
    // Use strict comparison for subnormal bounds. Also exclude the smallest
    // normal value, which a subnormal product can round to when long double
    // has binary64 precision. Out-of-range bounds likewise grant no allowance.
    if (value <= std::numeric_limits<double>::min() || value > std::numeric_limits<double>::max())
        return 0;
    const auto rounded = static_cast<double>(value);
    return rounded < value ? std::nextafter(rounded, std::numeric_limits<double>::infinity()) : rounded;
}

SIMPLEFLUID_SOLVERS_LOCAL inline std::array<double, 2> fourier_temperature_bounds(
    double owner, double neighbor, long double integrated_conductance)
{
    const auto spacing = [](double temperature)
    {
        const long double below = temperature - std::nextafter(temperature, 0.);
        const double above = std::nextafter(temperature, std::numeric_limits<double>::infinity());
        return std::array{below, std::isfinite(above) ? static_cast<long double>(above) - temperature : below};
    };
    const auto [owner_down, owner_up] = spacing(owner);
    const auto [neighbor_down, neighbor_up] = spacing(neighbor);
    // The faces use owner minus neighbor. Each binary64 operand extends to
    // the midpoint toward its adjacent representable value in that direction.
    return {outward_temperature_resolution(integrated_conductance * (owner_down + neighbor_up) * .5L),
        outward_temperature_resolution(integrated_conductance * (owner_up + neighbor_down) * .5L)};
}

struct GrossComparison
{
    double residual = 0;
    double fourier_uncertainty = 0;
    std::optional<Flow::GrossResidualWitness> winner;

    void include(const GrossComparison& local)
    {
        const auto key = [](const Flow::GrossResidualWitness& value)
        {
            return std::tie(value.geometry_cell, value.channel_metadata.stage, value.channel_metadata.stage_index,
                value.channel, value.geometry_face);
        };
        if (local.winner && (!winner || local.residual > residual ||
                                (local.residual == residual && key(*local.winner) < key(*winner))))
            winner = local.winner;
        residual = std::max(residual, local.residual);
        fourier_uncertainty = std::max(fourier_uncertainty, local.fourier_uncertainty);
    }
};

SIMPLEFLUID_SOLVERS_LOCAL inline GrossComparison fourier_face_error(
    std::span<const Flow::FourierFaceResolution> current, std::span<const Flow::FourierFaceResolution> previous,
    double tolerance, Flow::Pack::global_ordinal_type cell = {}, size_t stage = 0, size_t channel = 0)
{
    require(current.size() == previous.size(), "Fourier resolution face count changed between correctors.");
    GrossComparison maximum;
    for (size_t i = 0; i < current.size(); ++i)
    {
        const auto& now = current[i];
        const auto& old = previous[i];
        require(now.geometry_identity == old.geometry_identity, "Fourier resolution canonical face changed.");
        const auto direction = [](double packet) { return (packet > 0) - (packet < 0); };
        const double scale = std::max(std::abs(now.energy), std::abs(old.energy));
        const Flow::GrossResidualWitness witness{.geometry_cell = cell, .geometry_face = now.geometry_identity,
            .fourier_stage = stage, .channel_metadata = {Flow::GrossResidualStage::FaceTransport, stage, std::nullopt},
            .channel = channel, .current_direction = direction(now.energy), .previous_direction = direction(old.energy),
            .current_packet = now.energy, .previous_packet = old.energy, .scale = scale, .relative_tolerance = tolerance,
            .current_decrease = now.decrease, .current_increase = now.increase,
            .previous_decrease = old.decrease, .previous_increase = old.increase};
        if (now.energy == old.energy)
        {
            maximum.include({0, 0, witness});
            continue;
        }
        const auto& greater = now.energy > old.energy ? now : old;
        const auto& lesser = now.energy > old.energy ? old : now;
        const long double high = greater.energy, low = lesser.energy;
        const long double decrease = greater.decrease, increase = lesser.increase;
        // Half-scale only at extreme magnitudes so even binary64 long double
        // can subtract a reversal or sum its bounds without overflow.
        constexpr long double half_maximum = std::numeric_limits<long double>::max() / 2;
        const long double factor =
            std::max({static_cast<long double>(scale), decrease, increase}) > half_maximum ? .5L : 1.L;
        const long double uncertainty = factor * decrease + factor * increase;
        const long double gap = std::max(0.L, (factor * high - factor * low) - uncertainty);
        maximum.include({normalized(gap, factor * scale, tolerance),
            static_cast<double>(std::min(uncertainty, factor * std::numeric_limits<double>::max()) / factor), witness});
    }
    return maximum;
}

SIMPLEFLUID_SOLVERS_LOCAL inline double event_error(
    const std::vector<std::vector<double>>& current, const std::vector<std::vector<double>>& previous, double tolerance)
{
    validate_event_shape(current, previous);
    double maximum = 0;
    for (size_t c = 0; c < current.size(); ++c)
        for (size_t i = 0; i < current[c].size(); ++i)
        {
            const long double now = current[c][i], old = previous[c][i];
            maximum = std::max(maximum, normalized(now - old, std::max(now, old), tolerance));
        }
    return maximum;
}
SIMPLEFLUID_SOLVERS_LOCAL inline GrossComparison event_error(const std::vector<std::vector<double>>& current,
    const std::vector<std::vector<double>>& previous, double tolerance,
    const Flow::FourierGrossResolutions& current_fourier, const Flow::FourierGrossResolutions& previous_fourier,
    const Flow::Mesh& mesh, std::span<const Flow::GrossChannelMetadata> metadata)
{
    require(current_fourier.size() == previous_fourier.size(), "Fourier gross resolution rows changed.");
    validate_event_shape(current, previous);
    GrossComparison maximum;
    for (size_t c = 0; c < current.size(); ++c)
    {
        const auto cell = mesh.cell_geometry_global_id(static_cast<Flow::Pack::local_ordinal_type>(c));
        using FourierStages = std::span<const Flow::FourierGrossResolution>;
        const auto now_stages = current_fourier.empty() ? FourierStages{} : FourierStages(current_fourier[c]);
        const auto old_stages = previous_fourier.empty() ? FourierStages{} : FourierStages(previous_fourier[c]);
        require(now_stages.size() == old_stages.size(), "Fourier gross resolution stage count changed.");
        for (size_t stage = 0; stage < now_stages.size(); ++stage)
            require(now_stages[stage].positive_channel() == old_stages[stage].positive_channel() &&
                        now_stages[stage].face_stage() == old_stages[stage].face_stage(),
                "Fourier gross resolution stage or channel changed.");
        size_t stage = 0;
        for (size_t i = 0; i < current[c].size(); ++i)
        {
            if (stage < now_stages.size() && i == now_stages[stage].positive_channel())
            {
                // The raw directional aggregates stay in the receipt. Compare
                // their constituent signed contacts with each face's own scale.
                const auto& now = now_stages[stage];
                maximum.include(fourier_face_error(
                    now.faces(), old_stages[stage].faces(), tolerance, cell, now.face_stage(), i));
                ++stage;
                ++i;
                continue;
            }
            const double now = current[c][i], old = previous[c][i], scale = std::max(now, old);
            const auto channel = metadata.empty() ? Flow::GrossChannelMetadata{} : metadata[i];
            const Flow::GrossResidualWitness witness{.geometry_cell = cell, .geometry_face = std::nullopt,
                .fourier_stage = std::nullopt, .channel_metadata = channel, .channel = i,
                .current_direction = channel.direction, .previous_direction = channel.direction,
                .current_packet = now, .previous_packet = old, .scale = scale, .relative_tolerance = tolerance};
            maximum.include({normalized(static_cast<long double>(now) - old, scale, tolerance), 0, witness});
        }
    }
    return maximum;
}

SIMPLEFLUID_SOLVERS_LOCAL inline double number_packet(std::span<const Flow::CellSources> cells, size_t c, size_t a)
{
    return cells.empty() || cells[c].population_number_changes.empty() ? 0 : cells[c].population_number_changes[a];
}
SIMPLEFLUID_SOLVERS_LOCAL inline QbmmShapeState shape_packet(
    std::span<const Flow::CellSources> cells, size_t c, size_t a)
{
    return cells.empty() || cells[c].qbmm_shape_changes.empty()
               ? QbmmShapeState{}
               : cells[c].qbmm_shape_changes[a].value_or(QbmmShapeState{});
}
SIMPLEFLUID_SOLVERS_LOCAL inline double shape_source_error(std::span<const Flow::CellSources> current,
    std::span<const Flow::CellSources> previous, const Flow::QbmmSourceCells& receipts,
    const Flow::QbmmSourceCells& previous_receipts, const Flow::TransportSnapshot& transported, double tolerance)
{
    double maximum = 0;
    for (size_t c = 0; c < transported.cells.size(); ++c)
        for (size_t a = 0; a < transported.qbmm_families.size(); ++a)
        {
            if (!transported.qbmm_families[a])
                continue;
            const auto now = shape_packet(current, c, a), old = shape_packet(previous, c, a);
            maximum = std::max({maximum,
                normalized(
                    static_cast<long double>(now.H2) - old.H2, std::max(std::abs(now.H2), std::abs(old.H2)), tolerance),
                normalized(static_cast<long double>(now.H3) - old.H3, std::max(std::abs(now.H3), std::abs(old.H3)),
                    tolerance)});
            // Compare the raw integrated event algebra too. A small event can be below
            // the ULP of a large stored shape moment in both consecutive candidates.
            for (size_t k = 0; k < 2; ++k)
            {
                const auto x = receipts.at(c).at(a).value().event_shape_change[k];
                const auto y = previous_receipts.at(c).at(a).value().event_shape_change[k];
                maximum = std::max(maximum, normalized(x - y, std::max(std::abs(x), std::abs(y)), tolerance));
            }
        }
    return maximum;
}
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_number_packets(
    std::span<const Flow::CellSources> sources, size_t cells, std::span<const Flow::PhaseRoleDescriptor> roles)
{
    require(sources.empty() || sources.size() == cells, "Population source owned-cell layout differs.");
    for (const auto& source : sources)
    {
        require(source.population_number_changes.empty() || source.population_number_changes.size() == roles.size(),
            "Population source physical-phase layout differs.");
        for (size_t a = 0; a < source.population_number_changes.size(); ++a)
        {
            const double change = source.population_number_changes[a];
            require(std::isfinite(change) && (roles[a].role == Flow::PhaseRole::DispersedGas || change == 0),
                "Population number sources must be finite and bulk sources exactly zero.");
        }
    }
}
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_phase_numbers(
    const Flow::CellState& cell, std::span<const Flow::PhaseRoleDescriptor> roles)
{
    require(cell.phases.size() == roles.size() && cell.phase_numbers.size() == roles.size(),
        "Population state physical-phase layout differs.");
    for (size_t a = 0; a < roles.size(); ++a)
    {
        const auto& number = cell.phase_numbers[a];
        if (roles[a].role != Flow::PhaseRole::DispersedGas)
        {
            require(!number, "Bulk phases do not own a population number.");
            continue;
        }
        require(number && std::isfinite(*number) && *number >= 0,
            "Dispersed phase requires an explicit finite nonnegative number.");
        const auto& phase = cell.phases[a];
        require(phase.thermodynamics && phase.moles.size() == phase.thermodynamics->catalog().species().size() &&
                    std::all_of(
                        phase.moles.begin(), phase.moles.end(), [](double n) { return std::isfinite(n) && n >= 0; }),
            "Dispersed number inventory has invalid species layout or amounts.");
        const bool occupied = std::any_of(phase.moles.begin(), phase.moles.end(), [](double n) { return n > 0; });
        require(occupied == (*number > 0), "Empty dispersed gas and zero population number must coincide exactly.");
        if (occupied)
        {
            const double phase_mass = static_cast<double>(mass(phase.moles, phase.thermodynamics->catalog()));
            require(std::isfinite(phase_mass) && phase_mass > 0,
                "Occupied dispersed mass is not representable; no numerical phase floor is supplied.");
        }
    }
}
struct SIMPLEFLUID_SOLVERS_LOCAL CaloricReference
{
    long double energy = 0, capacity = 0;
    bool available = false;
};
SIMPLEFLUID_SOLVERS_LOCAL inline CaloricReference caloric_reference(
    const Flow::TransportPhaseInventory& transported, const PhaseThermalState& guess, const PhaseThermalState& accepted)
{
    const auto& catalog = guess.thermodynamics->catalog();
    const auto guess_mass = mass(guess.moles, catalog), accepted_mass = mass(accepted.moles, catalog);
    const auto stage_mass = mass(transported.moles, catalog);
    const auto* phase = guess_mass > 0 && guess.temperature > 0         ? &guess
                        : accepted_mass > 0 && accepted.temperature > 0 ? &accepted
                                                                        : nullptr;
    if (!phase)
        return {};
    const long double phase_mass = phase == &guess ? guess_mass : accepted_mass;
    long double total = 0;
    for (const double n : phase->moles)
        total += n;
    std::vector<double> x;
    x.reserve(phase->moles.size());
    for (const double n : phase->moles)
        x.push_back(static_cast<double>(static_cast<long double>(n) / total));
    const auto point = phase->thermodynamics->evaluate(phase->temperature, phase->absolute_pressure, x);
    CaloricReference result;
    result.energy = stage_mass > 0 ? static_cast<long double>(transported.internal_energy) / stage_mass
                                   : static_cast<long double>(phase->internal_energy) / phase_mass;
    result.capacity = phase_mass * phase->temperature * point.internal_energy_temperature;
    result.available = true;
    require(std::isfinite(result.energy) && std::isfinite(result.capacity) && result.capacity > 0,
        "Source-stage caloric reference/capacity is invalid.");
    return result;
}
SIMPLEFLUID_SOLVERS_LOCAL inline double source_error(std::span<const Flow::CellSources> current,
    std::span<const Flow::CellSources> previous, const Flow::TransportSnapshot& transported, const Flow::State& guess,
    const Flow::State& accepted, const Flow::Options& options)
{
    double maximum = 0;
    for (size_t c = 0; c < transported.cells.size(); ++c)
    {
        for (size_t a = 0; a < transported.cells[c].phases.size(); ++a)
        {
            const auto& catalog = guess.cells[c].phases[a].thermodynamics->catalog();
            const long double number_now = number_packet(current, c, a), number_old = number_packet(previous, c, a);
            maximum = std::max(
                maximum, normalized(number_now - number_old, std::max(std::abs(number_now), std::abs(number_old)),
                             options.source_relative_tolerance));
            long double mass_current = 0, mass_previous = 0;
            bool active = false;
            for (size_t s = 0; s < catalog.species().size(); ++s)
            {
                const long double now = species_packet(current, c, a, s), old = species_packet(previous, c, a, s);
                active = active || now != 0 || old != 0;
                maximum = std::max(maximum,
                    normalized(now - old, std::max(std::abs(now), std::abs(old)), options.source_relative_tolerance));
                mass_current += now * catalog.species()[s].molarMass;
                mass_previous += old * catalog.species()[s].molarMass;
            }
            const long double energy_current = energy_packet(current, c, a),
                              energy_previous = energy_packet(previous, c, a);
            active = active || energy_current != 0 || energy_previous != 0;
            if (!active)
                continue;
            const auto reference = caloric_reference(
                transported.cells[c].phases[a], guess.cells[c].phases[a], accepted.cells[c].phases[a]);
            if (!reference.available)
            {
                maximum = std::numeric_limits<double>::infinity();
                continue;
            }
            // A common additive mass-energy reference cancels before comparison.
            // No absolute one-joule floor, and no formation-offset-sized tolerance.
            const long double now = energy_current - mass_current * reference.energy;
            const long double old = energy_previous - mass_previous * reference.energy;
            const long double difference =
                (energy_current - energy_previous) - (mass_current - mass_previous) * reference.energy;
            maximum = std::max(
                maximum, normalized(difference, reference.capacity + std::abs(now) + std::abs(old),
                             std::min(options.source_relative_tolerance, options.conservation_relative_tolerance)));
        }
        for (size_t axis = 0; axis < 3; ++axis)
        {
            const long double now = current.empty() ? 0 : current[c].momentum.component(axis);
            const long double old = previous.empty() ? 0 : previous[c].momentum.component(axis);
            maximum = std::max(maximum,
                normalized(now - old, std::max(std::abs(now), std::abs(old)), options.source_relative_tolerance));
        }
    }
    return maximum;
}
SIMPLEFLUID_SOLVERS_LOCAL inline double transport_error(const Flow::TransportSnapshot& current,
    const Flow::TransportSnapshot& previous, const Flow::State& guess, const Flow::State& accepted,
    const Flow::Options& options)
{
    require(current.cells.size() == previous.cells.size() &&
                current.accepted_generation == previous.accepted_generation &&
                current.old_geometry_epoch == previous.old_geometry_epoch &&
                current.new_geometry_epoch == previous.new_geometry_epoch && current.time_step == previous.time_step &&
                current.accepted_time == previous.accepted_time &&
                current.schema_fingerprint == previous.schema_fingerprint &&
                current.qbmm_families == previous.qbmm_families,
        "Transport snapshots do not belong to one immutable preview frame.");
    double maximum = 0;
    for (size_t c = 0; c < current.cells.size(); ++c)
    {
        require(current.old_control_volumes[c] == previous.old_control_volumes[c] &&
                    current.new_control_volumes[c] == previous.new_control_volumes[c],
            "Source corrector changed geometry authority.");
        for (size_t a = 0; a < current.cells[c].phases.size(); ++a)
        {
            const auto& now = current.cells[c].phases[a];
            const auto& old = previous.cells[c].phases[a];
            const auto& catalog = guess.cells[c].phases[a].thermodynamics->catalog();
            for (size_t s = 0; s < now.moles.size(); ++s)
                maximum = std::max(maximum,
                    normalized(static_cast<long double>(now.moles[s]) - old.moles[s],
                        std::max(std::abs(now.moles[s]), std::abs(old.moles[s])), options.source_relative_tolerance));
            const auto reference = caloric_reference(now, guess.cells[c].phases[a], accepted.cells[c].phases[a]);
            const long double delta_mass = mass(now.moles, catalog) - mass(old.moles, catalog);
            const long double delta_energy = static_cast<long double>(now.internal_energy) - old.internal_energy;
            if (reference.available)
                maximum = std::max(
                    maximum, normalized(delta_energy - delta_mass * reference.energy, reference.capacity,
                                 std::min(options.source_relative_tolerance, options.conservation_relative_tolerance)));
            else if (delta_energy != 0 || delta_mass != 0)
                maximum = std::numeric_limits<double>::infinity();
            maximum = std::max(maximum,
                normalized(static_cast<long double>(now.advected_material_volume) - old.advected_material_volume,
                    std::max(std::abs(now.advected_material_volume), std::abs(old.advected_material_volume)),
                    options.volume_relative_tolerance));
            require(now.population_number.has_value() == old.population_number.has_value(),
                "Population ownership changed within an immutable transport preview.");
            if (now.population_number)
                maximum = std::max(
                    maximum, normalized(static_cast<long double>(*now.population_number) - *old.population_number,
                                 std::max(std::abs(*now.population_number), std::abs(*old.population_number)),
                                 options.source_relative_tolerance));
            require(now.qbmm_shape.has_value() == old.qbmm_shape.has_value(),
                "QBMM shape ownership changed within an immutable transport preview.");
            if (now.qbmm_shape)
                for (auto moment : {&QbmmShapeState::H2, &QbmmShapeState::H3})
                {
                    const long double latest = (*now.qbmm_shape).*moment;
                    const long double prior = (*old.qbmm_shape).*moment;
                    maximum = std::max(maximum,
                        normalized(latest - prior, std::max(latest, prior), options.source_relative_tolerance));
                }
        }
    }
    return maximum;
}
SIMPLEFLUID_SOLVERS_LOCAL inline double thermal_error(
    const Flow::State& current, const Flow::State& previous, const Flow::Options& options)
{
    double maximum = 0;
    for (size_t c = 0; c < current.cells.size(); ++c)
        for (size_t a = 0; a < current.cells[c].phases.size(); ++a)
        {
            const auto& now = current.cells[c].phases[a];
            const auto& old = previous.cells[c].phases[a];
            maximum =
                std::max(maximum, normalized(static_cast<long double>(now.temperature) - old.temperature,
                                      std::max(now.temperature, old.temperature), options.thermal_relative_tolerance));
            if (now.temperature > 0 || old.temperature > 0)
            {
                maximum = std::max(maximum,
                    normalized(static_cast<long double>(now.absolute_pressure) - old.absolute_pressure,
                        std::max(now.absolute_pressure, old.absolute_pressure), options.thermal_relative_tolerance));
                // This draft has no capillary pressure-offset model. Later radius
                // integration must replace this common-phase-pressure identity.
                const long double actual =
                    static_cast<long double>(current.thermodynamic_pressure) + current.cells[c].mechanical_pressure;
                const long double prior =
                    static_cast<long double>(previous.thermodynamic_pressure) + previous.cells[c].mechanical_pressure;
                maximum = std::max(maximum, normalized(actual - prior, std::max(std::abs(actual), std::abs(prior)),
                                                options.thermal_relative_tolerance));
                maximum = std::max(
                    maximum, normalized(static_cast<long double>(now.absolute_pressure) - actual,
                                 std::max(std::abs(actual), std::abs(static_cast<long double>(now.absolute_pressure))),
                                 options.thermal_relative_tolerance));
            }
        }
    return maximum;
}
} // namespace SimpleFluid::Reacting::source_stage_detail
