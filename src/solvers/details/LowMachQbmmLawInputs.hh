/** @file LowMachQbmmLawInputs.hh
 * @brief MPI-free native law snapshots and endpoint gates; callers own collective fences.
 */
#pragma once

#include "equations/QbmmLawInputs.hh"
#include "equations/details/LowMachScaledFaceSum.hh"
#include "solvers/LowMachMixtureSolver.hh"

#include <numeric>

namespace SimpleFluid::Reacting::qbmm_native_law_detail
{
using Flow = LowMachMixtureSolver;

/** Applies also to count-only phases and native configurations with no QBMM schema. */
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_drift(const Flow::CellState& cell)
{
    if (cell.drift_velocity.size() != cell.phases.size())
        throw std::invalid_argument("Native supplied phase drift layout differs from physical phases.");
    for (const auto& velocity : cell.drift_velocity)
        for (std::size_t axis = 0; axis < 3; ++axis)
            if (!std::isfinite(velocity.component(axis)))
                throw std::invalid_argument("Native supplied phase drift must be finite, including count-only phases.");
}

SIMPLEFLUID_SOLVERS_LOCAL inline bool is_transport_gate(const QbmmLawMetadata& law)
{
    return law.kind != QbmmLawKind::NodeSlip && law.gate == QbmmApplicabilityGate::RejectOutside &&
           qbmm_is_transport_law(law);
}

/** The native state supplies a carrier-liquid basis only when its liquid slot is unique. */
SIMPLEFLUID_SOLVERS_LOCAL inline std::optional<std::size_t> carrier_liquid_index(
    std::span<const PhaseThermalState> phases)
{
    std::optional<std::size_t> liquid;
    for (std::size_t p = 0; p < phases.size(); ++p)
        if (phases[p].thermodynamics->kind() == EOSKind::Liquid)
        {
            if (liquid)
                return std::nullopt;
            liquid = p;
        }
    return liquid;
}

SIMPLEFLUID_SOLVERS_LOCAL inline ReactionLocation kinetic_location(
    const Flow::Mesh& mesh, Flow::Pack::local_ordinal_type cell)
{
    const auto id = mesh.cell_geometry_global_id(cell);
    if (!std::in_range<std::uint64_t>(id) || !std::in_range<std::size_t>(cell))
        throw std::invalid_argument("Kinetic QBMM cell identity is not representable.");
    const auto center = mesh.cell_centroid(cell);
    return {static_cast<std::uint64_t>(id), mesh.geometry_epoch(), {center.x, center.y, center.z},
        static_cast<std::size_t>(cell)};
}

SIMPLEFLUID_SOLVERS_LOCAL inline double narrow_kinetic(const qbmm_face_detail::ScaledFaceSum& value, double result)
{
    if (!std::isfinite(result) || (result == 0 && value.sign() != 0))
        throw std::invalid_argument("Kinetic QBMM constitutive quantity is not representable.");
    return result;
}

SIMPLEFLUID_SOLVERS_LOCAL inline QbmmCellView kinetic_cell_view(
    const QbmmPopulationSchema& family, const Flow::CellState& cell, std::size_t phase)
{
    return {family.population(), family.catalog().fingerprint(), cell.phases.at(phase).moles,
        cell.phase_numbers.at(phase).value(), cell.qbmm_shape.at(phase).value(), family.fingerprint()};
}

/** One physical mean for source inputs, property gates and callback-free endpoint reports. */
SIMPLEFLUID_SOLVERS_LOCAL inline Flow::Vector mean_drift_velocity(const Flow::CellState& cell, size_t p)
{
    if (cell.qbmm_kinetic_closure.empty() || !cell.qbmm_kinetic_closure.at(p))
        return cell.drift_velocity.at(p);
    const auto& closure = *cell.qbmm_kinetic_closure.at(p);
    if (closure.active_atoms == 0)
        return {};
    const auto& phase = cell.phases.at(p);
    qbmm_face_detail::ScaledFaceSum mass;
    for (size_t s = 0; s < phase.moles.size(); ++s)
        mass.add_product(phase.moles[s], phase.thermodynamics->catalog().species()[s].molarMass);
    Flow::Vector mean{};
    for (size_t axis = 0; axis < 3; ++axis)
    {
        qbmm_face_detail::ScaledFaceSum momentum;
        momentum.add_product(closure.drift_momentum.component(axis), 1.);
        mean.component(axis) = narrow_kinetic(momentum, momentum.rounded(mass));
    }
    return mean;
}

SIMPLEFLUID_SOLVERS_LOCAL inline double kinetic_speed(const Flow::Vector& velocity)
{
    const double speed = std::hypot(velocity.x, velocity.y, velocity.z);
    if (!std::isfinite(velocity.x) || !std::isfinite(velocity.y) || !std::isfinite(velocity.z) || !std::isfinite(speed))
        throw std::invalid_argument("Kinetic QBMM slip needs finite SI components and speed.");
    return speed;
}

SIMPLEFLUID_SOLVERS_LOCAL inline void accumulate_kinetic_drift(const QbmmPopulationSchema& family,
    const QbmmQuadrature& quadrature, const Flow::QbmmKineticClosure& closure,
    std::array<qbmm_face_detail::ScaledFaceSum, 3>& sums)
{
    if (!quadrature.valid() || closure.active_atoms != quadrature.active_atoms)
        throw std::invalid_argument("Kinetic QBMM closure disagrees with the represented quadrature rank.");
    for (std::size_t i = 0; i < closure.node_velocities.size(); ++i)
    {
        const auto& velocity = closure.node_velocities[i];
        const double speed = kinetic_speed(velocity);
        if (i >= closure.active_atoms)
        {
            if (speed != 0)
                throw std::invalid_argument("An inactive kinetic QBMM atom cannot retain a slip velocity.");
            continue;
        }
        const auto& atom = quadrature.atoms[i];
        qbmm_face_detail::ScaledFaceSum mass;
        mass.add_product(atom.count, atom.mass_over_scale, family.mass_scale_kg());
        for (std::size_t axis = 0; axis < sums.size(); ++axis)
            sums[axis].add_weighted(mass, velocity.component(axis));
    }
}

SIMPLEFLUID_SOLVERS_LOCAL inline Flow::Vector kinetic_drift_momentum(
    const QbmmPopulationSchema& family, const QbmmQuadrature& quadrature, const Flow::QbmmKineticClosure& closure)
{
    std::array<qbmm_face_detail::ScaledFaceSum, 3> sums;
    accumulate_kinetic_drift(family, quadrature, closure, sums);
    return {narrow_kinetic(sums[0], sums[0].rounded()), narrow_kinetic(sums[1], sums[1].rounded()),
        narrow_kinetic(sums[2], sums[2].rounded())};
}

SIMPLEFLUID_SOLVERS_LOCAL inline double kinetic_gas_fraction(
    const Flow::CellState& cell, std::span<const double> recovered_volumes, double bulk_volume_m3)
{
    qbmm_face_detail::ScaledFaceSum gas_volume;
    for (std::size_t a = 0; a < cell.phases.size(); ++a)
    {
        const double volume = recovered_volumes[a];
        if (!std::isfinite(volume) || volume < 0)
            throw std::invalid_argument("Kinetic QBMM closure needs finite nonnegative EOS volumes.");
        if (cell.phases[a].thermodynamics->kind() == EOSKind::Gas)
            gas_volume.add_product(volume, 1.);
    }
    return narrow_kinetic(gas_volume, gas_volume.rounded(bulk_volume_m3));
}

SIMPLEFLUID_SOLVERS_LOCAL inline double kinetic_specific_volume(const PhaseThermalState& donor, double occupied_volume)
{
    qbmm_face_detail::ScaledFaceSum mass, volume;
    for (std::size_t s = 0; s < donor.moles.size(); ++s)
        mass.add_product(donor.moles[s], donor.thermodynamics->catalog().species()[s].molarMass);
    if (mass.sign() <= 0 || occupied_volume <= 0)
        throw std::invalid_argument("An active kinetic QBMM atom needs positive donor mass and EOS volume.");
    volume.add_product(occupied_volume, 1.);
    return narrow_kinetic(volume, volume.rounded(mass));
}

using NodeSlipObservations = std::array<QbmmApplicabilityEvaluation, 2>;

SIMPLEFLUID_SOLVERS_LOCAL inline QbmmLawInputs make_kinetic_law_inputs(const QbmmPopulationSchema& family,
    const Flow::CellState& cell, std::size_t phase, double bulk_volume_m3, std::span<const double> recovered_volumes,
    const QbmmCellView& view, const QbmmQuadrature& quadrature, std::optional<ReactionLocation> location = {})
{
    std::optional<std::vector<double>> liquid_concentrations;
    std::optional<double> liquid_temperature;
    if (const auto liquid = carrier_liquid_index(cell.phases); liquid && recovered_volumes[*liquid] > 0)
    {
        liquid_concentrations = qbmm_phase_concentrations(cell.phases[*liquid].moles, recovered_volumes[*liquid]);
        liquid_temperature = cell.phases[*liquid].temperature;
    }
    const auto& donor = cell.phases[phase];
    const double occupied_volume = recovered_volumes[phase];
    const auto geometry = qbmm_geometry(family, view, bulk_volume_m3, kinetic_specific_volume(donor, occupied_volume));
    return make_qbmm_law_inputs(
        family, {quadrature, geometry, donor.moles, bulk_volume_m3, occupied_volume, donor.absolute_pressure,
                    donor.temperature, std::nullopt, kinetic_gas_fraction(cell, recovered_volumes, bulk_volume_m3),
                    std::move(liquid_concentrations), liquid_temperature, {}, location});
}

SIMPLEFLUID_SOLVERS_LOCAL inline void validate_node_slip_gate(
    const QbmmPopulationSchema& family, const QbmmApplicabilityEvaluation& evaluation)
{
    if (!evaluation.gate_passed)
        throw std::invalid_argument("Kinetic QBMM family '" + family.population() + "' law '" +
                                    evaluation.metadata.identity + "' is outside its declared node applicability.");
}

/** Evaluate on owned thermodynamic donor traces. The callback sees instantaneous physical inputs only. */
SIMPLEFLUID_SOLVERS_LOCAL inline Flow::QbmmKineticClosure evaluate_node_slip(const Flow::Options& options,
    const Flow::CellState& cell, std::size_t phase, double bulk_volume_m3, std::span<const double> recovered_volumes,
    std::optional<ReactionLocation> location = {}, NodeSlipObservations* observations = nullptr)
{
    const auto& family = *options.qbmm_families.at(phase);
    const auto& law = *options.qbmm_node_slip_laws.at(phase);
    const auto view = kinetic_cell_view(family, cell, phase);
    const auto quadrature = invert_qbmm_moments(qbmm_cell_moments(family, view));
    if (!quadrature.valid() || recovered_volumes.size() != cell.phases.size() || !std::isfinite(bulk_volume_m3) ||
        bulk_volume_m3 <= 0)
        throw std::invalid_argument("Kinetic QBMM closure needs represented donor stocks and EOS volumes.");
    Flow::QbmmKineticClosure closure{quadrature.active_atoms};
    if (closure.active_atoms == 0)
    {
        if (recovered_volumes[phase] != 0)
            throw std::invalid_argument("An absent kinetic QBMM phase cannot have occupied EOS volume.");
        return closure;
    }

    const double occupied_volume = recovered_volumes[phase];
    const auto inputs =
        make_kinetic_law_inputs(family, cell, phase, bulk_volume_m3, recovered_volumes, view, quadrature, location);
    for (std::size_t i = 0; i < closure.active_atoms; ++i)
    {
        validate_node_slip_gate(family, evaluate_qbmm_node_slip_applicability(law.metadata, inputs, i, std::nullopt));
        const auto velocity = law.evaluate(
            make_qbmm_node_slip_input(inputs, i, occupied_volume, law.required_body_acceleration_m_per_s2));
        auto& retained = closure.node_velocities[i];
        retained = {velocity[0], velocity[1], velocity[2]};
        auto applicability = evaluate_qbmm_node_slip_applicability(law.metadata, inputs, i, kinetic_speed(retained));
        validate_node_slip_gate(family, applicability);
        if (observations)
            (*observations)[i] = std::move(applicability);
    }
    closure.drift_momentum = kinetic_drift_momentum(family, quadrature, closure);
    return closure;
}

/** Finish the applicability gate after the communicator has selected the actual substep. */
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_node_slip_interval(const Flow::Options& options, std::size_t phase,
    const Flow::QbmmKineticClosure& closure, double substep_s, NodeSlipObservations* observations = nullptr)
{
    if (closure.active_atoms == 0)
        return;
    if (!std::isfinite(substep_s) || substep_s <= 0)
        throw std::invalid_argument("Kinetic QBMM requires a positive representable substep.");
    const auto& law = options.qbmm_node_slip_laws.at(phase)->metadata;
    for (std::size_t r = 0; r < law.ranges.size(); ++r)
    {
        const auto& range = law.ranges[r];
        if (range.variable != QbmmRangeVariable::TimeStepS)
            continue;
        const bool applicable = substep_s >= range.minimum && substep_s <= range.maximum;
        if (law.gate == QbmmApplicabilityGate::RejectOutside && !applicable)
            throw std::invalid_argument("Kinetic QBMM law '" + law.identity + "' rejects the actual substep size.");
        if (observations)
            for (std::size_t i = 0; i < closure.active_atoms; ++i)
            {
                auto& evaluation = observations->at(i);
                evaluation.applicable = evaluation.applicable && applicable;
                evaluation.ranges.at(r) = {range,
                    applicable ? QbmmRangeStatus::WithinRange : QbmmRangeStatus::OutsideRange, substep_s, substep_s};
            }
    }
}

SIMPLEFLUID_SOLVERS_LOCAL inline void evaluate_kinetic_closure(const Flow::Options& options, Flow::CellState& cell,
    double bulk_volume_m3, std::span<const double> recovered_volumes, std::optional<ReactionLocation> location = {})
{
    if (!uses_qbmm_kinetic_transport(options.qbmm_model))
        return;
    std::vector<std::optional<Flow::QbmmKineticClosure>> closures(options.qbmm_families.size());
    for (std::size_t a = 0; a < closures.size(); ++a)
        if (options.qbmm_families[a])
            closures[a] = evaluate_node_slip(options, cell, a, bulk_volume_m3, recovered_volumes, location);
    cell.qbmm_kinetic_eos_volumes.assign(recovered_volumes.begin(), recovered_volumes.end());
    cell.qbmm_kinetic_closure = std::move(closures);
}

/** Check a retained endpoint using canonical moments and recorded velocities only. */
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_kinetic_closure(
    const Flow::Options& options, const Flow::CellState& cell)
{
    const auto& closures = cell.qbmm_kinetic_closure;
    const auto& families = options.qbmm_families;
    if (closures.size() != (uses_qbmm_kinetic_transport(options.qbmm_model) ? families.size() : 0) ||
        cell.qbmm_kinetic_eos_volumes.size() != closures.size())
        throw std::invalid_argument("Kinetic QBMM endpoint closure layout differs from the configured integrator.");
    for (std::size_t a = 0; a < closures.size(); ++a)
    {
        const double volume = cell.qbmm_kinetic_eos_volumes[a];
        if (!std::isfinite(volume) || volume < 0 ||
            (volume > 0) != std::ranges::any_of(cell.phases[a].moles, [](double n) { return n > 0; }))
            throw std::invalid_argument("Retained kinetic EOS volume disagrees with phase occupancy.");
        if (bool(closures[a]) != bool(families[a]))
            throw std::invalid_argument("Kinetic QBMM endpoint closure must match the bound family slots.");
        if (!closures[a])
            continue;
        const auto& family = *families[a];
        const auto quadrature = invert_qbmm_moments(qbmm_cell_moments(family, kinetic_cell_view(family, cell, a)));
        if (kinetic_drift_momentum(family, quadrature, *closures[a]) != closures[a]->drift_momentum)
            throw std::invalid_argument("Kinetic QBMM endpoint drift momentum disagrees with its retained node slip.");
    }
}

/** Recheck instantaneous hard gates using the archived physical traces and node velocities. */
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_archived_node_slip_gates(
    const Flow::Options& options, const Flow::CellState& cell, double bulk_volume_m3)
{
    for (std::size_t a = 0; a < cell.qbmm_kinetic_closure.size(); ++a)
    {
        const auto& closure = cell.qbmm_kinetic_closure[a];
        if (!closure || closure->active_atoms == 0)
            continue;
        const auto& law = options.qbmm_node_slip_laws[a]->metadata;
        if (law.gate != QbmmApplicabilityGate::RejectOutside)
            continue;
        const auto& family = *options.qbmm_families[a];
        const auto view = kinetic_cell_view(family, cell, a);
        const auto quadrature = invert_qbmm_moments(qbmm_cell_moments(family, view));
        const auto inputs =
            make_kinetic_law_inputs(family, cell, a, bulk_volume_m3, cell.qbmm_kinetic_eos_volumes, view, quadrature);
        for (std::size_t i = 0; i < closure->active_atoms; ++i)
            validate_node_slip_gate(family,
                evaluate_qbmm_node_slip_applicability(law, inputs, i, kinetic_speed(closure->node_velocities[i])));
    }
}

/** Correctors supply recovered EOS volumes from the exact viscosity inputs. Endpoint
 * callers without a recovered snapshot sample the physical state for its EOS volumes.
 */
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_transport_gates(const Flow::Options& options,
    const Flow::CellState& cell, double bulk_volume_m3, double time_step_s,
    std::span<const double> recovered_volumes = {})
{
    const auto& families = options.qbmm_families;
    const auto active_gate = [&](std::size_t p, const QbmmLawMetadata& law)
    { return is_transport_gate(law) && (!law.requires_population || cell.phase_numbers.at(p).value() > 0); };
    bool has_gate = false, needs_liquid = false;
    for (std::size_t p = 0; p < families.size(); ++p)
        if (families[p])
            for (const auto& law : families[p]->closure_versions().laws)
                if (active_gate(p, law))
                {
                    has_gate = true;
                    needs_liquid = needs_liquid || qbmm_uses_carrier_liquid(law);
                }
    if (!has_gate)
        return;

    const bool kinetic = uses_qbmm_kinetic_transport(options.qbmm_model);
    if (kinetic)
        recovered_volumes = cell.qbmm_kinetic_eos_volumes;
    const auto liquid = needs_liquid ? carrier_liquid_index(cell.phases) : std::nullopt;
    if (!recovered_volumes.empty() && recovered_volumes.size() != cell.phases.size())
        throw std::invalid_argument("Native transport law recovered volume layout differs from physical phases.");
    std::vector<double> volumes(cell.phases.size()), specific_volumes(cell.phases.size());
    double gas_volume = 0;
    std::optional<std::vector<double>> liquid_concentrations;
    std::optional<double> liquid_temperature;
    for (std::size_t p = 0; p < cell.phases.size(); ++p)
    {
        const auto& phase = cell.phases[p];
        if (!phase.thermodynamics || phase.moles.size() != phase.thermodynamics->catalog().species().size())
            throw std::invalid_argument("Native law input material layout differs from its thermodynamic catalog.");
        const bool gas = phase.thermodynamics->kind() == EOSKind::Gas;
        if (!gas && liquid != p)
            continue;
        const double total_moles = std::accumulate(phase.moles.begin(), phase.moles.end(), 0.);
        if (total_moles == 0)
            continue;
        // Use the native report's mass and composition arithmetic at inclusive range boundaries.
        double mass = 0;
        for (std::size_t s = 0; s < phase.moles.size(); ++s)
            mass += phase.moles[s] * phase.thermodynamics->catalog().species()[s].molarMass;
        if (recovered_volumes.empty())
        {
            auto composition = phase.moles;
            for (auto& amount : composition)
                amount /= total_moles;
            const auto point = phase.thermodynamics->evaluate(phase.temperature, phase.absolute_pressure, composition);
            volumes[p] = mass / point.density;
            specific_volumes[p] = 1 / point.density;
        }
        else
        {
            volumes[p] = recovered_volumes[p];
            specific_volumes[p] = kinetic ? kinetic_specific_volume(phase, volumes[p]) : volumes[p] / mass;
        }
        if (!std::isfinite(volumes[p]) || volumes[p] <= 0)
            throw std::invalid_argument("Native law input physical phase volume is not representable.");
        if (gas)
            gas_volume += volumes[p];
        else
        {
            liquid_concentrations = qbmm_phase_concentrations(phase.moles, volumes[p]);
            liquid_temperature = phase.temperature;
        }
    }
    for (std::size_t p = 0; p < families.size(); ++p)
    {
        const auto active = [&](const auto& law) { return active_gate(p, law); };
        if (!families[p] || std::ranges::none_of(families[p]->closure_versions().laws, active))
            continue;
        const auto& family = *families[p];
        const auto& phase = cell.phases.at(p);
        const QbmmCellView view{family.population(), family.catalog().fingerprint(), phase.moles,
            cell.phase_numbers.at(p).value(), cell.qbmm_shape.at(p).value(), family.fingerprint()};
        const auto quadrature = invert_qbmm_moments(qbmm_cell_moments(family, view));
        const std::optional<double> specific_volume =
            volumes[p] > 0 ? std::optional<double>(specific_volumes[p]) : std::nullopt;
        const auto drift = mean_drift_velocity(cell, p);
        const auto inputs = make_qbmm_law_inputs(family,
            {quadrature, qbmm_geometry(family, view, bulk_volume_m3, specific_volume), phase.moles, bulk_volume_m3,
                volumes[p], phase.absolute_pressure, phase.temperature, time_step_s,
                (kinetic ? kinetic_gas_fraction(cell, recovered_volumes, bulk_volume_m3) : gas_volume / bulk_volume_m3),
                liquid_concentrations, liquid_temperature,
                {drift.component(0), drift.component(1), drift.component(2)}});
        for (const auto& law : family.closure_versions().laws)
            if (active(law) && !evaluate_qbmm_applicability(law, inputs).gate_passed)
                throw std::invalid_argument("Native QBMM family '" + family.population() + "' law '" + law.identity +
                                            "' is outside its declared applicability or lacks a required input.");
    }
}
} // namespace SimpleFluid::Reacting::qbmm_native_law_detail
