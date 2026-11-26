/** @file QbmmLawInputs.hh
 * @brief Pure construction of law inputs from frozen local physical quantities.
 */
#pragma once

#include "equations/QbmmBulkGasPool.hh"

namespace SimpleFluid::Reacting
{
[[nodiscard]] inline bool qbmm_is_transport_law(const QbmmLawMetadata& law)
{
    return law.kind == QbmmLawKind::PhaseSlip || law.kind == QbmmLawKind::TransportProperty ||
           law.kind == QbmmLawKind::NodeSlip;
}

/** Prepared by the state owner. Volumes are physical EOS volumes, never transport work volumes.
 * Quadrature/geometry may be reused from an existing local report. No EOS, rate law,
 * field, clock or collective is accessed by the builder.
 */
struct QbmmFrozenLawState
{
    QbmmQuadrature quadrature;
    std::optional<QbmmGeometryView> geometry;
    std::span<const double> family_moles;
    double bulk_volume_m3, phase_volume_m3, absolute_pressure_pa;
    std::optional<double> temperature_K, time_step_s, gas_volume_fraction;
    std::optional<std::vector<double>> carrier_liquid_concentration_mol_per_m3;
    std::optional<double> carrier_liquid_temperature_K;
    std::array<double, 3> drift_m_per_s;
    std::optional<ReactionLocation> location{};
    std::optional<double> carrier_liquid_volume_m3{};
    std::optional<std::vector<double>> carrier_liquid_dissolved_moles{};
    std::optional<double> bulk_gas_pool_volume_m3{};
    std::optional<std::vector<double>> bulk_gas_pool_moles{};
    std::optional<std::vector<double>> bulk_gas_pool_concentration_mol_per_m3{};
    std::optional<double> bulk_gas_pool_temperature_K{};
    std::optional<std::uint64_t> bulk_gas_pool_fingerprint{};
};

[[nodiscard]] inline QbmmLawInputs make_qbmm_law_inputs(const QbmmPopulationSchema& family, QbmmFrozenLawState frozen)
{
    const bool absent = frozen.quadrature.status == InversionStatus::Absent;
    if (!frozen.quadrature.valid() || !std::isfinite(frozen.bulk_volume_m3) || frozen.bulk_volume_m3 <= 0 ||
        !std::isfinite(frozen.phase_volume_m3) || frozen.phase_volume_m3 < 0 ||
        !std::ranges::all_of(frozen.drift_m_per_s, [](double value) { return std::isfinite(value); }))
        throw std::invalid_argument("QBMM law input snapshot needs valid geometry, quadrature and finite drift.");
    if (absent != (frozen.phase_volume_m3 == 0))
        throw std::invalid_argument("QBMM law input population absence disagrees with the physical phase volume.");
    if (frozen.carrier_liquid_volume_m3 == 0.0)
    {
        if (frozen.carrier_liquid_concentration_mol_per_m3 || frozen.carrier_liquid_temperature_K ||
            frozen.carrier_liquid_dissolved_moles)
            throw std::invalid_argument("QBMM law input cannot carry a liquid trace at exact dryout.");
    }
    else if (frozen.carrier_liquid_volume_m3)
    {
        const auto& concentration = frozen.carrier_liquid_concentration_mol_per_m3;
        if (!std::isfinite(*frozen.carrier_liquid_volume_m3) || *frozen.carrier_liquid_volume_m3 <= 0 ||
            !concentration || concentration->size() != family.catalog().species().size() ||
            !std::ranges::all_of(*concentration, [](double value) { return std::isfinite(value) && value >= 0; }))
            throw std::invalid_argument(
                "QBMM law input needs positive finite carrier liquid EOS volume and concentrations.");
    }
    if (const auto& dissolved = frozen.carrier_liquid_dissolved_moles;
        dissolved &&
        (!frozen.carrier_liquid_volume_m3 || dissolved->size() != family.catalog().species().size() ||
            !std::ranges::all_of(*dissolved, [](double value) { return std::isfinite(value) && value >= 0; })))
        throw std::invalid_argument("QBMM law input needs finite catalogued Dissolved moles on occupied liquid.");
    const auto species_count = family.catalog().species().size();
    const auto& pool_moles = frozen.bulk_gas_pool_moles;
    const auto& pool_volume = frozen.bulk_gas_pool_volume_m3;
    const auto& pool_concentration = frozen.bulk_gas_pool_concentration_mol_per_m3;
    const auto& pool_temperature = frozen.bulk_gas_pool_temperature_K;
    if (!pool_moles && frozen.bulk_gas_pool_fingerprint)
        throw std::invalid_argument("QBMM pool authority requires a frozen species row.");
    if (pool_volume)
    {
        const double pool_fraction = *pool_volume / frozen.bulk_volume_m3;
        if (!std::isfinite(pool_fraction) || pool_fraction <= 0 || !pool_moles || pool_moles->size() != species_count ||
            qbmm_phase_concentrations(*pool_moles, *pool_volume) != pool_concentration || !pool_temperature ||
            !std::isfinite(*pool_temperature) || *pool_temperature <= 0)
            throw std::invalid_argument("QBMM law input has an inconsistent bulk gas pool trace.");
        for (std::size_t i = 0; i < species_count; ++i)
            if ((*pool_moles)[i] != 0 &&
                !family.catalog().allows(family.catalog().species()[i].id, SpeciesPhase::Resolved))
                throw std::invalid_argument("QBMM law input pool moles must be permitted in Resolved.");
    }
    else if (pool_concentration || pool_temperature ||
             (pool_moles && (pool_moles->size() != species_count ||
                                !std::ranges::all_of(*pool_moles, [](double value) { return value == 0; }))))
        throw std::invalid_argument("An absent QBMM pool can carry only exact zero catalogued moles.");

    const double speed = std::hypot(frozen.drift_m_per_s[0], frozen.drift_m_per_s[1], frozen.drift_m_per_s[2]);
    const double fraction = frozen.phase_volume_m3 / frozen.bulk_volume_m3;
    if (!std::isfinite(speed) || !std::isfinite(fraction) || (frozen.phase_volume_m3 > 0 && fraction == 0))
        throw std::invalid_argument("QBMM law input phase fraction or slip speed is not representable.");
    std::optional<std::vector<double>> original_family;
    if (pool_moles)
    {
        (void) family.scaled_mass(frozen.family_moles);
        original_family.emplace(frozen.family_moles.begin(), frozen.family_moles.end());
    }
    return {frozen.quadrature, std::move(frozen.geometry), family.mass_scale_kg(), frozen.bulk_volume_m3,
        frozen.absolute_pressure_pa, absent ? std::nullopt : frozen.temperature_K, frozen.time_step_s, fraction,
        frozen.gas_volume_fraction, qbmm_phase_concentrations(frozen.family_moles, frozen.phase_volume_m3),
        std::move(frozen.carrier_liquid_concentration_mol_per_m3), frozen.carrier_liquid_temperature_K, speed,
        family.fingerprint(), family.catalog().fingerprint(), family.population(), frozen.location, {}, {},
        frozen.carrier_liquid_volume_m3, std::move(frozen.carrier_liquid_dissolved_moles),
        frozen.bulk_gas_pool_volume_m3, std::move(frozen.bulk_gas_pool_moles),
        std::move(frozen.bulk_gas_pool_concentration_mol_per_m3), frozen.bulk_gas_pool_temperature_K,
        std::move(original_family), frozen.bulk_gas_pool_fingerprint};
}

[[nodiscard]] inline QbmmLawInputs make_qbmm_law_inputs(
    const QbmmPopulationSchema& family, QbmmFrozenLawState frozen, const QbmmBulkGasPoolState& pool)
{
    frozen.bulk_gas_pool_moles = pool.moles;
    frozen.bulk_gas_pool_volume_m3 = pool.volume_m3 != 0 ? std::optional{pool.volume_m3} : std::nullopt;
    frozen.bulk_gas_pool_concentration_mol_per_m3 = pool.concentration_mol_per_m3;
    frozen.bulk_gas_pool_temperature_K = pool.temperature_K;
    frozen.bulk_gas_pool_fingerprint = pool.binding_fingerprint;
    return make_qbmm_law_inputs(family, std::move(frozen));
}

[[nodiscard]] inline QbmmNodeSlipInput make_qbmm_node_slip_input(const QbmmLawInputs& input, std::size_t atom,
    double phase_volume_m3, std::optional<vec3<double>> body_acceleration_m_per_s2 = {})
{
    if (!input.geometry || atom >= input.quadrature.active_atoms || atom >= input.geometry->active_atoms ||
        !input.phase_concentration_mol_per_m3 || !input.temperature_K || !input.gas_volume_fraction ||
        !std::isfinite(phase_volume_m3) || phase_volume_m3 <= 0 || !std::isfinite(input.absolute_pressure_pa) ||
        input.absolute_pressure_pa <= 0 || !std::isfinite(*input.temperature_K) || *input.temperature_K <= 0 ||
        !std::isfinite(*input.gas_volume_fraction) || *input.gas_volume_fraction <= 0)
        throw std::invalid_argument("QBMM node slip needs an active atom and a physical donor thermodynamic trace.");
    std::optional<std::span<const double>> liquid;
    if (input.carrier_liquid_concentration_mol_per_m3)
        liquid = *input.carrier_liquid_concentration_mol_per_m3;
    return {input.geometry->mass_kg_per_bubble[atom], input.geometry->diameter_m[atom],
        input.quadrature.atoms[atom].count, input.bulk_volume_m3, phase_volume_m3, input.absolute_pressure_pa,
        *input.temperature_K, *input.gas_volume_fraction, input.geometry->number_density, input.geometry->area_density,
        *input.phase_concentration_mol_per_m3, liquid, input.carrier_liquid_temperature_K, input.family_fingerprint,
        input.catalog_fingerprint, input.population, input.location, body_acceleration_m_per_s2};
}

[[nodiscard]] inline QbmmApplicabilityEvaluation evaluate_qbmm_node_slip_applicability(
    const QbmmLawMetadata& law, const QbmmLawInputs& input, std::size_t atom, std::optional<double> speed)
{
    if (law.kind != QbmmLawKind::NodeSlip || !input.geometry || atom >= input.quadrature.active_atoms ||
        atom >= input.geometry->active_atoms)
        throw std::invalid_argument("QBMM node slip applicability needs a node-slip contract and an active atom.");
    const QbmmLawInputs frozen{input.quadrature, input.geometry, input.mass_scale_kg, input.bulk_volume_m3,
        input.absolute_pressure_pa, input.temperature_K, input.time_step_s, input.phase_volume_fraction,
        input.gas_volume_fraction, input.phase_concentration_mol_per_m3, input.carrier_liquid_concentration_mol_per_m3,
        input.carrier_liquid_temperature_K, speed, input.family_fingerprint, input.catalog_fingerprint,
        input.population, input.location, input.geometry->mass_kg_per_bubble[atom], input.geometry->diameter_m[atom],
        input.carrier_liquid_volume_m3, input.carrier_liquid_dissolved_moles, input.bulk_gas_pool_volume_m3,
        input.bulk_gas_pool_moles, input.bulk_gas_pool_concentration_mol_per_m3, input.bulk_gas_pool_temperature_K,
        input.authoritative_family_moles, input.bulk_gas_pool_fingerprint};
    return evaluate_qbmm_applicability(law, frozen);
}
} // namespace SimpleFluid::Reacting
