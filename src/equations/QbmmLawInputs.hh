/** @file QbmmLawInputs.hh
 * @brief Pure construction of law inputs from frozen local physical quantities.
 */
#pragma once

#include "equations/QbmmPopulationSchema.hh"

namespace SimpleFluid::Reacting
{
[[nodiscard]] inline bool qbmm_is_transport_law(const QbmmLawMetadata& law)
{
    return law.kind == QbmmLawKind::PhaseSlip || law.kind == QbmmLawKind::TransportProperty;
}

[[nodiscard]] inline bool qbmm_uses_carrier_liquid(const QbmmLawMetadata& law)
{
    return std::ranges::any_of(law.ranges,
        [](const auto& range)
        {
            return range.variable == QbmmRangeVariable::CarrierLiquidConcentrationMolPerM3 ||
                   range.variable == QbmmRangeVariable::CarrierLiquidTemperatureK;
        });
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
    const double speed = std::hypot(frozen.drift_m_per_s[0], frozen.drift_m_per_s[1], frozen.drift_m_per_s[2]);
    const double fraction = frozen.phase_volume_m3 / frozen.bulk_volume_m3;
    if (!std::isfinite(speed) || !std::isfinite(fraction) || (frozen.phase_volume_m3 > 0 && fraction == 0))
        throw std::invalid_argument("QBMM law input phase fraction or slip speed is not representable.");
    return {frozen.quadrature, std::move(frozen.geometry), family.mass_scale_kg(), frozen.bulk_volume_m3,
        frozen.absolute_pressure_pa, absent ? std::nullopt : frozen.temperature_K, frozen.time_step_s, fraction,
        frozen.gas_volume_fraction, qbmm_phase_concentrations(frozen.family_moles, frozen.phase_volume_m3),
        std::move(frozen.carrier_liquid_concentration_mol_per_m3), frozen.carrier_liquid_temperature_K, speed,
        family.fingerprint(), family.catalog().fingerprint(), family.population(), frozen.location};
}
} // namespace SimpleFluid::Reacting
