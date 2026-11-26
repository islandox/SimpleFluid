/** @file QbmmConstitutiveLaws.hh
 * @brief Supplied immutable SI law contracts and local applicability evaluation.
 */
#pragma once

#include "dataclass/vec3.hh"
#include "equations/QbmmMoments.hh"
#include "equations/ReactionMechanism.hh"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace SimpleFluid::Reacting
{
enum class QbmmLawKind
{
    IntegratedPopulation,
    Nucleation,
    MassTransfer,
    Coalescence,
    Breakup,
    PhaseSlip,
    TransportProperty,
    NodeSlip,
    BulkGasConversion
};

/** Each variable fixes its SI basis. Diameter bounds apply to every active atom. */
enum class QbmmRangeVariable
{
    AbsolutePressurePa,
    TemperatureK,
    TimeStepS,
    BulkVolumeM3,
    PhaseVolumeFraction,
    GasVolumeFraction,
    NumberDensityPerM3,
    DiameterM,
    AreaDensityPerM,
    PhaseConcentrationMolPerM3,
    CarrierLiquidConcentrationMolPerM3,
    CarrierLiquidTemperatureK,
    SlipSpeedMPerS,
    NodeMassKgPerBubble,
    BulkGasPoolVolumeM3,
    BulkGasPoolConcentrationMolPerM3,
    BulkGasPoolTemperatureK,
    BulkGasPoolVolumeFraction
};
enum class QbmmApplicabilityGate
{
    ReportOnly,
    RejectOutside
};
enum class QbmmRangeStatus
{
    WithinRange,
    OutsideRange,
    MissingInput,
    NotApplicable
};

struct QbmmSiRange
{
    QbmmRangeVariable variable = QbmmRangeVariable::AbsolutePressurePa;
    double minimum = 0, maximum = 0;          ///< Inclusive, finite bounds, without numerical tolerance or clipping.
    std::optional<std::size_t> species_index; ///< Required for a concentration range, in catalog order.
    bool operator==(const QbmmSiRange&) const = default;
};

/** Metadata is copied into the immutable family schema, never inferred from species names.
 * input_basis/output_units declare the supplied law's physical interpretation, including
 * any efficiency or dissipation basis. A declared range is not empirical validation.
 */
struct QbmmLawMetadata
{
    std::string identity;
    std::uint64_t revision = 1;
    QbmmLawKind kind = QbmmLawKind::IntegratedPopulation;
    std::string input_basis, output_units;
    QbmmApplicabilityGate gate = QbmmApplicabilityGate::ReportOnly;
    bool requires_population = true;
    std::vector<QbmmSiRange> ranges;
    bool operator==(const QbmmLawMetadata&) const = default;
};

[[nodiscard]] inline bool qbmm_uses_carrier_liquid(const QbmmLawMetadata& law)
{
    return std::ranges::any_of(law.ranges,
        [](const auto& range)
        {
            return range.variable == QbmmRangeVariable::CarrierLiquidConcentrationMolPerM3 ||
                   range.variable == QbmmRangeVariable::CarrierLiquidTemperatureK;
        });
}

[[nodiscard]] inline bool qbmm_uses_bulk_gas_pool(const QbmmLawMetadata& law)
{
    return law.kind == QbmmLawKind::BulkGasConversion ||
           std::ranges::any_of(law.ranges,
               [](const auto& range)
               {
                   return range.variable == QbmmRangeVariable::BulkGasPoolVolumeM3 ||
                          range.variable == QbmmRangeVariable::BulkGasPoolConcentrationMolPerM3 ||
                          range.variable == QbmmRangeVariable::BulkGasPoolTemperatureK ||
                          range.variable == QbmmRangeVariable::BulkGasPoolVolumeFraction;
               });
}

inline void validate_qbmm_law_metadata(const QbmmLawMetadata& law, std::size_t species_count)
{
    if (law.identity.empty() || law.revision == 0 || law.input_basis.empty() || law.output_units.empty() ||
        law.kind < QbmmLawKind::IntegratedPopulation || law.kind > QbmmLawKind::BulkGasConversion ||
        (law.kind == QbmmLawKind::NodeSlip && (!law.requires_population || law.output_units != "m/s")) ||
        (law.gate != QbmmApplicabilityGate::ReportOnly && law.gate != QbmmApplicabilityGate::RejectOutside))
        throw std::invalid_argument("QBMM law needs an identity, revision and explicit SI input/output basis.");
    for (const auto& range : law.ranges)
    {
        const bool concentration = range.variable == QbmmRangeVariable::PhaseConcentrationMolPerM3 ||
                                   range.variable == QbmmRangeVariable::CarrierLiquidConcentrationMolPerM3 ||
                                   range.variable == QbmmRangeVariable::BulkGasPoolConcentrationMolPerM3;
        if (range.variable < QbmmRangeVariable::AbsolutePressurePa ||
            range.variable > QbmmRangeVariable::BulkGasPoolVolumeFraction ||
            (range.variable == QbmmRangeVariable::NodeMassKgPerBubble && law.kind != QbmmLawKind::NodeSlip) ||
            !std::isfinite(range.minimum) || !std::isfinite(range.maximum) || range.minimum < 0 ||
            range.maximum < range.minimum || concentration != range.species_index.has_value() ||
            (concentration && *range.species_index >= species_count))
            throw std::invalid_argument("Invalid QBMM SI applicability range or concentration species index.");
    }
}

/** Stable binary metadata encoding, shared by family and controller compatibility signatures.
 * Bounds are encoded by their exact double bits; identities and vector sizes are delimited.
 */
[[nodiscard]] inline std::string qbmm_law_signature(const QbmmLawMetadata& law)
{
    std::string result;
    const auto number = [&](std::uint64_t value)
    {
        for (unsigned shift = 0; shift < 64; shift += 8)
            result.push_back(static_cast<char>(value >> shift));
    };
    const auto string = [&](const std::string& value)
    {
        number(value.size());
        result += value;
    };
    string(law.identity);
    number(law.revision);
    number(static_cast<std::uint64_t>(law.kind));
    string(law.input_basis);
    string(law.output_units);
    number(static_cast<std::uint64_t>(law.gate));
    number(law.requires_population);
    number(law.ranges.size());
    for (const auto& range : law.ranges)
    {
        number(static_cast<std::uint64_t>(range.variable));
        number(std::bit_cast<std::uint64_t>(range.minimum));
        number(std::bit_cast<std::uint64_t>(range.maximum));
        number(range.species_index.has_value());
        number(range.species_index.value_or(0));
    }
    return result;
}

/** Owned, frozen local values. Helpers never consult an EOS, fields, clocks or MPI.
 * A missing duration (e.g. initialized/restarted output) is not an assumed unit step.
 * Phase concentrations and temperature are disengaged for exact phase absence.
 */
struct QbmmLawInputs
{
    const QbmmQuadrature quadrature;
    const std::optional<QbmmGeometryView> geometry;
    const double mass_scale_kg, bulk_volume_m3, absolute_pressure_pa;
    const std::optional<double> temperature_K, time_step_s;
    const double phase_volume_fraction;
    const std::optional<double> gas_volume_fraction;
    const std::optional<std::vector<double>> phase_concentration_mol_per_m3;
    const std::optional<std::vector<double>> carrier_liquid_concentration_mol_per_m3{}; ///< Physical liquid EOS basis.
    const std::optional<double> carrier_liquid_temperature_K{}; ///< Disengaged for absent physical liquid.
    const std::optional<double> slip_speed_m_per_s{};           ///< Supplied common family drift, not carrier speed.
    const std::uint64_t family_fingerprint = 0, catalog_fingerprint = 0;
    const std::string population{}; ///< Stable family identity; quadrature indices carry no material identity.
    const std::optional<ReactionLocation> location{}; ///< Source-donor centroid, cell identity and geometry epoch.
    const std::optional<double> node_mass_kg_per_bubble{}, node_diameter_m{};
    const std::optional<double> carrier_liquid_volume_m3{};                    ///< Occupied physical liquid EOS volume.
    const std::optional<std::vector<double>> carrier_liquid_dissolved_moles{}; ///< Original Dissolved mol/cell.
    const std::optional<double> bulk_gas_pool_volume_m3{};                     ///< Occupied physical pool EOS volume.
    const std::optional<std::vector<double>> bulk_gas_pool_moles{};            ///< All zeros for a bound, absent pool.
    const std::optional<std::vector<double>> bulk_gas_pool_concentration_mol_per_m3{};
    const std::optional<double> bulk_gas_pool_temperature_K{};
    const std::optional<std::vector<double>> authoritative_family_moles{}; ///< Original stock for exact pool drainage.
    const std::optional<std::uint64_t> bulk_gas_pool_fingerprint{}; ///< Explicit authority for conversion events.
};

/** Instantaneous SI input for a single active atom. Concentration spans are borrowed for this call.
 * The law may depend on physical geometry and thermodynamic state. The numerical substep is not an input.
 */
struct QbmmNodeSlipInput
{
    double mass_kg_per_bubble, diameter_m, count;
    double bulk_volume_m3, phase_volume_m3, absolute_pressure_pa, temperature_K;
    double gas_volume_fraction, number_density_per_m3, area_density_per_m;
    std::span<const double> phase_concentration_mol_per_m3; ///< Occupied donor gas EOS volume basis.
    std::optional<std::span<const double>> carrier_liquid_concentration_mol_per_m3; ///< Liquid EOS basis.
    std::optional<double> carrier_liquid_temperature_K;
    std::uint64_t family_fingerprint, catalog_fingerprint;
    std::string_view population;
    std::optional<ReactionLocation> location;
    std::optional<vec3<double>> body_acceleration_m_per_s2{}; ///< Engaged only for an explicitly dependent law.
};

/** Immutable transport closure. The supplied callback must be deterministic and MPI-free. */
struct QbmmNodeSlipLaw
{
    const QbmmLawMetadata metadata;
    const std::function<std::array<double, 3>(const QbmmNodeSlipInput&)> evaluate; ///< Relative velocity in m/s.
    /** Immutable SI dependency. When engaged, the solver supplies this vector to the law and requires
     * the reacting controller to use the same acceleration. A zero vector is an explicit dependency.
     */
    const std::optional<vec3<double>> required_body_acceleration_m_per_s2{};
};

/** Extend the original kinetic law encoding only when an acceleration dependency is declared. */
[[nodiscard]] inline std::string qbmm_node_slip_signature(const QbmmNodeSlipLaw& law)
{
    auto result = qbmm_law_signature(law.metadata);
    if (const auto& acceleration = law.required_body_acceleration_m_per_s2)
    {
        result += "body-acceleration-si-v1";
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const double value = acceleration->component(axis);
            const auto bits = std::bit_cast<std::uint64_t>(value);
            for (unsigned shift = 0; shift < 64; shift += 8)
                result.push_back(static_cast<char>(bits >> shift));
        }
    }
    return result;
}

/** No epsilon volume or concentration exists for an absent phase. */
[[nodiscard]] inline std::optional<std::vector<double>> qbmm_phase_concentrations(
    std::span<const double> moles, double phase_volume_m3)
{
    if (!std::isfinite(phase_volume_m3) || phase_volume_m3 < 0)
        throw std::invalid_argument("QBMM concentration needs a nonnegative finite physical phase volume.");
    std::vector<double> result;
    result.reserve(moles.size());
    for (const double amount : moles)
    {
        if (!std::isfinite(amount) || amount < 0 || (phase_volume_m3 == 0 && amount != 0))
            throw std::invalid_argument("QBMM concentration has invalid material or inconsistent phase absence.");
        if (phase_volume_m3 > 0)
        {
            const double concentration = amount / phase_volume_m3;
            if (!std::isfinite(concentration) || (amount > 0 && concentration == 0))
                throw std::invalid_argument("QBMM phase concentration is not representable.");
            result.push_back(concentration);
        }
    }
    if (phase_volume_m3 == 0)
        return std::nullopt;
    if (std::ranges::all_of(moles, [](double amount) { return amount == 0; }))
        throw std::invalid_argument("A material-empty QBMM phase cannot have positive physical volume.");
    return result;
}

struct QbmmRangeEvaluation
{
    QbmmSiRange declared;
    QbmmRangeStatus status = QbmmRangeStatus::MissingInput;
    std::optional<double> observed_minimum, observed_maximum;
};
struct QbmmApplicabilityEvaluation
{
    QbmmLawMetadata metadata;
    std::vector<QbmmRangeEvaluation> ranges;
    bool applicable = false; ///< All applicable declared ranges evaluated and satisfied.
    bool gate_passed = true;
};

[[nodiscard]] inline QbmmApplicabilityEvaluation evaluate_qbmm_applicability(
    const QbmmLawMetadata& law, const QbmmLawInputs& input)
{
    // Catalog-index validity is checked when binding the immutable schema. Standalone
    // evaluation still validates the complete structural/range contract, even at absence.
    validate_qbmm_law_metadata(law, std::numeric_limits<std::size_t>::max());
    QbmmApplicabilityEvaluation result{law, {}, true, true};
    const bool absent = input.quadrature.status == InversionStatus::Absent;
    const bool dryout = input.carrier_liquid_volume_m3 == 0.0 && qbmm_uses_carrier_liquid(law);
    const bool inactive = (absent && law.requires_population) || dryout;
    const bool absent_pool = !input.bulk_gas_pool_volume_m3 && input.bulk_gas_pool_moles &&
                             std::ranges::all_of(*input.bulk_gas_pool_moles, [](double amount) { return amount == 0; });
    if (inactive)
        result.applicable = false;
    for (const auto& range : law.ranges)
    {
        QbmmRangeEvaluation evaluation{range, QbmmRangeStatus::MissingInput, {}, {}};
        const bool absent_pool_range =
            absent_pool && (range.variable == QbmmRangeVariable::BulkGasPoolConcentrationMolPerM3 ||
                               range.variable == QbmmRangeVariable::BulkGasPoolTemperatureK);
        if (absent_pool_range)
            result.applicable = false;
        if (inactive || absent_pool_range ||
            (law.kind == QbmmLawKind::NodeSlip &&
                ((range.variable == QbmmRangeVariable::TimeStepS && !input.time_step_s) ||
                    (range.variable == QbmmRangeVariable::SlipSpeedMPerS && !input.slip_speed_m_per_s))))
            evaluation.status = QbmmRangeStatus::NotApplicable;
        else
        {
            std::optional<double> lo, hi;
            switch (range.variable)
            {
                case QbmmRangeVariable::AbsolutePressurePa:
                    lo = input.absolute_pressure_pa;
                    break;
                case QbmmRangeVariable::TemperatureK:
                    lo = input.temperature_K;
                    break;
                case QbmmRangeVariable::TimeStepS:
                    lo = input.time_step_s;
                    break;
                case QbmmRangeVariable::BulkVolumeM3:
                    lo = input.bulk_volume_m3;
                    break;
                case QbmmRangeVariable::PhaseVolumeFraction:
                    lo = input.phase_volume_fraction;
                    break;
                case QbmmRangeVariable::GasVolumeFraction:
                    lo = input.gas_volume_fraction;
                    break;
                case QbmmRangeVariable::NumberDensityPerM3:
                    if (input.geometry)
                        lo = input.geometry->number_density;
                    break;
                case QbmmRangeVariable::AreaDensityPerM:
                    if (input.geometry)
                        lo = input.geometry->area_density;
                    break;
                case QbmmRangeVariable::DiameterM:
                    if (input.node_diameter_m)
                        lo = input.node_diameter_m;
                    else if (input.geometry)
                    {
                        lo = input.geometry->min_diameter;
                        hi = input.geometry->max_diameter;
                    }
                    break;
                case QbmmRangeVariable::PhaseConcentrationMolPerM3:
                    if (input.phase_concentration_mol_per_m3 && range.species_index &&
                        *range.species_index < input.phase_concentration_mol_per_m3->size())
                        lo = input.phase_concentration_mol_per_m3->at(*range.species_index);
                    break;
                case QbmmRangeVariable::CarrierLiquidConcentrationMolPerM3:
                    if (input.carrier_liquid_concentration_mol_per_m3 && range.species_index &&
                        *range.species_index < input.carrier_liquid_concentration_mol_per_m3->size())
                        lo = input.carrier_liquid_concentration_mol_per_m3->at(*range.species_index);
                    break;
                case QbmmRangeVariable::CarrierLiquidTemperatureK:
                    lo = input.carrier_liquid_temperature_K;
                    break;
                case QbmmRangeVariable::SlipSpeedMPerS:
                    lo = input.slip_speed_m_per_s;
                    break;
                case QbmmRangeVariable::NodeMassKgPerBubble:
                    lo = input.node_mass_kg_per_bubble;
                    break;
                case QbmmRangeVariable::BulkGasPoolVolumeM3:
                    lo = absent_pool ? std::optional<double>{0} : input.bulk_gas_pool_volume_m3;
                    break;
                case QbmmRangeVariable::BulkGasPoolConcentrationMolPerM3:
                    if (input.bulk_gas_pool_concentration_mol_per_m3 && range.species_index &&
                        *range.species_index < input.bulk_gas_pool_concentration_mol_per_m3->size())
                        lo = input.bulk_gas_pool_concentration_mol_per_m3->at(*range.species_index);
                    break;
                case QbmmRangeVariable::BulkGasPoolTemperatureK:
                    lo = input.bulk_gas_pool_temperature_K;
                    break;
                case QbmmRangeVariable::BulkGasPoolVolumeFraction:
                    if (absent_pool)
                        lo = 0;
                    else if (input.bulk_gas_pool_volume_m3 && input.bulk_volume_m3 > 0)
                        lo = *input.bulk_gas_pool_volume_m3 / input.bulk_volume_m3;
                    break;
            }
            if (!hi)
                hi = lo;
            if (lo && hi && std::isfinite(*lo) && std::isfinite(*hi))
            {
                evaluation.observed_minimum = lo;
                evaluation.observed_maximum = hi;
                evaluation.status = *lo >= range.minimum && *hi <= range.maximum ? QbmmRangeStatus::WithinRange
                                                                                 : QbmmRangeStatus::OutsideRange;
            }
            if (evaluation.status != QbmmRangeStatus::WithinRange)
            {
                result.applicable = false;
                if (law.gate == QbmmApplicabilityGate::RejectOutside)
                    result.gate_passed = false;
            }
        }
        result.ranges.push_back(std::move(evaluation));
    }
    return result;
}

struct QbmmSlipEvaluation
{
    std::array<double, 3> velocity_m_per_s;
    QbmmApplicabilityEvaluation applicability;
};

/** Evaluate an already supplied local slip field, including during initialization.
 * No callback, field write, EOS evaluation, clock access or MPI is involved. Nonfinite
 * components/speed always reject. Declared ranges follow the metadata's gating policy.
 */
[[nodiscard]] inline QbmmSlipEvaluation evaluate_qbmm_supplied_slip(
    const QbmmLawMetadata& metadata, const QbmmLawInputs& input, const std::array<double, 3>& velocity_m_per_s)
{
    validate_qbmm_law_metadata(metadata, std::numeric_limits<std::size_t>::max());
    if (metadata.kind != QbmmLawKind::PhaseSlip ||
        !std::ranges::all_of(velocity_m_per_s, [](double value) { return std::isfinite(value); }))
        throw std::invalid_argument("QBMM supplied slip needs a phase-slip contract and finite SI components.");
    const double speed = std::hypot(velocity_m_per_s[0], velocity_m_per_s[1], velocity_m_per_s[2]);
    if (!std::isfinite(speed))
        throw std::invalid_argument("QBMM supplied slip speed is not representable.");
    const QbmmLawInputs frozen{input.quadrature, input.geometry, input.mass_scale_kg, input.bulk_volume_m3,
        input.absolute_pressure_pa, input.temperature_K, input.time_step_s, input.phase_volume_fraction,
        input.gas_volume_fraction, input.phase_concentration_mol_per_m3, input.carrier_liquid_concentration_mol_per_m3,
        input.carrier_liquid_temperature_K, speed, input.family_fingerprint, input.catalog_fingerprint,
        input.population, input.location, {}, {}, input.carrier_liquid_volume_m3, input.carrier_liquid_dissolved_moles,
        input.bulk_gas_pool_volume_m3, input.bulk_gas_pool_moles, input.bulk_gas_pool_concentration_mol_per_m3,
        input.bulk_gas_pool_temperature_K, input.authoritative_family_moles, input.bulk_gas_pool_fingerprint};
    auto applicability = evaluate_qbmm_applicability(metadata, frozen);
    if (!applicability.gate_passed)
        throw std::invalid_argument(
            "QBMM supplied slip is outside its declared applicability or lacks required inputs.");
    return {velocity_m_per_s, std::move(applicability)};
}
} // namespace SimpleFluid::Reacting
