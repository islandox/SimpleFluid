/** @file PressureConditionedThermal.hh
 * Frozen events and MPI-free pressure-conditioned constitutive evaluation.
 */
#pragma once
#include "equations/ReactingThermodynamics.hh"
#include <array>
#include <cstddef>
#include <limits>

namespace SimpleFluid::Reacting
{
struct ThermalLawIdentity
{
    std::string identity;
    std::uint64_t revision = 1, schema_fingerprint = 0;
    bool operator==(const ThermalLawIdentity&) const = default;
};
struct PartialMolarEnthalpyPoint
{
    std::vector<double> value, temperature, pressure; ///< J/mol, J/(mol K), m3/mol, catalog order.
};
struct PartialMolarEnthalpyLaw
{
    const ThermalLawIdentity metadata;
    const ThermodynamicBounds bounds;
    /** TRUE dH/dn_i at fixed T,p,n_others, not the mean mixture h reused per species.
     * This is pure MPI-free property evaluation, never source/event integration.
     */
    const std::function<PartialMolarEnthalpyPoint(double, double, std::span<const double>)> evaluate;
};
enum class InterphaseStageKind
{
    Heat,
    Material
};
struct MaterialTransferSlot
{
    SpeciesId species = 0;
    std::size_t donor = 0, recipient = 0;
    bool operator==(const MaterialTransferSlot&) const = default;
};
struct HeatTransferSlot
{
    std::size_t first = 0, second = 0;
    bool operator==(const HeatTransferSlot&) const = default;
};
struct ConfiguredInterphaseStage
{
    std::string identity;
    InterphaseStageKind kind = InterphaseStageKind::Material;
    std::vector<MaterialTransferSlot> material;
    std::vector<HeatTransferSlot> heat;
};
/** Constructor-copied immutable ordered slots and constitutive law metadata.
 * Kinetic callbacks may return AMOUNTS only; they cannot alter these bindings.
 */
struct ConditionedCaloricDomainProvider;
struct InterphaseConfiguration
{
    ThermalLawIdentity metadata;
    std::vector<ConfiguredInterphaseStage> stages;
    std::vector<std::shared_ptr<const PartialMolarEnthalpyLaw>> phase_enthalpy;
    std::shared_ptr<const ConditionedCaloricDomainProvider>
        caloric_domain; ///< No old affine inference for a composite plan.
};
struct FrozenInterphasePlan
{
    std::shared_ptr<const InterphaseConfiguration> configuration;
    std::vector<std::vector<double>> pre_moles, final_moles;
    std::vector<std::vector<double>> amounts; ///< Material nonnegative mol; heat signed J, configured slot order.
    std::vector<std::vector<long double>> raw_static_energy; ///< Original per-phase channels, never rounded sums.
    std::vector<long double> raw_declared_energy;
    std::vector<std::vector<std::array<long double, 2>>> raw_pre_moles; ///< Owner-local predecessor high/correction.
};
struct ConditionedCaloricQuery
{
    const FrozenInterphasePlan& plan;
    std::span<const PhaseThermalState> final_phases;
    std::span<const double> advected_energy, advected_volumes;
    CaloricPressureInterval reference_pressure;
    double mechanical_offset = 0;
    PhaseThermalMode mode = PhaseThermalMode::Equilibrium;
};
struct ConditionedCaloricDomainProvider
{
    const ThermalLawIdentity metadata;
    const std::function<CaloricPressureDomain(const ConditionedCaloricQuery&)>
        evaluate; ///< MPI-free whole-plan certificate/hint.
};
struct ThermalBatchIdentity
{
    std::string provider_identity;
    std::uint64_t provider_revision = 0, invocation = 0, corrector = 0, schema_fingerprint = 0;
    std::uint64_t accepted_generation = 0, old_geometry_epoch = 0, new_geometry_epoch = 0;
    bool operator==(const ThermalBatchIdentity&) const = default;
};
struct PressureConditionedThermalBatch
{
    std::shared_ptr<const InterphaseConfiguration> configuration;
    ThermalBatchIdentity source;
    std::vector<FrozenInterphasePlan> cells;
};
class SIMPLEFLUID_SOLVERS_EXPORT ThermalCaloricRangeError : public std::domain_error
{
public:
    using std::domain_error::domain_error;
};
struct ThermalEvaluationControls
{
    double physical_relative_tolerance = 2e-10;
    double caloric_relative_tolerance = 2e-9;
};
struct ThermalAnchorReceipt
{
    std::vector<double> temperature, temperature_pressure_derivative;
    std::vector<long double>
        enthalpy; ///< Compositions/immutable providers belong to the frozen plan, not duplicated per stage.
};
struct MaterialThermalReceipt
{
    std::size_t stage = 0, slot = 0;
    double molar_enthalpy = 0, enthalpy_temperature = 0, enthalpy_pressure = 0, donor_temperature_pressure = 0;
    long double carried_energy = 0, carried_energy_pressure_derivative = 0;
};
struct PressureConditionedThermalEvaluation
{
    std::vector<PhaseThermalState> phases;
    std::vector<double> volumes, applied_phase_energy;
    std::vector<std::array<long double, 2>>
        raw_applied_phase_energy; ///< Compensated high/correction before final narrowing.
    std::vector<ThermalAnchorReceipt> anchors;
    std::vector<MaterialThermalReceipt>
        material_events; ///< Exact active slot constitutive values, no reevaluation needed for audit.
    std::vector<long double> raw_material_enthalpy, raw_conductive_heat, raw_equilibrium_redistribution;
    double total_volume = 0, volume_pressure_derivative = 0, pressure_work = 0, work_magnitude = 0,
           pressure_operands = 0;
    double empty_phase_energy_roundoff = 0, energy_operand_roundoff_bound = 0, caloric_scale = 0;
    double raw_channel_roundoff_bound = 0; ///< Extra compensated static/dynamic channel uncertainty, J.
    double absolute_pressure = 0;
};
/** Validate configured topology/laws before callbacks. All phases share common p; sigma=0. */
SIMPLEFLUID_SOLVERS_EXPORT void validate_interphase_configuration(
    const InterphaseConfiguration& configuration, std::span<const PhaseThermalState> phase_templates);
/** Pure finite donor reservation. No heat recovery or rate callback; no incoming credit within a material stage. */
[[nodiscard]] SIMPLEFLUID_SOLVERS_EXPORT FrozenInterphasePlan compile_interphase_plan(
    std::shared_ptr<const InterphaseConfiguration> configuration, std::span<const PhaseThermalState> pre_phases,
    std::vector<std::vector<double>> amounts, std::vector<std::vector<long double>> raw_static_energy,
    std::vector<long double> raw_declared_energy);
/** Actual raw predecessor from earlier accepted local source stages. Phase/catalog
 * dimensions and double projection must match pre_phases. No MPI serialization.
 */
[[nodiscard]] SIMPLEFLUID_SOLVERS_EXPORT FrozenInterphasePlan compile_interphase_plan(
    std::shared_ptr<const InterphaseConfiguration> configuration, std::span<const PhaseThermalState> pre_phases,
    std::vector<std::vector<double>> amounts, std::vector<std::vector<long double>> raw_static_energy,
    std::vector<long double> raw_declared_energy,
    std::span<const std::vector<std::array<long double, 2>>> predecessor_parts);
/** Pressure-only constitutive/anchor evaluation; transferred amounts NEVER change here.
 * final_templates carries the applied rounded material endpoint. No kinetic/count provider is callable here.
 */
[[nodiscard]] SIMPLEFLUID_SOLVERS_EXPORT PressureConditionedThermalEvaluation evaluate_pressure_conditioned_thermal(
    const FrozenInterphasePlan& plan, std::span<const PhaseThermalState> final_templates,
    std::span<const double> advected_energy, std::span<const double> advected_volumes, double pressure,
    PhaseThermalMode mode, ThermalEvaluationControls controls = {});
/** Constitutive pressure bounds include occupied intermediate phases, not just final surviving phases. */
[[nodiscard]] SIMPLEFLUID_SOLVERS_EXPORT CaloricPressureInterval interphase_pressure_bounds(
    const FrozenInterphasePlan& plan, std::span<const PhaseThermalState> phase_templates);
} // namespace SimpleFluid::Reacting
