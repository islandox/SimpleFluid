/** @file ReactingThermodynamics.hh
 * @brief Composition-aware caloric EOS and conservative local phase-energy transactions.
 */
#pragma once

#include "SimpleFluidExport.hh"
#include "equations/ReactionMechanism.hh"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SimpleFluid::Reacting
{

struct ThermodynamicBounds
{
    double minimum_temperature = 0, maximum_temperature = 0; ///< K.
    double minimum_pressure = 0, maximum_pressure = 0;       ///< Pa absolute.
};

enum class EOSKind
{
    Liquid,
    Gas
};
enum class PhaseThermalMode
{
    Equilibrium,
    SeparatePhaseTemperatures
};

/** Explicit immutable EOS capability; Gas kind alone never implies ideality. */
enum class EnthalpyPressureDependence
{
    Unspecified,
    PressureIndependent
};

/** Mass-specific properties in one common caloric reference; derivatives hold composition fixed. */
struct MixtureThermodynamicPoint
{
    double density = 0;                       ///< kg/m3.
    double internal_energy = 0, enthalpy = 0; ///< J/kg; h=e+p/rho.
    double internal_energy_temperature = 0;   ///< (de/dT)_p, J/(kg K).
    double internal_energy_pressure = 0;      ///< (de/dp)_T, J/(kg Pa).
    double density_temperature = 0;           ///< (drho/dT)_p.
    double density_pressure = 0;    ///< (drho/dp)_T; gas positive, liquid may use the exact incompressible limit.
    std::vector<double> fugacities; ///< Species order, Pa; absent species have zero fugacity.
    std::vector<double> chemical_potentials;                ///< Species order, J/mol; optional equilibrium hook result.
    double dynamic_viscosity = 0, thermal_conductivity = 0; ///< Pa s, W/(m K); supplied transport closure.
};

/** Immutable shared provider. Nonideal gas requires fugacities or an explicit chemical-potential hook.
 * Captures must own immutable property data for the provider lifetime. No physical correlations are inferred.
 */
class SIMPLEFLUID_SOLVERS_EXPORT MixtureThermodynamics
{
public:
    using Evaluator = std::function<MixtureThermodynamicPoint(double, double, std::span<const double>)>;
    using ChemicalPotentialEvaluator = std::function<std::vector<double>(double, double, std::span<const double>)>;
    MixtureThermodynamics(std::shared_ptr<const SpeciesCatalog> catalog, ThermodynamicBounds bounds, EOSKind kind,
        Evaluator evaluator, EnergyConvention convention = EnergyConvention::IncludesFormationEnergy,
        ChemicalPotentialEvaluator chemical_potentials = {});
    MixtureThermodynamics(std::shared_ptr<const SpeciesCatalog> catalog, ThermodynamicBounds bounds, EOSKind kind,
        EnthalpyPressureDependence enthalpy_pressure_dependence, Evaluator evaluator,
        EnergyConvention convention = EnergyConvention::IncludesFormationEnergy,
        ChemicalPotentialEvaluator chemical_potentials = {});
    [[nodiscard]] EnthalpyPressureDependence enthalpy_pressure_dependence() const noexcept;
    [[nodiscard]] MixtureThermodynamicPoint evaluate(
        double temperature, double absolute_pressure, std::span<const double> mole_fractions) const;
    [[nodiscard]] const ThermodynamicBounds& bounds() const noexcept;
    [[nodiscard]] const SpeciesCatalog& catalog() const noexcept;
    [[nodiscard]] EOSKind kind() const noexcept;
    [[nodiscard]] EnergyConvention energy_convention() const noexcept;

private:
    std::shared_ptr<const SpeciesCatalog> catalog_;
    ThermodynamicBounds bounds_;
    EOSKind kind_;
    EnthalpyPressureDependence enthalpy_pressure_dependence_;
    Evaluator evaluator_;
    EnergyConvention convention_;
    ChemicalPotentialEvaluator chemical_potentials_;
};

struct IdealGasCaloricSpecies
{
    double molar_heat_capacity_v = 0;     ///< J/(mol K).
    double reference_temperature = 0;     ///< K.
    double reference_internal_energy = 0; ///< J/mol, including formation energy when selected.
};

using MixtureTransportEvaluator = std::function<std::pair<double, double>(double, double, std::span<const double>)>;
/** An omitted transport closure means an inviscid, nonconducting gas. */
[[nodiscard]] SIMPLEFLUID_SOLVERS_EXPORT std::shared_ptr<const MixtureThermodynamics> make_ideal_gas_mixture(
    std::shared_ptr<const SpeciesCatalog> catalog, std::vector<IdealGasCaloricSpecies> caloric,
    ThermodynamicBounds bounds, EnergyConvention convention = EnergyConvention::IncludesFormationEnergy,
    double gas_constant = 8.31446261815324, MixtureTransportEvaluator transport = {});

/** Supplied mixture property law; mass-specific Gibbs-consistent linear local liquid approximation.
 * The property callback sees composition; reference energy uses the same convention in every phase.
 * Zero compressibility is the exact constant-volume liquid limit and requires zero thermal expansion.
 */
struct LinearCompressibleMixtureLiquid
{
    double reference_temperature = 0, reference_pressure = 0, reference_density = 0;
    double isobaric_heat_capacity = 0, thermal_expansion = 0, compressibility = 0;
    double reference_internal_energy = 0;
    double dynamic_viscosity = 0, thermal_conductivity = 0;
};
using LiquidMixturePropertyEvaluator = std::function<LinearCompressibleMixtureLiquid(std::span<const double>)>;
[[nodiscard]] SIMPLEFLUID_SOLVERS_EXPORT std::shared_ptr<const MixtureThermodynamics>
make_linear_compressible_mixture_liquid(std::shared_ptr<const SpeciesCatalog> catalog, ThermodynamicBounds bounds,
    LiquidMixturePropertyEvaluator properties, EnergyConvention convention = EnergyConvention::IncludesFormationEnergy);

/** All quantities are extensive per cell/parcel. Each population bin gets an independent entry.
 * An empty phase has exactly zero energy and zero temperature; it never receives an invented caloric floor.
 */
struct PhaseThermalState
{
    std::shared_ptr<const MixtureThermodynamics> thermodynamics;
    std::vector<double> moles;
    double absolute_pressure = 0, internal_energy = 0, temperature = 0;
};

struct CaloricPressureInterval
{
    double lower = 0, upper = 0;
};
enum class CaloricDomainKnowledge
{
    CertifiedConnected,
    CertifiedEmpty,
    Unresolved
};
struct CaloricPressureDomain
{
    CaloricDomainKnowledge knowledge = CaloricDomainKnowledge::Unresolved;
    /** CertifiedConnected describes the entire connected feasible set in the query bounds.
     * For Unresolved, an optional interval is only a supplied bounded search bracket;
     * failure inside it does not prove that another physical branch is absent.
     */
    std::optional<CaloricPressureInterval> interval;
};
struct CaloricPressureQuery
{
    std::span<const PhaseThermalState> phases; ///< Actual transported+source inventories, before solver p*dV.
    std::span<const double> advected_volumes;
    std::span<const double> pressure_offsets; ///< p_phase = p0 + offset; no temperature/radius-dependent offsets.
    CaloricPressureInterval pressure;
    PhaseThermalMode mode = PhaseThermalMode::Equilibrium;
};
struct CaloricPressureMargins
{
    std::vector<long double> lower, upper; ///< U_packet+p_phase*Vadv-H(Tmin), H(Tmax)-U_packet-p_phase*Vadv.
};
/** Pure local query. EOS/derivative/provider errors propagate; no exception is a search signal here. */
[[nodiscard]] SIMPLEFLUID_SOLVERS_EXPORT CaloricPressureMargins evaluate_caloric_pressure_margins(
    const CaloricPressureQuery& query, double reference_pressure);
/** Exact affine feasibility only for explicitly pressure-independent enthalpy EOS.
 * Common-T mode sums physical phases once at the common T bounds. Other EOS return Unresolved.
 */
[[nodiscard]] SIMPLEFLUID_SOLVERS_EXPORT CaloricPressureDomain exact_caloric_pressure_domain(
    const CaloricPressureQuery& query);
struct CaloricPressureDomainProvider
{
    const std::string identity;
    const std::uint64_t revision;
    /** MPI-free, immutable and deterministic; receives the whole local physical phase group. */
    const std::function<CaloricPressureDomain(const CaloricPressureQuery&)> evaluate;
};
class SIMPLEFLUID_SOLVERS_EXPORT CaloricDomainUnresolved : public std::domain_error
{
public:
    using std::domain_error::domain_error;
};

struct PhaseThermalRecovery
{
    std::vector<PhaseThermalState> phases;
    std::vector<double> volumes;
    double total_internal_energy = 0, total_volume = 0;
    double volume_pressure_susceptibility = 0; ///< d(sum V)/dp at fixed moles and total/individual U.
};

/** Bounded inversion; throws without changing input on invalid states or unreachable energy.
 * Equilibrium consumes total_internal_energy; SeparatePhaseTemperatures requires it to equal sum phase U.
 * Chemical heat must not be added when already included in species caloric references.
 */
[[nodiscard]] SIMPLEFLUID_SOLVERS_EXPORT PhaseThermalRecovery recover_phase_temperatures(
    std::span<const PhaseThermalState> phases, double total_internal_energy,
    PhaseThermalMode mode = PhaseThermalMode::Equilibrium);

struct PhasePressureRecovery
{
    double reference_pressure = 0; ///< Absolute pressure of entry zero; other phase pressure offsets retained.
    PhaseThermalRecovery thermal;
};
/** Bounded stable-branch p(V,U,n) inversion with a common pressure increment across phases.
 * Phase pressure offsets (e.g. supplied capillary jumps) stay fixed for this local inversion.
 * Geometry-dependent bubble pressure/radius coupling requires an outer coupled solve.
 */
[[nodiscard]] SIMPLEFLUID_SOLVERS_EXPORT PhasePressureRecovery solve_phase_pressure(
    std::span<const PhaseThermalState> phases, double total_internal_energy, double target_volume,
    double minimum_reference_pressure, double maximum_reference_pressure,
    PhaseThermalMode mode = PhaseThermalMode::Equilibrium);

struct BubbleThermodynamicState
{
    double radius = 0, volume = 0, absolute_pressure = 0;
    MixtureThermodynamicPoint properties;
};
/** Inverts N*4*pi*r^3/3 = mass/rho(T,p_l+2*sigma/r,x); no ideal-gas shortcut. */
[[nodiscard]] SIMPLEFLUID_SOLVERS_EXPORT BubbleThermodynamicState solve_bubble_radius(const MixtureThermodynamics& gas,
    std::span<const double> moles, double temperature, double liquid_pressure, double surface_tension,
    double bubble_number, double minimum_radius, double maximum_radius);

/** Integrated closure packet. Positive moles/enthalpy/heat flow from phase first to phase second.
 * Carried enthalpy is supplied once per species (J/mol); heat is additional conductive exchange only.
 * With formation-energy caloric references, no extra latent/chemical heat may accompany this packet.
 */
struct IsobaricPhaseVolumeWork
{
    double first_volume_change = 0, second_volume_change = 0; ///< Supplied coupled EOS trial, m3.
};

struct InterphaseExchange
{
    std::vector<double> moles_first_to_second;
    std::vector<double> carried_molar_enthalpy;
    double heat_first_to_second = 0;
    std::optional<IsobaricPhaseVolumeWork> phase_volume_work = std::nullopt;
};
using InterphaseExchangeClosure = std::function<InterphaseExchange(const PhaseThermalState&,
    const MixtureThermodynamicPoint&, const PhaseThermalState&, const MixtureThermodynamicPoint&)>;

struct InterphaseExchangeResult
{
    PhaseThermalRecovery thermal;
    double mass_first_to_second = 0, enthalpy_first_to_second = 0, heat_first_to_second = 0;
    double pressure_work_on_first = 0, pressure_work_on_second = 0, external_mechanical_work = 0;
    double empty_phase_energy_roundoff = 0; ///< Numerical operand cancellation, never a physical heat source.
};
/** Pure atomic preview: joint donors and energy/range gates reject overdraw, never clip.
 * Two specified phases may be liquid/resolved gas or any explicitly owned gas population.
 * Optional isobaric phase-volume work is -p*deltaV; candidate EOS volumes must match the supplied trial.
 * Its net mechanical work is an external energy ledger, separate from paired interfacial enthalpy/heat.
 * Bubble collapse is not species destruction. A zeroed phase must carry exactly zero U.
 */
[[nodiscard]] SIMPLEFLUID_SOLVERS_EXPORT InterphaseExchangeResult preview_interphase_exchange(
    std::span<const PhaseThermalState> phases, std::size_t first, std::size_t second,
    const InterphaseExchangeClosure& closure, PhaseThermalMode mode = PhaseThermalMode::Equilibrium);

} // namespace SimpleFluid::Reacting
