/** @file VoFLowMach.hh
 * @brief Thermodynamic and extensive-inventory contracts for low-Mach VoF.
 */
#pragma once

#include "FVM/GeometricVoF.hh"
#include "SimpleFluidExport.hh"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
namespace SimpleFluid::Meshes::AMR
{
class CartesianAMR;
}

namespace SimpleFluid::VoF
{

struct ThermodynamicBounds
{
    double minimum_temperature = 0; ///< K; caller must supply the validity range.
    double maximum_temperature = 0;
    double minimum_pressure = 0; ///< Pa absolute.
    double maximum_pressure = 0;
};

struct LiquidThermodynamicPoint
{
    double density = 0;                     ///< kg/m3 of true aqueous liquid.
    double internal_energy = 0;             ///< J/kg, using the declared common reference.
    double enthalpy = 0;                    ///< J/kg; h=e+p/rho.
    double internal_energy_temperature = 0; ///< (de/dT)_p, J/(kg K), positive.
    double dynamic_viscosity = 0;           ///< Pa s.
    double thermal_conductivity = 0;        ///< W/(m K).
};

/** Immutable, rank-local callback; implementations must be thermodynamically consistent. */
class SIMPLEFLUID_SOLVERS_EXPORT LiquidThermodynamics
{
public:
    using Evaluator = std::function<LiquidThermodynamicPoint(double, double)>;
    LiquidThermodynamics(ThermodynamicBounds bounds, Evaluator evaluator);
    [[nodiscard]] const ThermodynamicBounds& bounds() const noexcept;
    [[nodiscard]] LiquidThermodynamicPoint evaluate(double temperature, double absolute_pressure) const;

private:
    ThermodynamicBounds bounds_;
    Evaluator evaluator_;
};

/** Parameters of a Gibbs-consistent local liquid approximation, not aqueous correlations. */
struct LinearCompressibleLiquid
{
    ThermodynamicBounds bounds;
    double reference_temperature = 0;
    double reference_pressure = 0;
    double reference_density = 0;
    double isobaric_heat_capacity = 0;
    double thermal_expansion = 0; ///< Reference volumetric alpha, 1/K.
    double compressibility = 0;   ///< Reference isothermal beta, 1/Pa; positive.
    double reference_internal_energy = 0;
    double dynamic_viscosity = 0;
    double thermal_conductivity = 0;
};

[[nodiscard]] SIMPLEFLUID_SOLVERS_EXPORT std::shared_ptr<const LiquidThermodynamics> make_linear_compressible_liquid(
    LinearCompressibleLiquid parameters);

struct IdealGasSpecies
{
    double molar_mass = 0;                ///< kg/mol; explicitly supplied for H2 and inert species.
    double molar_heat_capacity_v = 0;     ///< J/(mol K), positive.
    double reference_temperature = 0;     ///< K.
    double reference_internal_energy = 0; ///< J/mol.
    double dynamic_viscosity = 0;
    double thermal_conductivity = 0;
};

struct LowMachOptions
{
    std::shared_ptr<const LiquidThermodynamics> liquid;
    IdealGasSpecies hydrogen;
    IdealGasSpecies inert;
    double gas_constant = 8.31446261815324; ///< J/(mol K).
    double minimum_bubble_radius = 1e-12;
    double maximum_bubble_radius = 1.0;
    int maximum_correctors = 80;
    int maximum_thermodynamic_iterations = 100;
    double volume_relative_tolerance = 2e-9;
    double energy_relative_tolerance = 2e-10;
    double inventory_relative_tolerance = 2e-11;
    double relaxation = 0.5;
};

/** Every number/molar quantity is extensive per owned cell. */
struct DispersedPopulation
{
    std::string name;
    std::vector<double> number;
    std::vector<double> hydrogen_moles;
    std::vector<Vector> drift_velocity;    ///< Relative to the liquid/continuous velocity, m/s.
    std::vector<double> volume;            ///< Derived true gas volume, m3.
    std::vector<double> radius;            ///< Derived from N,n,T,p_l and Laplace pressure, m.
    std::vector<double> absolute_pressure; ///< Pa; retained phase history while representation relaxation is pending.
};

struct LowMachInitialState
{
    double thermodynamic_pressure = 0; ///< Uniform offset P0, Pa absolute.
    std::vector<double> temperature;
    std::vector<double> resolved_fraction;
    std::vector<double> resolved_hydrogen_mole_fraction; ///< Empty means pure H2.
    std::vector<double> dissolved_hydrogen_moles;
    std::vector<double> donor_hydrogen_moles; ///< Finite chemically bound H2-equivalent stock.
    std::vector<DispersedPopulation> dispersed;
    std::vector<Vector> velocity; ///< Continuous/liquid velocity, not barycentric velocity.
};

struct LowMachState
{
    bool pending_representation_relaxation = false;
    double thermodynamic_pressure = 0;
    std::vector<double> liquid_mass;
    std::vector<double> resolved_hydrogen_moles;
    std::vector<double> resolved_inert_moles;
    std::vector<double> dissolved_hydrogen_moles;
    std::vector<double> donor_hydrogen_moles;
    std::vector<DispersedPopulation> dispersed;
    std::vector<double> internal_energy; ///< Total liquid/resolved/dispersed internal energy, J.
    std::vector<Vector> momentum;        ///< Physical kg m/s: rho V u + sum(m_d w_d).
    std::vector<double> temperature;
    std::vector<double> resolved_fraction;
    std::vector<double> dispersed_fraction;
    std::vector<double> liquid_fraction;
    std::vector<double> liquid_density;
    std::vector<double> resolved_density;
    std::vector<double> liquid_absolute_pressure;
    std::vector<double> resolved_absolute_pressure;
    std::vector<double> resolved_hydrogen_partial_pressure;
};

/** A phase parcel is measured against the accepted source phase volume. */
struct RepresentationGasParcel
{
    std::uint64_t source_cell = 0, target_cell = 0;
    int source_population = -1, target_population = -1; ///< -1 is resolved gas; otherwise an existing population index.
    double volume = 0;                                  ///< Source-state gas volume, m3.
    double target_number = 0; ///< Required for resolved-to-dispersed conversion; no implicit number creation.
};

/** Entire remaining carrier, sampled after gas withdrawals and before gas deposits. */
struct RepresentationCarrierParcel
{
    std::uint64_t source_cell = 0, target_cell = 0;
    double volume = 0; ///< Liquid plus all remaining dispersed gas, m3.
};

/** Sparse parcels occur once on their event owner; referenced cells may be remote. */
struct RepresentationChangeRequest
{
    std::shared_ptr<const SimpleFluid::Meshes::AMR::CartesianAMR> source_hierarchy;
    std::uint64_t accepted_state_revision = 0;
    std::vector<RepresentationGasParcel> gas_parcels;
    std::vector<RepresentationCarrierParcel> carrier_parcels;
};

struct RepresentationDiagnostics
{
    double gas_volume_promoted = 0, gas_volume_demoted = 0;
    double hydrogen_promoted = 0, hydrogen_demoted = 0;
    double dispersed_number_removed = 0, dispersed_number_added = 0;
    double carrier_volume_moved = 0;
    double mass_error = 0, hydrogen_error = 0, inert_error = 0, internal_energy_error = 0;
    double internal_energy_roundoff_bound = 0; ///< Packet-operand forward-error allowance, J.
    Vector momentum_error{}, momentum_roundoff_bound{};
    double maximum_volume_error = 0;
    double maximum_face_divergence = 0;
    double resolved_area_change = 0, dispersed_area_change = 0;
    double surface_energy_change = 0; ///< Diagnostic sigma*delta(area); never an automatic heat source.
};

struct PopulationSource
{
    std::string name;
    std::vector<double> number_change;
    std::vector<double> hydrogen_change;
    std::vector<Vector> drift_velocity; ///< Empty retains the accepted drift.
};

/**
 * Immutable integrated source packet for one accepted-old -> candidate trial.
 * Empty arrays mean zero. Internal exchange and external venting are distinct.
 * The controller derives chemical liquid-mass changes; diffusion may be
 * counterdiffusive with zero net liquid-mass change. Internal intercell
 * momentum/enthalpy transfers must sum to zero globally.
 */
struct LowMachSources
{
    std::vector<double> liquid_mass_change;
    std::vector<double> resolved_hydrogen_change;
    std::vector<double> resolved_inert_change;
    std::vector<double> dissolved_hydrogen_change;
    std::vector<double> donor_hydrogen_change;
    std::vector<PopulationSource> dispersed;
    std::vector<double> net_thermal_energy;  ///< J; chemical-energy convention belongs to the caller.
    std::vector<double> radiolysis_energy;   ///< J used for chemistry diagnostics; NEVER added as heat again.
    std::vector<double> enthalpy_transfer;   ///< Signed internal intercell transfer, J.
    std::vector<Vector> momentum_transfer;   ///< Signed internal transfer, kg m/s.
    std::vector<double> vent_hydrogen_moles; ///< Nonnegative withdrawal from designated resolved gas.
    std::vector<double> vent_inert_moles;
    std::vector<double> vent_enthalpy;     ///< Signed enthalpy leaving in the chosen EOS reference, J.
    std::vector<Vector> vent_momentum;     ///< Physical momentum leaving the domain.
    std::vector<double> produced_hydrogen; ///< Diagnostics only, mol.
    std::vector<double> captured_hydrogen; ///< Diagnostics only, mol.
    std::vector<double> surface_tension;   ///< Optional validation: must equal the configured constant sigma.
};

/** Signed integrated canonical-owner-outward face amounts; overlap ordering. */
struct LowMachTransport
{
    std::vector<double> resolved_volume, carrier_volume;
    std::vector<double> liquid_mass, resolved_hydrogen, resolved_inert;
    std::vector<double> dissolved_hydrogen, donor_hydrogen;
    std::vector<std::vector<double>> dispersed_number, dispersed_hydrogen, dispersed_slip_volume;
    std::vector<double> internal_energy;
    std::vector<double> conductive_energy;
    std::vector<Vector> momentum;
    std::vector<double> gas_volume_increment, carrier_volume_increment; ///< Raw owned physical changes, m3.
    double time_step = 0;
};

struct LowMachDiagnostics
{
    Vector momentum_error{};          ///< Measured periodic physical-momentum balance [kg m/s].
    Vector momentum_roundoff_bound{}; ///< Forward-error bound from assembled impulse terms [kg m/s].
    int correctors = 0;
    double maximum_occupancy_error = 0;
    double global_volume_error = 0;
    double maximum_continuity_residual = 0; ///< 1/s, relative to the nonzero target.
    double total_mass = 0;
    double hydrogen_inventory = 0; ///< Includes chemically bound donor stock.
    double inert_inventory = 0;
    double internal_energy = 0;
    double pressure_work = 0; ///< Integrated -sum(p_a Q_a), J.
    double mass_error = 0;
    double hydrogen_error = 0;
    double energy_error = 0;
    double produced_hydrogen = 0;
    double captured_hydrogen = 0;
    double escaped_hydrogen = 0;
    double escaped_inert = 0;
    double escaped_enthalpy = 0;
    double net_thermal_energy = 0;
    double radiolysis_energy = 0;
    double raw_compatibility_residual = 0;        ///< Sum of raw phase-volume increments, m3.
    double compatibility_roundoff_correction = 0; ///< Reported numerical zero-mode correction, m3.
};

} // namespace SimpleFluid::VoF
