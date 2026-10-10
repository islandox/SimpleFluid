/** @file LumpedInnerCavityModel.hh
 *  @brief Lumped moist air above a liquid free surface and inside vessel walls.
 */
#pragma once

#include "SimpleFluidExport.hh"
#include "equations/ConvectionBoundary.hh"

#include <concepts>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace SimpleFluid
{

class InnerCavityFieldInput;

enum class CavityMoistureExchange
{
    Impermeable,
    LiquidSurface,   ///< Evaporation or condensation; liquid inventory belongs to the caller.
    CondensationOnly ///< Dry vessel wall; collected condensate cannot re-evaporate.
};

/** @brief One prescribed, isothermal surface; all quantities use SI units. */
struct InnerCavitySurface
{
    std::string name;
    real_t area = 1.0;             ///< m^2
    real_t temperature = 300;      ///< K
    ConvectionBoundary convection; ///< Constant h or vertical-plate natural convection; ambient is the cavity.
    /** Optional geometry-specific h(T_surface, T_air) [W/(m^2 K)]. Overrides convection. */
    std::function<real_t(real_t, real_t)> convection_coefficient;
    CavityMoistureExchange moisture_exchange = CavityMoistureExchange::Impermeable;
    real_t mass_transfer_coefficient = 0.0; ///< m/s; supplied gas-film transfer coefficient.
    real_t emissivity = 1.0;                ///< (0,1] when radiation is enabled.
};

namespace cavity_detail
{
/** Borrowed batch properties with per-surface numerical data. */
struct SurfaceSample
{
    const InnerCavitySurface* properties;
    real_t area, temperature;
};
} // namespace cavity_detail

struct LumpedInnerCavityOptions
{
    real_t volume = 1.0;                    ///< Fixed gas volume, m^3; excludes the liquid.
    real_t dry_air_mass = 1.0;              ///< Conserved dry-air inventory, kg.
    real_t dry_air_gas_constant = 287.05;   ///< J/(kg K)
    real_t vapor_gas_constant = 461.5;      ///< J/(kg K)
    real_t dry_air_cp = 1005.0;             ///< J/(kg K), constant caloric properties.
    real_t vapor_cp = 1860.0;               ///< J/(kg K)
    real_t liquid_cp = 4180.0;              ///< J/(kg K)
    real_t reference_temperature = 273.15;  ///< Enthalpy reference, K.
    real_t reference_latent_heat = 2.501e6; ///< h_v - h_l at reference_temperature, J/kg.
    real_t minimum_temperature = 273.15;    ///< Supported air-temperature bracket, K.
    real_t maximum_temperature = 373.15;    ///< K; also bounds accepted initial states.
    /** Required pure material callback: water saturation pressure [Pa] at temperature [K]. */
    std::function<real_t(real_t)> saturation_pressure;
};

struct LumpedInnerCavityState
{
    real_t temperature = 300.0;  ///< K
    real_t humidity_ratio = 0.0; ///< kg water vapor / kg dry air; not relative humidity.
};

/** @brief Compact numeric rates in input-surface order; positive leaves the surface. */
struct InnerCavitySurfaceExchange
{
    real_t heat_transfer_coefficient = 0.0; ///< W/(m^2 K)
    real_t vapor_mass_rate = 0.0;           ///< kg/s; negative for condensation.
    real_t sensible_heat_rate = 0.0;        ///< W, convection to air.
    real_t latent_heat_rate = 0.0;          ///< W, m_dot (h_v(Ts) - h_l(Ts)).
    real_t radiation_heat_rate = 0.0;       ///< W, net radiation to other surfaces, not air.
    real_t heat_rate = 0.0;                 ///< W, sensible + latent + radiation.
    real_t energy_rate = 0.0;               ///< W, heat_rate + m_dot h_l(Ts).
};

struct InnerCavityFaceExchange
{
    global_index_t face_global_id = 0;
    std::size_t batch_slot = 0; ///< Position in the mesh boundary batch.
    real_t area = 0.0;
    InnerCavitySurfaceExchange exchange;
};

/** Local owned-face transfers for one boundary batch, including on empty ranks. */
struct InnerCavityBoundaryExchange
{
    std::shared_ptr<const void> mesh;
    int batch_id = -1;
    std::vector<InnerCavityFaceExchange> faces;
};

struct LumpedInnerCavityExchange
{
    std::vector<InnerCavitySurfaceExchange> surfaces;
    real_t vapor_mass_rate = 0.0;                        ///< kg/s, sum over surfaces.
    real_t air_energy_rate = 0.0;                        ///< W, convection + transported vapor enthalpy.
    std::vector<InnerCavityBoundaryExchange> boundaries; ///< Populated by the field-batch API.
};

struct LumpedInnerCavityStep
{
    LumpedInnerCavityState state;
    LumpedInnerCavityExchange exchange;   ///< Backward-Euler rates; multiply by dt for transfers.
    real_t mass_balance_residual = 0.0;   ///< kg
    real_t energy_balance_residual = 0.0; ///< J
};

/**
 * @brief Fixed-volume ideal moist-air control volume with prescribed surfaces.
 *
 * Temperature and humidity are uniform; dry-air mass is conserved and pressure
 * follows the ideal-gas law. Radiation uses a transparent-gas, diffuse-gray
 * enclosure. A nonempty row-major view-factor matrix must close and obey area
 * reciprocity. An empty matrix explicitly disables radiation.
 *
 * advance() uses backward Euler for mass and internal energy, with h and film
 * transfer conductance frozen at the old air temperature. preview() is pure;
 * failed advances preserve state. Callers own liquid/condensate inventories,
 * surface thermal solves and coupled-step acceptance. Scalar-surface calls
 * leave MPI reductions to callers; field-batch calls handle them collectively. See
 * docs/modeling/lumped_inner_cavity.md for the conservation/coupling contract.
 */
class SIMPLEFLUID_EQUATIONS_EXPORT LumpedInnerCavityModel
{
public:
    LumpedInnerCavityModel(LumpedInnerCavityOptions options, LumpedInnerCavityState initial_state);

    const LumpedInnerCavityState& state() const noexcept { return d_state; }
    const LumpedInnerCavityOptions& options() const noexcept { return d_options; }
    void set_state(LumpedInnerCavityState state);

    real_t vapor_mass() const;
    real_t vapor_pressure() const;
    real_t pressure() const;
    real_t relative_humidity() const; ///< Fraction; may exceed one (no bulk fog model).
    real_t internal_energy() const;   ///< J, using the documented common enthalpy reference.

    LumpedInnerCavityExchange evaluate(
        std::span<const InnerCavitySurface> surfaces, std::span<const real_t> view_factors = {}) const;
    LumpedInnerCavityStep preview(real_t time_step, std::span<const InnerCavitySurface> surfaces,
        std::span<const real_t> view_factors = {}) const;
    LumpedInnerCavityStep advance(
        real_t time_step, std::span<const InnerCavitySurface> surfaces, std::span<const real_t> view_factors = {});

    // Definitions live in LumpedInnerCavityCoupling.hh. Constrained templates
    // preserve the original evaluate({}) / preview(dt, {}) overloads.
    template<class Batch>
        requires std::same_as<Batch, InnerCavityFieldInput>
    LumpedInnerCavityExchange evaluate(const Batch& batch, std::span<const real_t> view_factors = {}) const;
    template<class Batch>
        requires std::same_as<Batch, InnerCavityFieldInput>
    LumpedInnerCavityStep preview(
        real_t time_step, const Batch& batch, std::span<const real_t> view_factors = {}) const;
    template<class Batch>
        requires std::same_as<Batch, InnerCavityFieldInput>
    LumpedInnerCavityStep advance(real_t time_step, const Batch& batch, std::span<const real_t> view_factors = {});

private:
    friend class InnerCavityFieldInput;
    LumpedInnerCavityExchange evaluate_samples(
        std::span<const cavity_detail::SurfaceSample> surfaces, std::span<const real_t> view_factors) const;
    LumpedInnerCavityStep preview_samples(real_t time_step, std::span<const cavity_detail::SurfaceSample> surfaces,
        std::span<const real_t> view_factors) const;
    LumpedInnerCavityOptions d_options;
    LumpedInnerCavityState d_state;
};

} // namespace SimpleFluid
