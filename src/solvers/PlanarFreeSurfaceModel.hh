/**
 * @file PlanarFreeSurfaceModel.hh
 * @brief Fixed-grid planar liquid-level and ideal-gas headspace models.
 */
#pragma once

#include "SimpleFluidExport.hh"
#include "dataclass/Database.hh"
#include "dataclass/TpetraTypes.hh"
#include "dataclass/typedefs.hh"
#include "fields/MeshFieldTraits.hh"
#include "geometry/Mesh.hh"

#include <Teuchos_CommHelpers.hpp>

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/** Stable setup diagnostic for the deliberately unsupported Milestone-B mode. */
inline constexpr std::string_view planar_ale_immutable_mesh_diagnostic =
    "free_surface_model 'planarALE' is unavailable: SimpleFluid meshes do not support fixed-topology geometry motion.";

/** Action taken when a requested height or volume is outside the vessel map. */
enum class FreeSurfaceRangePolicy
{
    Error,
    ClampAndReport
};

/** Runtime free-surface selection. */
enum class FreeSurfaceMode
{
    Fixed,
    PlanarVolumeBudget,
    PlanarALE
};

/** Vessel-profile representation. */
enum class VesselVolumeMapMode
{
    ConstantArea,
    Tabulated
};

/** Liquid-material volume representation. */
enum class LiquidVolumeMode
{
    GlobalConstantMass,
    CellMassInventory
};

/** Headspace boundary condition. */
enum class HeadspaceMode
{
    Vented,
    Closed
};

/** Policy used to obtain the lumped headspace temperature. */
enum class HeadspaceTemperatureMode
{
    Fixed,
    Prescribed,
    BulkLiquid
};

/**
 * Result of a vessel-map range check.
 *
 * `underflow` and `overflow` use the units of `requested`; the evaluated
 * `value` uses the units of the requested map result.
 */
struct VesselRangeEvaluation
{
    real_t requested = 0.0;
    real_t accepted = 0.0;
    real_t value = 0.0;
    real_t underflow = 0.0;
    real_t overflow = 0.0;

    [[nodiscard]] bool clamped() const noexcept { return underflow > 0.0 || overflow > 0.0; }
};

/** Monotone vessel volume/height relation used by the planar closure. */
class SIMPLEFLUID_SOLVERS_EXPORT VesselVolumeMap
{
public:
    explicit VesselVolumeMap(FreeSurfaceRangePolicy range_policy = FreeSurfaceRangePolicy::Error) noexcept;
    virtual ~VesselVolumeMap() = default;

    [[nodiscard]] virtual real_t bottomElevation() const noexcept = 0;
    [[nodiscard]] virtual real_t topElevation() const noexcept = 0;
    [[nodiscard]] virtual real_t totalUsableVolume() const noexcept = 0;

    /** Evaluate volume below a height and retain an explicit clamp report. */
    [[nodiscard]] VesselRangeEvaluation evaluateVolumeBelow(real_t height) const;

    /** Evaluate level for a volume and retain an explicit clamp report. */
    [[nodiscard]] VesselRangeEvaluation evaluateLevelForVolume(real_t volume) const;

    [[nodiscard]] real_t volumeBelow(real_t height) const;
    [[nodiscard]] real_t areaAt(real_t height) const;
    [[nodiscard]] real_t levelForVolume(real_t volume) const;
    [[nodiscard]] FreeSurfaceRangePolicy rangePolicy() const noexcept;

    /** Last scalar/evaluation range event; inspect this for clampAndReport. */
    [[nodiscard]] const VesselRangeEvaluation& lastRangeEvaluation() const noexcept;

protected:
    [[nodiscard]] virtual real_t volumeBelowInRange(real_t height) const = 0;
    [[nodiscard]] virtual real_t areaAtInRange(real_t height) const = 0;
    [[nodiscard]] virtual real_t levelForVolumeInRange(real_t volume) const = 0;

private:
    [[nodiscard]] VesselRangeEvaluation boundAndEvaluate(real_t requested, real_t lower, real_t upper,
        std::string_view quantity, const std::function<real_t(real_t)>& evaluator) const;

    FreeSurfaceRangePolicy d_range_policy;
    mutable VesselRangeEvaluation d_last_range_evaluation;
};

/** Analytic cylindrical/prismatic volume map. */
class SIMPLEFLUID_SOLVERS_EXPORT ConstantAreaVesselVolumeMap final : public VesselVolumeMap
{
public:
    ConstantAreaVesselVolumeMap(real_t bottom_elevation, real_t top_elevation, real_t cross_section_area,
        FreeSurfaceRangePolicy range_policy = FreeSurfaceRangePolicy::Error);

    [[nodiscard]] real_t bottomElevation() const noexcept override;
    [[nodiscard]] real_t topElevation() const noexcept override;
    [[nodiscard]] real_t totalUsableVolume() const noexcept override;

protected:
    [[nodiscard]] real_t volumeBelowInRange(real_t height) const override;
    [[nodiscard]] real_t areaAtInRange(real_t height) const override;
    [[nodiscard]] real_t levelForVolumeInRange(real_t volume) const override;

private:
    real_t d_bottom_elevation;
    real_t d_top_elevation;
    real_t d_cross_section_area;
};

/** Piecewise-linear monotone vessel map for general profiles. */
class SIMPLEFLUID_SOLVERS_EXPORT TabulatedVesselVolumeMap final : public VesselVolumeMap
{
public:
    TabulatedVesselVolumeMap(
        ArrReal heights, ArrReal volumes, FreeSurfaceRangePolicy range_policy = FreeSurfaceRangePolicy::Error);

    [[nodiscard]] real_t bottomElevation() const noexcept override;
    [[nodiscard]] real_t topElevation() const noexcept override;
    [[nodiscard]] real_t totalUsableVolume() const noexcept override;

protected:
    [[nodiscard]] real_t volumeBelowInRange(real_t height) const override;
    [[nodiscard]] real_t areaAtInRange(real_t height) const override;
    [[nodiscard]] real_t levelForVolumeInRange(real_t volume) const override;

private:
    [[nodiscard]] size_t heightSegment(real_t height) const;
    [[nodiscard]] size_t volumeSegment(real_t volume) const;

    ArrReal d_heights;
    ArrReal d_volumes;
};

/** Vessel configuration; all dimensional values use SI units. */
struct FreeSurfaceVesselOptions
{
    VesselVolumeMapMode mode = VesselVolumeMapMode::ConstantArea;
    real_t bottom_elevation = 0.0;                                           ///< [m]
    real_t top_elevation = 0.0;                                              ///< [m]
    real_t cross_section_area = 0.0;                                         ///< [m^2]
    real_t total_internal_volume = std::numeric_limits<real_t>::quiet_NaN(); ///< [m^3]
    ArrReal height_table;                                                    ///< [m]
    ArrReal volume_table;                                                    ///< [m^3]
};

/** Global-constant-mass fallback configuration. */
struct LiquidMassInventoryOptions
{
    LiquidVolumeMode mode = LiquidVolumeMode::GlobalConstantMass;
    std::optional<real_t> initial_liquid_mass; ///< [kg]
    FreeSurfaceRangePolicy depletion_policy = FreeSurfaceRangePolicy::Error;
};

/** Lumped ideal-gas headspace configuration in SI units. */
struct HeadspaceOptions
{
    HeadspaceMode mode = HeadspaceMode::Vented;
    HeadspaceTemperatureMode temperature_mode = HeadspaceTemperatureMode::Fixed;
    real_t ambient_pressure = 101325.0;     ///< [Pa absolute]
    real_t initial_pressure = 101325.0;     ///< [Pa absolute]
    real_t initial_temperature = 293.15;    ///< [K]
    real_t gas_constant = 8.31446261815324; ///< [J/(mol K)]
    real_t compressibility_factor = 1.0;
    real_t total_internal_volume = std::numeric_limits<real_t>::quiet_NaN(); ///< [m^3]
    std::map<std::string, real_t> initial_moles;                             ///< [mol/species]
    bool infer_initial_moles = true;
    ArrReal prescribed_temperature_times;  ///< [s]
    ArrReal prescribed_temperature_values; ///< [K]
};

/** Safeguards for the closed pressure/level solve. */
struct FreeSurfaceCouplingOptions
{
    int maximum_correctors = 100;
    real_t absolute_tolerance = 1.0e-8; ///< Pressure residual [Pa].
    real_t relative_tolerance = 1.0e-10;
    real_t relaxation = 1.0;
    real_t minimum_absolute_pressure = 1.0;     ///< [Pa absolute]
    real_t maximum_absolute_pressure = 1.0e9;   ///< [Pa absolute]
    real_t volume_absolute_tolerance = 1.0e-12; ///< [m^3]
    real_t volume_relative_tolerance = 1.0e-10;
    real_t gas_absolute_tolerance = 1.0e-12; ///< [mol/species]
    real_t gas_relative_tolerance = 1.0e-10;
};

/** Complete fixed-grid free-surface configuration. */
struct FreeSurfaceOptions
{
    bool enabled = false;
    FreeSurfaceMode mode = FreeSurfaceMode::Fixed;
    Dimension gravity_axis = Dimension::Z;
    real_t validity_warning_relative_level_change = 0.05;
    FreeSurfaceRangePolicy range_policy = FreeSurfaceRangePolicy::Error;
    std::optional<real_t> initial_liquid_volume; ///< [m^3]
    std::optional<real_t> initial_clear_level;   ///< [m]
    FreeSurfaceVesselOptions vessel;
    LiquidMassInventoryOptions liquid_mass;
    HeadspaceOptions headspace;
    FreeSurfaceCouplingOptions coupling;
};

/** Parse and validate flat `free_surface_*` Database keys. */
SIMPLEFLUID_SOLVERS_EXPORT FreeSurfaceOptions free_surface_options_from_database(const Database& database);

/** Validate programmatically assembled free-surface options. */
SIMPLEFLUID_SOLVERS_EXPORT void validate_free_surface_options(const FreeSurfaceOptions& options);

/** Construct the configured volume map. */
SIMPLEFLUID_SOLVERS_EXPORT std::shared_ptr<const VesselVolumeMap> make_vessel_volume_map(
    const FreeSurfaceOptions& options);

/** Return configured initial liquid volume, converting clear level if set. */
SIMPLEFLUID_SOLVERS_EXPORT std::optional<real_t> configured_initial_liquid_volume(const FreeSurfaceOptions& options);

/** Linearly interpolate a validated prescribed headspace history [K]. */
SIMPLEFLUID_SOLVERS_EXPORT real_t prescribed_headspace_temperature(const HeadspaceOptions& options, real_t time);

using GasMolesBySpecies = std::map<std::string, real_t>;
using GasMolesByPopulation = std::map<std::string, GasMolesBySpecies>;

/** Prospective or accepted lumped headspace state. */
struct HeadspaceState
{
    real_t pressure = 0.0;    ///< [Pa absolute]
    real_t volume = 0.0;      ///< [m^3]
    real_t temperature = 0.0; ///< [K]
    real_t total_moles = 0.0; ///< [mol]
};

/** Runtime-selectable owner of escaped gas and headspace thermodynamics. */
class SIMPLEFLUID_SOLVERS_EXPORT HeadspaceModel
{
public:
    explicit HeadspaceModel(HeadspaceOptions options);
    virtual ~HeadspaceModel() = default;

    [[nodiscard]] virtual std::unique_ptr<HeadspaceModel> clone() const = 0;
    [[nodiscard]] virtual HeadspaceMode mode() const noexcept = 0;
    virtual void initialize(real_t pool_volume, real_t supplied_temperature) = 0;
    [[nodiscard]] virtual HeadspaceState trialState(
        real_t pool_volume, real_t supplied_temperature, const GasMolesBySpecies& escaped_moles) const = 0;
    virtual void commit(const HeadspaceState& state, const GasMolesBySpecies& escaped_moles) = 0;

    [[nodiscard]] const HeadspaceOptions& options() const noexcept;
    [[nodiscard]] const HeadspaceState& state() const noexcept;
    [[nodiscard]] virtual const GasMolesBySpecies& headspaceMoles() const noexcept = 0;
    [[nodiscard]] virtual const GasMolesBySpecies& ventedMoles() const noexcept = 0;

protected:
    [[nodiscard]] real_t temperature(real_t supplied_temperature) const;
    [[nodiscard]] real_t availableVolume(real_t pool_volume) const;
    void setState(HeadspaceState state) noexcept;

    HeadspaceOptions d_options;
    HeadspaceState d_state;
};

/** Constant-pressure vent whose escaped species inventory remains auditable. */
class SIMPLEFLUID_SOLVERS_EXPORT VentedHeadspaceModel final : public HeadspaceModel
{
public:
    explicit VentedHeadspaceModel(HeadspaceOptions options);
    [[nodiscard]] std::unique_ptr<HeadspaceModel> clone() const override;
    [[nodiscard]] HeadspaceMode mode() const noexcept override;
    void initialize(real_t pool_volume, real_t supplied_temperature) override;
    [[nodiscard]] HeadspaceState trialState(
        real_t pool_volume, real_t supplied_temperature, const GasMolesBySpecies& escaped_moles) const override;
    void commit(const HeadspaceState& state, const GasMolesBySpecies& escaped_moles) override;
    [[nodiscard]] const GasMolesBySpecies& headspaceMoles() const noexcept override;
    [[nodiscard]] const GasMolesBySpecies& ventedMoles() const noexcept override;

private:
    GasMolesBySpecies d_empty_headspace;
    GasMolesBySpecies d_vented_moles;
};

/** Closed lumped ideal-gas headspace. */
class SIMPLEFLUID_SOLVERS_EXPORT ClosedIdealGasHeadspaceModel final : public HeadspaceModel
{
public:
    explicit ClosedIdealGasHeadspaceModel(HeadspaceOptions options);
    [[nodiscard]] std::unique_ptr<HeadspaceModel> clone() const override;
    [[nodiscard]] HeadspaceMode mode() const noexcept override;
    void initialize(real_t pool_volume, real_t supplied_temperature) override;
    [[nodiscard]] HeadspaceState trialState(
        real_t pool_volume, real_t supplied_temperature, const GasMolesBySpecies& escaped_moles) const override;
    void commit(const HeadspaceState& state, const GasMolesBySpecies& escaped_moles) override;
    [[nodiscard]] const GasMolesBySpecies& headspaceMoles() const noexcept override;
    [[nodiscard]] const GasMolesBySpecies& ventedMoles() const noexcept override;

private:
    GasMolesBySpecies d_headspace_moles;
    GasMolesBySpecies d_empty_vent;
};

/** Construct a vented or closed headspace from validated options. */
SIMPLEFLUID_SOLVERS_EXPORT std::unique_ptr<HeadspaceModel> make_headspace_model(const FreeSurfaceOptions& options);

/** Externally owned current gas inventories and exact escaped transfer. */
struct FreeSurfaceGasInventory
{
    GasMolesBySpecies initial_moles;
    GasMolesBySpecies generated_moles; ///< Cumulative generated gas [mol].
    GasMolesBySpecies dissolved_moles;
    GasMolesBySpecies submerged_moles;
    GasMolesByPopulation submerged_population_moles;
    /** Exact per-step submerged decrement, transferred once [mol]. */
    GasMolesBySpecies escaped_moles_this_step;
    GasMolesBySpecies other_sink_moles; ///< Cumulative documented sinks [mol].
};

/** Immutable-by-value state snapshot for reporting and tests. */
struct FreeSurfaceDiagnostics
{
    real_t time = 0.0;                       ///< Accepted physical time [s]
    real_t time_step = 0.0;                  ///< Accepted step size [s]
    real_t liquid_volume = 0.0;              ///< [m^3]
    real_t submerged_bubble_volume = 0.0;    ///< [m^3]
    real_t pool_volume = 0.0;                ///< [m^3]
    real_t old_clear_level = 0.0;            ///< [m]
    real_t clear_level = 0.0;                ///< [m]
    real_t old_pool_level = 0.0;             ///< [m]
    real_t pool_level = 0.0;                 ///< [m]
    real_t clear_level_rate = 0.0;           ///< [m/s]
    real_t pool_level_rate = 0.0;            ///< [m/s]
    real_t surface_area = 0.0;               ///< [m^2]
    real_t overflow_volume = 0.0;            ///< [m^3]
    real_t dryout_deficit = 0.0;             ///< [m^3]
    real_t configured_level_underflow = 0.0; ///< [m]
    real_t configured_level_overflow = 0.0;  ///< [m]
    real_t volume_closure_residual = 0.0;    ///< [m^3]
    real_t normalized_volume_closure_residual = 0.0;
    HeadspaceState headspace;
    int nonlinear_iterations = 0;
    real_t nonlinear_residual = 0.0; ///< [Pa]
    real_t relative_level_change = 0.0;
    bool validity_warning = false;
    GasMolesBySpecies generated_gas_moles;
    GasMolesBySpecies dissolved_gas_moles;
    GasMolesBySpecies submerged_gas_moles;
    GasMolesByPopulation submerged_population_gas_moles;
    GasMolesBySpecies escaped_gas_moles_this_step;
    GasMolesBySpecies other_sink_gas_moles;
    GasMolesBySpecies headspace_gas_moles;
    GasMolesBySpecies vented_gas_moles;
    GasMolesBySpecies gas_closure_by_species;
    GasMolesBySpecies normalized_gas_closure_by_species;
    real_t gas_closure_residual = 0.0; ///< [mol]
    real_t normalized_gas_closure_residual = 0.0;
};

/** Callback inputs for one accepted free-surface inventory update. */
struct FreeSurfaceUpdate
{
    using VolumeAtPressure = std::function<real_t(real_t)>;

    VolumeAtPressure liquid_volume_at_pressure;
    VolumeAtPressure bubble_volume_at_pressure;
    FreeSurfaceGasInventory gas;
    /** Lowest absolute pressure for which every coupled callback is valid. */
    real_t minimum_valid_absolute_pressure = 0.0; ///< [Pa absolute]
    /** Volume-equivalent unmet liquid demand supplied by the inventory. */
    real_t liquid_volume_deficit = 0.0; ///< [m^3]
    real_t headspace_temperature = std::numeric_limits<real_t>::quiet_NaN();
    real_t time = 0.0;      ///< Accepted physical time [s]
    real_t time_step = 0.0; ///< Accepted step size [s]
};

/**
 * Fixed-grid planar inventory closure.
 *
 * This is deliberately a weak geometry-feedback approximation: it reports a
 * planar level but does not mask cells, move the mesh, or modify transport.
 */
class SIMPLEFLUID_SOLVERS_EXPORT PlanarFreeSurfaceModel
{
public:
    PlanarFreeSurfaceModel(std::shared_ptr<const VesselVolumeMap> volume_map, std::unique_ptr<HeadspaceModel> headspace,
        FreeSurfaceCouplingOptions coupling = {}, real_t validity_warning_relative_level_change = 0.05,
        real_t configured_level_underflow = 0.0, real_t configured_level_overflow = 0.0);

    void initialize(const FreeSurfaceUpdate& update);
    void update(const FreeSurfaceUpdate& update);

    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] FreeSurfaceDiagnostics diagnostics() const;
    [[nodiscard]] real_t headspacePressure() const noexcept;
    [[nodiscard]] const VesselVolumeMap& volumeMap() const noexcept;
    [[nodiscard]] const HeadspaceModel& headspace() const noexcept;
    /** Cumulative exact gas transfer committed through this model [mol/species]. */
    [[nodiscard]] const GasMolesBySpecies& committedEscapedMoles() const noexcept;
    [[nodiscard]] static constexpr std::string_view approximationLabel() noexcept
    {
        return "fixed-grid planar volume budget (weak geometry-feedback approximation)";
    }

private:
    struct ClosureResult
    {
        real_t liquid_volume = 0.0;
        real_t bubble_volume = 0.0;
        real_t dryout_deficit = 0.0;
        HeadspaceState headspace;
        int iterations = 0;
        real_t residual = 0.0;
    };

    [[nodiscard]] ClosureResult solveClosure(const FreeSurfaceUpdate& update) const;
    [[nodiscard]] ClosureResult solveVented(const FreeSurfaceUpdate& update) const;
    [[nodiscard]] ClosureResult solveClosed(const FreeSurfaceUpdate& update) const;
    [[nodiscard]] ClosureResult evaluateClosedTrial(const FreeSurfaceUpdate& update, real_t pressure) const;
    [[nodiscard]] FreeSurfaceDiagnostics makeDiagnostics(
        const ClosureResult& closure, const FreeSurfaceUpdate& update, bool initializing) const;
    void validateUpdate(const FreeSurfaceUpdate& update, bool initializing) const;

    std::shared_ptr<const VesselVolumeMap> d_volume_map;
    std::unique_ptr<HeadspaceModel> d_headspace;
    FreeSurfaceCouplingOptions d_coupling;
    real_t d_validity_warning_relative_level_change;
    real_t d_configured_level_underflow;
    real_t d_configured_level_overflow;
    bool d_initialized = false;
    real_t d_initial_pool_level = 0.0;
    GasMolesBySpecies d_initial_gas_moles;
    GasMolesBySpecies d_committed_escaped_moles;
    FreeSurfaceDiagnostics d_diagnostics;
};

/** Construct the enabled planar-volume-budget model; disabled returns null. */
SIMPLEFLUID_SOLVERS_EXPORT std::unique_ptr<PlanarFreeSurfaceModel> make_planar_free_surface_model(
    const FreeSurfaceOptions& options);

/** Diagnostics for the globalConstantMass liquid-material approximation. */
template<class Scalar> struct LiquidMassInventoryDiagnostics
{
    Scalar initial_mass = {};                  ///< [kg]
    Scalar total_mass = {};                    ///< [kg]
    Scalar cumulative_evaporated_mass = {};    ///< [kg]
    Scalar cumulative_condensed_mass = {};     ///< [kg]
    Scalar dryout_mass_deficit = {};           ///< [kg]
    Scalar mass_weighted_specific_volume = {}; ///< [m^3/kg]
    Scalar liquid_volume = {};                 ///< [m^3]
    Scalar mass_balance_residual = {};         ///< [kg]
    Scalar normalized_mass_balance_residual = {};
};

/**
 * MPI-consistent globalConstantMass liquid inventory.
 *
 * Fixed reference mass fractions are initialized from pure-liquid density
 * times cell volume. The fallback therefore evaluates
 * `V_l=M_l sum(w_c/rho_l,c)` without claiming local conservative transport.
 * Mixture density and void fraction are intentionally absent from this API.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes, class MeshType = Mesh<Pack>> class LiquidMassInventory
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using mesh_type = MeshType;
    using field_type = typename MeshFieldTraits<Pack, mesh_type>::scalar_cell_type;
    using diagnostics_type = LiquidMassInventoryDiagnostics<scalar_type>;

    /** Opaque, single-generation phase-change transaction prepared collectively. */
    class PhaseChangePreview
    {
    public:
        [[nodiscard]] const diagnostics_type& diagnostics() const noexcept { return d_diagnostics; }

    private:
        friend class LiquidMassInventory;
        PhaseChangePreview(const LiquidMassInventory* owner, size_t generation, diagnostics_type diagnostics) noexcept
            : d_owner(owner), d_generation(generation), d_diagnostics(std::move(diagnostics))
        {
        }

        const LiquidMassInventory* d_owner;
        size_t d_generation;
        diagnostics_type d_diagnostics;
    };

    LiquidMassInventory(SP<const mesh_type> mesh, LiquidMassInventoryOptions options = {})
        : d_mesh(checkedMesh(std::move(mesh))), d_options(std::move(options)),
          d_pure_liquid_density(d_mesh, "rhoLiquid", true)
    {
    }

    /**
     * Initialize mass fractions and mass from pure density [kg/m^3].
     *
     * If no mass is configured, `initial_liquid_volume` [m^3] is multiplied
     * by the volume-weighted pure-liquid density over owned cells.
     */
    template<class DensityEvaluator>
    void initialize(scalar_type initial_liquid_volume, DensityEvaluator&& pure_liquid_density)
    {
        const int local_initialized = d_initialized ? 1 : 0;
        int minimum_initialized = 0;
        int maximum_initialized = 0;
        const auto communicator = d_mesh->owned_cell_map()->getComm();
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &local_initialized, &minimum_initialized);
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_initialized, &maximum_initialized);
        if (minimum_initialized != maximum_initialized)
        {
            throw std::logic_error("LiquidMassInventory initialized state must match on every rank.");
        }
        if (maximum_initialized != 0)
        {
            throw std::logic_error("LiquidMassInventory is already initialized.");
        }

        const int local_invalid_volume =
            !std::isfinite(initial_liquid_volume) || initial_liquid_volume < scalar_type{} ? 1 : 0;
        int any_invalid_volume = 0;
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_invalid_volume, &any_invalid_volume);
        if (any_invalid_volume != 0)
        {
            throw std::invalid_argument("Initial liquid volume must be finite and non-negative.");
        }
        requireReplicated(initial_liquid_volume, "initial liquid volume");
        const int local_invalid_options =
            d_options.mode != LiquidVolumeMode::GlobalConstantMass ||
                    (d_options.initial_liquid_mass && (!std::isfinite(*d_options.initial_liquid_mass) ||
                                                          *d_options.initial_liquid_mass < scalar_type{}))
                ? 1
                : 0;
        int any_invalid_options = 0;
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_invalid_options, &any_invalid_options);
        if (any_invalid_options != 0)
        {
            throw std::invalid_argument(
                "LiquidMassInventory requires globalConstantMass mode and a finite non-negative initial mass.");
        }
        const std::array<scalar_type, 4> local_options{static_cast<scalar_type>(d_options.mode),
            static_cast<scalar_type>(d_options.depletion_policy),
            d_options.initial_liquid_mass ? scalar_type{1} : scalar_type{},
            static_cast<scalar_type>(d_options.initial_liquid_mass.value_or(real_t{}))};
        std::array<scalar_type, 4> minimum_options{};
        std::array<scalar_type, 4> maximum_options{};
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, static_cast<int>(local_options.size()),
            local_options.data(), minimum_options.data());
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, static_cast<int>(local_options.size()),
            local_options.data(), maximum_options.data());
        if (minimum_options != maximum_options)
        {
            throw std::invalid_argument("LiquidMassInventory options must match on every rank.");
        }

        const auto count = d_mesh->num_owned_cells();
        d_density_scratch.resize(count);
        int local_invalid = 0;
        for (size_t owned = 0; owned < count; ++owned)
        {
            const auto cell = static_cast<local_ordinal_type>(owned);
            try
            {
                d_density_scratch[owned] = pure_liquid_density(cell);
            }
            catch (...)
            {
                local_invalid = 1;
                d_density_scratch[owned] = scalar_type{};
            }
            if (!std::isfinite(d_density_scratch[owned]) || d_density_scratch[owned] <= scalar_type{})
            {
                local_invalid = 1;
            }
        }
        requireCollectivelyValidDensity(local_invalid);

        scalar_type local_reference_mass = {};
        scalar_type local_mesh_volume = {};
        for (size_t owned = 0; owned < count; ++owned)
        {
            const auto cell = static_cast<local_ordinal_type>(owned);
            const auto cell_volume = static_cast<scalar_type>(d_mesh->cell_volume(cell));
            if (!std::isfinite(cell_volume) || cell_volume <= scalar_type{})
            {
                local_invalid = 1;
                continue;
            }
            local_reference_mass += d_density_scratch[owned] * cell_volume;
            local_mesh_volume += cell_volume;
        }
        requireCollectivelyValidDensity(local_invalid);
        const auto reference_mass = globalSum(local_reference_mass);
        const auto mesh_volume = globalSum(local_mesh_volume);
        if (!(reference_mass > scalar_type{}) || !(mesh_volume > scalar_type{}))
        {
            throw std::invalid_argument("LiquidMassInventory requires positive global mesh volume and reference mass.");
        }

        d_reference_mass_fraction.resize(count);
        for (size_t owned = 0; owned < count; ++owned)
        {
            const auto cell = static_cast<local_ordinal_type>(owned);
            d_reference_mass_fraction[owned] =
                d_density_scratch[owned] * static_cast<scalar_type>(d_mesh->cell_volume(cell)) / reference_mass;
            d_pure_liquid_density.set_value(cell, d_density_scratch[owned]);
        }
        d_pure_liquid_density.sync_ghosts();

        d_diagnostics.initial_mass = d_options.initial_liquid_mass
                                         ? static_cast<scalar_type>(*d_options.initial_liquid_mass)
                                         : initial_liquid_volume * reference_mass / mesh_volume;
        d_diagnostics.total_mass = d_diagnostics.initial_mass;
        d_initialized = true;
        d_phase_change_generation = 0;
        updateVolumeFromStoredDensity();
        updateMassBalance();
    }

    /** Refresh pure density and the mass-weighted liquid material volume. */
    template<class DensityEvaluator> void updatePureLiquidDensity(DensityEvaluator&& pure_liquid_density)
    {
        requireInitialized();
        const auto count = d_mesh->num_owned_cells();
        if (d_density_scratch.size() != count)
        {
            throw std::logic_error("LiquidMassInventory mesh ownership changed after initialization.");
        }
        int local_invalid = 0;
        for (size_t owned = 0; owned < count; ++owned)
        {
            const auto cell = static_cast<local_ordinal_type>(owned);
            try
            {
                d_density_scratch[owned] = pure_liquid_density(cell);
            }
            catch (...)
            {
                local_invalid = 1;
                d_density_scratch[owned] = scalar_type{};
            }
            if (!std::isfinite(d_density_scratch[owned]) || d_density_scratch[owned] <= scalar_type{})
            {
                local_invalid = 1;
            }
        }
        requireCollectivelyValidDensity(local_invalid);
        for (size_t owned = 0; owned < count; ++owned)
        {
            d_pure_liquid_density.set_value(static_cast<local_ordinal_type>(owned), d_density_scratch[owned]);
        }
        d_pure_liquid_density.sync_ghosts();
        updateVolumeFromStoredDensity();
    }

    /**
     * Apply globally integrated, already accepted phase-change masses [kg].
     * Evaporation is removed exactly once; condensate is an explicit return.
     */
    [[nodiscard]] PhaseChangePreview previewPhaseChange(
        scalar_type evaporated_mass, scalar_type condensed_mass = scalar_type{}) const
    {
        requireInitialized();
        const int local_invalid = !std::isfinite(evaporated_mass) || evaporated_mass < scalar_type{} ||
                                          !std::isfinite(condensed_mass) || condensed_mass < scalar_type{}
                                      ? 1
                                      : 0;
        int any_invalid = 0;
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_invalid, &any_invalid);
        if (any_invalid != 0)
        {
            throw std::invalid_argument("Phase-change masses must be finite and non-negative.");
        }
        requireReplicated(evaporated_mass, "evaporated mass");
        requireReplicated(condensed_mass, "condensed mass");
        requireReplicated(static_cast<scalar_type>(d_phase_change_generation), "phase-change generation");

        auto preview = d_diagnostics;
        const auto available = preview.total_mass + condensed_mass;
        auto accepted_evaporation = evaporated_mass;
        auto deficit = scalar_type{};
        if (evaporated_mass > available)
        {
            deficit = evaporated_mass - available;
            if (d_options.depletion_policy == FreeSurfaceRangePolicy::Error)
            {
                throw std::out_of_range("Liquid evaporation exceeds the available liquid mass.");
            }
            accepted_evaporation = available;
        }

        preview.total_mass = available - accepted_evaporation;
        preview.cumulative_evaporated_mass += accepted_evaporation;
        preview.cumulative_condensed_mass += condensed_mass;
        preview.dryout_mass_deficit += deficit;
        preview.liquid_volume = preview.total_mass * preview.mass_weighted_specific_volume;
        updateMassBalance(preview);
        return PhaseChangePreview(this, d_phase_change_generation, std::move(preview));
    }

    /** Commit a diagnostics value returned by previewPhaseChange(). */
    void commitPhaseChange(const PhaseChangePreview& preview)
    {
        const int local_invalid = preview.d_owner != this || preview.d_generation != d_phase_change_generation ? 1 : 0;
        int any_invalid = 0;
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_invalid, &any_invalid);
        if (any_invalid != 0)
        {
            throw std::logic_error(
                "LiquidMassInventory phase-change preview is stale, foreign, or inconsistent across ranks.");
        }
        d_diagnostics = preview.d_diagnostics;
        ++d_phase_change_generation;
    }

    void updatePhaseChange(scalar_type evaporated_mass, scalar_type condensed_mass = scalar_type{})
    {
        const auto preview = previewPhaseChange(evaporated_mass, condensed_mass);
        commitPhaseChange(preview);
    }

    [[nodiscard]] bool initialized() const noexcept { return d_initialized; }

    [[nodiscard]] diagnostics_type diagnostics() const
    {
        requireInitialized();
        return d_diagnostics;
    }

    [[nodiscard]] scalar_type totalMass() const { return diagnostics().total_mass; }

    [[nodiscard]] scalar_type liquidVolume() const { return diagnostics().liquid_volume; }

    [[nodiscard]] const field_type& pureLiquidDensity() const noexcept { return d_pure_liquid_density; }

    /** Canonical pure-liquid material-density output field [kg/m^3]. */
    [[nodiscard]] const field_type& rhoLiquid() const noexcept { return d_pure_liquid_density; }

private:
    [[nodiscard]] static SP<const mesh_type> checkedMesh(SP<const mesh_type> mesh)
    {
        if (!mesh)
        {
            throw std::invalid_argument("LiquidMassInventory requires a non-null mesh.");
        }
        return mesh;
    }

    [[nodiscard]] scalar_type globalSum(scalar_type local_value) const
    {
        scalar_type global_value = {};
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_value, &global_value);
        return global_value;
    }

    void requireCollectivelyValidDensity(int local_invalid) const
    {
        int any_invalid = 0;
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_invalid, &any_invalid);
        if (any_invalid != 0)
        {
            throw std::invalid_argument(
                "Pure-liquid density and cell volume must be finite and positive on every rank.");
        }
    }

    void requireReplicated(scalar_type value, std::string_view quantity) const
    {
        scalar_type minimum = {};
        scalar_type maximum = {};
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &value, &minimum);
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &value, &maximum);
        if (minimum != maximum)
        {
            throw std::invalid_argument(
                "LiquidMassInventory global " + std::string(quantity) + " must be identical on every rank.");
        }
    }

    void updateVolumeFromStoredDensity()
    {
        scalar_type local_specific_volume = {};
        for (size_t owned = 0; owned < d_reference_mass_fraction.size(); ++owned)
        {
            const auto density = d_pure_liquid_density.value(static_cast<local_ordinal_type>(owned));
            local_specific_volume += d_reference_mass_fraction[owned] / density;
        }
        d_diagnostics.mass_weighted_specific_volume = globalSum(local_specific_volume);
        d_diagnostics.liquid_volume = d_diagnostics.total_mass * d_diagnostics.mass_weighted_specific_volume;
    }

    static void updateMassBalance(diagnostics_type& diagnostics)
    {
        diagnostics.mass_balance_residual = diagnostics.initial_mass + diagnostics.cumulative_condensed_mass -
                                            diagnostics.cumulative_evaporated_mass - diagnostics.total_mass;
        const auto scale =
            std::max(scalar_type{1}, std::abs(diagnostics.initial_mass) + diagnostics.cumulative_condensed_mass);
        diagnostics.normalized_mass_balance_residual = diagnostics.mass_balance_residual / scale;
        const auto tolerance = scalar_type{64} * std::numeric_limits<scalar_type>::epsilon() * scale;
        if (std::abs(diagnostics.mass_balance_residual) > tolerance)
        {
            throw std::logic_error("LiquidMassInventory mass closure exceeded roundoff tolerance.");
        }
    }

    void updateMassBalance() { updateMassBalance(d_diagnostics); }

    void requireInitialized() const
    {
        if (!d_initialized)
        {
            throw std::logic_error("LiquidMassInventory must be initialized before use.");
        }
    }

    SP<const mesh_type> d_mesh;
    LiquidMassInventoryOptions d_options;
    field_type d_pure_liquid_density;
    std::vector<scalar_type> d_reference_mass_fraction;
    std::vector<scalar_type> d_density_scratch;
    diagnostics_type d_diagnostics;
    size_t d_phase_change_generation = 0;
    bool d_initialized = false;
};

} // namespace SimpleFluid
