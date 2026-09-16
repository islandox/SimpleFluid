/**
 * @file PlanarFreeSurfaceModel.hh
 * @brief Fixed-grid planar liquid-level and ideal-gas headspace models.
 */
#pragma once

#include "FVM/TransportSystem.hh"
#include "SimpleFluidExport.hh"
#include "dataclass/Database.hh"
#include "dataclass/TpetraTypes.hh"
#include "dataclass/typedefs.hh"
#include "fields/MeshFieldTraits.hh"
#include "geometry/Mesh.hh"
#include "geometry/MeshQuality.hh"
#include "solvers/BelosLinearSolver.hh"
#include "utils/CompensatedSum.hh"

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

/** Stable setup diagnostic for a solver that lacks mutable ALE ownership. */
inline constexpr std::string_view planar_ale_unavailable_diagnostic =
    "free_surface_model 'planarALE' requires the mutable native solver constructor and the supported conservative "
    "ALE model matrix.";

/** Compatibility alias retained for callers of the earlier diagnostic name. */
inline constexpr std::string_view planar_ale_immutable_mesh_diagnostic = planar_ale_unavailable_diagnostic;

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
    [[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL VesselRangeEvaluation boundAndEvaluate(
        real_t requested, real_t lower, real_t upper,
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
    [[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL real_t volumeBelowInRange(real_t height) const override;
    [[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL real_t areaAtInRange(real_t height) const override;
    [[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL real_t levelForVolumeInRange(real_t volume) const override;

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
    [[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL real_t volumeBelowInRange(real_t height) const override;
    [[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL real_t areaAtInRange(real_t height) const override;
    [[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL real_t levelForVolumeInRange(real_t volume) const override;

private:
    [[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL size_t heightSegment(real_t height) const;
    [[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL size_t volumeSegment(real_t volume) const;

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

/** Runtime liquid-mass inventory configuration. */
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

/** Solver-integrated planar-ALE controls; all dimensional values use SI units. */
struct FreeSurfaceALEOptions
{
    /** Name of the planar moving liquid/headspace boundary batch. */
    std::string top_boundary;
    /** Elevation below which mesh points remain fixed [m]. */
    std::optional<real_t> deformation_start_elevation;
    /** Maximum accepted-to-trial surface displacement [m]. */
    std::optional<real_t> maximum_level_change;
    real_t gcl_absolute_tolerance = 1.0e-12; ///< [m^3/s]
    real_t gcl_relative_tolerance = 1.0e-10;
    int maximum_correctors = 12;
    real_t level_absolute_tolerance = 1.0e-10; ///< [m]
    real_t level_relative_tolerance = 1.0e-10;
    real_t relaxation = 0.5;
    MeshQualityLimits quality_limits;
};

/** Complete fixed-grid or solver-integrated planar free-surface configuration. */
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
    FreeSurfaceALEOptions ale;
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
    /** Opaque accepted-state snapshot used by an outer solver transaction. */
    class StateSnapshot
    {
    public:
        StateSnapshot(StateSnapshot&&) noexcept = default;
        StateSnapshot& operator=(StateSnapshot&&) noexcept = default;
        StateSnapshot(const StateSnapshot&) = delete;
        StateSnapshot& operator=(const StateSnapshot&) = delete;
        ~StateSnapshot() = default;

    private:
        friend class PlanarFreeSurfaceModel;
        StateSnapshot() = default;

        const PlanarFreeSurfaceModel* d_owner = nullptr;
        std::unique_ptr<HeadspaceModel> d_headspace;
        bool d_initialized = false;
        real_t d_initial_pool_level = 0.0;
        GasMolesBySpecies d_initial_gas_moles;
        GasMolesBySpecies d_committed_escaped_moles;
        FreeSurfaceDiagnostics d_diagnostics;
        std::size_t d_update_generation = 0;
    };

    /** Opaque non-mutating closure result committed by generation. */
    class UpdatePreview
    {
    public:
        [[nodiscard]] const FreeSurfaceDiagnostics& diagnostics() const noexcept { return d_diagnostics; }
        [[nodiscard]] real_t headspacePressure() const noexcept { return d_headspace.pressure; }

    private:
        friend class PlanarFreeSurfaceModel;
        UpdatePreview(const PlanarFreeSurfaceModel* owner, std::size_t generation, HeadspaceState headspace,
            GasMolesBySpecies escaped_moles, FreeSurfaceDiagnostics diagnostics) noexcept
            : d_owner(owner), d_generation(generation), d_headspace(std::move(headspace)),
              d_escaped_moles(std::move(escaped_moles)), d_diagnostics(std::move(diagnostics))
        {
        }

        const PlanarFreeSurfaceModel* d_owner = nullptr;
        std::size_t d_generation = 0;
        HeadspaceState d_headspace;
        GasMolesBySpecies d_escaped_moles;
        FreeSurfaceDiagnostics d_diagnostics;
    };

    PlanarFreeSurfaceModel(std::shared_ptr<const VesselVolumeMap> volume_map, std::unique_ptr<HeadspaceModel> headspace,
        FreeSurfaceCouplingOptions coupling = {}, real_t validity_warning_relative_level_change = 0.05,
        real_t configured_level_underflow = 0.0, real_t configured_level_overflow = 0.0);

    void initialize(const FreeSurfaceUpdate& update);
    [[nodiscard]] UpdatePreview previewUpdate(const FreeSurfaceUpdate& update) const;
    void commitUpdate(const UpdatePreview& preview);
    void update(const FreeSurfaceUpdate& update);

    /** Capture every mutable accepted ledger and headspace state. */
    [[nodiscard]] StateSnapshot snapshot() const;
    /** Restore accepted state and monotonically invalidate all outstanding previews. */
    void restore(const StateSnapshot& snapshot);

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

    [[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL ClosureResult solveClosure(const FreeSurfaceUpdate& update) const;
    [[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL ClosureResult solveVented(const FreeSurfaceUpdate& update) const;
    [[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL ClosureResult solveClosed(const FreeSurfaceUpdate& update) const;
    [[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL ClosureResult evaluateClosedTrial(
        const FreeSurfaceUpdate& update, real_t pressure) const;
    [[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL FreeSurfaceDiagnostics makeDiagnostics(
        const ClosureResult& closure, const FreeSurfaceUpdate& update, bool initializing) const;
    SIMPLEFLUID_SOLVERS_LOCAL void validateUpdate(const FreeSurfaceUpdate& update, bool initializing) const;

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
    std::size_t d_update_generation = 0;
};

/** Construct the enabled planar-volume-budget model; disabled returns null. */
SIMPLEFLUID_SOLVERS_EXPORT std::unique_ptr<PlanarFreeSurfaceModel> make_planar_free_surface_model(
    const FreeSurfaceOptions& options);

/** Diagnostics for the selected liquid-material inventory. */
template<class Scalar> struct LiquidMassInventoryDiagnostics
{
    Scalar initial_mass = {};                  ///< [kg]
    Scalar total_mass = {};                    ///< [kg]
    Scalar cumulative_evaporated_mass = {};    ///< [kg]
    Scalar cumulative_condensed_mass = {};     ///< [kg]
    Scalar dryout_mass_deficit = {};           ///< [kg]
    Scalar mass_weighted_specific_volume = {}; ///< [m^3/kg]
    Scalar liquid_volume = {};                 ///< [m^3]
    Scalar step_mass_balance_residual = {};    ///< [kg]
    Scalar normalized_step_mass_balance_residual = {};
    Scalar mass_balance_residual = {}; ///< [kg]
    Scalar normalized_mass_balance_residual = {};
};

/**
 * MPI-consistent global or cellwise liquid-mass inventory.
 *
 * Fixed reference mass fractions are initialized from pure-liquid density
 * times cell volume. The global fallback therefore evaluates
 * `V_l=M_l sum(w_c/rho_l,c)` without claiming local conservative transport.
 * On a fixed mesh, cellwise mode stores `m_l_star` [kg/m^3 reference volume].
 * With an active ALE descriptor the same compatibility field is explicitly a
 * current-control-volume mass density whose conserved product is `V*m_l_star`;
 * accepted-old and trial-new volumes are never applied twice. Mixture density
 * and void fraction are intentionally absent from both paths.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes, class MeshType = Mesh<Pack>> class SIMPLEFLUID_SOLVERS_EXPORT LiquidMassInventory
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using mesh_type = MeshType;
    using field_type = typename MeshFieldTraits<Pack, mesh_type>::scalar_cell_type;
    using face_flux_field_type = typename MeshFieldTraits<Pack, mesh_type>::scalar_face_type;
    using diagnostics_type = LiquidMassInventoryDiagnostics<scalar_type>;

    /** Opaque accepted-state snapshot used by the ALE step transaction. */
    class StateSnapshot
    {
    private:
        friend class LiquidMassInventory;
        const LiquidMassInventory* d_owner = nullptr;
        std::vector<scalar_type> d_pure_density;
        std::vector<scalar_type> d_cell_mass;
        std::vector<scalar_type> d_trial_cell_mass;
        diagnostics_type d_diagnostics;
        size_t d_phase_change_generation = 0;
        size_t d_cellwise_trial_nonce = 0;
        bool d_initialized = false;
    };

    /** Opaque, single-generation phase-change transaction prepared collectively. */
    class PhaseChangePreview
    {
    public:
        [[nodiscard]] const diagnostics_type& diagnostics() const noexcept { return d_diagnostics; }
        [[nodiscard]] const std::optional<LinearSolveStatistics>& transportStatistics() const noexcept
        {
            return d_transport_statistics;
        }

    private:
        friend class LiquidMassInventory;
        PhaseChangePreview(const LiquidMassInventory* owner, size_t generation, diagnostics_type diagnostics,
            bool cellwise, size_t trial_nonce,
            std::optional<LinearSolveStatistics> transport_statistics = std::nullopt) noexcept
            : d_owner(owner), d_generation(generation), d_diagnostics(std::move(diagnostics)), d_cellwise(cellwise),
              d_trial_nonce(trial_nonce), d_transport_statistics(std::move(transport_statistics))
        {
        }

        const LiquidMassInventory* d_owner;
        size_t d_generation;
        diagnostics_type d_diagnostics;
        bool d_cellwise;
        size_t d_trial_nonce;
        std::optional<LinearSolveStatistics> d_transport_statistics;
    };

    LiquidMassInventory(SP<const mesh_type> mesh, LiquidMassInventoryOptions options = {})
        : d_mesh(checkedMesh(std::move(mesh))), d_options(std::move(options)),
          d_pure_liquid_density(d_mesh, "rhoLiquid", true), d_cell_mass_inventory(d_mesh, "liquidMassInventory", true),
          d_trial_cell_mass_inventory(d_mesh, "liquidMassInventoryTrial", true),
          d_unit_transport_weight(d_mesh, scalar_type{1}, "liquidMassUnitWeight"),
          d_zero_diffusivity(d_mesh, scalar_type{}, "liquidMassZeroDiffusivity")
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
        initialize_impl(initial_liquid_volume, [&](local_ordinal_type cell)
        {
            return pure_liquid_density(cell);
        });
    }

    /** Refresh pure density and the mass-weighted liquid material volume. */
    template<class DensityEvaluator> void updatePureLiquidDensity(DensityEvaluator&& pure_liquid_density)
    {
        update_pure_liquid_density_impl([&](local_ordinal_type cell)
        {
            return pure_liquid_density(cell);
        });
    }

    /**
     * Apply globally integrated, already accepted phase-change masses [kg].
     * Evaporation is removed exactly once; condensate is an explicit return.
     */
    [[nodiscard]] PhaseChangePreview previewPhaseChange(
        scalar_type evaporated_mass, scalar_type condensed_mass = scalar_type{}) const;

    /**
     * Conservatively preview one cellwise liquid-mass transport/source step.
     *
     * `m_l_star` and both optional phase-change rates use kg/m^3 of fixed
     * reference control volume and kg/(m^3 s), respectively. Physical
     * boundary flux is deliberately unsupported until an inlet composition
     * contract exists; all boundary face fluxes must therefore be zero.
     * Accepted state is unchanged until commitPhaseChange() receives the
     * returned nonce-bearing token.
     */
    [[nodiscard]] PhaseChangePreview previewCellwiseAdvance(scalar_type time_step,
        const face_flux_field_type& liquid_face_flux, const field_type* evaporation_mass_rate = nullptr,
        const field_type* condensation_mass_rate = nullptr, const LinearSolverOptions& linear_options = {},
        const FVM::ALEControlVolumeState* ale = nullptr);

    /** Commit a diagnostics value returned by previewPhaseChange(). */
    void commitPhaseChange(const PhaseChangePreview& preview);

    void updatePhaseChange(scalar_type evaporated_mass, scalar_type condensed_mass = scalar_type{})
    {
        const auto preview = previewPhaseChange(evaporated_mass, condensed_mass);
        commitPhaseChange(preview);
    }

    [[nodiscard]] bool initialized() const noexcept { return d_initialized; }

    [[nodiscard]] LiquidVolumeMode mode() const noexcept { return d_options.mode; }

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

    /** Fixed-reference or ALE current-control-volume mass density [kg/m^3]. */
    [[nodiscard]] const field_type& cellMassInventory() const noexcept { return d_cell_mass_inventory; }

    /** Discard numeric transport state after a geometry-epoch transition. */
    void refresh_geometry() { invalidateGeometry(); }

    /**
     * Trial current-control-volume mass density owned by a live cellwise preview.
     *
     * The returned field remains valid only until another cellwise preview,
     * commit, restore, density update, or reinitialization invalidates the token.
     */
    [[nodiscard]] const field_type& trialCellMassInventory(const PhaseChangePreview& preview) const;

    /** Capture all mutable accepted/trial inventory state without communication. */
    [[nodiscard]] StateSnapshot snapshot() const;

    /** Restore a snapshot and invalidate numeric transport state. */
    void restore(const StateSnapshot& snapshot);

    /** Invalidate geometry-dependent matrix and retained solver setup. */
    void invalidateGeometry()
    {
        d_transport_matrix = Teuchos::null;
        d_transport_solver = {};
    }

private:
    void initialize_impl(scalar_type initial_liquid_volume,
        const std::function<scalar_type(local_ordinal_type)>& pure_liquid_density);
    void update_pure_liquid_density_impl(
        const std::function<scalar_type(local_ordinal_type)>& pure_liquid_density);

    [[nodiscard]] static size_t invalidatedCounter(size_t current, size_t saved, std::string_view name)
    {
        const auto newest = std::max(current, saved);
        if (newest == std::numeric_limits<size_t>::max())
        {
            throw std::overflow_error("LiquidMassInventory " + std::string(name) + " exhausted.");
        }
        return newest + 1;
    }

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

    SIMPLEFLUID_SOLVERS_LOCAL
    void updateVolumeFromStoredDensity();

    [[nodiscard]] static scalar_type massClosureTolerance(scalar_type scale)
    {
        // This is a physical acceptance gate, not a linear-solver convergence
        // setting.  In particular, a caller cannot relax conservation by
        // requesting a loose Krylov tolerance.
        constexpr scalar_type maximum_normalized_mass_error = scalar_type{1.0e-10};
        const auto normalized =
            std::max(scalar_type{4096} * std::numeric_limits<scalar_type>::epsilon(), maximum_normalized_mass_error);
        return normalized * std::max(scalar_type{1}, scale);
    }

    SIMPLEFLUID_SOLVERS_LOCAL
    static void updateMassBalance(diagnostics_type& diagnostics);

    void updateMassBalance() { updateMassBalance(d_diagnostics); }

    void requireInitialized() const
    {
        if (!d_initialized)
        {
            throw std::logic_error("LiquidMassInventory must be initialized before use.");
        }
    }

    void updateGlobalCellMassField()
    {
        for (size_t owned = 0; owned < d_reference_mass_fraction.size(); ++owned)
        {
            const auto cell = static_cast<local_ordinal_type>(owned);
            const auto volume = static_cast<scalar_type>(d_mesh->cell_volume(cell));
            d_cell_mass_inventory.set_owned_value(
                cell, d_diagnostics.total_mass * d_reference_mass_fraction[owned] / volume);
        }
        d_cell_mass_inventory.sync_ghosts();
    }

    SP<const mesh_type> d_mesh;
    LiquidMassInventoryOptions d_options;
    field_type d_pure_liquid_density;
    field_type d_cell_mass_inventory;
    field_type d_trial_cell_mass_inventory;
    field_type d_unit_transport_weight;
    field_type d_zero_diffusivity;
    std::vector<scalar_type> d_reference_mass_fraction;
    std::vector<scalar_type> d_density_scratch;
    diagnostics_type d_diagnostics;
    BelosLinearSolver<Pack> d_transport_solver;
    Teuchos::RCP<typename Pack::matrix_type> d_transport_matrix = Teuchos::null;
    size_t d_phase_change_generation = 0;
    size_t d_cellwise_trial_nonce = 0;
    bool d_initialized = false;
};
extern template class LiquidMassInventory<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
extern template class LiquidMassInventory<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;


} // namespace SimpleFluid
