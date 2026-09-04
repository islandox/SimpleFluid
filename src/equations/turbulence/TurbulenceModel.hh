/**
 * @file TurbulenceModel.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Runtime ownership and transport coupling for two-equation RANS models.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "SimpleFluidExport.hh"
#include "FVM/CellGradientScheme.hh"
#include "FVM/FaceFlux.hh"
#include "dataclass/Database.hh"
#include "dataclass/vec3.hh"
#include "equations/BoundaryConditions.hh"
#include "equations/BoussinesqModel.hh"
#include "equations/turbulence/TurbulenceEquations.hh"
#include "equations/turbulence/TurbulenceScalarTransportEquation.hh"
#include "equations/turbulence/TurbulenceWallTreatment.hh"
#include "fields/MeshFieldTraits.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/WallYPlusStatistics.hh"
#include "solvers/BelosLinearSolver.hh"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

namespace SimpleFluid
{

/** @brief Runtime-selectable turbulence closure. */
enum class TurbulenceModelType
{
    Laminar,
    StandardKEpsilon,
    RNGKEpsilon,
    RealizableKEpsilon,
    StandardKOmega,     ///< Standard k-omega closure.
    BSLKOmega,          ///< Baseline blended k-omega closure.
    SSTKOmega           ///< Shear-stress-transport k-omega closure.
};

/** @brief Return the canonical database name of a turbulence closure. */
SIMPLEFLUID_EQUATIONS_EXPORT std::string_view
to_string(TurbulenceModelType model) noexcept;

/**
 * @brief Parse a turbulence model name.
 *
 * Accepted spellings are `laminar`, `standardKEpsilon`, `RNGKEpsilon`,
 * `realizableKEpsilon`, `standardKOmega`, `BSLKOmega`, and `SSTKOmega`.
 */
SIMPLEFLUID_EQUATIONS_EXPORT TurbulenceModelType
parse_turbulence_model_type(const std::string& value);

/** @brief Runtime selection of the wall treatment paired with a closure. */
enum class TurbulenceWallTreatmentType
{
    None,
    ResolvedLowReSST,              ///< Resolve SST through the viscous sublayer.
    StandardHighReKEpsilon,        ///< Apply high-Re k-epsilon wall functions.
    ResolvedLowReKEpsilon          ///< Resolve k-epsilon through the viscous sublayer.
};

/** @brief Return the canonical database name of a wall treatment. */
SIMPLEFLUID_EQUATIONS_EXPORT std::string_view
to_string(TurbulenceWallTreatmentType treatment) noexcept;

/**
 * @brief Parse `none`, `resolvedLowReSST`, `resolvedLowReKEpsilon`, or
 * `standardHighReKEpsilon`.
 */
SIMPLEFLUID_EQUATIONS_EXPORT TurbulenceWallTreatmentType
parse_turbulence_wall_treatment_type(
    const std::string& value);

/** @brief Optional direct buoyancy production in the turbulence equations. */
enum class TurbulenceBuoyancyModel
{
    None,
    OpenFOAMBoussinesq
};

/** @brief Return the canonical database name of a buoyancy-production model. */
SIMPLEFLUID_EQUATIONS_EXPORT std::string_view
to_string(TurbulenceBuoyancyModel model) noexcept;

/** @brief Parse `none` or `OpenFOAMBoussinesq`. */
SIMPLEFLUID_EQUATIONS_EXPORT TurbulenceBuoyancyModel
parse_turbulence_buoyancy_model(
    const std::string& value);

/** @brief Initial conditions, floors, and turbulent heat-flux controls. */
struct TurbulenceModelOptions
{
    TurbulenceModelType model = TurbulenceModelType::Laminar;
    real_t initial_turbulent_kinetic_energy = 1.0e-6;
    real_t initial_dissipation_rate = 1.0e-6;
    real_t initial_specific_dissipation_rate = 1.0;
    real_t min_turbulent_kinetic_energy = 1.0e-12; ///< Positive @f$k@f$ floor.
    real_t min_dissipation_rate = 1.0e-12; ///< Positive @f$\epsilon@f$ floor.
    real_t min_specific_dissipation_rate = 1.0e-12; ///< Positive @f$\omega@f$ floor.
    real_t turbulent_prandtl_number = 0.9;
    FVM::CellGradientScheme gradient_scheme =
        FVM::CellGradientScheme::LeastSquares;
    FVM::FaceCoefficientInterpolation coefficient_interpolation =
        FVM::FaceCoefficientInterpolation::Harmonic;
    /** Optional uniform override; otherwise BSL/SST solve wall distance. */
    std::optional<real_t> initial_wall_distance;
    /**
     * Explicit wall-distance anchors. Every name must identify a no-slip
     * velocity patch and is unioned with all other no-slip patches.
     */
    ArrString wall_distance_boundaries;
    /** Numerical controls for the automatic BSL/SST Poisson reconstruction. */
    WallDistanceEquationOptions wall_distance_equation;
    TurbulenceBuoyancyModel buoyancy_model =
        TurbulenceBuoyancyModel::None;
    /** Multiplier on OpenFOAM-compatible turbulent buoyancy production. */
    real_t buoyancy_coefficient = 1.0;
    TurbulenceWallTreatmentType wall_treatment =
        TurbulenceWallTreatmentType::None;
    /** Wall patches/constants; overlapping closure constants are coordinated. */
    TurbulenceWallTreatmentOptions wall_options;
};

/** @brief Validate a turbulence configuration before allocating model state. */
SIMPLEFLUID_EQUATIONS_EXPORT void
validate_turbulence_model_options(const TurbulenceModelOptions& options);

/** @brief Parse and validate flat turbulence database keys. */
SIMPLEFLUID_EQUATIONS_EXPORT TurbulenceModelOptions
turbulence_model_options_from_database(const Database& database);

/**
 * @brief Boussinesq fields and forcing used by direct turbulence buoyancy.
 *
 * Temperature-gradient production is used normally. When density feedback is
 * active, the equivalent reference-density gradient form is used so the
 * turbulence and momentum equations respond to the same material state.
 * With kinematic eddy viscosity, both forms have units of specific
 * turbulent-kinetic-energy production, @f$\mathrm{m^2\,s^{-3}}@f$:
 * @f[
 * G_b=C_b\beta\frac{\nu_t}{Pr_t}\mathbf{g}\mathbin{\cdot}\nabla T,
 * \qquad
 * G_b=-C_b\frac{\nu_t}{Pr_t\rho_\mathrm{ref}}
 *          \mathbf{g}\mathbin{\cdot}\nabla\rho.
 * @f]
 * Positive terms are explicit production; negative terms are linearized as
 * implicit sinks. The epsilon and omega equations receive the corresponding
 * OpenFOAM incompressible `fv::buoyancyTurbSource` secondary source. Its
 * epsilon orientation factor is
 * @f$C_3=\tanh((|U_\perp|+\mathrm{SMALL})/|U_\parallel|)@f$; this is
 * intentionally distinct from the opposite convention in OpenFOAM's
 * compressible `buoyantKEpsilon` closure.
 * @tparam Pack Tpetra type pack used for scalar storage.
 * @tparam MeshType Mesh and associated field-storage backend.
 */
template <TpetraTypePack Pack = DefaultTpetraTypes,
          class MeshType = Mesh<Pack>>
struct TurbulenceBuoyancyContext
{
    using scalar_type = typename Pack::scalar_type;
    using field_type = typename MeshFieldTraits<Pack, MeshType>::scalar_cell_type;

    const field_type* temperature = nullptr;
    const BoundaryConditionMap* temperature_boundary_conditions = nullptr;
    vec3<scalar_type> gravity{};
    scalar_type thermal_expansion{};
    bool density_feedback_enabled = false;
};

/**
 * @brief Owns turbulence state, two scalar transport equations, and effective properties.
 *
 * Molecular fields remain authoritative in MaterialPropertyFields. This model
 * publishes separate effective viscosity and conductivity fields for the
 * momentum and temperature equations. The implemented closures use shear
 * production and a gradient-diffusion turbulent heat flux. Optional resolved
 * SST, resolved standard/realizable k-epsilon, and high-Re standard-k-epsilon
 * wall treatments provide dynamic scalar data and face transport coefficients.
 * Optional signed Boussinesq production is coupled to both transported
 * turbulence equations.
 * The isotropic Reynolds stress is supplied explicitly as
 * @f$-2/3\,\nabla k@f$, so the solver pressure remains mechanical pressure.
 * @tparam Pack Tpetra type pack used for mesh and field storage.
 * @tparam MeshType Mesh and associated field-storage backend.
 */
template <TpetraTypePack Pack = DefaultTpetraTypes,
          class MeshType = Mesh<Pack>>
class SIMPLEFLUID_EQUATIONS_EXPORT TurbulenceModel
{
public:
    using mesh_type = MeshType;
    using field_traits = MeshFieldTraits<Pack, mesh_type>;
    using field_type = typename field_traits::scalar_cell_type;
    using velocity_field_type = typename field_traits::vector_cell_type;
    using tensor_field_type = typename field_traits::tensor_cell_type;
    using face_flux_field_type = typename field_traits::scalar_face_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using material_type = MaterialPropertyFields<Pack, mesh_type>;
    using velocity_boundary_cache_type = std::conditional_t<
        std::is_same_v<mesh_type, Mesh<Pack>>,
        FVM::VelocityBoundaryCache<Pack>,
        FVM::FieldStoredVelocityBoundaryCache<Pack, mesh_type>>;
    using boundary_cache_type = FVM::MeshBoundaryCache<Pack, mesh_type>;

    TurbulenceModel(SP<const mesh_type> mesh, const BoundaryConditionSet& boundary_conditions);
    ~TurbulenceModel();

    TurbulenceModel(const TurbulenceModel&) = delete;
    TurbulenceModel& operator=(const TurbulenceModel&) = delete;
    TurbulenceModel(TurbulenceModel&&) = delete;
    TurbulenceModel& operator=(TurbulenceModel&&) = delete;

    /**
     * @brief Replace the active closure and reset its transported fields.
     * @note This is collective over the mesh communicator.
     */
    void configure(const TurbulenceModelOptions& options, const material_type& material,
                   scalar_type reference_density);

    /**
     * @brief Parse and replace the active closure from flat database keys.
     * @note Parsing and configuration are collective over the mesh communicator.
     */
    void configure(const Database& database, const material_type& material,
                   scalar_type reference_density);

    /**
     * @brief Disable turbulence and release its lazily allocated state.
     * @note Invoke consistently on every mesh rank.
     */
    bool disable() noexcept;

    bool enabled() const noexcept;
    TurbulenceModelType type() const noexcept;
    const TurbulenceModelOptions& options() const noexcept;

    /**
     * @brief Advance k and epsilon/omega using the projected volumetric face flux.
     *
     * Both candidate fields are published only after both linear solves
     * converge. Eddy viscosity and effective properties are then refreshed.
     * This operation is collective over the mesh communicator.
     */
    LinearSolveSummary advance(const velocity_field_type& velocity,
                               const face_flux_field_type& projected_face_fluxes,
                               const velocity_boundary_cache_type& velocity_boundary_cache,
                               scalar_type time_step, const material_type& material,
                               scalar_type reference_density, FVM::NonOrthogonalTreatment treatment,
                               const LinearSolverOptions& linear_options = {},
                               const TurbulenceBuoyancyContext<Pack, mesh_type>*
                                   buoyancy_context = nullptr);

    /** Rebuild mu_eff and lambda_eff from current molecular and turbulent fields. */
    void refresh_effective_properties(const material_type& material, scalar_type reference_density);

    /**
     * @brief Transactionally restore transported and eddy-viscosity fields.
     *
     * This restart-oriented operation validates and publishes k, epsilon or
     * omega, and nu_t together, then rebuilds gradients, wall data, and
     * effective properties from the supplied accepted velocity.
     */
    void restore_transported_state(
        const field_type& turbulent_kinetic_energy,
        const field_type& secondary,
        const field_type& turbulent_kinematic_viscosity,
        const velocity_field_type& velocity,
        const material_type& material,
        scalar_type reference_density);

    /**
     * @brief Replace the positive BSL/SST wall-distance field and refresh derived properties.
     *
     * The replacement distance, closure eddy viscosity, and effective
     * properties are validated and staged before any accepted field is
     * published.
     * @note This is collective over the mesh communicator.
     */
    void set_wall_distance(const field_type& wall_distance,
                           const material_type& material,
                           scalar_type reference_density);

    const field_type& turbulent_kinetic_energy() const;
    /** Gradient used for the isotropic Reynolds-stress acceleration. */
    const velocity_field_type& turbulent_kinetic_energy_gradient() const;
    const field_type* dissipation_rate() const noexcept;
    const field_type* specific_dissipation_rate() const noexcept;
    const field_type& turbulent_kinematic_viscosity() const;
    const field_type& effective_dynamic_viscosity() const;
    const field_type& effective_thermal_conductivity() const;
    /** Signed accepted buoyancy production, or null when disabled. */
    const field_type* buoyancy_production() const noexcept;
    /** Explicit source assembled for the accepted k transport state. */
    const field_type* turbulent_kinetic_energy_source() const noexcept;
    /** Implicit sink coefficient assembled for the accepted k state. */
    const field_type* turbulent_kinetic_energy_sink() const noexcept;
    /** Explicit source assembled for epsilon or omega transport. */
    const field_type* secondary_source() const noexcept;
    /** Implicit sink coefficient assembled for epsilon or omega transport. */
    const field_type* secondary_sink() const noexcept;
    /** Reconstructed or explicitly supplied Menter wall distance, if active. */
    const field_type* wall_distance() const noexcept;

    /** Sparse face viscosities supplied by an active wall treatment. */
    const boundary_cache_type*
    effective_dynamic_viscosity_boundary_cache() const noexcept;

    /** Sparse face conductivities supplied by an active wall treatment. */
    const boundary_cache_type*
    effective_thermal_conductivity_boundary_cache() const noexcept;

    /** Cell diagnostic containing the maximum incident-wall y+ when active. */
    const field_type* wall_y_plus() const noexcept;

    /** Globally reduced accepted y+ statistics in configured patch order. */
    const Arr<WallYPlusStatistics>&
    wall_y_plus_statistics() const noexcept;

    /** Active fields suitable for opt-in solution output. */
    const std::map<std::string, const field_type*>& output_fields() const noexcept;

private:
    /** @brief Lazily allocated closure, transport, and derived-field state. */
    struct SIMPLEFLUID_EQUATIONS_LOCAL State;

    SIMPLEFLUID_EQUATIONS_LOCAL State& require_state();
    SIMPLEFLUID_EQUATIONS_LOCAL const State& require_state() const;
    SIMPLEFLUID_EQUATIONS_LOCAL
    void stage_effective_properties(State& state, const field_type& turbulent_kinematic_viscosity,
                                    const material_type& material, scalar_type reference_density,
                                    scalar_type turbulent_prandtl_number) const;
    SIMPLEFLUID_EQUATIONS_LOCAL
    void stage_menter_eddy_viscosity(
        State& state, const field_type& wall_distance,
        const material_type& material, scalar_type reference_density,
        std::string_view context) const;
    SIMPLEFLUID_EQUATIONS_LOCAL
    void commit_effective_properties(State& state) const;

    SP<const mesh_type> d_mesh;
    VectorBoundaryConditionMap d_velocity_boundary_conditions;
    velocity_boundary_cache_type d_wall_velocity_boundary_cache;
    TurbulenceBoundaryConditionSet d_boundary_conditions;
    TurbulenceModelOptions d_options;
    std::unique_ptr<State> d_state;
    std::map<std::string, const field_type*> d_empty_output_fields;
    Arr<WallYPlusStatistics> d_empty_wall_y_plus_statistics;
};

extern template class TurbulenceModel<DefaultTpetraTypes>;
extern template class TurbulenceModel<
    DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;

} // namespace SimpleFluid
