/**
 * @file TurbulenceModel.hh
 * @brief Runtime ownership and transport coupling for two-equation RANS models.
 */

#pragma once

#include "FVM/FaceFlux.hh"
#include "dataclass/Database.hh"
#include "equations/BoundaryConditions.hh"
#include "equations/BoussinesqModel.hh"
#include "equations/turbulence/TurbulenceEquations.hh"
#include "equations/turbulence/TurbulenceScalarTransportEquation.hh"
#include "equations/turbulence/TurbulenceWallTreatment.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/VectorCellField.hh"
#include "geometry/Mesh.hh"
#include "solvers/BelosLinearSolver.hh"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace SimpleFluid
{

/** @brief Runtime-selectable turbulence closure. */
enum class TurbulenceModelType
{
    Laminar,
    StandardKEpsilon,
    RNGKEpsilon,
    RealizableKEpsilon,
    StandardKOmega,
    BSLKOmega,
    SSTKOmega
};

/** @brief Return the canonical database name of a turbulence closure. */
std::string_view to_string(TurbulenceModelType model) noexcept;

/**
 * @brief Parse a turbulence model name.
 *
 * Accepted spellings are `laminar`, `standardKEpsilon`, `RNGKEpsilon`,
 * `realizableKEpsilon`, `standardKOmega`, `BSLKOmega`, and `SSTKOmega`.
 */
TurbulenceModelType parse_turbulence_model_type(const std::string& value);

/** @brief Runtime selection of the wall treatment paired with a closure. */
enum class TurbulenceWallTreatmentType
{
    None,
    ResolvedLowReSST,
    StandardHighReKEpsilon
};

/** @brief Return the canonical database name of a wall treatment. */
std::string_view to_string(TurbulenceWallTreatmentType treatment) noexcept;

/** @brief Parse `none`, `resolvedLowReSST`, or `standardHighReKEpsilon`. */
TurbulenceWallTreatmentType parse_turbulence_wall_treatment_type(
    const std::string& value);

/** @brief Initial conditions, floors, and turbulent heat-flux controls. */
struct TurbulenceModelOptions
{
    TurbulenceModelType model = TurbulenceModelType::Laminar;
    real_t initial_turbulent_kinetic_energy = 1.0e-6;
    real_t initial_dissipation_rate = 1.0e-6;
    real_t initial_specific_dissipation_rate = 1.0;
    real_t min_turbulent_kinetic_energy = 1.0e-12;
    real_t min_dissipation_rate = 1.0e-12;
    real_t min_specific_dissipation_rate = 1.0e-12;
    real_t turbulent_prandtl_number = 0.9;
    /** Required by BSL and SST until a distributed wall-distance solver exists. */
    std::optional<real_t> initial_wall_distance;
    TurbulenceWallTreatmentType wall_treatment =
        TurbulenceWallTreatmentType::None;
    /** Wall patches/constants; overlapping closure constants are coordinated. */
    TurbulenceWallTreatmentOptions wall_options;
};

/** @brief Validate a turbulence configuration before allocating model state. */
void validate_turbulence_model_options(const TurbulenceModelOptions& options);

/** @brief Parse and validate flat turbulence database keys. */
TurbulenceModelOptions turbulence_model_options_from_database(const Database& database);

/**
 * @brief Owns turbulence state, two scalar transport equations, and effective properties.
 *
 * Molecular fields remain authoritative in MaterialPropertyFields. This model
 * publishes separate effective viscosity and conductivity fields for the
 * momentum and temperature equations. The implemented closures use shear
 * production and a gradient-diffusion turbulent heat flux. Optional resolved
 * SST and high-Re standard-k-epsilon wall treatments provide dynamic scalar
 * data and face transport coefficients. Buoyancy production remains outside
 * this coupling layer.
 * The isotropic Reynolds stress is supplied explicitly as
 * @f$-2/3\,\nabla k@f$, so the solver pressure remains mechanical pressure.
 */
template <TpetraTypePack Pack = DefaultTpetraTypes> class TurbulenceModel
{
public:
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;
    using velocity_field_type = VectorCellField<Pack>;
    using face_flux_field_type = FaceField<Pack>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using material_type = MaterialPropertyFields<Pack>;

    TurbulenceModel(SP<const mesh_type> mesh, const BoundaryConditionSet& boundary_conditions);
    ~TurbulenceModel();

    TurbulenceModel(const TurbulenceModel&) = delete;
    TurbulenceModel& operator=(const TurbulenceModel&) = delete;
    TurbulenceModel(TurbulenceModel&&) = delete;
    TurbulenceModel& operator=(TurbulenceModel&&) = delete;

    /**
     * Replace the active closure and reset its transported fields.
     * @note This is collective over the mesh communicator.
     */
    void configure(const TurbulenceModelOptions& options, const material_type& material,
                   scalar_type reference_density);

    /**
     * Parse and replace the active closure from flat database keys.
     * @note Parsing and configuration are collective over the mesh communicator.
     */
    void configure(const Database& database, const material_type& material,
                   scalar_type reference_density);

    /**
     * Disable turbulence and release its lazily allocated state.
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
                               const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
                               scalar_type time_step, const material_type& material,
                               scalar_type reference_density, FVM::NonOrthogonalTreatment treatment,
                               const LinearSolverOptions& linear_options = {});

    /** Rebuild mu_eff and lambda_eff from current molecular and turbulent fields. */
    void refresh_effective_properties(const material_type& material, scalar_type reference_density);

    /**
     * Replace the positive BSL/SST wall-distance field.
     * @note This is collective over the mesh communicator.
     */
    void set_wall_distance(const field_type& wall_distance);

    const field_type& turbulent_kinetic_energy() const;
    /** Gradient used for the isotropic Reynolds-stress acceleration. */
    const velocity_field_type& turbulent_kinetic_energy_gradient() const;
    const field_type* dissipation_rate() const noexcept;
    const field_type* specific_dissipation_rate() const noexcept;
    const field_type& turbulent_kinematic_viscosity() const;
    const field_type& effective_dynamic_viscosity() const;
    const field_type& effective_thermal_conductivity() const;

    /** Sparse face viscosities supplied by an active wall treatment. */
    const FVM::BoundaryCache<Pack>*
    effective_dynamic_viscosity_boundary_cache() const noexcept;

    /** Sparse face conductivities supplied by an active wall treatment. */
    const FVM::BoundaryCache<Pack>*
    effective_thermal_conductivity_boundary_cache() const noexcept;

    /** Cell diagnostic containing the maximum incident-wall y+ when active. */
    const field_type* wall_y_plus() const noexcept;

    /** Active fields suitable for opt-in solution output. */
    const std::map<std::string, const field_type*>& output_fields() const noexcept;

private:
    struct State;

    State& require_state();
    const State& require_state() const;
    void stage_effective_properties(State& state, const field_type& turbulent_kinematic_viscosity,
                                    const material_type& material, scalar_type reference_density,
                                    scalar_type turbulent_prandtl_number) const;
    void commit_effective_properties(State& state) const;

    SP<const mesh_type> d_mesh;
    VectorBoundaryConditionMap d_velocity_boundary_conditions;
    FVM::VelocityBoundaryCache<Pack> d_wall_velocity_boundary_cache;
    TurbulenceBoundaryConditionSet d_boundary_conditions;
    TurbulenceModelOptions d_options;
    std::unique_ptr<State> d_state;
    std::map<std::string, const field_type*> d_empty_output_fields;
};

} // namespace SimpleFluid
