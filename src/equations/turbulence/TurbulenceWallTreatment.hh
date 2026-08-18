/**
 * @file TurbulenceWallTreatment.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Policy-based wall treatments for two-equation RANS closures.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "SimpleFluidExport.hh"
#include "FVM/BoundaryCache.hh"
#include "FVM/FaceFlux.hh"
#include "dataclass/typedefs.hh"
#include "equations/BoundaryConditions.hh"
#include "equations/BoussinesqModel.hh"
#include "fields/MeshFieldTraits.hh"
#include "geometry/MeshHandle.hh"

#include <optional>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/** @brief Momentum wall-law roughness applied to a configured no-slip patch. */
enum class TurbulenceWallRoughnessModel
{
    Smooth,
    SandGrain
};

/** @brief Return the canonical database name of a wall roughness model. */
SIMPLEFLUID_EQUATIONS_EXPORT std::string_view
to_string(TurbulenceWallRoughnessModel model) noexcept;

/** @brief Parse `smooth` or `sandGrain`. */
SIMPLEFLUID_EQUATIONS_EXPORT TurbulenceWallRoughnessModel
parse_turbulence_wall_roughness_model(
    const std::string& value);

/** @brief Thermal wall-law closure applied by a high-Re wall treatment. */
enum class TurbulenceThermalWallLaw
{
    TurbulentPrandtl,
    Jayatilleke
};

/** @brief Return the canonical database name of a thermal wall law. */
SIMPLEFLUID_EQUATIONS_EXPORT std::string_view
to_string(TurbulenceThermalWallLaw law) noexcept;

/** @brief Parse `turbulentPrandtl` or `Jayatilleke`. */
SIMPLEFLUID_EQUATIONS_EXPORT TurbulenceThermalWallLaw
parse_turbulence_thermal_wall_law(
    const std::string& value);

/** @brief Constants and explicitly selected no-slip patches for wall treatment.
 */
struct TurbulenceWallTreatmentOptions
{
    ArrString boundary_names;
    /**
     * @brief Patch-aligned roughness modes.
     *
     * An empty vector selects smooth walls on every configured patch.
     * Otherwise this, `roughness_heights`, and `roughness_constants` must each
     * have exactly `boundary_names.size()` entries.
     */
    Arr<TurbulenceWallRoughnessModel> roughness_models;
    /** Equivalent sand-grain roughness height @f$K_s@f$ [m], patch aligned. */
    ArrReal roughness_heights;
    /** Sand-grain roughness constant @f$C_s@f$, patch aligned. */
    ArrReal roughness_constants;
    /** Standard k-epsilon Cmu, shared with that closure when selected. */
    real_t c_mu = 0.09;
    /** Wall-law kappa, also shared with the resolved SST closure. */
    real_t kappa = 0.41;
    real_t log_layer_e = 9.8; ///< Wall-law logarithmic-layer constant.
    /** Thermal wall law; the default preserves the prior constant-Prt flux. */
    TurbulenceThermalWallLaw thermal_wall_law =
        TurbulenceThermalWallLaw::TurbulentPrandtl;
    /**
     * @brief Optional wall-specific turbulent Prandtl number.
     *
     * When absent, `evaluate()` uses the turbulence-model Prt passed by the
     * caller. The same value enters the Jayatilleke correlation.
     */
    std::optional<real_t> thermal_turbulent_prandtl_number;
    /** Match OpenFOAM.com's optional epsilonWallFunction low-Re correction. */
    bool epsilon_low_re_correction = false;
    /** SST inner beta coefficient, shared with the paired closure. */
    real_t sst_beta_1 = 0.075;
    /** Menter's wall-face coefficient; 60 is distinct from a 6-coefficient cell
     * constraint. */
    real_t sst_omega_wall_coefficient = 60.0;
};

/** @brief Validate policy-independent wall-treatment constants and patch names.
 */
SIMPLEFLUID_EQUATIONS_EXPORT void
validate_turbulence_wall_treatment_options(
    const TurbulenceWallTreatmentOptions& options);

/** @brief OpenFOAM.com v2606 ten-iteration sublayer intersection. */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
openfoam_y_plus_lam(real_t kappa, real_t log_layer_e);

/** @brief Tag selecting integration-to-the-wall SST boundary data. */
struct SIMPLEFLUID_PUBLIC_TYPE ResolvedLowReSSTWallPolicy
{
    static constexpr std::string_view name = "resolvedLowReSST";
};

/**
 * @brief Tag selecting resolved viscous-sublayer k-epsilon boundary data.
 *
 * This policy supplies @f$k=0@f$ at the wall, the adjacent-cell constraint
 * @f$\epsilon=2\nu k/y^2@f$, zero wall-adjacent shear production, and
 * molecular-only wall transport. It is intended for near-wall meshes with
 * @f$y^+\approx 1@f$.
 */
struct SIMPLEFLUID_PUBLIC_TYPE ResolvedLowReKEpsilonWallPolicy
{
    static constexpr std::string_view name = "resolvedLowReKEpsilon";
};

/** @brief Tag selecting the OpenFOAM.com v2606 stepwise high-Re k-epsilon
 * wall-function set. */
struct SIMPLEFLUID_PUBLIC_TYPE StandardHighReKEpsilonWallPolicy
{
    static constexpr std::string_view name = "standardHighReKEpsilon";
};

/**
 * @brief Values staged for one configured wall face.
 * @tparam Scalar Floating-point scalar type used by the evaluation.
 */
template <class Scalar> struct TurbulenceWallFaceEvaluation
{
    BoundaryCondition turbulent_kinetic_energy{};
    BoundaryCondition secondary{}; ///< Wall condition for @f$\epsilon@f$ or @f$\omega@f$.
    Scalar wall_distance{};
    Scalar y_plus{};
    Scalar roughness_height{};
    Scalar roughness_constant{};
    Scalar roughness_height_plus{};
    Scalar effective_log_layer_e{};
    Scalar turbulent_kinematic_viscosity{};
    Scalar turbulent_thermal_diffusivity{};
    Scalar jayatilleke_p{};
    Scalar thermal_y_plus_transition{};
    Scalar effective_dynamic_viscosity{};
    Scalar effective_thermal_conductivity{};
    /** Per-face contribution before equal-count corner averaging. */
    std::optional<Scalar> secondary_constraint;
    /** Per-face contribution before equal-count corner averaging. */
    std::optional<Scalar> production_override;
};

/**
 * @brief Policy-based evaluator for resolved SST and k-epsilon walls.
 *
 * The object owns immutable geometry/configuration only. `evaluate()` creates
 * an independent, copyable staging object so a caller can preserve accepted
 * wall data until all candidate turbulence fields have been validated.
 * Construction and evaluation are collective over the mesh communicator.
 * @tparam Pack Tpetra type pack used for mesh and field storage.
 * @tparam Policy Wall-treatment policy tag.
 * @tparam MeshType Mesh and associated field-storage backend.
 */
template <TpetraTypePack Pack, class Policy,
          class MeshType = Mesh<Pack>>
class SIMPLEFLUID_EQUATIONS_EXPORT TurbulenceWallTreatment
{
public:
    using mesh_type = MeshType;
    using field_traits = MeshFieldTraits<Pack, mesh_type>;
    using field_type = typename field_traits::scalar_cell_type;
    using velocity_field_type = typename field_traits::vector_cell_type;
    using material_type = MaterialPropertyFields<Pack, mesh_type>;
    using velocity_boundary_cache_type = std::conditional_t<
        std::is_same_v<mesh_type, Mesh<Pack>>,
        FVM::VelocityBoundaryCache<Pack>,
        FVM::FieldStoredVelocityBoundaryCache<Pack, mesh_type>>;
    using boundary_cache_type = FVM::MeshBoundaryCache<Pack, mesh_type>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using face_evaluation_type = TurbulenceWallFaceEvaluation<scalar_type>;

    /** @brief Copyable/movable result of one wall evaluation. */
    class Evaluation
    {
    public:
        Evaluation() = default;
        Evaluation(const Evaluation&) = default;
        SIMPLEFLUID_EQUATIONS_LOCAL Evaluation(Evaluation&&) noexcept = default;
        Evaluation& operator=(const Evaluation&) = default;
        Evaluation& operator=(Evaluation&&) noexcept = default;

        /** Return staged data for one configured local wall face. */
        const face_evaluation_type& face(int batch_id, size_t in_batch_id) const;

        /** True if this rank has staged data for the requested batch face. */
        bool contains_face(int batch_id, size_t in_batch_id) const noexcept;

        /** Equal-face-count averaged epsilon constraint for an owned cell. */
        std::optional<scalar_type> secondary_constraint(local_ordinal_type cell_lid) const;

        /** Equal-face-count averaged wall production for an owned cell. */
        std::optional<scalar_type> production_override(local_ordinal_type cell_lid) const;

        /** Maximum y+ among configured wall faces incident on an owned cell. */
        std::optional<scalar_type> cell_y_plus(local_ordinal_type cell_lid) const;

        /** Sparse per-wall-face effective viscosity cache for momentum transport.
         */
        const boundary_cache_type& boundary_dynamic_viscosity() const noexcept
        {
            return d_boundary_dynamic_viscosity;
        }

        /** Sparse per-wall-face effective conductivity cache for heat transport. */
        const boundary_cache_type& boundary_thermal_conductivity() const noexcept
        {
            return d_boundary_thermal_conductivity;
        }

        /** Sparse molecular wall diffusivity for resolved turbulence scalars. */
        const boundary_cache_type& boundary_scalar_diffusivity() const noexcept
        {
            return d_boundary_scalar_diffusivity;
        }

        const SP<const mesh_type>& mesh_ptr() const noexcept { return d_mesh; }

    private:
        friend class TurbulenceWallTreatment<Pack, Policy, MeshType>;

        SIMPLEFLUID_EQUATIONS_LOCAL
        explicit Evaluation(SP<const mesh_type> mesh);
        SIMPLEFLUID_EQUATIONS_LOCAL
        void check_owned_cell(local_ordinal_type cell_lid) const;

        SP<const mesh_type> d_mesh;
        std::unordered_map<int, Arr<face_evaluation_type>> d_faces;
        Arr<std::optional<scalar_type>> d_secondary_constraints;
        Arr<std::optional<scalar_type>> d_production_overrides;
        Arr<std::optional<scalar_type>> d_cell_y_plus;
        boundary_cache_type d_boundary_dynamic_viscosity;
        boundary_cache_type d_boundary_thermal_conductivity;
        boundary_cache_type d_boundary_scalar_diffusivity;
    };

    TurbulenceWallTreatment(SP<const mesh_type> mesh, TurbulenceWallTreatmentOptions options,
                            const VectorBoundaryConditionMap& velocity_boundary_conditions);

    TurbulenceWallTreatment(SP<const mesh_type> mesh, TurbulenceWallTreatmentOptions options,
                            const velocity_boundary_cache_type& velocity_boundary_cache);

    /**
     * @brief Evaluate all configured local wall faces without mutating prior
     * data.
     *
     * SST uses k=0 and omega=60 nu/(beta1 y^2) at the face. Resolved k-epsilon
     * uses k=0, homogeneous-Neumann epsilon, molecular wall transport, and the
     * viscous-sublayer epsilon/G constraints in adjacent owned cells. High-Re
     * standard k-epsilon uses homogeneous-Neumann k/epsilon face data, v2606
     * stepwise nut, and equal-face-count epsilon/G constraints. Its optional
     * epsilon low-Re correction is disabled by default, as in OpenFOAM.com
     * v2606.
     */
    Evaluation evaluate(const field_type& turbulent_kinetic_energy,
                        const velocity_field_type& velocity,
                        const velocity_boundary_cache_type& velocity_boundary_cache,
                        const material_type& material, scalar_type reference_density,
                        scalar_type turbulent_prandtl_number,
                        const Evaluation* accepted_evaluation = nullptr) const;

    const TurbulenceWallTreatmentOptions& options() const noexcept { return d_options; }

    /** OpenFOAM.com v2606 sublayer intersection for the configured constants. */
    scalar_type y_plus_lam() const noexcept { return static_cast<scalar_type>(d_y_plus_lam); }

private:
    /** @brief Rank-local batch identifier paired with its configured name. */
    struct LocalWallBatch
    {
        int id{};
        std::string name;
        TurbulenceWallRoughnessModel roughness_model =
            TurbulenceWallRoughnessModel::Smooth;
        real_t roughness_height{};
        real_t roughness_constant{};
    };

    SIMPLEFLUID_EQUATIONS_LOCAL
    void initialize(
        const std::unordered_map<std::string, BoundaryConditionType>&
            boundary_types);
    SIMPLEFLUID_EQUATIONS_LOCAL
    void validate_velocity_cache(const velocity_boundary_cache_type& velocity_boundary_cache) const;

    SP<const mesh_type> d_mesh;
    TurbulenceWallTreatmentOptions d_options;
    real_t d_y_plus_lam{};
    std::vector<LocalWallBatch> d_local_wall_batches;
};

template <TpetraTypePack Pack = DefaultTpetraTypes,
          class MeshType = Mesh<Pack>>
using ResolvedLowReSSTWallTreatment =
    TurbulenceWallTreatment<Pack, ResolvedLowReSSTWallPolicy, MeshType>;

template <TpetraTypePack Pack = DefaultTpetraTypes,
          class MeshType = Mesh<Pack>>
using ResolvedLowReKEpsilonWallTreatment =
    TurbulenceWallTreatment<Pack, ResolvedLowReKEpsilonWallPolicy, MeshType>;

template <TpetraTypePack Pack = DefaultTpetraTypes,
          class MeshType = Mesh<Pack>>
using StandardHighReKEpsilonWallTreatment =
    TurbulenceWallTreatment<Pack, StandardHighReKEpsilonWallPolicy, MeshType>;

extern template class TurbulenceWallTreatment<
    DefaultTpetraTypes, ResolvedLowReSSTWallPolicy>;
extern template class TurbulenceWallTreatment<
    DefaultTpetraTypes, ResolvedLowReKEpsilonWallPolicy>;
extern template class TurbulenceWallTreatment<
    DefaultTpetraTypes, StandardHighReKEpsilonWallPolicy>;
extern template class TurbulenceWallTreatment<
    DefaultTpetraTypes, ResolvedLowReSSTWallPolicy,
    MeshHandle<DefaultTpetraTypes>>;
extern template class TurbulenceWallTreatment<
    DefaultTpetraTypes, ResolvedLowReKEpsilonWallPolicy,
    MeshHandle<DefaultTpetraTypes>>;
extern template class TurbulenceWallTreatment<
    DefaultTpetraTypes, StandardHighReKEpsilonWallPolicy,
    MeshHandle<DefaultTpetraTypes>>;

} // namespace SimpleFluid
