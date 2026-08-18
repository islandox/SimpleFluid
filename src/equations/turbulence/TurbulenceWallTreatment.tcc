/**
 * @file TurbulenceWallTreatment.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Template implementation of policy-based turbulence wall treatment.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "equations/turbulence/TurbulenceWallTreatment.hh"

#include "FVM/details/OperatorDetails.hh"
#include "equations/turbulence/TurbulenceCollectiveValidation.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace SimpleFluid
{
namespace turbulence_wall_detail
{

template <class Policy>
inline constexpr bool supported_policy_v =
    std::is_same_v<Policy, ResolvedLowReSSTWallPolicy> ||
    std::is_same_v<Policy, ResolvedLowReKEpsilonWallPolicy> ||
    std::is_same_v<Policy, StandardHighReKEpsilonWallPolicy>;

template <class Policy>
inline constexpr bool resolved_wall_policy_v =
    std::is_same_v<Policy, ResolvedLowReSSTWallPolicy> ||
    std::is_same_v<Policy, ResolvedLowReKEpsilonWallPolicy>;

template <class Policy>
inline constexpr bool epsilon_wall_policy_v =
    std::is_same_v<Policy, ResolvedLowReKEpsilonWallPolicy> ||
    std::is_same_v<Policy, StandardHighReKEpsilonWallPolicy>;

/**
 * @brief Test whether every component of a vector is finite.
 * @tparam Scalar Vector scalar type.
 * @param value Vector to inspect.
 * @return True when all components are finite.
 */
template <class Scalar> bool finite_vector(const vec3<Scalar>& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

/**
 * @brief Require one configuration string to be identical on every rank.
 * @tparam Pack Tpetra type pack used by the mesh communicator.
 * @param mesh Distributed mesh.
 * @param value Rank-local string value.
 * @param context Diagnostic label.
 */
template <class MeshType>
void require_uniform_string(const MeshType& mesh, const std::string& value,
                            const std::string& context)
{
    turbulence_detail::require_uniform_integral(mesh, static_cast<int>(value.size()),
                                                context + " length");
    for (const auto character : value)
    {
        turbulence_detail::require_uniform_integral(mesh, static_cast<unsigned char>(character),
                                                    context + " characters");
    }
}

/**
 * @brief Physical inputs needed to evaluate one wall face.
 * @tparam Scalar Floating-point scalar type.
 * @tparam Vec Three-component vector type.
 */
template <class Scalar, class Vec> struct FaceInputs
{
    Scalar k{};
    Scalar wall_distance{};
    Scalar molecular_nu{};
    Vec owner_velocity{};
    Vec wall_velocity{};
    Vec outward_normal{};
};

/** @brief Patch-local equivalent sand-grain roughness parameters. */
template <class Scalar> struct RoughnessInputs
{
    TurbulenceWallRoughnessModel model = TurbulenceWallRoughnessModel::Smooth;
    Scalar height{};
    Scalar constant{};
    Scalar accepted_turbulent_nu{};
};

/**
 * @brief Compute a shear-based resolved-wall y+ using molecular viscosity.
 */
template <class Scalar, class Vec>
Scalar resolved_wall_y_plus(const FaceInputs<Scalar, Vec>& input)
{
    const auto velocity_difference = input.owner_velocity - input.wall_velocity;
    const auto normal_velocity =
        input.outward_normal * velocity_difference.dot(input.outward_normal);
    const auto tangential_velocity = velocity_difference - normal_velocity;
    const auto tangential_gradient_magnitude =
        tangential_velocity.norm() / input.wall_distance;
    const auto friction_velocity =
        std::sqrt(input.molecular_nu * tangential_gradient_magnitude);
    return input.wall_distance * friction_velocity / input.molecular_nu;
}

/**
 * @brief Evaluate integration-to-the-wall SST boundary data.
 * @tparam Scalar Floating-point scalar type.
 * @tparam Vec Three-component vector type.
 * @param input Validated physical face inputs.
 * @param options Wall-treatment constants.
 * @return Staged resolved-SST wall values.
 */
template <class Scalar, class Vec>
TurbulenceWallFaceEvaluation<Scalar>
evaluate_resolved_sst_face(const FaceInputs<Scalar, Vec>& input,
                           const TurbulenceWallTreatmentOptions& options)
{
    TurbulenceWallFaceEvaluation<Scalar> result;
    result.turbulent_kinetic_energy = {BoundaryConditionType::Dirichlet, 0.0};
    result.secondary = {BoundaryConditionType::Dirichlet,
                        options.sst_omega_wall_coefficient *
                            static_cast<real_t>(input.molecular_nu) /
                            (options.sst_beta_1 * static_cast<real_t>(input.wall_distance) *
                             static_cast<real_t>(input.wall_distance))};
    result.wall_distance = input.wall_distance;
    result.turbulent_kinematic_viscosity = Scalar{};
    result.y_plus = resolved_wall_y_plus(input);
    return result;
}

/**
 * @brief Evaluate resolved viscous-sublayer k-epsilon boundary data.
 * @tparam Scalar Floating-point scalar type.
 * @tparam Vec Three-component vector type.
 * @param input Validated physical face inputs.
 * @return Staged resolved k-epsilon wall values and cell constraints.
 */
template <class Scalar, class Vec>
TurbulenceWallFaceEvaluation<Scalar>
evaluate_resolved_k_epsilon_face(const FaceInputs<Scalar, Vec>& input)
{
    TurbulenceWallFaceEvaluation<Scalar> result;
    result.turbulent_kinetic_energy = {BoundaryConditionType::Dirichlet, 0.0};
    result.secondary = {BoundaryConditionType::Neumann, 0.0};
    result.wall_distance = input.wall_distance;
    result.y_plus = resolved_wall_y_plus(input);
    result.turbulent_kinematic_viscosity = Scalar{};
    result.secondary_constraint =
        Scalar{2} * input.k * input.molecular_nu /
        (input.wall_distance * input.wall_distance);
    result.production_override = Scalar{};
    return result;
}

/**
 * @brief Evaluate the standard high-Re k-epsilon wall function.
 * @tparam Scalar Floating-point scalar type.
 * @tparam Vec Three-component vector type.
 * @param input Validated physical face inputs.
 * @param options Wall-treatment constants.
 * @param y_plus_lam Viscous/log-layer intersection.
 * @return Staged k-epsilon wall values and cell constraints.
 */
template <class Scalar, class Vec>
TurbulenceWallFaceEvaluation<Scalar>
evaluate_standard_k_epsilon_face(const FaceInputs<Scalar, Vec>& input,
                                 const TurbulenceWallTreatmentOptions& options, Scalar y_plus_lam,
                                 const RoughnessInputs<Scalar>& roughness)
{
    TurbulenceWallFaceEvaluation<Scalar> result;
    result.turbulent_kinetic_energy = {BoundaryConditionType::Neumann, 0.0};
    result.secondary = {BoundaryConditionType::Neumann, 0.0};
    result.wall_distance = input.wall_distance;

    const auto c_mu_quarter = static_cast<Scalar>(std::pow(options.c_mu, 0.25));
    const auto c_mu_three_quarters = static_cast<Scalar>(std::pow(options.c_mu, 0.75));
    const auto sqrt_k = std::sqrt(input.k);
    const auto k_based_y_plus = c_mu_quarter * input.wall_distance * sqrt_k / input.molecular_nu;
    const auto velocity_difference = input.owner_velocity - input.wall_velocity;
    const auto normal_velocity =
        input.outward_normal * velocity_difference.dot(input.outward_normal);
    const auto tangential_velocity = velocity_difference - normal_velocity;
    const auto wall_shear_gradient_magnitude =
        tangential_velocity.norm() / input.wall_distance;

    result.y_plus = k_based_y_plus;
    result.roughness_height = roughness.height;
    result.roughness_constant = roughness.constant;
    result.effective_log_layer_e = static_cast<Scalar>(options.log_layer_e);

    const bool rough_wall = roughness.model == TurbulenceWallRoughnessModel::SandGrain;
    const bool in_log_layer = k_based_y_plus > y_plus_lam;
    if (rough_wall)
    {
        const auto friction_velocity = c_mu_quarter * sqrt_k;
        const auto roughness_height_plus =
            friction_velocity * roughness.height / input.molecular_nu;
        result.roughness_height_plus = roughness_height_plus;

        Scalar roughness_multiplier{1};
        if (roughness_height_plus > Scalar{2.25} &&
            roughness_height_plus < Scalar{90})
        {
            const auto base =
                (roughness_height_plus - Scalar{2.25}) / Scalar{87.75} +
                roughness.constant * roughness_height_plus;
            const auto exponent = std::sin(
                Scalar{0.4258} * (std::log(roughness_height_plus) - Scalar{0.811}));
            roughness_multiplier = std::pow(base, exponent);
        }
        else if (roughness_height_plus >= Scalar{90})
        {
            roughness_multiplier =
                Scalar{1} + roughness.constant * roughness_height_plus;
        }
        result.effective_log_layer_e =
            static_cast<Scalar>(options.log_layer_e) / roughness_multiplier;
        const auto logarithm = std::log(std::max(
            result.effective_log_layer_e * k_based_y_plus, Scalar{1.0001}));
        const auto raw_nut =
            input.molecular_nu *
            (k_based_y_plus * static_cast<Scalar>(options.kappa) / logarithm - Scalar{1});
        const auto limiter_scale =
            std::max(roughness.accepted_turbulent_nu, input.molecular_nu);
        result.turbulent_kinematic_viscosity =
            std::max(std::min(raw_nut, Scalar{2} * limiter_scale),
                     Scalar{0.5} * limiter_scale);
    }
    else if (in_log_layer)
    {
        const auto logarithm = std::log(
            std::max(static_cast<Scalar>(options.log_layer_e) * k_based_y_plus, Scalar{1.0001}));
        result.turbulent_kinematic_viscosity =
            input.molecular_nu *
            std::max(k_based_y_plus * static_cast<Scalar>(options.kappa) / logarithm - Scalar{1},
                     Scalar{});
    }
    else
    {
        result.turbulent_kinematic_viscosity = Scalar{};
    }

    if (k_based_y_plus < y_plus_lam)
    {
        const auto effective_nu =
            input.molecular_nu + result.turbulent_kinematic_viscosity;
        result.y_plus =
            input.wall_distance *
            std::sqrt(effective_nu * wall_shear_gradient_magnitude) /
            input.molecular_nu;
    }

    if (!options.epsilon_low_re_correction || k_based_y_plus >= y_plus_lam)
    {
        result.secondary_constraint = c_mu_three_quarters * input.k * sqrt_k /
                                      (static_cast<Scalar>(options.kappa) * input.wall_distance);
    }
    else
    {
        result.secondary_constraint =
            Scalar{2} * input.k * input.molecular_nu / (input.wall_distance * input.wall_distance);
    }

    if (!options.epsilon_low_re_correction || in_log_layer)
    {
        result.production_override = (result.turbulent_kinematic_viscosity + input.molecular_nu) *
                                     wall_shear_gradient_magnitude * c_mu_quarter * sqrt_k /
                                     (static_cast<Scalar>(options.kappa) * input.wall_distance);
    }
    else
    {
        result.production_override = Scalar{};
    }
    return result;
}

/**
 * @brief Jayatilleke log-layer offset for a molecular/turbulent Prandtl ratio.
 */
template <class Scalar> Scalar jayatilleke_p(Scalar prandtl_ratio)
{
    return Scalar{9.24} * (std::pow(prandtl_ratio, Scalar{0.75}) - Scalar{1}) *
           (Scalar{1} + Scalar{0.28} * std::exp(Scalar{-0.007} * prandtl_ratio));
}

/**
 * @brief Solve the Jayatilleke viscous/log thermal-layer intersection.
 */
template <class Scalar>
Scalar jayatilleke_y_plus_thermal(Scalar p, Scalar prandtl_ratio, Scalar kappa,
                                  Scalar log_layer_e)
{
    constexpr Scalar convergence_tolerance{0.01};
    Scalar y_plus_thermal{11};
    for (int iteration = 0; iteration < 10; ++iteration)
    {
        const auto logarithm =
            std::log(log_layer_e * y_plus_thermal);
        const auto residual =
            y_plus_thermal - (logarithm / kappa + p) / prandtl_ratio;
        const auto derivative =
            Scalar{1} - Scalar{1} / (kappa * prandtl_ratio * y_plus_thermal);
        const auto next_y_plus_thermal =
            y_plus_thermal - residual / derivative;
        if (next_y_plus_thermal <= Scalar{})
        {
            return Scalar{};
        }
        if (std::abs(next_y_plus_thermal - y_plus_thermal) <
            convergence_tolerance)
        {
            return next_y_plus_thermal;
        }
        y_plus_thermal = next_y_plus_thermal;
    }
    return y_plus_thermal;
}

} // namespace turbulence_wall_detail

/**
 * @brief Construct empty staged wall data tied to a mesh.
 * @tparam Pack Tpetra type pack used by the treatment.
 * @tparam Policy Wall-treatment policy tag.
 * @param mesh Computational mesh retained by the result.
 */
template <TpetraTypePack Pack, class Policy, class MeshType>
TurbulenceWallTreatment<Pack, Policy, MeshType>::Evaluation::Evaluation(SP<const mesh_type> mesh)
    : d_mesh(std::move(mesh)), d_secondary_constraints(d_mesh ? d_mesh->num_owned_cells() : 0),
      d_production_overrides(d_mesh ? d_mesh->num_owned_cells() : 0),
      d_cell_y_plus(d_mesh ? d_mesh->num_owned_cells() : 0),
      d_boundary_dynamic_viscosity{{}, d_mesh}, d_boundary_thermal_conductivity{{}, d_mesh},
      d_boundary_scalar_diffusivity{{}, d_mesh}
{
}

/**
 * @brief Return staged data for one configured local wall face.
 * @tparam Pack Tpetra type pack used by the treatment.
 * @tparam Policy Wall-treatment policy tag.
 * @param batch_id Boundary batch identifier.
 * @param in_batch_id Face index within the batch.
 * @return Staged evaluation for the requested face.
 * @throws std::out_of_range if the batch or face is absent.
 */
template <TpetraTypePack Pack, class Policy, class MeshType>
auto TurbulenceWallTreatment<Pack, Policy, MeshType>::Evaluation::face(int batch_id,
                                                                   size_t in_batch_id) const
    -> const face_evaluation_type&
{
    const auto iter = d_faces.find(batch_id);
    if (iter == d_faces.end())
    {
        throw std::out_of_range("Turbulence wall evaluation does not contain the "
                                "requested boundary batch.");
    }
    return iter->second.at(in_batch_id);
}

/**
 * @brief Test whether staged data contains a boundary face.
 * @tparam Pack Tpetra type pack used by the treatment.
 * @tparam Policy Wall-treatment policy tag.
 * @param batch_id Boundary batch identifier.
 * @param in_batch_id Face index within the batch.
 * @return True when the face has staged data on this rank.
 */
template <TpetraTypePack Pack, class Policy, class MeshType>
bool TurbulenceWallTreatment<Pack, Policy, MeshType>::Evaluation::contains_face(
    int batch_id, size_t in_batch_id) const noexcept
{
    const auto iter = d_faces.find(batch_id);
    return iter != d_faces.end() && in_batch_id < iter->second.size();
}

/**
 * @brief Require an owned local cell identifier for staged cell data.
 * @tparam Pack Tpetra type pack used by the treatment.
 * @tparam Policy Wall-treatment policy tag.
 * @param cell_lid Cell identifier to validate.
 * @throws std::out_of_range if the mesh is absent or the cell is not owned.
 */
template <TpetraTypePack Pack, class Policy, class MeshType>
void TurbulenceWallTreatment<Pack, Policy, MeshType>::Evaluation::check_owned_cell(
    local_ordinal_type cell_lid) const
{
    if (!d_mesh || !d_mesh->is_owned_cell(cell_lid))
    {
        throw std::out_of_range("Turbulence wall evaluation requires an owned cell local ID.");
    }
}

/**
 * @brief Return the averaged secondary-variable wall constraint for a cell.
 * @tparam Pack Tpetra type pack used by the treatment.
 * @tparam Policy Wall-treatment policy tag.
 * @param cell_lid Owned local cell identifier.
 * @return Constraint when a configured wall contributes to the cell.
 * @throws std::out_of_range if @p cell_lid is not owned.
 */
template <TpetraTypePack Pack, class Policy, class MeshType>
auto TurbulenceWallTreatment<Pack, Policy, MeshType>::Evaluation::secondary_constraint(
    local_ordinal_type cell_lid) const -> std::optional<scalar_type>
{
    check_owned_cell(cell_lid);
    return d_secondary_constraints.at(static_cast<size_t>(cell_lid));
}

/**
 * @brief Return the averaged wall-production override for a cell.
 * @tparam Pack Tpetra type pack used by the treatment.
 * @tparam Policy Wall-treatment policy tag.
 * @param cell_lid Owned local cell identifier.
 * @return Production override when a configured wall contributes.
 * @throws std::out_of_range if @p cell_lid is not owned.
 */
template <TpetraTypePack Pack, class Policy, class MeshType>
auto TurbulenceWallTreatment<Pack, Policy, MeshType>::Evaluation::production_override(
    local_ordinal_type cell_lid) const -> std::optional<scalar_type>
{
    check_owned_cell(cell_lid);
    return d_production_overrides.at(static_cast<size_t>(cell_lid));
}

/**
 * @brief Return the maximum staged y+ incident on an owned cell.
 * @tparam Pack Tpetra type pack used by the treatment.
 * @tparam Policy Wall-treatment policy tag.
 * @param cell_lid Owned local cell identifier.
 * @return Maximum y+ when a configured wall contributes.
 * @throws std::out_of_range if @p cell_lid is not owned.
 */
template <TpetraTypePack Pack, class Policy, class MeshType>
auto TurbulenceWallTreatment<Pack, Policy, MeshType>::Evaluation::cell_y_plus(
    local_ordinal_type cell_lid) const -> std::optional<scalar_type>
{
    check_owned_cell(cell_lid);
    return d_cell_y_plus.at(static_cast<size_t>(cell_lid));
}

/**
 * @brief Construct wall treatment from configured velocity conditions.
 * @tparam Pack Tpetra type pack used by the treatment.
 * @tparam Policy Wall-treatment policy tag.
 * @param mesh Distributed computational mesh.
 * @param options Wall patches and model constants.
 * @param velocity_boundary_conditions Configured velocity conditions by name.
 * @throws std::invalid_argument if mesh, options, or patches are invalid.
 */
template <TpetraTypePack Pack, class Policy, class MeshType>
TurbulenceWallTreatment<Pack, Policy, MeshType>::TurbulenceWallTreatment(
    SP<const mesh_type> mesh, TurbulenceWallTreatmentOptions options,
    const VectorBoundaryConditionMap& velocity_boundary_conditions)
    : d_mesh(std::move(mesh)), d_options(std::move(options))
{
    std::unordered_map<std::string, BoundaryConditionType> boundary_types;
    boundary_types.reserve(velocity_boundary_conditions.size());
    for (const auto& [name, condition] : velocity_boundary_conditions)
    {
        boundary_types.emplace(name, condition.type);
    }
    initialize(boundary_types);
}

/**
 * @brief Construct wall treatment from a validated velocity boundary cache.
 * @tparam Pack Tpetra type pack used by the treatment.
 * @tparam Policy Wall-treatment policy tag.
 * @param mesh Distributed computational mesh.
 * @param options Wall patches and model constants.
 * @param velocity_boundary_cache Cached velocity conditions by boundary.
 * @throws std::invalid_argument if mesh, options, patches, or cache are invalid.
 */
template <TpetraTypePack Pack, class Policy, class MeshType>
TurbulenceWallTreatment<Pack, Policy, MeshType>::TurbulenceWallTreatment(
    SP<const mesh_type> mesh, TurbulenceWallTreatmentOptions options,
    const velocity_boundary_cache_type& velocity_boundary_cache)
    : d_mesh(std::move(mesh)), d_options(std::move(options))
{
    initialize(velocity_boundary_cache.type_by_name);
    turbulence_detail::collective_local_validation(
        *d_mesh, "Turbulence wall velocity-cache validation",
        [&]
        {
            if (velocity_boundary_cache.mesh.get() != d_mesh.get())
            {
                throw std::invalid_argument("Turbulence wall treatment received a "
                                            "velocity cache on the wrong mesh.");
            }
        });
}

/**
 * @brief Validate collective configuration and discover rank-local wall batches.
 * @tparam Pack Tpetra type pack used by the treatment.
 * @tparam Policy Wall-treatment policy tag.
 * @param boundary_types Velocity condition type indexed by boundary name.
 * @throws std::invalid_argument if configuration or mesh patches are invalid.
 */
template <TpetraTypePack Pack, class Policy, class MeshType>
void TurbulenceWallTreatment<Pack, Policy, MeshType>::initialize(
    const std::unordered_map<std::string, BoundaryConditionType>& boundary_types)
{
    static_assert(turbulence_wall_detail::supported_policy_v<Policy>,
                  "TurbulenceWallTreatment requires a supported policy tag.");
    if (!d_mesh)
    {
        throw std::invalid_argument("TurbulenceWallTreatment requires a non-null mesh.");
    }

    turbulence_detail::collective_local_validation(
        *d_mesh, "Turbulence wall option validation",
        [&] { validate_turbulence_wall_treatment_options(d_options); });
    turbulence_detail::require_uniform_integral(*d_mesh,
                                                static_cast<int>(d_options.boundary_names.size()),
                                                "Turbulence wall boundary-name count");
    for (size_t index = 0; index < d_options.boundary_names.size(); ++index)
    {
        turbulence_wall_detail::require_uniform_string(*d_mesh, d_options.boundary_names[index],
                                                       "Turbulence wall boundary name " +
                                                           std::to_string(index));
    }
    turbulence_detail::require_uniform_real(*d_mesh, d_options.c_mu, "Turbulence wall Cmu");
    turbulence_detail::require_uniform_real(*d_mesh, d_options.kappa, "Turbulence wall kappa");
    turbulence_detail::require_uniform_real(*d_mesh, d_options.log_layer_e, "Turbulence wall E");
    turbulence_detail::require_uniform_integral(
        *d_mesh, static_cast<int>(d_options.thermal_wall_law),
        "Turbulence thermal wall law");
    turbulence_detail::require_uniform_integral(
        *d_mesh, d_options.thermal_turbulent_prandtl_number.has_value() ? 1 : 0,
        "Turbulence thermal wall Prandtl presence");
    if (d_options.thermal_turbulent_prandtl_number.has_value())
    {
        turbulence_detail::require_uniform_real(
            *d_mesh, *d_options.thermal_turbulent_prandtl_number,
            "Turbulence thermal wall Prandtl number");
    }
    turbulence_detail::require_uniform_integral(
        *d_mesh, static_cast<int>(d_options.roughness_models.size()),
        "Turbulence wall roughness-model count");
    turbulence_detail::require_uniform_integral(
        *d_mesh, static_cast<int>(d_options.roughness_heights.size()),
        "Turbulence wall roughness-height count");
    turbulence_detail::require_uniform_integral(
        *d_mesh, static_cast<int>(d_options.roughness_constants.size()),
        "Turbulence wall roughness-constant count");
    for (size_t index = 0; index < d_options.roughness_models.size(); ++index)
    {
        turbulence_detail::require_uniform_integral(
            *d_mesh, static_cast<int>(d_options.roughness_models[index]),
            "Turbulence wall roughness model " + std::to_string(index));
        turbulence_detail::require_uniform_real(
            *d_mesh, d_options.roughness_heights[index],
            "Turbulence wall roughness height " + std::to_string(index));
        turbulence_detail::require_uniform_real(
            *d_mesh, d_options.roughness_constants[index],
            "Turbulence wall roughness constant " + std::to_string(index));
    }
    turbulence_detail::require_uniform_integral(*d_mesh,
                                                d_options.epsilon_low_re_correction ? 1 : 0,
                                                "Turbulence wall epsilon low-Re correction");
    turbulence_detail::require_uniform_real(*d_mesh, d_options.sst_beta_1, "Turbulence wall beta1");
    turbulence_detail::require_uniform_real(*d_mesh, d_options.sst_omega_wall_coefficient,
                                            "Turbulence wall omega coefficient");

    if constexpr (turbulence_wall_detail::resolved_wall_policy_v<Policy>)
    {
        turbulence_detail::collective_local_validation(
            *d_mesh, "Resolved wall-treatment compatibility",
            [&]
            {
                const auto has_unresolved_roughness =
                    std::any_of(d_options.roughness_models.begin(),
                                d_options.roughness_models.end(),
                                [](const auto model)
                                {
                                    return model !=
                                           TurbulenceWallRoughnessModel::Smooth;
                                }) ||
                    std::any_of(d_options.roughness_heights.begin(),
                                d_options.roughness_heights.end(),
                                [](const auto height) { return height != 0.0; });
                if (has_unresolved_roughness)
                {
                    throw std::invalid_argument(
                        "Resolved low-Re wall treatment does not accept an "
                        "unresolved rough-wall law.");
                }
                if (d_options.thermal_wall_law !=
                    TurbulenceThermalWallLaw::TurbulentPrandtl)
                {
                    throw std::invalid_argument(
                        "Resolved low-Re wall treatment requires molecular wall "
                        "heat transport.");
                }
            });
    }

    turbulence_detail::collective_local_validation(
        *d_mesh, "Turbulence wall boundary-condition validation",
        [&]
        {
            for (const auto& name : d_options.boundary_names)
            {
                const auto iter = boundary_types.find(name);
                if (iter == boundary_types.end() || iter->second != BoundaryConditionType::NoSlip)
                {
                    throw std::invalid_argument(
                        "Turbulence wall boundary '" + name +
                        "' must have a configured NoSlip velocity condition.");
                }
            }
        });

    for (size_t boundary_index = 0;
         boundary_index < d_options.boundary_names.size(); ++boundary_index)
    {
        const auto& name = d_options.boundary_names[boundary_index];
        const auto has_roughness = !d_options.roughness_models.empty();
        const auto roughness_model =
            has_roughness ? d_options.roughness_models[boundary_index]
                          : TurbulenceWallRoughnessModel::Smooth;
        const auto roughness_height =
            has_roughness ? d_options.roughness_heights[boundary_index] : 0.0;
        const auto roughness_constant =
            has_roughness ? d_options.roughness_constants[boundary_index] : 0.0;
        int local_found = 0;
        for (const auto& [batch_id, batch] : d_mesh->boundary_batches())
        {
            (void)batch;
            if (d_mesh->boundary_batch_name(batch_id) == name)
            {
                ++local_found;
                d_local_wall_batches.push_back(
                    {batch_id, name, roughness_model, roughness_height,
                     roughness_constant});
            }
        }
        int global_found = 0;
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1,
                           &local_found, &global_found);
        if (global_found == 0)
        {
            throw std::invalid_argument("Turbulence wall boundary '" + name +
                                        "' does not exist on the distributed mesh.");
        }
    }
    d_y_plus_lam = openfoam_y_plus_lam(d_options.kappa, d_options.log_layer_e);
}

/**
 * @brief Validate mesh identity and completeness of cached wall velocities.
 * @tparam Pack Tpetra type pack used by the treatment.
 * @tparam Policy Wall-treatment policy tag.
 * @param velocity_boundary_cache Boundary cache to validate.
 * @throws std::invalid_argument if a selected wall cache is invalid.
 */
template <TpetraTypePack Pack, class Policy, class MeshType>
void TurbulenceWallTreatment<Pack, Policy, MeshType>::validate_velocity_cache(
    const velocity_boundary_cache_type& velocity_boundary_cache) const
{
    turbulence_detail::collective_local_validation(
        *d_mesh, "Turbulence wall velocity-cache validation",
        [&]
        {
            if (velocity_boundary_cache.mesh.get() != d_mesh.get())
            {
                throw std::invalid_argument("Turbulence wall treatment received a "
                                            "velocity cache on the wrong mesh.");
            }
            for (const auto& wall_batch : d_local_wall_batches)
            {
                const auto& batch = d_mesh->boundary_batches().at(wall_batch.id);
                const auto type = velocity_boundary_cache.type.find(wall_batch.id);
                const auto values = velocity_boundary_cache.value.find(wall_batch.id);
                if (type == velocity_boundary_cache.type.end() ||
                    type->second != BoundaryConditionType::NoSlip ||
                    values == velocity_boundary_cache.value.end() ||
                    values->second.size() != batch.face_lids.size())
                {
                    throw std::invalid_argument("Turbulence wall boundary '" + wall_batch.name +
                                                "' is not a complete NoSlip velocity cache batch.");
                }
                for (const auto& value : values->second)
                {
                    if (!turbulence_wall_detail::finite_vector(value))
                    {
                        throw std::invalid_argument("Turbulence wall velocity cache "
                                                    "contains a non-finite value.");
                    }
                }
            }
        });
}

/**
 * @brief Evaluate and aggregate all configured local wall faces.
 * @tparam Pack Tpetra type pack used by the treatment.
 * @tparam Policy Wall-treatment policy tag.
 * @param turbulent_kinetic_energy Accepted k field.
 * @param velocity Accepted liquid velocity field.
 * @param velocity_boundary_cache Cached no-slip wall velocities.
 * @param material Physical material-property fields.
 * @param reference_density Momentum reference density.
 * @param turbulent_prandtl_number Turbulent Prandtl number for heat flux.
 * @return Independent staged wall evaluation.
 * @throws std::invalid_argument if a field, cache, or physical input is invalid.
 * @throws std::overflow_error if a wall formula produces invalid data.
 */
template <TpetraTypePack Pack, class Policy, class MeshType>
auto TurbulenceWallTreatment<Pack, Policy, MeshType>::evaluate(
    const field_type& turbulent_kinetic_energy, const velocity_field_type& velocity,
    const velocity_boundary_cache_type& velocity_boundary_cache, const material_type& material,
    scalar_type reference_density, scalar_type turbulent_prandtl_number,
    const Evaluation* accepted_evaluation) const -> Evaluation
{
    turbulence_detail::require_uniform_real(*d_mesh, static_cast<real_t>(reference_density),
                                            "Turbulence wall reference density");
    turbulence_detail::require_uniform_real(*d_mesh, static_cast<real_t>(turbulent_prandtl_number),
                                            "Turbulence wall turbulent Prandtl number");
    validate_velocity_cache(velocity_boundary_cache);

    turbulence_detail::collective_local_validation(
        *d_mesh, "Turbulence wall input validation",
        [&]
        {
            const field_type* scalar_fields[] = {
                &turbulent_kinetic_energy, &material.density, &material.specific_heat_capacity,
                &material.dynamic_viscosity, &material.thermal_conductivity};
            for (const auto* field : scalar_fields)
            {
                if (&field->mesh() != d_mesh.get())
                {
                    throw std::invalid_argument("Turbulence wall treatment field mesh mismatch.");
                }
            }
            if (&velocity.mesh() != d_mesh.get())
            {
                throw std::invalid_argument("Turbulence wall treatment velocity mesh mismatch.");
            }
            if (accepted_evaluation != nullptr &&
                accepted_evaluation->mesh_ptr().get() != d_mesh.get())
            {
                throw std::invalid_argument(
                    "Accepted turbulence wall evaluation mesh mismatch.");
            }
            if (!std::isfinite(reference_density) || reference_density <= scalar_type{} ||
                !std::isfinite(turbulent_prandtl_number) ||
                turbulent_prandtl_number <= scalar_type{})
            {
                throw std::invalid_argument("Turbulence wall treatment requires "
                                            "positive finite density and Prt.");
            }
        });

    Evaluation result(d_mesh);
    Arr<size_t> face_counts(d_mesh->num_owned_cells(), 0);
    Arr<scalar_type> secondary_sums(d_mesh->num_owned_cells(), scalar_type{});
    Arr<scalar_type> production_sums(d_mesh->num_owned_cells(), scalar_type{});
    Arr<scalar_type> combined_wall_distances(
        d_mesh->num_local_cells(),
        std::numeric_limits<scalar_type>::infinity());
    const auto k_values = turbulent_kinetic_energy.local_read_view();
    const auto density_values = material.density.local_read_view();
    const auto heat_capacity_values =
        material.specific_heat_capacity.local_read_view();
    const auto molecular_viscosity_values =
        material.dynamic_viscosity.local_read_view();
    const auto molecular_conductivity_values =
        material.thermal_conductivity.local_read_view();
    const auto velocity_values = velocity.local_read_view();

    turbulence_detail::collective_local_validation(
        *d_mesh, "Turbulence wall face evaluation",
        [&]
        {
            if constexpr (
                std::is_same_v<Policy,
                               StandardHighReKEpsilonWallPolicy>)
            {
                // OpenFOAM's nearWallDist searches one combined wall patch.
                // For a cell incident on multiple configured wall faces this
                // makes every face use the nearest incident-wall distance.
                for (const auto& wall_batch : d_local_wall_batches)
                {
                    const auto& batch =
                        d_mesh->boundary_batches().at(wall_batch.id);
                    for (const auto face_lid : batch.face_lids)
                    {
                        if (!d_mesh->is_boundary_face(face_lid))
                        {
                            throw std::invalid_argument(
                                "Configured turbulence wall batch contains "
                                "a non-boundary face.");
                        }
                        const auto owner = d_mesh->owner_cell(face_lid);
                        const auto distance =
                            static_cast<scalar_type>(
                                FVM::detail::boundary_normal_distance(
                                    *d_mesh, face_lid, owner));
                        if (!std::isfinite(distance)
                            || distance <= scalar_type{})
                        {
                            throw std::invalid_argument(
                                "Turbulence wall treatment requires a "
                                "finite positive wall distance.");
                        }
                        auto& combined_distance =
                            combined_wall_distances.at(
                                static_cast<size_t>(owner));
                        combined_distance =
                            std::min(combined_distance, distance);
                    }
                }
            }

            for (const auto& wall_batch : d_local_wall_batches)
            {
                const auto& batch = d_mesh->boundary_batches().at(wall_batch.id);
                auto& face_results = result.d_faces[wall_batch.id];
                face_results.resize(batch.face_lids.size());
                auto& dynamic_viscosities =
                    result.d_boundary_dynamic_viscosity.value[wall_batch.id];
                auto& thermal_conductivities =
                    result.d_boundary_thermal_conductivity.value[wall_batch.id];
                auto& scalar_diffusivities =
                    result.d_boundary_scalar_diffusivity.value[wall_batch.id];
                dynamic_viscosities.resize(batch.face_lids.size());
                thermal_conductivities.resize(batch.face_lids.size());
                scalar_diffusivities.resize(batch.face_lids.size());

                for (size_t in_batch_id = 0; in_batch_id < batch.face_lids.size(); ++in_batch_id)
                {
                    const auto face_lid = batch.face_lids[in_batch_id];
                    if (!d_mesh->is_boundary_face(face_lid))
                    {
                        throw std::invalid_argument("Configured turbulence wall batch "
                                                    "contains a non-boundary face.");
                    }
                    const auto owner = d_mesh->owner_cell(face_lid);
                    auto distance = static_cast<scalar_type>(
                        FVM::detail::boundary_normal_distance(
                            *d_mesh, face_lid, owner));
                    if constexpr (
                        std::is_same_v<
                            Policy,
                            StandardHighReKEpsilonWallPolicy>)
                    {
                        distance = combined_wall_distances.at(
                            static_cast<size_t>(owner));
                    }
                    const auto k = k_values(owner, 0);
                    const auto density = density_values(owner, 0);
                    const auto heat_capacity =
                        heat_capacity_values(owner, 0);
                    const auto molecular_viscosity =
                        molecular_viscosity_values(owner, 0);
                    const auto molecular_conductivity =
                        molecular_conductivity_values(owner, 0);
                    const typename velocity_field_type::vec_type
                        owner_velocity{velocity_values(owner, 0),
                                       velocity_values(owner, 1),
                                       velocity_values(owner, 2)};
                    const auto wall_velocity =
                        velocity_boundary_cache.value.at(wall_batch.id).at(in_batch_id);
                    const auto outward_normal = d_mesh->face_normal_outward(face_lid, owner);

                    if (!std::isfinite(distance) || distance <= scalar_type{} ||
                        !std::isfinite(k) || k < scalar_type{} || !std::isfinite(density) ||
                        density <= scalar_type{} || !std::isfinite(heat_capacity) ||
                        heat_capacity <= scalar_type{} || !std::isfinite(molecular_viscosity) ||
                        molecular_viscosity <= scalar_type{} ||
                        !std::isfinite(molecular_conductivity) ||
                        molecular_conductivity < scalar_type{} ||
                        !turbulence_wall_detail::finite_vector(owner_velocity) ||
                        !turbulence_wall_detail::finite_vector(wall_velocity) ||
                        !turbulence_wall_detail::finite_vector(outward_normal))
                    {
                        throw std::invalid_argument(
                            "Turbulence wall treatment requires finite physical wall "
                            "inputs, "
                            "positive distance/nu/rho/Cp, and non-negative k/lambda.");
                    }

                    const auto molecular_nu = molecular_viscosity / reference_density;
                    turbulence_wall_detail::FaceInputs<scalar_type,
                                                       typename velocity_field_type::vec_type>
                        inputs{k,
                               distance,
                               molecular_nu,
                               owner_velocity,
                               wall_velocity,
                               outward_normal};

                    face_evaluation_type face_result;
                    if constexpr (std::is_same_v<Policy, ResolvedLowReSSTWallPolicy>)
                    {
                        face_result =
                            turbulence_wall_detail::evaluate_resolved_sst_face(inputs, d_options);
                    }
                    else if constexpr (
                        std::is_same_v<Policy, ResolvedLowReKEpsilonWallPolicy>)
                    {
                        face_result =
                            turbulence_wall_detail::evaluate_resolved_k_epsilon_face(inputs);
                    }
                    else
                    {
                        scalar_type accepted_turbulent_nu{};
                        if (accepted_evaluation != nullptr &&
                            accepted_evaluation->contains_face(wall_batch.id, in_batch_id))
                        {
                            accepted_turbulent_nu =
                                accepted_evaluation->face(wall_batch.id, in_batch_id)
                                    .turbulent_kinematic_viscosity;
                        }
                        face_result = turbulence_wall_detail::evaluate_standard_k_epsilon_face(
                            inputs, d_options, static_cast<scalar_type>(d_y_plus_lam),
                            {wall_batch.roughness_model,
                             static_cast<scalar_type>(wall_batch.roughness_height),
                             static_cast<scalar_type>(wall_batch.roughness_constant),
                             accepted_turbulent_nu});
                    }

                    const auto wall_prandtl = static_cast<scalar_type>(
                        d_options.thermal_turbulent_prandtl_number.value_or(
                            static_cast<real_t>(turbulent_prandtl_number)));
                    if constexpr (std::is_same_v<Policy,
                                                 StandardHighReKEpsilonWallPolicy>)
                    {
                        if (d_options.thermal_wall_law ==
                            TurbulenceThermalWallLaw::Jayatilleke)
                        {
                            if (molecular_conductivity <= scalar_type{})
                            {
                                throw std::invalid_argument(
                                    "Jayatilleke wall treatment requires positive "
                                    "molecular thermal conductivity.");
                            }
                            const auto molecular_prandtl =
                                molecular_viscosity * heat_capacity /
                                molecular_conductivity;
                            const auto prandtl_ratio =
                                molecular_prandtl / wall_prandtl;
                            face_result.jayatilleke_p =
                                turbulence_wall_detail::jayatilleke_p(prandtl_ratio);
                            face_result.thermal_y_plus_transition =
                                turbulence_wall_detail::jayatilleke_y_plus_thermal(
                                    face_result.jayatilleke_p, prandtl_ratio,
                                    static_cast<scalar_type>(d_options.kappa),
                                    static_cast<scalar_type>(d_options.log_layer_e));
                            if (face_result.y_plus >
                                face_result.thermal_y_plus_transition)
                            {
                                const auto temperature_plus =
                                    wall_prandtl *
                                    (std::log(std::max(
                                         static_cast<scalar_type>(
                                             d_options.log_layer_e) *
                                             face_result.y_plus,
                                         scalar_type{1.0001})) /
                                         static_cast<scalar_type>(d_options.kappa) +
                                     face_result.jayatilleke_p);
                                face_result.turbulent_thermal_diffusivity =
                                    molecular_nu *
                                    std::max(face_result.y_plus / temperature_plus -
                                                 scalar_type{1} /
                                                     molecular_prandtl,
                                             scalar_type{});
                            }
                        }
                        else
                        {
                            face_result.turbulent_thermal_diffusivity =
                                face_result.turbulent_kinematic_viscosity /
                                wall_prandtl;
                        }
                    }

                    face_result.effective_dynamic_viscosity =
                        molecular_viscosity +
                        reference_density * face_result.turbulent_kinematic_viscosity;
                    face_result.effective_thermal_conductivity =
                        molecular_conductivity +
                        density * heat_capacity *
                            face_result.turbulent_thermal_diffusivity;

                    const scalar_type finite_values[] = {
                        face_result.wall_distance,
                        face_result.y_plus,
                        face_result.roughness_height,
                        face_result.roughness_constant,
                        face_result.roughness_height_plus,
                        face_result.effective_log_layer_e,
                        face_result.turbulent_kinematic_viscosity,
                        face_result.turbulent_thermal_diffusivity,
                        face_result.thermal_y_plus_transition,
                        face_result.effective_dynamic_viscosity,
                        face_result.effective_thermal_conductivity,
                        static_cast<scalar_type>(face_result.turbulent_kinetic_energy.value),
                        static_cast<scalar_type>(face_result.secondary.value)};
                    for (const auto value : finite_values)
                    {
                        if (!std::isfinite(value) || value < scalar_type{})
                        {
                            throw std::overflow_error("Turbulence wall treatment produced "
                                                      "a non-finite or negative value.");
                        }
                    }
                    if (!std::isfinite(face_result.jayatilleke_p))
                    {
                        throw std::overflow_error(
                            "Turbulence wall treatment produced a non-finite "
                            "Jayatilleke offset.");
                    }
                    if ((face_result.secondary_constraint.has_value() &&
                         (!std::isfinite(*face_result.secondary_constraint) ||
                          *face_result.secondary_constraint < scalar_type{})) ||
                        (face_result.production_override.has_value() &&
                         (!std::isfinite(*face_result.production_override) ||
                          *face_result.production_override < scalar_type{})))
                    {
                        throw std::overflow_error("Turbulence wall treatment produced an "
                                                  "invalid cell contribution.");
                    }

                    face_results[in_batch_id] = face_result;
                    dynamic_viscosities[in_batch_id] = face_result.effective_dynamic_viscosity;
                    thermal_conductivities[in_batch_id] =
                        face_result.effective_thermal_conductivity;
                    scalar_diffusivities[in_batch_id] = molecular_nu;

                    if (!d_mesh->is_owned_cell(owner))
                    {
                        continue;
                    }
                    const auto owned = static_cast<size_t>(owner);
                    ++face_counts[owned];
                    auto& cell_y_plus = result.d_cell_y_plus[owned];
                    cell_y_plus = cell_y_plus.has_value()
                                      ? std::max(*cell_y_plus, face_result.y_plus)
                                      : face_result.y_plus;
                    if (face_result.secondary_constraint.has_value())
                    {
                        secondary_sums[owned] += *face_result.secondary_constraint;
                    }
                    if (face_result.production_override.has_value())
                    {
                        production_sums[owned] += *face_result.production_override;
                    }
                }
            }

            if constexpr (turbulence_wall_detail::epsilon_wall_policy_v<Policy>)
            {
                for (size_t owned = 0; owned < face_counts.size(); ++owned)
                {
                    if (face_counts[owned] == 0)
                    {
                        continue;
                    }
                    const auto inverse_count =
                        scalar_type{1} / static_cast<scalar_type>(face_counts[owned]);
                    result.d_secondary_constraints[owned] = secondary_sums[owned] * inverse_count;
                    result.d_production_overrides[owned] = production_sums[owned] * inverse_count;
                }
            }
        });

    return result;
}

} // namespace SimpleFluid
