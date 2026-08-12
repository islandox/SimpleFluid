/**
 * @file TurbulenceModel.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Template implementation of runtime two-equation turbulence coupling.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "TurbulenceCollectiveValidation.hh"
#include "TurbulenceModel.hh"

#include "FVM/CellOperators.hh"
#include "fields/TensorCellField.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace SimpleFluid
{

namespace turbulence_detail
{

template<TpetraTypePack Pack,
         class BoundaryConditionProvider,
         class BoundaryValueProvider>
void reconstruct_gradient(
    FVM::CellGradientScheme scheme,
    const CellField<Pack>& field,
    BoundaryConditionProvider boundary_condition,
    BoundaryValueProvider boundary_value,
    VectorCellField<Pack>& gradient,
    const FVM::CellGradientCache<Pack>& cache)
{
    if (scheme == FVM::CellGradientScheme::GaussLinear)
    {
        FVM::gauss_linear_cell_gradient(
            field, boundary_condition, boundary_value, gradient);
        return;
    }
    FVM::cell_gradient(
        field, boundary_condition, boundary_value, gradient, cache);
}

template<TpetraTypePack Pack>
void reconstruct_gradient(
    FVM::CellGradientScheme scheme,
    const CellField<Pack>& field,
    const BoundaryConditionMap& boundary_conditions,
    VectorCellField<Pack>& gradient,
    const FVM::CellGradientCache<Pack>& cache)
{
    if (scheme == FVM::CellGradientScheme::GaussLinear)
    {
        FVM::gauss_linear_cell_gradient(
            field, boundary_conditions, gradient);
        return;
    }
    FVM::cell_gradient(
        field, boundary_conditions, gradient, cache);
}

template<TpetraTypePack Pack>
void reconstruct_gradient(
    FVM::CellGradientScheme scheme,
    const CellField<Pack>& field,
    VectorCellField<Pack>& gradient,
    const FVM::CellGradientCache<Pack>& cache)
{
    if (scheme == FVM::CellGradientScheme::GaussLinear)
    {
        FVM::gauss_linear_cell_gradient(field, gradient);
        return;
    }
    FVM::cell_gradient(field, gradient, cache);
}

template<TpetraTypePack Pack, class BoundaryValueProvider>
void reconstruct_gradient(
    FVM::CellGradientScheme scheme,
    const VectorCellField<Pack>& field,
    BoundaryValueProvider boundary_value,
    TensorCellField<Pack>& gradient,
    const FVM::CellGradientCache<Pack>& cache)
{
    if (scheme == FVM::CellGradientScheme::GaussLinear)
    {
        FVM::gauss_linear_cell_gradient(
            field, boundary_value, gradient);
        return;
    }
    FVM::cell_gradient(
        field, boundary_value, gradient, cache);
}

} // namespace turbulence_detail

/**
 * @brief Owns all fields, closures, equations, and staged wall data.
 * @tparam Pack Tpetra type pack used by the enclosing model.
 */
template <TpetraTypePack Pack>
struct SIMPLEFLUID_EQUATIONS_LOCAL TurbulenceModel<Pack>::State
{
    using closure_type =
        std::variant<StandardKEpsilonEquation, RNGKEpsilonEquation, RealizableKEpsilonEquation,
                     StandardKOmegaEquation, BSLKOmegaEquation, SSTKOmegaEquation>;
    using resolved_sst_wall_type = ResolvedLowReSSTWallTreatment<Pack>;
    using resolved_k_epsilon_wall_type =
        ResolvedLowReKEpsilonWallTreatment<Pack>;
    using high_re_wall_type = StandardHighReKEpsilonWallTreatment<Pack>;
    using wall_treatment_type =
        std::variant<std::monostate, resolved_sst_wall_type,
                     resolved_k_epsilon_wall_type, high_re_wall_type>;
    using wall_evaluation_type =
        std::variant<std::monostate,
                     typename resolved_sst_wall_type::Evaluation,
                     typename resolved_k_epsilon_wall_type::Evaluation,
                     typename high_re_wall_type::Evaluation>;
    struct wall_publication_type
    {
        wall_evaluation_type evaluation;
        Arr<WallYPlusStatistics> statistics;
    };

    State(SP<const mesh_type> mesh, const TurbulenceBoundaryConditionSet& boundary_conditions,
          const VectorBoundaryConditionMap& velocity_boundary_conditions,
          const TurbulenceModelOptions& options)
        : epsilon_family(options.model == TurbulenceModelType::StandardKEpsilon ||
                         options.model == TurbulenceModelType::RNGKEpsilon ||
                         options.model == TurbulenceModelType::RealizableKEpsilon),
          menter_family(options.model == TurbulenceModelType::BSLKOmega ||
                        options.model == TurbulenceModelType::SSTKOmega),
          wall_boundary_names(options.wall_options.boundary_names),
          closure(make_closure(options)),
          wall_treatment(make_wall_treatment(mesh, options, velocity_boundary_conditions)),
          gradient_cache(mesh),
          k(mesh, static_cast<scalar_type>(options.initial_turbulent_kinetic_energy), "k"),
          secondary(mesh,
                    static_cast<scalar_type>(epsilon_family
                                                 ? options.initial_dissipation_rate
                                                 : options.initial_specific_dissipation_rate),
                    epsilon_family ? "epsilon" : "omega"),
          candidate_k(mesh, static_cast<scalar_type>(options.initial_turbulent_kinetic_energy),
                      "k_candidate"),
          candidate_secondary(
              mesh,
              static_cast<scalar_type>(epsilon_family ? options.initial_dissipation_rate
                                                      : options.initial_specific_dissipation_rate),
              "turbulence_secondary_candidate"),
          nu_t(mesh, "nu_t"), candidate_nu_t(mesh, "nu_t_candidate"),
          effective_dynamic_viscosity(mesh, "mu_eff"),
          candidate_effective_dynamic_viscosity(mesh, "mu_eff_candidate"),
          effective_thermal_conductivity(mesh, "lambda_eff"),
          candidate_effective_thermal_conductivity(mesh, "lambda_eff_candidate"),
          k_diffusivity(mesh, "k_effective_diffusivity"),
          secondary_diffusivity(mesh, "turbulence_secondary_effective_diffusivity"),
          k_source(mesh, "k_explicit_source"), k_sink(mesh, "k_implicit_sink"),
          secondary_source(mesh, "turbulence_secondary_explicit_source"),
          secondary_sink(mesh, "turbulence_secondary_implicit_sink"),
          buoyancy_production(mesh, scalar_type{},
                              "buoyancy_production"),
          candidate_buoyancy_production(
              mesh, scalar_type{}, "buoyancy_production_candidate"),
          wall_distance(mesh, static_cast<scalar_type>(options.initial_wall_distance.value_or(1.0)),
                        "wall_distance"),
          wall_y_plus(mesh, scalar_type{}, "wall_y_plus"),
          candidate_wall_y_plus(mesh, scalar_type{},
                                "wall_y_plus_candidate"),
          wall_velocity(mesh, "turbulence_wall_velocity"),
          candidate_wall_velocity(
              mesh, "turbulence_wall_velocity_candidate"),
          velocity_gradient(mesh, "velocity_gradient"),
          candidate_velocity_gradient(
              mesh, "velocity_gradient_candidate"),
          buoyancy_gradient(mesh, "turbulence_buoyancy_gradient"),
          k_gradient(mesh, "k_gradient"),
          secondary_gradient(mesh, "turbulence_secondary_gradient"),
          candidate_k_gradient(mesh, "k_candidate_gradient"),
          candidate_secondary_gradient(mesh, "turbulence_secondary_candidate_gradient"),
          k_equation(mesh, boundary_conditions.turbulent_kinetic_energy),
          secondary_equation(mesh, epsilon_family ? boundary_conditions.dissipation_rate
                                                  : boundary_conditions.specific_dissipation_rate)
    {
        const auto initial_k = options.initial_turbulent_kinetic_energy;
        const auto initial_secondary = epsilon_family ? options.initial_dissipation_rate
                                                      : options.initial_specific_dissipation_rate;
        real_t initial_nu_t{};
        switch (options.model)
        {
        case TurbulenceModelType::StandardKEpsilon:
            initial_nu_t =
                std::get<StandardKEpsilonEquation>(closure).turbulent_kinematic_viscosity(
                    {initial_k, initial_secondary});
            break;
        case TurbulenceModelType::RNGKEpsilon:
            initial_nu_t = std::get<RNGKEpsilonEquation>(closure).turbulent_kinematic_viscosity(
                {initial_k, initial_secondary});
            break;
        case TurbulenceModelType::RealizableKEpsilon:
            initial_nu_t =
                std::get<RealizableKEpsilonEquation>(closure).turbulent_kinematic_viscosity(
                    {initial_k, initial_secondary}, {});
            break;
        case TurbulenceModelType::StandardKOmega:
            initial_nu_t = std::get<StandardKOmegaEquation>(closure).turbulent_kinematic_viscosity(
                {initial_k, initial_secondary});
            break;
        case TurbulenceModelType::BSLKOmega:
            initial_nu_t = std::get<BSLKOmegaEquation>(closure).turbulent_kinematic_viscosity(
                {initial_k, initial_secondary});
            break;
        case TurbulenceModelType::SSTKOmega:
            initial_nu_t = std::get<SSTKOmegaEquation>(closure).turbulent_kinematic_viscosity(
                {initial_k, initial_secondary},
                {0.0, options.initial_wall_distance.value_or(1.0), 0.0, 0.0});
            break;
        case TurbulenceModelType::Laminar:
            throw std::logic_error("Laminar mode cannot allocate turbulence state.");
        }
        nu_t.put_scalar(static_cast<scalar_type>(initial_nu_t));
        candidate_nu_t.put_scalar(static_cast<scalar_type>(initial_nu_t));

        output_fields = {{"k", &k},
                         {epsilon_family ? "epsilon" : "omega", &secondary},
                         {"nu_t", &nu_t},
                         {"mu_eff", &effective_dynamic_viscosity},
                         {"lambda_eff", &effective_thermal_conductivity}};
        if (options.wall_treatment != TurbulenceWallTreatmentType::None)
        {
            output_fields.emplace("wall_y_plus", &wall_y_plus);
        }
        if (menter_family)
        {
            output_fields.emplace("wall_distance", &wall_distance);
        }
        if (options.buoyancy_model != TurbulenceBuoyancyModel::None)
        {
            output_fields.emplace(
                "buoyancy_production", &buoyancy_production);
        }
    }

    static closure_type make_closure(const TurbulenceModelOptions& options)
    {
        switch (options.model)
        {
        case TurbulenceModelType::StandardKEpsilon:
        {
            auto coefficients = StandardKEpsilonEquation::Coefficients{};
            if (options.wall_treatment ==
                    TurbulenceWallTreatmentType::StandardHighReKEpsilon ||
                options.wall_treatment ==
                    TurbulenceWallTreatmentType::ResolvedLowReKEpsilon)
            {
                coefficients.c_mu = options.wall_options.c_mu;
            }
            return StandardKEpsilonEquation{coefficients};
        }
        case TurbulenceModelType::RNGKEpsilon:
            return RNGKEpsilonEquation{};
        case TurbulenceModelType::RealizableKEpsilon:
            return RealizableKEpsilonEquation{};
        case TurbulenceModelType::StandardKOmega:
            return StandardKOmegaEquation{};
        case TurbulenceModelType::BSLKOmega:
            return BSLKOmegaEquation{};
        case TurbulenceModelType::SSTKOmega:
        {
            auto coefficients = SSTKOmegaEquation::Coefficients{};
            if (options.wall_treatment ==
                TurbulenceWallTreatmentType::ResolvedLowReSST)
            {
                coefficients.beta_1 = options.wall_options.sst_beta_1;
                coefficients.kappa = options.wall_options.kappa;
            }
            return SSTKOmegaEquation{coefficients};
        }
        case TurbulenceModelType::Laminar:
            break;
        }
        throw std::logic_error("Laminar mode does not have a turbulence closure.");
    }

    static wall_treatment_type make_wall_treatment(
        const SP<const mesh_type>& mesh, const TurbulenceModelOptions& options,
        const VectorBoundaryConditionMap& velocity_boundary_conditions)
    {
        switch (options.wall_treatment)
        {
        case TurbulenceWallTreatmentType::None:
            return std::monostate{};
        case TurbulenceWallTreatmentType::ResolvedLowReSST:
            return resolved_sst_wall_type(
                mesh, options.wall_options, velocity_boundary_conditions);
        case TurbulenceWallTreatmentType::StandardHighReKEpsilon:
            return high_re_wall_type(mesh, options.wall_options,
                                     velocity_boundary_conditions);
        case TurbulenceWallTreatmentType::ResolvedLowReKEpsilon:
            return resolved_k_epsilon_wall_type(
                mesh, options.wall_options, velocity_boundary_conditions);
        }
        throw std::logic_error("Unknown turbulence wall-treatment type.");
    }

    wall_evaluation_type evaluate_wall(
        const field_type& k_field, const velocity_field_type& velocity,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        const material_type& material, scalar_type reference_density,
        scalar_type turbulent_prandtl_number,
        const wall_evaluation_type* accepted_evaluation = nullptr) const
    {
        wall_evaluation_type result;
        std::visit(
            [&](const auto& treatment)
            {
                using treatment_type = std::remove_cvref_t<decltype(treatment)>;
                if constexpr (!std::is_same_v<treatment_type, std::monostate>)
                {
                    using evaluation_type = typename treatment_type::Evaluation;
                    const auto* accepted =
                        accepted_evaluation != nullptr
                            ? std::get_if<evaluation_type>(accepted_evaluation)
                            : nullptr;
                    result = treatment.evaluate(k_field, velocity,
                                                velocity_boundary_cache, material,
                                                reference_density,
                                                turbulent_prandtl_number,
                                                accepted);
                }
            },
            wall_treatment);
        return result;
    }

    static std::optional<BoundaryCondition> wall_boundary_condition(
        const wall_evaluation_type& evaluation, int batch_id,
        size_t in_batch_id, bool k_field)
    {
        return std::visit(
            [&](const auto& values) -> std::optional<BoundaryCondition>
            {
                using evaluation_type = std::remove_cvref_t<decltype(values)>;
                if constexpr (std::is_same_v<evaluation_type, std::monostate>)
                {
                    return std::nullopt;
                }
                else
                {
                    if (!values.contains_face(batch_id, in_batch_id))
                    {
                        return std::nullopt;
                    }
                    const auto& face = values.face(batch_id, in_batch_id);
                    return k_field ? face.turbulent_kinetic_energy
                                   : face.secondary;
                }
            },
            evaluation);
    }

    static std::optional<scalar_type> wall_secondary_constraint(
        const wall_evaluation_type& evaluation, local_ordinal_type cell_lid)
    {
        return std::visit(
            [&](const auto& values) -> std::optional<scalar_type>
            {
                using evaluation_type = std::remove_cvref_t<decltype(values)>;
                if constexpr (std::is_same_v<evaluation_type, std::monostate>)
                    return std::nullopt;
                else
                    return values.secondary_constraint(cell_lid);
            },
            evaluation);
    }

    static std::optional<scalar_type> wall_production_override(
        const wall_evaluation_type& evaluation, local_ordinal_type cell_lid)
    {
        return std::visit(
            [&](const auto& values) -> std::optional<scalar_type>
            {
                using evaluation_type = std::remove_cvref_t<decltype(values)>;
                if constexpr (std::is_same_v<evaluation_type, std::monostate>)
                    return std::nullopt;
                else
                    return values.production_override(cell_lid);
            },
            evaluation);
    }

    static const FVM::BoundaryCache<Pack>* boundary_dynamic_viscosity(
        const wall_evaluation_type& evaluation) noexcept
    {
        return std::visit(
            [](const auto& values) -> const FVM::BoundaryCache<Pack>*
            {
                using evaluation_type = std::remove_cvref_t<decltype(values)>;
                if constexpr (std::is_same_v<evaluation_type, std::monostate>)
                    return nullptr;
                else
                    return &values.boundary_dynamic_viscosity();
            },
            evaluation);
    }

    static const FVM::BoundaryCache<Pack>* boundary_thermal_conductivity(
        const wall_evaluation_type& evaluation) noexcept
    {
        return std::visit(
            [](const auto& values) -> const FVM::BoundaryCache<Pack>*
            {
                using evaluation_type = std::remove_cvref_t<decltype(values)>;
                if constexpr (std::is_same_v<evaluation_type, std::monostate>)
                    return nullptr;
                else
                    return &values.boundary_thermal_conductivity();
            },
            evaluation);
    }

    static const FVM::BoundaryCache<Pack>* boundary_scalar_diffusivity(
        const wall_evaluation_type& evaluation) noexcept
    {
        return std::visit(
            [](const auto& values) -> const FVM::BoundaryCache<Pack>*
            {
                using evaluation_type = std::remove_cvref_t<decltype(values)>;
                if constexpr (std::is_same_v<evaluation_type, std::monostate>)
                    return nullptr;
                else
                    return &values.boundary_scalar_diffusivity();
            },
            evaluation);
    }

    wall_publication_type
    stage_wall_publication(wall_evaluation_type evaluation)
    {
        WallYPlusSamplesByPatch local_samples;
        Arr<WallYPlusStatistics> statistics;
        if (!wall_boundary_names.empty())
        {
            std::visit(
                [&](const auto& values)
                {
                    using evaluation_type =
                        std::remove_cvref_t<decltype(values)>;
                    if constexpr (!std::is_same_v<
                                      evaluation_type,
                                      std::monostate>)
                    {
                        for (const auto& [batch_id, batch] :
                             wall_y_plus.mesh().boundary_batches())
                        {
                            const auto& name =
                                wall_y_plus.mesh()
                                    .boundary_batch_name(batch_id);
                            if (std::find(
                                    wall_boundary_names.begin(),
                                    wall_boundary_names.end(),
                                    name)
                                == wall_boundary_names.end())
                            {
                                continue;
                            }
                            auto& samples = local_samples[name];
                            for (size_t in_batch_id = 0;
                                 in_batch_id < batch.face_lids.size();
                                 ++in_batch_id)
                            {
                                const auto face_lid =
                                    batch.face_lids[in_batch_id];
                                if (!wall_y_plus.mesh().is_owned_face(
                                        face_lid)
                                    || !values.contains_face(
                                        batch_id, in_batch_id))
                                {
                                    continue;
                                }
                                samples.push_back(
                                    {static_cast<real_t>(
                                         values.face(
                                             batch_id,
                                             in_batch_id)
                                             .y_plus),
                                     wall_y_plus.mesh().face_area(
                                         face_lid)});
                            }
                        }
                    }
                },
                evaluation);
            statistics =
                reduce_wall_y_plus_statistics(
                    *wall_y_plus.mesh()
                         .owned_cell_map()
                         ->getComm(),
                    std::span<const std::string>{
                        wall_boundary_names},
                    local_samples);
        }
        {
            auto candidate_values =
                candidate_wall_y_plus.owned_write_view();
            for (size_t owned = 0;
                 owned < candidate_wall_y_plus.num_owned_cells(); ++owned)
            {
                const auto cell_lid =
                    static_cast<local_ordinal_type>(owned);
                candidate_values(cell_lid, 0) = std::visit(
                    [&](const auto& values) -> scalar_type
                    {
                        using evaluation_type =
                            std::remove_cvref_t<decltype(values)>;
                        if constexpr (std::is_same_v<
                                          evaluation_type,
                                          std::monostate>)
                            return scalar_type{};
                        else
                            return values.cell_y_plus(cell_lid)
                                .value_or(scalar_type{});
                    },
                    evaluation);
            }
        }
        candidate_wall_y_plus.sync_ghosts();
        return {std::move(evaluation), std::move(statistics)};
    }

    template<class Field>
    static void publish_synced_field(
        Field& accepted, const Field& candidate) noexcept
    {
        accepted.owned_data().update(
            scalar_type{1}, candidate.owned_data(), scalar_type{0});
        accepted.overlap_data().update(
            scalar_type{1}, candidate.overlap_data(), scalar_type{0});
    }

    void publish_wall_publication(
        wall_publication_type publication) noexcept
    {
        publish_synced_field(wall_y_plus, candidate_wall_y_plus);
        wall_statistics = std::move(publication.statistics);
        wall_evaluation = std::move(publication.evaluation);
    }

    bool epsilon_family;
    bool menter_family;
    ArrString wall_boundary_names;
    Arr<WallYPlusStatistics> wall_statistics;
    closure_type closure;
    wall_treatment_type wall_treatment;
    wall_evaluation_type wall_evaluation;
    FVM::CellGradientCache<Pack> gradient_cache;
    field_type k;
    field_type secondary;
    field_type candidate_k;
    field_type candidate_secondary;
    field_type nu_t;
    field_type candidate_nu_t;
    field_type effective_dynamic_viscosity;
    field_type candidate_effective_dynamic_viscosity;
    field_type effective_thermal_conductivity;
    field_type candidate_effective_thermal_conductivity;
    field_type k_diffusivity;
    field_type secondary_diffusivity;
    field_type k_source;
    field_type k_sink;
    field_type secondary_source;
    field_type secondary_sink;
    field_type buoyancy_production;
    field_type candidate_buoyancy_production;
    field_type wall_distance;
    field_type wall_y_plus;
    field_type candidate_wall_y_plus;
    velocity_field_type wall_velocity;
    velocity_field_type candidate_wall_velocity;
    TensorCellField<Pack> velocity_gradient;
    TensorCellField<Pack> candidate_velocity_gradient;
    VectorCellField<Pack> buoyancy_gradient;
    VectorCellField<Pack> k_gradient;
    VectorCellField<Pack> secondary_gradient;
    VectorCellField<Pack> candidate_k_gradient;
    VectorCellField<Pack> candidate_secondary_gradient;
    TurbulenceScalarTransportEquation<Pack> k_equation;
    TurbulenceScalarTransportEquation<Pack> secondary_equation;
    std::map<std::string, const field_type*> output_fields;
};

/**
 * @brief Construct a runtime turbulence model on a mesh.
 * @tparam Pack Tpetra type pack used by the model.
 * @param mesh Distributed computational mesh.
 * @param boundary_conditions Velocity and turbulence scalar boundaries.
 * @throws std::invalid_argument if the mesh or boundary types are invalid.
 */
template <TpetraTypePack Pack>
TurbulenceModel<Pack>::TurbulenceModel(SP<const mesh_type> mesh,
                                       const BoundaryConditionSet& boundary_conditions)
    : d_mesh(std::move(mesh)), d_velocity_boundary_conditions(boundary_conditions.velocity),
      d_wall_velocity_boundary_cache(d_mesh),
      d_boundary_conditions(boundary_conditions.turbulence)
{
    if (!d_mesh)
    {
        throw std::invalid_argument("TurbulenceModel requires a non-null mesh.");
    }
    d_wall_velocity_boundary_cache =
        FVM::cache_velocity_boundary_conditions<Pack>(d_mesh, boundary_conditions);

    turbulence_detail::collective_local_validation(
        *d_mesh, "Turbulence boundary-condition validation",
        [&]
        {
            auto validate_boundaries = [](const BoundaryConditionMap& conditions)
            {
                for (const auto& [name, condition] : conditions)
                {
                    if (condition.type != BoundaryConditionType::Dirichlet &&
                        condition.type != BoundaryConditionType::Neumann)
                    {
                        throw std::invalid_argument("Turbulence scalar boundary '" + name +
                                                    "' must be Dirichlet or Neumann.");
                    }
                }
            };
            validate_boundaries(d_boundary_conditions.turbulent_kinetic_energy);
            validate_boundaries(d_boundary_conditions.dissipation_rate);
            validate_boundaries(d_boundary_conditions.specific_dissipation_rate);
        });
}

template <TpetraTypePack Pack> TurbulenceModel<Pack>::~TurbulenceModel() = default;

/**
 * @brief Parse database options and replace the active closure collectively.
 * @tparam Pack Tpetra type pack used by the model.
 * @param database Source configuration database.
 * @param material Material fields used to initialize effective properties.
 * @param reference_density Positive momentum reference density.
 * @throws std::invalid_argument if parsing or configuration validation fails.
 * @throws std::overflow_error if a derived property is invalid.
 */
template <TpetraTypePack Pack>
void TurbulenceModel<Pack>::configure(const Database& database,
                                      const material_type& material,
                                      scalar_type reference_density)
{
    TurbulenceModelOptions parsed_options;
    turbulence_detail::collective_local_validation(
        *d_mesh, "Turbulence database parsing",
        [&]
        { parsed_options = turbulence_model_options_from_database(database); });
    configure(parsed_options, material, reference_density);
}

/**
 * @brief Replace the active closure and reset its transported state.
 * @tparam Pack Tpetra type pack used by the model.
 * @param options Closure, floor, and wall-treatment options.
 * @param material Material fields used to initialize effective properties.
 * @param reference_density Positive momentum reference density.
 * @throws std::invalid_argument if options, boundaries, or material inputs fail.
 * @throws std::logic_error if an invalid closure state is requested.
 * @throws std::overflow_error if a derived property is invalid.
 */
template <TpetraTypePack Pack>
void TurbulenceModel<Pack>::configure(const TurbulenceModelOptions& options,
                                      const material_type& material, scalar_type reference_density)
{
    turbulence_detail::collective_local_validation(*d_mesh, "Turbulence model option validation",
                                                   [&]
                                                   { validate_turbulence_model_options(options); });
    turbulence_detail::require_uniform_integral(*d_mesh, static_cast<int>(options.model),
                                                "Turbulence model type");
    turbulence_detail::require_uniform_integral(
        *d_mesh, static_cast<int>(options.wall_treatment),
        "Turbulence wall-treatment type");
    turbulence_detail::require_uniform_integral(
        *d_mesh, static_cast<int>(options.buoyancy_model),
        "Turbulence buoyancy model");
    turbulence_detail::require_uniform_integral(
        *d_mesh, static_cast<int>(options.gradient_scheme),
        "Turbulence cell-gradient scheme");
    turbulence_detail::require_uniform_integral(
        *d_mesh, static_cast<int>(options.coefficient_interpolation),
        "Turbulence face-coefficient interpolation");
    turbulence_detail::require_uniform_real(
        *d_mesh, options.buoyancy_coefficient,
        "Turbulence buoyancy coefficient");
    turbulence_detail::require_uniform_integral(
        *d_mesh,
        static_cast<int>(
            options.wall_distance_equation
                .non_orthogonal_treatment),
        "Turbulence wall-distance non-orthogonal treatment");
    turbulence_detail::require_uniform_integral(
        *d_mesh,
        options.wall_distance_equation.non_orthogonal_correctors,
        "Turbulence wall-distance correctors");
    turbulence_detail::require_uniform_integral(
        *d_mesh,
        options.wall_distance_equation.linear_solver.max_iterations,
        "Turbulence wall-distance maximum iterations");
    turbulence_detail::require_uniform_real(
        *d_mesh,
        options.wall_distance_equation.linear_solver.tolerance,
        "Turbulence wall-distance linear tolerance");
    turbulence_detail::require_uniform_integral(
        *d_mesh,
        options.wall_distance_equation.linear_solver.verbosity,
        "Turbulence wall-distance linear verbosity");
    turbulence_detail::require_uniform_integral(
        *d_mesh,
        static_cast<int>(
            options.wall_distance_equation.linear_solver.preconditioner),
        "Turbulence wall-distance preconditioner");
    turbulence_detail::require_uniform_integral(
        *d_mesh,
        options.wall_distance_equation.linear_solver
                .reuse_preconditioner
            ? 1
            : 0,
        "Turbulence wall-distance preconditioner reuse");
    const real_t wall_scalar_options[] = {
        options.wall_options.c_mu,
        options.wall_options.kappa,
        options.wall_options.log_layer_e,
        options.wall_options.sst_beta_1,
        options.wall_options.sst_omega_wall_coefficient};
    for (const auto value : wall_scalar_options)
    {
        turbulence_detail::require_uniform_real(
            *d_mesh, value, "Turbulence wall scalar options");
    }
    turbulence_detail::require_uniform_integral(
        *d_mesh, options.wall_options.epsilon_low_re_correction ? 1 : 0,
        "Turbulence wall epsilon low-Re correction");
    const std::array<real_t, 7> scalar_options{options.initial_turbulent_kinetic_energy,
                                               options.initial_dissipation_rate,
                                               options.initial_specific_dissipation_rate,
                                               options.min_turbulent_kinetic_energy,
                                               options.min_dissipation_rate,
                                               options.min_specific_dissipation_rate,
                                               options.turbulent_prandtl_number};
    for (const auto value : scalar_options)
    {
        turbulence_detail::require_uniform_real(*d_mesh, value, "Turbulence scalar options");
    }
    turbulence_detail::require_uniform_integral(*d_mesh,
                                                options.initial_wall_distance.has_value() ? 1 : 0,
                                                "Turbulence wall-distance presence");
    if (options.initial_wall_distance)
    {
        turbulence_detail::require_uniform_real(*d_mesh, *options.initial_wall_distance,
                                                "Turbulence wall distance");
    }
    if (options.model == TurbulenceModelType::Laminar)
    {
        d_options = options;
        d_state.reset();
        return;
    }
    turbulence_detail::collective_local_validation(
        *d_mesh, "Active turbulence boundary-condition validation",
        [&]
        {
            for (const auto& name : options.wall_distance_boundaries)
            {
                const auto velocity_condition =
                    d_velocity_boundary_conditions.find(name);
                if (velocity_condition
                        == d_velocity_boundary_conditions.end()
                    || velocity_condition->second.type
                        != BoundaryConditionType::NoSlip)
                {
                    throw std::invalid_argument(
                        "Wall-distance boundary '" + name
                        + "' must be a no-slip velocity patch.");
                }
            }

            auto validate_scalar_boundaries =
                [&](const BoundaryConditionMap& conditions, real_t floor,
                    std::string_view field_name)
            {
                for (const auto& [name, condition] : conditions)
                {
                    if (condition.type != BoundaryConditionType::Dirichlet &&
                        condition.type != BoundaryConditionType::Neumann)
                    {
                        throw std::invalid_argument(
                            std::string(field_name) + " boundary '" + name +
                            "' must be Dirichlet or Neumann.");
                    }
                    if (!std::isfinite(condition.value) ||
                        !std::isfinite(condition.robin_coefficient))
                    {
                        throw std::invalid_argument(
                            std::string(field_name) + " boundary '" + name +
                            "' contains non-finite data.");
                    }
                    const auto wall_owned =
                        std::find(options.wall_options.boundary_names.begin(),
                                  options.wall_options.boundary_names.end(), name) !=
                        options.wall_options.boundary_names.end();
                    if (condition.type == BoundaryConditionType::Dirichlet &&
                        condition.value < floor && !wall_owned)
                    {
                        throw std::invalid_argument(
                            std::string(field_name) + " Dirichlet boundary '" + name +
                            "' lies below its positive floor.");
                    }
                }
            };

            validate_scalar_boundaries(
                d_boundary_conditions.turbulent_kinetic_energy,
                options.min_turbulent_kinetic_energy,
                "Turbulent kinetic energy");
            const auto epsilon_family =
                options.model == TurbulenceModelType::StandardKEpsilon ||
                options.model == TurbulenceModelType::RNGKEpsilon ||
                options.model == TurbulenceModelType::RealizableKEpsilon;
            validate_scalar_boundaries(
                epsilon_family ? d_boundary_conditions.dissipation_rate
                               : d_boundary_conditions.specific_dissipation_rate,
                epsilon_family ? options.min_dissipation_rate
                               : options.min_specific_dissipation_rate,
                epsilon_family ? "Dissipation rate" : "Specific dissipation rate");
        });
    turbulence_detail::collective_local_validation(
        *d_mesh, "Turbulence reference-density validation",
        [&]
        {
            if (!std::isfinite(reference_density) || reference_density <= scalar_type{})
            {
                throw std::invalid_argument("TurbulenceModel requires a positive finite reference "
                                            "density.");
            }
        });
    turbulence_detail::require_uniform_real(*d_mesh, static_cast<real_t>(reference_density),
                                            "Turbulence reference density");

    BoundaryConditionSet configured_boundaries;
    configured_boundaries.velocity = d_velocity_boundary_conditions;
    auto configured_velocity_boundary_cache =
        FVM::cache_velocity_boundary_conditions<Pack>(
            d_mesh, configured_boundaries);
    auto candidate = std::make_unique<State>(d_mesh, d_boundary_conditions,
                                             d_velocity_boundary_conditions, options);
    if (candidate->menter_family
        && !options.initial_wall_distance.has_value())
    {
        auto wall_names = options.wall_distance_boundaries;
        wall_names.insert(
            wall_names.end(),
            options.wall_options.boundary_names.begin(),
            options.wall_options.boundary_names.end());
        for (const auto& [name, condition] :
             d_velocity_boundary_conditions)
        {
            if (condition.type == BoundaryConditionType::NoSlip)
            {
                wall_names.push_back(name);
            }
        }
        std::sort(wall_names.begin(), wall_names.end());
        wall_names.erase(
            std::unique(wall_names.begin(), wall_names.end()),
            wall_names.end());
        PoissonWallDistanceEquation<Pack>{d_mesh}.solve(
            wall_names, candidate->wall_distance,
            options.wall_distance_equation);
    }
    auto initial_wall_evaluation = candidate->evaluate_wall(
        candidate->k, candidate->wall_velocity,
        configured_velocity_boundary_cache, material, reference_density,
        static_cast<scalar_type>(options.turbulent_prandtl_number));
    turbulence_detail::collective_local_validation(
        *d_mesh, "Initial turbulence-gradient reconstruction",
        [&]
        {
            auto k_condition = [&](int batch_id, size_t in_batch_id)
            {
                const auto wall = State::wall_boundary_condition(
                    initial_wall_evaluation, batch_id, in_batch_id, true);
                if (wall)
                    return *wall;
                const auto& name = d_mesh->boundary_batch_name(batch_id);
                const auto iter = d_boundary_conditions.turbulent_kinetic_energy.find(name);
                return iter == d_boundary_conditions.turbulent_kinetic_energy.end()
                     ? BoundaryCondition{}
                     : iter->second;
            };
            auto k_value = [&](int batch_id, size_t in_batch_id)
            { return static_cast<scalar_type>(k_condition(batch_id, in_batch_id).value); };
            turbulence_detail::reconstruct_gradient(
                options.gradient_scheme,
                candidate->k, k_condition, k_value,
                candidate->k_gradient, candidate->gradient_cache);
            if (candidate->menter_family)
            {
                auto secondary_condition = [&](int batch_id, size_t in_batch_id)
                {
                    const auto wall = State::wall_boundary_condition(
                        initial_wall_evaluation, batch_id, in_batch_id, false);
                    if (wall)
                        return *wall;
                    const auto& name = d_mesh->boundary_batch_name(batch_id);
                    const auto iter =
                        d_boundary_conditions.specific_dissipation_rate.find(name);
                    return iter == d_boundary_conditions.specific_dissipation_rate.end()
                         ? BoundaryCondition{}
                         : iter->second;
                };
                auto secondary_value = [&](int batch_id, size_t in_batch_id)
                {
                    return static_cast<scalar_type>(
                        secondary_condition(batch_id, in_batch_id).value);
                };
                turbulence_detail::reconstruct_gradient(
                    options.gradient_scheme,
                    candidate->secondary, secondary_condition,
                    secondary_value,
                    candidate->secondary_gradient,
                    candidate->gradient_cache);
            }
        });
    candidate->k_gradient.sync_ghosts();
    if (candidate->menter_family)
    {
        candidate->secondary_gradient.sync_ghosts();
    }

    if (options.model == TurbulenceModelType::SSTKOmega)
    {
        const auto wall_velocity_values =
            candidate->wall_velocity.local_read_view();
        auto initial_boundary_velocity =
            [&](int batch_id, size_t in_batch_id) ->
            typename velocity_field_type::vec_type
        {
            const auto& batch =
                d_mesh->boundary_batches().at(batch_id);
            const auto face_lid =
                batch.face_lids.at(in_batch_id);
            const auto type =
                configured_velocity_boundary_cache.type.at(batch_id);
            if (type == BoundaryConditionType::Slip)
            {
                return FVM::slip_face_velocity(
                    candidate->wall_velocity, face_lid);
            }
            if (type == BoundaryConditionType::Periodic)
            {
                const auto owner = d_mesh->owner_cell(face_lid);
                return {wall_velocity_values(owner, 0),
                        wall_velocity_values(owner, 1),
                        wall_velocity_values(owner, 2)};
            }
            if (type == BoundaryConditionType::Neumann)
            {
                const auto owner = d_mesh->owner_cell(face_lid);
                return {wall_velocity_values(owner, 0),
                        wall_velocity_values(owner, 1),
                        wall_velocity_values(owner, 2)};
            }
            return configured_velocity_boundary_cache.value
                .at(batch_id)
                .at(in_batch_id);
        };
        turbulence_detail::collective_local_validation(
            *d_mesh, "Initial SST eddy-viscosity evaluation",
            [&]
            {
                if (&material.dynamic_viscosity.mesh() != d_mesh.get())
                {
                    throw std::invalid_argument(
                        "TurbulenceModel material field mesh mismatch.");
                }
                turbulence_detail::reconstruct_gradient(
                    options.gradient_scheme,
                    candidate->wall_velocity,
                    initial_boundary_velocity,
                    candidate->velocity_gradient,
                    candidate->gradient_cache);
                const auto& closure =
                    std::get<SSTKOmegaEquation>(
                        candidate->closure);
                const auto molecular_viscosity_values =
                    material.dynamic_viscosity.owned_read_view();
                const auto velocity_gradient_values =
                    candidate->velocity_gradient.owned_read_view();
                const auto k_gradient_values =
                    candidate->k_gradient.owned_read_view();
                const auto omega_gradient_values =
                    candidate->secondary_gradient.owned_read_view();
                const auto wall_distance_values =
                    candidate->wall_distance.owned_read_view();
                const auto k_values =
                    candidate->k.owned_read_view();
                const auto omega_values =
                    candidate->secondary.owned_read_view();
                auto turbulent_viscosity_values =
                    candidate->candidate_nu_t.owned_write_view();
                for (size_t owned = 0;
                     owned < d_mesh->num_owned_cells(); ++owned)
                {
                    const auto cell_lid =
                        static_cast<local_ordinal_type>(owned);
                    const auto molecular_viscosity =
                        static_cast<real_t>(
                            molecular_viscosity_values(cell_lid, 0));
                    if (!std::isfinite(molecular_viscosity)
                        || molecular_viscosity < 0.0)
                    {
                        throw std::invalid_argument(
                            "TurbulenceModel requires finite "
                            "non-negative molecular viscosity.");
                    }

                    real_t rotation_squared{};
                    for (size_t row = 0; row < 3; ++row)
                    {
                        for (size_t column = 0; column < 3;
                             ++column)
                        {
                            const auto gij = static_cast<real_t>(
                                velocity_gradient_values(
                                    cell_lid, row * 3 + column));
                            const auto gji = static_cast<real_t>(
                                velocity_gradient_values(
                                    cell_lid, column * 3 + row));
                            const auto wij = 0.5 * (gij - gji);
                            rotation_squared += wij * wij;
                        }
                    }
                    const typename velocity_field_type::vec_type
                        k_gradient{k_gradient_values(cell_lid, 0),
                                   k_gradient_values(cell_lid, 1),
                                   k_gradient_values(cell_lid, 2)};
                    const typename velocity_field_type::vec_type
                        omega_gradient{
                            omega_gradient_values(cell_lid, 0),
                            omega_gradient_values(cell_lid, 1),
                            omega_gradient_values(cell_lid, 2)};
                    const MenterKOmegaInvariants invariants{
                        molecular_viscosity
                            / static_cast<real_t>(reference_density),
                        static_cast<real_t>(
                            wall_distance_values(cell_lid, 0)),
                        static_cast<real_t>(
                            k_gradient.dot(omega_gradient)),
                        std::sqrt(2.0 * rotation_squared)};
                    const auto nu_t =
                        closure.turbulent_kinematic_viscosity(
                            {static_cast<real_t>(
                                 k_values(cell_lid, 0)),
                             static_cast<real_t>(
                                 omega_values(cell_lid, 0))},
                            invariants);
                    if (!std::isfinite(nu_t) || nu_t < 0.0)
                    {
                        throw std::runtime_error(
                            "Initial SST evaluation produced an "
                            "invalid eddy viscosity.");
                    }
                    turbulent_viscosity_values(cell_lid, 0) =
                        static_cast<scalar_type>(nu_t);
                }
            });
        candidate->velocity_gradient.sync_ghosts();
        candidate->candidate_nu_t.sync_ghosts();
    }

    stage_effective_properties(*candidate, candidate->candidate_nu_t, material,
                               reference_density,
                               static_cast<scalar_type>(options.turbulent_prandtl_number));
    auto initial_wall_publication =
        candidate->stage_wall_publication(
            std::move(initial_wall_evaluation));
    State::publish_synced_field(
        candidate->nu_t, candidate->candidate_nu_t);
    commit_effective_properties(*candidate);
    candidate->publish_wall_publication(
        std::move(initial_wall_publication));
    d_wall_velocity_boundary_cache =
        std::move(configured_velocity_boundary_cache);
    d_options = options;
    d_state = std::move(candidate);
}

/**
 * @brief Disable turbulence and release the allocated closure state.
 * @tparam Pack Tpetra type pack used by the model.
 * @return True when an active state was released.
 */
template <TpetraTypePack Pack> bool TurbulenceModel<Pack>::disable() noexcept
{
    const auto was_enabled = static_cast<bool>(d_state);
    d_state.reset();
    d_options = TurbulenceModelOptions{};
    return was_enabled;
}

/**
 * @brief Report whether a non-laminar closure is active.
 * @tparam Pack Tpetra type pack used by the model.
 * @return True when turbulence state is allocated.
 */
template <TpetraTypePack Pack> bool TurbulenceModel<Pack>::enabled() const noexcept
{
    return static_cast<bool>(d_state);
}

/**
 * @brief Return the configured turbulence model type.
 * @tparam Pack Tpetra type pack used by the model.
 * @return Runtime model identifier.
 */
template <TpetraTypePack Pack> TurbulenceModelType TurbulenceModel<Pack>::type() const noexcept
{
    return d_options.model;
}

/**
 * @brief Return the currently configured model options.
 * @tparam Pack Tpetra type pack used by the model.
 * @return Stored model options.
 */
template <TpetraTypePack Pack>
const TurbulenceModelOptions& TurbulenceModel<Pack>::options() const noexcept
{
    return d_options;
}

/**
 * @brief Require and return mutable active turbulence state.
 * @tparam Pack Tpetra type pack used by the model.
 * @return Mutable active state.
 * @throws std::logic_error if the model is disabled.
 */
template <TpetraTypePack Pack> auto TurbulenceModel<Pack>::require_state() -> State&
{
    if (!d_state)
    {
        throw std::logic_error("TurbulenceModel is disabled in laminar mode.");
    }
    return *d_state;
}

/**
 * @brief Require and return immutable active turbulence state.
 * @tparam Pack Tpetra type pack used by the model.
 * @return Immutable active state.
 * @throws std::logic_error if the model is disabled.
 */
template <TpetraTypePack Pack> auto TurbulenceModel<Pack>::require_state() const -> const State&
{
    if (!d_state)
    {
        throw std::logic_error("TurbulenceModel is disabled in laminar mode.");
    }
    return *d_state;
}

/**
 * @brief Stage effective viscosity and conductivity from molecular fields.
 * @tparam Pack Tpetra type pack used by the model.
 * @param[in,out] state State receiving candidate effective properties.
 * @param turbulent_kinematic_viscosity Eddy-viscosity field to apply.
 * @param material Molecular material-property fields.
 * @param reference_density Positive momentum reference density.
 * @param turbulent_prandtl_number Positive turbulent Prandtl number.
 * @throws std::invalid_argument if fields or physical inputs are invalid.
 * @throws std::overflow_error if a derived property is invalid.
 */
template <TpetraTypePack Pack>
void TurbulenceModel<Pack>::stage_effective_properties(
    State& state, const field_type& turbulent_kinematic_viscosity, const material_type& material,
    scalar_type reference_density, scalar_type turbulent_prandtl_number) const
{
    turbulence_detail::collective_local_validation(
        *d_mesh, "Turbulence effective-property validation",
        [&]
        {
            if (!std::isfinite(reference_density) || reference_density <= scalar_type{})
            {
                throw std::invalid_argument("TurbulenceModel requires a positive finite reference "
                                            "density.");
            }
            if (!std::isfinite(turbulent_prandtl_number) ||
                turbulent_prandtl_number <= scalar_type{})
            {
                throw std::invalid_argument("TurbulenceModel requires a positive finite turbulent "
                                            "Prandtl number.");
            }
            const field_type* fields[] = {
                &material.density, &material.specific_heat_capacity, &material.dynamic_viscosity,
                &material.thermal_conductivity, &turbulent_kinematic_viscosity};
            for (const auto* field : fields)
            {
                if (&field->mesh() != d_mesh.get())
                {
                    throw std::invalid_argument("TurbulenceModel material field mesh mismatch.");
                }
            }

            const auto density_values =
                material.density.owned_read_view();
            const auto heat_capacity_values =
                material.specific_heat_capacity.owned_read_view();
            const auto molecular_viscosity_values =
                material.dynamic_viscosity.owned_read_view();
            const auto molecular_conductivity_values =
                material.thermal_conductivity.owned_read_view();
            const auto turbulent_viscosity_values =
                turbulent_kinematic_viscosity.owned_read_view();
            auto effective_viscosity_values =
                state.candidate_effective_dynamic_viscosity
                    .owned_write_view();
            auto effective_conductivity_values =
                state.candidate_effective_thermal_conductivity
                    .owned_write_view();
            for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell_lid = static_cast<local_ordinal_type>(owned);
                const auto density = density_values(cell_lid, 0);
                const auto heat_capacity =
                    heat_capacity_values(cell_lid, 0);
                const auto molecular_viscosity =
                    molecular_viscosity_values(cell_lid, 0);
                const auto molecular_conductivity =
                    molecular_conductivity_values(cell_lid, 0);
                const auto nu_t =
                    turbulent_viscosity_values(cell_lid, 0);
                const scalar_type values[] = {density, heat_capacity, molecular_viscosity,
                                              molecular_conductivity, nu_t};
                for (const auto value : values)
                {
                    if (!std::isfinite(value) || value < scalar_type{})
                    {
                        throw std::invalid_argument(
                            "TurbulenceModel effective-property inputs must "
                            "be finite and non-negative.");
                    }
                }
                const auto effective_viscosity = molecular_viscosity + reference_density * nu_t;
                const auto effective_conductivity =
                    molecular_conductivity +
                    density * heat_capacity * nu_t / turbulent_prandtl_number;
                if (!std::isfinite(effective_viscosity) || effective_viscosity < scalar_type{} ||
                    !std::isfinite(effective_conductivity) ||
                    effective_conductivity < scalar_type{})
                {
                    throw std::overflow_error("TurbulenceModel effective-property calculation "
                                              "produced a non-finite or negative result.");
                }
                effective_viscosity_values(cell_lid, 0) =
                    effective_viscosity;
                effective_conductivity_values(cell_lid, 0) =
                    effective_conductivity;
            }
        });
    state.candidate_effective_dynamic_viscosity.sync_ghosts();
    state.candidate_effective_thermal_conductivity.sync_ghosts();
}

/**
 * @brief Publish staged effective properties to accepted fields.
 * @tparam Pack Tpetra type pack used by the model.
 * @param[in,out] state State whose candidate properties are committed.
 */
template <TpetraTypePack Pack>
void TurbulenceModel<Pack>::commit_effective_properties(State& state) const
{
    State::publish_synced_field(
        state.effective_dynamic_viscosity,
        state.candidate_effective_dynamic_viscosity);
    State::publish_synced_field(
        state.effective_thermal_conductivity,
        state.candidate_effective_thermal_conductivity);
}

/**
 * @brief Stage accepted-state BSL/SST eddy viscosity for new molecular data.
 */
template <TpetraTypePack Pack>
void TurbulenceModel<Pack>::stage_menter_eddy_viscosity(
    State& state, const field_type& wall_distance,
    const material_type& material, scalar_type reference_density,
    std::string_view context) const
{
    turbulence_detail::collective_local_validation(
        *d_mesh, context,
        [&]
        {
            if (!state.menter_family)
            {
                throw std::logic_error(
                    "TurbulenceModel wall distance is available only for "
                    "BSL or SST k-omega closures.");
            }
            if (&wall_distance.mesh() != d_mesh.get()
                || &material.dynamic_viscosity.mesh() != d_mesh.get())
            {
                throw std::invalid_argument(
                    "TurbulenceModel Menter refresh field mesh mismatch.");
            }
            if (!std::isfinite(reference_density)
                || reference_density <= scalar_type{})
            {
                throw std::invalid_argument(
                    "TurbulenceModel Menter refresh requires a positive "
                    "finite reference density.");
            }

            const auto molecular_viscosity_values =
                material.dynamic_viscosity.owned_read_view();
            const auto velocity_gradient_values =
                state.velocity_gradient.owned_read_view();
            const auto k_values = state.k.owned_read_view();
            const auto secondary_values =
                state.secondary.owned_read_view();
            const auto k_gradient_values =
                state.k_gradient.owned_read_view();
            const auto secondary_gradient_values =
                state.secondary_gradient.owned_read_view();
            const auto wall_distance_values =
                wall_distance.owned_read_view();
            auto turbulent_viscosity_values =
                state.candidate_nu_t.owned_write_view();
            for (size_t owned = 0;
                 owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell_lid =
                    static_cast<local_ordinal_type>(owned);
                const auto molecular_viscosity =
                    static_cast<real_t>(
                        molecular_viscosity_values(cell_lid, 0));
                if (!std::isfinite(molecular_viscosity)
                    || molecular_viscosity < 0.0)
                {
                    throw std::invalid_argument(
                        "TurbulenceModel Menter refresh requires finite "
                        "non-negative molecular viscosity.");
                }

                real_t rotation_squared{};
                for (size_t row = 0; row < 3; ++row)
                {
                    for (size_t column = 0; column < 3; ++column)
                    {
                        const auto gij = static_cast<real_t>(
                            velocity_gradient_values(
                                cell_lid, row * 3 + column));
                        const auto gji = static_cast<real_t>(
                            velocity_gradient_values(
                                cell_lid, column * 3 + row));
                        const auto wij = 0.5 * (gij - gji);
                        rotation_squared += wij * wij;
                    }
                }
                const KOmegaState local{
                    static_cast<real_t>(k_values(cell_lid, 0)),
                    static_cast<real_t>(
                        secondary_values(cell_lid, 0))};
                const typename velocity_field_type::vec_type
                    k_gradient{k_gradient_values(cell_lid, 0),
                               k_gradient_values(cell_lid, 1),
                               k_gradient_values(cell_lid, 2)};
                const typename velocity_field_type::vec_type
                    omega_gradient{
                        secondary_gradient_values(cell_lid, 0),
                        secondary_gradient_values(cell_lid, 1),
                        secondary_gradient_values(cell_lid, 2)};
                const MenterKOmegaInvariants invariants{
                    molecular_viscosity
                        / static_cast<real_t>(reference_density),
                    static_cast<real_t>(
                        wall_distance_values(cell_lid, 0)),
                    static_cast<real_t>(
                        k_gradient.dot(omega_gradient)),
                    std::sqrt(2.0 * rotation_squared)};

                real_t nu_t{};
                if (d_options.model == TurbulenceModelType::BSLKOmega)
                {
                    nu_t = std::get<BSLKOmegaEquation>(
                               state.closure)
                               .turbulent_kinematic_viscosity(local);
                }
                else if (d_options.model
                         == TurbulenceModelType::SSTKOmega)
                {
                    nu_t = std::get<SSTKOmegaEquation>(
                               state.closure)
                               .turbulent_kinematic_viscosity(
                                   local, invariants);
                }
                else
                {
                    throw std::logic_error(
                        "TurbulenceModel lost its active Menter closure.");
                }
                if (!std::isfinite(nu_t) || nu_t < 0.0)
                {
                    throw std::runtime_error(
                        "TurbulenceModel Menter refresh produced an "
                        "invalid eddy viscosity.");
                }
                turbulent_viscosity_values(cell_lid, 0) =
                    static_cast<scalar_type>(nu_t);
            }
        });
    state.candidate_nu_t.sync_ghosts();
}

/**
 * @brief Rebuild accepted effective properties without advancing turbulence.
 * @tparam Pack Tpetra type pack used by the model.
 * @param material Current molecular material-property fields.
 * @param reference_density Positive momentum reference density.
 * @throws std::logic_error if the model is disabled.
 * @throws std::invalid_argument if material or wall inputs are invalid.
 * @throws std::overflow_error if a derived property is invalid.
 */
template <TpetraTypePack Pack>
void TurbulenceModel<Pack>::refresh_effective_properties(const material_type& material,
                                                         scalar_type reference_density)
{
    turbulence_detail::require_uniform_integral(*d_mesh, enabled() ? 1 : 0,
                                                "Turbulence enabled state");
    auto& state = require_state();
    turbulence_detail::require_uniform_real(*d_mesh, static_cast<real_t>(reference_density),
                                            "Turbulence reference density");
    auto wall_evaluation = state.evaluate_wall(
        state.k, state.wall_velocity, d_wall_velocity_boundary_cache,
        material, reference_density,
        static_cast<scalar_type>(d_options.turbulent_prandtl_number),
        &state.wall_evaluation);
    auto wall_publication =
        state.stage_wall_publication(std::move(wall_evaluation));
    const field_type* effective_nu_t = &state.nu_t;
    if (state.menter_family)
    {
        stage_menter_eddy_viscosity(
            state, state.wall_distance, material, reference_density,
            "Turbulence Menter effective-property refresh");
        effective_nu_t = &state.candidate_nu_t;
    }
    stage_effective_properties(state, *effective_nu_t, material,
                               reference_density,
                               static_cast<scalar_type>(d_options.turbulent_prandtl_number));
    if (state.menter_family)
    {
        State::publish_synced_field(state.nu_t, state.candidate_nu_t);
    }
    commit_effective_properties(state);
    state.publish_wall_publication(std::move(wall_publication));
}

/**
 * @brief Restore accepted turbulence fields and all directly dependent data.
 * @tparam Pack Tpetra type pack used by the model.
 * @param turbulent_kinetic_energy Positive restart k field.
 * @param secondary Positive restart epsilon or omega field.
 * @param turbulent_kinematic_viscosity Non-negative restart nu_t field.
 * @param velocity Accepted restart velocity used by wall functions.
 * @param material Current molecular material-property fields.
 * @param reference_density Positive momentum reference density.
 */
template <TpetraTypePack Pack>
void TurbulenceModel<Pack>::restore_transported_state(
    const field_type& turbulent_kinetic_energy,
    const field_type& secondary,
    const field_type& turbulent_kinematic_viscosity,
    const velocity_field_type& velocity,
    const material_type& material,
    scalar_type reference_density)
{
    turbulence_detail::require_uniform_integral(
        *d_mesh, enabled() ? 1 : 0,
        "Turbulence enabled state");
    turbulence_detail::require_uniform_real(
        *d_mesh, static_cast<real_t>(reference_density),
        "Turbulence reference density");
    auto& state = require_state();

    turbulence_detail::collective_local_validation(
        *d_mesh, "Turbulence restart-state validation",
        [&]
        {
            const field_type* scalar_fields[] = {
                &turbulent_kinetic_energy,
                &secondary,
                &turbulent_kinematic_viscosity};
            for (const auto* field : scalar_fields)
            {
                if (&field->mesh() != d_mesh.get())
                {
                    throw std::invalid_argument(
                        "Turbulence restart field mesh mismatch.");
                }
            }
            if (&velocity.mesh() != d_mesh.get()
                || &material.density.mesh() != d_mesh.get()
                || &material.specific_heat_capacity.mesh() != d_mesh.get()
                || &material.dynamic_viscosity.mesh() != d_mesh.get()
                || &material.thermal_conductivity.mesh() != d_mesh.get())
            {
                throw std::invalid_argument(
                    "Turbulence restart velocity or material mesh mismatch.");
            }
            if (!std::isfinite(reference_density)
                || reference_density <= scalar_type{})
            {
                throw std::invalid_argument(
                    "Turbulence restart requires a positive finite "
                    "reference density.");
            }

            const auto k_values =
                turbulent_kinetic_energy.owned_read_view();
            const auto secondary_values =
                secondary.owned_read_view();
            const auto nu_t_values =
                turbulent_kinematic_viscosity.owned_read_view();
            const auto k_floor = static_cast<scalar_type>(
                d_options.min_turbulent_kinetic_energy);
            const auto secondary_floor = static_cast<scalar_type>(
                state.epsilon_family
                    ? d_options.min_dissipation_rate
                    : d_options.min_specific_dissipation_rate);
            for (size_t owned = 0;
                 owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell_lid =
                    static_cast<local_ordinal_type>(owned);
                const auto k = k_values(cell_lid, 0);
                const auto secondary_value =
                    secondary_values(cell_lid, 0);
                const auto nu_t = nu_t_values(cell_lid, 0);
                if (!std::isfinite(k) || k < k_floor
                    || !std::isfinite(secondary_value)
                    || secondary_value < secondary_floor
                    || !std::isfinite(nu_t) || nu_t < scalar_type{})
                {
                    throw std::invalid_argument(
                        "Turbulence restart fields must be finite and "
                        "respect their configured lower bounds.");
                }
            }
        });

    state.candidate_k.owned_data().update(
        scalar_type{1},
        turbulent_kinetic_energy.owned_data(),
        scalar_type{0});
    state.candidate_secondary.owned_data().update(
        scalar_type{1}, secondary.owned_data(), scalar_type{0});
    state.candidate_nu_t.owned_data().update(
        scalar_type{1},
        turbulent_kinematic_viscosity.owned_data(),
        scalar_type{0});
    state.candidate_k.sync_ghosts();
    state.candidate_secondary.sync_ghosts();
    state.candidate_nu_t.sync_ghosts();

    BoundaryConditionSet configured_boundaries;
    configured_boundaries.velocity =
        d_velocity_boundary_conditions;
    auto velocity_boundary_cache =
        FVM::cache_velocity_boundary_conditions<Pack>(
            d_mesh, configured_boundaries);
    auto wall_evaluation = state.evaluate_wall(
        state.candidate_k, velocity, velocity_boundary_cache,
        material, reference_density,
        static_cast<scalar_type>(
            d_options.turbulent_prandtl_number),
        &state.wall_evaluation);
    auto wall_publication =
        state.stage_wall_publication(std::move(wall_evaluation));

    turbulence_detail::collective_local_validation(
        *d_mesh, "Turbulence restart-gradient reconstruction",
        [&]
        {
            auto k_condition =
                [&](int batch_id, size_t in_batch_id)
            {
                const auto wall =
                    State::wall_boundary_condition(
                        wall_publication.evaluation,
                        batch_id, in_batch_id, true);
                if (wall)
                    return *wall;
                const auto& name =
                    d_mesh->boundary_batch_name(batch_id);
                const auto iter =
                    d_boundary_conditions
                        .turbulent_kinetic_energy.find(name);
                return iter
                        == d_boundary_conditions
                               .turbulent_kinetic_energy.end()
                    ? BoundaryCondition{}
                    : iter->second;
            };
            auto k_value =
                [&](int batch_id, size_t in_batch_id)
            {
                return static_cast<scalar_type>(
                    k_condition(batch_id, in_batch_id).value);
            };
            turbulence_detail::reconstruct_gradient(
                d_options.gradient_scheme,
                state.candidate_k, k_condition, k_value,
                state.candidate_k_gradient,
                state.gradient_cache);

            if (state.menter_family)
            {
                auto secondary_condition =
                    [&](int batch_id, size_t in_batch_id)
                {
                    const auto wall =
                        State::wall_boundary_condition(
                            wall_publication.evaluation,
                            batch_id, in_batch_id, false);
                    if (wall)
                        return *wall;
                    const auto& name =
                        d_mesh->boundary_batch_name(batch_id);
                    const auto iter =
                        d_boundary_conditions
                            .specific_dissipation_rate.find(name);
                    return iter
                            == d_boundary_conditions
                                   .specific_dissipation_rate.end()
                        ? BoundaryCondition{}
                        : iter->second;
                };
                auto secondary_value =
                    [&](int batch_id, size_t in_batch_id)
                {
                    return static_cast<scalar_type>(
                        secondary_condition(
                            batch_id, in_batch_id).value);
                };
                turbulence_detail::reconstruct_gradient(
                    d_options.gradient_scheme,
                    state.candidate_secondary,
                    secondary_condition, secondary_value,
                    state.candidate_secondary_gradient,
                    state.gradient_cache);
            }
        });
    state.candidate_k_gradient.sync_ghosts();
    if (state.menter_family)
    {
        state.candidate_secondary_gradient.sync_ghosts();
    }

    stage_effective_properties(
        state, state.candidate_nu_t, material,
        reference_density,
        static_cast<scalar_type>(
            d_options.turbulent_prandtl_number));
    state.candidate_wall_velocity.owned_data().update(
        scalar_type{1}, velocity.owned_data(), scalar_type{0});
    state.candidate_wall_velocity.overlap_data().update(
        scalar_type{1}, velocity.overlap_data(), scalar_type{0});

    State::publish_synced_field(
        state.k, state.candidate_k);
    State::publish_synced_field(
        state.secondary, state.candidate_secondary);
    State::publish_synced_field(
        state.nu_t, state.candidate_nu_t);
    State::publish_synced_field(
        state.k_gradient, state.candidate_k_gradient);
    if (state.menter_family)
    {
        State::publish_synced_field(
            state.secondary_gradient,
            state.candidate_secondary_gradient);
    }
    commit_effective_properties(state);
    State::publish_synced_field(
        state.wall_velocity, state.candidate_wall_velocity);
    state.publish_wall_publication(
        std::move(wall_publication));
    d_wall_velocity_boundary_cache =
        std::move(velocity_boundary_cache);
}

/**
 * @brief Advance both turbulence scalars and publish derived properties.
 * @tparam Pack Tpetra type pack used by the model.
 * @param velocity Accepted liquid velocity field.
 * @param projected_face_fluxes Pressure-projected volumetric face fluxes.
 * @param velocity_boundary_cache Cached velocity boundary values.
 * @param time_step Positive physical time step.
 * @param material Current molecular material-property fields.
 * @param reference_density Positive momentum reference density.
 * @param treatment Non-orthogonal scalar diffusion treatment.
 * @param linear_options Linear-solver configuration.
 * @return Aggregated statistics from the two scalar solves.
 * @throws std::logic_error if the model is disabled or loses its closure.
 * @throws std::invalid_argument if fields or physical inputs are invalid.
 * @throws std::runtime_error if a solve or closure evaluation fails.
 * @throws std::overflow_error if a derived property is invalid.
 */
template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::advance(const velocity_field_type& velocity,
                                    const face_flux_field_type& projected_face_fluxes,
                                    const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
                                    scalar_type time_step, const material_type& material,
                                    scalar_type reference_density,
                                    FVM::NonOrthogonalTreatment treatment,
                                    const LinearSolverOptions& linear_options,
                                    const TurbulenceBuoyancyContext<Pack>*
                                        buoyancy_context) -> LinearSolveSummary
{
    turbulence_detail::require_uniform_integral(*d_mesh, enabled() ? 1 : 0,
                                                "Turbulence enabled state");
    auto& state = require_state();
    turbulence_detail::require_uniform_integral(*d_mesh, static_cast<int>(d_options.model),
                                                "Active turbulence model type");
    turbulence_detail::require_uniform_integral(*d_mesh, static_cast<int>(treatment),
                                                "Turbulence non-orthogonal treatment");
    turbulence_detail::require_uniform_real(*d_mesh, static_cast<real_t>(time_step),
                                            "Turbulence time step");
    turbulence_detail::require_uniform_real(*d_mesh, static_cast<real_t>(reference_density),
                                            "Turbulence reference density");
    const auto direct_buoyancy =
        d_options.buoyancy_model
        == TurbulenceBuoyancyModel::OpenFOAMBoussinesq;
    turbulence_detail::require_uniform_integral(
        *d_mesh, buoyancy_context != nullptr ? 1 : 0,
        "Turbulence buoyancy-context presence");
    if (direct_buoyancy && buoyancy_context != nullptr)
    {
        turbulence_detail::require_uniform_real(
            *d_mesh,
            static_cast<real_t>(
                buoyancy_context->thermal_expansion),
            "Turbulence thermal expansion");
        turbulence_detail::require_uniform_real(
            *d_mesh,
            static_cast<real_t>(buoyancy_context->gravity.x),
            "Turbulence gravity x");
        turbulence_detail::require_uniform_real(
            *d_mesh,
            static_cast<real_t>(buoyancy_context->gravity.y),
            "Turbulence gravity y");
        turbulence_detail::require_uniform_real(
            *d_mesh,
            static_cast<real_t>(buoyancy_context->gravity.z),
            "Turbulence gravity z");
        turbulence_detail::require_uniform_integral(
            *d_mesh,
            buoyancy_context->density_feedback_enabled ? 1 : 0,
            "Turbulence density-feedback mode");
    }
    turbulence_detail::collective_local_validation(
        *d_mesh, "Turbulence advance input validation",
        [&]
        {
            if (&velocity.mesh() != d_mesh.get() || &projected_face_fluxes.mesh() != d_mesh.get() ||
                velocity_boundary_cache.mesh.get() != d_mesh.get())
            {
                throw std::invalid_argument(
                    "TurbulenceModel advance inputs must use the model mesh.");
            }
            if (!std::isfinite(reference_density) || reference_density <= scalar_type{})
            {
                throw std::invalid_argument("TurbulenceModel requires a positive finite reference "
                                            "density.");
            }
            if (&material.dynamic_viscosity.mesh() != d_mesh.get())
            {
                throw std::invalid_argument("TurbulenceModel material field mesh mismatch.");
            }
            if (direct_buoyancy)
            {
                if (buoyancy_context == nullptr)
                {
                    throw std::invalid_argument(
                        "OpenFOAMBoussinesq turbulence production requires "
                        "a buoyancy context.");
                }
                const auto& context = *buoyancy_context;
                const auto finite_vector =
                    std::isfinite(context.gravity.x)
                    && std::isfinite(context.gravity.y)
                    && std::isfinite(context.gravity.z);
                if (!finite_vector
                    || !std::isfinite(context.thermal_expansion)
                    || context.thermal_expansion < scalar_type{})
                {
                    throw std::invalid_argument(
                        "Turbulence buoyancy requires finite gravity and a "
                        "finite non-negative thermal expansion.");
                }
                if (context.density_feedback_enabled)
                {
                    if (&material.density.mesh() != d_mesh.get())
                    {
                        throw std::invalid_argument(
                            "Turbulence buoyancy density field mesh "
                            "mismatch.");
                    }
                }
                else if (context.temperature == nullptr
                         || context.temperature_boundary_conditions
                            == nullptr
                         || &context.temperature->mesh() != d_mesh.get())
                {
                    throw std::invalid_argument(
                        "Temperature-based turbulence buoyancy requires a "
                        "temperature field and boundary map on the model "
                        "mesh.");
                }
            }
        });
    auto accepted_velocity_boundary_cache = velocity_boundary_cache;

    const auto velocity_local_values = velocity.local_read_view();
    auto boundary_velocity = [&](int batch_id, size_t in_batch_id) ->
        typename velocity_field_type::vec_type
    {
        const auto& batch = d_mesh->boundary_batches().at(batch_id);
        const auto face_lid = batch.face_lids.at(in_batch_id);
        const auto type = velocity_boundary_cache.type.at(batch_id);
        if (type == BoundaryConditionType::Slip)
        {
            return FVM::slip_face_velocity(velocity, face_lid);
        }
        if (type == BoundaryConditionType::Periodic)
        {
            const auto owner = d_mesh->owner_cell(face_lid);
            return {velocity_local_values(owner, 0),
                    velocity_local_values(owner, 1),
                    velocity_local_values(owner, 2)};
        }
        if (type == BoundaryConditionType::Neumann)
        {
            const auto owner = d_mesh->owner_cell(face_lid);
            const typename velocity_field_type::vec_type owner_value{
                velocity_local_values(owner, 0),
                velocity_local_values(owner, 1),
                velocity_local_values(owner, 2)};
            const auto name = d_mesh->boundary_batch_name(batch_id);
            const auto condition = d_velocity_boundary_conditions.find(name);
            if (condition == d_velocity_boundary_conditions.end())
            {
                return owner_value;
            }
            const auto distance = FVM::detail::boundary_normal_distance(*d_mesh, face_lid, owner);
            const typename velocity_field_type::vec_type derivative{
                static_cast<scalar_type>(condition->second.value.x),
                static_cast<scalar_type>(condition->second.value.y),
                static_cast<scalar_type>(condition->second.value.z)};
            return owner_value + derivative * static_cast<scalar_type>(distance);
        }
        return velocity_boundary_cache.value.at(batch_id).at(in_batch_id);
    };
    auto current_wall_evaluation = state.evaluate_wall(
        state.k, velocity, velocity_boundary_cache, material,
        reference_density,
        static_cast<scalar_type>(d_options.turbulent_prandtl_number),
        &state.wall_evaluation);
    auto update_scalar_gradients =
        [&, this](const field_type& k_field, const field_type& secondary_field,
                  velocity_field_type& k_gradient,
                  velocity_field_type& secondary_gradient,
                  const typename State::wall_evaluation_type& wall_evaluation)
    {
        auto k_condition = [&](int batch_id, size_t in_batch_id)
        {
            const auto wall = State::wall_boundary_condition(
                wall_evaluation, batch_id, in_batch_id, true);
            if (wall)
                return *wall;
            const auto& name = d_mesh->boundary_batch_name(batch_id);
            const auto iter = d_boundary_conditions.turbulent_kinetic_energy.find(name);
            return iter == d_boundary_conditions.turbulent_kinetic_energy.end()
                 ? BoundaryCondition{}
                 : iter->second;
        };
        auto k_value = [&](int batch_id, size_t in_batch_id)
        { return static_cast<scalar_type>(k_condition(batch_id, in_batch_id).value); };
        turbulence_detail::reconstruct_gradient(
            d_options.gradient_scheme,
            k_field, k_condition, k_value, k_gradient,
            state.gradient_cache);
        if (state.menter_family)
        {
            auto secondary_condition = [&](int batch_id, size_t in_batch_id)
            {
                const auto wall = State::wall_boundary_condition(
                    wall_evaluation, batch_id, in_batch_id, false);
                if (wall)
                    return *wall;
                const auto& name = d_mesh->boundary_batch_name(batch_id);
                const auto iter =
                    d_boundary_conditions.specific_dissipation_rate.find(name);
                return iter == d_boundary_conditions.specific_dissipation_rate.end()
                     ? BoundaryCondition{}
                     : iter->second;
            };
            auto secondary_value = [&](int batch_id, size_t in_batch_id)
            {
                return static_cast<scalar_type>(
                    secondary_condition(batch_id, in_batch_id).value);
            };
            turbulence_detail::reconstruct_gradient(
                d_options.gradient_scheme,
                secondary_field, secondary_condition,
                secondary_value, secondary_gradient,
                state.gradient_cache);
        }
    };
    auto sync_scalar_gradients = [&](velocity_field_type& k_gradient,
                                     velocity_field_type& secondary_gradient)
    {
        k_gradient.sync_ghosts();
        if (state.menter_family)
        {
            secondary_gradient.sync_ghosts();
        }
    };
    turbulence_detail::collective_local_validation(
        *d_mesh, "Turbulence gradient reconstruction",
        [&]
        {
            turbulence_detail::reconstruct_gradient(
                d_options.gradient_scheme,
                velocity, boundary_velocity,
                state.candidate_velocity_gradient,
                state.gradient_cache);
            update_scalar_gradients(state.k, state.secondary,
                                    state.candidate_k_gradient,
                                    state.candidate_secondary_gradient,
                                    current_wall_evaluation);
            if (direct_buoyancy)
            {
                if (buoyancy_context->density_feedback_enabled)
                {
                    turbulence_detail::reconstruct_gradient(
                        d_options.gradient_scheme,
                        material.density, state.buoyancy_gradient,
                        state.gradient_cache);
                }
                else
                {
                    turbulence_detail::reconstruct_gradient(
                        d_options.gradient_scheme,
                        *buoyancy_context->temperature,
                        *buoyancy_context
                             ->temperature_boundary_conditions,
                        state.buoyancy_gradient,
                        state.gradient_cache);
                }
            }
        });
    state.candidate_velocity_gradient.sync_ghosts();
    sync_scalar_gradients(
        state.candidate_k_gradient,
        state.candidate_secondary_gradient);
    if (direct_buoyancy)
    {
        state.buoyancy_gradient.sync_ghosts();
    }

    auto evaluate_closure =
        [&, this](const field_type& k_field, const field_type& secondary_field,
                  const velocity_field_type& k_gradient,
                  const velocity_field_type& secondary_gradient,
                  const typename State::wall_evaluation_type& wall_evaluation)
    {
        turbulence_detail::collective_local_validation(
            *d_mesh, "Turbulence closure evaluation",
            [&]
            {
                const auto k_values = k_field.owned_read_view();
                const auto secondary_values =
                    secondary_field.owned_read_view();
                const auto molecular_viscosity_values =
                    material.dynamic_viscosity.owned_read_view();
                const auto velocity_gradient_values =
                    state.candidate_velocity_gradient.owned_read_view();
                const auto k_gradient_values =
                    k_gradient.owned_read_view();
                const auto secondary_gradient_values =
                    secondary_gradient.owned_read_view();
                const auto wall_distance_values =
                    state.wall_distance.owned_read_view();
                const auto buoyancy_gradient_values =
                    state.buoyancy_gradient.owned_read_view();
                const auto velocity_values =
                    velocity.owned_read_view();

                auto turbulent_viscosity_values =
                    state.candidate_nu_t.owned_write_view();
                auto k_diffusivity_values =
                    state.k_diffusivity.owned_write_view();
                auto secondary_diffusivity_values =
                    state.secondary_diffusivity.owned_write_view();
                auto k_source_values =
                    state.k_source.owned_write_view();
                auto k_sink_values =
                    state.k_sink.owned_write_view();
                auto secondary_source_values =
                    state.secondary_source.owned_write_view();
                auto secondary_sink_values =
                    state.secondary_sink.owned_write_view();
                auto buoyancy_production_values =
                    state.candidate_buoyancy_production
                        .owned_write_view();
                for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
                {
                    const auto cell_lid = static_cast<local_ordinal_type>(owned);
                    const auto k =
                        static_cast<real_t>(k_values(cell_lid, 0));
                    const auto secondary = static_cast<real_t>(
                        secondary_values(cell_lid, 0));
                    const auto molecular_viscosity =
                        static_cast<real_t>(
                            molecular_viscosity_values(cell_lid, 0));
                    if (!std::isfinite(molecular_viscosity) || molecular_viscosity < 0.0)
                    {
                        throw std::invalid_argument(
                            "TurbulenceModel requires finite non-negative molecular "
                            "viscosity.");
                    }
                    const auto molecular_nu =
                        molecular_viscosity / static_cast<real_t>(reference_density);

                    std::array<std::array<real_t, 3>, 3> strain{};
                    real_t strain_squared{};
                    real_t rotation_squared{};
                    for (size_t row = 0; row < 3; ++row)
                    {
                        for (size_t column = 0; column < 3; ++column)
                        {
                            const auto gij = static_cast<real_t>(
                                velocity_gradient_values(
                                    cell_lid, row * 3 + column));
                            const auto gji = static_cast<real_t>(
                                velocity_gradient_values(
                                    cell_lid, column * 3 + row));
                            const auto sij = 0.5 * (gij + gji);
                            const auto wij = 0.5 * (gij - gji);
                            strain[row][column] = sij;
                            strain_squared += sij * sij;
                            rotation_squared += wij * wij;
                        }
                    }
                    const auto strain_magnitude = std::sqrt(2.0 * strain_squared);
                    const auto vorticity_magnitude = std::sqrt(2.0 * rotation_squared);
                    const auto u_star = std::sqrt(strain_squared + rotation_squared);
                    real_t strain_cubed_trace{};
                    for (size_t i = 0; i < 3; ++i)
                        for (size_t j = 0; j < 3; ++j)
                            for (size_t l = 0; l < 3; ++l)
                                strain_cubed_trace += strain[i][j] * strain[j][l] * strain[l][i];
                    const auto strain_denominator = std::pow(strain_squared, 1.5);
                    const auto normalized_strain_third_invariant =
                        strain_denominator > 0.0 ? strain_cubed_trace / strain_denominator : 0.0;

                    real_t nu_t{};
                    real_t k_diffusivity{};
                    real_t secondary_diffusivity{};
                    real_t explicit_k_source{};
                    real_t implicit_k_sink{};
                    real_t explicit_secondary_source{};
                    real_t implicit_secondary_sink{};
                    real_t buoyancy_secondary_multiplier{};
                    switch (d_options.model)
                    {
                    case TurbulenceModelType::StandardKEpsilon:
                    {
                        const auto& closure = std::get<StandardKEpsilonEquation>(state.closure);
                        const KEpsilonState local{k, secondary};
                        nu_t = closure.turbulent_kinematic_viscosity(local);
                        const auto diffusion = closure.diffusivities(local, molecular_nu);
                        const auto production = State::wall_production_override(
                                                    wall_evaluation, cell_lid)
                                                    .value_or(
                                                        nu_t * strain_magnitude *
                                                        strain_magnitude);
                        const auto& coefficients = closure.coefficients();
                        k_diffusivity = diffusion.k;
                        secondary_diffusivity = diffusion.epsilon;
                        explicit_k_source = production;
                        implicit_k_sink = secondary / k;
                        explicit_secondary_source =
                            coefficients.c_epsilon_1 * production * secondary / k;
                        implicit_secondary_sink = coefficients.c_epsilon_2 * secondary / k;
                        buoyancy_secondary_multiplier =
                            coefficients.c_epsilon_1 * secondary / k;
                        break;
                    }
                    case TurbulenceModelType::RNGKEpsilon:
                    {
                        const auto& closure = std::get<RNGKEpsilonEquation>(state.closure);
                        const KEpsilonState local{k, secondary};
                        nu_t = closure.turbulent_kinematic_viscosity(local);
                        const auto diffusion = closure.diffusivities(local, molecular_nu);
                        const auto production = nu_t * strain_magnitude * strain_magnitude;
                        const auto correction = closure.rng_correction(local, strain_magnitude);
                        const auto& coefficients = closure.coefficients();
                        k_diffusivity = diffusion.k;
                        secondary_diffusivity = diffusion.epsilon;
                        explicit_k_source = production;
                        implicit_k_sink = secondary / k;
                        explicit_secondary_source =
                            coefficients.c_epsilon_1 * production * secondary / k +
                            std::max(-correction, 0.0);
                        implicit_secondary_sink = coefficients.c_epsilon_2 * secondary / k +
                                                  std::max(correction / secondary, 0.0);
                        buoyancy_secondary_multiplier =
                            coefficients.c_epsilon_1 * secondary / k;
                        break;
                    }
                    case TurbulenceModelType::RealizableKEpsilon:
                    {
                        const auto& closure = std::get<RealizableKEpsilonEquation>(state.closure);
                        const KEpsilonState local{k, secondary};
                        const RealizableKEpsilonInvariants invariants{
                            strain_magnitude, u_star, normalized_strain_third_invariant,
                            molecular_nu};
                        nu_t = closure.turbulent_kinematic_viscosity(local, invariants);
                        const auto diffusion = closure.diffusivities(local, invariants);
                        const auto production =
                            State::wall_production_override(
                                wall_evaluation, cell_lid)
                                .value_or(
                                    nu_t * strain_magnitude *
                                    strain_magnitude);
                        const auto epsilon_denominator = k + std::sqrt(molecular_nu * secondary);
                        const auto& coefficients = closure.coefficients();
                        k_diffusivity = diffusion.k;
                        secondary_diffusivity = diffusion.epsilon;
                        explicit_k_source = production;
                        implicit_k_sink = secondary / k;
                        explicit_secondary_source =
                            closure.epsilon_production_coefficient(local, strain_magnitude) *
                            strain_magnitude * secondary;
                        implicit_secondary_sink =
                            coefficients.c_epsilon_2 * secondary / epsilon_denominator;
                        buoyancy_secondary_multiplier =
                            1.44 * secondary / k;
                        break;
                    }
                    case TurbulenceModelType::StandardKOmega:
                    {
                        const auto& closure = std::get<StandardKOmegaEquation>(state.closure);
                        const KOmegaState local{k, secondary};
                        nu_t = closure.turbulent_kinematic_viscosity(local);
                        const auto diffusion = closure.diffusivities(local, molecular_nu);
                        const auto production = nu_t * strain_magnitude * strain_magnitude;
                        const auto& coefficients = closure.coefficients();
                        k_diffusivity = diffusion.k;
                        secondary_diffusivity = diffusion.omega;
                        explicit_k_source = production;
                        implicit_k_sink = coefficients.beta_star * secondary;
                        explicit_secondary_source = coefficients.gamma * production * secondary / k;
                        implicit_secondary_sink = coefficients.beta * secondary;
                        buoyancy_secondary_multiplier =
                            coefficients.gamma
                          / (nu_t
                             + std::numeric_limits<real_t>::epsilon());
                        break;
                    }
                    case TurbulenceModelType::BSLKOmega:
                    case TurbulenceModelType::SSTKOmega:
                    {
                        const KOmegaState local{k, secondary};
                        const typename velocity_field_type::vec_type
                            local_k_gradient{
                                k_gradient_values(cell_lid, 0),
                                k_gradient_values(cell_lid, 1),
                                k_gradient_values(cell_lid, 2)};
                        const typename velocity_field_type::vec_type
                            omega_gradient{
                                secondary_gradient_values(cell_lid, 0),
                                secondary_gradient_values(cell_lid, 1),
                                secondary_gradient_values(cell_lid, 2)};
                        const MenterKOmegaInvariants invariants{
                            molecular_nu,
                            static_cast<real_t>(
                                wall_distance_values(cell_lid, 0)),
                            static_cast<real_t>(local_k_gradient.dot(omega_gradient)),
                            vorticity_magnitude};
                        if (d_options.model == TurbulenceModelType::BSLKOmega)
                        {
                            const auto& closure = std::get<BSLKOmegaEquation>(state.closure);
                            nu_t = closure.turbulent_kinematic_viscosity(local);
                            const auto diffusion = closure.diffusivities(local, invariants);
                            const auto production = nu_t * strain_magnitude * strain_magnitude;
                            const auto f1 = closure.blending_function_1(local, invariants);
                            const auto coefficients = closure.blended_coefficients(f1);
                            const auto cross_diffusion =
                                closure.cross_diffusion_source(local, invariants, f1);
                            k_diffusivity = diffusion.k;
                            secondary_diffusivity = diffusion.omega;
                            explicit_k_source = closure.limited_production(local, production);
                            implicit_k_sink = closure.coefficients().beta_star * secondary;
                            explicit_secondary_source = coefficients.gamma * production / nu_t +
                                                        std::max(cross_diffusion, 0.0);
                            implicit_secondary_sink = coefficients.beta * secondary +
                                                      std::max(-cross_diffusion / secondary, 0.0);
                            buoyancy_secondary_multiplier =
                                coefficients.gamma
                              / (nu_t
                                 + std::numeric_limits<real_t>::epsilon());
                        }
                        else
                        {
                            const auto& closure = std::get<SSTKOmegaEquation>(state.closure);
                            nu_t = closure.turbulent_kinematic_viscosity(local, invariants);
                            const auto diffusion = closure.diffusivities(local, invariants);
                            const auto production = nu_t * strain_magnitude * strain_magnitude;
                            const auto f1 = closure.blending_function_1(local, invariants);
                            const auto coefficients = closure.blended_coefficients(f1);
                            const auto cross_diffusion =
                                closure.cross_diffusion_source(local, invariants, f1);
                            k_diffusivity = diffusion.k;
                            secondary_diffusivity = diffusion.omega;
                            explicit_k_source = closure.limited_production(local, production);
                            implicit_k_sink = closure.coefficients().beta_star * secondary;
                            explicit_secondary_source = coefficients.gamma * production / nu_t +
                                                        std::max(cross_diffusion, 0.0);
                            implicit_secondary_sink = coefficients.beta * secondary +
                                                      std::max(-cross_diffusion / secondary, 0.0);
                            buoyancy_secondary_multiplier =
                                coefficients.gamma
                              / (nu_t
                                 + std::numeric_limits<real_t>::epsilon());
                        }
                        break;
                    }
                    case TurbulenceModelType::Laminar:
                        throw std::logic_error("TurbulenceModel lost its active closure.");
                    }

                    real_t local_buoyancy_production{};
                    if (direct_buoyancy)
                    {
                        const typename velocity_field_type::vec_type
                            gradient{
                                buoyancy_gradient_values(cell_lid, 0),
                                buoyancy_gradient_values(cell_lid, 1),
                                buoyancy_gradient_values(cell_lid, 2)};
                        const auto gravity = buoyancy_context->gravity;
                        const auto gravity_dot_gradient =
                            static_cast<real_t>(
                                gravity.dot(gradient));
                        const auto turbulent_thermal_diffusivity =
                            nu_t / d_options.turbulent_prandtl_number;
                        if (buoyancy_context->density_feedback_enabled)
                        {
                            local_buoyancy_production =
                                -d_options.buoyancy_coefficient
                              * turbulent_thermal_diffusivity
                              * gravity_dot_gradient
                              / static_cast<real_t>(reference_density);
                        }
                        else
                        {
                            // OpenFOAM's incompressible
                            // fv::buoyancyTurbSource defines
                            // B = beta*alphat*(grad(T) & g), then inserts
                            // it with `eqn -= Sp(B/k, k)`.  In this
                            // transport assembly, positive values are
                            // explicit production and negative values are
                            // implicit sinks, so the equivalent signed
                            // source is -B.
                            local_buoyancy_production =
                                -d_options.buoyancy_coefficient
                              * static_cast<real_t>(
                                    buoyancy_context->thermal_expansion)
                              * turbulent_thermal_diffusivity
                              * gravity_dot_gradient;
                        }

                        real_t c3 = 1.0;
                        if (state.epsilon_family)
                        {
                            const auto gravity_magnitude =
                                std::sqrt(
                                    static_cast<real_t>(
                                        gravity.dot(gravity)));
                            if (gravity_magnitude
                                <= std::numeric_limits<real_t>::epsilon())
                            {
                                c3 = 0.0;
                            }
                            else
                            {
                                const typename velocity_field_type::vec_type
                                    velocity_value{
                                        velocity_values(cell_lid, 0),
                                        velocity_values(cell_lid, 1),
                                        velocity_values(cell_lid, 2)};
                                const auto parallel =
                                    static_cast<real_t>(
                                        velocity_value.dot(gravity))
                                    / gravity_magnitude;
                                const auto velocity_squared =
                                    static_cast<real_t>(
                                        velocity_value.dot(
                                            velocity_value));
                                const auto perpendicular =
                                    std::sqrt(std::max(
                                        velocity_squared
                                      - parallel * parallel,
                                        real_t{}));
                                const auto parallel_magnitude =
                                    std::abs(parallel);
                                // Match OpenFOAM's incompressible
                                // fv::buoyancyTurbSource.  Its Boussinesq
                                // temperature source uses
                                // tanh(|U_perpendicular + SMALL|
                                //      / |U_parallel|), which is the
                                // opposite orientation convention from the
                                // compressible buoyantKEpsilon model.
                                c3 =
                                    parallel_magnitude
                                        <= std::numeric_limits<
                                               real_t>::epsilon()
                                      ? real_t{1}
                                      : std::tanh(
                                            (perpendicular
                                             + std::numeric_limits<
                                                   real_t>::epsilon())
                                            / parallel_magnitude);
                            }
                        }

                        const auto secondary_buoyancy =
                            buoyancy_secondary_multiplier
                          * c3
                          * local_buoyancy_production;
                        if (local_buoyancy_production >= 0.0)
                        {
                            explicit_k_source +=
                                local_buoyancy_production;
                        }
                        else
                        {
                            implicit_k_sink +=
                                -local_buoyancy_production / k;
                        }
                        if (secondary_buoyancy >= 0.0)
                        {
                            explicit_secondary_source +=
                                secondary_buoyancy;
                        }
                        else
                        {
                            implicit_secondary_sink +=
                                -secondary_buoyancy / secondary;
                        }
                    }

                    const real_t values[] = {nu_t,
                                             k_diffusivity,
                                             secondary_diffusivity,
                                             explicit_k_source,
                                             implicit_k_sink,
                                             explicit_secondary_source,
                                             implicit_secondary_sink,
                                             local_buoyancy_production};
                    for (const auto value : values)
                    {
                        if (!std::isfinite(value))
                        {
                            throw std::runtime_error(
                                "Turbulence closure produced a non-finite value.");
                        }
                    }
                    if (nu_t < 0.0 || k_diffusivity < 0.0 || secondary_diffusivity < 0.0 ||
                        explicit_k_source < 0.0 || implicit_k_sink < 0.0 ||
                        explicit_secondary_source < 0.0 || implicit_secondary_sink < 0.0)
                    {
                        throw std::runtime_error(
                            "Turbulence closure produced a negative source, sink, "
                            "or transport coefficient.");
                    }

                    turbulent_viscosity_values(cell_lid, 0) =
                        static_cast<scalar_type>(nu_t);
                    k_diffusivity_values(cell_lid, 0) =
                        static_cast<scalar_type>(k_diffusivity);
                    secondary_diffusivity_values(cell_lid, 0) =
                        static_cast<scalar_type>(secondary_diffusivity);
                    k_source_values(cell_lid, 0) =
                        static_cast<scalar_type>(explicit_k_source);
                    k_sink_values(cell_lid, 0) =
                        static_cast<scalar_type>(implicit_k_sink);
                    secondary_source_values(cell_lid, 0) =
                        static_cast<scalar_type>(
                            explicit_secondary_source);
                    secondary_sink_values(cell_lid, 0) =
                        static_cast<scalar_type>(
                            implicit_secondary_sink);
                    buoyancy_production_values(cell_lid, 0) =
                        static_cast<scalar_type>(
                            local_buoyancy_production);
                }
            });
        state.candidate_nu_t.sync_ghosts();
        state.k_diffusivity.sync_ghosts();
        state.secondary_diffusivity.sync_ghosts();
        state.candidate_buoyancy_production.sync_ghosts();
    };
    evaluate_closure(
        state.k, state.secondary, state.candidate_k_gradient,
        state.candidate_secondary_gradient, current_wall_evaluation);

    auto summary = [&]
    {
        const auto k_source_values =
            state.k_source.owned_read_view();
        const auto k_sink_values =
            state.k_sink.owned_read_view();
        const auto secondary_source_values =
            state.secondary_source.owned_read_view();
        const auto secondary_sink_values =
            state.secondary_sink.owned_read_view();
        auto k_source = [&](local_ordinal_type cell_lid)
        { return k_source_values(cell_lid, 0); };
        auto k_sink = [&](local_ordinal_type cell_lid)
        { return k_sink_values(cell_lid, 0); };
        auto secondary_source = [&](local_ordinal_type cell_lid)
        { return secondary_source_values(cell_lid, 0); };
        auto secondary_sink = [&](local_ordinal_type cell_lid)
        { return secondary_sink_values(cell_lid, 0); };

        TurbulenceScalarBoundaryOverrides<Pack> k_wall_overrides;
        k_wall_overrides.boundary_condition =
            [&](int batch_id, size_t in_batch_id)
        {
            return State::wall_boundary_condition(
                current_wall_evaluation, batch_id, in_batch_id, true);
        };
        k_wall_overrides.boundary_value =
            [&](int batch_id, size_t in_batch_id)
                -> std::optional<scalar_type>
        {
            const auto condition = State::wall_boundary_condition(
                current_wall_evaluation, batch_id, in_batch_id, true);
            return condition &&
                           condition->type ==
                               BoundaryConditionType::Dirichlet
                     ? std::optional<scalar_type>{
                           static_cast<scalar_type>(condition->value)}
                     : std::nullopt;
        };
        k_wall_overrides.allow_zero_dirichlet =
            d_options.wall_treatment ==
                TurbulenceWallTreatmentType::ResolvedLowReSST ||
            d_options.wall_treatment ==
                TurbulenceWallTreatmentType::ResolvedLowReKEpsilon;
        k_wall_overrides.boundary_diffusivity =
            State::boundary_scalar_diffusivity(
                current_wall_evaluation);

        TurbulenceScalarBoundaryOverrides<Pack>
            secondary_wall_overrides;
        secondary_wall_overrides.boundary_condition =
            [&](int batch_id, size_t in_batch_id)
        {
            return State::wall_boundary_condition(
                current_wall_evaluation, batch_id, in_batch_id,
                false);
        };
        secondary_wall_overrides.boundary_value =
            [&](int batch_id, size_t in_batch_id)
                -> std::optional<scalar_type>
        {
            const auto condition = State::wall_boundary_condition(
                current_wall_evaluation, batch_id, in_batch_id,
                false);
            return condition &&
                           condition->type ==
                               BoundaryConditionType::Dirichlet
                     ? std::optional<scalar_type>{
                           static_cast<scalar_type>(condition->value)}
                     : std::nullopt;
        };
        secondary_wall_overrides.fixed_cell_value =
            [&](local_ordinal_type cell_lid)
        {
            auto value = State::wall_secondary_constraint(
                current_wall_evaluation, cell_lid);
            if (value)
            {
                *value = std::max(
                    *value,
                    static_cast<scalar_type>(
                        state.epsilon_family
                            ? d_options.min_dissipation_rate
                            : d_options
                                  .min_specific_dissipation_rate));
            }
            return value;
        };
        secondary_wall_overrides.boundary_diffusivity =
            State::boundary_scalar_diffusivity(
                current_wall_evaluation);

        LinearSolveSummary result;
        result.add(state.k_equation.advance(
            state.k, projected_face_fluxes, time_step,
            state.k_diffusivity, state.candidate_k, k_source,
            k_sink,
            static_cast<scalar_type>(
                d_options.min_turbulent_kinetic_energy),
            treatment, linear_options, &k_wall_overrides,
            d_options.coefficient_interpolation));
        result.add(state.secondary_equation.advance(
            state.secondary, projected_face_fluxes, time_step,
            state.secondary_diffusivity,
            state.candidate_secondary, secondary_source,
            secondary_sink,
            static_cast<scalar_type>(
                state.epsilon_family
                    ? d_options.min_dissipation_rate
                    : d_options.min_specific_dissipation_rate),
            treatment, linear_options, &secondary_wall_overrides,
            d_options.coefficient_interpolation));
        return result;
    }();

    auto candidate_wall_evaluation = state.evaluate_wall(
        state.candidate_k, velocity, velocity_boundary_cache, material,
        reference_density,
        static_cast<scalar_type>(d_options.turbulent_prandtl_number),
        &state.wall_evaluation);
    auto candidate_wall_publication =
        state.stage_wall_publication(
            std::move(candidate_wall_evaluation));

    turbulence_detail::collective_local_validation(
        *d_mesh, "Turbulence candidate-gradient reconstruction",
        [&]
        {
            update_scalar_gradients(state.candidate_k, state.candidate_secondary,
                                    state.candidate_k_gradient,
                                    state.candidate_secondary_gradient,
                                    candidate_wall_publication.evaluation);
        });
    sync_scalar_gradients(state.candidate_k_gradient,
                          state.candidate_secondary_gradient);
    evaluate_closure(state.candidate_k, state.candidate_secondary,
                     state.candidate_k_gradient,
                     state.candidate_secondary_gradient,
                     candidate_wall_publication.evaluation);
    stage_effective_properties(state, state.candidate_nu_t, material, reference_density,
                               static_cast<scalar_type>(d_options.turbulent_prandtl_number));
    state.candidate_wall_velocity.owned_data().update(
        scalar_type{1}, velocity.owned_data(), scalar_type{0});
    state.candidate_wall_velocity.overlap_data().update(
        scalar_type{1}, velocity.overlap_data(), scalar_type{0});

    // Publish only after both solves and all derived-field validation pass.
    State::publish_synced_field(state.k, state.candidate_k);
    State::publish_synced_field(
        state.secondary, state.candidate_secondary);
    State::publish_synced_field(state.nu_t, state.candidate_nu_t);
    State::publish_synced_field(
        state.buoyancy_production,
        state.candidate_buoyancy_production);
    State::publish_synced_field(
        state.k_gradient, state.candidate_k_gradient);
    if (state.menter_family)
    {
        State::publish_synced_field(
            state.secondary_gradient,
            state.candidate_secondary_gradient);
    }
    State::publish_synced_field(
        state.velocity_gradient,
        state.candidate_velocity_gradient);
    commit_effective_properties(state);
    State::publish_synced_field(
        state.wall_velocity, state.candidate_wall_velocity);
    state.publish_wall_publication(
        std::move(candidate_wall_publication));
    d_wall_velocity_boundary_cache =
        std::move(accepted_velocity_boundary_cache);
    return summary;
}

/**
 * @brief Transactionally replace the wall distance and its dependent properties.
 * @tparam Pack Tpetra type pack used by the model.
 * @param wall_distance Positive distance-to-wall field.
 * @param material Current molecular material-property fields.
 * @param reference_density Positive momentum reference density.
 * @throws std::logic_error if the model is disabled.
 * @throws std::logic_error if the active closure is not BSL or SST k-omega.
 * @throws std::invalid_argument if a field mesh or physical value is invalid.
 * @throws std::overflow_error if an effective-property calculation is invalid.
 */
template <TpetraTypePack Pack>
void TurbulenceModel<Pack>::set_wall_distance(
    const field_type& wall_distance, const material_type& material,
    scalar_type reference_density)
{
    turbulence_detail::require_uniform_integral(*d_mesh, enabled() ? 1 : 0,
                                                "Turbulence enabled state");
    auto& state = require_state();
    turbulence_detail::require_uniform_integral(
        *d_mesh, static_cast<int>(d_options.model),
        "Active turbulence model type");
    turbulence_detail::require_uniform_real(
        *d_mesh, static_cast<real_t>(reference_density),
        "Turbulence reference density");
    turbulence_detail::collective_local_validation(
        *d_mesh, "Turbulence wall-distance validation",
        [&]
        {
            if (!state.menter_family)
            {
                throw std::logic_error(
                    "TurbulenceModel wall distance is available only for "
                    "BSL or SST k-omega closures.");
            }
            if (&wall_distance.mesh() != d_mesh.get())
            {
                throw std::invalid_argument("TurbulenceModel wall-distance mesh mismatch.");
            }
            const field_type* material_fields[] = {
                &material.density,
                &material.specific_heat_capacity,
                &material.dynamic_viscosity,
                &material.thermal_conductivity};
            for (const auto* field : material_fields)
            {
                if (&field->mesh() != d_mesh.get())
                {
                    throw std::invalid_argument(
                        "TurbulenceModel wall-distance material mesh mismatch.");
                }
            }
            if (!std::isfinite(reference_density)
                || reference_density <= scalar_type{})
            {
                throw std::invalid_argument(
                    "TurbulenceModel wall-distance refresh requires a "
                    "positive finite reference density.");
            }
            for (size_t owned = 0;
                 owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell_lid =
                    static_cast<local_ordinal_type>(owned);
                const auto value = wall_distance.value(cell_lid);
                if (!std::isfinite(value) || value <= scalar_type{})
                {
                    throw std::invalid_argument("TurbulenceModel wall distance must be finite and "
                                                "positive.");
                }
            }
        });

    field_type candidate_wall_distance(
        d_mesh, "wall_distance_candidate");
    candidate_wall_distance.owned_data().update(
        scalar_type{1}, wall_distance.owned_data(), scalar_type{0});
    candidate_wall_distance.sync_ghosts();

    turbulence_detail::collective_local_validation(
        *d_mesh, "Synchronized turbulence wall-distance validation",
        [&]
        {
            for (size_t local = 0;
                 local < d_mesh->num_local_cells(); ++local)
            {
                const auto cell_lid =
                    static_cast<local_ordinal_type>(local);
                const auto value =
                    candidate_wall_distance.local_value(cell_lid);
                if (!std::isfinite(value)
                    || value <= scalar_type{})
                {
                    throw std::invalid_argument(
                        "TurbulenceModel synchronized wall distance must "
                        "be finite and positive.");
                }
            }
        });
    stage_menter_eddy_viscosity(
        state, candidate_wall_distance, material, reference_density,
        "Turbulence wall-distance closure evaluation");
    stage_effective_properties(
        state, state.candidate_nu_t, material, reference_density,
        static_cast<scalar_type>(
            d_options.turbulent_prandtl_number));

    // Publish only after the replacement and every dependent field validate.
    State::publish_synced_field(
        state.wall_distance, candidate_wall_distance);
    State::publish_synced_field(state.nu_t, state.candidate_nu_t);
    commit_effective_properties(state);
}

/**
 * @brief Return the accepted turbulent kinetic-energy field.
 * @tparam Pack Tpetra type pack used by the model.
 * @return Accepted k field.
 * @throws std::logic_error if the model is disabled.
 */
template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::turbulent_kinetic_energy() const -> const field_type&
{
    return require_state().k;
}

/**
 * @brief Return the accepted turbulent kinetic-energy gradient.
 * @tparam Pack Tpetra type pack used by the model.
 * @return Accepted gradient of k.
 * @throws std::logic_error if the model is disabled.
 */
template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::turbulent_kinetic_energy_gradient() const
    -> const velocity_field_type&
{
    return require_state().k_gradient;
}

/**
 * @brief Return the accepted epsilon field for epsilon-family closures.
 * @tparam Pack Tpetra type pack used by the model.
 * @return Epsilon field, or null for disabled and omega-family models.
 */
template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::dissipation_rate() const noexcept -> const field_type*
{
    return d_state && d_state->epsilon_family ? &d_state->secondary : nullptr;
}

/**
 * @brief Return the accepted omega field for omega-family closures.
 * @tparam Pack Tpetra type pack used by the model.
 * @return Omega field, or null for disabled and epsilon-family models.
 */
template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::specific_dissipation_rate() const noexcept -> const field_type*
{
    return d_state && !d_state->epsilon_family ? &d_state->secondary : nullptr;
}

/**
 * @brief Return the accepted turbulent kinematic-viscosity field.
 * @tparam Pack Tpetra type pack used by the model.
 * @return Accepted eddy-viscosity field.
 * @throws std::logic_error if the model is disabled.
 */
template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::turbulent_kinematic_viscosity() const -> const field_type&
{
    return require_state().nu_t;
}

/**
 * @brief Return the accepted effective dynamic-viscosity field.
 * @tparam Pack Tpetra type pack used by the model.
 * @return Molecular plus turbulent dynamic viscosity.
 * @throws std::logic_error if the model is disabled.
 */
template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::effective_dynamic_viscosity() const -> const field_type&
{
    return require_state().effective_dynamic_viscosity;
}

/**
 * @brief Return the accepted effective thermal-conductivity field.
 * @tparam Pack Tpetra type pack used by the model.
 * @return Molecular plus turbulent thermal conductivity.
 * @throws std::logic_error if the model is disabled.
 */
template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::effective_thermal_conductivity() const -> const field_type&
{
    return require_state().effective_thermal_conductivity;
}

template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::buoyancy_production() const noexcept
    -> const field_type*
{
    return d_state
            && d_options.buoyancy_model
               != TurbulenceBuoyancyModel::None
         ? &d_state->buoyancy_production
         : nullptr;
}

template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::turbulent_kinetic_energy_source() const noexcept
    -> const field_type*
{
    return d_state ? &d_state->k_source : nullptr;
}

template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::turbulent_kinetic_energy_sink() const noexcept
    -> const field_type*
{
    return d_state ? &d_state->k_sink : nullptr;
}

template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::secondary_source() const noexcept
    -> const field_type*
{
    return d_state ? &d_state->secondary_source : nullptr;
}

template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::secondary_sink() const noexcept
    -> const field_type*
{
    return d_state ? &d_state->secondary_sink : nullptr;
}

template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::wall_distance() const noexcept
    -> const field_type*
{
    return d_state && d_state->menter_family
         ? &d_state->wall_distance
         : nullptr;
}

/**
 * @brief Return sparse wall-face effective viscosities when available.
 * @tparam Pack Tpetra type pack used by the model.
 * @return Active wall viscosity cache, or null.
 */
template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::effective_dynamic_viscosity_boundary_cache() const noexcept
    -> const FVM::BoundaryCache<Pack>*
{
    return d_state
         ? State::boundary_dynamic_viscosity(d_state->wall_evaluation)
         : nullptr;
}

/**
 * @brief Return sparse wall-face effective conductivities when available.
 * @tparam Pack Tpetra type pack used by the model.
 * @return Active wall conductivity cache, or null.
 */
template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::effective_thermal_conductivity_boundary_cache() const noexcept
    -> const FVM::BoundaryCache<Pack>*
{
    return d_state
         ? State::boundary_thermal_conductivity(d_state->wall_evaluation)
         : nullptr;
}

/**
 * @brief Return the accepted maximum incident-wall y+ diagnostic.
 * @tparam Pack Tpetra type pack used by the model.
 * @return y+ field when wall treatment is active, or null.
 */
template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::wall_y_plus() const noexcept -> const field_type*
{
    return d_state && d_options.wall_treatment != TurbulenceWallTreatmentType::None
         ? &d_state->wall_y_plus
         : nullptr;
}

template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::wall_y_plus_statistics() const noexcept
    -> const Arr<WallYPlusStatistics>&
{
    return d_state
         ? d_state->wall_statistics
         : d_empty_wall_y_plus_statistics;
}

/**
 * @brief Return active turbulence fields exposed for solution output.
 * @tparam Pack Tpetra type pack used by the model.
 * @return Stable name-to-field mapping, empty when disabled.
 */
template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::output_fields() const noexcept
    -> const std::map<std::string, const field_type*>&
{
    return d_state ? d_state->output_fields : d_empty_output_fields;
}

} // namespace SimpleFluid
