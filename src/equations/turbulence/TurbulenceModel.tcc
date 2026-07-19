/**
 * @file TurbulenceModel.tcc
 * @brief Template implementation of runtime two-equation turbulence coupling.
 */

#include "TurbulenceCollectiveValidation.hh"
#include "TurbulenceModel.hh"

#include "FVM/CellOperators.hh"
#include "fields/TensorCellField.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <variant>

namespace SimpleFluid
{

template <TpetraTypePack Pack> struct TurbulenceModel<Pack>::State
{
    using closure_type =
        std::variant<StandardKEpsilonEquation, RNGKEpsilonEquation, RealizableKEpsilonEquation,
                     StandardKOmegaEquation, BSLKOmegaEquation, SSTKOmegaEquation>;

    State(SP<const mesh_type> mesh, const TurbulenceBoundaryConditionSet& boundary_conditions,
          const TurbulenceModelOptions& options)
        : epsilon_family(options.model == TurbulenceModelType::StandardKEpsilon ||
                         options.model == TurbulenceModelType::RNGKEpsilon ||
                         options.model == TurbulenceModelType::RealizableKEpsilon),
          menter_family(options.model == TurbulenceModelType::BSLKOmega ||
                        options.model == TurbulenceModelType::SSTKOmega),
          closure(make_closure(options.model)),
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
          wall_distance(mesh, static_cast<scalar_type>(options.initial_wall_distance.value_or(1.0)),
                        "wall_distance"),
          velocity_gradient(mesh, "velocity_gradient"), k_gradient(mesh, "k_gradient"),
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
    }

    static closure_type make_closure(TurbulenceModelType model)
    {
        switch (model)
        {
        case TurbulenceModelType::StandardKEpsilon:
            return StandardKEpsilonEquation{};
        case TurbulenceModelType::RNGKEpsilon:
            return RNGKEpsilonEquation{};
        case TurbulenceModelType::RealizableKEpsilon:
            return RealizableKEpsilonEquation{};
        case TurbulenceModelType::StandardKOmega:
            return StandardKOmegaEquation{};
        case TurbulenceModelType::BSLKOmega:
            return BSLKOmegaEquation{};
        case TurbulenceModelType::SSTKOmega:
            return SSTKOmegaEquation{};
        case TurbulenceModelType::Laminar:
            break;
        }
        throw std::logic_error("Laminar mode does not have a turbulence closure.");
    }

    bool epsilon_family;
    bool menter_family;
    closure_type closure;
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
    field_type wall_distance;
    TensorCellField<Pack> velocity_gradient;
    VectorCellField<Pack> k_gradient;
    VectorCellField<Pack> secondary_gradient;
    VectorCellField<Pack> candidate_k_gradient;
    VectorCellField<Pack> candidate_secondary_gradient;
    TurbulenceScalarTransportEquation<Pack> k_equation;
    TurbulenceScalarTransportEquation<Pack> secondary_equation;
    std::map<std::string, const field_type*> output_fields;
};

template <TpetraTypePack Pack>
TurbulenceModel<Pack>::TurbulenceModel(SP<const mesh_type> mesh,
                                       const BoundaryConditionSet& boundary_conditions)
    : d_mesh(std::move(mesh)), d_velocity_boundary_conditions(boundary_conditions.velocity),
      d_boundary_conditions(boundary_conditions.turbulence)
{
    if (!d_mesh)
    {
        throw std::invalid_argument("TurbulenceModel requires a non-null mesh.");
    }

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

template <TpetraTypePack Pack>
void TurbulenceModel<Pack>::configure(const TurbulenceModelOptions& options,
                                      const material_type& material, scalar_type reference_density)
{
    turbulence_detail::collective_local_validation(*d_mesh, "Turbulence model option validation",
                                                   [&]
                                                   { validate_turbulence_model_options(options); });
    turbulence_detail::require_uniform_integral(*d_mesh, static_cast<int>(options.model),
                                                "Turbulence model type");
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
            auto validate_scalar_boundaries =
                [](const BoundaryConditionMap& conditions, real_t floor,
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
                    if (condition.type == BoundaryConditionType::Dirichlet &&
                        condition.value < floor)
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

    auto candidate = std::make_unique<State>(d_mesh, d_boundary_conditions, options);
    turbulence_detail::collective_local_validation(
        *d_mesh, "Initial turbulence-gradient reconstruction",
        [&]
        {
            FVM::cell_gradient(
                candidate->k,
                d_boundary_conditions.turbulent_kinetic_energy,
                candidate->k_gradient);
            if (candidate->menter_family)
            {
                FVM::cell_gradient(
                    candidate->secondary,
                    d_boundary_conditions.specific_dissipation_rate,
                    candidate->secondary_gradient);
            }
        });
    candidate->k_gradient.sync_ghosts();
    if (candidate->menter_family)
    {
        candidate->secondary_gradient.sync_ghosts();
    }
    stage_effective_properties(*candidate, candidate->nu_t, material, reference_density,
                               static_cast<scalar_type>(options.turbulent_prandtl_number));
    commit_effective_properties(*candidate);
    d_options = options;
    d_state = std::move(candidate);
}

template <TpetraTypePack Pack> bool TurbulenceModel<Pack>::disable() noexcept
{
    const auto was_enabled = static_cast<bool>(d_state);
    d_state.reset();
    d_options.model = TurbulenceModelType::Laminar;
    return was_enabled;
}

template <TpetraTypePack Pack> bool TurbulenceModel<Pack>::enabled() const noexcept
{
    return static_cast<bool>(d_state);
}

template <TpetraTypePack Pack> TurbulenceModelType TurbulenceModel<Pack>::type() const noexcept
{
    return d_options.model;
}

template <TpetraTypePack Pack>
const TurbulenceModelOptions& TurbulenceModel<Pack>::options() const noexcept
{
    return d_options;
}

template <TpetraTypePack Pack> auto TurbulenceModel<Pack>::require_state() -> State&
{
    if (!d_state)
    {
        throw std::logic_error("TurbulenceModel is disabled in laminar mode.");
    }
    return *d_state;
}

template <TpetraTypePack Pack> auto TurbulenceModel<Pack>::require_state() const -> const State&
{
    if (!d_state)
    {
        throw std::logic_error("TurbulenceModel is disabled in laminar mode.");
    }
    return *d_state;
}

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

            for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell_lid = static_cast<local_ordinal_type>(owned);
                const auto density = material.density.value(cell_lid);
                const auto heat_capacity = material.specific_heat_capacity.value(cell_lid);
                const auto molecular_viscosity = material.dynamic_viscosity.value(cell_lid);
                const auto molecular_conductivity = material.thermal_conductivity.value(cell_lid);
                const auto nu_t = turbulent_kinematic_viscosity.value(cell_lid);
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
                state.candidate_effective_dynamic_viscosity.set_owned_value(cell_lid,
                                                                            effective_viscosity);
                state.candidate_effective_thermal_conductivity.set_owned_value(
                    cell_lid, effective_conductivity);
            }
        });
    state.candidate_effective_dynamic_viscosity.sync_ghosts();
    state.candidate_effective_thermal_conductivity.sync_ghosts();
}

template <TpetraTypePack Pack>
void TurbulenceModel<Pack>::commit_effective_properties(State& state) const
{
    state.effective_dynamic_viscosity.owned_data().update(
        scalar_type{1}, state.candidate_effective_dynamic_viscosity.owned_data(), scalar_type{0});
    state.effective_thermal_conductivity.owned_data().update(
        scalar_type{1}, state.candidate_effective_thermal_conductivity.owned_data(),
        scalar_type{0});
    state.effective_dynamic_viscosity.sync_ghosts();
    state.effective_thermal_conductivity.sync_ghosts();
}

template <TpetraTypePack Pack>
void TurbulenceModel<Pack>::refresh_effective_properties(const material_type& material,
                                                         scalar_type reference_density)
{
    turbulence_detail::require_uniform_integral(*d_mesh, enabled() ? 1 : 0,
                                                "Turbulence enabled state");
    auto& state = require_state();
    turbulence_detail::require_uniform_real(*d_mesh, static_cast<real_t>(reference_density),
                                            "Turbulence reference density");
    stage_effective_properties(state, state.nu_t, material, reference_density,
                               static_cast<scalar_type>(d_options.turbulent_prandtl_number));
    commit_effective_properties(state);
}

template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::advance(const velocity_field_type& velocity,
                                    const face_flux_field_type& projected_face_fluxes,
                                    const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
                                    scalar_type time_step, const material_type& material,
                                    scalar_type reference_density,
                                    FVM::NonOrthogonalTreatment treatment,
                                    const LinearSolverOptions& linear_options) -> LinearSolveSummary
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
        });

    auto boundary_velocity = [&](int batch_id, size_t in_batch_id) ->
        typename velocity_field_type::vec_type
    {
        const auto& batch = d_mesh->boundary_batches().at(batch_id);
        const auto face_lid = batch.face_lids.at(in_batch_id);
        const auto type = velocity_boundary_cache.type.at(batch_id);
        if (type == BoundaryConditionType::Slip)
        {
            return FVM::detail::slip_face_velocity(velocity, face_lid);
        }
        if (type == BoundaryConditionType::Periodic)
        {
            return velocity.local_value(d_mesh->owner_cell(face_lid));
        }
        if (type == BoundaryConditionType::Neumann)
        {
            const auto owner = d_mesh->owner_cell(face_lid);
            const auto owner_value = velocity.local_value(owner);
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
    auto update_scalar_gradients =
        [&, this](const field_type& k_field, const field_type& secondary_field,
                  velocity_field_type& k_gradient,
                  velocity_field_type& secondary_gradient)
    {
        FVM::cell_gradient(k_field, d_boundary_conditions.turbulent_kinetic_energy,
                           k_gradient);
        if (state.menter_family)
        {
            FVM::cell_gradient(secondary_field, d_boundary_conditions.specific_dissipation_rate,
                               secondary_gradient);
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
            FVM::cell_gradient(velocity, boundary_velocity, state.velocity_gradient);
            update_scalar_gradients(state.k, state.secondary, state.k_gradient,
                                    state.secondary_gradient);
        });
    sync_scalar_gradients(state.k_gradient, state.secondary_gradient);

    auto evaluate_closure =
        [&, this](const field_type& k_field, const field_type& secondary_field,
                  const velocity_field_type& k_gradient,
                  const velocity_field_type& secondary_gradient)
    {
        turbulence_detail::collective_local_validation(
            *d_mesh, "Turbulence closure evaluation",
            [&]
            {
                for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
                {
                    const auto cell_lid = static_cast<local_ordinal_type>(owned);
                    const auto k = static_cast<real_t>(k_field.value(cell_lid));
                    const auto secondary = static_cast<real_t>(secondary_field.value(cell_lid));
                    const auto molecular_viscosity =
                        static_cast<real_t>(material.dynamic_viscosity.value(cell_lid));
                    if (!std::isfinite(molecular_viscosity) || molecular_viscosity < 0.0)
                    {
                        throw std::invalid_argument(
                            "TurbulenceModel requires finite non-negative molecular "
                            "viscosity.");
                    }
                    const auto molecular_nu =
                        molecular_viscosity / static_cast<real_t>(reference_density);

                    const auto gradient = state.velocity_gradient.value(cell_lid);
                    std::array<std::array<real_t, 3>, 3> strain{};
                    real_t strain_squared{};
                    real_t rotation_squared{};
                    for (size_t row = 0; row < 3; ++row)
                    {
                        for (size_t column = 0; column < 3; ++column)
                        {
                            const auto gij = static_cast<real_t>(gradient[row].component(column));
                            const auto gji = static_cast<real_t>(gradient[column].component(row));
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
                    switch (d_options.model)
                    {
                    case TurbulenceModelType::StandardKEpsilon:
                    {
                        const auto& closure = std::get<StandardKEpsilonEquation>(state.closure);
                        const KEpsilonState local{k, secondary};
                        nu_t = closure.turbulent_kinematic_viscosity(local);
                        const auto diffusion = closure.diffusivities(local, molecular_nu);
                        const auto production = nu_t * strain_magnitude * strain_magnitude;
                        const auto& coefficients = closure.coefficients();
                        k_diffusivity = diffusion.k;
                        secondary_diffusivity = diffusion.epsilon;
                        explicit_k_source = production;
                        implicit_k_sink = secondary / k;
                        explicit_secondary_source =
                            coefficients.c_epsilon_1 * production * secondary / k;
                        implicit_secondary_sink = coefficients.c_epsilon_2 * secondary / k;
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
                        const auto production = nu_t * strain_magnitude * strain_magnitude;
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
                        break;
                    }
                    case TurbulenceModelType::BSLKOmega:
                    case TurbulenceModelType::SSTKOmega:
                    {
                        const KOmegaState local{k, secondary};
                        const auto local_k_gradient = k_gradient.value(cell_lid);
                        const auto omega_gradient = secondary_gradient.value(cell_lid);
                        const MenterKOmegaInvariants invariants{
                            molecular_nu, static_cast<real_t>(state.wall_distance.value(cell_lid)),
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
                        }
                        break;
                    }
                    case TurbulenceModelType::Laminar:
                        throw std::logic_error("TurbulenceModel lost its active closure.");
                    }

                    const real_t values[] = {nu_t,
                                             k_diffusivity,
                                             secondary_diffusivity,
                                             explicit_k_source,
                                             implicit_k_sink,
                                             explicit_secondary_source,
                                             implicit_secondary_sink};
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

                    state.candidate_nu_t.set_owned_value(cell_lid, static_cast<scalar_type>(nu_t));
                    state.k_diffusivity.set_owned_value(cell_lid,
                                                        static_cast<scalar_type>(k_diffusivity));
                    state.secondary_diffusivity.set_owned_value(
                        cell_lid, static_cast<scalar_type>(secondary_diffusivity));
                    state.k_source.set_owned_value(cell_lid,
                                                   static_cast<scalar_type>(explicit_k_source));
                    state.k_sink.set_owned_value(cell_lid,
                                                 static_cast<scalar_type>(implicit_k_sink));
                    state.secondary_source.set_owned_value(
                        cell_lid, static_cast<scalar_type>(explicit_secondary_source));
                    state.secondary_sink.set_owned_value(
                        cell_lid, static_cast<scalar_type>(implicit_secondary_sink));
                }
            });
        state.candidate_nu_t.sync_ghosts();
        state.k_diffusivity.sync_ghosts();
        state.secondary_diffusivity.sync_ghosts();
    };
    evaluate_closure(state.k, state.secondary, state.k_gradient,
                     state.secondary_gradient);

    auto k_source = [&](local_ordinal_type cell_lid) { return state.k_source.value(cell_lid); };
    auto k_sink = [&](local_ordinal_type cell_lid) { return state.k_sink.value(cell_lid); };
    auto secondary_source = [&](local_ordinal_type cell_lid)
    { return state.secondary_source.value(cell_lid); };
    auto secondary_sink = [&](local_ordinal_type cell_lid)
    { return state.secondary_sink.value(cell_lid); };

    LinearSolveSummary summary;
    summary.add(state.k_equation.advance(
        state.k, projected_face_fluxes, time_step, state.k_diffusivity, state.candidate_k, k_source,
        k_sink, static_cast<scalar_type>(d_options.min_turbulent_kinetic_energy), treatment,
        linear_options));
    summary.add(state.secondary_equation.advance(
        state.secondary, projected_face_fluxes, time_step, state.secondary_diffusivity,
        state.candidate_secondary, secondary_source, secondary_sink,
        static_cast<scalar_type>(state.epsilon_family ? d_options.min_dissipation_rate
                                                      : d_options.min_specific_dissipation_rate),
        treatment, linear_options));

    turbulence_detail::collective_local_validation(
        *d_mesh, "Turbulence candidate-gradient reconstruction",
        [&]
        {
            update_scalar_gradients(state.candidate_k, state.candidate_secondary,
                                    state.candidate_k_gradient,
                                    state.candidate_secondary_gradient);
        });
    sync_scalar_gradients(state.candidate_k_gradient,
                          state.candidate_secondary_gradient);
    evaluate_closure(state.candidate_k, state.candidate_secondary,
                     state.candidate_k_gradient,
                     state.candidate_secondary_gradient);
    stage_effective_properties(state, state.candidate_nu_t, material, reference_density,
                               static_cast<scalar_type>(d_options.turbulent_prandtl_number));

    // Publish only after both solves and all derived-field validation pass.
    state.k.owned_data().update(scalar_type{1}, state.candidate_k.owned_data(), scalar_type{0});
    state.secondary.owned_data().update(scalar_type{1}, state.candidate_secondary.owned_data(),
                                        scalar_type{0});
    state.nu_t.owned_data().update(scalar_type{1}, state.candidate_nu_t.owned_data(),
                                   scalar_type{0});
    state.k_gradient.owned_data().update(
        scalar_type{1}, state.candidate_k_gradient.owned_data(), scalar_type{0});
    if (state.menter_family)
    {
        state.secondary_gradient.owned_data().update(
            scalar_type{1}, state.candidate_secondary_gradient.owned_data(), scalar_type{0});
    }
    state.k.sync_ghosts();
    state.secondary.sync_ghosts();
    state.nu_t.sync_ghosts();
    state.k_gradient.sync_ghosts();
    if (state.menter_family)
    {
        state.secondary_gradient.sync_ghosts();
    }
    commit_effective_properties(state);
    return summary;
}

template <TpetraTypePack Pack>
void TurbulenceModel<Pack>::set_wall_distance(const field_type& wall_distance)
{
    turbulence_detail::require_uniform_integral(*d_mesh, enabled() ? 1 : 0,
                                                "Turbulence enabled state");
    auto& state = require_state();
    turbulence_detail::collective_local_validation(
        *d_mesh, "Turbulence wall-distance validation",
        [&]
        {
            if (&wall_distance.mesh() != d_mesh.get())
            {
                throw std::invalid_argument("TurbulenceModel wall-distance mesh mismatch.");
            }
            for (size_t local = 0; local < d_mesh->num_local_cells(); ++local)
            {
                const auto cell_lid = static_cast<local_ordinal_type>(local);
                const auto value = wall_distance.local_value(cell_lid);
                if (!std::isfinite(value) || value <= scalar_type{})
                {
                    throw std::invalid_argument("TurbulenceModel wall distance must be finite and "
                                                "positive.");
                }
            }
        });
    state.wall_distance.owned_data().update(scalar_type{1}, wall_distance.owned_data(),
                                            scalar_type{0});
    state.wall_distance.sync_ghosts();
}

template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::turbulent_kinetic_energy() const -> const field_type&
{
    return require_state().k;
}

template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::turbulent_kinetic_energy_gradient() const
    -> const velocity_field_type&
{
    return require_state().k_gradient;
}

template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::dissipation_rate() const noexcept -> const field_type*
{
    return d_state && d_state->epsilon_family ? &d_state->secondary : nullptr;
}

template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::specific_dissipation_rate() const noexcept -> const field_type*
{
    return d_state && !d_state->epsilon_family ? &d_state->secondary : nullptr;
}

template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::turbulent_kinematic_viscosity() const -> const field_type&
{
    return require_state().nu_t;
}

template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::effective_dynamic_viscosity() const -> const field_type&
{
    return require_state().effective_dynamic_viscosity;
}

template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::effective_thermal_conductivity() const -> const field_type&
{
    return require_state().effective_thermal_conductivity;
}

template <TpetraTypePack Pack>
auto TurbulenceModel<Pack>::output_fields() const noexcept
    -> const std::map<std::string, const field_type*>&
{
    return d_state ? d_state->output_fields : d_empty_output_fields;
}

} // namespace SimpleFluid
