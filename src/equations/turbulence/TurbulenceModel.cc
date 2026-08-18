/**
 * @file TurbulenceModel.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Explicit template instantiation for TurbulenceModel.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "equations/turbulence/TurbulenceModel.hh"
#include "equations/turbulence/TurbulenceModel.tcc"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace SimpleFluid
{

/**
 * @brief Return the canonical database name of a turbulence model.
 * @param model Runtime model identifier.
 * @return Canonical model name, or `unknown` for an invalid enumerator.
 */
std::string_view to_string(TurbulenceModelType model) noexcept
{
    switch (model)
    {
    case TurbulenceModelType::Laminar:
        return "laminar";
    case TurbulenceModelType::StandardKEpsilon:
        return "standardKEpsilon";
    case TurbulenceModelType::RNGKEpsilon:
        return "RNGKEpsilon";
    case TurbulenceModelType::RealizableKEpsilon:
        return "realizableKEpsilon";
    case TurbulenceModelType::StandardKOmega:
        return "standardKOmega";
    case TurbulenceModelType::BSLKOmega:
        return "BSLKOmega";
    case TurbulenceModelType::SSTKOmega:
        return "SSTKOmega";
    }
    return "unknown";
}

/**
 * @brief Parse a turbulence model name or supported alias.
 * @param value Configured model name.
 * @return Parsed turbulence model identifier.
 * @throws std::invalid_argument if @p value is unknown.
 */
TurbulenceModelType parse_turbulence_model_type(const std::string& value)
{
    if (value == "laminar" || value == "none")
        return TurbulenceModelType::Laminar;
    if (value == "standardKEpsilon" || value == "kEpsilon")
        return TurbulenceModelType::StandardKEpsilon;
    if (value == "RNGKEpsilon" || value == "rngKEpsilon")
        return TurbulenceModelType::RNGKEpsilon;
    if (value == "realizableKEpsilon")
        return TurbulenceModelType::RealizableKEpsilon;
    if (value == "standardKOmega" || value == "kOmega")
        return TurbulenceModelType::StandardKOmega;
    if (value == "BSLKOmega" || value == "bslKOmega")
        return TurbulenceModelType::BSLKOmega;
    if (value == "SSTKOmega" || value == "sstKOmega")
        return TurbulenceModelType::SSTKOmega;
    throw std::invalid_argument("Unknown turbulence model '" + value + "'.");
}

/**
 * @brief Return the canonical database name of a wall treatment.
 * @param treatment Runtime wall-treatment identifier.
 * @return Canonical treatment name, or `unknown` for an invalid enumerator.
 */
std::string_view to_string(TurbulenceWallTreatmentType treatment) noexcept
{
    switch (treatment)
    {
    case TurbulenceWallTreatmentType::None:
        return "none";
    case TurbulenceWallTreatmentType::ResolvedLowReSST:
        return "resolvedLowReSST";
    case TurbulenceWallTreatmentType::StandardHighReKEpsilon:
        return "standardHighReKEpsilon";
    case TurbulenceWallTreatmentType::ResolvedLowReKEpsilon:
        return "resolvedLowReKEpsilon";
    }
    return "unknown";
}

/**
 * @brief Parse a turbulence wall-treatment name or supported alias.
 * @param value Configured wall-treatment name.
 * @return Parsed wall-treatment identifier.
 * @throws std::invalid_argument if @p value is unknown.
 */
TurbulenceWallTreatmentType parse_turbulence_wall_treatment_type(
    const std::string& value)
{
    if (value == "none" || value == "off")
        return TurbulenceWallTreatmentType::None;
    if (value == "resolvedLowReSST" || value == "resolvedSST")
        return TurbulenceWallTreatmentType::ResolvedLowReSST;
    if (value == "resolvedLowReKEpsilon" || value == "resolvedKEpsilon")
        return TurbulenceWallTreatmentType::ResolvedLowReKEpsilon;
    if (value == "standardHighReKEpsilon" || value == "OpenFOAMKEpsilon")
        return TurbulenceWallTreatmentType::StandardHighReKEpsilon;
    throw std::invalid_argument(
        "Unknown turbulence wall treatment '" + value + "'.");
}

std::string_view to_string(TurbulenceBuoyancyModel model) noexcept
{
    switch (model)
    {
    case TurbulenceBuoyancyModel::None:
        return "none";
    case TurbulenceBuoyancyModel::OpenFOAMBoussinesq:
        return "OpenFOAMBoussinesq";
    }
    return "unknown";
}

TurbulenceBuoyancyModel parse_turbulence_buoyancy_model(
    const std::string& value)
{
    if (value == "none" || value == "off")
        return TurbulenceBuoyancyModel::None;
    if (value == "OpenFOAMBoussinesq"
        || value == "openFOAMBoussinesq"
        || value == "boussinesq")
    {
        return TurbulenceBuoyancyModel::OpenFOAMBoussinesq;
    }
    throw std::invalid_argument(
        "Unknown turbulence buoyancy model '" + value + "'.");
}

/**
 * @brief Validate model floors, wall distance, and closure-policy pairing.
 * @param options Turbulence model options to validate.
 * @throws std::invalid_argument if any active option is inconsistent.
 */
void validate_turbulence_model_options(const TurbulenceModelOptions& options)
{
    switch (options.gradient_scheme)
    {
        case FVM::CellGradientScheme::LeastSquares:
        case FVM::CellGradientScheme::GaussLinear:
            break;
        default:
            throw std::invalid_argument(
                "Unknown turbulence cell-gradient scheme.");
    }
    switch (options.coefficient_interpolation)
    {
        case FVM::FaceCoefficientInterpolation::Harmonic:
        case FVM::FaceCoefficientInterpolation::Linear:
            break;
        default:
            throw std::invalid_argument(
                "Unknown turbulence face-coefficient interpolation.");
    }
    const real_t positive_values[] = {options.initial_turbulent_kinetic_energy,
                                      options.initial_dissipation_rate,
                                      options.initial_specific_dissipation_rate,
                                      options.min_turbulent_kinetic_energy,
                                      options.min_dissipation_rate,
                                      options.min_specific_dissipation_rate,
                                      options.turbulent_prandtl_number};
    for (const auto value : positive_values)
    {
        if (!std::isfinite(value) || value <= 0.0)
        {
            throw std::invalid_argument("Turbulence initial values, floors, and turbulent Prandtl "
                                        "number must be finite and positive.");
        }
    }
    if (options.initial_turbulent_kinetic_energy < options.min_turbulent_kinetic_energy ||
        options.initial_dissipation_rate < options.min_dissipation_rate ||
        options.initial_specific_dissipation_rate < options.min_specific_dissipation_rate)
    {
        throw std::invalid_argument("Turbulence initial values must be greater than or equal to "
                                    "their configured floors.");
    }

    if (options.initial_wall_distance.has_value() &&
        (!std::isfinite(*options.initial_wall_distance) || *options.initial_wall_distance <= 0.0))
    {
        throw std::invalid_argument("Turbulence wall distance must be finite and positive.");
    }
    if (!std::isfinite(options.buoyancy_coefficient)
        || options.buoyancy_coefficient <= 0.0)
    {
        throw std::invalid_argument(
            "Turbulence buoyancy coefficient must be finite and positive.");
    }
    if (options.model == TurbulenceModelType::Laminar
        && options.buoyancy_model != TurbulenceBuoyancyModel::None)
    {
        throw std::invalid_argument(
            "A turbulence buoyancy model requires an active turbulence "
            "closure.");
    }
    const auto menter_model =
        options.model == TurbulenceModelType::BSLKOmega
        || options.model == TurbulenceModelType::SSTKOmega;
    if (!menter_model && !options.wall_distance_boundaries.empty())
    {
        throw std::invalid_argument(
            "Wall-distance boundaries are only used by BSL and SST "
            "k-omega models.");
    }
    auto sorted_wall_distance_boundaries = options.wall_distance_boundaries;
    for (const auto& name : sorted_wall_distance_boundaries)
    {
        if (name.empty())
        {
            throw std::invalid_argument(
                "Wall-distance boundary names cannot be empty.");
        }
    }
    std::sort(
        sorted_wall_distance_boundaries.begin(),
        sorted_wall_distance_boundaries.end());
    if (std::adjacent_find(
            sorted_wall_distance_boundaries.begin(),
            sorted_wall_distance_boundaries.end())
        != sorted_wall_distance_boundaries.end())
    {
        throw std::invalid_argument(
            "Wall-distance boundary names must be unique.");
    }
    validate_wall_distance_equation_options(
        options.wall_distance_equation);

    switch (options.wall_treatment)
    {
    case TurbulenceWallTreatmentType::None:
        if (!options.wall_options.boundary_names.empty())
        {
            throw std::invalid_argument(
                "Turbulence wall boundaries require an active wall treatment.");
        }
        break;
    case TurbulenceWallTreatmentType::ResolvedLowReSST:
        if (options.model != TurbulenceModelType::SSTKOmega)
        {
            throw std::invalid_argument(
                "resolvedLowReSST wall treatment requires the SST k-omega model.");
        }
        validate_turbulence_wall_treatment_options(options.wall_options);
        if (options.wall_options.thermal_wall_law !=
            TurbulenceThermalWallLaw::TurbulentPrandtl ||
            std::any_of(options.wall_options.roughness_models.begin(),
                        options.wall_options.roughness_models.end(),
                        [](const auto model)
                        {
                            return model != TurbulenceWallRoughnessModel::Smooth;
                        }) ||
            std::any_of(options.wall_options.roughness_heights.begin(),
                        options.wall_options.roughness_heights.end(),
                        [](const auto height) { return height != 0.0; }))
        {
            throw std::invalid_argument(
                "resolvedLowReSST requires smooth walls and molecular wall heat transport.");
        }
        {
            auto coefficients = SSTKOmegaEquation::Coefficients{};
            coefficients.beta_1 = options.wall_options.sst_beta_1;
            coefficients.kappa = options.wall_options.kappa;
            static_cast<void>(SSTKOmegaEquation{coefficients});
        }
        break;
    case TurbulenceWallTreatmentType::StandardHighReKEpsilon:
        if (options.model != TurbulenceModelType::StandardKEpsilon)
        {
            throw std::invalid_argument(
                "standardHighReKEpsilon wall treatment requires the standard k-epsilon model.");
        }
        validate_turbulence_wall_treatment_options(options.wall_options);
        break;
    case TurbulenceWallTreatmentType::ResolvedLowReKEpsilon:
        if (options.model != TurbulenceModelType::StandardKEpsilon &&
            options.model != TurbulenceModelType::RealizableKEpsilon)
        {
            throw std::invalid_argument(
                "resolvedLowReKEpsilon wall treatment requires the standard or "
                "realizable k-epsilon model.");
        }
        validate_turbulence_wall_treatment_options(options.wall_options);
        if (options.wall_options.thermal_wall_law !=
                TurbulenceThermalWallLaw::TurbulentPrandtl ||
            std::any_of(options.wall_options.roughness_models.begin(),
                        options.wall_options.roughness_models.end(),
                        [](const auto model)
                        {
                            return model != TurbulenceWallRoughnessModel::Smooth;
                        }) ||
            std::any_of(options.wall_options.roughness_heights.begin(),
                        options.wall_options.roughness_heights.end(),
                        [](const auto height) { return height != 0.0; }))
        {
            throw std::invalid_argument(
                "resolvedLowReKEpsilon requires smooth walls and molecular wall "
                "heat transport.");
        }
        break;
    }
}

/**
 * @brief Parse and validate turbulence model options from a database.
 * @param database Source configuration database.
 * @return Validated turbulence model options.
 * @throws std::invalid_argument if an option is ill-typed or invalid.
 */
TurbulenceModelOptions turbulence_model_options_from_database(const Database& database)
{
    TurbulenceModelOptions options;
    options.model = parse_turbulence_model_type(
        detail::database_value_or<std::string>(database, "turbulence_model", "laminar"));
    options.initial_turbulent_kinetic_energy = detail::database_value_or<real_t>(
        database, "initial_turbulent_kinetic_energy", options.initial_turbulent_kinetic_energy);
    options.initial_dissipation_rate = detail::database_value_or<real_t>(
        database, "initial_dissipation_rate", options.initial_dissipation_rate);
    options.initial_specific_dissipation_rate = detail::database_value_or<real_t>(
        database, "initial_specific_dissipation_rate", options.initial_specific_dissipation_rate);
    options.min_turbulent_kinetic_energy = detail::database_value_or<real_t>(
        database, "min_turbulent_kinetic_energy", options.min_turbulent_kinetic_energy);
    options.min_dissipation_rate = detail::database_value_or<real_t>(
        database, "min_dissipation_rate", options.min_dissipation_rate);
    options.min_specific_dissipation_rate = detail::database_value_or<real_t>(
        database, "min_specific_dissipation_rate", options.min_specific_dissipation_rate);
    options.turbulent_prandtl_number = detail::database_value_or<real_t>(
        database, "turbulent_prandtl_number", options.turbulent_prandtl_number);
    if (database.contains("wall_distance"))
    {
        options.initial_wall_distance = database.get<real_t>("wall_distance");
    }
    if (database.contains("wall_distance_boundaries"))
    {
        options.wall_distance_boundaries =
            database.get<ArrString>("wall_distance_boundaries");
    }
    options.wall_distance_equation.non_orthogonal_treatment =
        FVM::non_orthogonal_treatment_from_string(
            detail::database_value_or<std::string>(
                database,
                "wall_distance_non_orthogonal_treatment",
                std::string{FVM::to_string(
                    options.wall_distance_equation
                        .non_orthogonal_treatment)}));
    options.wall_distance_equation.non_orthogonal_correctors =
        detail::database_value_or<int>(
            database,
            "wall_distance_non_orthogonal_correctors",
            options.wall_distance_equation.non_orthogonal_correctors);
    options.wall_distance_equation.linear_solver.max_iterations =
        detail::database_value_or<int>(
            database,
            "wall_distance_linear_solver_max_iterations",
            options.wall_distance_equation.linear_solver.max_iterations);
    options.wall_distance_equation.linear_solver.tolerance =
        detail::database_value_or<real_t>(
            database,
            "wall_distance_linear_solver_tolerance",
            options.wall_distance_equation.linear_solver.tolerance);
    options.wall_distance_equation.linear_solver.verbosity =
        detail::database_value_or<int>(
            database,
            "wall_distance_linear_solver_verbosity",
            options.wall_distance_equation.linear_solver.verbosity);
    options.wall_distance_equation.linear_solver.backend =
        parse_linear_solver_backend(
            detail::database_value_or<std::string>(
                database,
                "wall_distance_linear_solver_backend",
                std::string{to_string(
                    options.wall_distance_equation.linear_solver
                        .backend)}));
    options.wall_distance_equation.linear_solver.preconditioner =
        parse_linear_preconditioner(
            detail::database_value_or<std::string>(
                database,
                "wall_distance_linear_solver_preconditioner",
                std::string{to_string(
                    options.wall_distance_equation.linear_solver
                        .preconditioner)}));
    options.wall_distance_equation.linear_solver.reuse_preconditioner =
        detail::database_value_or<bool>(
            database,
            "wall_distance_linear_solver_reuse_preconditioner",
            options.wall_distance_equation.linear_solver
                .reuse_preconditioner);
    options.buoyancy_model = parse_turbulence_buoyancy_model(
        detail::database_value_or<std::string>(
            database, "turbulence_buoyancy_model", "none"));
    options.buoyancy_coefficient = detail::database_value_or<real_t>(
        database, "turbulence_buoyancy_coefficient",
        options.buoyancy_coefficient);
    options.wall_treatment = parse_turbulence_wall_treatment_type(
        detail::database_value_or<std::string>(database, "wall_treatment", "none"));
    if (database.contains("wall_boundaries"))
    {
        options.wall_options.boundary_names = database.get<ArrString>("wall_boundaries");
    }
    if (database.contains("wall_roughness_model"))
    {
        const auto configured_models =
            database.get<ArrString>("wall_roughness_model");
        options.wall_options.roughness_models.reserve(configured_models.size());
        for (const auto& model : configured_models)
        {
            options.wall_options.roughness_models.push_back(
                parse_turbulence_wall_roughness_model(model));
        }
    }
    if (database.contains("wall_roughness_heights"))
    {
        options.wall_options.roughness_heights =
            database.get<ArrReal>("wall_roughness_heights");
    }
    if (database.contains("wall_roughness_constants"))
    {
        options.wall_options.roughness_constants =
            database.get<ArrReal>("wall_roughness_constants");
    }
    options.wall_options.c_mu = detail::database_value_or<real_t>(
        database, "wall_c_mu", options.wall_options.c_mu);
    options.wall_options.kappa = detail::database_value_or<real_t>(
        database, "wall_kappa", options.wall_options.kappa);
    options.wall_options.log_layer_e = detail::database_value_or<real_t>(
        database, "wall_e", options.wall_options.log_layer_e);
    options.wall_options.thermal_wall_law = parse_turbulence_thermal_wall_law(
        detail::database_value_or<std::string>(
            database, "wall_thermal_law",
            std::string{to_string(options.wall_options.thermal_wall_law)}));
    if (database.contains("wall_thermal_turbulent_prandtl_number"))
    {
        options.wall_options.thermal_turbulent_prandtl_number =
            database.get<real_t>("wall_thermal_turbulent_prandtl_number");
    }
    options.wall_options.epsilon_low_re_correction =
        detail::database_value_or<bool>(
            database, "wall_epsilon_low_re_correction",
            options.wall_options.epsilon_low_re_correction);
    options.wall_options.sst_beta_1 = detail::database_value_or<real_t>(
        database, "wall_beta_1", options.wall_options.sst_beta_1);
    options.wall_options.sst_omega_wall_coefficient = detail::database_value_or<real_t>(
        database, "wall_sst_omega_face_coefficient",
        options.wall_options.sst_omega_wall_coefficient);
    validate_turbulence_model_options(options);
    return options;
}

template class TurbulenceModel<DefaultTpetraTypes>;
template class TurbulenceModel<
    DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid
