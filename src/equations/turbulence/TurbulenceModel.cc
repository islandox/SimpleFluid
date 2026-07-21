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
    if (value == "standardHighReKEpsilon" || value == "OpenFOAMKEpsilon")
        return TurbulenceWallTreatmentType::StandardHighReKEpsilon;
    throw std::invalid_argument(
        "Unknown turbulence wall treatment '" + value + "'.");
}

/**
 * @brief Validate model floors, wall distance, and closure-policy pairing.
 * @param options Turbulence model options to validate.
 * @throws std::invalid_argument if any active option is inconsistent.
 */
void validate_turbulence_model_options(const TurbulenceModelOptions& options)
{
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
    if ((options.model == TurbulenceModelType::BSLKOmega ||
         options.model == TurbulenceModelType::SSTKOmega) &&
        !options.initial_wall_distance.has_value())
    {
        throw std::invalid_argument("BSL and SST k-omega models require a positive wall distance.");
    }

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
    options.wall_treatment = parse_turbulence_wall_treatment_type(
        detail::database_value_or<std::string>(database, "wall_treatment", "none"));
    if (database.contains("wall_boundaries"))
    {
        options.wall_options.boundary_names = database.get<ArrString>("wall_boundaries");
    }
    options.wall_options.c_mu = detail::database_value_or<real_t>(
        database, "wall_c_mu", options.wall_options.c_mu);
    options.wall_options.kappa = detail::database_value_or<real_t>(
        database, "wall_kappa", options.wall_options.kappa);
    options.wall_options.log_layer_e = detail::database_value_or<real_t>(
        database, "wall_e", options.wall_options.log_layer_e);
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
} // namespace SimpleFluid
