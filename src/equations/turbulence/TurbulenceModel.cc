/**
 * @file TurbulenceModel.cc
 * @brief Explicit template instantiation for TurbulenceModel.
 */

#include "equations/turbulence/TurbulenceModel.hh"
#include "equations/turbulence/TurbulenceModel.tcc"

#include <cmath>
#include <stdexcept>

namespace SimpleFluid
{

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
}

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
    validate_turbulence_model_options(options);
    return options;
}

template class TurbulenceModel<DefaultTpetraTypes>;
} // namespace SimpleFluid
