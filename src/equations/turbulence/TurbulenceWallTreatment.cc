/**
 * @file TurbulenceWallTreatment.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Validation and explicit instantiation for turbulence wall treatment.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "equations/turbulence/TurbulenceWallTreatment.hh"
#include "equations/turbulence/TurbulenceWallTreatment.tcc"

#include <cmath>
#include <set>
#include <stdexcept>

namespace SimpleFluid
{

std::string_view to_string(TurbulenceWallRoughnessModel model) noexcept
{
    switch (model)
    {
    case TurbulenceWallRoughnessModel::Smooth:
        return "smooth";
    case TurbulenceWallRoughnessModel::SandGrain:
        return "sandGrain";
    }
    return "unknown";
}

TurbulenceWallRoughnessModel parse_turbulence_wall_roughness_model(
    const std::string& value)
{
    if (value == "smooth" || value == "none")
        return TurbulenceWallRoughnessModel::Smooth;
    if (value == "sandGrain" || value == "sand-grain" || value == "rough")
        return TurbulenceWallRoughnessModel::SandGrain;
    throw std::invalid_argument("Unknown turbulence wall roughness model '" + value + "'.");
}

std::string_view to_string(TurbulenceThermalWallLaw law) noexcept
{
    switch (law)
    {
    case TurbulenceThermalWallLaw::TurbulentPrandtl:
        return "turbulentPrandtl";
    case TurbulenceThermalWallLaw::Jayatilleke:
        return "Jayatilleke";
    }
    return "unknown";
}

TurbulenceThermalWallLaw parse_turbulence_thermal_wall_law(
    const std::string& value)
{
    if (value == "turbulentPrandtl" || value == "constantPrt")
        return TurbulenceThermalWallLaw::TurbulentPrandtl;
    if (value == "Jayatilleke" || value == "jayatilleke")
        return TurbulenceThermalWallLaw::Jayatilleke;
    throw std::invalid_argument("Unknown turbulence thermal wall law '" + value + "'.");
}

/**
 * @brief Validate selected wall patches and policy-independent constants.
 * @param options Wall-treatment options to validate.
 * @throws std::invalid_argument if a name or constant is invalid.
 */
void validate_turbulence_wall_treatment_options(const TurbulenceWallTreatmentOptions& options)
{
    if (options.boundary_names.empty())
    {
        throw std::invalid_argument("Turbulence wall treatment requires at least "
                                    "one explicit boundary name.");
    }
    std::set<std::string> unique_names;
    for (const auto& name : options.boundary_names)
    {
        if (name.empty())
        {
            throw std::invalid_argument("Turbulence wall boundary names cannot be empty.");
        }
        if (!unique_names.insert(name).second)
        {
            throw std::invalid_argument("Duplicate turbulence wall boundary name '" + name + "'.");
        }
    }

    const auto has_roughness_configuration =
        !options.roughness_models.empty() || !options.roughness_heights.empty() ||
        !options.roughness_constants.empty();
    if (has_roughness_configuration &&
        (options.roughness_models.size() != options.boundary_names.size() ||
         options.roughness_heights.size() != options.boundary_names.size() ||
         options.roughness_constants.size() != options.boundary_names.size()))
    {
        throw std::invalid_argument(
            "Turbulence wall roughness modes, heights, and constants must all be "
            "patch aligned.");
    }
    for (size_t index = 0; index < options.roughness_heights.size(); ++index)
    {
        if (options.roughness_models[index] !=
                TurbulenceWallRoughnessModel::Smooth &&
            options.roughness_models[index] !=
                TurbulenceWallRoughnessModel::SandGrain)
        {
            throw std::invalid_argument(
                "Turbulence wall roughness model is invalid.");
        }
        const auto height = options.roughness_heights[index];
        const auto constant = options.roughness_constants[index];
        if (!std::isfinite(height) || height < 0.0 || !std::isfinite(constant) ||
            constant < 0.0)
        {
            throw std::invalid_argument(
                "Turbulence wall roughness heights and constants must be finite "
                "and non-negative.");
        }
        if (options.roughness_models[index] == TurbulenceWallRoughnessModel::Smooth &&
            height != 0.0)
        {
            throw std::invalid_argument(
                "A smooth turbulence wall cannot have non-zero roughness height.");
        }
    }
    if (options.thermal_wall_law !=
            TurbulenceThermalWallLaw::TurbulentPrandtl &&
        options.thermal_wall_law != TurbulenceThermalWallLaw::Jayatilleke)
    {
        throw std::invalid_argument("Turbulence thermal wall law is invalid.");
    }

    const real_t constants[] = {options.c_mu, options.kappa, options.log_layer_e,
                                options.sst_beta_1, options.sst_omega_wall_coefficient};
    for (const auto value : constants)
    {
        if (!std::isfinite(value) || value <= 0.0)
        {
            throw std::invalid_argument("Turbulence wall constants must be finite and positive.");
        }
    }
    if (options.thermal_turbulent_prandtl_number.has_value() &&
        (!std::isfinite(*options.thermal_turbulent_prandtl_number) ||
         *options.thermal_turbulent_prandtl_number <= 0.0))
    {
        throw std::invalid_argument(
            "Turbulence thermal wall Prandtl number must be finite and positive.");
    }
}

/**
 * @brief Compute the OpenFOAM.com log-layer intersection by fixed iteration.
 * @param kappa Von Karman constant.
 * @param log_layer_e Log-law roughness constant.
 * @return Positive sublayer intersection y+.
 * @throws std::invalid_argument if an input is not finite and positive.
 * @throws std::overflow_error if the iteration produces an invalid value.
 */
real_t openfoam_y_plus_lam(real_t kappa, real_t log_layer_e)
{
    if (!std::isfinite(kappa) || kappa <= 0.0 || !std::isfinite(log_layer_e) || log_layer_e <= 0.0)
    {
        throw std::invalid_argument("OpenFOAM yPlusLam requires positive finite kappa and E.");
    }

    real_t y_plus_lam = 11.0;
    for (int iteration = 0; iteration < 10; ++iteration)
    {
        y_plus_lam = std::log(std::max(log_layer_e * y_plus_lam, 1.0)) / kappa;
    }
    if (!std::isfinite(y_plus_lam) || y_plus_lam <= 0.0)
    {
        throw std::overflow_error("OpenFOAM yPlusLam iteration produced an invalid value.");
    }
    return y_plus_lam;
}

template class TurbulenceWallTreatment<DefaultTpetraTypes, ResolvedLowReSSTWallPolicy>;
template class TurbulenceWallTreatment<DefaultTpetraTypes, ResolvedLowReKEpsilonWallPolicy>;
template class TurbulenceWallTreatment<DefaultTpetraTypes, StandardHighReKEpsilonWallPolicy>;
template class TurbulenceWallTreatment<
    DefaultTpetraTypes, ResolvedLowReSSTWallPolicy,
    MeshHandle<DefaultTpetraTypes>>;
template class TurbulenceWallTreatment<
    DefaultTpetraTypes, ResolvedLowReKEpsilonWallPolicy,
    MeshHandle<DefaultTpetraTypes>>;
template class TurbulenceWallTreatment<
    DefaultTpetraTypes, StandardHighReKEpsilonWallPolicy,
    MeshHandle<DefaultTpetraTypes>>;

} // namespace SimpleFluid
