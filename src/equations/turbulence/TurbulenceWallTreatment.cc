/**
 * @file TurbulenceWallTreatment.cc
 * @brief Validation and explicit instantiation for turbulence wall treatment.
 */

#include "equations/turbulence/TurbulenceWallTreatment.hh"
#include "equations/turbulence/TurbulenceWallTreatment.tcc"

#include <cmath>
#include <set>
#include <stdexcept>

namespace SimpleFluid
{

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

    const real_t constants[] = {options.c_mu, options.kappa, options.log_layer_e,
                                options.sst_beta_1, options.sst_omega_wall_coefficient};
    for (const auto value : constants)
    {
        if (!std::isfinite(value) || value <= 0.0)
        {
            throw std::invalid_argument("Turbulence wall constants must be finite and positive.");
        }
    }
}

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
template class TurbulenceWallTreatment<DefaultTpetraTypes, StandardHighReKEpsilonWallPolicy>;

} // namespace SimpleFluid
