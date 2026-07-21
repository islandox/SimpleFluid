/**
 * @file StandardKOmegaEquation.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Wilcox-1988 standard k-omega closure equation terms.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "equations/turbulence/TurbulenceEquationCommon.hh"

namespace SimpleFluid
{

/** @brief Standard Wilcox-1988 k-omega equation closure. */
class StandardKOmegaEquation
{
public:
    /** @brief Coefficients controlling the standard k-omega closure. */
    struct Coefficients
    {
        real_t beta_star = 0.09;
        real_t beta = 0.075;
        real_t gamma = 5.0 / 9.0;
        real_t sigma_k = 0.5;
        real_t sigma_omega = 0.5;
    };

    StandardKOmegaEquation() : StandardKOmegaEquation(Coefficients{}) {}

    explicit StandardKOmegaEquation(Coefficients coefficients) : d_coefficients(coefficients)
    {
        validate_coefficients();
    }

    const Coefficients& coefficients() const noexcept
    {
        return d_coefficients;
    }

    real_t turbulent_kinematic_viscosity(const KOmegaState& state) const
    {
        turbulence_detail::validate(state);
        return state.k / state.omega;
    }

    KOmegaDiffusivities effective_diffusivities(const KOmegaState& state,
                                                real_t molecular_kinematic_viscosity) const
    {
        turbulence_detail::require_non_negative(molecular_kinematic_viscosity,
                                                "Molecular kinematic viscosity");
        const auto nu_t = turbulent_kinematic_viscosity(state);
        return {molecular_kinematic_viscosity + d_coefficients.sigma_k * nu_t,
                molecular_kinematic_viscosity + d_coefficients.sigma_omega * nu_t};
    }

    KOmegaDiffusivities diffusivities(const KOmegaState& state,
                                      real_t molecular_kinematic_viscosity) const
    {
        return effective_diffusivities(state, molecular_kinematic_viscosity);
    }

    KOmegaSourceTerms source_terms(const KOmegaState& state, real_t production) const
    {
        turbulence_detail::validate(state);
        turbulence_detail::validate_production(production);
        return {production - d_coefficients.beta_star * state.k * state.omega,
                d_coefficients.gamma * production * state.omega / state.k -
                    d_coefficients.beta * state.omega * state.omega};
    }

private:
    void validate_coefficients() const
    {
        turbulence_detail::require_positive(d_coefficients.beta_star, "Standard k-omega beta-star");
        turbulence_detail::require_positive(d_coefficients.beta, "Standard k-omega beta");
        turbulence_detail::require_positive(d_coefficients.gamma, "Standard k-omega gamma");
        turbulence_detail::require_positive(d_coefficients.sigma_k, "Standard k-omega sigma-k");
        turbulence_detail::require_positive(d_coefficients.sigma_omega,
                                            "Standard k-omega sigma-omega");
    }

    Coefficients d_coefficients;
};

} // namespace SimpleFluid
