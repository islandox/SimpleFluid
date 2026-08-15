/**
 * @file StandardKEpsilonEquation.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Standard high-Reynolds-number k-epsilon closure equation terms.
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

/**
 * @brief Launder-Spalding standard k-epsilon equation closure.
 *
 * All returned quantities use the density-divided incompressible equations.
 * Consequently, viscosity and diffusivity values are kinematic.
 */
class StandardKEpsilonEquation
{
public:
    /** @brief Coefficients controlling the standard k-epsilon closure. */
    struct Coefficients
    {
        real_t c_mu = 0.09; ///< Eddy-viscosity coefficient.
        real_t c_epsilon_1 = 1.44; ///< Epsilon production coefficient.
        real_t c_epsilon_2 = 1.92; ///< Epsilon destruction coefficient.
        real_t sigma_k = 1.0; ///< Turbulent Prandtl number for @f$k@f$.
        real_t sigma_epsilon = 1.3; ///< Turbulent Prandtl number for @f$\epsilon@f$.
    };

    StandardKEpsilonEquation() : StandardKEpsilonEquation(Coefficients{}) {}

    explicit StandardKEpsilonEquation(Coefficients coefficients) : d_coefficients(coefficients)
    {
        validate_coefficients();
    }

    const Coefficients& coefficients() const noexcept
    {
        return d_coefficients;
    }

    real_t turbulent_kinematic_viscosity(const KEpsilonState& state) const
    {
        turbulence_detail::validate(state);
        return d_coefficients.c_mu * state.k * state.k / state.epsilon;
    }

    KEpsilonDiffusivities effective_diffusivities(const KEpsilonState& state,
                                                  real_t molecular_kinematic_viscosity) const
    {
        turbulence_detail::require_non_negative(molecular_kinematic_viscosity,
                                                "Molecular kinematic viscosity");
        const auto nu_t = turbulent_kinematic_viscosity(state);
        return {molecular_kinematic_viscosity + nu_t / d_coefficients.sigma_k,
                molecular_kinematic_viscosity + nu_t / d_coefficients.sigma_epsilon};
    }

    KEpsilonDiffusivities diffusivities(const KEpsilonState& state,
                                        real_t molecular_kinematic_viscosity) const
    {
        return effective_diffusivities(state, molecular_kinematic_viscosity);
    }

    KEpsilonSourceTerms source_terms(const KEpsilonState& state, real_t production) const
    {
        turbulence_detail::validate(state);
        turbulence_detail::validate_production(production);
        return {production - state.epsilon,
                d_coefficients.c_epsilon_1 * production * state.epsilon / state.k -
                    d_coefficients.c_epsilon_2 * state.epsilon * state.epsilon / state.k};
    }

private:
    void validate_coefficients() const
    {
        turbulence_detail::require_positive(d_coefficients.c_mu, "Standard k-epsilon C-mu");
        turbulence_detail::require_positive(d_coefficients.c_epsilon_1,
                                            "Standard k-epsilon C-epsilon-1");
        turbulence_detail::require_positive(d_coefficients.c_epsilon_2,
                                            "Standard k-epsilon C-epsilon-2");
        turbulence_detail::require_positive(d_coefficients.sigma_k, "Standard k-epsilon sigma-k");
        turbulence_detail::require_positive(d_coefficients.sigma_epsilon,
                                            "Standard k-epsilon sigma-epsilon");
    }

    Coefficients d_coefficients;
};

} // namespace SimpleFluid
