/**
 * @file RNGKEpsilonEquation.hh
 * @brief Renormalization-group k-epsilon closure equation terms.
 */

#pragma once

#include "equations/turbulence/TurbulenceEquationCommon.hh"

#include <cmath>

namespace SimpleFluid
{

/**
 * @brief High-Reynolds-number RNG k-epsilon closure.
 *
 * This form includes the RNG rapid-strain correction but no optional swirl
 * or low-Reynolds-number refinements.
 */
class RNGKEpsilonEquation
{
public:
    struct Coefficients
    {
        real_t c_mu = 0.0845;
        real_t c_epsilon_1 = 1.42;
        real_t c_epsilon_2 = 1.68;
        real_t sigma_k = 0.71942;
        real_t sigma_epsilon = 0.71942;
        real_t eta_zero = 4.38;
        real_t beta = 0.012;
    };

    RNGKEpsilonEquation() : RNGKEpsilonEquation(Coefficients{}) {}

    explicit RNGKEpsilonEquation(Coefficients coefficients) : d_coefficients(coefficients)
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

    real_t eta(const KEpsilonState& state, real_t strain_rate_magnitude) const
    {
        turbulence_detail::validate(state);
        turbulence_detail::require_non_negative(strain_rate_magnitude, "Strain-rate magnitude");
        return strain_rate_magnitude * state.k / state.epsilon;
    }

    /** @brief Return @f$R_\epsilon@f$, which is subtracted from the RHS. */
    real_t rng_correction(const KEpsilonState& state, real_t strain_rate_magnitude) const
    {
        const auto eta_value = eta(state, strain_rate_magnitude);
        const auto eta_cubed = eta_value * eta_value * eta_value;
        return d_coefficients.c_mu * eta_cubed * (1.0 - eta_value / d_coefficients.eta_zero) /
               (1.0 + d_coefficients.beta * eta_cubed) * state.epsilon * state.epsilon / state.k;
    }

    KEpsilonSourceTerms source_terms(const KEpsilonState& state, real_t production,
                                     real_t strain_rate_magnitude) const
    {
        turbulence_detail::validate(state);
        turbulence_detail::validate_production(production);
        const auto correction = rng_correction(state, strain_rate_magnitude);
        return {production - state.epsilon,
                d_coefficients.c_epsilon_1 * production * state.epsilon / state.k -
                    d_coefficients.c_epsilon_2 * state.epsilon * state.epsilon / state.k -
                    correction};
    }

private:
    void validate_coefficients() const
    {
        turbulence_detail::require_positive(d_coefficients.c_mu, "RNG k-epsilon C-mu");
        turbulence_detail::require_positive(d_coefficients.c_epsilon_1,
                                            "RNG k-epsilon C-epsilon-1");
        turbulence_detail::require_positive(d_coefficients.c_epsilon_2,
                                            "RNG k-epsilon C-epsilon-2");
        turbulence_detail::require_positive(d_coefficients.sigma_k, "RNG k-epsilon sigma-k");
        turbulence_detail::require_positive(d_coefficients.sigma_epsilon,
                                            "RNG k-epsilon sigma-epsilon");
        turbulence_detail::require_positive(d_coefficients.eta_zero, "RNG k-epsilon eta-zero");
        turbulence_detail::require_positive(d_coefficients.beta, "RNG k-epsilon beta");
    }

    Coefficients d_coefficients;
};

} // namespace SimpleFluid
