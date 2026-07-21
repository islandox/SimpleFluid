/**
 * @file BSLKOmegaEquation.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Original Menter-1994 baseline k-omega closure equation terms.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "equations/turbulence/TurbulenceEquationCommon.hh"

#include <algorithm>
#include <cmath>

namespace SimpleFluid
{

/**
 * @brief Menter-1994 baseline (BSL) k-omega equation closure.
 *
 * The recommended k-equation-only production cap
 * @f$20\beta^*k\omega@f$ is enabled by default.
 */
class BSLKOmegaEquation
{
public:
    /** @brief Coefficients controlling the BSL k-omega closure. */
    struct Coefficients
    {
        real_t beta_star = 0.09;
        real_t kappa = 0.41;
        real_t sigma_k_1 = 0.5;
        real_t sigma_omega_1 = 0.5;
        real_t beta_1 = 0.075;
        real_t sigma_k_2 = 1.0;
        real_t sigma_omega_2 = 0.856;
        real_t beta_2 = 0.0828;
        real_t production_limit_factor = 20.0;
        /** @brief Dimensional @f$CD_{k\omega}@f$ floor, in 1/s2. */
        real_t cross_diffusion_floor = 1.0e-20;
    };

    BSLKOmegaEquation() : BSLKOmegaEquation(Coefficients{}) {}

    explicit BSLKOmegaEquation(Coefficients coefficients) : d_coefficients(coefficients)
    {
        validate_coefficients();
    }

    const Coefficients& coefficients() const noexcept
    {
        return d_coefficients;
    }

    real_t gamma_1() const
    {
        return turbulence_detail::menter_gamma(d_coefficients.beta_1, d_coefficients.beta_star,
                                               d_coefficients.sigma_omega_1, d_coefficients.kappa);
    }

    real_t gamma_2() const
    {
        return turbulence_detail::menter_gamma(d_coefficients.beta_2, d_coefficients.beta_star,
                                               d_coefficients.sigma_omega_2, d_coefficients.kappa);
    }

    real_t cross_diffusion_coefficient(const KOmegaState& state,
                                       const MenterKOmegaInvariants& invariants) const
    {
        return turbulence_detail::menter_cross_diffusion_coefficient(
            state, invariants, d_coefficients.sigma_omega_2, d_coefficients.cross_diffusion_floor);
    }

    real_t blending_function_1(const KOmegaState& state,
                               const MenterKOmegaInvariants& invariants) const
    {
        return turbulence_detail::menter_blending_function_1(
            state, invariants, d_coefficients.beta_star, d_coefficients.sigma_omega_2,
            d_coefficients.cross_diffusion_floor);
    }

    MenterBlendedCoefficients blended_coefficients(real_t f1) const
    {
        return {turbulence_detail::blend(d_coefficients.sigma_k_1, d_coefficients.sigma_k_2, f1),
                turbulence_detail::blend(d_coefficients.sigma_omega_1, d_coefficients.sigma_omega_2,
                                         f1),
                turbulence_detail::blend(d_coefficients.beta_1, d_coefficients.beta_2, f1),
                turbulence_detail::blend(gamma_1(), gamma_2(), f1)};
    }

    real_t cross_diffusion_source(const KOmegaState& state,
                                  const MenterKOmegaInvariants& invariants, real_t f1) const
    {
        return turbulence_detail::menter_cross_diffusion_source(state, invariants,
                                                                d_coefficients.sigma_omega_2, f1);
    }

    real_t turbulent_kinematic_viscosity(const KOmegaState& state) const
    {
        turbulence_detail::validate(state);
        return state.k / state.omega;
    }

    real_t production_limit(const KOmegaState& state) const
    {
        turbulence_detail::validate(state);
        return d_coefficients.production_limit_factor * d_coefficients.beta_star * state.k *
               state.omega;
    }

    real_t limited_production(const KOmegaState& state, real_t production) const
    {
        turbulence_detail::validate_production(production);
        return std::min(production, production_limit(state));
    }

    KOmegaDiffusivities effective_diffusivities(const KOmegaState& state,
                                                const MenterKOmegaInvariants& invariants) const
    {
        const auto f1 = blending_function_1(state, invariants);
        const auto coefficients = blended_coefficients(f1);
        const auto nu_t = turbulent_kinematic_viscosity(state);
        return {invariants.kinematic_viscosity + coefficients.sigma_k * nu_t,
                invariants.kinematic_viscosity + coefficients.sigma_omega * nu_t};
    }

    KOmegaDiffusivities diffusivities(const KOmegaState& state,
                                      const MenterKOmegaInvariants& invariants) const
    {
        return effective_diffusivities(state, invariants);
    }

    KOmegaSourceTerms source_terms(const KOmegaState& state, real_t production,
                                   const MenterKOmegaInvariants& invariants) const
    {
        turbulence_detail::validate(state);
        turbulence_detail::validate_production(production);
        const auto f1 = blending_function_1(state, invariants);
        const auto coefficients = blended_coefficients(f1);
        const auto nu_t = turbulent_kinematic_viscosity(state);
        return {limited_production(state, production) -
                    d_coefficients.beta_star * state.k * state.omega,
                coefficients.gamma * production / nu_t -
                    coefficients.beta * state.omega * state.omega +
                    cross_diffusion_source(state, invariants, f1)};
    }

private:
    void validate_coefficients() const
    {
        turbulence_detail::require_positive(d_coefficients.beta_star, "BSL beta-star");
        turbulence_detail::require_positive(d_coefficients.kappa, "BSL von Karman constant");
        turbulence_detail::require_positive(d_coefficients.sigma_k_1, "BSL inner sigma-k");
        turbulence_detail::require_positive(d_coefficients.sigma_omega_1, "BSL inner sigma-omega");
        turbulence_detail::require_positive(d_coefficients.beta_1, "BSL inner beta");
        turbulence_detail::require_positive(d_coefficients.sigma_k_2, "BSL outer sigma-k");
        turbulence_detail::require_positive(d_coefficients.sigma_omega_2, "BSL outer sigma-omega");
        turbulence_detail::require_positive(d_coefficients.beta_2, "BSL outer beta");
        turbulence_detail::require_positive(d_coefficients.production_limit_factor,
                                            "BSL production-limit factor");
        turbulence_detail::require_positive(d_coefficients.cross_diffusion_floor,
                                            "BSL cross-diffusion floor");
        turbulence_detail::require_positive(gamma_1(), "BSL inner gamma");
        turbulence_detail::require_positive(gamma_2(), "BSL outer gamma");
    }

    Coefficients d_coefficients;
};

} // namespace SimpleFluid
