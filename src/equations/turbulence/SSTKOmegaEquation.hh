/**
 * @file SSTKOmegaEquation.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Original Menter-1994 shear-stress-transport closure equation terms.
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
 * @brief Original Menter-1994 SST k-omega equation closure.
 *
 * This class uses the original vorticity-based eddy-viscosity limiter and
 * the recommended k-equation-only production cap of
 * @f$20\beta^*k\omega@f$.  It intentionally does not mix in the distinct
 * SST-2003 strain limiter and ten-times production cap.
 */
class SSTKOmegaEquation
{
public:
    /** @brief Coefficients controlling the SST k-omega closure. */
    struct Coefficients
    {
        real_t beta_star = 0.09; ///< Kinetic-energy destruction coefficient.
        real_t kappa = 0.41; ///< von Karman constant.
        real_t a_one = 0.31; ///< Eddy-viscosity limiter coefficient.
        real_t sigma_k_1 = 0.85; ///< Inner @f$k@f$ diffusion coefficient.
        real_t sigma_omega_1 = 0.5; ///< Inner @f$\omega@f$ diffusion coefficient.
        real_t beta_1 = 0.075; ///< Inner omega destruction coefficient.
        real_t sigma_k_2 = 1.0; ///< Outer @f$k@f$ diffusion coefficient.
        real_t sigma_omega_2 = 0.856; ///< Outer @f$\omega@f$ diffusion coefficient.
        real_t beta_2 = 0.0828; ///< Outer omega destruction coefficient.
        real_t production_limit_factor = 20.0; ///< Kinetic-energy production cap factor.
        /** @brief Dimensional @f$CD_{k\omega}@f$ floor, in 1/s2. */
        real_t cross_diffusion_floor = 1.0e-20;
    };

    SSTKOmegaEquation() : SSTKOmegaEquation(Coefficients{}) {}

    explicit SSTKOmegaEquation(Coefficients coefficients) : d_coefficients(coefficients)
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

    real_t blending_function_2(const KOmegaState& state,
                               const MenterKOmegaInvariants& invariants) const
    {
        return turbulence_detail::menter_blending_function_2(state, invariants,
                                                             d_coefficients.beta_star);
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

    real_t turbulent_kinematic_viscosity(const KOmegaState& state,
                                         const MenterKOmegaInvariants& invariants) const
    {
        turbulence_detail::validate(state);
        turbulence_detail::validate_menter_wall_inputs(invariants);
        turbulence_detail::require_non_negative(invariants.vorticity_magnitude,
                                                "Vorticity magnitude");
        const auto f2 = blending_function_2(state, invariants);
        const auto denominator =
            std::max(d_coefficients.a_one * state.omega, invariants.vorticity_magnitude * f2);
        return d_coefficients.a_one * state.k / denominator;
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
        const auto nu_t = turbulent_kinematic_viscosity(state, invariants);
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
        turbulence_detail::validate(invariants);
        turbulence_detail::validate_production(production);
        const auto f1 = blending_function_1(state, invariants);
        const auto coefficients = blended_coefficients(f1);
        const auto nu_t = turbulent_kinematic_viscosity(state, invariants);
        return {limited_production(state, production) -
                    d_coefficients.beta_star * state.k * state.omega,
                coefficients.gamma * production / nu_t -
                    coefficients.beta * state.omega * state.omega +
                    cross_diffusion_source(state, invariants, f1)};
    }

private:
    void validate_coefficients() const
    {
        turbulence_detail::require_positive(d_coefficients.beta_star, "SST beta-star");
        turbulence_detail::require_positive(d_coefficients.kappa, "SST von Karman constant");
        turbulence_detail::require_positive(d_coefficients.a_one, "SST a-one");
        turbulence_detail::require_positive(d_coefficients.sigma_k_1, "SST inner sigma-k");
        turbulence_detail::require_positive(d_coefficients.sigma_omega_1, "SST inner sigma-omega");
        turbulence_detail::require_positive(d_coefficients.beta_1, "SST inner beta");
        turbulence_detail::require_positive(d_coefficients.sigma_k_2, "SST outer sigma-k");
        turbulence_detail::require_positive(d_coefficients.sigma_omega_2, "SST outer sigma-omega");
        turbulence_detail::require_positive(d_coefficients.beta_2, "SST outer beta");
        turbulence_detail::require_positive(d_coefficients.production_limit_factor,
                                            "SST production-limit factor");
        turbulence_detail::require_positive(d_coefficients.cross_diffusion_floor,
                                            "SST cross-diffusion floor");
        turbulence_detail::require_positive(gamma_1(), "SST inner gamma");
        turbulence_detail::require_positive(gamma_2(), "SST outer gamma");
    }

    Coefficients d_coefficients;
};

} // namespace SimpleFluid
