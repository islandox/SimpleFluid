/**
 * @file RealizableKEpsilonEquation.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Shih realizable k-epsilon closure equation terms.
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
 * @brief Original Shih et al. realizable k-epsilon closure.
 *
 * The canonical @f$A_0=4.04@f$ value is used.  Some later implementations,
 * including OpenFOAM, use 4.0 and can select it through Coefficients.
 */
class RealizableKEpsilonEquation
{
public:
    /** @brief Coefficients controlling the realizable k-epsilon closure. */
    struct Coefficients
    {
        real_t a_zero = 4.04;
        real_t c_epsilon_2 = 1.9;
        real_t sigma_k = 1.0;
        real_t sigma_epsilon = 1.2;
    };

    RealizableKEpsilonEquation() : RealizableKEpsilonEquation(Coefficients{}) {}

    explicit RealizableKEpsilonEquation(Coefficients coefficients) : d_coefficients(coefficients)
    {
        validate_coefficients();
    }

    const Coefficients& coefficients() const noexcept
    {
        return d_coefficients;
    }

    real_t a_s(const RealizableKEpsilonInvariants& invariants) const
    {
        turbulence_detail::validate(invariants);
        constexpr real_t sqrt_six = 2.4494897427831780982;
        const auto acos_argument =
            std::clamp(sqrt_six * invariants.normalized_strain_third_invariant, -1.0, 1.0);
        const auto phi = std::acos(acos_argument) / 3.0;
        return sqrt_six * std::cos(phi);
    }

    real_t c_mu(const KEpsilonState& state, const RealizableKEpsilonInvariants& invariants) const
    {
        turbulence_detail::validate(state);
        turbulence_detail::validate(invariants);
        return 1.0 / (d_coefficients.a_zero +
                      a_s(invariants) * invariants.u_star * state.k / state.epsilon);
    }

    real_t turbulent_kinematic_viscosity(const KEpsilonState& state,
                                         const RealizableKEpsilonInvariants& invariants) const
    {
        return c_mu(state, invariants) * state.k * state.k / state.epsilon;
    }

    KEpsilonDiffusivities
    effective_diffusivities(const KEpsilonState& state,
                            const RealizableKEpsilonInvariants& invariants) const
    {
        const auto nu_t = turbulent_kinematic_viscosity(state, invariants);
        return {invariants.kinematic_viscosity + nu_t / d_coefficients.sigma_k,
                invariants.kinematic_viscosity + nu_t / d_coefficients.sigma_epsilon};
    }

    KEpsilonDiffusivities diffusivities(const KEpsilonState& state,
                                        const RealizableKEpsilonInvariants& invariants) const
    {
        return effective_diffusivities(state, invariants);
    }

    real_t epsilon_production_coefficient(const KEpsilonState& state,
                                          real_t strain_rate_magnitude) const
    {
        turbulence_detail::validate(state);
        turbulence_detail::require_non_negative(strain_rate_magnitude, "Strain-rate magnitude");
        const auto eta = strain_rate_magnitude * state.k / state.epsilon;
        return std::max(0.43, eta / (eta + 5.0));
    }

    KEpsilonSourceTerms source_terms(const KEpsilonState& state, real_t production,
                                     const RealizableKEpsilonInvariants& invariants) const
    {
        turbulence_detail::validate(state);
        turbulence_detail::validate(invariants);
        turbulence_detail::validate_production(production);
        const auto c1 = epsilon_production_coefficient(state, invariants.strain_rate_magnitude);
        const auto epsilon_denominator =
            state.k + std::sqrt(invariants.kinematic_viscosity * state.epsilon);
        return {production - state.epsilon, c1 * invariants.strain_rate_magnitude * state.epsilon -
                                                d_coefficients.c_epsilon_2 * state.epsilon *
                                                    state.epsilon / epsilon_denominator};
    }

private:
    void validate_coefficients() const
    {
        turbulence_detail::require_positive(d_coefficients.a_zero, "Realizable k-epsilon A-zero");
        turbulence_detail::require_positive(d_coefficients.c_epsilon_2,
                                            "Realizable k-epsilon C-epsilon-2");
        turbulence_detail::require_positive(d_coefficients.sigma_k, "Realizable k-epsilon sigma-k");
        turbulence_detail::require_positive(d_coefficients.sigma_epsilon,
                                            "Realizable k-epsilon sigma-epsilon");
    }

    Coefficients d_coefficients;
};

} // namespace SimpleFluid
