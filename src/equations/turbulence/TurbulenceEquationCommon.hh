/**
 * @file TurbulenceEquationCommon.hh
 * @brief Shared value types and validation for two-equation turbulence
 * closures.
 */

#pragma once

#include "dataclass/typedefs.hh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

namespace SimpleFluid
{

/**
 * @brief Local state for a k-epsilon closure evaluation.
 *
 * @p k has units m2/s2 and @p epsilon has units m2/s3.
 * These are positive interior values; the caller owns any dimensional
 * turbulence floors or wall-function limiting applied before evaluation.
 */
struct KEpsilonState
{
    real_t k = {};
    real_t epsilon = {};
};

/**
 * @brief Net right-hand-side terms for the k and epsilon equations.
 *
 * The entries have units m2/s3 and m2/s4, respectively.
 */
struct KEpsilonSourceTerms
{
    real_t k = {};
    real_t epsilon = {};
};

/** @brief Effective kinematic diffusivities for k and epsilon, in m2/s. */
struct KEpsilonDiffusivities
{
    real_t k = {};
    real_t epsilon = {};
};

/**
 * @brief Local state for a k-omega closure evaluation.
 *
 * @p k has units m2/s2 and @p omega has units 1/s.
 * These are positive interior values; the caller owns any dimensional
 * turbulence floors or wall-function limiting applied before evaluation.
 */
struct KOmegaState
{
    real_t k = {};
    real_t omega = {};
};

/**
 * @brief Net right-hand-side terms for the k and omega equations.
 *
 * The entries have units m2/s3 and 1/s2, respectively.
 */
struct KOmegaSourceTerms
{
    real_t k = {};
    real_t omega = {};
};

/** @brief Effective kinematic diffusivities for k and omega, in m2/s. */
struct KOmegaDiffusivities
{
    real_t k = {};
    real_t omega = {};
};

/**
 * @brief Velocity-gradient invariants needed by the realizable k-epsilon model.
 *
 * The strain magnitude is @f$S=\sqrt{2S_{ij}S_{ij}}@f$.  @p u_star is
 * @f$U^*=\sqrt{S_{ij}S_{ij}+\widetilde\Omega_{ij}\widetilde\Omega_{ij}}@f$
 * and @p normalized_strain_third_invariant is
 * @f$W=S_{ij}S_{jk}S_{ki}/(S_{ij}S_{ij})^{3/2}@f$.
 */
struct RealizableKEpsilonInvariants
{
    real_t strain_rate_magnitude = {};
    real_t u_star = {};
    real_t normalized_strain_third_invariant = {};
    real_t kinematic_viscosity = {};
};

/**
 * @brief Local quantities needed by the Menter BSL and SST models.
 *
 * The vorticity magnitude is @f$\Omega=\sqrt{2W_{ij}W_{ij}}@f$ and
 * @p grad_k_dot_grad_omega is the signed scalar product
 * @f$\nabla k\cdot\nabla\omega@f$.
 */
struct MenterKOmegaInvariants
{
    real_t kinematic_viscosity = {};
    real_t wall_distance = {};
    real_t grad_k_dot_grad_omega = {};
    real_t vorticity_magnitude = {};
};

/** @brief One set of coefficients after Menter inner/outer blending. */
struct MenterBlendedCoefficients
{
    real_t sigma_k = {};
    real_t sigma_omega = {};
    real_t beta = {};
    real_t gamma = {};
};

namespace turbulence_detail
{

inline void require_finite(real_t value, std::string_view name)
{
    if (!std::isfinite(value))
    {
        throw std::invalid_argument(std::string(name) + " must be finite.");
    }
}

inline void require_non_negative(real_t value, std::string_view name)
{
    require_finite(value, name);
    if (value < 0.0)
    {
        throw std::invalid_argument(std::string(name) + " must be non-negative.");
    }
}

inline void require_positive(real_t value, std::string_view name)
{
    require_finite(value, name);
    if (value <= 0.0)
    {
        throw std::invalid_argument(std::string(name) + " must be positive.");
    }
}

inline void validate(const KEpsilonState& state)
{
    require_positive(state.k, "Turbulent kinetic energy k");
    require_positive(state.epsilon, "Dissipation rate epsilon");
}

inline void validate(const KOmegaState& state)
{
    require_positive(state.k, "Turbulent kinetic energy k");
    require_positive(state.omega, "Specific dissipation rate omega");
}

inline void validate(const RealizableKEpsilonInvariants& invariants)
{
    require_non_negative(invariants.strain_rate_magnitude, "Strain-rate magnitude");
    require_non_negative(invariants.u_star, "Realizable U-star");
    require_finite(invariants.normalized_strain_third_invariant,
                   "Normalized strain third invariant");
    require_non_negative(invariants.kinematic_viscosity, "Molecular kinematic viscosity");
}

inline void validate(const MenterKOmegaInvariants& invariants)
{
    require_non_negative(invariants.kinematic_viscosity, "Molecular kinematic viscosity");
    require_positive(invariants.wall_distance, "Wall distance");
    require_finite(invariants.grad_k_dot_grad_omega, "Gradient dot product");
    require_non_negative(invariants.vorticity_magnitude, "Vorticity magnitude");
}

inline void validate_menter_wall_inputs(const MenterKOmegaInvariants& invariants)
{
    require_non_negative(invariants.kinematic_viscosity, "Molecular kinematic viscosity");
    require_positive(invariants.wall_distance, "Wall distance");
}

inline void validate_production(real_t production)
{
    require_non_negative(production, "Turbulent kinetic-energy production");
}

inline real_t blend(real_t inner, real_t outer, real_t f1)
{
    require_finite(f1, "Menter blending factor");
    if (f1 < 0.0 || f1 > 1.0)
    {
        throw std::invalid_argument("Menter blending factor must be in [0, 1].");
    }
    return f1 * inner + (1.0 - f1) * outer;
}

inline void validate_blending_factor(real_t f1)
{
    (void)blend(0.0, 0.0, f1);
}

inline real_t menter_gamma(real_t beta, real_t beta_star, real_t sigma_omega, real_t kappa)
{
    return beta / beta_star - sigma_omega * kappa * kappa / std::sqrt(beta_star);
}

inline real_t menter_cross_diffusion_coefficient(const KOmegaState& state,
                                                 const MenterKOmegaInvariants& invariants,
                                                 real_t sigma_omega_2, real_t floor)
{
    validate(state);
    require_finite(invariants.grad_k_dot_grad_omega, "Gradient dot product");
    require_positive(sigma_omega_2, "Outer omega diffusion coefficient");
    require_positive(floor, "Cross-diffusion floor");
    return std::max(2.0 * sigma_omega_2 * invariants.grad_k_dot_grad_omega / state.omega, floor);
}

inline real_t menter_blending_function_1(const KOmegaState& state,
                                         const MenterKOmegaInvariants& invariants, real_t beta_star,
                                         real_t sigma_omega_2, real_t cross_diffusion_floor)
{
    validate(state);
    validate_menter_wall_inputs(invariants);
    require_finite(invariants.grad_k_dot_grad_omega, "Gradient dot product");
    require_positive(beta_star, "Menter beta-star");
    const auto cd =
        menter_cross_diffusion_coefficient(state, invariants, sigma_omega_2, cross_diffusion_floor);
    const auto y2 = invariants.wall_distance * invariants.wall_distance;
    const auto viscous = 500.0 * invariants.kinematic_viscosity / (y2 * state.omega);
    const auto turbulent =
        std::sqrt(state.k) / (beta_star * state.omega * invariants.wall_distance);
    const auto cross_limit = 4.0 * sigma_omega_2 * state.k / (cd * y2);
    const auto argument = std::min(std::max(turbulent, viscous), cross_limit);
    const auto argument_squared = argument * argument;
    return std::tanh(argument_squared * argument_squared);
}

inline real_t menter_blending_function_2(const KOmegaState& state,
                                         const MenterKOmegaInvariants& invariants, real_t beta_star)
{
    validate(state);
    validate_menter_wall_inputs(invariants);
    require_positive(beta_star, "Menter beta-star");
    const auto y2 = invariants.wall_distance * invariants.wall_distance;
    const auto argument =
        std::max(2.0 * std::sqrt(state.k) / (beta_star * state.omega * invariants.wall_distance),
                 500.0 * invariants.kinematic_viscosity / (y2 * state.omega));
    return std::tanh(argument * argument);
}

inline real_t menter_cross_diffusion_source(const KOmegaState& state,
                                            const MenterKOmegaInvariants& invariants,
                                            real_t sigma_omega_2, real_t f1)
{
    validate(state);
    require_finite(invariants.grad_k_dot_grad_omega, "Gradient dot product");
    require_positive(sigma_omega_2, "Outer omega diffusion coefficient");
    validate_blending_factor(f1);
    return 2.0 * (1.0 - f1) * sigma_omega_2 * invariants.grad_k_dot_grad_omega / state.omega;
}

} // namespace turbulence_detail

/**
 * @brief Linear eddy-viscosity production approximation
 *        @f$P_k=\nu_t S^2@f$.
 *
 * This is the Boussinesq stress hypothesis, not the thermal Boussinesq
 * buoyancy approximation. The result has units m2/s3.
 */
inline real_t turbulence_strain_production(real_t turbulent_kinematic_viscosity,
                                           real_t strain_rate_magnitude)
{
    turbulence_detail::require_non_negative(turbulent_kinematic_viscosity,
                                            "Turbulent kinematic viscosity");
    turbulence_detail::require_non_negative(strain_rate_magnitude, "Strain-rate magnitude");
    return turbulent_kinematic_viscosity * strain_rate_magnitude * strain_rate_magnitude;
}

} // namespace SimpleFluid
