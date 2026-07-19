/**
 * @file testTurbulenceEquations.cc
 * @brief Formula and validation tests for two-equation turbulence closures.
 */

#include <gtest/gtest.h>

#include "equations/turbulence/TurbulenceEquations.hh"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{

using SimpleFluid::BSLKOmegaEquation;
using SimpleFluid::KEpsilonState;
using SimpleFluid::KOmegaState;
using SimpleFluid::MenterKOmegaInvariants;
using SimpleFluid::RealizableKEpsilonEquation;
using SimpleFluid::RealizableKEpsilonInvariants;
using SimpleFluid::RNGKEpsilonEquation;
using SimpleFluid::SSTKOmegaEquation;
using SimpleFluid::StandardKEpsilonEquation;
using SimpleFluid::StandardKOmegaEquation;

} // namespace

TEST(TurbulenceEquationCommonTest, StrainProductionHasKinematicUnits)
{
    EXPECT_DOUBLE_EQ(SimpleFluid::turbulence_strain_production(0.25, 2.0), 1.0);
    EXPECT_THROW(SimpleFluid::turbulence_strain_production(-0.25, 2.0), std::invalid_argument);
    EXPECT_THROW(
        SimpleFluid::turbulence_strain_production(0.25, std::numeric_limits<double>::infinity()),
        std::invalid_argument);
}

TEST(StandardKEpsilonEquationTest, EvaluatesCanonicalClosure)
{
    const StandardKEpsilonEquation equation;
    const KEpsilonState state{2.0, 0.5};

    EXPECT_DOUBLE_EQ(equation.coefficients().c_mu, 0.09);
    EXPECT_DOUBLE_EQ(equation.coefficients().c_epsilon_1, 1.44);
    EXPECT_NEAR(equation.turbulent_kinematic_viscosity(state), 0.72, 1.0e-14);

    const auto diffusivities = equation.effective_diffusivities(state, 0.1);
    EXPECT_NEAR(diffusivities.k, 0.82, 1.0e-14);
    EXPECT_NEAR(diffusivities.epsilon, 0.1 + 0.72 / 1.3, 1.0e-14);

    const auto sources = equation.source_terms(state, 0.8);
    EXPECT_NEAR(sources.k, 0.3, 1.0e-14);
    EXPECT_NEAR(sources.epsilon, 0.048, 1.0e-14);
}

TEST(RNGKEpsilonEquationTest, AppliesSignedRapidStrainCorrection)
{
    const RNGKEpsilonEquation equation;
    const KEpsilonState state{2.0, 0.5};

    EXPECT_DOUBLE_EQ(equation.coefficients().c_mu, 0.0845);
    EXPECT_NEAR(equation.eta(state, 0.5), 2.0, 1.0e-14);

    const auto eta = 2.0;
    const auto eta_cubed = eta * eta * eta;
    const auto expected_correction =
        0.0845 * eta_cubed * (1.0 - eta / 4.38) / (1.0 + 0.012 * eta_cubed) * 0.5 * 0.5 / 2.0;
    EXPECT_NEAR(equation.rng_correction(state, 0.5), expected_correction, 1.0e-14);
    EXPECT_GT(equation.rng_correction(state, 0.5), 0.0);
    EXPECT_LT(equation.rng_correction(state, 2.0), 0.0);
    const auto eta_zero_strain = equation.coefficients().eta_zero * state.epsilon / state.k;
    EXPECT_NEAR(equation.rng_correction(state, eta_zero_strain), 0.0, 1.0e-14);

    const auto sources = equation.source_terms(state, 0.8, 0.5);
    const auto expected_epsilon =
        1.42 * 0.8 * 0.5 / 2.0 - 1.68 * 0.5 * 0.5 / 2.0 - expected_correction;
    EXPECT_NEAR(sources.k, 0.3, 1.0e-14);
    EXPECT_NEAR(sources.epsilon, expected_epsilon, 1.0e-14);

    const auto nu_t = 0.0845 * 2.0 * 2.0 / 0.5;
    const auto diffusivities = equation.diffusivities(state, 0.1);
    EXPECT_NEAR(diffusivities.k, 0.1 + nu_t / 0.71942, 1.0e-14);
    EXPECT_NEAR(diffusivities.epsilon, diffusivities.k, 1.0e-14);
}

TEST(RealizableKEpsilonEquationTest, ComputesVariableCmuAndEpsilonSource)
{
    const RealizableKEpsilonEquation equation;
    const KEpsilonState state{2.0, 0.5};
    const RealizableKEpsilonInvariants invariants{.strain_rate_magnitude = 3.0,
                                                  .u_star = 2.0,
                                                  .normalized_strain_third_invariant = 0.0,
                                                  .kinematic_viscosity = 0.1};

    const auto expected_a_s = std::sqrt(4.5);
    const auto expected_c_mu = 1.0 / (4.04 + expected_a_s * 2.0 * 2.0 / 0.5);
    EXPECT_NEAR(equation.a_s(invariants), expected_a_s, 1.0e-14);
    EXPECT_NEAR(equation.c_mu(state, invariants), expected_c_mu, 1.0e-14);
    EXPECT_NEAR(equation.turbulent_kinematic_viscosity(state, invariants),
                expected_c_mu * 2.0 * 2.0 / 0.5, 1.0e-14);

    const auto eta = 3.0 * 2.0 / 0.5;
    const auto expected_c1 = std::max(0.43, eta / (eta + 5.0));
    EXPECT_NEAR(equation.epsilon_production_coefficient(state, 3.0), expected_c1, 1.0e-14);

    const auto sources = equation.source_terms(state, 0.8, invariants);
    const auto expected_epsilon =
        expected_c1 * 3.0 * 0.5 - 1.9 * 0.5 * 0.5 / (2.0 + std::sqrt(0.1 * 0.5));
    EXPECT_NEAR(sources.k, 0.3, 1.0e-14);
    EXPECT_NEAR(sources.epsilon, expected_epsilon, 1.0e-14);

    const auto nu_t = expected_c_mu * state.k * state.k / state.epsilon;
    const auto diffusivities = equation.diffusivities(state, invariants);
    EXPECT_NEAR(diffusivities.k, 0.1 + nu_t, 1.0e-14);
    EXPECT_NEAR(diffusivities.epsilon, 0.1 + nu_t / 1.2, 1.0e-14);
    EXPECT_DOUBLE_EQ(equation.epsilon_production_coefficient(state, 0.1), 0.43);
}

TEST(RealizableKEpsilonEquationTest, EvaluatesPhysicalNonzeroThirdInvariant)
{
    const RealizableKEpsilonEquation equation;
    const RealizableKEpsilonInvariants invariants{.strain_rate_magnitude = 1.0,
                                                  .u_star = 1.0,
                                                  .normalized_strain_third_invariant = 0.1,
                                                  .kinematic_viscosity = 0.0};

    const auto sqrt_six = std::sqrt(6.0);
    const auto expected = sqrt_six * std::cos(std::acos(sqrt_six * 0.1) / 3.0);
    EXPECT_NEAR(equation.a_s(invariants), expected, 1.0e-14);
}

TEST(RealizableKEpsilonEquationTest, ClampsThirdInvariantForAcos)
{
    const RealizableKEpsilonEquation equation;
    auto invariants = RealizableKEpsilonInvariants{.strain_rate_magnitude = 1.0,
                                                   .u_star = 1.0,
                                                   .normalized_strain_third_invariant = 10.0,
                                                   .kinematic_viscosity = 0.0};
    EXPECT_NEAR(equation.a_s(invariants), std::sqrt(6.0), 1.0e-14);

    invariants.normalized_strain_third_invariant = -10.0;
    EXPECT_NEAR(equation.a_s(invariants), std::sqrt(1.5), 1.0e-14);
}

TEST(StandardKOmegaEquationTest, EvaluatesWilcox1988Closure)
{
    const StandardKOmegaEquation equation;
    const KOmegaState state{2.0, 4.0};

    EXPECT_DOUBLE_EQ(equation.coefficients().beta_star, 0.09);
    EXPECT_DOUBLE_EQ(equation.coefficients().gamma, 5.0 / 9.0);
    EXPECT_NEAR(equation.turbulent_kinematic_viscosity(state), 0.5, 1.0e-14);

    const auto diffusivities = equation.diffusivities(state, 0.1);
    EXPECT_NEAR(diffusivities.k, 0.35, 1.0e-14);
    EXPECT_NEAR(diffusivities.omega, 0.35, 1.0e-14);

    const auto sources = equation.source_terms(state, 0.8);
    EXPECT_NEAR(sources.k, 0.08, 1.0e-14);
    EXPECT_NEAR(sources.omega, (5.0 / 9.0) * 0.8 * 4.0 / 2.0 - 0.075 * 16.0, 1.0e-14);
}

TEST(BSLKOmegaEquationTest, BlendsInnerAndOuterMenterCoefficients)
{
    const BSLKOmegaEquation equation;
    const auto inner = equation.blended_coefficients(1.0);
    const auto outer = equation.blended_coefficients(0.0);

    EXPECT_DOUBLE_EQ(inner.sigma_k, 0.5);
    EXPECT_DOUBLE_EQ(inner.sigma_omega, 0.5);
    EXPECT_DOUBLE_EQ(inner.beta, 0.075);
    EXPECT_NEAR(equation.gamma_1(), 0.5531666666666667, 1.0e-14);
    EXPECT_NEAR(inner.gamma, 0.5531666666666667, 1.0e-14);
    EXPECT_DOUBLE_EQ(outer.sigma_k, 1.0);
    EXPECT_DOUBLE_EQ(outer.sigma_omega, 0.856);
    EXPECT_DOUBLE_EQ(outer.beta, 0.0828);
    EXPECT_NEAR(equation.gamma_2(), 0.4403546666666667, 1.0e-14);
    EXPECT_NEAR(outer.gamma, 0.4403546666666667, 1.0e-14);
}

TEST(BSLKOmegaEquationTest, EvaluatesBlendedSourcesAndDiffusivities)
{
    const BSLKOmegaEquation equation;
    const KOmegaState state{2.0, 4.0};
    const MenterKOmegaInvariants invariants{.kinematic_viscosity = 1.0e-5,
                                            .wall_distance = 10.0,
                                            .grad_k_dot_grad_omega = 0.1,
                                            .vorticity_magnitude = 0.0};

    const auto cd = 2.0 * 0.856 * 0.1 / 4.0;
    const auto turbulent_argument = std::sqrt(2.0) / (0.09 * 4.0 * 10.0);
    const auto viscous_argument = 500.0e-5 / (10.0 * 10.0 * 4.0);
    const auto cross_argument = 4.0 * 0.856 * 2.0 / (cd * 10.0 * 10.0);
    const auto f1_argument =
        std::min(std::max(turbulent_argument, viscous_argument), cross_argument);
    const auto expected_f1 = std::tanh(std::pow(f1_argument, 4));
    const auto f1 = equation.blending_function_1(state, invariants);
    EXPECT_NEAR(f1, expected_f1, 1.0e-14);
    EXPECT_NEAR(equation.cross_diffusion_coefficient(state, invariants), cd, 1.0e-14);

    const auto sigma_k = expected_f1 * 0.5 + (1.0 - expected_f1) * 1.0;
    const auto sigma_omega = expected_f1 * 0.5 + (1.0 - expected_f1) * 0.856;
    const auto beta = expected_f1 * 0.075 + (1.0 - expected_f1) * 0.0828;
    const auto gamma = expected_f1 * 0.5531666666666667 + (1.0 - expected_f1) * 0.4403546666666667;
    const auto cross_source = 2.0 * (1.0 - expected_f1) * 0.856 * 0.1 / 4.0;

    const auto diffusivities = equation.diffusivities(state, invariants);
    EXPECT_NEAR(diffusivities.k, 1.0e-5 + sigma_k * 0.5, 1.0e-14);
    EXPECT_NEAR(diffusivities.omega, 1.0e-5 + sigma_omega * 0.5, 1.0e-14);

    const auto sources = equation.source_terms(state, 0.8, invariants);
    EXPECT_NEAR(sources.k, 0.8 - 0.09 * 2.0 * 4.0, 1.0e-14);
    EXPECT_NEAR(sources.omega, gamma * 0.8 / 0.5 - beta * 16.0 + cross_source, 1.0e-14);

    EXPECT_NEAR(equation.production_limit(state), 14.4, 1.0e-14);
    EXPECT_NEAR(equation.limited_production(state, 100.0), 14.4, 1.0e-14);
    const auto limited_sources = equation.source_terms(state, 100.0, invariants);
    EXPECT_NEAR(limited_sources.k, 14.4 - 0.09 * 2.0 * 4.0, 1.0e-14);
    EXPECT_NEAR(limited_sources.omega, gamma * 100.0 / 0.5 - beta * 16.0 + cross_source, 1.0e-12);

    auto negative_gradient = invariants;
    negative_gradient.grad_k_dot_grad_omega = -0.1;
    EXPECT_LT(equation.cross_diffusion_source(state, negative_gradient, 0.25), 0.0);
    EXPECT_DOUBLE_EQ(equation.cross_diffusion_coefficient(state, negative_gradient),
                     equation.coefficients().cross_diffusion_floor);
}

TEST(BSLKOmegaEquationTest, ExercisesViscousAndCrossLimitF1Branches)
{
    const BSLKOmegaEquation equation;
    const KOmegaState viscous_state{1.0, 3.0};
    const MenterKOmegaInvariants viscous_invariants{.kinematic_viscosity = 0.45,
                                                    .wall_distance = 10.0,
                                                    .grad_k_dot_grad_omega = 0.0,
                                                    .vorticity_magnitude = 0.0};

    const auto viscous_argument = 500.0 * 0.45 / (10.0 * 10.0 * 3.0);
    EXPECT_NEAR(viscous_argument, 0.75, 1.0e-14);
    EXPECT_NEAR(equation.blending_function_1(viscous_state, viscous_invariants),
                std::tanh(std::pow(viscous_argument, 4)), 1.0e-14);

    const KOmegaState cross_state{2.0, 4.0};
    const MenterKOmegaInvariants cross_invariants{.kinematic_viscosity = 1.0e-5,
                                                  .wall_distance = 10.0,
                                                  .grad_k_dot_grad_omega = 1.6,
                                                  .vorticity_magnitude = 0.0};
    const auto cd = 2.0 * 0.856 * 1.6 / 4.0;
    const auto cross_argument = 4.0 * 0.856 * 2.0 / (cd * 10.0 * 10.0);
    EXPECT_NEAR(cross_argument, 0.1, 1.0e-14);
    EXPECT_NEAR(equation.blending_function_1(cross_state, cross_invariants),
                std::tanh(std::pow(cross_argument, 4)), 1.0e-14);
}

TEST(SSTKOmegaEquationTest, AppliesOriginalVorticityAndProductionLimiters)
{
    const SSTKOmegaEquation equation;
    const KOmegaState state{2.0, 4.0};
    const MenterKOmegaInvariants invariants{.kinematic_viscosity = 1.0e-5,
                                            .wall_distance = 10.0,
                                            .grad_k_dot_grad_omega = 0.1,
                                            .vorticity_magnitude = 3.0};

    const auto f2_argument =
        std::max(2.0 * std::sqrt(2.0) / (0.09 * 4.0 * 10.0), 500.0e-5 / (10.0 * 10.0 * 4.0));
    const auto expected_f2 = std::tanh(f2_argument * f2_argument);
    const auto f2 = equation.blending_function_2(state, invariants);
    const auto expected_nu_t = 0.31 * 2.0 / std::max(0.31 * 4.0, 3.0 * expected_f2);
    EXPECT_NEAR(f2, expected_f2, 1.0e-14);
    EXPECT_NEAR(equation.turbulent_kinematic_viscosity(state, invariants), expected_nu_t, 1.0e-14);

    const auto cd = 2.0 * 0.856 * 0.1 / 4.0;
    const auto turbulent_argument = std::sqrt(2.0) / (0.09 * 4.0 * 10.0);
    const auto viscous_argument = 500.0e-5 / (10.0 * 10.0 * 4.0);
    const auto cross_argument = 4.0 * 0.856 * 2.0 / (cd * 10.0 * 10.0);
    const auto f1_argument =
        std::min(std::max(turbulent_argument, viscous_argument), cross_argument);
    const auto expected_f1 = std::tanh(std::pow(f1_argument, 4));
    const auto sources = equation.source_terms(state, 100.0, invariants);
    EXPECT_NEAR(equation.blending_function_1(state, invariants), expected_f1, 1.0e-14);
    EXPECT_NEAR(equation.production_limit(state), 14.4, 1.0e-14);
    EXPECT_NEAR(equation.limited_production(state, 100.0), 14.4, 1.0e-14);
    EXPECT_NEAR(sources.k, 14.4 - 0.09 * 2.0 * 4.0, 1.0e-14);

    const auto beta = expected_f1 * 0.075 + (1.0 - expected_f1) * 0.0828;
    const auto gamma = expected_f1 * 0.5531666666666667 + (1.0 - expected_f1) * 0.4403546666666667;
    const auto cross_source = 2.0 * (1.0 - expected_f1) * 0.856 * 0.1 / 4.0;
    const auto expected_omega = gamma * 100.0 / expected_nu_t - beta * 16.0 + cross_source;
    EXPECT_NEAR(sources.omega, expected_omega, 1.0e-11);

    const auto sigma_k = expected_f1 * 0.85 + (1.0 - expected_f1) * 1.0;
    const auto sigma_omega = expected_f1 * 0.5 + (1.0 - expected_f1) * 0.856;
    const auto diffusivities = equation.diffusivities(state, invariants);
    EXPECT_NEAR(diffusivities.k, 1.0e-5 + sigma_k * expected_nu_t, 1.0e-14);
    EXPECT_NEAR(diffusivities.omega, 1.0e-5 + sigma_omega * expected_nu_t, 1.0e-14);

    auto inactive_limiter = invariants;
    inactive_limiter.vorticity_magnitude = 0.0;
    EXPECT_NEAR(equation.turbulent_kinematic_viscosity(state, inactive_limiter), 0.5, 1.0e-14);
    const auto uncapped_sources = equation.source_terms(state, 0.8, inactive_limiter);
    EXPECT_NEAR(uncapped_sources.k, 0.8 - 0.09 * 2.0 * 4.0, 1.0e-14);
}

TEST(SSTKOmegaEquationTest, ExercisesViscousF2Branch)
{
    const SSTKOmegaEquation equation;
    const KOmegaState state{1.0, 3.0};
    const MenterKOmegaInvariants invariants{.kinematic_viscosity = 0.45,
                                            .wall_distance = 10.0,
                                            .grad_k_dot_grad_omega = 0.0,
                                            .vorticity_magnitude = 0.0};

    const auto viscous_argument = 500.0 * 0.45 / (10.0 * 10.0 * 3.0);
    EXPECT_NEAR(viscous_argument, 0.75, 1.0e-14);
    EXPECT_NEAR(equation.blending_function_2(state, invariants),
                std::tanh(viscous_argument * viscous_argument), 1.0e-14);
}

TEST(TurbulenceEquationCommonTest, MenterCrossDiffusionUsesOnlyRequiredInvariants)
{
    auto invariants = MenterKOmegaInvariants{};
    invariants.grad_k_dot_grad_omega = 0.5;
    EXPECT_NEAR(BSLKOmegaEquation{}.cross_diffusion_source({1.0, 1.0}, invariants, 0.5), 0.428,
                1.0e-14);
}

TEST(TurbulenceEquationCommonTest, RejectsInvalidStatesAndCoefficients)
{
    EXPECT_THROW(StandardKEpsilonEquation(StandardKEpsilonEquation::Coefficients{.sigma_k = 0.0}),
                 std::invalid_argument);
    EXPECT_THROW(StandardKEpsilonEquation{}.source_terms({0.0, 1.0}, 0.0), std::invalid_argument);
    EXPECT_THROW(StandardKOmegaEquation{}.source_terms({1.0, 1.0}, -1.0), std::invalid_argument);
    EXPECT_THROW(BSLKOmegaEquation(BSLKOmegaEquation::Coefficients{.kappa = 1.0}),
                 std::invalid_argument);
    EXPECT_THROW(SSTKOmegaEquation(SSTKOmegaEquation::Coefficients{.kappa = 1.0}),
                 std::invalid_argument);

    const auto invariants = MenterKOmegaInvariants{};
    EXPECT_THROW(BSLKOmegaEquation{}.blending_function_1({1.0, 1.0}, invariants),
                 std::invalid_argument);

    auto realizable = RealizableKEpsilonInvariants{};
    realizable.normalized_strain_third_invariant = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(RealizableKEpsilonEquation{}.c_mu({1.0, 1.0}, realizable), std::invalid_argument);
}
