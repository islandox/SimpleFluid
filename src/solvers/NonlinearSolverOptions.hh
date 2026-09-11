#pragma once

namespace SimpleFluid
{

enum class NonlinearBackend
{
    NOX
};
enum class CoupledLinearization
{
    Picard,
    AnalyticNewton
};

/** Controls for the opt-in fixed-mesh nonlinear pressure-velocity solve.
 * Reference scales are positive and fixed throughout one physical step.
 * Pressure scale applies to the internal pressure p/reference_density.
 * Residual scales apply to volume-normalized momentum and continuity.
 */
struct NonlinearSolverOptions
{
    NonlinearBackend backend = NonlinearBackend::NOX;
    CoupledLinearization linearization = CoupledLinearization::AnalyticNewton;
    int maximum_iterations = 20;
    int maximum_backtracks = 12;
    int krylov_restart = 80;
    double relative_tolerance = 1.0e-8;
    double absolute_tolerance = 1.0e-10;
    double continuity_tolerance = 1.0e-10; // maximum integrated cell imbalance [m^3/s]
    double velocity_scale = 1.0;
    double pressure_scale = 1.0;
    double momentum_residual_scale = 1.0;
    double continuity_residual_scale = 1.0;
    double forcing_initial = 0.1;
    double forcing_minimum = 1.0e-6;
    double forcing_maximum = 0.5;
    double armijo = 1.0e-4;
    double backtrack_factor = 0.5;
    double minimum_step = 1.0e-8;
};

} // namespace SimpleFluid
