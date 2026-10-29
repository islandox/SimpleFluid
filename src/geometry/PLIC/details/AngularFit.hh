/** @file AngularFit.hh
 * @brief Private PLIC angular qualification and optional-trial control.
 */
#pragma once
#include "geometry/PLIC/GeometryError.hh"
#include <array>
#include <span>
#include <vector>

namespace SimpleFluid::PLIC::angular_detail
{
using Vector = std::array<double, 3>;
struct Qualification
{
    double first_step = 0, second_step = 0, step = 0, condition = 0, probe_step = 0;
    double normal_aa = 0, normal_ab = 0, normal_bb = 0, gradient_a = 0, gradient_b = 0;
    double linearized_rms = 0;
};
struct AxialProbes
{
    std::array<std::vector<double>, 4> residuals;
};
class Unobservable : public std::invalid_argument
{
public:
    using std::invalid_argument::invalid_argument;
};
enum class IterationResult
{
    Converged,
    Advanced,
    Stalled
};
/** Required derivative probes use only the three fixed step sizes. Arithmetic
 * fallback and refinement of sub-probe steps or stalled searches share evaluator accounting.
 * A computed rank/conditioning failure is never retried. */
template<class Evaluate>
Qualification qualify_normal(const Vector& normal, const Vector& tangent, const Vector& second,
    std::span<const double> residual, double maximum_condition, Evaluate& evaluate, double maximum_probe_step = 1e-4,
    Qualification* evidence = nullptr, AxialProbes* axial = nullptr);
/** Qualification precedes convergence. An optional bounded curvature trial
 * may improve the qualified normal. Failed optional trials retain current
 * normal/evaluation and consume the existing work allowance. */
template<class Evaluation, class Evaluate, class Normalize>
IterationResult advance(Vector& normal, Evaluation& current, double maximum_condition, double fit_tolerance,
    double orientation_tolerance, Evaluate& evaluate, Normalize normalize, Qualification& qualification);
} // namespace SimpleFluid::PLIC::angular_detail

#include "geometry/PLIC/details/AngularFit.ipp"
