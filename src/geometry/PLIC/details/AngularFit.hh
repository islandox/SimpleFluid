/** @file AngularFit.hh
 * @brief Private PLIC angular qualification and optional-trial control.
 */
#pragma once
#include "geometry/PLIC/GeometryError.hh"
#include <array>
#include <span>

namespace SimpleFluid::PLIC::angular_detail
{
using Vector = std::array<double, 3>;
struct Qualification
{
    double first_step = 0, second_step = 0, step = 0, condition = 0, probe_step = 0;
};
enum class IterationResult
{
    Converged,
    Advanced,
    Stalled
};
/** Required derivative probes may use only the three fixed step sizes, and only
 * after a typed unresolved-arithmetic failure. Evaluator accounting is shared
 * across all attempts. A computed rank/conditioning failure is never retried. */
template<class Evaluate>
Qualification qualify_normal(const Vector& normal, const Vector& tangent, const Vector& second,
    std::span<const double> residual, double maximum_condition, Evaluate& evaluate);
/** Qualification precedes convergence. Optional failed trials retain current
 * normal/evaluation and consume the evaluator's existing work allowance. */
template<class Evaluation, class Evaluate, class Normalize>
IterationResult advance(Vector& normal, Evaluation& current, double maximum_condition, double fit_tolerance,
    double orientation_tolerance, Evaluate& evaluate, Normalize normalize, Qualification& qualification);
} // namespace SimpleFluid::PLIC::angular_detail

#include "geometry/PLIC/details/AngularFit.ipp"
