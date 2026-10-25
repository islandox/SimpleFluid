/** @file AngularFit.ipp
 * @brief Callback-dependent finite PLIC trial orchestration.
 */
#pragma once
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace SimpleFluid::PLIC::angular_detail
{
template<class Evaluate>
Qualification qualify_normal(const Vector& normal, const Vector& tangent, const Vector& second,
    std::span<const double> residual, double maximum_condition, Evaluate& evaluate)
{
    using Evaluation = std::invoke_result_t<Evaluate&, Vector>;
    std::optional<std::array<Evaluation, 4>> probes;
    double h = 0;
    for (const auto attempted_h : {1e-4, 5e-5, 2.5e-5})
    {
        Vector a = normal, b = normal, am = normal, bm = normal;
        for (unsigned i = 0; i < 3; ++i)
        {
            a[i] += attempted_h * tangent[i];
            b[i] += attempted_h * second[i];
            am[i] -= attempted_h * tangent[i];
            bm[i] -= attempted_h * second[i];
        }
        try
        {
            probes.emplace(std::array<Evaluation, 4>{evaluate(a), evaluate(b), evaluate(am), evaluate(bm)});
            h = attempted_h;
            break;
        }
        catch (const GeometryError& error)
        {
            if (!error.recoverable_trial())
                throw;
        }
    }
    if (!probes)
        throw GeometryError(GeometryErrorCode::UnresolvedArithmetic,
            "PLIC current-normal qualification exhausted its finite-difference arithmetic probes.");
    for (const auto& probe : *probes)
        if (probe.residual.size() != residual.size())
            throw GeometryError(GeometryErrorCode::InvalidInput, "PLIC angular probe residual shape differs.");
    const auto& [ea, eb, ema, emb] = *probes;
    double aa = 0, ab = 0, bb = 0, ra = 0, rb = 0;
    for (std::size_t i = 0; i < residual.size(); ++i)
    {
        const double da = (ea.residual[i] - ema.residual[i]) / (2 * h);
        const double db = (eb.residual[i] - emb.residual[i]) / (2 * h);
        aa += da * da;
        ab += da * db;
        bb += db * db;
        ra += da * residual[i];
        rb += db * residual[i];
    }
    const double det = aa * bb - ab * ab;
    if (!(det > std::numeric_limits<double>::epsilon() * aa * bb))
        throw std::invalid_argument("PLIC volume-fit orientation is underdetermined.");
    const double largest = .5 * (aa + bb + std::hypot(aa - bb, 2 * ab)), smallest = det / largest;
    if (largest / smallest > maximum_condition ||
        std::sqrt(smallest / residual.size()) <= 64 * std::numeric_limits<double>::epsilon() / h)
        throw std::invalid_argument("PLIC clipped-volume orientation is ill-conditioned or unobservable.");
    Qualification result;
    result.first_step = (-bb * ra + ab * rb) / det;
    result.second_step = (ab * ra - aa * rb) / det;
    result.step = std::hypot(result.first_step, result.second_step);
    result.condition = largest / smallest;
    result.probe_step = h;
    return result;
}

template<class Evaluation, class Evaluate, class Normalize>
IterationResult advance(Vector& normal, Evaluation& current, double maximum_condition, double fit_tolerance,
    double orientation_tolerance, Evaluate& evaluate, Normalize normalize, Qualification& qualification)
{
    const Vector axis = std::abs(normal[0]) < .8 ? Vector{1, 0, 0} : Vector{0, 1, 0};
    const double projection = axis[0] * normal[0] + axis[1] * normal[1] + axis[2] * normal[2];
    Vector tangent;
    for (unsigned i = 0; i < 3; ++i)
        tangent[i] = axis[i] - normal[i] * projection;
    tangent = normalize(tangent);
    const Vector second{normal[1] * tangent[2] - normal[2] * tangent[1],
        normal[2] * tangent[0] - normal[0] * tangent[2], normal[0] * tangent[1] - normal[1] * tangent[0]};
    qualification = qualify_normal(normal, tangent, second, current.residual, maximum_condition, evaluate);
    if (std::sqrt(current.objective_upper / current.residual.size()) <= fit_tolerance ||
        qualification.step <= orientation_tolerance)
        return IterationResult::Converged;
    for (double scale = 1; scale >= 1. / 128; scale *= .5)
    {
        Vector trial = normal;
        for (unsigned i = 0; i < 3; ++i)
            trial[i] += scale * (qualification.first_step * tangent[i] + qualification.second_step * second[i]);
        trial = normalize(trial);
        std::optional<Evaluation> next;
        try
        {
            next.emplace(evaluate(trial));
        }
        catch (const GeometryError& error)
        {
            if (!error.recoverable_trial())
                throw;
            continue;
        }
        if (next->objective < current.objective)
        {
            normal = trial;
            current = std::move(*next);
            return IterationResult::Advanced;
        }
    }
    return IterationResult::Stalled;
}
} // namespace SimpleFluid::PLIC::angular_detail
