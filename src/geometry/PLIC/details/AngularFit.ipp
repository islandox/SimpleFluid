/** @file AngularFit.ipp
 * @brief Callback-dependent finite PLIC trial orchestration.
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace SimpleFluid::PLIC::angular_detail
{
template<class Evaluate>
Qualification qualify_normal(const Vector& normal, const Vector& tangent, const Vector& second,
    std::span<const double> residual, double maximum_condition, Evaluate& evaluate, double maximum_probe_step,
    Qualification* evidence, AxialProbes* axial)
{
    using Evaluation = std::invoke_result_t<Evaluate&, Vector>;
    std::optional<std::array<Evaluation, 4>> probes;
    double h = 0;
    for (const auto attempted_h : {1e-4, 5e-5, 2.5e-5})
    {
        if (attempted_h > maximum_probe_step)
            continue;
        if (evidence)
        {
            evidence->probe_step = attempted_h;
            evidence->step = std::numeric_limits<double>::infinity();
        }
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
    if (evidence)
    {
        evidence->probe_step = h;
        evidence->condition = std::numeric_limits<double>::infinity();
        evidence->step = std::numeric_limits<double>::infinity();
    }
    if (!(det > std::numeric_limits<double>::epsilon() * aa * bb))
        throw Unobservable("PLIC volume-fit orientation is underdetermined.");
    const double largest = .5 * (aa + bb + std::hypot(aa - bb, 2 * ab)), smallest = det / largest;
    if (evidence)
        evidence->condition = largest / smallest;
    if (largest / smallest > maximum_condition ||
        std::sqrt(smallest / residual.size()) <= 64 * std::numeric_limits<double>::epsilon() / h)
        throw Unobservable("PLIC clipped-volume orientation is ill-conditioned or unobservable.");
    Qualification result;
    result.first_step = (-bb * ra + ab * rb) / det;
    result.second_step = (ab * ra - aa * rb) / det;
    result.step = std::hypot(result.first_step, result.second_step);
    result.condition = largest / smallest;
    result.probe_step = h;
    result.normal_aa = aa;
    result.normal_ab = ab;
    result.normal_bb = bb;
    result.gradient_a = ra;
    result.gradient_b = rb;
    long double linearized_squared = 0;
    for (std::size_t i = 0; i < residual.size(); ++i)
    {
        const long double da = (static_cast<long double>(ea.residual[i]) - ema.residual[i]) / (2 * h);
        const long double db = (static_cast<long double>(eb.residual[i]) - emb.residual[i]) / (2 * h);
        const long double value = residual[i] + da * result.first_step + db * result.second_step;
        linearized_squared += value * value;
    }
    result.linearized_rms = static_cast<double>(std::sqrt(linearized_squared / residual.size()));
    if (axial)
        for (std::size_t i = 0; i < probes->size(); ++i)
            axial->residuals[i] = (*probes)[i].residual;
    return result;
}

template<class Evaluation, class Evaluate, class Normalize>
bool observable_trial(const Vector& normal, const Evaluation& trial, std::size_t residual_size,
    double maximum_condition, Evaluate& evaluate, Normalize normalize)
{
    if (trial.residual.size() != residual_size)
        throw GeometryError(GeometryErrorCode::InvalidInput, "PLIC angular trial residual shape differs.");
    const Vector axis = std::abs(normal[0]) < .8 ? Vector{1, 0, 0} : Vector{0, 1, 0};
    const double projection = axis[0] * normal[0] + axis[1] * normal[1] + axis[2] * normal[2];
    Vector tangent;
    for (unsigned i = 0; i < 3; ++i)
        tangent[i] = axis[i] - normal[i] * projection;
    tangent = normalize(tangent);
    const Vector second{normal[1] * tangent[2] - normal[2] * tangent[1],
        normal[2] * tangent[0] - normal[0] * tangent[2], normal[0] * tangent[1] - normal[1] * tangent[0]};
    try
    {
        (void) qualify_normal(normal, tangent, second, trial.residual, maximum_condition, evaluate);
        return true;
    }
    catch (const Unobservable&)
    {
        return false;
    }
    catch (const GeometryError& error)
    {
        if (!error.recoverable_trial())
            throw;
        return false;
    }
}

inline std::array<double, 2> dogleg(const Qualification& q, double radius)
{
    const auto fallback = [&] { return std::array<double, 2>{radius * q.first_step, radius * q.second_step}; };
    const long double ga = q.gradient_a, gb = q.gradient_b;
    const long double g2 = ga * ga + gb * gb, gnorm = std::sqrt(g2);
    const long double gGg = ga * (q.normal_aa * ga + q.normal_ab * gb) +
                           gb * (q.normal_ab * ga + q.normal_bb * gb);
    if (!std::isfinite(g2) || !std::isfinite(gGg) || !(g2 > 0) || !(gGg > 0))
        return fallback();
    const long double cauchy_scale = g2 / gGg;
    const long double ca = -cauchy_scale * ga, cb = -cauchy_scale * gb;
    if (std::hypot(ca, cb) >= radius)
        return {static_cast<double>(-radius * ga / gnorm), static_cast<double>(-radius * gb / gnorm)};
    const long double da = q.first_step - ca, db = q.second_step - cb;
    const long double cd = ca * da + cb * db, d2 = da * da + db * db;
    const long double remainder = radius * radius - ca * ca - cb * cb;
    if (!(d2 > 0) || !(remainder > 0))
        return fallback();
    const long double root = std::sqrt(cd * cd + d2 * remainder);
    const long double tau = cd >= 0 ? remainder / (cd + root) : (root - cd) / d2;
    const long double a = ca + tau * da, b = cb + tau * db;
    if (!std::isfinite(a) || !std::isfinite(b) || !(ga * a + gb * b < 0))
        return fallback();
    return {static_cast<double>(a), static_cast<double>(b)};
}

// Optional local curvature search. The axial samples and the GN gate remain
// those of the current qualification; all additional evaluations are charged.
template<class Evaluation, class Evaluate, class Normalize>
std::optional<Evaluation> curvature_trial(const Vector& normal, const Vector& tangent, const Vector& second,
    const Evaluation& current, const Qualification& qualification, const AxialProbes& axial,
    double maximum_condition, Evaluate& evaluate, Normalize normalize, Vector& accepted_normal)
{
    const double h = qualification.probe_step;
    try
    {
        const auto probe = [&](double a, double b)
        {
            Vector trial = normal;
            for (unsigned i = 0; i < 3; ++i)
                trial[i] += h * (a * tangent[i] + b * second[i]);
            auto value = evaluate(trial);
            if (value.residual.size() != current.residual.size())
                throw GeometryError(GeometryErrorCode::InvalidInput, "PLIC angular probe residual shape differs.");
            return value;
        };
        const auto pp = probe(1, 1), pm = probe(1, -1), mp = probe(-1, 1), mm = probe(-1, -1);
        long double aa = 0, ab = 0, bb = 0, ga = 0, gb = 0;
        const long double lh = h, h2 = lh * lh;
        for (std::size_t i = 0; i < current.residual.size(); ++i)
        {
            const long double r = current.residual[i];
            const long double ap = axial.residuals[0][i], bp = axial.residuals[1][i];
            const long double am = axial.residuals[2][i], bm = axial.residuals[3][i];
            const long double da = (ap - am) / (2 * lh), db = (bp - bm) / (2 * lh);
            aa += da * da + r * (ap - 2 * r + am) / h2;
            bb += db * db + r * (bp - 2 * r + bm) / h2;
            ab += da * db + r * (static_cast<long double>(pp.residual[i]) - pm.residual[i] - mp.residual[i] +
                                    mm.residual[i]) / (4 * h2);
            ga += r * da;
            gb += r * db;
        }
        if (!std::isfinite(aa) || !std::isfinite(ab) || !std::isfinite(bb) ||
            !std::isfinite(ga) || !std::isfinite(gb))
            return std::nullopt;
        const long double magnitude = std::max({std::abs(aa), std::abs(ab), std::abs(bb)});
        if (!(magnitude > 0))
            return std::nullopt;
        // Scaling retains the eigenspaces and absolute spectral condition.
        // Recover the other eigenvalue from the determinant rather than
        // subtracting nearly equal trace/radius terms.
        const long double ha = aa / magnitude, hab = ab / magnitude, hb = bb / magnitude;
        const long double trace = (ha + hb) / 2, radius = std::hypot((ha - hb) / 2, hab);
        const long double outer = trace + std::copysign(radius, trace);
        const long double inner = outer != 0 ? std::fma(ha, hb, -hab * hab) / outer : 0;
        const long double minimum = std::min(outer, inner), maximum = std::max(outer, inner);
        if (!std::isfinite(minimum) || !std::isfinite(maximum))
            return std::nullopt;
        if (minimum < 0)
        {
            const long double least = std::min(std::abs(minimum), std::abs(maximum));
            const long double greatest = std::max(std::abs(minimum), std::abs(maximum));
            if (!(least > 0) || greatest / least > maximum_condition)
                return std::nullopt;
            long double a = 0, b = 0;
            if (radius == 0)
            {
                // The negative isotropic eigenspace has no preferred axis.
                const long double gradient = std::hypot(ga, gb);
                if (!std::isfinite(gradient))
                    return std::nullopt;
                a = gradient > 0 ? -ga / gradient : 1;
                b = gradient > 0 ? -gb / gradient : 0;
            }
            else
            {
                a = hab;
                b = minimum - ha;
                const long double other_a = minimum - hb, other_b = hab;
                if (std::hypot(other_a, other_b) > std::hypot(a, b))
                {
                    a = other_a;
                    b = other_b;
                }
                const long double length = std::hypot(a, b);
                if (!std::isfinite(length) || !(length > 0))
                    return std::nullopt;
                a /= length;
                b /= length;
            }
            // Fix an exactly gradient-orthogonal eigenvector's otherwise
            // arbitrary sign in physical coordinates. Both this direction and
            // the normal reverse on phase exchange; a tangent-coordinate sign
            // alone would privilege one phase. Gradient descent takes priority.
            std::array<long double, 3> direction{};
            unsigned direction_axis = 0, normal_axis = 0;
            for (unsigned i = 0; i < 3; ++i)
            {
                direction[i] = a * tangent[i] + b * second[i];
                if (std::abs(direction[i]) > std::abs(direction[direction_axis]))
                    direction_axis = i;
                if (std::abs(normal[i]) > std::abs(normal[normal_axis]))
                    normal_axis = i;
            }
            if (std::signbit(direction[direction_axis]) != std::signbit(normal[normal_axis]))
            {
                a = -a;
                b = -b;
            }
            const long double slope = ga * a + gb * b;
            if (!std::isfinite(slope))
                return std::nullopt;
            if (slope > 0)
            {
                a = -a;
                b = -b;
            }
            // Do not invert or shift an indefinite Hessian. Actual merit and
            // the unchanged trial-Jacobian qualification decide acceptance at
            // the same bounded dyadic radii as the qualified GN fallback.
            for (double scale = 1; scale >= 1. / 128; scale *= .5)
            {
                try
                {
                    Vector trial = normal;
                    for (unsigned i = 0; i < 3; ++i)
                        trial[i] += static_cast<double>(scale * (a * tangent[i] + b * second[i]));
                    trial = normalize(trial);
                    auto next = evaluate(trial);
                    if (next.residual.size() != current.residual.size())
                        throw GeometryError(GeometryErrorCode::InvalidInput,
                            "PLIC angular trial residual shape differs.");
                    if (next.objective < current.objective &&
                        observable_trial(trial, next, current.residual.size(), maximum_condition, evaluate, normalize))
                    {
                        accepted_normal = trial;
                        return next;
                    }
                }
                catch (const Unobservable&)
                {
                    // A smaller radius can retain the observed angular rank.
                }
                catch (const GeometryError& error)
                {
                    if (!error.recoverable_trial())
                        throw;
                }
            }
            return std::nullopt;
        }
        const long double det = aa * bb - ab * ab;
        const long double largest = (aa + bb + std::hypot(aa - bb, 2 * ab)) / 2;
        if (!std::isfinite(det) || !std::isfinite(largest) || !(aa > 0) || !(det > 0) ||
            !(largest > 0) || largest / (det / largest) > maximum_condition)
            return std::nullopt;
        const long double a = (-bb * ga + ab * gb) / det, b = (ab * ga - aa * gb) / det;
        if (!std::isfinite(a) || !std::isfinite(b) || !(ga * a + gb * b < 0) || std::hypot(a, b) > 1)
            return std::nullopt;
        Vector trial = normal;
        for (unsigned i = 0; i < 3; ++i)
            trial[i] += static_cast<double>(a * tangent[i] + b * second[i]);
        trial = normalize(trial);
        auto next = evaluate(trial);
        if (next.residual.size() != current.residual.size())
            throw GeometryError(GeometryErrorCode::InvalidInput, "PLIC angular trial residual shape differs.");
        if (next.objective < current.objective &&
            observable_trial(trial, next, current.residual.size(), maximum_condition, evaluate, normalize))
        {
            accepted_normal = trial;
            return next;
        }
    }
    catch (const GeometryError& error)
    {
        if (!error.recoverable_trial())
            throw;
    }
    return std::nullopt;
}

template<class Evaluation, class Evaluate, class Normalize>
IterationResult advance(Vector& normal, Evaluation& current, double maximum_condition, double fit_tolerance,
    double orientation_tolerance, Evaluate& evaluate, Normalize normalize, Qualification& qualification)
{
    qualification = {};
    qualification.step = std::numeric_limits<double>::infinity();
    const Vector axis = std::abs(normal[0]) < .8 ? Vector{1, 0, 0} : Vector{0, 1, 0};
    const double projection = axis[0] * normal[0] + axis[1] * normal[1] + axis[2] * normal[2];
    Vector tangent;
    for (unsigned i = 0; i < 3; ++i)
        tangent[i] = axis[i] - normal[i] * projection;
    tangent = normalize(tangent);
    const Vector second{normal[1] * tangent[2] - normal[2] * tangent[1],
        normal[2] * tangent[0] - normal[0] * tangent[2], normal[0] * tangent[1] - normal[1] * tangent[0]};
    // A nonzero-residual optimum can expose the O(h^2) derivative bias before
    // the unchanged angular-step gate is met. Reassess sub-probe steps and
    // stalled searches using the existing finer scales and shared budget.
    bool deferred_line_search = false;
    for (const auto maximum_probe_step : {1e-4, 5e-5, 2.5e-5})
    {
        // Arithmetic fallback may already have reached a finer scale.
        if (qualification.probe_step != 0 && maximum_probe_step >= qualification.probe_step)
            continue;
        const auto previous = qualification;
        AxialProbes axial;
        std::exception_ptr refinement_failure;
        try
        {
            qualification = qualify_normal(normal, tangent, second, current.residual, maximum_condition, evaluate,
                maximum_probe_step, &qualification, &axial);
        }
        catch (const GeometryError& error)
        {
            if (!deferred_line_search || !error.recoverable_trial())
                throw;
            refinement_failure = std::current_exception();
            qualification = previous;
        }
        deferred_line_search = false;
        if (std::sqrt(current.objective_upper / current.residual.size()) <= fit_tolerance ||
            qualification.step <= orientation_tolerance)
            return IterationResult::Converged;
        if (!refinement_failure && qualification.step < qualification.probe_step && qualification.probe_step > 2.5e-5)
        {
            deferred_line_search = true;
            continue;
        }
        if (!refinement_failure && qualification.linearized_rms > fit_tolerance)
        {
            Vector trial;
            if (auto next = curvature_trial(normal, tangent, second, current, qualification, axial,
                    maximum_condition, evaluate, normalize, trial))
            {
                normal = trial;
                current = std::move(*next);
                return IterationResult::Advanced;
            }
        }
        for (double scale = 1; scale >= 1. / 128; scale *= .5)
        {
            const auto step = qualification.step > 1 ? dogleg(qualification, scale)
                : std::array<double, 2>{scale * qualification.first_step, scale * qualification.second_step};
            Vector trial = normal;
            for (unsigned i = 0; i < 3; ++i)
                trial[i] += step[0] * tangent[i] + step[1] * second[i];
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
            if (next->objective < current.objective &&
                observable_trial(trial, *next, current.residual.size(), maximum_condition, evaluate, normalize))
            {
                normal = trial;
                current = std::move(*next);
                return IterationResult::Advanced;
            }
        }
        if (refinement_failure)
            std::rethrow_exception(refinement_failure);
    }
    return IterationResult::Stalled;
}
} // namespace SimpleFluid::PLIC::angular_detail
