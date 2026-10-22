/** @file PolyhedralReconstruction.cc
 * @brief Native convex-cell PLIC reconstruction with bounded geometric fitting.
 */
#include "FVM/PolyhedralReconstruction.hh"
#include "geometry/ExactPlanarFace.hh"
#include "geometry/ExactSweptGeometry.hh"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace SimpleFluid::VoF
{
namespace
{
using Point = Meshes::GeometryPoint;
double dot(Point a, Point b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Point minus(Point a, Point b)
{
    for (unsigned i = 0; i < 3; ++i)
        a[i] -= b[i];
    return a;
}
Point unit(Point a)
{
    const double norm = std::hypot(a[0], a[1], a[2]);
    if (!std::isfinite(norm) || norm == 0)
        throw std::invalid_argument("PLIC normal is zero or nonfinite.");
    for (auto& x : a)
        x /= norm;
    return a;
}
void validate_sample(const PolyhedralPhaseSample& sample, const PolyhedralReconstructionOptions& options)
{
    if (!sample.geometry || !std::isfinite(sample.gas_volume) || !std::isfinite(sample.carrier_volume) ||
        sample.gas_volume < 0 || sample.carrier_volume < 0 || !options.maximum_offset_iterations ||
        !options.maximum_fit_iterations || !options.maximum_stencil_cells || !options.maximum_clip_evaluations ||
        !std::isfinite(options.volume_tolerance) || !std::isfinite(options.fit_tolerance) ||
        !std::isfinite(options.maximum_condition) || !std::isfinite(options.orientation_tolerance) ||
        !std::isfinite(options.maximum_model_residual) || !(options.orientation_tolerance > 0) ||
        !(options.maximum_model_residual > 0 && options.maximum_model_residual <= .5) ||
        !(options.volume_tolerance > 0) || !(options.fit_tolerance > 0) || !(options.maximum_condition >= 1))
        throw std::invalid_argument("Invalid polyhedral PLIC sample/options.");
    const auto metric = sample.geometry->moments().volume;
    const long double discrepancy = static_cast<long double>(sample.gas_volume) + sample.carrier_volume - metric.value;
    const long double bound =
        metric.binary64_absolute_error() + 8 * std::numeric_limits<double>::epsilon() * metric.value;
    if (std::abs(discrepancy) > bound)
        throw std::invalid_argument("PLIC independent phases do not occupy actual cell.");
}
Meshes::ExactGeometryMeasure phase(
    const Meshes::ExactConvexFragment& fragment, const Meshes::GeometryHalfspace& plane, bool gas)
{
    auto measures = fragment.phases(plane);
    return gas ? measures.gas : measures.carrier;
}
// Pivoted elimination also rejects geometrically deficient physical stencils.
Point solve(std::array<Point, 3> matrix, Point rhs, double limit, double& condition)
{
    double largest = 0, smallest = std::numeric_limits<double>::infinity();
    for (unsigned k = 0; k < 3; ++k)
    {
        unsigned pivot = k;
        for (unsigned i = k + 1; i < 3; ++i)
            if (std::abs(matrix[i][k]) > std::abs(matrix[pivot][k]))
                pivot = i;
        std::swap(matrix[k], matrix[pivot]);
        std::swap(rhs[k], rhs[pivot]);
        const double diagonal = std::abs(matrix[k][k]);
        largest = std::max(largest, diagonal);
        smallest = std::min(smallest, diagonal);
        if (!std::isfinite(diagonal) || diagonal == 0 || largest / smallest > limit)
            throw std::invalid_argument("PLIC physical stencil is ill-conditioned.");
        for (unsigned i = k + 1; i < 3; ++i)
        {
            const auto factor = matrix[i][k] / matrix[k][k];
            for (unsigned j = k; j < 3; ++j)
                matrix[i][j] -= factor * matrix[k][j];
            rhs[i] -= factor * rhs[k];
        }
    }
    condition = largest / smallest;
    Point result{};
    for (int i = 2; i >= 0; --i)
    {
        double x = rhs[i];
        for (unsigned j = i + 1; j < 3; ++j)
            x -= matrix[i][j] * result[j];
        result[i] = x / matrix[i][i];
    }
    return result;
}
PolyhedralInterface offset_fit(const PolyhedralPhaseSample& sample, Point normal,
    PolyhedralReconstructionOptions options, std::size_t& evaluations)
{
    validate_sample(sample, options);
    PolyhedralInterface result;
    result.pure_gas = sample.carrier_volume == 0;
    result.pure_carrier = sample.gas_volume == 0;
    result.gas.normal = unit(normal);
    const bool gas = sample.gas_volume <= sample.carrier_volume;
    const auto center = sample.geometry->centroid();
    Point support{};
    double extremum = gas ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
    for (std::size_t f = 0; f < sample.geometry->num_faces(); ++f)
        for (const auto& id : sample.geometry->face_geometry(f)->vertex_identities())
        {
            const auto point = Meshes::exact_point_display(id);
            const auto projection = dot(result.gas.normal, minus(point, center));
            if ((gas && projection < extremum) || (!gas && projection > extremum))
            {
                extremum = projection;
                support = point;
            }
        }
    // The scalar offset now resolves a minority cut relative to its supporting
    // extremum, rather than adding a microscopic offset to a cell-scale distance.
    result.gas.anchor = support;
    double lower = std::numeric_limits<double>::infinity(), upper = -lower;
    for (std::size_t f = 0; f < sample.geometry->num_faces(); ++f)
        for (const auto& id : sample.geometry->face_geometry(f)->vertex_identities())
        {
            const auto point = Meshes::exact_point_display(id);
            const auto projection = dot(result.gas.normal, minus(point, support));
            lower = std::min(lower, projection);
            upper = std::max(upper, projection);
        }
    if (result.pure_gas || result.pure_carrier)
    {
        result.gas.offset = result.pure_gas ? upper : lower;
        return result;
    }
    const double target = gas ? sample.gas_volume : sample.carrier_volume;
    const auto volume = Meshes::ExactGeometryMeasure::volume(*sample.geometry);
    const auto fragment = Meshes::ExactConvexFragment::from_geometry(*sample.geometry);
    const auto requested = Meshes::ExactGeometryMeasure::from_value(target);
    const auto tolerance = requested.scaled(options.volume_tolerance);
    auto magnitude = [](const auto& x) { return x.sign() < 0 ? x.scaled(-1) : x; };
    Meshes::ExactGeometryMeasure best;
    bool have_best = false;
    // Work in positive distance from the relevant supporting extremum.
    // A power-law seed reaches subnormal axis cuts without 1000 linear halvings.
    double lo = 0, hi = gas ? upper : -lower;
    if (!std::isfinite(hi) || !(hi > 0))
        throw std::invalid_argument("PLIC supporting projection is not representable.");
    double distance = std::exp(std::log(hi) + requested.log_ratio(volume) / 3);
    if (!(distance > lo && distance < hi))
        distance = std::midpoint(lo, hi);
    double previous_distance = 0;
    Meshes::ExactGeometryMeasure previous_volume;
    for (unsigned iteration = 0; iteration < options.maximum_offset_iterations; ++iteration)
    {
        auto plane = result.gas;
        plane.offset = gas ? distance : -distance;
        if (evaluations++ >= options.maximum_clip_evaluations)
            throw std::invalid_argument("PLIC clip evaluation budget exhausted.");
        const auto measured = phase(*fragment, plane, gas);
        const auto residual = measured - requested;
        const auto absolute = magnitude(residual);
        if (!have_best || absolute.compare(best) < 0)
        {
            have_best = true;
            best = absolute;
            result.gas.offset = plane.offset;
            result.target_volume_difference = residual.enclosure();
            result.target_volume_residual = double(result.target_volume_difference.value);
        }
        if (best.compare(tolerance) <= 0)
            break;
        if (residual.sign() < 0)
            lo = distance;
        else
            hi = distance;
        if (std::nextafter(lo, hi) >= hi)
            break;
        double next = std::midpoint(lo, hi);
        if (measured.sign() > 0)
        {
            double degree = 1;
            if (previous_volume.sign() > 0 && previous_distance != distance)
            {
                const double ratio = distance / previous_distance;
                const auto denominator = std::isfinite(ratio) && ratio > 0
                                             ? std::log(ratio)
                                             : std::log(distance) - std::log(previous_distance);
                const auto estimate = measured.log_ratio(previous_volume) / denominator;
                if (std::isfinite(estimate))
                    degree = std::clamp(estimate, 1., 3.);
            }
            const double logarithm = requested.log_ratio(measured) / degree;
            const double factor = std::exp(logarithm);
            double predicted = distance * factor;
            if (!std::isfinite(predicted) || predicted == 0)
                predicted = std::exp(std::log(distance) + logarithm);
            if (predicted > lo && predicted < hi && predicted != distance)
                next = predicted;
        }
        previous_distance = distance;
        previous_volume = measured;
        distance = next;
    }
    if (!have_best || best.compare(tolerance) > 0)
        throw std::invalid_argument("PLIC target offset did not converge on actual polyhedron.");
    return result;
}
} // namespace
void PolyhedralReconstruction::validate(const PolyhedralPhaseSample& sample, PolyhedralReconstructionOptions options)
{
    validate_sample(sample, options);
}
PolyhedralInterface PolyhedralReconstruction::solve_offset(
    const PolyhedralPhaseSample& sample, Point normal, PolyhedralReconstructionOptions options)
{
    std::size_t evaluations = 0;
    return offset_fit(sample, normal, options, evaluations);
}
PolyhedralInterface PolyhedralReconstruction::reconstruct(const PolyhedralPhaseSample& target,
    std::span<const PolyhedralPhaseSample> stencil, PolyhedralReconstructionOptions options)
{
    validate_sample(target, options);
    if (target.gas_volume == 0 || target.carrier_volume == 0)
        return solve_offset(target, {1, 0, 0}, options);
    if (stencil.size() > options.maximum_stencil_cells)
        throw std::invalid_argument("PLIC physical stencil budget exhausted.");
    if (stencil.size() < 4)
        throw std::invalid_argument("PLIC physical stencil is insufficient.");
    const auto center = target.geometry->centroid();
    const double volume = target.geometry->moments().volume.as_double();
    const double alpha = target.gas_volume / volume;
    std::array<Point, 3> matrix{};
    Point rhs{};
    for (const auto& sample : stencil)
    {
        validate_sample(sample, options);
        const auto d = minus(sample.geometry->centroid(), center);
        const auto d2 = dot(d, d);
        if (d2 == 0)
            throw std::invalid_argument("PLIC stencil repeats the target center.");
        const double delta = sample.gas_volume / sample.geometry->moments().volume.as_double() - alpha;
        for (unsigned i = 0; i < 3; ++i)
        {
            rhs[i] -= d[i] * delta / d2;
            for (unsigned j = 0; j < 3; ++j)
                matrix[i][j] += d[i] * d[j] / d2;
        }
    }
    double condition = 0;
    Point normal = unit(solve(matrix, rhs, options.maximum_condition, condition));
    struct Evaluation
    {
        PolyhedralInterface plane;
        std::vector<double> residual;
        double objective = 0;
    };
    std::size_t evaluations = 0;
    const auto evaluate = [&](Point n)
    {
        Evaluation value;
        value.plane = offset_fit(target, unit(n), options, evaluations);
        value.residual.reserve(stencil.size());
        for (const auto& sample : stencil)
        {
            const bool gas = sample.gas_volume <= sample.carrier_volume;
            if (evaluations++ >= options.maximum_clip_evaluations)
                throw std::invalid_argument("PLIC clip evaluation budget exhausted.");
            const long double measured =
                phase(*Meshes::ExactConvexFragment::from_geometry(*sample.geometry), value.plane.gas, gas)
                    .enclosure()
                    .value;
            const double residual = (measured - (gas ? sample.gas_volume : sample.carrier_volume)) /
                                    sample.geometry->moments().volume.as_double();
            value.residual.push_back(residual);
            value.objective += residual * residual;
        }
        return value;
    };
    auto current = evaluate(normal);
    bool converged = false;
    unsigned iterations = 0;
    double last_step = std::numeric_limits<double>::infinity();
    for (unsigned iteration = 0; iteration < options.maximum_fit_iterations; ++iteration)
    {
        iterations = iteration + 1;
        Point axis = std::abs(normal[0]) < .8 ? Point{1, 0, 0} : Point{0, 1, 0};
        Point tangent = unit(
            minus(axis, {normal[0] * dot(axis, normal), normal[1] * dot(axis, normal), normal[2] * dot(axis, normal)}));
        Point second{normal[1] * tangent[2] - normal[2] * tangent[1], normal[2] * tangent[0] - normal[0] * tangent[2],
            normal[0] * tangent[1] - normal[1] * tangent[0]};
        constexpr double h = 1e-4;
        Point a = normal, b = normal, am = normal, bm = normal;
        for (unsigned i = 0; i < 3; ++i)
        {
            a[i] += h * tangent[i];
            b[i] += h * second[i];
            am[i] -= h * tangent[i];
            bm[i] -= h * second[i];
        }
        auto ea = evaluate(a), eb = evaluate(b), ema = evaluate(am), emb = evaluate(bm);
        double aa = 0, ab = 0, bb = 0, ra = 0, rb = 0;
        for (std::size_t i = 0; i < stencil.size(); ++i)
        {
            double da = (ea.residual[i] - ema.residual[i]) / (2 * h), db = (eb.residual[i] - emb.residual[i]) / (2 * h);
            aa += da * da;
            ab += da * db;
            bb += db * db;
            ra += da * current.residual[i];
            rb += db * current.residual[i];
        }
        const double det = aa * bb - ab * ab;
        if (!(det > std::numeric_limits<double>::epsilon() * aa * bb))
            throw std::invalid_argument("PLIC volume-fit orientation is underdetermined.");
        const double largest = .5 * (aa + bb + std::hypot(aa - bb, 2 * ab)), smallest = det / largest;
        if (largest / smallest > options.maximum_condition ||
            std::sqrt(smallest / stencil.size()) <= 64 * std::numeric_limits<double>::epsilon() / h)
            throw std::invalid_argument("PLIC clipped-volume orientation is ill-conditioned or unobservable.");
        condition = std::max(condition, largest / smallest);
        const double da = (-bb * ra + ab * rb) / det, db = (ab * ra - aa * rb) / det;
        last_step = std::hypot(da, db);
        if (std::sqrt(current.objective / stencil.size()) <= options.fit_tolerance ||
            last_step <= options.orientation_tolerance)
        {
            converged = true;
            break;
        }
        bool accepted = false;
        for (double scale = 1; scale >= 1. / 128; scale *= .5)
        {
            Point trial = normal;
            for (unsigned i = 0; i < 3; ++i)
                trial[i] += scale * (da * tangent[i] + db * second[i]);
            trial = unit(trial);
            auto next = evaluate(trial);
            if (next.objective < current.objective)
            {
                normal = trial;
                current = std::move(next);
                accepted = true;
                break;
            }
        }
        if (!accepted)
            throw std::invalid_argument("PLIC angular volume optimizer did not reach stationarity.");
    }
    current.plane.stencil_volume_residual = std::sqrt(current.objective / stencil.size());
    current.plane.stencil_condition = condition;
    current.plane.orientation_step = last_step;
    current.plane.fit_iterations = iterations;
    current.plane.fit_converged = converged;
    if (!converged)
        throw std::invalid_argument("PLIC angular volume optimizer iteration budget exhausted.");
    if (current.plane.stencil_volume_residual > options.maximum_model_residual)
        throw std::invalid_argument("PLIC physical plane-model defect exceeds admission bound.");
    return current.plane;
}
} // namespace SimpleFluid::VoF
