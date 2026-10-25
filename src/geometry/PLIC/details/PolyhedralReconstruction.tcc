/** @file PolyhedralReconstruction.tcc
 * @brief Resolution admission and mode-selected finite PLIC reconstruction.
 */
#include "geometry/PLIC/PolyhedralReconstruction.hh"
#include "geometry/PLIC/details/AngularFit.hh"
#include "geometry/PLIC/details/PhaseArithmetic.hh"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <vector>

namespace SimpleFluid::PLIC
{
namespace
{
using Vector = SimpleFluid::PLIC::GeometryPoint;
template<ArithmeticMode Mode> using Value = PLIC::GeometryValue<Mode>;
constexpr long double infinity = std::numeric_limits<long double>::infinity();

double dot(Vector a, Vector b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector minus(Vector a, Vector b)
{
    for (unsigned i = 0; i < 3; ++i)
        a[i] -= b[i];
    return a;
}
Vector unit(Vector a)
{
    const double norm = std::hypot(a[0], a[1], a[2]);
    if (!std::isfinite(norm) || norm == 0)
        throw std::invalid_argument("Finite PLIC normal is zero or nonfinite.");
    for (auto& x : a)
        x /= norm;
    return a;
}
long double certified_upward(long double x)
{
    if (!std::isfinite(x) || x < 0)
        throw std::invalid_argument("Finite PLIC arithmetic bound is unrepresentable.");
    return x == 0 ? 0 : std::nextafter(x, infinity);
}
Value<ArithmeticMode::BoundedCertification> certified_difference(
    Value<ArithmeticMode::BoundedCertification> a, long double b)
{
    const auto value = a.value - b;
    // Error-free TwoSum for the represented subtraction; original geometry
    // uncertainty remains separate and is never inferred from its rounded sign.
    const auto z = value - a.value;
    const auto residual = (a.value - (value - z)) + (-b - z);
    return {value, certified_upward(a.absolute_error + std::abs(residual))};
}
long double certified_fit_error(Value<ArithmeticMode::BoundedCertification> residual)
{
    return certified_upward(std::abs(residual.value) + residual.absolute_error);
}
long double certified_fit_allowance(long double requested, double tolerance)
{
    using Math = VoF::finite_detail::Arithmetic<ArithmeticMode::BoundedCertification>;
    return Math::lower(Math::multiply({requested, 0}, {tolerance, 0}));
}
Value<ArithmeticMode::Ordinary> ordinary_difference(Value<ArithmeticMode::Ordinary> a, long double b)
{
    const auto result = VoF::finite_detail::Arithmetic<ArithmeticMode::Ordinary>::make_value(a.value - b);
    return {result.value};
}
long double ordinary_fit_allowance(long double requested, double tolerance)
{
    const auto result = requested * static_cast<long double>(tolerance);
    if (!std::isfinite(result) || result < 0)
        throw std::invalid_argument("Ordinary PLIC target tolerance is unrepresentable.");
    return result;
}
template<ArithmeticMode Mode> Value<Mode> difference(Value<Mode> a, long double b)
{
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return ordinary_difference(a, b);
    else
        return certified_difference(a, b);
}
template<ArithmeticMode Mode> long double fit_error(Value<Mode> residual)
{
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return std::abs(residual.value);
    else
        return certified_fit_error(residual);
}
template<ArithmeticMode Mode> long double fit_allowance(long double requested, double tolerance)
{
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return ordinary_fit_allowance(requested, tolerance);
    else
        return certified_fit_allowance(requested, tolerance);
}
template<ArithmeticMode Mode> long double lower(Value<Mode> value)
{
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return value.value;
    else
        return value.lower();
}
template<ArithmeticMode Mode> long double upper(Value<Mode> value)
{
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return value.value;
    else
        return value.upper();
}
template<ArithmeticMode Mode> long double radius(Value<Mode> value)
{
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return 0;
    else
        return value.absolute_error;
}
template<ArithmeticMode Mode> PLIC::DynamicGeometryValue published_difference(Value<Mode> value)
{
    return {value.value, radius<Mode>(value), Mode};
}
double outward(long double x, bool upper)
{
    double result = static_cast<double>(x);
    if (upper ? static_cast<long double>(result) < x : static_cast<long double>(result) > x)
        result = std::nextafter(
            result, upper ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity());
    if (!std::isfinite(result))
        throw std::invalid_argument("Finite PLIC support cannot be represented.");
    return result;
}
template<ArithmeticMode Mode>
void validate_sample(const PolyhedralPhaseSample<Mode>& sample, const PolyhedralReconstructionOptions& options)
{
    if (Mode == ArithmeticMode::BoundedCertification && !PLIC::certification_environment())
        throw std::invalid_argument("Finite PLIC arithmetic environment is unsupported.");
    if (!sample.geometry || !std::isfinite(sample.gas_volume) || !std::isfinite(sample.carrier_volume) ||
        sample.gas_volume < 0 || sample.carrier_volume < 0 || !options.maximum_offset_iterations ||
        !options.maximum_fit_iterations || !options.maximum_stencil_cells || !options.maximum_clip_evaluations ||
        !std::isfinite(options.volume_tolerance) || !std::isfinite(options.fit_tolerance) ||
        !std::isfinite(options.maximum_condition) || !std::isfinite(options.orientation_tolerance) ||
        !std::isfinite(options.maximum_model_residual) || !(options.orientation_tolerance > 0) ||
        !(options.maximum_model_residual > 0 && options.maximum_model_residual <= .5) ||
        !(options.volume_tolerance > 0) || !(options.fit_tolerance > 0) || !(options.maximum_condition >= 1) ||
        !std::isfinite(options.minimum_phase_support_fraction) ||
        !(options.minimum_phase_support_fraction > 0 && options.minimum_phase_support_fraction <= .5))
        throw std::invalid_argument("Invalid finite polyhedral PLIC sample/options.");
    const auto volume = sample.geometry->volume();
    if (!(lower<Mode>(volume) > 0))
        throw std::invalid_argument("Finite PLIC cell volume is not established positive.");
    const auto total = static_cast<long double>(sample.gas_volume) + sample.carrier_volume;
    const auto residual = difference<Mode>(volume, total);
    const auto allowance = radius<Mode>(volume) + 8 * std::numeric_limits<double>::epsilon() * std::abs(volume.value);
    const auto bound = [&]
    {
        if constexpr (Mode == ArithmeticMode::Ordinary)
            return allowance;
        else
            return certified_upward(allowance);
    }();
    if (std::abs(residual.value) > bound)
        throw std::invalid_argument("Finite PLIC independent phases do not occupy the actual cell.");
}
std::string resolution_message(const PolyhedralResolutionAssessment& assessment)
{
    std::ostringstream message;
    message << "Finite PLIC requires mesh/stencil refinement";
    switch (assessment.reason)
    {
        case PolyhedralResolutionAssessment::Reason::InsufficientStencil:
            message << ": physical normal support is insufficient";
            break;
        case PolyhedralResolutionAssessment::Reason::InsufficientPhaseSupport:
            message << ": an interface phase lacks resolved neighborhood support";
            break;
        case PolyhedralResolutionAssessment::Reason::UnresolvedPlaneModel:
            message << ": the local plane model exceeds the unchanged defect bound";
            break;
        default:
            message << ": invalid resolution assessment";
            break;
    }
    message << ". Refine the interface and stencil neighborhood (requested maximum span "
            << assessment.requested_cell_span
            << "), recover missing initial geometry if needed, and reassess; accepted inventories are unchanged.";
    return message.str();
}
Vector solve(std::array<Vector, 3> matrix, Vector rhs, double limit, double& condition)
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
            throw std::invalid_argument("Finite PLIC physical stencil is ill-conditioned.");
        for (unsigned i = k + 1; i < 3; ++i)
        {
            const auto factor = matrix[i][k] / matrix[k][k];
            for (unsigned j = k; j < 3; ++j)
                matrix[i][j] -= factor * matrix[k][j];
            rhs[i] -= factor * rhs[k];
        }
    }
    condition = largest / smallest;
    Vector result{};
    for (int i = 2; i >= 0; --i)
    {
        double x = rhs[i];
        for (unsigned j = i + 1; j < 3; ++j)
            x -= matrix[i][j] * result[j];
        result[i] = x / matrix[i][i];
    }
    return result;
}
template<ArithmeticMode Mode>
PolyhedralInterface offset_fit(const PolyhedralPhaseSample<Mode>& sample, Vector normal,
    PolyhedralReconstructionOptions options, std::size_t& evaluations)
{
    PolyhedralInterface result;
    result.finite_geometry = true;
    result.finite_target_volume_difference = published_difference<Mode>({});
    result.pure_gas = sample.carrier_volume == 0;
    result.pure_carrier = sample.gas_volume == 0;
    result.gas.normal = unit(normal);
    const bool gas = sample.gas_volume <= sample.carrier_volume;
    const auto reference = sample.geometry->origin();
    const auto points = sample.geometry->local_points();
    long double extremum = gas ? infinity : -infinity;
    bool have_anchor = false;
    for (const auto& point : points)
    {
        long double projection = 0;
        for (unsigned axis = 0; axis < 3; ++axis)
            projection += static_cast<long double>(result.gas.normal[axis]) * point.value[axis];
        if (!have_anchor || (gas ? projection < extremum : projection > extremum))
        {
            have_anchor = true;
            extremum = projection;
            for (unsigned axis = 0; axis < 3; ++axis)
            {
                result.gas.anchor[axis] =
                    static_cast<double>(static_cast<long double>(reference[axis]) + point.value[axis]);
                if (!std::isfinite(result.gas.anchor[axis]))
                    throw std::invalid_argument("Finite PLIC support anchor exceeds its represented range.");
            }
        }
    }
    if (!have_anchor)
        throw std::invalid_argument("Finite PLIC geometry has no support anchor.");
    // This vertex choice only conditions the scalar representation. The full
    // enclosed support query below still proves a signed bracket, including
    // nonbinary support behind the displayed anchor. Never assume a zero bound.
    const auto support = sample.geometry->support(result.gas.normal, result.gas.anchor);
    double lo = outward(lower<Mode>(support.lower), false);
    double hi = outward(upper<Mode>(support.upper), true);
    if (result.pure_gas || result.pure_carrier)
    {
        result.gas.offset = result.pure_gas ? hi : lo;
        return result;
    }
    if (hi < lo)
        throw GeometryError(GeometryErrorCode::InvalidTopology, "Finite PLIC support bracket is inverted.");
    if (hi == lo)
        throw GeometryError(GeometryErrorCode::UnresolvedArithmetic, "Finite PLIC support bracket is unresolved.");
    const long double requested = gas ? sample.gas_volume : sample.carrier_volume;
    const long double allowed = fit_allowance<Mode>(requested, options.volume_tolerance);
    const auto bulk = sample.geometry->volume();
    const double fraction = static_cast<double>(requested / bulk.value);
    double offset = gas ? lo + (hi - lo) * std::cbrt(fraction) : hi - (hi - lo) * std::cbrt(fraction);
    if (!(offset > lo && offset < hi))
        offset = std::midpoint(lo, hi);
    long double best_bound = infinity;
    bool have_best = false, arithmetic_stalled = false;
    unsigned offset_iterations = 0;
    for (unsigned iteration = 0; iteration < options.maximum_offset_iterations; ++iteration)
    {
        ++offset_iterations;
        if (evaluations++ >= options.maximum_clip_evaluations)
            throw GeometryError(GeometryErrorCode::ResourceLimit, "Finite PLIC clip evaluation budget exhausted.");
        auto plane = result.gas;
        plane.offset = offset;
        const auto phases = sample.geometry->phases(plane);
        const auto measured = gas ? phases.gas : phases.carrier;
        const auto residual = difference<Mode>(measured, requested);
        const auto error = fit_error<Mode>(residual);
        if (error < best_bound)
        {
            have_best = true;
            best_bound = error;
            result.gas = plane;
            result.finite_target_volume_difference = published_difference<Mode>(residual);
            result.target_volume_residual = static_cast<double>(residual.value);
        }
        if (best_bound <= allowed)
            return result;
        // Update only when the whole enclosure determines the monotone side.
        // An unresolved comparison cannot be converted to an exact volume.
        if (lower<Mode>(residual) > 0)
        {
            if (gas)
                hi = offset;
            else
                lo = offset;
        }
        else if (upper<Mode>(residual) < 0)
        {
            if (gas)
                lo = offset;
            else
                hi = offset;
        }
        else
        {
            arithmetic_stalled = true;
            break;
        }
        if (std::nextafter(lo, hi) >= hi)
        {
            arithmetic_stalled = true;
            break;
        }
        offset = std::midpoint(lo, hi);
    }
    if (!have_best || best_bound > allowed)
    {
        std::ostringstream message;
        message.precision(std::numeric_limits<long double>::max_digits10);
        const bool exhausted = offset_iterations >= options.maximum_offset_iterations;
        message << (exhausted ? "Finite PLIC offset iteration budget exhausted before target-volume acceptance: target="
                              : "Finite PLIC arithmetic cannot establish the original target-volume tolerance: target=")
                << requested << ", residual=" << result.finite_target_volume_difference.value
                << ", radius=" << result.finite_target_volume_difference.absolute_error << ", bound=" << best_bound
                << ", allowed=" << allowed << ", bracket=[" << lo << "," << hi << "].";
        throw GeometryError(!exhausted && arithmetic_stalled ? GeometryErrorCode::UnresolvedArithmetic
                                                             : GeometryErrorCode::ResourceLimit,
            message.str());
    }
    return result;
}
} // namespace

PolyhedralResolutionRequired::PolyhedralResolutionRequired(PolyhedralResolutionAssessment assessment)
    : std::invalid_argument(resolution_message(assessment)), assessment_(assessment)
{
}

template<ArithmeticMode Mode>
void PolyhedralReconstruction<Mode>::validate(
    const PolyhedralPhaseSample<Mode>& sample, PolyhedralReconstructionOptions options)
{
    validate_sample<Mode>(sample, options);
}
template<ArithmeticMode Mode>
PolyhedralResolutionAssessment PolyhedralReconstruction<Mode>::assess_resolution(
    const PolyhedralPhaseSample<Mode>& target, std::span<const PolyhedralPhaseSample<Mode>> stencil,
    PolyhedralReconstructionOptions options)
{
    validate_sample<Mode>(target, options);
    if (stencil.size() > options.maximum_stencil_cells)
        throw std::invalid_argument("Finite PLIC physical stencil budget exhausted.");
    PolyhedralResolutionAssessment result;
    result.stencil_cells = stencil.size();
    for (const auto& span : target.geometry->axis_spans())
        result.local_cell_span = std::max(result.local_cell_span, outward(upper<Mode>(span), true));
    result.requested_cell_span = result.local_cell_span * .5;
    const auto examine = [&](const PolyhedralPhaseSample<Mode>& sample)
    {
        validate_sample<Mode>(sample, options);
        const long double total = static_cast<long double>(sample.gas_volume) + sample.carrier_volume;
        if (!(total > 0))
            throw std::invalid_argument("Finite PLIC phase support has zero inventory.");
        // Independent inventories supply the support fractions. No threshold
        // changes their stored amounts or converts a mixed target to a pure one.
        result.gas_support_fraction =
            std::max(result.gas_support_fraction, double(static_cast<long double>(sample.gas_volume) / total));
        result.carrier_support_fraction =
            std::max(result.carrier_support_fraction, double(static_cast<long double>(sample.carrier_volume) / total));
    };
    examine(target);
    if (target.gas_volume == 0 || target.carrier_volume == 0)
        return result;
    for (const auto& sample : stencil)
        examine(sample);
    if (stencil.size() < 4)
        result.reason = PolyhedralResolutionAssessment::Reason::InsufficientStencil;
    else if (result.gas_support_fraction < options.minimum_phase_support_fraction ||
             result.carrier_support_fraction < options.minimum_phase_support_fraction)
        result.reason = PolyhedralResolutionAssessment::Reason::InsufficientPhaseSupport;
    if (result.admitted())
    {
        const auto center = target.geometry->centroid();
        const auto volume = target.geometry->volume().as_double();
        const double alpha = target.gas_volume / volume;
        std::array<Vector, 3> matrix{};
        Vector rhs{};
        for (const auto& sample : stencil)
        {
            const auto d = minus(sample.geometry->centroid(), center);
            const auto d2 = dot(d, d);
            if (d2 == 0 || !std::isfinite(d2))
                throw std::invalid_argument(
                    "Finite PLIC stencil repeats the target center or exceeds arithmetic range.");
            const double delta = sample.gas_volume / sample.geometry->volume().as_double() - alpha;
            for (unsigned i = 0; i < 3; ++i)
            {
                rhs[i] -= d[i] * delta / d2;
                for (unsigned j = 0; j < 3; ++j)
                    matrix[i][j] += d[i] * d[j] / d2;
            }
        }
        double condition = 0;
        try
        {
            (void) unit(solve(matrix, rhs, options.maximum_condition, condition));
        }
        catch (const std::invalid_argument&)
        {
            result.reason = PolyhedralResolutionAssessment::Reason::InsufficientStencil;
        }
    }
    return result;
}
template<ArithmeticMode Mode>
PolyhedralInterface PolyhedralReconstruction<Mode>::solve_offset(
    const PolyhedralPhaseSample<Mode>& sample, Vector normal, PolyhedralReconstructionOptions options)
{
    validate_sample<Mode>(sample, options);
    std::size_t evaluations = 0;
    return offset_fit<Mode>(sample, normal, options, evaluations);
}
template<ArithmeticMode Mode>
PolyhedralInterface PolyhedralReconstruction<Mode>::reconstruct(const PolyhedralPhaseSample<Mode>& target,
    std::span<const PolyhedralPhaseSample<Mode>> stencil, PolyhedralReconstructionOptions options)
{
    auto resolution = assess_resolution(target, stencil, options);
    if (!resolution.admitted())
        throw PolyhedralResolutionRequired(resolution);
    if (target.gas_volume == 0 || target.carrier_volume == 0)
    {
        auto result = solve_offset(target, {1, 0, 0}, options);
        result.resolution = resolution;
        return result;
    }
    const auto center = target.geometry->centroid();
    const double volume = target.geometry->volume().as_double();
    const double alpha = target.gas_volume / volume;
    std::array<Vector, 3> matrix{};
    Vector rhs{};
    std::vector<double> volumes;
    volumes.reserve(stencil.size());
    for (const auto& sample : stencil)
    {
        const auto d = minus(sample.geometry->centroid(), center);
        const double d2 = dot(d, d);
        if (d2 == 0 || !std::isfinite(d2))
            throw std::invalid_argument("Finite PLIC stencil repeats the target center or exceeds arithmetic range.");
        const double v = sample.geometry->volume().as_double();
        volumes.push_back(v);
        const double delta = sample.gas_volume / v - alpha;
        for (unsigned i = 0; i < 3; ++i)
        {
            rhs[i] -= d[i] * delta / d2;
            for (unsigned j = 0; j < 3; ++j)
                matrix[i][j] += d[i] * d[j] / d2;
        }
    }
    double condition = 0;
    Vector normal;
    try
    {
        normal = unit(solve(matrix, rhs, options.maximum_condition, condition));
    }
    catch (const std::invalid_argument&)
    {
        resolution.reason = PolyhedralResolutionAssessment::Reason::InsufficientStencil;
        throw PolyhedralResolutionRequired(resolution);
    }
    struct Evaluation
    {
        PolyhedralInterface plane;
        std::vector<double> residual;
        double objective = 0;
        long double objective_upper = 0;
    };
    std::size_t evaluations = 0;
    const auto evaluate = [&](Vector n)
    {
        Evaluation value;
        value.plane = offset_fit<Mode>(target, unit(n), options, evaluations);
        value.residual.reserve(stencil.size());
        for (std::size_t i = 0; i < stencil.size(); ++i)
        {
            const auto& sample = stencil[i];
            const bool gas = sample.gas_volume <= sample.carrier_volume;
            if (evaluations++ >= options.maximum_clip_evaluations)
                throw GeometryError(GeometryErrorCode::ResourceLimit, "Finite PLIC clip evaluation budget exhausted.");
            const auto phases = sample.geometry->phases(value.plane.gas);
            const auto residual =
                difference<Mode>(gas ? phases.gas : phases.carrier, gas ? sample.gas_volume : sample.carrier_volume);
            const double normalized = static_cast<double>(residual.value / volumes[i]);
            value.residual.push_back(normalized);
            value.objective += normalized * normalized;
            if constexpr (Mode == ArithmeticMode::Ordinary)
            {
                const auto magnitude = std::abs(residual.value) / volumes[i];
                value.objective_upper += magnitude * magnitude;
            }
            else
            {
                const auto upper = certified_upward((std::abs(residual.value) + residual.absolute_error) / volumes[i]);
                value.objective_upper = certified_upward(value.objective_upper + certified_upward(upper * upper));
            }
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
        angular_detail::Qualification qualification;
        const auto status = angular_detail::advance(
            normal, current, options.maximum_condition, options.fit_tolerance, options.orientation_tolerance, evaluate,
            [](Vector value) { return unit(value); }, qualification);
        condition = std::max(condition, qualification.condition);
        last_step = qualification.step;
        if (status == angular_detail::IterationResult::Converged)
        {
            converged = true;
            break;
        }
        if (status == angular_detail::IterationResult::Stalled)
            throw std::invalid_argument("PLIC angular volume optimizer did not reach stationarity.");
    }
    current.plane.stencil_volume_residual = static_cast<double>(std::sqrt(current.objective_upper / stencil.size()));
    current.plane.stencil_condition = condition;
    current.plane.orientation_step = last_step;
    current.plane.fit_iterations = iterations;
    current.plane.fit_converged = converged;
    if (!converged)
        throw GeometryError(
            GeometryErrorCode::ResourceLimit, "PLIC angular volume optimizer iteration budget exhausted.");
    if (current.plane.stencil_volume_residual > options.maximum_model_residual)
    {
        resolution.reason = PolyhedralResolutionAssessment::Reason::UnresolvedPlaneModel;
        throw PolyhedralResolutionRequired(resolution);
    }
    current.plane.resolution = resolution;
    return current.plane;
}
} // namespace SimpleFluid::PLIC
