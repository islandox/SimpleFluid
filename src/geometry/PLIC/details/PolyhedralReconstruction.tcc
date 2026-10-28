/** @file PolyhedralReconstruction.tcc
 * @brief Resolution admission and mode-selected finite PLIC reconstruction.
 */
#include "geometry/PLIC/PlaneMeasure.hh"
#include "geometry/PLIC/PolyhedralReconstruction.hh"
#include "geometry/PLIC/details/AngularFit.hh"
#include "geometry/PLIC/details/PhaseArithmetic.hh"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <vector>

namespace SimpleFluid::PLIC
{
namespace
{
using Vector = SimpleFluid::PLIC::GeometryPoint;
template<ArithmeticMode Mode> using Value = PLIC::GeometryValue<Mode>;
constexpr long double infinity = std::numeric_limits<long double>::infinity();

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
        !(options.minimum_phase_support_fraction > 0 && options.minimum_phase_support_fraction <= .5) ||
        !std::isfinite(options.minimum_optimized_phase_length_ratio) ||
        !(options.minimum_optimized_phase_length_ratio >= 0 && options.minimum_optimized_phase_length_ratio <= 1))
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
        case PolyhedralResolutionAssessment::Reason::RankDeficientStencil:
            message << ": centroid directions do not span three dimensions";
            break;
        case PolyhedralResolutionAssessment::Reason::IllConditionedStencil:
            message << ": physical normal support is ill-conditioned";
            break;
        case PolyhedralResolutionAssessment::Reason::NoDirectionalSignal:
            message << ": phase differences do not establish a meaningful direction";
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
struct DirectionAnalysis
{
    PolyhedralResolutionAssessment resolution;
    Vector normal{};
    std::vector<double> volumes;
};
long double roundoff_factor(std::size_t operations)
{
    const auto product = operations * static_cast<long double>(std::numeric_limits<double>::epsilon());
    return product / (1 - product);
}
template<ArithmeticMode Mode>
DirectionAnalysis analyze_direction(const PolyhedralPhaseSample<Mode>& target,
    std::span<const PolyhedralPhaseSample<Mode>> stencil, const PolyhedralReconstructionOptions& options)
{
    validate_sample<Mode>(target, options);
    if (stencil.size() > options.maximum_stencil_cells)
        throw std::invalid_argument("Finite PLIC physical stencil budget exhausted.");
    DirectionAnalysis analysis;
    auto& result = analysis.resolution;
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
        result.gas_support_fraction =
            std::max(result.gas_support_fraction, double(static_cast<long double>(sample.gas_volume) / total));
        result.carrier_support_fraction =
            std::max(result.carrier_support_fraction, double(static_cast<long double>(sample.carrier_volume) / total));
    };
    examine(target);
    if (target.gas_volume == 0 || target.carrier_volume == 0)
        return analysis;
    for (const auto& sample : stencil)
        examine(sample);
    if (stencil.size() < 4)
    {
        result.reason = PolyhedralResolutionAssessment::Reason::InsufficientStencil;
        return analysis;
    }
    // Select one independent phase channel throughout the solve. In particular,
    // a carrier minority is never recovered by subtracting gas fractions near 1.
    const bool gas = target.gas_volume <= target.carrier_volume;
    const auto fraction = [&](const PolyhedralPhaseSample<Mode>& sample)
    {
        const long double total = static_cast<long double>(sample.gas_volume) + sample.carrier_volume;
        return static_cast<long double>(gas ? sample.gas_volume : sample.carrier_volume) / total;
    };
    const auto alpha = fraction(target);
    const auto center = target.geometry->centroid();
    using LongVector = std::array<long double, 3>;
    std::array<LongVector, 3> assembled{}, matrix_magnitude{}, matrix_error{};
    LongVector assembled_rhs{}, rhs_magnitude{}, rhs_error{};
    analysis.volumes.reserve(stencil.size());
    const auto assembly_roundoff = roundoff_factor(8 + stencil.size());
    const auto fraction_roundoff = roundoff_factor(3);
    for (const auto& sample : stencil)
    {
        const auto sample_center = sample.geometry->centroid();
        LongVector d{};
        long double d2 = 0;
        for (unsigned i = 0; i < 3; ++i)
        {
            d[i] = (static_cast<long double>(sample_center[i]) - center[i]) / result.local_cell_span;
            d2 += d[i] * d[i];
        }
        if (d2 == 0 || !std::isfinite(d2))
            throw std::invalid_argument("Finite PLIC stencil repeats the target center or exceeds arithmetic range.");
        analysis.volumes.push_back(sample.geometry->volume().as_double());
        const auto sample_fraction = fraction(sample);
        const auto delta = sample_fraction - alpha;
        // Relative input/ratio roundoff scales with the chosen phase itself,
        // so a resolvable dilute signal has no arbitrary absolute epsilon floor.
        const auto delta_error = fraction_roundoff * (std::abs(sample_fraction) + std::abs(alpha)) +
                                 std::numeric_limits<long double>::epsilon() * std::abs(delta);
        for (unsigned i = 0; i < 3; ++i)
        {
            const auto coefficient = d[i] / d2;
            const auto term = (gas ? -1 : 1) * coefficient * delta;
            assembled_rhs[i] += term;
            rhs_magnitude[i] += std::abs(term);
            rhs_error[i] += std::abs(coefficient) * delta_error;
            for (unsigned j = 0; j < 3; ++j)
            {
                const auto entry = d[i] * d[j] / d2;
                assembled[i][j] += entry;
                matrix_magnitude[i][j] += std::abs(entry);
            }
        }
    }
    std::array<Vector, 3> matrix{}, inverse{};
    Vector rhs{};
    long double matrix_norm = 0, matrix_error_norm = 0;
    for (unsigned i = 0; i < 3; ++i)
    {
        rhs[i] = static_cast<double>(assembled_rhs[i]);
        rhs_error[i] += assembly_roundoff * rhs_magnitude[i] + std::abs(assembled_rhs[i] - rhs[i]);
        long double row_norm = 0, row_error = 0;
        for (unsigned j = 0; j < 3; ++j)
        {
            matrix[i][j] = static_cast<double>(assembled[i][j]);
            matrix_error[i][j] = assembly_roundoff * matrix_magnitude[i][j] + std::abs(assembled[i][j] - matrix[i][j]);
            row_norm += std::abs(matrix[i][j]);
            row_error += matrix_error[i][j];
        }
        matrix_norm = std::max(matrix_norm, row_norm);
        matrix_error_norm = std::max(matrix_error_norm, row_error);
    }
    double pivot_condition = 0;
    Vector gradient{};
    try
    {
        gradient = solve(matrix, rhs, std::numeric_limits<double>::infinity(), pivot_condition);
        for (unsigned j = 0; j < 3; ++j)
        {
            Vector basis{};
            basis[j] = 1;
            const auto column = solve(matrix, basis, std::numeric_limits<double>::infinity(), pivot_condition);
            for (unsigned i = 0; i < 3; ++i)
                inverse[i][j] = column[i];
        }
    }
    catch (const std::invalid_argument&)
    {
        result.reason = PolyhedralResolutionAssessment::Reason::RankDeficientStencil;
        return analysis;
    }
    long double inverse_norm = 0;
    for (const auto& row : inverse)
    {
        long double row_norm = 0;
        for (const auto entry : row)
            row_norm += std::abs(entry);
        inverse_norm = std::max(inverse_norm, row_norm);
    }
    const auto condition = matrix_norm * inverse_norm;
    if (!std::isfinite(condition) || condition > options.maximum_condition)
    {
        result.stencil_condition =
            static_cast<double>(std::min(condition, static_cast<long double>(std::numeric_limits<double>::max())));
        result.reason = PolyhedralResolutionAssessment::Reason::IllConditionedStencil;
        return analysis;
    }
    result.stencil_condition = static_cast<double>(condition);
    LongVector perturbation{}, gradient_error{};
    for (unsigned i = 0; i < 3; ++i)
    {
        long double residual = rhs[i];
        perturbation[i] = rhs_error[i];
        for (unsigned j = 0; j < 3; ++j)
        {
            residual -= static_cast<long double>(matrix[i][j]) * gradient[j];
            perturbation[i] += matrix_error[i][j] * std::abs(gradient[j]);
        }
        perturbation[i] += std::abs(residual);
    }
    const auto denominator = 1 - inverse_norm * matrix_error_norm;
    if (!(denominator > 0))
    {
        result.reason = PolyhedralResolutionAssessment::Reason::IllConditionedStencil;
        return analysis;
    }
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j)
            gradient_error[i] += std::abs(inverse[i][j]) * perturbation[j];
    const auto strength = std::hypot(gradient[0], gradient[1], gradient[2]);
    // The perturbation denominator uses infinity norms. Convert that uniform
    // component bound to a Euclidean bound before comparing with the strength.
    const auto uncertainty =
        std::sqrt(3.L) * *std::max_element(gradient_error.begin(), gradient_error.end()) / denominator;
    if (!std::isfinite(strength) || !std::isfinite(uncertainty) || uncertainty > std::numeric_limits<double>::max())
        throw std::invalid_argument("Finite PLIC directional arithmetic exceeds representable range.");
    result.directional_strength = strength;
    result.directional_uncertainty = static_cast<double>(uncertainty);
    if (!(strength > uncertainty))
    {
        result.reason = PolyhedralResolutionAssessment::Reason::NoDirectionalSignal;
        return analysis;
    }
    analysis.normal = unit(gradient);
    result.angular_fit_eligible = result.gas_support_fraction >= options.minimum_phase_support_fraction &&
                                  result.carrier_support_fraction >= options.minimum_phase_support_fraction;
    return analysis;
}
template<ArithmeticMode Mode>
PolyhedralInterface offset_fit(const PolyhedralPhaseSample<Mode>& sample, Vector normal,
    PolyhedralReconstructionOptions options, std::size_t& evaluations)
{
    std::optional<OrdinaryPlaneMeasure> ordinary_measure;
    if constexpr (Mode == ArithmeticMode::Ordinary)
        if (options.measure_only)
            ordinary_measure.emplace(*sample.geometry);
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
        long double section_area = 0, numerical_error = 0;
        const auto measured = [&]() -> Value<Mode>
        {
            if constexpr (Mode == ArithmeticMode::Ordinary)
            {
                if (ordinary_measure)
                {
                    auto minority = plane;
                    if (!gas)
                    {
                        for (auto& component : minority.normal)
                            component = -component;
                        minority.offset = -minority.offset;
                    }
                    const auto volume = ordinary_measure->below(minority, true);
                    section_area = volume.section_area;
                    numerical_error = volume.numerical_error_estimate;
                    return {volume.volume};
                }
                const auto phases = sample.geometry->phases(plane);
                return gas ? phases.gas : phases.carrier;
            }
            else
            {
                const auto phases = sample.geometry->phases(plane);
                return gas ? phases.gas : phases.carrier;
            }
        }();
        const auto residual = difference<Mode>(measured, requested);
        const auto error = fit_error<Mode>(residual) + numerical_error;
        if (error < best_bound)
        {
            have_best = true;
            best_bound = error;
            result.gas = plane;
            result.finite_target_volume_difference = published_difference<Mode>(residual);
            result.target_volume_residual = static_cast<double>(residual.value);
            result.target_volume_numerical_error_estimate = numerical_error;
        }
        if (best_bound <= allowed)
            return result;
        if constexpr (Mode == ArithmeticMode::Ordinary)
            if (numerical_error > allowed && std::abs(residual.value) <= numerical_error)
            {
                arithmetic_stalled = true;
                break;
            }
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
        double next = std::midpoint(lo, hi);
        if constexpr (Mode == ArithmeticMode::Ordinary)
        {
            // A Newton proposal is optional acceleration. Every accepted trial
            // remains strictly inside the updated bracket; periodic bisection
            // guarantees progress even for very small sections.
            if (section_area > 0 && iteration % 4 != 3)
            {
                const auto step = residual.value / section_area;
                const double candidate = static_cast<double>(offset + (gas ? -step : step));
                if (candidate > lo && candidate < hi && candidate != offset)
                    next = candidate;
            }
        }
        offset = next;
    }
    if (!have_best || best_bound > allowed)
    {
        std::ostringstream message;
        message.precision(std::numeric_limits<long double>::max_digits10);
        const bool exhausted = offset_iterations >= options.maximum_offset_iterations;
        message << (exhausted ? "Finite PLIC offset iteration budget exhausted before target-volume acceptance: target="
                              : "Finite PLIC arithmetic cannot establish the original target-volume tolerance: target=")
                << requested << ", residual=" << result.finite_target_volume_difference.value
                << ", radius=" << result.finite_target_volume_difference.absolute_error
                << ", numerical_error_estimate=" << result.target_volume_numerical_error_estimate
                << ", bound=" << best_bound << ", allowed=" << allowed << ", bracket=[" << lo << "," << hi << "].";
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
    return analyze_direction<Mode>(target, stencil, options).resolution;
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
    auto direction = analyze_direction<Mode>(target, stencil, options);
    auto resolution = direction.resolution;
    if (!resolution.admitted())
        throw PolyhedralResolutionRequired(resolution);
    if (target.gas_volume == 0 || target.carrier_volume == 0)
    {
        auto result = solve_offset(target, {1, 0, 0}, options);
        result.resolution = resolution;
        return result;
    }
    auto normal = direction.normal;
    auto& volumes = direction.volumes;
    double condition = resolution.stencil_condition;
    std::size_t evaluations = 0;
    const auto phase_length_ratio =
        std::cbrt(static_cast<long double>(std::min(target.gas_volume, target.carrier_volume))) /
        std::cbrt(target.geometry->volume().value);
    if (phase_length_ratio < options.minimum_optimized_phase_length_ratio)
    {
        // The stencil establishes the gradient and its support; tiny target
        // phases do not supply an angular volume-model objective. No neighbor
        // clipping, angular qualification or model-RMS gate runs on this path.
        auto result = offset_fit<Mode>(target, normal, options, evaluations);
        result.orientation_method = PolyhedralOrientationMethod::Gradient;
        result.stencil_condition = condition;
        result.resolution = resolution;
        return result;
    }
    std::vector<OrdinaryPlaneMeasure> ordinary_stencil;
    if constexpr (Mode == ArithmeticMode::Ordinary)
        if (options.measure_only)
        {
            ordinary_stencil.reserve(stencil.size());
            for (const auto& sample : stencil)
                ordinary_stencil.emplace_back(*sample.geometry);
        }
    struct Evaluation
    {
        PolyhedralInterface plane;
        std::vector<double> residual;
        double objective = 0;
        long double objective_upper = 0;
    };
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
            const auto measured = [&]() -> Value<Mode>
            {
                if constexpr (Mode == ArithmeticMode::Ordinary)
                {
                    if (options.measure_only)
                    {
                        auto plane = value.plane.gas;
                        if (!gas)
                        {
                            for (auto& component : plane.normal)
                                component = -component;
                            plane.offset = -plane.offset;
                        }
                        return {ordinary_stencil[i].below(plane).volume};
                    }
                    const auto phases = sample.geometry->phases(value.plane.gas);
                    return gas ? phases.gas : phases.carrier;
                }
                else
                {
                    const auto phases = sample.geometry->phases(value.plane.gas);
                    return gas ? phases.gas : phases.carrier;
                }
            }();
            const auto residual = difference<Mode>(measured, gas ? sample.gas_volume : sample.carrier_volume);
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
    if (!resolution.angular_fit_eligible)
    {
        current.plane.stencil_volume_residual =
            static_cast<double>(std::sqrt(current.objective_upper / stencil.size()));
        current.plane.stencil_volume_residual_measured = true;
        current.plane.stencil_condition = condition;
        current.plane.orientation_method = PolyhedralOrientationMethod::Gradient;
        if (current.plane.stencil_volume_residual > options.maximum_model_residual)
        {
            resolution.reason = PolyhedralResolutionAssessment::Reason::UnresolvedPlaneModel;
            throw PolyhedralResolutionRequired(resolution);
        }
        current.plane.resolution = resolution;
        return current.plane;
    }
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
    current.plane.stencil_volume_residual_measured = true;
    current.plane.stencil_condition = condition;
    current.plane.orientation_step = last_step;
    current.plane.fit_iterations = iterations;
    current.plane.orientation_method = PolyhedralOrientationMethod::AngularVolumeFit;
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
