/** @file BoundedPolyhedralReconstruction.inc
 * @brief Bounded floating-point PLIC implementation.
 * Included only by FinitePolyhedralReconstruction.cc at file scope.
 */
namespace SimpleFluid::VoF
{
namespace
{
using Point = Meshes::GeometryPoint;
using Value = Meshes::BoundedGeometryValue;
constexpr long double infinity = std::numeric_limits<long double>::infinity();

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
        throw std::invalid_argument("Finite PLIC normal is zero or nonfinite.");
    for (auto& x : a)
        x /= norm;
    return a;
}
long double upward(long double x)
{
    if (!std::isfinite(x) || x < 0)
        throw std::invalid_argument("Finite PLIC arithmetic bound is unrepresentable.");
    return x == 0 ? 0 : std::nextafter(x, infinity);
}
Value difference(Value a, long double b)
{
    const auto value = a.value - b;
    // Error-free TwoSum for the represented subtraction; original geometry
    // uncertainty remains separate and is never inferred from its rounded sign.
    const auto z = value - a.value;
    const auto residual = (a.value - (value - z)) + (-b - z);
    return {value, upward(a.absolute_error + std::abs(residual))};
}
double outward(long double x, bool upper)
{
    double result = static_cast<double>(x);
    if (upper ? static_cast<long double>(result) < x : static_cast<long double>(result) > x)
        result = std::nextafter(result, upper ? std::numeric_limits<double>::infinity()
                                             : -std::numeric_limits<double>::infinity());
    if (!std::isfinite(result))
        throw std::invalid_argument("Finite PLIC support cannot be represented.");
    return result;
}
void validate_sample(const FinitePolyhedralPhaseSample& sample, const PolyhedralReconstructionOptions& options)
{
    if (!Meshes::bounded_geometry_environment())
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
    if (!(volume.lower() > 0))
        throw std::invalid_argument("Finite PLIC cell volume is not established positive.");
    const auto total = static_cast<long double>(sample.gas_volume) + sample.carrier_volume;
    const auto residual = difference(volume, total);
    const auto bound = upward(volume.absolute_error +
        8 * std::numeric_limits<double>::epsilon() * std::abs(volume.value));
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
PolyhedralInterface offset_fit(const FinitePolyhedralPhaseSample& sample, Point normal,
    PolyhedralReconstructionOptions options, std::size_t& evaluations)
{
    PolyhedralInterface result;
    result.finite_geometry = true;
    result.pure_gas = sample.carrier_volume == 0;
    result.pure_carrier = sample.gas_volume == 0;
    result.gas.normal = unit(normal);
    result.gas.anchor = sample.geometry->origin();
    const auto support = sample.geometry->support(result.gas.normal, result.gas.anchor);
    double lo = outward(support.lower.lower(), false);
    double hi = outward(support.upper.upper(), true);
    if (result.pure_gas || result.pure_carrier)
    {
        result.gas.offset = result.pure_gas ? hi : lo;
        return result;
    }
    if (!(hi > lo))
        throw std::invalid_argument("Finite PLIC support bracket is unresolved.");
    const bool gas = sample.gas_volume <= sample.carrier_volume;
    const long double requested = gas ? sample.gas_volume : sample.carrier_volume;
    const long double allowed = requested * options.volume_tolerance;
    const auto bulk = sample.geometry->volume();
    const double fraction = static_cast<double>(requested / bulk.value);
    double offset = gas ? lo + (hi - lo) * std::cbrt(fraction) : hi - (hi - lo) * std::cbrt(fraction);
    if (!(offset > lo && offset < hi))
        offset = std::midpoint(lo, hi);
    long double best_bound = infinity;
    bool have_best = false;
    for (unsigned iteration = 0; iteration < options.maximum_offset_iterations; ++iteration)
    {
        if (evaluations++ >= options.maximum_clip_evaluations)
            throw std::invalid_argument("Finite PLIC clip evaluation budget exhausted.");
        auto plane = result.gas;
        plane.offset = offset;
        const auto phases = sample.geometry->phases(plane);
        const auto measured = gas ? phases.gas : phases.carrier;
        const auto residual = difference(measured, requested);
        const auto error = upward(std::abs(residual.value) + residual.absolute_error);
        if (error < best_bound)
        {
            have_best = true;
            best_bound = error;
            result.gas = plane;
            result.finite_target_volume_difference = residual;
            result.target_volume_residual = static_cast<double>(residual.value);
        }
        if (best_bound <= allowed)
            return result;
        // Update only when the whole enclosure determines the monotone side.
        // An unresolved comparison cannot be converted to an exact volume.
        if (residual.lower() > 0)
        {
            if (gas)
                hi = offset;
            else
                lo = offset;
        }
        else if (residual.upper() < 0)
        {
            if (gas)
                lo = offset;
            else
                hi = offset;
        }
        else
            break;
        if (std::nextafter(lo, hi) >= hi)
            break;
        offset = std::midpoint(lo, hi);
    }
    if (!have_best || best_bound > allowed)
        throw std::invalid_argument("Finite PLIC arithmetic cannot establish the original target-volume tolerance.");
    return result;
}
} // namespace

PolyhedralResolutionRequired::PolyhedralResolutionRequired(PolyhedralResolutionAssessment assessment)
    : std::invalid_argument(resolution_message(assessment)), assessment_(assessment)
{
}

void PolyhedralReconstruction::validate(
    const FinitePolyhedralPhaseSample& sample, PolyhedralReconstructionOptions options)
{
    validate_sample(sample, options);
}
PolyhedralResolutionAssessment PolyhedralReconstruction::assess_resolution(
    const FinitePolyhedralPhaseSample& target, std::span<const FinitePolyhedralPhaseSample> stencil,
    PolyhedralReconstructionOptions options)
{
    validate_sample(target, options);
    if (stencil.size() > options.maximum_stencil_cells)
        throw std::invalid_argument("Finite PLIC physical stencil budget exhausted.");
    PolyhedralResolutionAssessment result;
    result.stencil_cells = stencil.size();
    for (const auto& span : target.geometry->axis_spans())
        result.local_cell_span = std::max(result.local_cell_span, outward(span.upper(), true));
    result.requested_cell_span = result.local_cell_span * .5;
    const auto examine = [&](const FinitePolyhedralPhaseSample& sample)
    {
        validate_sample(sample, options);
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
        std::array<Point, 3> matrix{};
        Point rhs{};
        for (const auto& sample : stencil)
        {
            const auto d = minus(sample.geometry->centroid(), center);
            const auto d2 = dot(d, d);
            if (d2 == 0 || !std::isfinite(d2))
                throw std::invalid_argument("Finite PLIC stencil repeats the target center or exceeds arithmetic range.");
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
PolyhedralInterface PolyhedralReconstruction::solve_offset(
    const FinitePolyhedralPhaseSample& sample, Point normal, PolyhedralReconstructionOptions options)
{
    validate_sample(sample, options);
    std::size_t evaluations = 0;
    return offset_fit(sample, normal, options, evaluations);
}
PolyhedralInterface PolyhedralReconstruction::reconstruct(const FinitePolyhedralPhaseSample& target,
    std::span<const FinitePolyhedralPhaseSample> stencil, PolyhedralReconstructionOptions options)
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
    std::array<Point, 3> matrix{};
    Point rhs{};
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
    Point normal;
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
    const auto evaluate = [&](Point n)
    {
        Evaluation value;
        value.plane = offset_fit(target, unit(n), options, evaluations);
        value.residual.reserve(stencil.size());
        for (std::size_t i = 0; i < stencil.size(); ++i)
        {
            const auto& sample = stencil[i];
            const bool gas = sample.gas_volume <= sample.carrier_volume;
            if (evaluations++ >= options.maximum_clip_evaluations)
                throw std::invalid_argument("Finite PLIC clip evaluation budget exhausted.");
            const auto phases = sample.geometry->phases(value.plane.gas);
            const auto residual = difference(gas ? phases.gas : phases.carrier,
                gas ? sample.gas_volume : sample.carrier_volume);
            const double normalized = static_cast<double>(residual.value / volumes[i]);
            value.residual.push_back(normalized);
            value.objective += normalized * normalized;
            const auto upper = upward((std::abs(residual.value) + residual.absolute_error) / volumes[i]);
            value.objective_upper = upward(value.objective_upper + upward(upper * upper));
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
        if (std::sqrt(current.objective_upper / stencil.size()) <= options.fit_tolerance ||
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
    current.plane.stencil_volume_residual =
        static_cast<double>(std::sqrt(current.objective_upper / stencil.size()));
    current.plane.stencil_condition = condition;
    current.plane.orientation_step = last_step;
    current.plane.fit_iterations = iterations;
    current.plane.fit_converged = converged;
    if (!converged)
        throw std::invalid_argument("PLIC angular volume optimizer iteration budget exhausted.");
    if (current.plane.stencil_volume_residual > options.maximum_model_residual)
    {
        resolution.reason = PolyhedralResolutionAssessment::Reason::UnresolvedPlaneModel;
        throw PolyhedralResolutionRequired(resolution);
    }
    current.plane.resolution = resolution;
    return current.plane;
}
} // namespace SimpleFluid::VoF
