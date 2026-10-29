/** @file GeneralVoFCurvature.cc
 * @brief Physical polygon centroids and gated local quadratic surface curvature.
 */
#include "solvers/details/GeneralVoFCurvature.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
namespace SimpleFluid::detail
{
namespace
{
using Point = std::array<long double, 3>;
Point point(const SimpleFluid::PLIC::GeometryPoint& p)
{
    return {p[0], p[1], p[2]};
}
long double dot(const Point& a, const Point& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Point difference(const Point& a, const Point& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
Point cross(const Point& a, const Point& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Point unit(Point p)
{
    const auto scale = std::max({std::abs(p[0]), std::abs(p[1]), std::abs(p[2])});
    if (!(scale > 0) || !std::isfinite(scale))
        throw std::runtime_error("General VoF curvature has a zero normal.");
    for (auto& v : p)
        v /= scale;
    const auto length = std::sqrt(dot(p, p));
    if (!std::isfinite(length) || !(length > 0))
        throw std::runtime_error("General VoF curvature has a nonfinite normal.");
    for (auto& v : p)
        v /= length;
    return p;
}
std::array<Point, 3> frame(const SimpleFluid::PLIC::GeometryPoint& normal)
{
    auto n = unit(point(normal));
    unsigned axis = 0;
    for (unsigned a = 1; a < 3; ++a)
        if (std::abs(n[a]) < std::abs(n[axis]))
            axis = a;
    Point reference{};
    reference[axis] = 1;
    auto x = unit(cross(reference, n));
    return {x, cross(n, x), n};
}
void require(bool value, const char* message)
{
    if (!value)
        throw std::runtime_error(message);
}
} // namespace
template<>
std::optional<GeneralVoFCurvaturePatch> GeneralVoFCurvature::patch<ArithmeticMode::Exact>(std::uint64_t cell,
    const PLIC::ConvexCellGeometry<ArithmeticMode::Exact>& geometry, const PLIC::PolyhedralInterface& plane)
{
    if (plane.pure_gas || plane.pure_carrier)
        return {};
    const auto* source = &geometry;
    const auto section = source->clip(plane.gas).section;
    require(section.vertices.size() >= 3, "A mixed general VoF cell has no physical PLIC polygon.");
    auto vertices = section.vertices;
    const auto basis = frame(plane.gas.normal);
    Point origin{};
    for (const auto& vertex : vertices)
        for (unsigned a = 0; a < 3; ++a)
            origin[a] += vertex[a] / vertices.size();
    std::sort(vertices.begin(), vertices.end(),
        [&](const Point& a, const Point& b)
        {
            const auto left = difference(a, origin), right = difference(b, origin);
            return std::atan2(dot(left, basis[1]), dot(left, basis[0])) <
                   std::atan2(dot(right, basis[1]), dot(right, basis[0]));
        });
    Point centroid{};
    long double total = 0;
    for (std::size_t i = 0; i < vertices.size(); ++i)
    {
        const auto& a = vertices[i];
        const auto& b = vertices[(i + 1) % vertices.size()];
        const long double twice_area = std::abs(dot(cross(difference(a, origin), difference(b, origin)), basis[2]));
        total += twice_area;
        for (unsigned axis = 0; axis < 3; ++axis)
            centroid[axis] += twice_area * ((a[axis] - origin[axis]) + (b[axis] - origin[axis])) / 3;
    }
    require(std::isfinite(total) && total > 0, "General VoF PLIC polygon has no representable physical area.");
    GeneralVoFCurvaturePatch result;
    result.cell = cell;
    result.key = VoF::GeneralVoFSurfaceKey{VoF::GeneralVoFSurfaceKind::MixedCell, cell};
    result.area = section.area.as_double();
    for (unsigned a = 0; a < 3; ++a)
    {
        result.centroid[a] = static_cast<double>(origin[a] + centroid[a] / total);
        result.normal[a] = static_cast<double>(basis[2][a]);
    }
    const auto spans = source->axis_spans();
    result.physical_spans = {spans[0].as_double(), spans[1].as_double(), spans[2].as_double()};
    result.cell_scale = std::min({spans[0].as_double(), spans[1].as_double(), spans[2].as_double()});
    long double surface = 0;
    for (std::size_t face = 0; face < source->num_faces(); ++face)
        surface += source->face_geometry(face)->area().value;
    result.cell_scale = std::min(result.cell_scale, static_cast<double>(2 * source->moments().volume.value / surface));
    require(std::isfinite(result.cell_scale) && result.cell_scale > 0,
        "General VoF curvature has no positive physical cell scale.");
    return result;
}
template<ArithmeticMode Mode>
std::optional<GeneralVoFCurvaturePatch> GeneralVoFCurvature::patch(
    std::uint64_t cell, const PLIC::ConvexCellGeometry<Mode>& geometry, const PLIC::PolyhedralInterface& plane)
{
    if (plane.pure_gas || plane.pure_carrier)
        return {};
    const auto section = geometry.clip(plane.gas).section;
    require(section.vertices.size() >= 3, "A mixed general VoF cell has no physical PLIC polygon.");
    const auto basis = frame(plane.gas.normal);
    const auto area_error = [&]
    {
        if constexpr (Mode == ArithmeticMode::Ordinary)
            return 0.L;
        else
            return section.area.absolute_error;
    }();
    require(section.area.value > area_error, "Finite general VoF PLIC polygon area has unresolved positivity.");
    GeneralVoFCurvaturePatch result;
    result.cell = cell;
    result.key = VoF::GeneralVoFSurfaceKey{VoF::GeneralVoFSurfaceKind::MixedCell, cell};
    result.area = section.area.as_double();
    for (unsigned a = 0; a < 3; ++a)
    {
        // The bounded kernel already integrates the local first moment. Do
        // not replace its numerical construction by an unbounded replay here.
        result.centroid[a] = static_cast<double>(section.centroid.value[a]);
        result.normal[a] = static_cast<double>(basis[2][a]);
    }
    const auto spans = geometry.axis_spans();
    result.physical_spans = {spans[0].as_double(), spans[1].as_double(), spans[2].as_double()};
    result.cell_scale = std::min({spans[0].as_double(), spans[1].as_double(), spans[2].as_double()});
    long double surface = 0;
    const auto& points = geometry.points();
    for (const auto& face : geometry.faces())
    {
        Point area{};
        for (std::size_t i = 1; i + 1 < face.size(); ++i)
        {
            const auto triangle = cross(difference(points[face[i]].value, points[face[0]].value),
                difference(points[face[i + 1]].value, points[face[0]].value));
            for (unsigned a = 0; a < 3; ++a)
                area[a] += triangle[a] / 2;
        }
        surface += std::sqrt(dot(area, area));
    }
    result.cell_scale = std::min(result.cell_scale, static_cast<double>(2 * geometry.volume().value / surface));
    require(std::isfinite(result.cell_scale) && result.cell_scale > 0,
        "General VoF curvature has no positive physical cell scale.");
    return result;
}
template std::optional<GeneralVoFCurvaturePatch> GeneralVoFCurvature::patch<ArithmeticMode::Ordinary>(
    std::uint64_t, const PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>&, const PLIC::PolyhedralInterface&);
template std::optional<GeneralVoFCurvaturePatch> GeneralVoFCurvature::patch<ArithmeticMode::BoundedCertification>(
    std::uint64_t, const PLIC::ConvexCellGeometry<ArithmeticMode::BoundedCertification>&,
    const PLIC::PolyhedralInterface&);

GeneralVoFCurvatureResult GeneralVoFCurvature::fit(const GeneralVoFCurvaturePatch& target,
    std::span<const GeneralVoFCurvaturePatch> support, GeneralVoFCurvatureOptions options)
{
    require(options.minimum_samples >= 6 && std::isfinite(options.support_radius) && options.support_radius > 0 &&
                std::isfinite(options.minimum_normal_cosine) && options.minimum_normal_cosine > 0 &&
                options.minimum_normal_cosine <= 1 && std::isfinite(options.maximum_condition) &&
                options.maximum_condition >= 1 && std::isfinite(options.maximum_normalized_residual) &&
                options.maximum_normalized_residual >= 0 && std::isfinite(options.maximum_slope) &&
                options.maximum_slope > 0,
        "Invalid general VoF surface-fit controls.");
    require(std::isfinite(target.cell_scale) && target.cell_scale > 0, "Invalid target physical curvature scale.");
    const auto basis = frame(target.normal);
    const auto origin = point(target.centroid);
    const long double h = target.cell_scale;
    const auto extents = point(target.physical_spans);
    const bool physical_support = extents[0] != 0 || extents[1] != 0 || extents[2] != 0;
    std::array<long double, 2> tangent_span{};
    if (physical_support)
    {
        for (auto span : extents)
            require(std::isfinite(span) && span > 0, "Invalid curvature physical span.");
        for (unsigned tangent = 0; tangent < 2; ++tangent)
            for (unsigned axis = 0; axis < 3; ++axis)
                tangent_span[tangent] += std::abs(basis[tangent][axis]) * extents[axis];
    }
    struct Sample
    {
        std::array<long double, 6> row;
        long double value, weight;
        Point normal;
    };
    std::vector<Sample> samples;
    const auto identity = [](const GeneralVoFCurvaturePatch& patch)
    { return patch.key.value_or(VoF::GeneralVoFSurfaceKey{VoF::GeneralVoFSurfaceKind::MixedCell, patch.cell}); };
    std::set<VoF::GeneralVoFSurfaceKey> seen;
    std::size_t outside_count = 0, incoherent_count = 0;
    const auto append = [&](const GeneralVoFCurvaturePatch& p)
    {
        require(std::isfinite(p.area) && p.area > 0, "Invalid general VoF curvature patch area.");
        auto r = difference(point(p.centroid), origin);
        const auto distance_squared = dot(r, r) / (h * h);
        require(std::isfinite(distance_squared), "Nonfinite general VoF curvature patch position.");
        bool outside = distance_squared > options.support_radius * options.support_radius;
        auto weight_distance = distance_squared;
        if (physical_support)
        {
            // Tangential geometry controls reconstruction support; stability
            // thickness remains solely the fit's dimensional normalization.
            const long double rx = (options.support_radius / 3) * tangent_span[0];
            const long double ry = (options.support_radius / 3) * tangent_span[1];
            const auto sx = dot(r, basis[0]) / rx, sy = dot(r, basis[1]) / ry;
            outside = sx * sx + sy * sy > 1;
            weight_distance =
                std::pow(dot(r, basis[0]) / tangent_span[0], 2) + std::pow(dot(r, basis[1]) / tangent_span[1], 2);
        }
        const auto normal = unit(point(p.normal));
        if (outside)
        {
            ++outside_count;
            return;
        }
        if (dot(normal, basis[2]) < options.minimum_normal_cosine)
        {
            ++incoherent_count;
            return;
        }
        const auto x = dot(r, basis[0]) / h, y = dot(r, basis[1]) / h, z = dot(r, basis[2]) / h;
        samples.push_back({{1, x, y, .5 * x * x, x * y, .5 * y * y}, z,
            static_cast<long double>(p.area) / (h * h * (1 + weight_distance)),
            {dot(normal, basis[0]), dot(normal, basis[1]), dot(normal, basis[2])}});
    };
    append(target);
    for (const auto& p : support)
    {
        require(seen.insert(identity(p)).second, "Duplicate general VoF curvature support patch.");
        if (identity(p) == identity(target))
        {
            require(p.centroid == target.centroid && p.normal == target.normal && p.area == target.area &&
                        p.cell_scale == target.cell_scale && p.physical_spans == target.physical_spans,
                "General VoF curvature target identity has contradictory patch geometry.");
            continue;
        }
        append(p);
    }
    if (samples.size() < options.minimum_samples)
    {
        std::ostringstream message;
        message << "General VoF curvature has insufficient coherent physical patches: samples=" << samples.size()
                << " required=" << options.minimum_samples << " outside_radius=" << outside_count
                << " incoherent=" << incoherent_count << " supplied=" << support.size();
        throw std::runtime_error(message.str());
    }
    // Resolve each tangential direction independently before forming polynomial
    // columns. Stability thickness h can be orders of magnitude smaller than
    // the two physical surface lengths on anisotropic cells.
    std::array<long double, 2> coordinate_scale{};
    long double maximum_weight = 0;
    for (const auto& sample : samples)
    {
        coordinate_scale[0] = std::max(coordinate_scale[0], std::abs(sample.row[1]));
        coordinate_scale[1] = std::max(coordinate_scale[1], std::abs(sample.row[2]));
        maximum_weight = std::max(maximum_weight, sample.weight);
    }
    require(coordinate_scale[0] > 0 && coordinate_scale[1] > 0 && std::isfinite(maximum_weight) && maximum_weight > 0,
        "General VoF curvature surface stencil is rank deficient.");
    const std::array<long double, 6> column_scale{1, coordinate_scale[0], coordinate_scale[1],
        coordinate_scale[0] * coordinate_scale[0], coordinate_scale[0] * coordinate_scale[1],
        coordinate_scale[1] * coordinate_scale[1]};
    std::vector<std::array<long double, 7>> matrix(samples.size());
    for (std::size_t row = 0; row < samples.size(); ++row)
    {
        const auto weight = std::sqrt(samples[row].weight / maximum_weight);
        for (unsigned column = 0; column < 6; ++column)
            matrix[row][column] = weight * samples[row].row[column] / column_scale[column];
        matrix[row][6] = weight * samples[row].value;
    }
    // Column-pivoted Householder QR avoids squaring the stencil's condition in
    // the solve. All numerical decisions use the conditioned rectangular data.
    std::array<unsigned, 6> permutation{0, 1, 2, 3, 4, 5};
    long double first_pivot = 0;
    for (unsigned column = 0; column < 6; ++column)
    {
        unsigned pivot = column;
        long double pivot_norm = 0;
        for (unsigned candidate = column; candidate < 6; ++candidate)
        {
            long double norm = 0;
            for (std::size_t row = column; row < matrix.size(); ++row)
                norm = std::hypot(norm, matrix[row][candidate]);
            if (norm > pivot_norm)
            {
                pivot = candidate;
                pivot_norm = norm;
            }
        }
        if (column == 0)
            first_pivot = pivot_norm;
        require(
            std::isfinite(pivot_norm) && pivot_norm > 64 * std::numeric_limits<long double>::epsilon() * first_pivot,
            "General VoF curvature surface stencil is rank deficient.");
        if (pivot != column)
        {
            for (auto& row : matrix)
                std::swap(row[column], row[pivot]);
            std::swap(permutation[column], permutation[pivot]);
        }
        const auto diagonal = -std::copysign(pivot_norm, matrix[column][column]);
        std::vector<long double> reflector(matrix.size() - column);
        for (std::size_t row = column; row < matrix.size(); ++row)
            reflector[row - column] = matrix[row][column] / pivot_norm;
        reflector[0] -= diagonal / pivot_norm;
        long double denominator = 0;
        for (auto value : reflector)
            denominator += value * value;
        for (unsigned other = column + 1; other < 7; ++other)
        {
            long double projection = 0;
            for (std::size_t row = column; row < matrix.size(); ++row)
                projection += reflector[row - column] * matrix[row][other];
            projection *= 2 / denominator;
            for (std::size_t row = column; row < matrix.size(); ++row)
                matrix[row][other] -= projection * reflector[row - column];
        }
        matrix[column][column] = diagonal;
        for (std::size_t row = column + 1; row < matrix.size(); ++row)
            matrix[row][column] = 0;
    }
    std::array<long double, 6> pivoted{}, coefficients{};
    std::array<std::array<long double, 6>, 6> inverse{};
    for (int row = 5; row >= 0; --row)
    {
        long double value = matrix[row][6];
        for (unsigned column = row + 1; column < 6; ++column)
            value -= matrix[row][column] * pivoted[column];
        pivoted[row] = value / matrix[row][row];
        for (unsigned rhs = 0; rhs < 6; ++rhs)
        {
            value = row == static_cast<int>(rhs) ? 1 : 0;
            for (unsigned column = row + 1; column < 6; ++column)
                value -= matrix[row][column] * inverse[column][rhs];
            inverse[row][rhs] = value / matrix[row][row];
        }
    }
    long double norm = 0, inverse_norm = 0;
    for (unsigned row = 0; row < 6; ++row)
    {
        coefficients[permutation[row]] = pivoted[row] / column_scale[permutation[row]];
        long double sum = 0, inverse_sum = 0;
        for (unsigned column = 0; column < 6; ++column)
        {
            sum += std::abs(matrix[row][column]);
            inverse_sum += std::abs(inverse[row][column]);
        }
        norm = std::max(norm, sum);
        inverse_norm = std::max(inverse_norm, inverse_sum);
    }
    GeneralVoFCurvatureResult result;
    result.samples = samples.size();
    result.condition = static_cast<double>(std::pow(norm * inverse_norm, 2));
    require(std::isfinite(result.condition) && result.condition <= options.maximum_condition,
        "General VoF curvature surface stencil is ill-conditioned.");
    long double residual = 0, weight = 0;
    for (const auto& sample : samples)
    {
        long double predicted = 0;
        for (unsigned a = 0; a < 6; ++a)
            predicted += sample.row[a] * coefficients[a];
        const auto defect = predicted - sample.value;
        residual += sample.weight * defect * defect;
        weight += sample.weight;
    }
    result.normalized_residual = static_cast<double>(std::sqrt(residual / weight));
    if (!std::isfinite(result.normalized_residual) || result.normalized_residual > options.maximum_normalized_residual)
    {
        std::ostringstream message;
        message << "General VoF curvature surface fit exceeds its physical residual gate: residual="
                << result.normalized_residual << " limit=" << options.maximum_normalized_residual
                << " samples=" << result.samples << " condition=" << result.condition;
        throw std::runtime_error(message.str());
    }
    const auto x = coefficients[1], y = coefficients[2], xx = coefficients[3] / h, xy = coefficients[4] / h,
               yy = coefficients[5] / h;
    require(std::hypot(x, y) <= options.maximum_slope,
        "General VoF curvature fit disagrees with its reconstructed target normal.");
    for (const auto& sample : samples)
    {
        const auto slope_x = coefficients[1] + coefficients[3] * sample.row[1] + coefficients[4] * sample.row[2];
        const auto slope_y = coefficients[2] + coefficients[4] * sample.row[1] + coefficients[5] * sample.row[2];
        const auto fitted_normal = unit(Point{-slope_x, -slope_y, 1});
        require(dot(fitted_normal, sample.normal) >= options.minimum_normal_cosine,
            "General VoF curvature surface fit contradicts its patch normals.");
    }
    result.curvature = static_cast<double>(
        -((1 + y * y) * xx - 2 * x * y * xy + (1 + x * x) * yy) / std::pow(1 + x * x + y * y, 1.5L));
    require(std::isfinite(result.curvature), "General VoF curvature is nonfinite.");
    return result;
}
} // namespace SimpleFluid::detail
