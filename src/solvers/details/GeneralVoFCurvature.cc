/** @file GeneralVoFCurvature.cc
 * @brief Physical polygon centroids and gated local quadratic surface curvature.
 */
#include "solvers/details/GeneralVoFCurvature.hh"
#include "geometry/ExactPlanarFace.hh"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
namespace SimpleFluid::detail
{
namespace
{
using Point = std::array<long double, 3>;
Point point(const Meshes::GeometryPoint& p)
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
std::array<Point, 3> frame(const Meshes::GeometryPoint& normal)
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
std::optional<GeneralVoFCurvaturePatch> GeneralVoFCurvature::patch(
    std::uint64_t cell, const Meshes::ConvexCellGeometry& geometry, const VoF::PolyhedralInterface& plane)
{
    if (plane.pure_gas || plane.pure_carrier)
        return {};
    const auto section = geometry.clip(plane.gas).section;
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
    result.area = section.area.as_double();
    for (unsigned a = 0; a < 3; ++a)
    {
        result.centroid[a] = static_cast<double>(origin[a] + centroid[a] / total);
        result.normal[a] = static_cast<double>(basis[2][a]);
    }
    const auto spans = geometry.axis_spans();
    result.physical_spans = {spans[0].as_double(), spans[1].as_double(), spans[2].as_double()};
    result.cell_scale = std::min({spans[0].as_double(), spans[1].as_double(), spans[2].as_double()});
    long double surface = 0;
    for (std::size_t face = 0; face < geometry.num_faces(); ++face)
        surface += geometry.face_geometry(face)->area().value;
    result.cell_scale = std::min(result.cell_scale, static_cast<double>(2 * geometry.moments().volume.value / surface));
    require(std::isfinite(result.cell_scale) && result.cell_scale > 0,
        "General VoF curvature has no positive physical cell scale.");
    return result;
}
std::optional<GeneralVoFCurvaturePatch> GeneralVoFCurvature::patch_finite(
    std::uint64_t cell, const Meshes::BoundedConvexGeometry& geometry, const VoF::PolyhedralInterface& plane)
{
    if (plane.pure_gas || plane.pure_carrier)
        return {};
    const auto section = geometry.clip(plane.gas).section;
    require(section.vertices.size() >= 3, "A mixed general VoF cell has no physical PLIC polygon.");
    const auto basis = frame(plane.gas.normal);
    require(section.area.value > section.area.absolute_error,
        "Finite general VoF PLIC polygon area has unresolved positivity.");
    GeneralVoFCurvaturePatch result;
    result.cell = cell;
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
            for (unsigned a = 0; a < 3; ++a) area[a] += triangle[a] / 2;
        }
        surface += std::sqrt(dot(area, area));
    }
    result.cell_scale = std::min(result.cell_scale, static_cast<double>(2 * geometry.volume().value / surface));
    require(std::isfinite(result.cell_scale) && result.cell_scale > 0,
        "General VoF curvature has no positive physical cell scale.");
    return result;
}
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
    };
    std::vector<Sample> samples;
    std::set<std::uint64_t> seen;
    const auto append = [&](const GeneralVoFCurvaturePatch& p)
    {
        require(std::isfinite(p.area) && p.area > 0, "Invalid general VoF curvature patch area.");
        auto r = difference(point(p.centroid), origin);
        const auto distance_squared = dot(r, r) / (h * h);
        require(std::isfinite(distance_squared), "Nonfinite general VoF curvature patch position.");
        bool outside = distance_squared > options.support_radius * options.support_radius;
        if (physical_support)
        {
            // Tangential geometry controls reconstruction support; stability
            // thickness remains solely the fit's dimensional normalization.
            const long double rx = (options.support_radius / 3) * tangent_span[0];
            const long double ry = (options.support_radius / 3) * tangent_span[1];
            const auto sx = dot(r, basis[0]) / rx, sy = dot(r, basis[1]) / ry;
            outside = sx * sx + sy * sy > 1;
        }
        if (outside || dot(unit(point(p.normal)), basis[2]) < options.minimum_normal_cosine)
            return;
        require(seen.insert(p.cell).second, "Duplicate general VoF curvature support cell.");
        const auto x = dot(r, basis[0]) / h, y = dot(r, basis[1]) / h, z = dot(r, basis[2]) / h;
        samples.push_back({{1, x, y, .5 * x * x, x * y, .5 * y * y}, z,
            static_cast<long double>(p.area) / (h * h * (1 + distance_squared))});
    };
    append(target);
    for (const auto& p : support)
        if (p.cell != target.cell)
            append(p);
    require(
        samples.size() >= options.minimum_samples, "General VoF curvature has insufficient coherent physical patches.");
    std::array<std::array<long double, 13>, 6> matrix{};
    for (const auto& sample : samples)
        for (unsigned a = 0; a < 6; ++a)
        {
            matrix[a][6] += sample.weight * sample.row[a] * sample.value;
            for (unsigned b = 0; b < 6; ++b)
                matrix[a][b] += sample.weight * sample.row[a] * sample.row[b];
        }
    long double norm = 0;
    for (unsigned a = 0; a < 6; ++a)
    {
        long double sum = 0;
        for (unsigned b = 0; b < 6; ++b)
            sum += std::abs(matrix[a][b]);
        norm = std::max(norm, sum);
        matrix[a][7 + a] = 1;
    }
    for (unsigned a = 0; a < 6; ++a)
    {
        unsigned pivot = a;
        for (unsigned b = a + 1; b < 6; ++b)
            if (std::abs(matrix[b][a]) > std::abs(matrix[pivot][a]))
                pivot = b;
        require(std::isfinite(matrix[pivot][a]) &&
                    std::abs(matrix[pivot][a]) > 64 * std::numeric_limits<long double>::epsilon() * norm,
            "General VoF curvature surface stencil is rank deficient.");
        std::swap(matrix[a], matrix[pivot]);
        const auto diagonal = matrix[a][a];
        for (unsigned b = a; b < 13; ++b)
            matrix[a][b] /= diagonal;
        for (unsigned row = 0; row < 6; ++row)
            if (row != a)
            {
                const auto factor = matrix[row][a];
                for (unsigned b = a; b < 13; ++b)
                    matrix[row][b] -= factor * matrix[a][b];
            }
    }
    long double inverse_norm = 0;
    for (unsigned a = 0; a < 6; ++a)
    {
        long double sum = 0;
        for (unsigned b = 7; b < 13; ++b)
            sum += std::abs(matrix[a][b]);
        inverse_norm = std::max(inverse_norm, sum);
    }
    GeneralVoFCurvatureResult result;
    result.samples = samples.size();
    result.condition = static_cast<double>(norm * inverse_norm);
    require(std::isfinite(result.condition) && result.condition <= options.maximum_condition,
        "General VoF curvature surface stencil is ill-conditioned.");
    long double residual = 0, weight = 0;
    for (const auto& sample : samples)
    {
        long double predicted = 0;
        for (unsigned a = 0; a < 6; ++a)
            predicted += sample.row[a] * matrix[a][6];
        const auto defect = predicted - sample.value;
        residual += sample.weight * defect * defect;
        weight += sample.weight;
    }
    result.normalized_residual = static_cast<double>(std::sqrt(residual / weight));
    require(
        std::isfinite(result.normalized_residual) && result.normalized_residual <= options.maximum_normalized_residual,
        "General VoF curvature surface fit exceeds its physical residual gate.");
    const auto x = matrix[1][6], y = matrix[2][6], xx = matrix[3][6] / h, xy = matrix[4][6] / h, yy = matrix[5][6] / h;
    require(std::hypot(x, y) <= options.maximum_slope,
        "General VoF curvature fit disagrees with its reconstructed target normal.");
    result.curvature = static_cast<double>(
        -((1 + y * y) * xx - 2 * x * y * xy + (1 + x * x) * yy) / std::pow(1 + x * x + y * y, 1.5L));
    require(std::isfinite(result.curvature), "General VoF curvature is nonfinite.");
    return result;
}
} // namespace SimpleFluid::detail
