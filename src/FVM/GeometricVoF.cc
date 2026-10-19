/**
 * @file GeometricVoF.cc
 * @brief Compiled Cartesian PLIC geometry and conservative split transport.
 */

#include "FVM/GeometricVoF.hh"
#include "FVM/details/VoFPhaseBudget.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::VoF
{
namespace
{
constexpr double fraction_roundoff = 2.0e-12;

std::size_t checked_product(std::size_t a, std::size_t b)
{
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("Cartesian grid index count overflows size_t.");
    return a * b;
}

void check_axis(std::size_t axis)
{
    if (axis >= 3)
        throw std::out_of_range("Cartesian grid axis must be in [0,2].");
}

void check_fraction(const UniformGrid& grid, const std::vector<double>& fraction)
{
    if (fraction.size() != grid.cell_count())
        throw std::invalid_argument("VoF fraction size does not match the grid.");
    for (const double value : fraction)
        if (!std::isfinite(value) || value < -fraction_roundoff || value > 1.0 + fraction_roundoff)
            throw std::invalid_argument("VoF fractions must be finite and in [0,1].");
}

std::size_t image_index(std::ptrdiff_t index, std::size_t count, bool periodic)
{
    const auto n = static_cast<std::ptrdiff_t>(count);
    if (periodic)
        return static_cast<std::size_t>((index % n + n) % n);
    // Mirror boundary cells: the implied scalar boundary condition is symmetry.
    const auto period = 2 * n;
    auto folded = (index % period + period) % period;
    if (folded >= n)
        folded = period - 1 - folded;
    return static_cast<std::size_t>(folded);
}

double sample(const UniformGrid& grid, const std::vector<double>& field, const std::array<std::ptrdiff_t, 3>& position)
{
    std::array<std::size_t, 3> p{};
    for (std::size_t a = 0; a < 3; ++a)
        p[a] = image_index(position[a], grid.cells[a], grid.periodic[a]);
    return field[grid.cell_index(p[0], p[1], p[2])];
}

Vector physical_gradient(const UniformGrid& grid, const std::vector<double>& field, std::size_t cell)
{
    const auto p = grid.cell_coordinates(cell);
    Vector gradient{};
    constexpr std::array<int, 3> weight{1, 2, 1};
    for (std::size_t a = 0; a < 3; ++a)
    {
        if (grid.cells[a] == 1)
            continue;
        const auto b = (a + 1) % 3;
        const auto c = (a + 2) % 3;
        long double difference = 0.0;
        for (int j = -1; j <= 1; ++j)
            for (int k = -1; k <= 1; ++k)
            {
                std::array<std::ptrdiff_t, 3> q{static_cast<std::ptrdiff_t>(p[0]), static_cast<std::ptrdiff_t>(p[1]),
                    static_cast<std::ptrdiff_t>(p[2])};
                q[b] += j;
                q[c] += k;
                ++q[a];
                const double upper = sample(grid, field, q);
                q[a] -= 2;
                difference += weight[j + 1] * weight[k + 1] * (upper - sample(grid, field, q));
            }
        gradient[a] = static_cast<double>(difference / (32.0L * grid.spacing[a]));
    }
    return gradient;
}

Vector normal_from_gradient(const Vector& gradient)
{
    const double magnitude = std::hypot(gradient[0], gradient[1], gradient[2]);
    if (magnitude == 0.0)
        return {0.0, 0.0, 0.0};
    return {-gradient[0] / magnitude, -gradient[1] / magnitude, -gradient[2] / magnitude};
}

double dot(const Vector& a, const Vector& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vector cross(const Vector& a, const Vector& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

Vector subtract(const Vector& a, const Vector& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Vector tangent(const Vector& normal)
{
    std::size_t axis = 0;
    for (std::size_t a = 1; a < 3; ++a)
        if (std::abs(normal[a]) < std::abs(normal[axis]))
            axis = a;
    Vector direction{};
    direction[axis] = 1.0;
    Vector result = cross(direction, normal);
    const double length = std::hypot(result[0], result[1], result[2]);
    for (auto& value : result)
        value /= length;
    return result;
}

/** An area centroid of the PLIC cut polygon, relative to its cell center. */
struct InterfaceCentroid
{
    Vector point{};
    double area = 0.0;
};

InterfaceCentroid interface_centroid(const UniformGrid& grid, const InterfacePlane& plane)
{
    if (plane.normal == Vector{0.0, 0.0, 0.0})
        return {};
    std::vector<Vector> vertices;
    vertices.reserve(6);
    auto add_vertex = [&](const Vector& candidate)
    {
        for (const auto& vertex : vertices)
            if (std::hypot((candidate[0] - vertex[0]) / grid.spacing[0], (candidate[1] - vertex[1]) / grid.spacing[1],
                    (candidate[2] - vertex[2]) / grid.spacing[2]) < 64.0 * std::numeric_limits<double>::epsilon())
                return;
        vertices.push_back(candidate);
    };
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const auto b = (axis + 1) % 3;
        const auto c = (axis + 2) % 3;
        for (const double sb : {-0.5, 0.5})
            for (const double sc : {-0.5, 0.5})
            {
                Vector lo{};
                lo[axis] = -0.5 * grid.spacing[axis];
                lo[b] = sb * grid.spacing[b];
                lo[c] = sc * grid.spacing[c];
                Vector hi = lo;
                hi[axis] = 0.5 * grid.spacing[axis];
                const double dlo = dot(plane.normal, lo) - plane.offset;
                const double dhi = dot(plane.normal, hi) - plane.offset;
                if (dlo == 0.0)
                    add_vertex(lo);
                if (dhi == 0.0)
                    add_vertex(hi);
                if ((dlo < 0.0 && dhi > 0.0) || (dlo > 0.0 && dhi < 0.0))
                {
                    Vector intersection = lo;
                    intersection[axis] += grid.spacing[axis] * (dlo / (dlo - dhi));
                    add_vertex(intersection);
                }
            }
    }
    if (vertices.size() < 3)
        return {};
    Vector center{};
    for (const auto& vertex : vertices)
        for (std::size_t a = 0; a < 3; ++a)
            center[a] += vertex[a] / static_cast<double>(vertices.size());
    const auto t1 = tangent(plane.normal);
    const auto t2 = cross(plane.normal, t1);
    std::sort(vertices.begin(), vertices.end(),
        [&](const Vector& a, const Vector& b)
        {
            const auto da = subtract(a, center);
            const auto db = subtract(b, center);
            return std::atan2(dot(da, t2), dot(da, t1)) < std::atan2(dot(db, t2), dot(db, t1));
        });
    InterfaceCentroid result;
    for (std::size_t i = 0; i < vertices.size(); ++i)
    {
        const auto& a = vertices[i];
        const auto& b = vertices[(i + 1) % vertices.size()];
        const auto product = cross(subtract(a, center), subtract(b, center));
        const double area = 0.5 * std::hypot(product[0], product[1], product[2]);
        const double total = result.area + area;
        if (!(total > 0.0))
            continue;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const double triangle_center = center[axis] / 3.0 + a[axis] / 3.0 + b[axis] / 3.0;
            result.point[axis] += (area / total) * (triangle_center - result.point[axis]);
        }
        result.area = total;
    }
    if (!(result.area > 0.0) || !std::isfinite(result.area))
        return {};
    return result;
}

/** A dimensionless weighted sample for a local quadratic height graph. */
struct SurfaceFitPoint
{
    std::array<double, 6> basis{};
    double height = 0.0;
    double weight = 0.0;
};

bool quadratic_fit(const std::vector<SurfaceFitPoint>& points, std::size_t terms, std::array<double, 6>& coefficients)
{
    if (points.size() < (terms == 3 ? 3U : 9U))
        return false;
    std::array<std::vector<double>, 6> columns;
    std::array<std::array<double, 6>, 6> r{};
    std::array<std::size_t, 6> permutation{0, 1, 2, 3, 4, 5};
    std::array<double, 6> projected{};
    std::vector<double> rhs(points.size());
    double maximum_weight = 0.0;
    for (const auto& point : points)
        maximum_weight = std::max(maximum_weight, point.weight);
    if (!(maximum_weight > 0.0))
        return false;
    for (std::size_t j = 0; j < terms; ++j)
        columns[j].resize(points.size());
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        const double weight = std::sqrt(points[i].weight / maximum_weight);
        rhs[i] = weight * points[i].height;
        for (std::size_t j = 0; j < terms; ++j)
            columns[j][i] = weight * points[i].basis[j];
    }
    double largest_diagonal = 0.0;
    for (std::size_t k = 0; k < terms; ++k)
    {
        std::size_t pivot = k;
        double norm2 = -1.0;
        for (std::size_t j = k; j < terms; ++j)
        {
            const double candidate = std::inner_product(columns[j].begin(), columns[j].end(), columns[j].begin(), 0.0);
            if (candidate > norm2)
            {
                pivot = j;
                norm2 = candidate;
            }
        }
        std::swap(columns[k], columns[pivot]);
        std::swap(permutation[k], permutation[pivot]);
        for (std::size_t j = 0; j < k; ++j)
            std::swap(r[j][k], r[j][pivot]);
        const double diagonal = std::sqrt(norm2);
        largest_diagonal = std::max(largest_diagonal, diagonal);
        // Pivoted, reorthogonalized QR avoids squaring the design condition
        // number. Rank-deficient or nearly one-dimensional patches are rejected.
        if (!std::isfinite(diagonal) || diagonal <= 1.0e-8 * largest_diagonal)
            return false;
        r[k][k] = diagonal;
        for (auto& value : columns[k])
            value /= diagonal;
        projected[k] = std::inner_product(columns[k].begin(), columns[k].end(), rhs.begin(), 0.0);
        for (std::size_t j = k + 1; j < terms; ++j)
            for (int pass = 0; pass < 2; ++pass)
            {
                const double projection =
                    std::inner_product(columns[k].begin(), columns[k].end(), columns[j].begin(), 0.0);
                r[k][j] += projection;
                for (std::size_t i = 0; i < points.size(); ++i)
                    columns[j][i] -= projection * columns[k][i];
            }
    }
    std::array<double, 6> solution{};
    for (std::size_t row = terms; row-- > 0;)
    {
        double value = projected[row];
        for (std::size_t j = row + 1; j < terms; ++j)
            value -= r[row][j] * solution[j];
        solution[row] = value / r[row][row];
        if (!std::isfinite(solution[row]))
            return false;
    }
    for (std::size_t j = 0; j < terms; ++j)
        coefficients[permutation[j]] = solution[j];
    double residual2 = 0.0;
    double weight_sum = 0.0;
    for (const auto& point : points)
    {
        const double fitted =
            std::inner_product(point.basis.begin(), point.basis.begin() + terms, coefficients.begin(), 0.0);
        const double residual = fitted - point.height;
        const double weight = point.weight / maximum_weight;
        residual2 += weight * residual * residual;
        weight_sum += weight;
    }
    // A single local graph cannot represent separated surfaces or a folded neck.
    // Keep such patches on the explicit last fallback, rather than bridging them.
    return residual2 <= 0.0625 * weight_sum;
}

bool surface_fit_curvature(const UniformGrid& grid, const std::vector<InterfacePlane>& planes,
    const std::vector<InterfaceCentroid>& centroids, std::size_t cell, const Vector& normal, double& value)
{
    std::size_t dimensions = 0;
    std::size_t inactive_axis = 0;
    for (std::size_t a = 0; a < 3; ++a)
        if (grid.cells[a] > 1)
            ++dimensions;
        else
            inactive_axis = a;
    if (dimensions <= 1)
    {
        value = 0.0;
        return true;
    }
    if (normal == Vector{0.0, 0.0, 0.0} || !(centroids[cell].area > 0.0))
        return false;
    Vector t1 = tangent(normal);
    if (dimensions == 2)
    {
        Vector inactive{};
        inactive[inactive_axis] = 1.0;
        t1 = cross(inactive, normal);
        const double length = std::hypot(t1[0], t1[1], t1[2]);
        for (auto& component : t1)
            component /= length;
    }
    const Vector t2 = cross(normal, t1);
    auto physical_scale = [&](const Vector& direction)
    {
        return std::hypot(
            direction[0] * grid.spacing[0], direction[1] * grid.spacing[1], direction[2] * grid.spacing[2]);
    };
    const double h1 = physical_scale(t1);
    const double h2 = physical_scale(t2);
    const double hn = physical_scale(normal);
    const auto p = grid.cell_coordinates(cell);
    for (int radius = 1; radius <= 2; ++radius)
    {
        std::array<int, 3> lo{};
        std::array<int, 3> hi{};
        for (std::size_t a = 0; a < 3; ++a)
            if (grid.periodic[a])
            {
                // One representative per periodic cell, including even extents.
                lo[a] = -static_cast<int>(std::min<std::size_t>(radius, (grid.cells[a] - 1) / 2));
                hi[a] = static_cast<int>(std::min<std::size_t>(radius, grid.cells[a] / 2));
            }
            else
            {
                lo[a] = -static_cast<int>(std::min<std::size_t>(radius, p[a]));
                hi[a] = static_cast<int>(std::min<std::size_t>(radius, grid.cells[a] - 1 - p[a]));
            }
        std::vector<SurfaceFitPoint> points;
        points.reserve(125);
        for (int k = lo[2]; k <= hi[2]; ++k)
            for (int j = lo[1]; j <= hi[1]; ++j)
                for (int i = lo[0]; i <= hi[0]; ++i)
                {
                    const std::array<int, 3> shift{i, j, k};
                    std::array<std::size_t, 3> q{};
                    for (std::size_t a = 0; a < 3; ++a)
                        q[a] =
                            image_index(static_cast<std::ptrdiff_t>(p[a]) + shift[a], grid.cells[a], grid.periodic[a]);
                    const auto neighbor = grid.cell_index(q[0], q[1], q[2]);
                    if (!(centroids[neighbor].area > 0.0) || dot(planes[neighbor].normal, normal) < 0.5)
                        continue;
                    Vector displacement{};
                    double distance2 = 0.0;
                    for (std::size_t a = 0; a < 3; ++a)
                    {
                        displacement[a] =
                            shift[a] * grid.spacing[a] + centroids[neighbor].point[a] - centroids[cell].point[a];
                        if (grid.periodic[a])
                        {
                            const double length = static_cast<double>(grid.cells[a]) * grid.spacing[a];
                            displacement[a] -= length * std::round(displacement[a] / length);
                        }
                        distance2 += std::pow(displacement[a] / grid.spacing[a], 2);
                    }
                    const double x = dot(displacement, t1) / h1;
                    const double y = dot(displacement, t2) / h2;
                    SurfaceFitPoint point;
                    point.basis = dimensions == 2 ? std::array<double, 6>{1.0, x, x * x, 0.0, 0.0, 0.0}
                                                  : std::array<double, 6>{1.0, x, y, x * x, x * y, y * y};
                    point.height = dot(displacement, normal) / hn;
                    point.weight = centroids[neighbor].area / (1.0 + distance2);
                    points.push_back(point);
                }
        std::array<double, 6> fit{};
        if (!quadratic_fit(points, dimensions == 2 ? 3 : 6, fit))
            continue;
        const double dx = hn * fit[1] / h1;
        const double dxx = 2.0 * hn * fit[dimensions == 2 ? 2 : 3] / (h1 * h1);
        if (dimensions == 2)
            value = -dxx / std::pow(1.0 + dx * dx, 1.5);
        else
        {
            const double dy = hn * fit[2] / h2;
            const double dyy = 2.0 * hn * fit[5] / (h2 * h2);
            const double dxy = hn * fit[4] / (h1 * h2);
            value = -(dxx * (1.0 + dy * dy) + dyy * (1.0 + dx * dx) - 2.0 * dx * dy * dxy) /
                    std::pow(1.0 + dx * dx + dy * dy, 1.5);
        }
        if (std::isfinite(value))
            return true;
    }
    return false;
}

// Integral of the two-uniform CDF. The piecewise polynomial avoids the severe
// cancellation of the usual inclusion/exclusion formula near axial normals.
long double integrated_two_uniform(long double s, long double a, long double b)
{
    if (s <= 0.0L)
        return 0.0L;
    if (s < a)
        return s * s * s / (6.0L * a * b);
    if (s < b)
        return (s * (s - a) + a * a / 3.0L) / (2.0L * b);
    if (s < a + b)
    {
        const long double tail = a + b - s;
        return s - (a + b) / 2.0L + tail * tail * tail / (6.0L * a * b);
    }
    return s - (a + b) / 2.0L;
}

long double box_fraction(const Vector& lower, const Vector& upper, const Vector& normal, double offset)
{
    std::array<long double, 3> lengths{};
    std::size_t dimensions = 0;
    long double distance = offset;
    long double total = 0.0L;
    for (std::size_t a = 0; a < 3; ++a)
    {
        const long double n = normal[a];
        distance -= n * (n >= 0.0L ? lower[a] : upper[a]);
        const long double length = std::abs(n) * (static_cast<long double>(upper[a]) - lower[a]);
        if (length > 0.0L)
        {
            lengths[dimensions++] = length;
            total += length;
        }
    }
    if (distance < 0.0L)
        return 0.0L;
    if (distance >= total)
        return 1.0L;
    if (distance == 0.0L)
        return 0.0L;
    std::sort(lengths.begin(), lengths.begin() + dimensions);
    const bool complement = distance > total / 2.0L;
    if (complement)
        distance = total - distance;
    long double fraction = 0.0L;
    if (dimensions == 1)
        fraction = distance / lengths[0];
    else if (dimensions == 2)
    {
        const auto a = lengths[0];
        const auto b = lengths[1];
        fraction = distance < a ? distance * distance / (2.0L * a * b) : (distance - a / 2.0L) / b;
    }
    else if (dimensions == 3)
    {
        const auto a = lengths[0];
        const auto b = lengths[1];
        const auto c = lengths[2];
        fraction = (integrated_two_uniform(distance, a, b) - integrated_two_uniform(distance - c, a, b)) / c;
    }
    return complement ? 1.0L - fraction : fraction;
}

bool height_curvature(
    const UniformGrid& grid, const std::vector<double>& fraction, std::size_t cell, const Vector& normal, double& value)
{
    std::size_t axis = 0;
    for (std::size_t a = 1; a < 3; ++a)
        if (std::abs(normal[a]) * grid.spacing[a] > std::abs(normal[axis]) * grid.spacing[axis])
            axis = a;
    if (normal[axis] == 0.0 || grid.cells[axis] < 5)
        return false;
    const int radius = static_cast<int>(std::min<std::size_t>(3, (grid.cells[axis] - 1) / 2));
    const auto b = (axis + 1) % 3;
    const auto c = (axis + 2) % 3;
    const auto p = grid.cell_coordinates(cell);
    const bool gas_below = normal[axis] > 0.0;
    double heights[3][3]{};
    for (int j = -1; j <= 1; ++j)
        for (int k = -1; k <= 1; ++k)
        {
            std::array<std::ptrdiff_t, 3> q{static_cast<std::ptrdiff_t>(p[0]), static_cast<std::ptrdiff_t>(p[1]),
                static_cast<std::ptrdiff_t>(p[2])};
            q[b] += j;
            q[c] += k;
            double previous = gas_below ? 1.0 : 0.0;
            double sum = 0.0;
            for (int offset = -radius; offset <= radius; ++offset)
            {
                q[axis] = static_cast<std::ptrdiff_t>(p[axis]) + offset;
                const double f = sample(grid, fraction, q);
                // Require a complete, single-valued interface in every column.
                if ((gas_below && f > previous + fraction_roundoff) || (!gas_below && f < previous - fraction_roundoff))
                    return false;
                if (offset == -radius && std::abs(f - (gas_below ? 1.0 : 0.0)) > fraction_roundoff)
                    return false;
                if (offset == radius && std::abs(f - (gas_below ? 0.0 : 1.0)) > fraction_roundoff)
                    return false;
                previous = f;
                sum += f;
            }
            // The common column-origin constant cancels in all derivatives.
            heights[j + 1][k + 1] = (gas_below ? sum : -sum) * grid.spacing[axis];
        }
    const double db = grid.cells[b] == 1 ? 0.0 : (heights[2][1] - heights[0][1]) / (2.0 * grid.spacing[b]);
    const double dc = grid.cells[c] == 1 ? 0.0 : (heights[1][2] - heights[1][0]) / (2.0 * grid.spacing[c]);
    double dbb = grid.cells[b] == 1
                     ? 0.0
                     : (heights[2][1] - 2.0 * heights[1][1] + heights[0][1]) / (grid.spacing[b] * grid.spacing[b]);
    double dcc = grid.cells[c] == 1
                     ? 0.0
                     : (heights[1][2] - 2.0 * heights[1][1] + heights[1][0]) / (grid.spacing[c] * grid.spacing[c]);
    if (grid.cells[b] > 1 && grid.cells[c] > 1)
    {
        // Transverse second-derivative filtering suppresses a 3D alternating
        // mode without altering quadratic graphs or invariant 2D interfaces.
        // The 0.2 neighbor weight follows https://basilisk.fr/src/curvature.h;
        // our heights have physical units, with a separate denominator per axis.
        constexpr std::array<double, 3> weights{0.2, 1.0, 0.2};
        dbb = 0.0;
        dcc = 0.0;
        for (std::size_t offset = 0; offset < 3; ++offset)
        {
            dbb += weights[offset] * (heights[2][offset] - 2.0 * heights[1][offset] + heights[0][offset]);
            dcc += weights[offset] * (heights[offset][2] - 2.0 * heights[offset][1] + heights[offset][0]);
        }
        dbb /= 1.4 * grid.spacing[b] * grid.spacing[b];
        dcc /= 1.4 * grid.spacing[c] * grid.spacing[c];
    }
    const double dbc = grid.cells[b] == 1 || grid.cells[c] == 1
                           ? 0.0
                           : (heights[2][2] - heights[2][0] - heights[0][2] + heights[0][0]) /
                                 (4.0 * grid.spacing[b] * grid.spacing[c]);
    const double sign = gas_below ? -1.0 : 1.0;
    value = sign * (dbb * (1.0 + dc * dc) + dcc * (1.0 + db * db) - 2.0 * db * dc * dbc) /
            std::pow(1.0 + db * db + dc * dc, 1.5);
    return true;
}
} // namespace

void UniformGrid::validate() const
{
    for (std::size_t a = 0; a < 3; ++a)
    {
        if (cells[a] == 0 || cells[a] >= static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max() / 2))
            throw std::invalid_argument("Cartesian grid sizes must be positive and indexable.");
        if (!std::isfinite(origin[a]) || !std::isfinite(spacing[a]) || spacing[a] <= 0.0 ||
            !std::isfinite(1.0 / spacing[a]) || !std::isfinite(origin[a] + static_cast<double>(cells[a]) * spacing[a]))
            throw std::invalid_argument("Cartesian grid coordinates must be finite with positive spacing.");
        if (!std::isfinite(face_area(a)) || face_area(a) <= 0.0)
            throw std::invalid_argument("Cartesian face areas must be finite and positive.");
        (void) face_count(a);
    }
    const double volume = cell_volume();
    if (!std::isfinite(volume) || volume <= 0.0)
        throw std::invalid_argument("Cartesian cell volume is not finite and positive.");
    (void) cell_count();
}

std::size_t UniformGrid::cell_count() const
{
    return checked_product(checked_product(cells[0], cells[1]), cells[2]);
}

std::size_t UniformGrid::face_count(std::size_t axis) const
{
    check_axis(axis);
    auto size = cells;
    if (size[axis] == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Cartesian face index count overflows size_t.");
    ++size[axis];
    return checked_product(checked_product(size[0], size[1]), size[2]);
}

std::size_t UniformGrid::cell_index(std::size_t i, std::size_t j, std::size_t k) const
{
    (void) cell_count();
    if (i >= cells[0] || j >= cells[1] || k >= cells[2])
        throw std::out_of_range("Cartesian cell coordinate is outside the grid.");
    return i + cells[0] * (j + cells[1] * k);
}

std::array<std::size_t, 3> UniformGrid::cell_coordinates(std::size_t id) const
{
    if (id >= cell_count())
        throw std::out_of_range("Cartesian cell index is outside the grid.");
    return {id % cells[0], (id / cells[0]) % cells[1], id / (cells[0] * cells[1])};
}

std::size_t UniformGrid::face_index(std::size_t axis, std::size_t i, std::size_t j, std::size_t k) const
{
    (void) face_count(axis);
    auto size = cells;
    ++size[axis];
    if (i >= size[0] || j >= size[1] || k >= size[2])
        throw std::out_of_range("Cartesian face coordinate is outside the grid.");
    return i + size[0] * (j + size[1] * k);
}

double UniformGrid::cell_volume() const
{
    return spacing[0] * spacing[1] * spacing[2];
}

double UniformGrid::face_area(std::size_t axis) const
{
    check_axis(axis);
    return spacing[(axis + 1) % 3] * spacing[(axis + 2) % 3];
}

Vector UniformGrid::cell_center(std::size_t id) const
{
    const auto p = cell_coordinates(id);
    return {origin[0] + (static_cast<double>(p[0]) + 0.5) * spacing[0],
        origin[1] + (static_cast<double>(p[1]) + 0.5) * spacing[1],
        origin[2] + (static_cast<double>(p[2]) + 0.5) * spacing[2]};
}

GeometricVoF::GeometricVoF(UniformGrid grid) : grid_(std::move(grid))
{
    grid_.validate();
}

const UniformGrid& GeometricVoF::grid() const noexcept
{
    return grid_;
}

double GeometricVoF::box_plane_volume(const Vector& lower, const Vector& upper, const Vector& normal, double offset)
{
    long double volume = 1.0L;
    if (!std::isfinite(offset))
        throw std::invalid_argument("PLIC plane offset must be finite.");
    for (std::size_t a = 0; a < 3; ++a)
    {
        if (!std::isfinite(lower[a]) || !std::isfinite(upper[a]) || !std::isfinite(normal[a]) || upper[a] < lower[a])
            throw std::invalid_argument("PLIC box and normal must be finite and ordered.");
        volume *= static_cast<long double>(upper[a]) - lower[a];
    }
    const double result = static_cast<double>(volume * box_fraction(lower, upper, normal, offset));
    if (!std::isfinite(result))
        throw std::invalid_argument("PLIC box volume overflows double.");
    return result;
}

PlaneSection GeometricVoF::plane_section(const Vector& lower, const Vector& upper, const Vector& normal, double offset)
{
    if (!std::isfinite(offset))
        throw std::invalid_argument("PLIC section offset must be finite.");
    UniformGrid local;
    Vector center{};
    double scale = 0.0;
    for (std::size_t a = 0; a < 3; ++a)
    {
        local.spacing[a] = upper[a] - lower[a];
        if (!std::isfinite(lower[a]) || !std::isfinite(upper[a]) || !std::isfinite(normal[a]) ||
            !std::isfinite(local.spacing[a]) || !(local.spacing[a] > 0.0))
            throw std::invalid_argument("PLIC section requires finite normal and positive box widths.");
        center[a] = std::midpoint(lower[a], upper[a]);
        scale = std::max(scale, std::abs(normal[a]));
    }
    PlaneSection section;
    section.centroid = center;
    if (scale == 0.0)
        return section;
    Vector unit{};
    for (std::size_t a = 0; a < 3; ++a)
        unit[a] = normal[a] / scale;
    const double magnitude = std::hypot(unit[0], unit[1], unit[2]);
    for (auto& component : unit)
        component /= magnitude;
    const double local_offset = (offset / scale) / magnitude - dot(unit, center);
    if (!std::isfinite(local_offset))
        throw std::invalid_argument("PLIC section offset is not representable in local coordinates.");
    const auto cut = interface_centroid(local, InterfacePlane{unit, local_offset});
    section.normal = unit;
    section.area = cut.area;
    if (cut.area > 0.0)
        for (std::size_t a = 0; a < 3; ++a)
            section.centroid[a] += cut.point[a];
    return section;
}

std::vector<InterfacePlane> GeometricVoF::reconstruct(const std::vector<double>& fraction) const
{
    std::vector<std::size_t> cells(fraction.size());
    std::iota(cells.begin(), cells.end(), 0);
    return reconstruct(fraction, cells);
}

std::vector<InterfacePlane> GeometricVoF::reconstruct(
    const std::vector<double>& fraction, std::span<const std::size_t> cells) const
{
    check_fraction(grid_, fraction);
    std::vector<InterfacePlane> planes(cells.size());
    Vector lower{};
    Vector upper{};
    for (std::size_t a = 0; a < 3; ++a)
    {
        lower[a] = -0.5 * grid_.spacing[a];
        upper[a] = 0.5 * grid_.spacing[a];
    }
    for (std::size_t index = 0; index < cells.size(); ++index)
    {
        const auto cell = cells[index];
        if (cell >= fraction.size())
            throw std::out_of_range("Requested PLIC cell is outside the grid.");
        const double f = fraction[cell];
        if (f <= 0.0 || f >= 1.0)
        {
            planes[index].offset = f <= 0.0 ? -1.0 : 1.0;
            continue;
        }
        Vector normal = normal_from_gradient(physical_gradient(grid_, fraction, cell));
        if (normal == Vector{0.0, 0.0, 0.0})
        {
            // A symmetric isolated fraction has no identifiable orientation.
            // Choose the first active axis, retaining the exact cell volume.
            std::size_t axis = 0;
            while (axis < 2 && grid_.cells[axis] == 1)
                ++axis;
            normal[axis] = 1.0;
        }
        double extent = 0.0;
        for (std::size_t a = 0; a < 3; ++a)
            extent += std::abs(normal[a]) * upper[a];
        double lo = -extent;
        double hi = extent;
        for (int iteration = 0; iteration < 55; ++iteration)
        {
            const double middle = std::midpoint(lo, hi);
            if (box_fraction(lower, upper, normal, middle) < f)
                lo = middle;
            else
                hi = middle;
        }
        planes[index] = {normal, std::midpoint(lo, hi)};
    }
    return planes;
}

std::vector<double> GeometricVoF::curvature(const std::vector<double>& fraction) const
{
    std::vector<std::size_t> cells(fraction.size());
    std::iota(cells.begin(), cells.end(), 0);
    return curvature(fraction, cells);
}

std::vector<double> GeometricVoF::curvature(
    const std::vector<double>& fraction, std::span<const std::size_t> cells) const
{
    check_fraction(grid_, fraction);
    std::vector<Vector> normals(fraction.size());
    for (std::size_t cell = 0; cell < fraction.size(); ++cell)
        normals[cell] = normal_from_gradient(physical_gradient(grid_, fraction, cell));
    std::vector<double> result(cells.size(), 0.0);
    std::vector<std::size_t> incomplete;
    for (std::size_t index = 0; index < cells.size(); ++index)
    {
        const auto cell = cells[index];
        if (cell >= fraction.size())
            throw std::out_of_range("Requested curvature cell is outside the grid.");
        if (fraction[cell] <= fraction_roundoff || fraction[cell] >= 1.0 - fraction_roundoff)
            continue;
        if (height_curvature(grid_, fraction, cell, normals[cell], result[index]))
            continue;
        incomplete.push_back(index);
    }
    if (incomplete.empty())
        return result;
    // Only the radius-two surface-fit neighborhoods can consume reconstructed
    // centroids. Their PLIC gradients add one fraction layer, giving radius three.
    std::vector<unsigned char> needed(fraction.size());
    for (const auto index : incomplete)
    {
        const auto p = grid_.cell_coordinates(cells[index]);
        for (int k = -2; k <= 2; ++k)
            for (int j = -2; j <= 2; ++j)
                for (int i = -2; i <= 2; ++i)
                {
                    const std::array<int, 3> shift{i, j, k};
                    std::array<std::size_t, 3> q{};
                    for (std::size_t a = 0; a < 3; ++a)
                        q[a] = image_index(
                            static_cast<std::ptrdiff_t>(p[a]) + shift[a], grid_.cells[a], grid_.periodic[a]);
                    const auto neighbor = grid_.cell_index(q[0], q[1], q[2]);
                    if (fraction[neighbor] > fraction_roundoff && fraction[neighbor] < 1.0 - fraction_roundoff)
                        needed[neighbor] = 1;
                }
    }
    std::vector<std::size_t> requested;
    for (std::size_t cell = 0; cell < needed.size(); ++cell)
        if (needed[cell])
            requested.push_back(cell);
    const auto selected_planes = reconstruct(fraction, requested);
    std::vector<InterfacePlane> planes(fraction.size());
    std::vector<InterfaceCentroid> centroids(fraction.size());
    for (std::size_t i = 0; i < requested.size(); ++i)
    {
        planes[requested[i]] = selected_planes[i];
        centroids[requested[i]] = interface_centroid(grid_, selected_planes[i]);
    }
    for (const auto index : incomplete)
    {
        const auto cell = cells[index];
        if (surface_fit_curvature(grid_, planes, centroids, cell, normals[cell], result[index]))
            continue;
        result[index] = 0.0;
        const auto p = grid_.cell_coordinates(cell);
        for (std::size_t a = 0; a < 3; ++a)
        {
            if (grid_.cells[a] == 1)
                continue;
            auto lo = p;
            auto hi = p;
            lo[a] = image_index(static_cast<std::ptrdiff_t>(p[a]) - 1, grid_.cells[a], grid_.periodic[a]);
            hi[a] = image_index(static_cast<std::ptrdiff_t>(p[a]) + 1, grid_.cells[a], grid_.periodic[a]);
            result[index] += (normals[grid_.cell_index(hi[0], hi[1], hi[2])][a] -
                                 normals[grid_.cell_index(lo[0], lo[1], lo[2])][a]) /
                             (2.0 * grid_.spacing[a]);
        }
    }
    return result;
}

AdvectionResult GeometricVoF::advect(
    const std::vector<double>& fraction, const FaceField& velocity, double dt, std::size_t first_axis) const
{
    check_fraction(grid_, fraction);
    check_axis(first_axis);
    if (!std::isfinite(dt) || dt < 0.0)
        throw std::invalid_argument("VoF timestep must be finite and nonnegative.");
    AdvectionResult result;
    result.fraction = fraction;
    std::array<bool, 3> active{};
    for (std::size_t a = 0; a < 3; ++a)
    {
        if (velocity[a].size() != grid_.face_count(a))
            throw std::invalid_argument("VoF face velocity size does not match the grid.");
        result.gas_volume[a].assign(velocity[a].size(), 0.0);
        result.liquid_volume[a].assign(velocity[a].size(), 0.0);
        for (const auto value : velocity[a])
        {
            if (!std::isfinite(value))
                throw std::invalid_argument("VoF face velocities must be finite.");
            active[a] = active[a] || value != 0.0;
        }
    }
    for (std::size_t cell = 0; cell < fraction.size(); ++cell)
    {
        const auto p = grid_.cell_coordinates(cell);
        double courant = 0.0;
        double divergence = 0.0;
        for (std::size_t a = 0; a < 3; ++a)
        {
            auto q = p;
            ++q[a];
            const double lo = velocity[a][grid_.face_index(a, p[0], p[1], p[2])];
            const double hi = velocity[a][grid_.face_index(a, q[0], q[1], q[2])];
            courant += std::max(std::abs(lo), std::abs(hi)) * dt / grid_.spacing[a];
            divergence += (hi - lo) * dt / grid_.spacing[a];
            if (p[a] == 0)
            {
                q = p;
                q[a] = grid_.cells[a];
                const double end = velocity[a][grid_.face_index(a, q[0], q[1], q[2])];
                if (grid_.periodic[a] && lo != end)
                    throw std::invalid_argument("VoF periodic endpoint velocities must agree exactly.");
                if (!grid_.periodic[a] && (lo != 0.0 || end != 0.0))
                    throw std::invalid_argument("VoF nonperiodic boundaries require zero normal velocity.");
            }
        }
        if (!std::isfinite(courant) || courant > 0.5 + 1.0e-14)
            throw std::runtime_error("VoF multidimensional Courant number exceeds 0.5.");
        result.maximum_volume_divergence = std::max(result.maximum_volume_divergence, std::abs(divergence));
    }
    if (result.maximum_volume_divergence > 1.0e-10)
        throw std::runtime_error("VoF geometric advection requires divergence-free face velocities.");
    if (dt == 0.0)
        return result;
    std::vector<double> compression(fraction.size());
    for (std::size_t cell = 0; cell < fraction.size(); ++cell)
        compression[cell] = fraction[cell] >= 0.5 ? 1.0 : 0.0;
    const double volume = grid_.cell_volume();
    for (std::size_t sweep = 0; sweep < 3; ++sweep)
    {
        const std::size_t a = (first_axis + sweep) % 3;
        if (!active[a])
            continue;
        const auto planes = reconstruct(result.fraction);
        auto size = grid_.cells;
        ++size[a];
        for (std::size_t k = 0; k < size[2]; ++k)
            for (std::size_t j = 0; j < size[1]; ++j)
                for (std::size_t i = 0; i < size[0]; ++i)
                {
                    std::array<std::size_t, 3> face{i, j, k};
                    const auto id = grid_.face_index(a, i, j, k);
                    const double u = velocity[a][id];
                    if (u == 0.0)
                        continue;
                    auto donor = face;
                    donor[a] = image_index(
                        static_cast<std::ptrdiff_t>(face[a]) - (u > 0.0 ? 1 : 0), grid_.cells[a], grid_.periodic[a]);
                    const auto cell = grid_.cell_index(donor[0], donor[1], donor[2]);
                    Vector lower{};
                    Vector upper{};
                    for (std::size_t b = 0; b < 3; ++b)
                    {
                        lower[b] = -0.5 * grid_.spacing[b];
                        upper[b] = 0.5 * grid_.spacing[b];
                    }
                    const double width = std::abs(u) * dt;
                    if (u > 0.0)
                        lower[a] = upper[a] - width;
                    else
                        upper[a] = lower[a] + width;
                    const double total_flux = u * dt * grid_.face_area(a);
                    double gas_flux = 0.0;
                    if (result.fraction[cell] >= 1.0)
                        gas_flux = total_flux;
                    else if (result.fraction[cell] > 0.0)
                        gas_flux = static_cast<double>(
                            total_flux * box_fraction(lower, upper, planes[cell].normal, planes[cell].offset));
                    result.gas_volume[a][id] = gas_flux;
                    result.liquid_volume[a][id] = total_flux - gas_flux;
                }
        for (unsigned correction = 0; correction < 4; ++correction)
        {
            std::vector<transport_detail::PhaseBudget> factors(fraction.size());
            bool changes = false;
            for (std::size_t cell = 0; cell < fraction.size(); ++cell)
            {
                const auto p = grid_.cell_coordinates(cell);
                auto q = p;
                ++q[a];
                const auto lo = grid_.face_index(a, p[0], p[1], p[2]);
                const auto hi = grid_.face_index(a, q[0], q[1], q[2]);
                const long double gas_out = std::max(0.0, -result.gas_volume[a][lo]) +
                                            static_cast<long double>(std::max(0.0, result.gas_volume[a][hi]));
                const long double carrier_out = std::max(0.0, -result.liquid_volume[a][lo]) +
                                                static_cast<long double>(std::max(0.0, result.liquid_volume[a][hi]));
                factors[cell] = transport_detail::phase_budget(
                    gas_out, carrier_out, result.fraction[cell], volume, fraction_roundoff);
                changes = changes || factors[cell].changes_flux();
            }
            if (!changes)
                break;
            if (correction == 3)
                throw std::runtime_error("VoF could not resolve a roundoff-only phase budget.");
            for (std::size_t k = 0; k < size[2]; ++k)
                for (std::size_t j = 0; j < size[1]; ++j)
                    for (std::size_t i = 0; i < size[0]; ++i)
                    {
                        std::array<std::size_t, 3> donor{i, j, k};
                        const auto id = grid_.face_index(a, i, j, k);
                        const double u = velocity[a][id];
                        if (u == 0.0)
                            continue;
                        donor[a] = image_index(static_cast<std::ptrdiff_t>(donor[a]) - (u > 0.0 ? 1 : 0),
                            grid_.cells[a], grid_.periodic[a]);
                        const auto cell = grid_.cell_index(donor[0], donor[1], donor[2]);
                        const double total = u * dt * grid_.face_area(a);
                        result.gas_volume[a][id] =
                            transport_detail::budgeted_gas(total, result.gas_volume[a][id], factors[cell]);
                        result.liquid_volume[a][id] = total - result.gas_volume[a][id];
                    }
        }
        std::vector<double> next = result.fraction;
        for (std::size_t cell = 0; cell < fraction.size(); ++cell)
        {
            const auto p = grid_.cell_coordinates(cell);
            auto q = p;
            ++q[a];
            const auto lo = grid_.face_index(a, p[0], p[1], p[2]);
            const auto hi = grid_.face_index(a, q[0], q[1], q[2]);
            const long double gas_out = std::max(0.0, -result.gas_volume[a][lo]) +
                                        static_cast<long double>(std::max(0.0, result.gas_volume[a][hi]));
            const long double gas_in = std::max(0.0, result.gas_volume[a][lo]) +
                                       static_cast<long double>(std::max(0.0, -result.gas_volume[a][hi]));
            const long double carrier_out = std::max(0.0, -result.liquid_volume[a][lo]) +
                                            static_cast<long double>(std::max(0.0, result.liquid_volume[a][hi]));
            const long double carrier_in = std::max(0.0, result.liquid_volume[a][lo]) +
                                           static_cast<long double>(std::max(0.0, -result.liquid_volume[a][hi]));
            if (compression[cell] == 0.0)
                next[cell] = static_cast<double>(
                    ((static_cast<long double>(result.fraction[cell]) * volume - gas_out) + gas_in) / volume);
            else
                next[cell] = static_cast<double>(
                    1.0L - (((1.0L - result.fraction[cell]) * volume - carrier_out) + carrier_in) / volume);
            if (!std::isfinite(next[cell]) || next[cell] < -fraction_roundoff || next[cell] > 1.0 + fraction_roundoff)
                throw std::runtime_error("VoF directional transport lost boundedness; reduce the timestep.");
        }
        result.fraction = std::move(next);
    }
    return result;
}

} // namespace SimpleFluid::VoF
