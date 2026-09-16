#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <variant>
#include <vector>

namespace SimpleFluid::mesh_transfer_detail
{

// Cartesian centroid, native cell volume, then lower/upper bounds for either
// x,y,z or r,theta,z. Theta lower bounds are reduced modulo 2 pi; upper bounds
// may cross that seam. All records follow the owned field-map order.
using Geometry = std::array<double, 10>;

inline constexpr double angular_period = 2.0 * std::numbers::pi_v<double>;

inline double normalized_angle(double angle)
{
    double result = std::fmod(angle, angular_period);
    if (result < 0) result += angular_period;
    // Adding the period to a tiny negative remainder can round to the period.
    return result >= angular_period ? 0.0 : result;
}

template<class Mesh>
std::vector<Geometry> geometry(const Mesh& mesh, bool conservative, bool& valid)
{
    const auto* cartesian =
        std::get_if<typename Mesh::CartesianPtr>(&mesh.variant());
    const auto* cylindrical =
        std::get_if<typename Mesh::CylindricalPtr>(&mesh.variant());
    valid = valid && (!conservative || cartesian || cylindrical);

    // Native cylindrical topology tolerates a roundoff-scale excess over one
    // turn. Its stored volumes still include that excess. For overlap geometry
    // only, represent the first physical turn once and leave the extra native
    // volume uncovered. In particular, never normalize weights or volumes to
    // hide this seam residual. A cell wholly inside the excess has zero overlap.
    bool trim_repeated_seam = false;
    double angular_origin = 0;
    if (conservative && cylindrical)
    {
        const auto& angles = (**cylindrical).cell_edges()[1];
        angular_origin = angles.front();
        const double span = angles.back() - angular_origin;
        constexpr double native_tolerance =
            64.0 * std::numeric_limits<double>::epsilon() * angular_period;
        valid = valid && std::isfinite(span) && span > 0
            && span <= angular_period + native_tolerance;
        trim_repeated_seam = span > angular_period;
    }

    std::vector<Geometry> result(mesh.num_owned_cells());
    for (std::size_t c = 0; c < result.size(); ++c)
    {
        const auto lid = static_cast<typename Mesh::local_ordinal_type>(c);
        auto& g = result[c];
        const auto centroid = mesh.cell_centroid(lid);
        for (std::size_t axis = 0; axis < 3; ++axis)
            g[axis] = centroid.component(axis);
        g[3] = mesh.cell_volume(lid);
        if (conservative)
        {
            const auto read_bounds = [&](const auto& native)
            {
                // Map IDs may change when a handle is reordered. Geometry IDs
                // remain attached to the original cell in the native backend.
                const auto cell = native.cell_id(
                    static_cast<std::size_t>(mesh.cell_geometry_global_id(lid)));
                const std::array<std::size_t, 3> indices{cell.i, cell.j, cell.k};
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    g[4 + 2 * axis] = native.cell_edges()[axis][indices[axis]];
                    g[5 + 2 * axis] = native.cell_edges()[axis][indices[axis] + 1];
                }
            };
            if (cartesian) read_bounds(**cartesian);
            else if (cylindrical)
            {
                read_bounds(**cylindrical);
                double angular_width = g[7] - g[6];
                if (trim_repeated_seam)
                {
                    const double lower =
                        std::min(angular_period, g[6] - angular_origin);
                    const double upper =
                        std::min(angular_period, g[7] - angular_origin);
                    angular_width = upper - lower;
                }
                g[6] = normalized_angle(g[6]);
                g[7] = g[6] + angular_width;
                valid = valid && g[4] > 0 && angular_width >= 0;
            }
        }
        for (double value : g) valid = valid && std::isfinite(value);
        valid = valid && g[3] > 0;
    }
    return result;
}

inline double angular_intersection(double a_lower, double a_upper,
                                   double b_lower, double b_upper)
{
    // Each arc is represented once and intersects at most the central image
    // and one adjacent periodic image. Use extended precision for seam sums;
    // no angular quadrature or symmetry multiplier is involved.
    const long double period = angular_period;
    const long double a0 = normalized_angle(a_lower);
    const long double b0 = normalized_angle(b_lower);
    const long double aw = std::min(period,
        static_cast<long double>(a_upper) - a_lower);
    const long double bw = std::min(period,
        static_cast<long double>(b_upper) - b_lower);
    if (aw <= 0 || bw <= 0) return 0;
    long double overlap = 0;
    for (int image = -1; image <= 1; ++image)
    {
        const long double start = b0 + image * period;
        overlap += std::max(0.0L,
            std::min(a0 + aw, start + bw) - std::max(a0, start));
    }
    return static_cast<double>(overlap);
}

inline double intersection(const Geometry& a, const Geometry& b, bool cylindrical)
{
    if (!cylindrical)
    {
        double volume = 1;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const auto low = 4 + 2 * axis;
            volume *= std::max(0.0, std::min(a[low + 1], b[low + 1])
                                       - std::max(a[low], b[low]));
        }
        return volume;
    }

    const double rlo = std::max(a[4], b[4]);
    const double rhi = std::min(a[5], b[5]);
    const double zlo = std::max(a[8], b[8]);
    const double zhi = std::min(a[9], b[9]);
    if (rhi <= rlo || zhi <= zlo) return 0;
    return 0.5 * (rhi - rlo) * (rhi + rlo)
        * angular_intersection(a[6], a[7], b[6], b[7]) * (zhi - zlo);
}

} // namespace SimpleFluid::mesh_transfer_detail
