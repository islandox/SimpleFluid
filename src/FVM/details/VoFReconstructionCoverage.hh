/** @file VoFReconstructionCoverage.hh
 * @brief Coverage of a reconstruction query expressed in represented coordinates.
 */
#pragma once

#include "geometry/CartesianAMR.hh"
#include "utils/CompensatedSum.hh"

#include <cmath>
#include <limits>
#include <span>

namespace SimpleFluid::VoF::geometry_detail
{
inline long double represented_box_volume(const Meshes::AMRBox& box)
{
    long double result = 1.0L;
    for (std::size_t a = 0; a < 3; ++a)
        result *= static_cast<long double>(box.upper[a]) - box.lower[a];
    return result;
}

/** The query has undergone the same boundary clamp as CartesianAMR::overlaps;
 * intersections are the disjoint, positive boxes returned by that query.
 * Compare represented endpoint geometry, rather than a nominal spacing product.
 * Three differences/two products and compensated positive accumulation need only
 * a volume-relative arithmetic bound, even where long double equals double.
 * The second-order term accounts for accumulation over all intersections.
 */
inline bool has_complete_reconstruction_coverage(
    const Meshes::AMRBox& query, std::span<const Meshes::AMROverlap> intersections)
{
    for (std::size_t a = 0; a < 3; ++a)
        if (!std::isfinite(query.lower[a]) || !std::isfinite(query.upper[a]) || !(query.upper[a] > query.lower[a]))
            return false;
    const long double requested = represented_box_volume(query);
    if (!(requested > 0.0L) || !std::isfinite(requested))
        return false;
    SimpleFluid::detail::CompensatedSum<long double> covered;
    for (const auto& intersection : intersections)
    {
        const long double v = represented_box_volume(intersection.box);
        if (!(v > 0.0L) || !std::isfinite(v))
            return false;
        covered += v;
    }
    const long double epsilon = std::numeric_limits<long double>::epsilon();
    const long double arithmetic_bound =
        64.0L * (epsilon + static_cast<long double>(intersections.size()) * epsilon * epsilon) * requested;
    return covered.value() > 0.0L && std::abs(covered.value() - requested) <= arithmetic_bound;
}
} // namespace SimpleFluid::VoF::geometry_detail
