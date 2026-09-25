/** @file MiteredCornerPatch.hh
 * @brief Defines clipped patch geometry for mitered annular corners.
 */
#pragma once

#include "geometry/MeshUtils.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes
{

/** @brief Local corner coordinates (x,y,0) and counter-clockwise convex quads.
 * x and y are the two inward wall-normal coordinates, in the caller's units.
 */
struct MiteredCornerPatch
{
    Arr<MeshUtils::Vec3> nodes;
    Arr<Arr<unsigned>> cells;
};

/**
 * @brief Replace the tensor overlap of two identical layer stacks by mitered quads.
 *
 * Offsets must start at zero and increase strictly to H. Every fine offset is
 * retained on x=H and y=H; the physical walls x=0 and y=0 use a subset of those
 * offsets with tangential gaps no larger than tangent_spacing (up to coordinate
 * roundoff). A request smaller than an existing fine interval is rejected.
 *
 * Coarse cuts are selected backwards: choose the earliest preceding fine
 * offset reachable within the requested spacing. Bottom and side strips meet
 * at x=y. At a terminating cut their two triangles are merged into one square,
 * so all output cells are convex quads and all interior edges conform. If no
 * fine intervals can be merged, return the original row-major tensor patch.
 *
 * The five offsets from 2 mm, growth 1.25, and a 10 mm tangential target select
 * cuts at 7.625 and 16.4140625 mm, giving 14 quads instead of 25.
 */
[[nodiscard]] inline MiteredCornerPatch mitered_corner_patch(
    const ArrReal& fine_offsets, real_t tangent_spacing)
{
    if (fine_offsets.size() < 2 || fine_offsets.front() != real_t{}
        || !std::isfinite(tangent_spacing) || !(tangent_spacing > real_t{}))
        throw std::invalid_argument("A mitered corner requires offsets from zero and positive finite tangent spacing.");
    const size_t n = fine_offsets.size() - 1;
    constexpr size_t index_limit = std::numeric_limits<unsigned>::max();
    if (n >= index_limit)
        throw std::overflow_error("Mitered corner offsets exceed supported node indices.");
    const auto fits = [&](real_t low, real_t high) {
        const long double scale = std::max({std::abs(static_cast<long double>(low)),
            std::abs(static_cast<long double>(high)), static_cast<long double>(tangent_spacing)});
        const long double roundoff = 8 * std::numeric_limits<real_t>::epsilon() * scale;
        return static_cast<long double>(high) - low <= static_cast<long double>(tangent_spacing) + roundoff;
    };
    for (size_t i = 0; i < fine_offsets.size(); ++i)
    {
        if (!std::isfinite(fine_offsets[i]) || (i && !(fine_offsets[i] > fine_offsets[i - 1])))
            throw std::invalid_argument("Mitered corner offsets must be finite and strictly increasing.");
        if (i && !fits(fine_offsets[i - 1], fine_offsets[i]))
            throw std::invalid_argument("Mitered corner tangent spacing is smaller than an existing fine interval.");
    }

    std::vector<size_t> cuts;
    for (size_t current = n; current != 0;)
    {
        cuts.push_back(current);
        size_t low = 0, high = current;
        while (low < high)
        {
            const size_t middle = low + (high - low) / 2;
            if (fits(fine_offsets[middle], fine_offsets[current])) high = middle;
            else low = middle + 1;
        }
        if (low == current)
            throw std::logic_error("Mitered corner cut selection did not advance.");
        current = low;
    }
    std::reverse(cuts.begin(), cuts.end());

    // For this all-quad disk, V = 2*sum(cuts) + n + 1 and
    // F = 2*sum(cuts) - number_of_cuts. Check before allocating connectivity.
    size_t node_count = n + 1;
    for (const auto cut : cuts)
    {
        if (cut > (index_limit - node_count) / 2)
            throw std::overflow_error("Mitered corner connectivity exceeds supported node indices.");
        node_count += 2 * cut;
    }
    const size_t cell_count = node_count - n - 1 - cuts.size();
    MiteredCornerPatch result;
    result.nodes.reserve(node_count);
    result.cells.reserve(cell_count);
    using Point = std::pair<size_t, size_t>;
    using Quad = std::array<Point, 4>;
    std::map<Point, unsigned> node_ids;
    const auto append = [&](const Quad& quad) {
        Arr<unsigned> cell;
        cell.reserve(4);
        for (const auto point : quad)
        {
            const auto [found, inserted] = node_ids.try_emplace(point, static_cast<unsigned>(result.nodes.size()));
            if (inserted)
                result.nodes.push_back({fine_offsets[point.first], fine_offsets[point.second], real_t{}});
            cell.push_back(found->second);
        }
        result.cells.push_back(std::move(cell));
    };
    if (cuts.size() == n)
    {
        // Includes the one-layer/no-savings case.
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                append(Quad{{{i, j}, {i + 1, j}, {i + 1, j + 1}, {i, j + 1}}});
    }
    else
        for (size_t layer = 0; layer < n; ++layer)
        {
            size_t left = layer;
            for (auto cut = std::upper_bound(cuts.begin(), cuts.end(), layer); cut != cuts.end(); ++cut)
            {
                const size_t right = *cut;
                if (left == layer && right == layer + 1)
                    append(Quad{{{layer, layer}, {right, layer}, {right, right}, {layer, right}}});
                else
                {
                    const Quad bottom{{{left, layer}, {right, layer},
                        {right, layer + 1}, {std::max(left, layer + 1), layer + 1}}};
                    append(bottom);
                    Quad side;
                    for (size_t i = 0; i < 4; ++i)
                        side[i] = {bottom[3 - i].second, bottom[3 - i].first};
                    append(side);
                }
                left = right;
            }
        }
    if (result.nodes.size() != node_count || result.cells.size() != cell_count)
        throw std::logic_error("Mitered corner connectivity count mismatch.");
    return result;
}

} // namespace SimpleFluid::Meshes
