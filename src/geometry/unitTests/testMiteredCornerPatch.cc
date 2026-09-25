/** @file testMiteredCornerPatch.cc
 * @brief Tests mitered-corner patch construction and geometric bounds.
 */
#include <gtest/gtest.h>

#include "geometry/mesh/MiteredCornerPatch.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace
{
using SimpleFluid::ArrReal;
using SimpleFluid::Meshes::MiteredCornerPatch;
using SimpleFluid::Meshes::mitered_corner_patch;
using Coordinate = std::pair<double, double>;

ArrReal default_offsets(size_t layers = 5)
{
    ArrReal result{0};
    double width = .002;
    for (size_t i = 0; i < layers; ++i)
    {
        result.push_back(result.back() + width);
        width *= 1.25;
    }
    return result;
}

std::vector<double> boundary_coordinates(const MiteredCornerPatch& patch, unsigned axis, double value)
{
    std::set<double> result;
    for (const auto& p : patch.nodes)
        if ((axis == 0 ? p.x : p.y) == value) result.insert(axis == 0 ? p.y : p.x);
    return {result.begin(), result.end()};
}

void check_patch(const MiteredCornerPatch& patch, const ArrReal& offsets, double tangent_spacing)
{
    ASSERT_FALSE(patch.cells.empty());
    const double h = offsets.back();
    for (const auto& p : patch.nodes)
    {
        EXPECT_TRUE(std::isfinite(p.x) && std::isfinite(p.y));
        EXPECT_GE(p.x, 0); EXPECT_LE(p.x, h);
        EXPECT_GE(p.y, 0); EXPECT_LE(p.y, h);
        EXPECT_EQ(p.z, 0);
    }
    struct Use { size_t cell; unsigned from, to; };
    std::map<std::pair<unsigned, unsigned>, std::vector<Use>> edges;
    std::set<unsigned> used_nodes;
    long double total_area = 0;
    for (size_t c = 0; c < patch.cells.size(); ++c)
    {
        const auto& cell = patch.cells[c];
        ASSERT_EQ(cell.size(), 4U);
        ASSERT_EQ(std::set<unsigned>(cell.begin(), cell.end()).size(), 4U);
        for (const auto node : cell) { ASSERT_LT(node, patch.nodes.size()); used_nodes.insert(node); }
        long double twice_area = 0;
        for (size_t i = 0; i < 4; ++i)
        {
            const auto from = cell[i], to = cell[(i + 1) % 4];
            const auto& a = patch.nodes[from]; const auto& b = patch.nodes[to];
            twice_area += static_cast<long double>(a.x) * b.y - static_cast<long double>(b.x) * a.y;
            edges[std::minmax(from, to)].push_back({c, from, to});
            for (const auto node : cell)
            {
                const auto& p = patch.nodes[node];
                const long double cross = (static_cast<long double>(b.x) - a.x) * (static_cast<long double>(p.y) - a.y)
                    - (static_cast<long double>(b.y) - a.y) * (static_cast<long double>(p.x) - a.x);
                EXPECT_GE(cross, -64 * std::numeric_limits<long double>::epsilon() * h * h);
            }
        }
        EXPECT_GT(twice_area, 0);
        total_area += twice_area / 2;
    }
    EXPECT_EQ(used_nodes.size(), patch.nodes.size());
    EXPECT_NEAR(static_cast<double>(total_area), h * h, 1e-12 * h * h);
    std::vector<std::vector<size_t>> neighbors(patch.cells.size());
    for (const auto& [edge, uses] : edges)
    {
        ASSERT_TRUE(uses.size() == 1 || uses.size() == 2);
        if (uses.size() == 2)
        {
            EXPECT_EQ(uses[0].from, uses[1].to);
            EXPECT_EQ(uses[0].to, uses[1].from);
            neighbors[uses[0].cell].push_back(uses[1].cell);
            neighbors[uses[1].cell].push_back(uses[0].cell);
        }
        else
        {
            const auto& a = patch.nodes[edge.first]; const auto& b = patch.nodes[edge.second];
            EXPECT_TRUE((a.x == b.x && (a.x == 0 || a.x == h))
                || (a.y == b.y && (a.y == 0 || a.y == h))) << "Unmatched interior edge / hanging node";
        }
    }
    std::set<size_t> visited;
    std::vector<size_t> stack{0};
    while (!stack.empty())
    {
        const auto c = stack.back(); stack.pop_back();
        if (visited.insert(c).second) stack.insert(stack.end(), neighbors[c].begin(), neighbors[c].end());
    }
    EXPECT_EQ(visited.size(), patch.cells.size());
    EXPECT_EQ(static_cast<long long>(patch.nodes.size()) - static_cast<long long>(edges.size())
        + static_cast<long long>(patch.cells.size()), 1);
    EXPECT_EQ(boundary_coordinates(patch, 0, h), offsets);
    EXPECT_EQ(boundary_coordinates(patch, 1, h), offsets);
    for (unsigned axis : {0U, 1U})
    {
        const auto wall = boundary_coordinates(patch, axis, 0);
        for (size_t i = 1; i < wall.size(); ++i)
            EXPECT_LE(wall[i] - wall[i - 1], tangent_spacing + 16 * std::numeric_limits<double>::epsilon() * h);
    }
}
} // namespace

TEST(MiteredCornerPatchTest, FiveLayersGiveFourteenConvexConformingQuads)
{
    const auto offsets = default_offsets();
    const auto patch = mitered_corner_patch(offsets, .01);
    EXPECT_EQ(patch.cells.size(), 14U);
    EXPECT_EQ(patch.nodes.size(), 22U);
    check_patch(patch, offsets, .01);
    const ArrReal coarse{0, offsets[3], offsets[5]};
    EXPECT_EQ(boundary_coordinates(patch, 0, 0), coarse);
    EXPECT_EQ(boundary_coordinates(patch, 1, 0), coarse);
    // The terminal diagonal triangle pairs become complete squares.
    for (const size_t low : {2U, 4U})
    {
        const std::set<Coordinate> square{{offsets[low], offsets[low]}, {offsets[low + 1], offsets[low]},
            {offsets[low + 1], offsets[low + 1]}, {offsets[low], offsets[low + 1]}};
        EXPECT_TRUE(std::any_of(patch.cells.begin(), patch.cells.end(), [&](const auto& cell) {
            std::set<Coordinate> points;
            for (const auto node : cell) points.emplace(patch.nodes[node].x, patch.nodes[node].y);
            return points == square;
        }));
    }
    const auto repeated = mitered_corner_patch(offsets, .01);
    EXPECT_EQ(repeated.cells, patch.cells);
    ASSERT_EQ(repeated.nodes.size(), patch.nodes.size());
    for (size_t i = 0; i < patch.nodes.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(repeated.nodes[i].x, patch.nodes[i].x);
        EXPECT_DOUBLE_EQ(repeated.nodes[i].y, patch.nodes[i].y);
    }
}

TEST(MiteredCornerPatchTest, SmallAndUnmergeableStacksRetainValidTopology)
{
    for (const size_t count : {1U, 2U, 3U})
    {
        const auto offsets = default_offsets(count);
        const auto patch = mitered_corner_patch(offsets, .02);
        EXPECT_EQ(patch.cells.size(), 2 * count - 1);
        check_patch(patch, offsets, .02);
    }
    const ArrReal offsets{0, .1, .2, .3, .4};
    const auto tensor = mitered_corner_patch(offsets, .1);
    EXPECT_EQ(tensor.cells.size(), 16U);
    EXPECT_EQ(tensor.nodes.size(), 25U);
    check_patch(tensor, offsets, .1);
    EXPECT_EQ(boundary_coordinates(tensor, 0, 0), offsets);
    EXPECT_EQ(boundary_coordinates(tensor, 1, 0), offsets);
}

TEST(MiteredCornerPatchTest, IrregularOffsetsAndLongerStacksKeepFineInterfaces)
{
    const ArrReal irregular{0, .001, .003, .007, .008, .014, .015};
    const auto patch = mitered_corner_patch(irregular, .009);
    EXPECT_EQ(patch.cells.size(), 16U);
    check_patch(patch, irregular, .009);
    const auto longer = default_offsets(8);
    check_patch(mitered_corner_patch(longer, .01), longer, .01);
    const ArrReal uniform{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const auto coarse = mitered_corner_patch(uniform, 6);
    EXPECT_EQ(coarse.cells.size(), 26U);
    EXPECT_EQ(boundary_coordinates(coarse, 0, 0), (ArrReal{0, 4, 10}));
    check_patch(coarse, uniform, 6);
}

TEST(MiteredCornerPatchTest, RejectsInvalidOffsetsAndUnattainableSpacing)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (const auto& offsets : std::vector<ArrReal>{{}, {0}, {1, 2}, {0, 0}, {0, 2, 1}, {0, nan}, {0, inf}})
        EXPECT_THROW((void)mitered_corner_patch(offsets, 10), std::invalid_argument);
    for (const double spacing : {0.0, -1.0, nan, inf})
        EXPECT_THROW((void)mitered_corner_patch({0, 1}, spacing), std::invalid_argument);
    EXPECT_THROW((void)mitered_corner_patch({0, 1, 3}, 1.5), std::invalid_argument);
}
