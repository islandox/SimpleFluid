/** @file PitzDailyMesh.hh
 * @brief Pure native construction of the retained pitzDaily block mesh.
 */
#pragma once
#include "geometry/details/NativeMeshGenerators.hh"

#include <algorithm>
#include <cmath>
#include <compare>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid::detail::pitz_daily_mesh
{
using MeshData = Meshes::UnstructuredMeshData;
using Point = MeshUtils::Vec3;

/**
 * @brief Scale a tutorial cell count while retaining at least one cell.
 *
 * @param tutorial_cells Cell count in the reference tutorial.
 * @param divisor Requested coarsening divisor.
 * @return Rounded-up scaled cell count.
 */
inline int scaled_cells(int tutorial_cells, int divisor)
{
    return std::max(1, (tutorial_cells + divisor - 1) / divisor);
}

/**
 * @brief Generate uniformly spaced coordinates over an interval.
 *
 * @param lower Lower interval endpoint.
 * @param upper Upper interval endpoint.
 * @param cells Number of cells.
 * @return Coordinate vector containing @p cells plus one edges.
 */
inline std::vector<double> uniform_edges(double lower, double upper, int cells)
{
    std::vector<double> edges(static_cast<size_t>(cells) + 1);
    for (int edge = 0; edge <= cells; ++edge)
    {
        edges[static_cast<size_t>(edge)] =
            lower + (upper - lower) * static_cast<double>(edge) / static_cast<double>(cells);
    }
    edges.front() = lower;
    edges.back() = upper;
    return edges;
}

/** @brief Generate geometrically graded coordinates over one interval. */
inline std::vector<double> graded_edges(double lower, double upper, int cells, double expansion_ratio)
{
    if (cells <= 0 || !(upper > lower) || !(expansion_ratio > 0.0) || !std::isfinite(expansion_ratio))
    {
        throw std::invalid_argument("Graded mesh intervals require positive finite dimensions and cell counts.");
    }
    if (cells == 1 || expansion_ratio == 1.0)
    {
        return uniform_edges(lower, upper, cells);
    }

    const double width_ratio = std::pow(expansion_ratio, 1.0 / static_cast<double>(cells - 1));
    double relative_width_sum = 0.0;
    double relative_width = 1.0;
    for (int cell = 0; cell < cells; ++cell)
    {
        relative_width_sum += relative_width;
        relative_width *= width_ratio;
    }

    std::vector<double> edges;
    edges.reserve(static_cast<size_t>(cells) + 1);
    edges.push_back(lower);
    relative_width = 1.0;
    const double width_scale = (upper - lower) / relative_width_sum;
    for (int cell = 0; cell < cells; ++cell)
    {
        edges.push_back(edges.back() + width_scale * relative_width);
        relative_width *= width_ratio;
    }
    edges.back() = upper;
    return edges;
}

/** @brief One OpenFOAM-style multi-grading section. */
struct GradingSection
{
    double length_fraction = 1.0;
    double cell_fraction = 1.0;
    double expansion_ratio = 1.0;
};

/**
 * @brief Generate unit-interval edges from OpenFOAM-style multi-grading sections.
 *
 * Section length and cell fractions are normalized independently. At least one
 * cell is retained per section; meshes too coarse to represent every section
 * fall back to uniform spacing.
 */
inline std::vector<double> multi_graded_fractions(int cells, const std::vector<GradingSection>& sections)
{
    if (cells <= 0 || sections.empty())
    {
        throw std::invalid_argument("Multi-grading requires cells and at least one section.");
    }
    if (cells < static_cast<int>(sections.size()))
    {
        return uniform_edges(0.0, 1.0, cells);
    }

    double total_length_fraction = 0.0;
    double total_cell_fraction = 0.0;
    for (const auto& section : sections)
    {
        if (!(section.length_fraction > 0.0) || !(section.cell_fraction > 0.0) || !(section.expansion_ratio > 0.0) ||
            !std::isfinite(section.length_fraction) || !std::isfinite(section.cell_fraction) ||
            !std::isfinite(section.expansion_ratio))
        {
            throw std::invalid_argument("Multi-grading fractions and expansion ratios must be finite and positive.");
        }
        total_length_fraction += section.length_fraction;
        total_cell_fraction += section.cell_fraction;
    }

    // Match OpenFOAM v2606 lineDivide: round each section, then place
    // the remainder in the first section with the largest cell fraction.
    // Largest-remainder allocation differs on coarse multi-graded meshes.
    std::vector<int> section_counts;
    int allocated_cells = 0;
    size_t largest_section = 0;
    for (size_t section = 0; section < sections.size(); ++section)
    {
        const double target = static_cast<double>(cells) * sections[section].cell_fraction / total_cell_fraction;
        section_counts.push_back(static_cast<int>(target + 0.5));
        allocated_cells += section_counts.back();
        if (sections[section].cell_fraction > sections[largest_section].cell_fraction)
        {
            largest_section = section;
        }
    }
    section_counts[largest_section] += cells - allocated_cells;
    if (std::any_of(section_counts.begin(), section_counts.end(), [](int count) { return count <= 0; }))
    {
        throw std::invalid_argument("Multi-grading section received no cells.");
    }

    std::vector<double> edges{0.0};
    edges.reserve(static_cast<size_t>(cells) + 1);
    double section_lower = 0.0;
    for (size_t section = 0; section < sections.size(); ++section)
    {
        const double section_upper = section + 1 == sections.size()
                                         ? 1.0
                                         : section_lower + sections[section].length_fraction / total_length_fraction;
        auto section_edges =
            graded_edges(section_lower, section_upper, section_counts[section], sections[section].expansion_ratio);
        edges.insert(edges.end(), section_edges.begin() + 1, section_edges.end());
        section_lower = section_upper;
    }
    edges.back() = 1.0;
    return edges;
}

/** @brief Quantized coordinates used to deduplicate mesh nodes. */
struct NodeKey
{
    std::int64_t x = 0;
    std::int64_t y = 0;
    std::int64_t z = 0;

    auto operator<=>(const NodeKey&) const = default;
};

/** @brief Assign stable source identifiers to unique geometric nodes. */
class NodeRegistry
{
public:
    MeshData::ID id(const Point& point)
    {
        constexpr double coordinate_scale = 1.0e12;
        const NodeKey key{std::llround(point.x * coordinate_scale), std::llround(point.y * coordinate_scale),
            std::llround(point.z * coordinate_scale)};
        const auto existing = d_ids.find(key);
        if (existing != d_ids.end())
        {
            return existing->second;
        }

        const auto id = d_next_id++;
        d_ids.emplace(key, id);
        d_coordinates.emplace_back(id, point);
        return id;
    }

    const std::vector<std::pair<MeshData::ID, Point>>& coordinates() const noexcept { return d_coordinates; }

private:
    MeshData::ID d_next_id = 1;
    std::map<NodeKey, MeshData::ID> d_ids;
    std::vector<std::pair<MeshData::ID, Point>> d_coordinates;
};

/** @brief Structured block geometry and boundary-label specification. */
struct BlockSpec
{
    std::vector<double> x_edges;
    std::vector<double> left_y_fractions;
    std::vector<double> right_y_fractions;
    double lower_left = 0.0;
    double lower_right = 0.0;
    double upper_left = 0.0;
    double upper_right = 0.0;
    bool inlet = false;
    bool outlet = false;
    bool left_wall = false;
    bool lower_wall = false;
    bool upper_wall = false;
};

/**
 * @brief Linearly interpolate between two scalar endpoints.
 *
 * @param left Value at fraction zero.
 * @param right Value at fraction one.
 * @param fraction Interpolation fraction.
 * @return Interpolated value.
 */
inline double interpolate(double left, double right, double fraction)
{
    return left + fraction * (right - left);
}

/** Build detached native geometry, preserving block grading and node identity. */
inline MeshData generate(int divisor, int multiplier)
{
    if (divisor <= 0 || multiplier <= 0)
        throw std::invalid_argument("pitzDaily divisor and multiplier must be positive.");
    std::vector<MeshData::Cell> cells;
    std::vector<SimpleFluid::detail::NativeBoundarySide> boundaries;
    const int upstream_x = scaled_cells(18 * multiplier, divisor);
    const int main_x = scaled_cells(180 * multiplier, divisor);
    const int outlet_x = scaled_cells(25 * multiplier, divisor);
    const int upper_y = scaled_cells(30 * multiplier, divisor);
    const int lower_y = scaled_cells(27 * multiplier, divisor);

    const auto upstream_edges = graded_edges(-0.0206, 0.0, upstream_x, 0.5);
    const auto main_edges = graded_edges(0.0, 0.206, main_x, 4.0);
    const auto outlet_edges = graded_edges(0.206, 0.290, outlet_x, 2.5);
    const auto negative_y_fractions = multi_graded_fractions(lower_y, {{2.0, 4.0, 1.0}, {1.0, 3.0, 0.3}});
    const auto positive_y_fractions =
        multi_graded_fractions(upper_y, {{1.0, 4.0, 2.0}, {2.0, 3.0, 4.0}, {2.0, 4.0, 0.25}});
    const auto positive_y_reduced_fractions = multi_graded_fractions(upper_y, {{2.0, 1.0, 1.0}, {1.0, 1.0, 0.25}});
    const auto uniform_lower_fractions = uniform_edges(0.0, 1.0, lower_y);
    const std::vector<BlockSpec> blocks{{upstream_edges, positive_y_fractions, positive_y_fractions, 0.0, 0.0, 0.0254,
                                            0.0254, true, false, false, true, true},
        {main_edges, negative_y_fractions, uniform_lower_fractions, -0.0254, -0.0254, 0.0, 0.0, false, false, true,
            true, false},
        {main_edges, positive_y_fractions, positive_y_reduced_fractions, 0.0, 0.0, 0.0254, 0.0254, false, false, false,
            false, true},
        {outlet_edges, uniform_lower_fractions, uniform_lower_fractions, -0.0254, -0.0166, 0.0, 0.0, false, true, false,
            true, false},
        {outlet_edges, positive_y_reduced_fractions, positive_y_reduced_fractions, 0.0, 0.0, 0.0254, 0.0166, false,
            true, false, false, true}};

    NodeRegistry nodes;
    MeshData::ID next_element_id = 1;
    constexpr double zmin = -0.0005;
    constexpr double zmax = 0.0005;

    auto declare_side = [&](MeshData::ID element, unsigned side, const std::string& boundary)
    { boundaries.push_back({element, side, boundary}); };

    for (const auto& block : blocks)
    {
        const int x_cells = static_cast<int>(block.x_edges.size()) - 1;
        if (block.left_y_fractions.size() != block.right_y_fractions.size() || block.left_y_fractions.size() < 2)
        {
            throw std::logic_error("pitzDaily block grading requires matching wall-normal edge counts.");
        }
        const int y_cells = static_cast<int>(block.left_y_fractions.size()) - 1;
        const double x_left = block.x_edges.front();
        const double inverse_width = 1.0 / (block.x_edges.back() - block.x_edges.front());
        for (int i = 0; i < x_cells; ++i)
        {
            const double xa = block.x_edges[static_cast<size_t>(i)];
            const double xb = block.x_edges[static_cast<size_t>(i + 1)];
            const double fa = (xa - x_left) * inverse_width;
            const double fb = (xb - x_left) * inverse_width;
            const double lower_a = interpolate(block.lower_left, block.lower_right, fa);
            const double lower_b = interpolate(block.lower_left, block.lower_right, fb);
            const double upper_a = interpolate(block.upper_left, block.upper_right, fa);
            const double upper_b = interpolate(block.upper_left, block.upper_right, fb);

            for (int j = 0; j < y_cells; ++j)
            {
                const double fraction_a0 = interpolate(block.left_y_fractions[static_cast<size_t>(j)],
                    block.right_y_fractions[static_cast<size_t>(j)], fa);
                const double fraction_a1 = interpolate(block.left_y_fractions[static_cast<size_t>(j + 1)],
                    block.right_y_fractions[static_cast<size_t>(j + 1)], fa);
                const double fraction_b0 = interpolate(block.left_y_fractions[static_cast<size_t>(j)],
                    block.right_y_fractions[static_cast<size_t>(j)], fb);
                const double fraction_b1 = interpolate(block.left_y_fractions[static_cast<size_t>(j + 1)],
                    block.right_y_fractions[static_cast<size_t>(j + 1)], fb);
                const double ya0 = interpolate(lower_a, upper_a, fraction_a0);
                const double ya1 = interpolate(lower_a, upper_a, fraction_a1);
                const double yb0 = interpolate(lower_b, upper_b, fraction_b0);
                const double yb1 = interpolate(lower_b, upper_b, fraction_b1);

                const std::vector<MeshData::ID> element_nodes{nodes.id({xa, ya0, zmin}), nodes.id({xb, yb0, zmin}),
                    nodes.id({xb, yb1, zmin}), nodes.id({xa, ya1, zmin}), nodes.id({xa, ya0, zmax}),
                    nodes.id({xb, yb0, zmax}), nodes.id({xb, yb1, zmax}), nodes.id({xa, ya1, zmax})};
                const auto element = next_element_id++;
                MeshData::Cell cell;
                cell.id = element;
                cell.type = SimpleFluid::MeshUtils::CellType::HEXAHEDRON;
                cell.node_ids = element_nodes;
                cells.push_back(std::move(cell));

                if (block.lower_wall && j == 0)
                {
                    declare_side(element, 0, "lowerWall");
                }
                if (block.outlet && i + 1 == x_cells)
                {
                    declare_side(element, 1, "outlet");
                }
                if (block.upper_wall && j + 1 == y_cells)
                {
                    declare_side(element, 2, "upperWall");
                }
                if (block.inlet && i == 0)
                {
                    declare_side(element, 3, "inlet");
                }
                if (block.left_wall && i == 0)
                {
                    declare_side(element, 3, "lowerWall");
                }
                declare_side(element, 4, "frontAndBack");
                declare_side(element, 5, "frontAndBack");
            }
        }
    }

    std::vector<MeshData::Node> native_nodes;
    native_nodes.reserve(nodes.coordinates().size());
    for (const auto& [node_id, point] : nodes.coordinates())
        native_nodes.push_back({node_id, point});
    return SimpleFluid::detail::build_native_mesh_data(std::move(native_nodes), std::move(cells), boundaries,
        {"inlet", "outlet", "upperWall", "lowerWall", "frontAndBack"});
}
} // namespace SimpleFluid::detail::pitz_daily_mesh
