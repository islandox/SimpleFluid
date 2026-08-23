/**
 * @file pitz_daily.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Transient standard-k-epsilon counterpart to OpenFOAM pitzDaily.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "equations/turbulence/TurbulenceModel.hh"
#include "geometry/STKMesh.hh"
#include "solvers/IncompressibleIsothermalSolver.hh"
#include "solvers/SolverProgress.hh"
#include "solvers/SteadyStateSearch.hh"

#include <Tpetra_Core.hpp>
#include <stk_io/IossBridge.hpp>
#include <stk_mesh/base/FEMHelpers.hpp>
#include <stk_mesh/base/FieldBase.hpp>

#include <algorithm>
#include <cmath>
#include <compare>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using Mesh = SimpleFluid::STKMesh<Pack>;
using Solver = SimpleFluid::IncompressibleIsothermalSolver<Pack>;
using Point = SimpleFluid::MeshUtils::Vec3;

/**
 * @brief Read a positive integer from an environment variable.
 *
 * @param name Environment-variable name.
 * @param fallback Value used when the variable is unset.
 * @return Parsed positive value or @p fallback.
 * @throws std::invalid_argument if the configured value is invalid or non-positive.
 * @throws std::out_of_range if the configured value exceeds the integer range.
 */
int positive_environment_integer(const char* name, int fallback)
{
    const char* text = std::getenv(name);
    if (text == nullptr)
    {
        return fallback;
    }

    const int value = std::stoi(text);
    if (value <= 0)
    {
        throw std::invalid_argument(std::string(name) + " must be positive.");
    }
    return value;
}

/** @brief Read a non-negative integer from an environment variable. */
int non_negative_environment_integer(const char* name, int fallback)
{
    const char* text = std::getenv(name);
    if (text == nullptr)
    {
        return fallback;
    }

    const int value = std::stoi(text);
    if (value < 0)
    {
        throw std::invalid_argument(std::string(name) + " cannot be negative.");
    }
    return value;
}

/**
 * @brief Read a finite positive floating-point value from the environment.
 *
 * @param name Environment-variable name.
 * @param fallback Value used when the variable is unset.
 * @return Parsed positive value or @p fallback.
 * @throws std::invalid_argument if the configured value is invalid, non-finite, or non-positive.
 * @throws std::out_of_range if the configured value exceeds the floating-point range.
 */
double positive_environment_real(const char* name, double fallback)
{
    const char* text = std::getenv(name);
    if (text == nullptr)
    {
        return fallback;
    }

    const double value = std::stod(text);
    if (!(value > 0.0) || !std::isfinite(value))
    {
        throw std::invalid_argument(std::string(name) + " must be finite and positive.");
    }
    return value;
}

/** @brief Read a strict boolean switch from an environment variable. */
bool environment_boolean(const char* name, bool fallback)
{
    const char* text = std::getenv(name);
    if (text == nullptr)
    {
        return fallback;
    }

    const std::string_view value{text};
    if (value == "1" || value == "true" || value == "yes" || value == "on")
    {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off")
    {
        return false;
    }
    throw std::invalid_argument(std::string(name) + " must be one of 0/1, false/true, no/yes, or off/on.");
}

/** @brief Identify a failed momentum predictor that is safe to retry. */
bool retryable_steady_state_failure(const std::runtime_error& error)
{
    const std::string_view message{error.what()};
    return message.find("IncompressibleMomentumEquation") != std::string_view::npos &&
           message.find("did not converge") != std::string_view::npos;
}

/**
 * @brief Scale a tutorial cell count while retaining at least one cell.
 *
 * @param tutorial_cells Cell count in the reference tutorial.
 * @param divisor Requested coarsening divisor.
 * @return Rounded-up scaled cell count.
 */
int scaled_cells(int tutorial_cells, int divisor)
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
std::vector<double> uniform_edges(double lower, double upper, int cells)
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
std::vector<double> graded_edges(double lower, double upper, int cells, double expansion_ratio)
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
std::vector<double> multi_graded_fractions(int cells, const std::vector<GradingSection>& sections)
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

    std::vector<double> target_counts;
    std::vector<int> section_counts;
    target_counts.reserve(sections.size());
    section_counts.reserve(sections.size());
    int allocated_cells = 0;
    for (const auto& section : sections)
    {
        const double target = static_cast<double>(cells) * section.cell_fraction / total_cell_fraction;
        target_counts.push_back(target);
        section_counts.push_back(std::max(1, static_cast<int>(std::floor(target))));
        allocated_cells += section_counts.back();
    }
    while (allocated_cells < cells)
    {
        size_t selected = 0;
        double largest_deficit = -std::numeric_limits<double>::infinity();
        for (size_t section = 0; section < sections.size(); ++section)
        {
            const double deficit = target_counts[section] - section_counts[section];
            if (deficit >= largest_deficit)
            {
                selected = section;
                largest_deficit = deficit;
            }
        }
        ++section_counts[selected];
        ++allocated_cells;
    }
    while (allocated_cells > cells)
    {
        size_t selected = 0;
        double largest_excess = -std::numeric_limits<double>::infinity();
        for (size_t section = 0; section < sections.size(); ++section)
        {
            if (section_counts[section] == 1)
            {
                continue;
            }
            const double excess = section_counts[section] - target_counts[section];
            if (excess > largest_excess)
            {
                selected = section;
                largest_excess = excess;
            }
        }
        --section_counts[selected];
        --allocated_cells;
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

/** @brief Quantized coordinates used to deduplicate STK mesh nodes. */
struct NodeKey
{
    std::int64_t x = 0;
    std::int64_t y = 0;
    std::int64_t z = 0;

    auto operator<=>(const NodeKey&) const = default;
};

/** @brief Assign stable STK identifiers to unique geometric nodes. */
class NodeRegistry
{
public:
    stk::mesh::EntityId id(const Point& point)
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

    const std::vector<std::pair<stk::mesh::EntityId, Point>>& coordinates() const noexcept { return d_coordinates; }

private:
    stk::mesh::EntityId d_next_id = 1;
    std::map<NodeKey, stk::mesh::EntityId> d_ids;
    std::vector<std::pair<stk::mesh::EntityId, Point>> d_coordinates;
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
double interpolate(double left, double right, double fraction)
{
    return left + fraction * (right - left);
}

/**
 * @brief Build a coarsened multi-block STK mesh for the pitzDaily geometry.
 *
 * @param divisor Coarsening divisor applied to tutorial cell counts.
 * @return Assembled mesh with inlet, outlet, wall, and spanwise boundaries.
 * @throws std::runtime_error if node coordinates cannot be assigned.
 */
SimpleFluid::SP<Mesh> make_pitz_daily_mesh(int divisor)
{
    auto mesh = std::make_shared<Mesh>();
    auto meta = mesh->meta();
    auto bulk = mesh->bulk();

    auto& coordinates = meta->declare_field<double>(stk::topology::NODE_RANK, "coordinates");
    stk::mesh::put_field_on_mesh(coordinates, meta->universal_part(), 3, nullptr);
    meta->set_coordinate_field(&coordinates);

    auto& hex_part = meta->declare_part_with_topology("pitzDaily_hexes", stk::topology::HEX_8);
    stk::io::put_io_part_attribute(hex_part);

    std::map<std::string, stk::mesh::Part*> boundary_parts;
    for (const auto* name : {"inlet", "outlet", "upperWall", "lowerWall", "frontAndBack"})
    {
        auto& part = meta->declare_part(name, meta->side_rank());
        stk::io::put_io_part_attribute(part);
        boundary_parts.emplace(name, &part);
    }

    const int upstream_x = scaled_cells(18, divisor);
    const int main_x = scaled_cells(180, divisor);
    const int outlet_x = scaled_cells(25, divisor);
    const int upper_y = scaled_cells(30, divisor);
    const int lower_y = scaled_cells(27, divisor);

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
    stk::mesh::EntityId next_element_id = 1;
    constexpr double zmin = -0.0005;
    constexpr double zmax = 0.0005;

    auto declare_side = [&](stk::mesh::Entity element, unsigned side, const std::string& boundary)
    {
        stk::mesh::PartVector parts{boundary_parts.at(boundary)};
        bulk->declare_element_side(element, side, parts);
    };

    bulk->modification_begin();
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

                const stk::mesh::EntityIdVector element_nodes{nodes.id({xa, ya0, zmin}), nodes.id({xb, yb0, zmin}),
                    nodes.id({xb, yb1, zmin}), nodes.id({xa, ya1, zmin}), nodes.id({xa, ya0, zmax}),
                    nodes.id({xb, yb0, zmax}), nodes.id({xb, yb1, zmax}), nodes.id({xa, ya1, zmax})};
                const auto element = stk::mesh::declare_element(*bulk, hex_part, next_element_id++, element_nodes);

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

    for (const auto& [node_id, point] : nodes.coordinates())
    {
        const auto node = bulk->get_entity(stk::topology::NODE_RANK, node_id);
        double* values = stk::mesh::field_data(coordinates, node);
        if (values == nullptr)
        {
            throw std::runtime_error("pitzDaily mesh failed to assign node coordinates.");
        }
        values[0] = point.x;
        values[1] = point.y;
        values[2] = point.z;
    }
    bulk->modification_end();
    mesh->assemble();
    return mesh;
}

/**
 * @brief Construct velocity, pressure, and turbulence boundary conditions.
 *
 * @return Boundary conditions matching the OpenFOAM pitzDaily case.
 */
SimpleFluid::BoundaryConditionSet pitz_daily_boundary_conditions()
{
    using Type = SimpleFluid::BoundaryConditionType;
    SimpleFluid::BoundaryConditionSet conditions;
    conditions.velocity["inlet"] = {Type::Dirichlet, {10.0, 0.0, 0.0}};
    conditions.velocity["outlet"] = {Type::Neumann, {}};
    conditions.pressure["inlet"] = {Type::Neumann, 0.0};
    conditions.pressure["outlet"] = {Type::Dirichlet, 0.0};
    conditions.turbulence.turbulent_kinetic_energy["inlet"] = {Type::Dirichlet, 0.375};
    conditions.turbulence.dissipation_rate["inlet"] = {Type::Dirichlet, 14.855};

    for (const auto* boundary : {"upperWall", "lowerWall"})
    {
        conditions.velocity[boundary] = {Type::NoSlip, {}};
        conditions.pressure[boundary] = {Type::Neumann, 0.0};
        conditions.turbulence.turbulent_kinetic_energy[boundary] = {Type::Neumann, 0.0};
        conditions.turbulence.dissipation_rate[boundary] = {Type::Neumann, 0.0};
    }

    conditions.velocity["frontAndBack"] = {Type::Slip, {}};
    conditions.pressure["frontAndBack"] = {Type::Neumann, 0.0};
    conditions.turbulence.turbulent_kinetic_energy["frontAndBack"] = {Type::Neumann, 0.0};
    conditions.turbulence.dissipation_rate["frontAndBack"] = {Type::Neumann, 0.0};
    conditions.turbulence.turbulent_kinetic_energy["outlet"] = {Type::Neumann, 0.0};
    conditions.turbulence.dissipation_rate["outlet"] = {Type::Neumann, 0.0};
    return conditions;
}

/**
 * @brief Write rank-local flow and turbulence cell data to CSV.
 *
 * @param mesh Mesh providing owned cell centers.
 * @param solver Solver providing velocity and pressure fields.
 * @param turbulence Turbulence model providing k, epsilon, and eddy viscosity.
 * @param rank MPI rank used in the output filename.
 * @throws std::runtime_error if output cannot be opened or epsilon is unavailable.
 */
void write_cells(const Mesh& mesh, const Solver& solver, const Solver::turbulence_model_type& turbulence, int rank)
{
    const char* configured_prefix = std::getenv("SIMPLEFLUID_PITZ_OUTPUT_PREFIX");
    const std::string prefix = configured_prefix == nullptr ? "simplefluid_cells" : configured_prefix;
    const std::string filename = prefix + "_rank" + std::to_string(rank) + ".csv";
    std::ofstream output(filename);
    if (!output)
    {
        throw std::runtime_error("Cannot open " + filename + " for writing.");
    }

    const auto* epsilon = turbulence.dissipation_rate();
    if (epsilon == nullptr)
    {
        throw std::runtime_error("pitzDaily requires an epsilon-family model.");
    }
    output << "x,y,z,ux,uy,uz,pressure,k,epsilon,nu_t\n";
    output << std::setprecision(17);
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(owned);
        const auto center = mesh.cell_centroid(lid);
        const auto velocity = solver.velocity().value(lid);
        output << center.x << ',' << center.y << ',' << center.z << ',' << velocity.x << ',' << velocity.y << ','
               << velocity.z << ',' << solver.pressure().value(lid) << ','
               << turbulence.turbulent_kinetic_energy().value(lid) << ',' << epsilon->value(lid) << ','
               << turbulence.turbulent_kinematic_viscosity().value(lid) << '\n';
    }
}

} // namespace

/**
 * @brief Run the configurable turbulent pitzDaily comparison case.
 *
 * @param argc Argument count passed to Tpetra.
 * @param argv Argument vector passed to Tpetra.
 * @return Process exit code, zero only when the requested run completes or
 *         reaches its configured steady-state criterion.
 */
int main(int argc, char** argv)
{
    Tpetra::ScopeGuard tpetra_scope(&argc, &argv);
    const int mesh_divisor = positive_environment_integer("SIMPLEFLUID_PITZ_MESH_DIVISOR", 4);
    const int steps = positive_environment_integer("SIMPLEFLUID_PITZ_STEPS", 200);
    const double time_step = positive_environment_real("SIMPLEFLUID_PITZ_DT", 1.0e-5);
    const bool search_for_steady_state = environment_boolean("SIMPLEFLUID_PITZ_STEADY_STATE", false);

    auto mesh = make_pitz_daily_mesh(mesh_divisor);
    auto boundary_conditions = pitz_daily_boundary_conditions();

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = time_step;
    time_options.steps = steps;
    time_options.kinematic_viscosity = 1.0e-5;
    time_options.pressure_velocity_coupling = SimpleFluid::PressureVelocityCoupling::SIMPLE;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 500;
    linear_options.tolerance =
        positive_environment_real("SIMPLEFLUID_PITZ_LINEAR_TOLERANCE", 1.0e-9);

    Solver solver(mesh, std::move(boundary_conditions), time_options, linear_options, 1.0);

    SimpleFluid::TurbulenceModelOptions turbulence_options;
    turbulence_options.model = SimpleFluid::TurbulenceModelType::StandardKEpsilon;
    turbulence_options.initial_turbulent_kinetic_energy = 0.375;
    turbulence_options.initial_dissipation_rate = 14.855;
    turbulence_options.wall_treatment = SimpleFluid::TurbulenceWallTreatmentType::StandardHighReKEpsilon;
    turbulence_options.wall_options.boundary_names = {"upperWall", "lowerWall"};
    auto& turbulence = solver.configure_turbulence(turbulence_options);

    const auto comm = Tpetra::getDefaultComm();
    const int rank = comm->getRank();
    bool steady_state_reached = false;
    std::optional<std::string> steady_state_failure;
    int rejected_steady_steps = 0;
    std::optional<SimpleFluid::SteadyStateStepStatistics<SimpleFluid::real_t>> final_steady_statistics;
    std::optional<SimpleFluid::real_t> last_requested_transport_linear_tolerance;
    if (!search_for_steady_state)
    {
        SimpleFluid::ProgressStream progress(std::cout);
        solver.run(steps, progress);
    }
    else
    {
        SimpleFluid::SteadyStateSearchOptions steady_options;
        steady_options.maximum_steps = steps;
        steady_options.required_consecutive_steps =
            positive_environment_integer("SIMPLEFLUID_PITZ_STEADY_CONSECUTIVE_STEPS", std::min(5, steps));
        steady_options.minimum_steps = positive_environment_integer("SIMPLEFLUID_PITZ_STEADY_MIN_STEPS",
            std::max(1, std::min(20, steps - steady_options.required_consecutive_steps + 1)));
        steady_options.maximum_retries_per_step =
            non_negative_environment_integer("SIMPLEFLUID_PITZ_STEADY_MAX_RETRIES", 4);
        steady_options.rejection_recovery_steps =
            non_negative_environment_integer("SIMPLEFLUID_PITZ_STEADY_REJECTION_RECOVERY_STEPS", 5);
        steady_options.relative_update_tolerance =
            positive_environment_real("SIMPLEFLUID_PITZ_STEADY_TOLERANCE", 1.0e-4);
        steady_options.minimum_time_step =
            positive_environment_real("SIMPLEFLUID_PITZ_STEADY_MIN_DT", time_step / 16.0);
        steady_options.maximum_time_step =
            positive_environment_real("SIMPLEFLUID_PITZ_STEADY_MAX_DT", std::max(time_step, 5.0e-2));
        steady_options.target_courant_number = positive_environment_real("SIMPLEFLUID_PITZ_STEADY_TARGET_COURANT", 0.8);
        steady_options.time_step_growth_factor = positive_environment_real("SIMPLEFLUID_PITZ_STEADY_DT_GROWTH", 1.5);
        steady_options.time_step_reduction_factor =
            positive_environment_real("SIMPLEFLUID_PITZ_STEADY_DT_REDUCTION", 0.5);
        steady_options.rejection_time_step_safety_factor =
            positive_environment_real("SIMPLEFLUID_PITZ_STEADY_REJECTION_SAFETY", 0.9);
        SimpleFluid::AdaptiveLinearToleranceOptions adaptive_linear_options;
        adaptive_linear_options.final_tolerance = linear_options.tolerance;
        adaptive_linear_options.relaxed_tolerance = positive_environment_real(
            "SIMPLEFLUID_PITZ_STEADY_RELAXED_LINEAR_TOLERANCE",
            std::max(1.0e-6, adaptive_linear_options.final_tolerance));
        adaptive_linear_options.full_accuracy_update_ratio = positive_environment_real(
            "SIMPLEFLUID_PITZ_STEADY_FULL_ACCURACY_UPDATE_RATIO", 10.0);
        const int progress_interval = positive_environment_integer("SIMPLEFLUID_PITZ_STEADY_PROGRESS_INTERVAL", 1);

        SimpleFluid::AdaptiveSteadyStateController controller(steady_options, time_step);
        SimpleFluid::AdaptiveLinearToleranceController linear_tolerance_controller(
            adaptive_linear_options, steady_options.relative_update_tolerance);
        auto adaptive_transport_linear_options = solver.linear_solver_options();
        adaptive_transport_linear_options.tolerance =
            linear_tolerance_controller.current_linear_tolerance();
        solver.set_linear_solver_options(adaptive_transport_linear_options);
        SimpleFluid::SteadyStateFieldMonitor<Pack> monitor(
            solver.velocity().mesh_ptr(), 0.0, steady_options.update_scales);
        const auto* epsilon = turbulence.dissipation_rate();
        if (epsilon == nullptr)
        {
            throw std::logic_error("Steady pitzDaily search requires a dissipation-rate turbulence field.");
        }
        monitor.initialize(solver.velocity(), {&turbulence.turbulent_kinetic_energy(), epsilon});
        SimpleFluid::SteadyStateProgressStream progress(std::cout);

        if (rank == 0)
        {
            std::cout << "steady_state_search: enabled=yes max_steps=" << steady_options.maximum_steps
                      << " tolerance=" << steady_options.relative_update_tolerance
                      << " min_steps=" << steady_options.minimum_steps
                      << " consecutive_steps=" << steady_options.required_consecutive_steps
                      << " max_retries=" << steady_options.maximum_retries_per_step
                      << " rejection_recovery_steps=" << steady_options.rejection_recovery_steps
                      << " rejection_safety=" << steady_options.rejection_time_step_safety_factor
                      << " target_Co=" << steady_options.target_courant_number << " dt_range=["
                      << steady_options.minimum_time_step << ',' << steady_options.maximum_time_step << ']'
                      << " transport_linear_tolerance=" << adaptive_linear_options.relaxed_tolerance << "->"
                      << adaptive_linear_options.final_tolerance
                      << " full_accuracy_update_ratio=" << adaptive_linear_options.full_accuracy_update_ratio
                      << " pressure_linear_tolerance=" << solver.pressure_linear_solver_options().tolerance << '\n';
        }

        for (int iteration = 0; iteration < steady_options.maximum_steps; ++iteration)
        {
            const auto requested_transport_linear_tolerance =
                linear_tolerance_controller.current_linear_tolerance();
            const bool steady_sample_eligible =
                linear_tolerance_controller.full_accuracy_requested();
            SimpleFluid::real_t accepted_time_step{};
            bool accepted = false;
            int retries = 0;
            while (!accepted)
            {
                accepted_time_step = solver.time_step();
                try
                {
                    solver.step();
                    accepted = true;
                }
                catch (const std::runtime_error& error)
                {
                    if (!retryable_steady_state_failure(error))
                    {
                        throw;
                    }

                    ++rejected_steady_steps;
                    const auto reduced_time_step = controller.rejected_time_step(accepted_time_step);
                    if (rank == 0)
                    {
                        progress.write_retry(iteration + 1,
                            std::min(retries + 1, steady_options.maximum_retries_per_step),
                            steady_options.maximum_retries_per_step, solver.time(), accepted_time_step,
                            reduced_time_step, error.what());
                    }
                    if (retries >= steady_options.maximum_retries_per_step || !(reduced_time_step < accepted_time_step))
                    {
                        steady_state_failure = error.what();
                        break;
                    }
                    ++retries;
                    solver.set_time_step(reduced_time_step);
                }
            }
            if (!accepted)
            {
                break;
            }

            const auto update_rates = monitor.observe(accepted_time_step);
            const bool solver_converged = solver.last_step_statistics().converged;
            const auto next_requested_transport_linear_tolerance =
                linear_tolerance_controller.observe(
                    static_cast<SimpleFluid::real_t>(update_rates.maximum()), solver_converged);
            const auto statistics =
                controller.observe(solver.time(), accepted_time_step, solver.maximum_courant_number(),
                    {static_cast<SimpleFluid::real_t>(update_rates.velocity),
                        static_cast<SimpleFluid::real_t>(update_rates.temperature),
                        static_cast<SimpleFluid::real_t>(update_rates.turbulence)},
                    solver_converged, steady_sample_eligible);
            final_steady_statistics = statistics;
            last_requested_transport_linear_tolerance = requested_transport_linear_tolerance;
            adaptive_transport_linear_options.tolerance =
                next_requested_transport_linear_tolerance;
            solver.set_linear_solver_options(adaptive_transport_linear_options);
            if (rank == 0 &&
                (statistics.iteration == 1 || statistics.steady || statistics.iteration % progress_interval == 0))
            {
                progress.write(statistics, solver.last_step_statistics(),
                    requested_transport_linear_tolerance,
                    next_requested_transport_linear_tolerance);
            }
            if (statistics.steady)
            {
                steady_state_reached = true;
                break;
            }
            solver.set_time_step(statistics.next_time_step);
        }
    }

    SimpleFluid::SolutionOutputOptions output_options;
    output_options.include_turbulence_fields = true;
    solver.write_parallel_solution_vtu("pitz_daily.vtu", output_options);
    write_cells(*mesh, solver, turbulence, rank);

    if (rank == 0)
    {
        std::cout << "pitzDaily: " << mesh->num_owned_cells() << " rank-zero owned cells, divisor=" << mesh_divisor
                  << ", steps=" << solver.step_index() << ", t=" << solver.time() << ", MPI ranks=" << comm->getSize()
                  << '\n';
        if (search_for_steady_state)
        {
            std::cout << "steady_state_search: reached=" << (steady_state_reached ? "yes" : "no")
                      << ", rejected_steps=" << rejected_steady_steps;
            if (final_steady_statistics)
            {
                const auto& statistics = *final_steady_statistics;
                std::cout << ", steps=" << statistics.iteration
                          << ", final_update_rate=" << statistics.update_rates.maximum()
                          << ", final_max_Co=" << statistics.maximum_courant_number
                          << ", final_dt=" << statistics.time_step;
                if (last_requested_transport_linear_tolerance)
                {
                    std::cout << ", last_transport_linear_tolerance="
                              << *last_requested_transport_linear_tolerance;
                }
            }
            else
            {
                std::cout << ", steps=0";
            }
            if (steady_state_failure)
            {
                std::cout << ", failure=\"" << *steady_state_failure << '\"';
            }
            std::cout << '\n';
        }
    }
    return search_for_steady_state && !steady_state_reached ? EXIT_FAILURE : EXIT_SUCCESS;
}
