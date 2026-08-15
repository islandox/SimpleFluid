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
#include "solvers/BoussinesqSolver.hh"
#include "solvers/SolverProgress.hh"

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
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using Mesh = SimpleFluid::STKMesh<Pack>;
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
        throw std::invalid_argument(
            std::string(name) + " must be finite and positive.");
    }
    return value;
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
            lower + (upper - lower) * static_cast<double>(edge)
                  / static_cast<double>(cells);
    }
    edges.front() = lower;
    edges.back() = upper;
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
        const NodeKey key{
            std::llround(point.x * coordinate_scale),
            std::llround(point.y * coordinate_scale),
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

    const std::vector<std::pair<stk::mesh::EntityId, Point>>&
    coordinates() const noexcept
    {
        return d_coordinates;
    }

private:
    stk::mesh::EntityId d_next_id = 1;
    std::map<NodeKey, stk::mesh::EntityId> d_ids;
    std::vector<std::pair<stk::mesh::EntityId, Point>> d_coordinates;
};

/** @brief Structured block geometry and boundary-label specification. */
struct BlockSpec
{
    std::vector<double> x_edges;
    int y_cells = 1;
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

    auto& coordinates = meta->declare_field<double>(
        stk::topology::NODE_RANK, "coordinates");
    stk::mesh::put_field_on_mesh(
        coordinates, meta->universal_part(), 3, nullptr);
    meta->set_coordinate_field(&coordinates);

    auto& hex_part = meta->declare_part_with_topology(
        "pitzDaily_hexes", stk::topology::HEX_8);
    stk::io::put_io_part_attribute(hex_part);

    std::map<std::string, stk::mesh::Part*> boundary_parts;
    for (const auto* name : {
             "inlet", "outlet", "upperWall", "lowerWall", "frontAndBack"})
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

    const auto upstream_edges = uniform_edges(-0.0206, 0.0, upstream_x);
    const auto main_edges = uniform_edges(0.0, 0.206, main_x);
    const auto outlet_edges = uniform_edges(0.206, 0.290, outlet_x);
    const std::vector<BlockSpec> blocks{
        {upstream_edges, upper_y, 0.0, 0.0, 0.0254, 0.0254,
         true, false, false, true, true},
        {main_edges, lower_y, -0.0254, -0.0254, 0.0, 0.0,
         false, false, true, true, false},
        {main_edges, upper_y, 0.0, 0.0, 0.0254, 0.0254,
         false, false, false, false, true},
        {outlet_edges, lower_y, -0.0254, -0.0166, 0.0, 0.0,
         false, true, false, true, false},
        {outlet_edges, upper_y, 0.0, 0.0, 0.0254, 0.0166,
         false, true, false, false, true}};

    NodeRegistry nodes;
    stk::mesh::EntityId next_element_id = 1;
    constexpr double zmin = -0.0005;
    constexpr double zmax = 0.0005;

    auto declare_side = [&](stk::mesh::Entity element,
                            unsigned side,
                            const std::string& boundary)
    {
        stk::mesh::PartVector parts{boundary_parts.at(boundary)};
        bulk->declare_element_side(element, side, parts);
    };

    bulk->modification_begin();
    for (const auto& block : blocks)
    {
        const int x_cells = static_cast<int>(block.x_edges.size()) - 1;
        const double x_left = block.x_edges.front();
        const double inverse_width =
            1.0 / (block.x_edges.back() - block.x_edges.front());
        for (int i = 0; i < x_cells; ++i)
        {
            const double xa = block.x_edges[static_cast<size_t>(i)];
            const double xb = block.x_edges[static_cast<size_t>(i + 1)];
            const double fa = (xa - x_left) * inverse_width;
            const double fb = (xb - x_left) * inverse_width;
            const double lower_a = interpolate(
                block.lower_left, block.lower_right, fa);
            const double lower_b = interpolate(
                block.lower_left, block.lower_right, fb);
            const double upper_a = interpolate(
                block.upper_left, block.upper_right, fa);
            const double upper_b = interpolate(
                block.upper_left, block.upper_right, fb);

            for (int j = 0; j < block.y_cells; ++j)
            {
                const double ya0 = interpolate(
                    lower_a, upper_a,
                    static_cast<double>(j) / block.y_cells);
                const double ya1 = interpolate(
                    lower_a, upper_a,
                    static_cast<double>(j + 1) / block.y_cells);
                const double yb0 = interpolate(
                    lower_b, upper_b,
                    static_cast<double>(j) / block.y_cells);
                const double yb1 = interpolate(
                    lower_b, upper_b,
                    static_cast<double>(j + 1) / block.y_cells);

                const stk::mesh::EntityIdVector element_nodes{
                    nodes.id({xa, ya0, zmin}),
                    nodes.id({xb, yb0, zmin}),
                    nodes.id({xb, yb1, zmin}),
                    nodes.id({xa, ya1, zmin}),
                    nodes.id({xa, ya0, zmax}),
                    nodes.id({xb, yb0, zmax}),
                    nodes.id({xb, yb1, zmax}),
                    nodes.id({xa, ya1, zmax})};
                const auto element = stk::mesh::declare_element(
                    *bulk, hex_part, next_element_id++, element_nodes);

                if (block.lower_wall && j == 0)
                {
                    declare_side(element, 0, "lowerWall");
                }
                if (block.outlet && i + 1 == x_cells)
                {
                    declare_side(element, 1, "outlet");
                }
                if (block.upper_wall && j + 1 == block.y_cells)
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
            throw std::runtime_error(
                "pitzDaily mesh failed to assign node coordinates.");
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
    conditions.turbulence.turbulent_kinetic_energy["inlet"] = {
        Type::Dirichlet, 0.375};
    conditions.turbulence.dissipation_rate["inlet"] = {
        Type::Dirichlet, 14.855};

    for (const auto* boundary : {"upperWall", "lowerWall"})
    {
        conditions.velocity[boundary] = {Type::NoSlip, {}};
        conditions.pressure[boundary] = {Type::Neumann, 0.0};
        conditions.turbulence.turbulent_kinetic_energy[boundary] = {
            Type::Neumann, 0.0};
        conditions.turbulence.dissipation_rate[boundary] = {
            Type::Neumann, 0.0};
    }

    conditions.velocity["frontAndBack"] = {Type::Slip, {}};
    conditions.pressure["frontAndBack"] = {Type::Neumann, 0.0};
    conditions.turbulence.turbulent_kinetic_energy["frontAndBack"] = {
        Type::Neumann, 0.0};
    conditions.turbulence.dissipation_rate["frontAndBack"] = {
        Type::Neumann, 0.0};
    conditions.turbulence.turbulent_kinetic_energy["outlet"] = {
        Type::Neumann, 0.0};
    conditions.turbulence.dissipation_rate["outlet"] = {
        Type::Neumann, 0.0};
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
void write_cells(
    const Mesh& mesh,
    const SimpleFluid::BoussinesqSolver<Pack>& solver,
    const SimpleFluid::TurbulenceModel<Pack>& turbulence,
    int rank)
{
    const char* configured_prefix =
        std::getenv("SIMPLEFLUID_PITZ_OUTPUT_PREFIX");
    const std::string prefix = configured_prefix == nullptr
                             ? "simplefluid_cells"
                             : configured_prefix;
    const std::string filename =
        prefix + "_rank" + std::to_string(rank) + ".csv";
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
        output << center.x << ',' << center.y << ',' << center.z << ','
               << velocity.x << ',' << velocity.y << ',' << velocity.z << ','
               << solver.pressure().value(lid) << ','
               << turbulence.turbulent_kinetic_energy().value(lid) << ','
               << epsilon->value(lid) << ','
               << turbulence.turbulent_kinematic_viscosity().value(lid)
               << '\n';
    }
}

} // namespace

/**
 * @brief Run the configurable turbulent pitzDaily comparison case.
 *
 * @param argc Argument count passed to Tpetra.
 * @param argv Argument vector passed to Tpetra.
 * @return Process exit code, zero on normal completion.
 */
int main(int argc, char** argv)
{
    Tpetra::ScopeGuard tpetra_scope(&argc, &argv);
    const int mesh_divisor = positive_environment_integer(
        "SIMPLEFLUID_PITZ_MESH_DIVISOR", 4);
    const int steps = positive_environment_integer(
        "SIMPLEFLUID_PITZ_STEPS", 200);
    const double time_step = positive_environment_real(
        "SIMPLEFLUID_PITZ_DT", 1.0e-5);

    auto mesh = make_pitz_daily_mesh(mesh_divisor);
    auto boundary_conditions = pitz_daily_boundary_conditions();

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = time_step;
    time_options.steps = steps;
    time_options.thermal_diffusivity = 0.0;
    time_options.kinematic_viscosity = 1.0e-5;
    time_options.thermal_expansion = 0.0;
    time_options.gravity_x = 0.0;
    time_options.gravity_y = 0.0;
    time_options.gravity_z = 0.0;
    time_options.pressure_velocity_coupling =
        SimpleFluid::PressureVelocityCoupling::SIMPLE;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 500;
    linear_options.tolerance = 1.0e-9;

    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.reference_density = 1.0;
    model_options.density = 1.0;
    model_options.specific_heat_capacity = 1.0;
    model_options.dynamic_viscosity = 1.0e-5;
    model_options.thermal_conductivity = 0.0;

    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh,
        std::move(boundary_conditions),
        time_options,
        linear_options,
        model_options);
    solver.initialize_heated_box(0.0, 0.0);

    SimpleFluid::TurbulenceModelOptions turbulence_options;
    turbulence_options.model =
        SimpleFluid::TurbulenceModelType::StandardKEpsilon;
    turbulence_options.initial_turbulent_kinetic_energy = 0.375;
    turbulence_options.initial_dissipation_rate = 14.855;
    turbulence_options.wall_treatment =
        SimpleFluid::TurbulenceWallTreatmentType::StandardHighReKEpsilon;
    turbulence_options.wall_options.boundary_names = {
        "upperWall", "lowerWall"};
    auto& turbulence = solver.configure_turbulence(turbulence_options);

    SimpleFluid::ProgressStream progress(std::cout);
    solver.run(steps, progress);

    const auto comm = Tpetra::getDefaultComm();
    const int rank = comm->getRank();
    SimpleFluid::SolutionOutputOptions output_options;
    output_options.include_turbulence_fields = true;
    solver.write_parallel_solution_vtu("pitz_daily.vtu", output_options);
    write_cells(*mesh, solver, turbulence, rank);

    if (rank == 0)
    {
        std::cout << "pitzDaily: " << mesh->num_owned_cells()
                  << " rank-zero owned cells, divisor=" << mesh_divisor
                  << ", steps=" << steps << ", t=" << solver.time()
                  << ", MPI ranks=" << comm->getSize() << '\n';
    }
    return 0;
}
