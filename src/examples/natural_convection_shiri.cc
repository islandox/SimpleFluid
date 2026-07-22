/**
 * @file natural_convection_shiri.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief MPI-capable Boussinesq counterpart to NC_Tutorial_Shiri.pdf.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "dataclass/Database.hh"
#include "geometry/MeshFactory.hh"
#include "solvers/BoussinesqSolver.hh"
#include "solvers/SolverProgress.hh"

#include <Tpetra_Core.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>

namespace
{

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
        return fallback;

    const int value = std::stoi(text);
    if (value <= 0)
        throw std::invalid_argument(std::string(name) + " must be positive.");
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
        return fallback;

    const double value = std::stod(text);
    if (!(value > 0.0) || !std::isfinite(value))
        throw std::invalid_argument(
            std::string(name) + " must be finite and positive.");
    return value;
}

/**
 * @brief Generate uniformly spaced coordinates over an interval.
 *
 * @param lower Lower interval endpoint.
 * @param upper Upper interval endpoint.
 * @param cells Number of cells.
 * @return Coordinate vector containing @p cells plus one edges.
 */
SimpleFluid::ArrReal uniform_edges(double lower, double upper, int cells)
{
    SimpleFluid::ArrReal edges;
    edges.reserve(static_cast<size_t>(cells) + 1);
    for (int edge = 0; edge <= cells; ++edge)
    {
        edges.push_back(
            lower + (upper - lower) * static_cast<double>(edge)
                  / static_cast<double>(cells));
    }
    return edges;
}

/**
 * @brief Generate radially graded annulus edges concentrated near the wall.
 *
 * @param cells Number of radial cells.
 * @return Radial edge coordinates from the inner to outer cylinder.
 */
SimpleFluid::ArrReal wall_graded_radial_edges(int cells)
{
    constexpr double inner_radius = 0.075;
    constexpr double outer_radius = 0.600;
    constexpr double grading = 3.0;

    SimpleFluid::ArrReal edges;
    edges.reserve(static_cast<size_t>(cells) + 1);
    edges.push_back(inner_radius);
    if (cells == 1)
    {
        edges.push_back(outer_radius);
        return edges;
    }
    const double ratio = std::pow(grading, 1.0 / (cells - 1));
    const double first_width = (outer_radius - inner_radius)
                             * (ratio - 1.0)
                             / (std::pow(ratio, cells) - 1.0);
    double width = first_width;
    for (int cell = 0; cell < cells; ++cell)
    {
        edges.push_back(edges.back() + width);
        width *= ratio;
    }
    edges.back() = outer_radius;
    return edges;
}

/**
 * @brief Write rank-local cylindrical solution profiles for comparison.
 *
 * @tparam Pack Tpetra type pack used by the mesh and solver.
 * @param mesh Mesh providing owned cell centers.
 * @param solver Solver providing temperature, velocity, and pressure fields.
 * @param rank MPI rank used in the output filename.
 * @throws std::runtime_error if the output file cannot be opened.
 */
template<class Pack>
void write_profile_cells(
    const SimpleFluid::MeshHandle<Pack>& mesh,
    const SimpleFluid::BoussinesqSolver<Pack>& solver,
    int rank)
{
    const char* configured_prefix = std::getenv("SIMPLEFLUID_SHIRI_OUTPUT_PREFIX");
    const std::string prefix = configured_prefix == nullptr
                             ? "simplefluid_cells"
                             : configured_prefix;
    const std::string filename =
        prefix + "_rank" + std::to_string(rank) + ".csv";
    std::ofstream output(filename);
    if (!output)
        throw std::runtime_error("Cannot open " + filename + " for writing.");

    output << "r,theta,z,temperature,ur,utheta,uz,pressure\n";
    output << std::setprecision(17);
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto lid = static_cast<typename Pack::local_ordinal_type>(owned);
        const auto center = mesh.cell_centroid(lid);
        const auto velocity = solver.velocity().value(lid);
        const double theta = std::atan2(center.y, center.x);
        const double cosine = std::cos(theta);
        const double sine = std::sin(theta);
        const double radial_velocity = velocity.x * cosine + velocity.y * sine;
        const double azimuthal_velocity = -velocity.x * sine + velocity.y * cosine;
        output << std::hypot(center.x, center.y) << ',' << theta << ','
               << center.z << ',' << solver.temperature().value(lid) << ','
               << radial_velocity << ',' << azimuthal_velocity << ','
               << velocity.z << ',' << solver.pressure().value(lid) << '\n';
    }
}

} // namespace

/**
 * @brief Run the configurable MPI annulus natural-convection case.
 *
 * @param argc Argument count passed to Tpetra.
 * @param argv Argument vector passed to Tpetra.
 * @return Process exit code, zero on normal completion.
 */
int main(int argc, char** argv)
{
    Tpetra::ScopeGuard tpetra_scope(&argc, &argv);
    using Pack = SimpleFluid::DefaultTpetraTypes;

    const int radial_cells =
        positive_environment_integer("SIMPLEFLUID_SHIRI_NR", 40);
    const int theta_cells =
        positive_environment_integer("SIMPLEFLUID_SHIRI_NTHETA", 20);
    const int axial_cells =
        positive_environment_integer("SIMPLEFLUID_SHIRI_NZ", 100);
    const int steps =
        positive_environment_integer("SIMPLEFLUID_SHIRI_STEPS", 200);
    const double time_step =
        positive_environment_real("SIMPLEFLUID_SHIRI_DT", 2.0e-3);

    auto database = std::make_shared<SimpleFluid::Database>();
    database->set("dimension", 3);
    database->set("mesh_size", SimpleFluid::real_t{1.0});
    database->set(
        "domain_type",
        static_cast<int>(SimpleFluid::MeshFactory::DomainType::ANNULUS));
    database->set("R", wall_graded_radial_edges(radial_cells));
    database->set(
        "Theta", uniform_edges(
            0.0, 0.25 * std::numbers::pi, theta_cells));
    database->set("Z", uniform_edges(0.0, 1.5, axial_cells));
    database->set(
        "domain_exterior_face_types",
        SimpleFluid::ArrString{
            "rmin", "rmax", "thetamin", "thetamax", "zmin", "zmax"});
    auto mesh = SimpleFluid::MeshFactory(database).build_handle<Pack>();

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["rmin"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 360.0};
    for (const auto* boundary : {"rmax", "zmin", "zmax"})
    {
        bcs.temperature[boundary] = {
            SimpleFluid::BoundaryConditionType::Dirichlet, 290.0};
        bcs.velocity[boundary] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }
    bcs.velocity["rmin"] = {
        SimpleFluid::BoundaryConditionType::NoSlip, {}};
    for (const auto* boundary : {"thetamin", "thetamax"})
    {
        bcs.temperature[boundary] = {
            SimpleFluid::BoundaryConditionType::Neumann, 0.0};
        bcs.velocity[boundary] = {
            SimpleFluid::BoundaryConditionType::Slip, {}};
    }

    constexpr double dynamic_viscosity = 1.8e-5;
    constexpr double reference_density = 1.198;
    constexpr double prandtl_number = 0.7;
    const double kinematic_viscosity =
        dynamic_viscosity / reference_density;
    const double thermal_diffusivity =
        kinematic_viscosity / prandtl_number;

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = time_step;
    time_options.steps = steps;
    time_options.thermal_diffusivity = thermal_diffusivity;
    time_options.kinematic_viscosity = kinematic_viscosity;
    time_options.thermal_expansion = 1.0 / 290.0;
    time_options.gravity_z = -9.81;
    time_options.reference_temperature = 290.0;
    time_options.n_pressure_correctors = 2;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 500;
    linear_options.tolerance = 1.0e-9;

    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.reference_density = reference_density;
    model_options.density = reference_density;
    model_options.specific_heat_capacity = 1000.0;
    model_options.dynamic_viscosity = dynamic_viscosity;
    model_options.thermal_conductivity =
        reference_density * 1000.0 * thermal_diffusivity;

    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, bcs, time_options, linear_options, model_options);
    solver.initialize_heated_box(290.0, 290.0);
    SimpleFluid::ProgressStream progress(std::cout);
    solver.run(steps, progress);

    const auto comm = Tpetra::getDefaultComm();
    const int rank = comm->getRank();
    solver.write_parallel_solution_vtu("natural_convection_shiri.vtu");
    write_profile_cells(*mesh, solver, rank);

    if (rank == 0)
    {
        std::cout << "Shiri annulus: " << radial_cells << 'x' << theta_cells
                  << 'x' << axial_cells << " cells, " << steps
                  << " steps, t=" << solver.time() << ", MPI ranks="
                  << comm->getSize() << '\n';
    }
    return 0;
}
