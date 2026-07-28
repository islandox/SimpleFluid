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
#include "solvers/SteadyStateSearch.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

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
 * @brief Read a non-negative integer from an environment variable.
 */
int non_negative_environment_integer(const char* name, int fallback)
{
    const char* text = std::getenv(name);
    if (text == nullptr)
        return fallback;

    const int value = std::stoi(text);
    if (value < 0)
        throw std::invalid_argument(
            std::string(name) + " cannot be negative.");
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
 * @brief Read a strict boolean switch from an environment variable.
 */
bool environment_boolean(const char* name, bool fallback)
{
    const char* text = std::getenv(name);
    if (text == nullptr)
        return fallback;

    const std::string_view value{text};
    if (value == "1" || value == "true" || value == "yes"
        || value == "on")
    {
        return true;
    }
    if (value == "0" || value == "false" || value == "no"
        || value == "off")
    {
        return false;
    }
    throw std::invalid_argument(
        std::string(name)
        + " must be one of 0/1, false/true, no/yes, or off/on.");
}

/**
 * @brief Identify a failed momentum predictor that is safe to retry.
 *
 * The momentum equation solves into a candidate and publishes velocity only
 * after convergence. A rejected predictor therefore leaves every accepted
 * primary and turbulence field unchanged.
 */
bool retryable_steady_state_failure(const std::runtime_error& error)
{
    const std::string_view message{error.what()};
    return message.find("IncompressibleMomentumEquation")
               != std::string_view::npos
        && message.find("did not converge")
               != std::string_view::npos;
}

/**
 * @brief Read a Belos backend name from an environment variable.
 */
SimpleFluid::LinearSolverBackend environment_linear_solver_backend(
    const char* name,
    SimpleFluid::LinearSolverBackend fallback)
{
    const char* text = std::getenv(name);
    return text == nullptr
        ? fallback
        : SimpleFluid::parse_linear_solver_backend(text);
}

/**
 * @brief Read a linear preconditioner name from an environment variable.
 */
SimpleFluid::LinearPreconditioner environment_linear_preconditioner(
    const char* name,
    SimpleFluid::LinearPreconditioner fallback)
{
    const char* text = std::getenv(name);
    return text == nullptr
        ? fallback
        : SimpleFluid::parse_linear_preconditioner(text);
}

/**
 * @brief Read a pressure-velocity coupling mode from the environment.
 */
SimpleFluid::PressureVelocityCoupling
environment_pressure_velocity_coupling(
    const char* name,
    SimpleFluid::PressureVelocityCoupling fallback)
{
    const char* text = std::getenv(name);
    return text == nullptr
        ? fallback
        : SimpleFluid::pressure_velocity_coupling_from_string(text);
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
 * @brief Generate geometrically graded coordinates over one mesh block.
 *
 * @param lower Lower interval endpoint.
 * @param upper Upper interval endpoint.
 * @param cells Number of cells.
 * @param expansion_ratio Last-cell width divided by first-cell width.
 * @return Coordinate vector containing @p cells plus one edges.
 */
SimpleFluid::ArrReal graded_edges(
    double lower, double upper, int cells, double expansion_ratio)
{
    SimpleFluid::ArrReal edges;
    edges.reserve(static_cast<size_t>(cells) + 1);
    edges.push_back(lower);
    if (cells == 1)
    {
        edges.push_back(upper);
        return edges;
    }

    if (expansion_ratio == 1.0)
        return uniform_edges(lower, upper, cells);

    const double ratio =
        std::pow(expansion_ratio, 1.0 / static_cast<double>(cells - 1));
    const double first_width =
        (upper - lower) * (ratio - 1.0)
        / (std::pow(ratio, cells) - 1.0);
    double width = first_width;
    for (int cell = 0; cell < cells; ++cell)
    {
        edges.push_back(edges.back() + width);
        width *= ratio;
    }
    edges.back() = upper;
    return edges;
}

/**
 * @brief Append a graded block without duplicating its shared first edge.
 */
void append_block(
    SimpleFluid::ArrReal& edges,
    double lower,
    double upper,
    int cells,
    double expansion_ratio)
{
    auto block = graded_edges(
        lower, upper, cells, expansion_ratio);
    edges.insert(edges.end(), block.begin() + 1, block.end());
}

/**
 * @brief Reproduce the two radial blocks in the OpenFOAM Shiri mesh.
 */
SimpleFluid::ArrReal shiri_radial_edges(int cells)
{
    if (cells == 1)
        return uniform_edges(0.075, 0.600, cells);

    const int inner_cells = cells / 2;
    const int outer_cells = cells - inner_cells;
    auto edges = graded_edges(
        0.075, 0.2625, inner_cells, 3.0);
    append_block(
        edges, 0.2625, 0.600, outer_cells, 1.0);
    return edges;
}

/**
 * @brief Reproduce the lower and upper axial blocks in the OpenFOAM mesh.
 */
SimpleFluid::ArrReal shiri_axial_edges(int cells)
{
    if (cells == 1)
        return uniform_edges(0.0, 1.5, cells);

    const int lower_cells = std::max(1, cells / 5);
    const int upper_cells = cells - lower_cells;
    auto edges = graded_edges(
        0.0, 0.120, lower_cells, 2.0);
    append_block(
        edges, 0.120, 1.5, upper_cells, 1.5);
    return edges;
}

/**
 * @brief Write rank-local cylindrical solution profiles for comparison.
 *
 * @tparam Pack Tpetra type pack used by the mesh and solver.
 * @param mesh Mesh providing owned cell centers.
 * @param solver Solver providing temperature, velocity, and pressure fields.
 * @param turbulence Standard k-epsilon state used by the comparison.
 * @param rank MPI rank used in the output filename.
 * @throws std::runtime_error if the output file cannot be opened.
 */
template<class Pack>
void write_profile_cells(
    const SimpleFluid::MeshHandle<Pack>& mesh,
    const SimpleFluid::BoussinesqSolver<Pack>& solver,
    const SimpleFluid::TurbulenceModel<Pack>& turbulence,
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

    const auto* epsilon = turbulence.dissipation_rate();
    if (epsilon == nullptr)
        throw std::logic_error(
            "Shiri standard k-epsilon output requires an epsilon field.");
    const auto* buoyancy_production =
        turbulence.buoyancy_production();
    const auto* wall_y_plus =
        turbulence.wall_y_plus();
    const auto* k_source =
        turbulence.turbulent_kinetic_energy_source();
    const auto* k_sink =
        turbulence.turbulent_kinetic_energy_sink();
    const auto* secondary_source =
        turbulence.secondary_source();
    const auto* secondary_sink =
        turbulence.secondary_sink();
    if (buoyancy_production == nullptr || wall_y_plus == nullptr
        || k_source == nullptr || k_sink == nullptr
        || secondary_source == nullptr || secondary_sink == nullptr)
        throw std::logic_error(
            "Shiri turbulent output requires buoyancy and wall diagnostics.");
    const double turbulent_prandtl =
        turbulence.options().turbulent_prandtl_number;

    output << "cell_geometry_gid,r,theta,z,temperature,ur,utheta,uz,pressure,"
              "k,epsilon,nut,alphat,buoyancy_production,wall_y_plus,"
              "k_source,k_sink,epsilon_source,epsilon_sink\n";
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
        const double turbulent_viscosity =
            turbulence.turbulent_kinematic_viscosity().value(lid);
        output << mesh.cell_geometry_global_id(lid) << ','
               << std::hypot(center.x, center.y) << ',' << theta << ','
               << center.z << ',' << solver.temperature().value(lid) << ','
               << radial_velocity << ',' << azimuthal_velocity << ','
               << velocity.z << ',' << solver.pressure().value(lid) << ','
               << turbulence.turbulent_kinetic_energy().value(lid) << ','
               << epsilon->value(lid) << ',' << turbulent_viscosity << ','
               << turbulent_viscosity / turbulent_prandtl << ','
               << buoyancy_production->value(lid) << ','
               << wall_y_plus->value(lid) << ','
               << k_source->value(lid) << ','
               << k_sink->value(lid) << ','
               << secondary_source->value(lid) << ','
               << secondary_sink->value(lid) << '\n';
    }
}

/** @brief Optionally write owned pressure-corrected face fluxes for diagnosis. */
template<class Pack>
void write_profile_face_fluxes(
    const SimpleFluid::MeshHandle<Pack>& mesh,
    const SimpleFluid::BoussinesqSolver<Pack>& solver,
    int rank)
{
    const char* configured_prefix =
        std::getenv("SIMPLEFLUID_SHIRI_FACE_OUTPUT_PREFIX");
    if (configured_prefix == nullptr)
        return;
    const std::string prefix = configured_prefix;
    if (prefix.empty())
    {
        throw std::invalid_argument(
            "SIMPLEFLUID_SHIRI_FACE_OUTPUT_PREFIX cannot be empty.");
    }
    const auto filename =
        prefix + "_rank" + std::to_string(rank) + ".csv";
    std::ofstream output(filename);
    if (!output)
        throw std::runtime_error("Cannot open " + filename + " for writing.");

    output << "face_geometry_gid,x,y,z,nx,ny,nz,area,flux\n";
    output << std::setprecision(17);
    const auto& fluxes =
        solver.pressure_corrected_face_fluxes();
    for (const auto face_lid : fluxes.owned_face_ids())
    {
        const auto center = mesh.face_centroid(face_lid);
        const auto normal = mesh.face_normal(face_lid);
        output << mesh.face_global_id(face_lid) << ','
               << center.x << ',' << center.y << ',' << center.z << ','
               << normal.x << ',' << normal.y << ',' << normal.z << ','
               << mesh.face_area(face_lid) << ','
               << fluxes.value(face_lid) << '\n';
    }
}

/** @brief Split one unquoted numeric CSV row. */
std::vector<std::string> split_csv_row(const std::string& line)
{
    std::vector<std::string> values;
    std::stringstream stream(line);
    std::string value;
    while (std::getline(stream, value, ','))
    {
        values.push_back(std::move(value));
    }
    return values;
}

/** @brief Parse one finite numeric restart value with full-token checking. */
double restart_real(
    const std::vector<std::string>& row,
    size_t column,
    const std::string& field_name,
    const std::string& filename,
    size_t line_number)
{
    size_t parsed = 0;
    const auto value = std::stod(row.at(column), &parsed);
    if (parsed != row.at(column).size() || !std::isfinite(value))
    {
        throw std::invalid_argument(
            filename + ':' + std::to_string(line_number)
            + " contains an invalid " + field_name + " value.");
    }
    return value;
}

struct RestartPlaneKey
{
    long long theta{};
    long long axial{};

    bool operator==(const RestartPlaneKey&) const = default;
};

struct RestartPlaneKeyHash
{
    size_t operator()(const RestartPlaneKey& key) const noexcept
    {
        const auto theta_hash = std::hash<long long>{}(key.theta);
        const auto axial_hash = std::hash<long long>{}(key.axial);
        return theta_hash
             ^ (axial_hash + 0x9e3779b9U
                + (theta_hash << 6U) + (theta_hash >> 2U));
    }
};

RestartPlaneKey restart_plane_key(double theta, double axial)
{
    constexpr double coordinate_resolution = 1.0e-8;
    return {
        std::llround(theta / coordinate_resolution),
        std::llround(axial / coordinate_resolution)};
}

/**
 * @brief Restore a Shiri state from all rank-local cell CSVs.
 *
 * New files use global cell IDs and therefore tolerate a changed partition.
 * Legacy files without that column are matched by cylindrical cell center.
 */
template<class Pack>
void restore_profile_cells(
    const SimpleFluid::MeshHandle<Pack>& geometry_mesh,
    const std::string& prefix,
    SimpleFluid::BoussinesqSolver<Pack>& solver,
    SimpleFluid::TurbulenceModel<Pack>& turbulence,
    double reference_density,
    int rank,
    int rank_count)
{
    using local_ordinal_type =
        typename Pack::local_ordinal_type;
    using global_ordinal_type =
        typename Pack::global_ordinal_type;
    using vector_type =
        typename SimpleFluid::VectorCellField<Pack>::vec_type;

    if (prefix.empty())
    {
        throw std::invalid_argument(
            "SIMPLEFLUID_SHIRI_RESTART_PREFIX cannot be empty.");
    }
    if (rank_count <= 0)
        throw std::invalid_argument("Shiri restart rank count must be positive.");

    const auto mesh = solver.temperature().mesh_ptr();
    SimpleFluid::CellField<Pack> restart_k(
        mesh, "restart_k");
    SimpleFluid::CellField<Pack> restart_epsilon(
        mesh, "restart_epsilon");
    SimpleFluid::CellField<Pack> restart_nu_t(
        mesh, "restart_nu_t");
    std::unordered_map<global_ordinal_type, local_ordinal_type>
        local_by_global;
    local_by_global.reserve(mesh->num_owned_cells());
    using LegacyCandidate =
        std::pair<double, local_ordinal_type>;
    std::unordered_map<
        RestartPlaneKey,
        std::vector<LegacyCandidate>,
        RestartPlaneKeyHash>
        local_by_plane;
    local_by_plane.reserve(mesh->num_owned_cells());
    for (size_t owned = 0;
         owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        local_by_global.emplace(
            geometry_mesh.cell_geometry_global_id(cell_lid), cell_lid);
        const auto center = mesh->cell_centroid(cell_lid);
        local_by_plane[restart_plane_key(
            std::atan2(center.y, center.x), center.z)]
            .emplace_back(
                std::hypot(center.x, center.y), cell_lid);
    }
    std::vector<bool> restored(
        mesh->num_owned_cells(), false);

    std::vector<std::string> expected_header;
    std::unordered_map<std::string, size_t> columns;
    bool uses_global_ids = false;
    size_t restored_count = 0;
    for (int checkpoint_rank = 0;
         checkpoint_rank < rank_count; ++checkpoint_rank)
    {
        const auto filename =
            prefix + "_rank" + std::to_string(checkpoint_rank)
            + ".csv";
        std::ifstream input(filename);
        if (!input)
        {
            throw std::runtime_error(
                "Cannot open Shiri restart file " + filename + '.');
        }
        std::string line;
        if (!std::getline(input, line))
        {
            throw std::invalid_argument(
                "Shiri restart file has no CSV header: "
                + filename);
        }
        const auto header = split_csv_row(line);
        if (checkpoint_rank == 0)
        {
            expected_header = header;
            for (size_t column = 0;
                 column < header.size(); ++column)
            {
                if (!columns.emplace(header[column], column).second)
                {
                    throw std::invalid_argument(
                        "Shiri restart file has a duplicate CSV column: "
                        + header[column]);
                }
            }
            uses_global_ids =
                columns.contains("cell_geometry_gid");
            for (const auto* required :
                 {"theta", "temperature", "ur", "utheta", "uz",
                  "pressure", "k", "epsilon", "nut"})
            {
                if (!columns.contains(required))
                {
                    throw std::invalid_argument(
                        "Shiri restart file is missing CSV column "
                        + std::string(required) + '.');
                }
            }
            if (!uses_global_ids
                && (!columns.contains("r")
                    || !columns.contains("z")))
            {
                throw std::invalid_argument(
                    "Legacy Shiri restart files require r and z "
                    "CSV columns.");
            }
        }
        else if (header != expected_header)
        {
            throw std::invalid_argument(
                "Shiri restart CSV headers do not match: "
                + filename);
        }

        size_t line_number = 1;
        while (std::getline(input, line))
        {
            ++line_number;
            if (line.empty())
                continue;
            const auto row = split_csv_row(line);
            if (row.size() != expected_header.size())
            {
                throw std::invalid_argument(
                    filename + ':' + std::to_string(line_number)
                    + " has a malformed CSV row.");
            }

            std::optional<local_ordinal_type> matched_lid;
            std::optional<double> parsed_theta;
            if (uses_global_ids)
            {
                size_t parsed = 0;
                const auto& gid_text =
                    row.at(columns.at("cell_geometry_gid"));
                const auto parsed_gid =
                    std::stoll(gid_text, &parsed);
                if (parsed != gid_text.size()
                    || !std::in_range<global_ordinal_type>(
                        parsed_gid))
                {
                    throw std::invalid_argument(
                        filename + ':'
                        + std::to_string(line_number)
                        + " contains an invalid cell_geometry_gid.");
                }
                const auto iter = local_by_global.find(
                    static_cast<global_ordinal_type>(
                        parsed_gid));
                if (iter == local_by_global.end())
                    continue;
                matched_lid = iter->second;
            }
            else
            {
                constexpr double radial_tolerance = 5.0e-4;
                const auto radial = restart_real(
                    row, columns.at("r"), "r",
                    filename, line_number);
                parsed_theta = restart_real(
                    row, columns.at("theta"), "theta",
                    filename, line_number);
                const auto axial = restart_real(
                    row, columns.at("z"), "z",
                    filename, line_number);
                const auto key =
                    restart_plane_key(*parsed_theta, axial);
                double closest_distance = radial_tolerance;
                for (long long theta_offset = -1;
                     theta_offset <= 1; ++theta_offset)
                {
                    for (long long axial_offset = -1;
                         axial_offset <= 1; ++axial_offset)
                    {
                        const auto iter = local_by_plane.find(
                            {key.theta + theta_offset,
                             key.axial + axial_offset});
                        if (iter == local_by_plane.end())
                            continue;
                        for (const auto& [candidate_r,
                                          candidate_lid] :
                             iter->second)
                        {
                            const auto distance =
                                std::abs(candidate_r - radial);
                            if (distance <= closest_distance)
                            {
                                closest_distance = distance;
                                matched_lid = candidate_lid;
                            }
                        }
                    }
                }
                if (!matched_lid)
                    continue;
            }

            const auto cell_lid = *matched_lid;
            const auto owned = static_cast<size_t>(cell_lid);
            if (restored.at(owned))
            {
                throw std::invalid_argument(
                    filename + ':'
                    + std::to_string(line_number)
                    + " restores one cell more than once.");
            }
            const auto theta = parsed_theta.value_or(
                restart_real(
                    row, columns.at("theta"), "theta",
                    filename, line_number));
            const auto radial_velocity = restart_real(
                row, columns.at("ur"), "ur",
                filename, line_number);
            const auto azimuthal_velocity = restart_real(
                row, columns.at("utheta"), "utheta",
                filename, line_number);
            const auto axial_velocity = restart_real(
                row, columns.at("uz"), "uz",
                filename, line_number);
            const auto cosine = std::cos(theta);
            const auto sine = std::sin(theta);
            solver.temperature().set_owned_value(
                cell_lid,
                restart_real(
                    row, columns.at("temperature"),
                    "temperature", filename, line_number));
            solver.velocity().set_owned_value(
                cell_lid,
                vector_type{
                    radial_velocity * cosine
                        - azimuthal_velocity * sine,
                    radial_velocity * sine
                        + azimuthal_velocity * cosine,
                    axial_velocity});
            solver.pressure().set_owned_value(
                cell_lid,
                restart_real(
                    row, columns.at("pressure"), "pressure",
                    filename, line_number));
            restart_k.set_owned_value(
                cell_lid,
                restart_real(
                    row, columns.at("k"), "k",
                    filename, line_number));
            restart_epsilon.set_owned_value(
                cell_lid,
                restart_real(
                    row, columns.at("epsilon"), "epsilon",
                    filename, line_number));
            restart_nu_t.set_owned_value(
                cell_lid,
                restart_real(
                    row, columns.at("nut"), "nut",
                    filename, line_number));
            restored[owned] = true;
            ++restored_count;
        }
    }
    if (restored_count != mesh->num_owned_cells()
        || std::find(restored.begin(), restored.end(), false)
               != restored.end())
    {
        size_t missing_count = 0;
        std::ostringstream missing_ids;
        for (size_t owned = 0;
             owned < mesh->num_owned_cells(); ++owned)
        {
            if (restored[owned])
                continue;
            if (missing_count < 5)
            {
                if (missing_count != 0)
                    missing_ids << ',';
                missing_ids << geometry_mesh.cell_geometry_global_id(
                    static_cast<local_ordinal_type>(owned));
            }
            ++missing_count;
        }
        throw std::invalid_argument(
            "Combined Shiri restart files do not contain every cell "
            "currently owned by rank " + std::to_string(rank)
            + " (restored=" + std::to_string(restored_count)
            + ", owned="
            + std::to_string(mesh->num_owned_cells())
            + ", missing=" + std::to_string(missing_count)
            + ", first missing global IDs="
            + missing_ids.str() + ").");
    }

    solver.temperature().sync_ghosts();
    solver.velocity().sync_ghosts();
    solver.pressure().sync_ghosts();
    restart_k.sync_ghosts();
    restart_epsilon.sync_ghosts();
    restart_nu_t.sync_ghosts();
    turbulence.restore_transported_state(
        restart_k, restart_epsilon, restart_nu_t,
        solver.velocity(), solver.material_properties(),
        static_cast<typename Pack::scalar_type>(
            reference_density));
}

/**
 * @brief Project the Shiri sector state onto its axisymmetric R-Z mean.
 *
 * The generated annular mesh has deterministic one-based geometry IDs with
 * radial index varying fastest, followed by theta and axial indices. The
 * projection averages scalar and cylindrical velocity components over theta
 * and republishes the transported turbulence state transactionally.
 */
template<class Pack>
void project_axisymmetric_shiri_state(
    SimpleFluid::BoussinesqSolver<Pack>& solver,
    SimpleFluid::TurbulenceModel<Pack>& turbulence,
    int radial_cells,
    int theta_cells,
    int axial_cells,
    double reference_density)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type =
        typename Pack::local_ordinal_type;
    using vector_type =
        typename SimpleFluid::VectorCellField<Pack>::vec_type;

    constexpr size_t count_component = 0;
    constexpr size_t temperature_component = 1;
    constexpr size_t radial_velocity_component = 2;
    constexpr size_t azimuthal_velocity_component = 3;
    constexpr size_t axial_velocity_component = 4;
    constexpr size_t pressure_component = 5;
    constexpr size_t k_component = 6;
    constexpr size_t secondary_component = 7;
    constexpr size_t nut_component = 8;
    constexpr size_t component_count = 9;

    const auto mesh = solver.temperature().mesh_ptr();
    const auto group_count =
        static_cast<size_t>(radial_cells)
      * static_cast<size_t>(axial_cells);
    if (group_count
        > static_cast<size_t>(std::numeric_limits<int>::max())
              / component_count)
    {
        throw std::overflow_error(
            "Shiri axisymmetric projection reduction is too large.");
    }
    std::vector<scalar_type> local(
        group_count * component_count, scalar_type{});
    std::vector<scalar_type> global(
        group_count * component_count, scalar_type{});
    int local_invalid_id = 0;
    const auto expected_cells =
        static_cast<size_t>(radial_cells)
      * static_cast<size_t>(theta_cells)
      * static_cast<size_t>(axial_cells);
    const auto* secondary =
        turbulence.dissipation_rate();
    if (secondary == nullptr)
    {
        secondary =
            turbulence.specific_dissipation_rate();
    }
    if (secondary == nullptr)
    {
        throw std::logic_error(
            "Shiri axisymmetric projection requires a secondary "
            "turbulence field.");
    }

    for (size_t owned = 0;
         owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto gid = mesh->cell_global_id(cell_lid);
        if (gid < 1
            || !std::in_range<size_t>(gid)
            || static_cast<size_t>(gid) > expected_cells)
        {
            local_invalid_id = 1;
            continue;
        }
        const auto ordinal = static_cast<size_t>(gid) - 1;
        const auto radial =
            ordinal % static_cast<size_t>(radial_cells);
        const auto axial =
            ordinal
            / (static_cast<size_t>(radial_cells)
               * static_cast<size_t>(theta_cells));
        const auto group =
            radial
          + static_cast<size_t>(radial_cells) * axial;
        const auto offset = component_count * group;
        const auto center = mesh->cell_centroid(cell_lid);
        const auto theta = std::atan2(center.y, center.x);
        const auto cosine = std::cos(theta);
        const auto sine = std::sin(theta);
        const auto velocity = solver.velocity().value(cell_lid);

        local[offset + count_component] += scalar_type{1};
        local[offset + temperature_component] +=
            solver.temperature().value(cell_lid);
        local[offset + radial_velocity_component] +=
            velocity.x * cosine + velocity.y * sine;
        local[offset + azimuthal_velocity_component] +=
            -velocity.x * sine + velocity.y * cosine;
        local[offset + axial_velocity_component] +=
            velocity.z;
        local[offset + pressure_component] +=
            solver.pressure().value(cell_lid);
        local[offset + k_component] +=
            turbulence.turbulent_kinetic_energy().value(
                cell_lid);
        local[offset + secondary_component] +=
            secondary->value(cell_lid);
        local[offset + nut_component] +=
            turbulence.turbulent_kinematic_viscosity().value(
                cell_lid);
    }

    const auto comm = Tpetra::getDefaultComm();
    int global_invalid_id = 0;
    Teuchos::reduceAll(
        *comm, Teuchos::REDUCE_MAX, 1,
        &local_invalid_id, &global_invalid_id);
    if (global_invalid_id != 0)
    {
        throw std::invalid_argument(
            "Shiri axisymmetric projection requires deterministic "
            "one-based annulus geometry IDs.");
    }
    Teuchos::reduceAll(
        *comm, Teuchos::REDUCE_SUM,
        static_cast<int>(local.size()),
        local.data(), global.data());

    for (size_t group = 0;
         group < group_count; ++group)
    {
        const auto count =
            global[component_count * group + count_component];
        if (!std::isfinite(count)
            || std::abs(
                   count
                 - static_cast<scalar_type>(theta_cells))
                   > scalar_type{0.5})
        {
            throw std::invalid_argument(
                "Shiri axisymmetric projection found an incomplete "
                "theta ring.");
        }
    }

    SimpleFluid::CellField<Pack> projected_k(
        mesh, "axisymmetric_k");
    SimpleFluid::CellField<Pack> projected_secondary(
        mesh, "axisymmetric_secondary");
    SimpleFluid::CellField<Pack> projected_nut(
        mesh, "axisymmetric_nu_t");
    for (size_t owned = 0;
         owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto ordinal =
            static_cast<size_t>(
                mesh->cell_global_id(cell_lid))
          - 1;
        const auto radial =
            ordinal % static_cast<size_t>(radial_cells);
        const auto axial =
            ordinal
            / (static_cast<size_t>(radial_cells)
               * static_cast<size_t>(theta_cells));
        const auto offset = component_count
          * (radial
             + static_cast<size_t>(radial_cells) * axial);
        const auto inverse_count =
            scalar_type{1}
          / global[offset + count_component];
        const auto center = mesh->cell_centroid(cell_lid);
        const auto theta = std::atan2(center.y, center.x);
        const auto cosine = std::cos(theta);
        const auto sine = std::sin(theta);
        const auto radial_velocity =
            global[offset + radial_velocity_component]
          * inverse_count;
        const auto azimuthal_velocity =
            global[offset + azimuthal_velocity_component]
          * inverse_count;

        solver.temperature().set_owned_value(
            cell_lid,
            global[offset + temperature_component]
              * inverse_count);
        solver.velocity().set_owned_value(
            cell_lid,
            vector_type{
                radial_velocity * cosine
                    - azimuthal_velocity * sine,
                radial_velocity * sine
                    + azimuthal_velocity * cosine,
                global[offset + axial_velocity_component]
                    * inverse_count});
        solver.pressure().set_owned_value(
            cell_lid,
            global[offset + pressure_component]
              * inverse_count);
        projected_k.set_owned_value(
            cell_lid,
            global[offset + k_component] * inverse_count);
        projected_secondary.set_owned_value(
            cell_lid,
            global[offset + secondary_component]
              * inverse_count);
        projected_nut.set_owned_value(
            cell_lid,
            global[offset + nut_component] * inverse_count);
    }

    solver.temperature().sync_ghosts();
    solver.velocity().sync_ghosts();
    solver.pressure().sync_ghosts();
    projected_k.sync_ghosts();
    projected_secondary.sync_ghosts();
    projected_nut.sync_ghosts();
    turbulence.restore_transported_state(
        projected_k, projected_secondary, projected_nut,
        solver.velocity(), solver.material_properties(),
        static_cast<scalar_type>(reference_density));
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
    const bool search_for_steady_state =
        environment_boolean(
            "SIMPLEFLUID_SHIRI_STEADY_STATE", false);
    const bool enforce_axisymmetry =
        environment_boolean(
            "SIMPLEFLUID_SHIRI_ENFORCE_AXISYMMETRY", false);
    if (enforce_axisymmetry && !search_for_steady_state)
    {
        throw std::invalid_argument(
            "SIMPLEFLUID_SHIRI_ENFORCE_AXISYMMETRY requires "
            "SIMPLEFLUID_SHIRI_STEADY_STATE=1.");
    }

    auto database = std::make_shared<SimpleFluid::Database>();
    database->set("dimension", 3);
    database->set("mesh_size", SimpleFluid::real_t{1.0});
    database->set(
        "domain_type",
        static_cast<int>(SimpleFluid::MeshFactory::DomainType::ANNULUS));
    database->set("R", shiri_radial_edges(radial_cells));
    database->set(
        "Theta", uniform_edges(
            0.0, 0.25 * std::numbers::pi, theta_cells));
    database->set("Z", shiri_axial_edges(axial_cells));
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
    time_options.coefficient_interpolation =
        SimpleFluid::FVM::FaceCoefficientInterpolation::Linear;
    time_options.pressure_gradient_scheme =
        SimpleFluid::FVM::CellGradientScheme::GaussLinear;
    time_options.n_pressure_correctors = 2;
    time_options.pressure_velocity_coupling =
        environment_pressure_velocity_coupling(
            "SIMPLEFLUID_SHIRI_PRESSURE_VELOCITY_COUPLING",
            search_for_steady_state
                ? SimpleFluid::PressureVelocityCoupling::SIMPLE
                : SimpleFluid::PressureVelocityCoupling::PISO);

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 500;
    linear_options.tolerance = 1.0e-9;
    linear_options.backend = environment_linear_solver_backend(
        "SIMPLEFLUID_SHIRI_LINEAR_SOLVER_BACKEND",
        SimpleFluid::LinearSolverBackend::BiCGStab);
    linear_options.preconditioner = environment_linear_preconditioner(
        "SIMPLEFLUID_SHIRI_LINEAR_PRECONDITIONER",
        SimpleFluid::LinearPreconditioner::Jacobi);

    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.reference_density = reference_density;
    model_options.density = reference_density;
    model_options.specific_heat_capacity = 1000.0;
    model_options.dynamic_viscosity = dynamic_viscosity;
    model_options.thermal_conductivity =
        reference_density * 1000.0 * thermal_diffusivity;

    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, bcs, time_options, linear_options, model_options);
    auto pressure_linear_options =
        solver.pressure_linear_solver_options();
    pressure_linear_options.backend =
        environment_linear_solver_backend(
            "SIMPLEFLUID_SHIRI_PRESSURE_LINEAR_SOLVER_BACKEND",
            SimpleFluid::LinearSolverBackend::BiCGStab);
    pressure_linear_options.preconditioner =
        environment_linear_preconditioner(
            "SIMPLEFLUID_SHIRI_PRESSURE_LINEAR_PRECONDITIONER",
            SimpleFluid::LinearPreconditioner::MueLu);
    pressure_linear_options.reuse_preconditioner =
        pressure_linear_options.preconditioner
            != SimpleFluid::LinearPreconditioner::None;
    solver.set_pressure_linear_solver_options(
        pressure_linear_options);
    solver.initialize_heated_box(290.0, 290.0);

    SimpleFluid::TurbulenceModelOptions turbulence_options;
    turbulence_options.model =
        SimpleFluid::TurbulenceModelType::StandardKEpsilon;
    turbulence_options.initial_turbulent_kinetic_energy = 0.00375;
    turbulence_options.initial_dissipation_rate = 0.00075;
    turbulence_options.turbulent_prandtl_number = 0.85;
    turbulence_options.gradient_scheme =
        SimpleFluid::FVM::CellGradientScheme::GaussLinear;
    turbulence_options.coefficient_interpolation =
        SimpleFluid::FVM::FaceCoefficientInterpolation::Linear;
    turbulence_options.buoyancy_model =
        SimpleFluid::TurbulenceBuoyancyModel::OpenFOAMBoussinesq;
    turbulence_options.buoyancy_coefficient = 1.0;
    turbulence_options.wall_treatment =
        SimpleFluid::TurbulenceWallTreatmentType::StandardHighReKEpsilon;
    turbulence_options.wall_options.boundary_names = {
        "rmin", "rmax", "zmin", "zmax"};
    turbulence_options.wall_options.thermal_wall_law =
        SimpleFluid::TurbulenceThermalWallLaw::Jayatilleke;
    turbulence_options.wall_options
        .thermal_turbulent_prandtl_number = 0.85;
    auto& turbulence =
        solver.configure_turbulence(turbulence_options);

    const auto comm = Tpetra::getDefaultComm();
    const int rank = comm->getRank();
    if (const char* restart_prefix =
            std::getenv("SIMPLEFLUID_SHIRI_RESTART_PREFIX"))
    {
        restore_profile_cells(
            *mesh, restart_prefix, solver, turbulence,
            reference_density, rank, comm->getSize());
        if (rank == 0)
        {
            std::cout
                << "restart: prefix=" << restart_prefix
                << ", ranks=" << comm->getSize() << '\n';
        }
    }
    if (rank == 0)
    {
        std::cout
            << "linear_solver: transport="
            << SimpleFluid::to_string(linear_options.backend) << '/'
            << SimpleFluid::to_string(linear_options.preconditioner)
            << ", pressure="
            << SimpleFluid::to_string(pressure_linear_options.backend)
            << '/'
            << SimpleFluid::to_string(
                   pressure_linear_options.preconditioner)
            << ", coupling="
            << SimpleFluid::to_string(
                   time_options.pressure_velocity_coupling)
            << '\n';
    }
    bool steady_state_reached = false;
    std::optional<std::string> steady_state_failure;
    int rejected_steady_steps = 0;
    std::optional<
        SimpleFluid::SteadyStateStepStatistics<
            SimpleFluid::real_t>>
        final_steady_statistics;
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
            positive_environment_integer(
                "SIMPLEFLUID_SHIRI_STEADY_CONSECUTIVE_STEPS",
                std::min(5, steps));
        steady_options.minimum_steps =
            positive_environment_integer(
                "SIMPLEFLUID_SHIRI_STEADY_MIN_STEPS",
                std::max(
                    1,
                    std::min(
                        20,
                        steps
                            - steady_options
                                  .required_consecutive_steps
                            + 1)));
        steady_options.maximum_retries_per_step =
            non_negative_environment_integer(
                "SIMPLEFLUID_SHIRI_STEADY_MAX_RETRIES", 4);
        steady_options.rejection_recovery_steps =
            non_negative_environment_integer(
                "SIMPLEFLUID_SHIRI_STEADY_REJECTION_RECOVERY_STEPS",
                5);
        steady_options.relative_update_tolerance =
            positive_environment_real(
                "SIMPLEFLUID_SHIRI_STEADY_TOLERANCE",
                1.0e-4);
        steady_options.minimum_time_step =
            positive_environment_real(
                "SIMPLEFLUID_SHIRI_STEADY_MIN_DT",
                time_step / 16.0);
        steady_options.maximum_time_step =
            positive_environment_real(
                "SIMPLEFLUID_SHIRI_STEADY_MAX_DT",
                std::max(time_step, 5.0e-2));
        steady_options.target_courant_number =
            positive_environment_real(
                "SIMPLEFLUID_SHIRI_STEADY_TARGET_COURANT",
                0.8);
        steady_options.time_step_growth_factor =
            positive_environment_real(
                "SIMPLEFLUID_SHIRI_STEADY_DT_GROWTH",
                1.5);
        steady_options.time_step_reduction_factor =
            positive_environment_real(
                "SIMPLEFLUID_SHIRI_STEADY_DT_REDUCTION",
                0.5);
        steady_options.rejection_time_step_safety_factor =
            positive_environment_real(
                "SIMPLEFLUID_SHIRI_STEADY_REJECTION_SAFETY",
                0.9);

        SimpleFluid::AdaptiveSteadyStateController controller(
            steady_options, time_step);
        SimpleFluid::SteadyStateFieldMonitor<Pack> monitor(
            solver.temperature().mesh_ptr(),
            time_options.reference_temperature,
            steady_options.update_scales);
        const auto* secondary =
            turbulence.dissipation_rate();
        if (secondary == nullptr)
        {
            secondary =
                turbulence.specific_dissipation_rate();
        }
        if (secondary == nullptr)
        {
            throw std::logic_error(
                "Steady turbulent Shiri search requires a secondary "
                "turbulence field.");
        }
        if (enforce_axisymmetry)
        {
            project_axisymmetric_shiri_state(
                solver, turbulence, radial_cells,
                theta_cells, axial_cells,
                reference_density);
        }
        monitor.initialize(
            solver.velocity(),
            solver.temperature(),
            {&turbulence.turbulent_kinetic_energy(),
             secondary});
        SimpleFluid::SteadyStateProgressStream progress(std::cout);

        if (rank == 0)
        {
            std::cout
                << "steady_state_search: enabled=yes max_steps="
                << steady_options.maximum_steps
                << " tolerance="
                << steady_options.relative_update_tolerance
                << " min_steps=" << steady_options.minimum_steps
                << " consecutive_steps="
                << steady_options.required_consecutive_steps
                << " max_retries="
                << steady_options.maximum_retries_per_step
                << " rejection_recovery_steps="
                << steady_options.rejection_recovery_steps
                << " rejection_safety="
                << steady_options
                       .rejection_time_step_safety_factor
                << " axisymmetric="
                << (enforce_axisymmetry ? "yes" : "no")
                << " target_Co="
                << steady_options.target_courant_number
                << " dt_range=["
                << steady_options.minimum_time_step << ','
                << steady_options.maximum_time_step << "]\n";
        }

        for (int iteration = 0;
             iteration < steady_options.maximum_steps;
             ++iteration)
        {
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
                        throw;

                    ++rejected_steady_steps;
                    const auto reduced_time_step =
                        controller.rejected_time_step(
                            accepted_time_step);
                    if (rank == 0)
                    {
                        progress.write_retry(
                            iteration + 1,
                            std::min(
                                retries + 1,
                                steady_options
                                    .maximum_retries_per_step),
                            steady_options
                                .maximum_retries_per_step,
                            solver.time(),
                            accepted_time_step,
                            reduced_time_step,
                            error.what());
                    }
                    if (retries
                            >= steady_options
                                .maximum_retries_per_step
                        || !(reduced_time_step
                             < accepted_time_step))
                    {
                        steady_state_failure = error.what();
                        break;
                    }
                    ++retries;
                    solver.set_time_step(reduced_time_step);
                }
            }
            if (!accepted)
                break;

            if (enforce_axisymmetry)
            {
                project_axisymmetric_shiri_state(
                    solver, turbulence, radial_cells,
                    theta_cells, axial_cells,
                    reference_density);
            }
            const auto update_rates =
                monitor.observe(accepted_time_step);
            const auto statistics = controller.observe(
                solver.time(),
                accepted_time_step,
                solver.maximum_courant_number(),
                {static_cast<SimpleFluid::real_t>(
                     update_rates.velocity),
                 static_cast<SimpleFluid::real_t>(
                     update_rates.temperature),
                 static_cast<SimpleFluid::real_t>(
                     update_rates.turbulence)},
                solver.last_step_statistics().converged);
            final_steady_statistics = statistics;
            if (rank == 0)
            {
                progress.write(
                    statistics,
                    solver.last_step_statistics());
            }
            if (statistics.steady)
            {
                steady_state_reached = true;
                break;
            }
            solver.set_time_step(
                statistics.next_time_step);
        }
    }

    SimpleFluid::SolutionOutputOptions output_options;
    output_options.include_turbulence_fields = true;
    solver.write_parallel_solution_vtu(
        "natural_convection_shiri.vtu", output_options);
    write_profile_cells(*mesh, solver, turbulence, rank);
    write_profile_face_fluxes(*mesh, solver, rank);

    if (rank == 0)
    {
        std::cout << "Shiri annulus: " << radial_cells << 'x' << theta_cells
                  << 'x' << axial_cells << " cells, "
                  << solver.step_index()
                  << " steps, t=" << solver.time() << ", MPI ranks="
                  << comm->getSize() << '\n';
        if (search_for_steady_state)
        {
            std::cout
                << "steady_state_search: reached="
                << (steady_state_reached ? "yes" : "no")
                << ", rejected_steps="
                << rejected_steady_steps;
            if (final_steady_statistics)
            {
                const auto& statistics =
                    *final_steady_statistics;
                std::cout
                    << ", steps=" << statistics.iteration
                    << ", final_update_rate="
                    << statistics.update_rates.maximum()
                    << ", final_max_Co="
                    << statistics.maximum_courant_number
                    << ", final_dt=" << statistics.time_step;
            }
            else
            {
                std::cout << ", steps=0";
            }
            if (steady_state_failure)
            {
                std::cout << ", failure=\""
                          << *steady_state_failure << '"';
            }
            std::cout << '\n';
        }
        for (const auto& statistics :
             turbulence.wall_y_plus_statistics())
        {
            std::cout << "wall_y_plus[" << statistics.boundary_name
                      << "]: min=" << statistics.minimum
                      << ", mean=" << statistics.area_weighted_mean
                      << ", max=" << statistics.maximum
                      << ", faces=" << statistics.global_face_count
                      << '\n';
        }
    }
    return search_for_steady_state && !steady_state_reached
        ? 2
        : 0;
}
