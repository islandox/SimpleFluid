#include "benchmarks/BenchmarkSupport.hh"
#include "equations/BoundaryConditions.hh"
#include "FVM/Operators.hh"
#include "geometry/MeshFactory.hh"
#include "geometry/STKMesh.hh"
#include "solvers/BelosLinearSolver.hh"
#include "solvers/BoussinesqSolver.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <unistd.h>
#include <stk_io/IossBridge.hpp>
#include <stk_mesh/base/FEMHelpers.hpp>
#include <stk_mesh/base/FieldBase.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef SIMPLEFLUID_GIT_COMMIT
#define SIMPLEFLUID_GIT_COMMIT "unknown"
#endif
#ifndef SIMPLEFLUID_GIT_DIRTY
#define SIMPLEFLUID_GIT_DIRTY "unknown"
#endif
#ifndef SIMPLEFLUID_BUILD_TYPE
#define SIMPLEFLUID_BUILD_TYPE "unknown"
#endif
#ifndef SIMPLEFLUID_COMPILER
#define SIMPLEFLUID_COMPILER "unknown"
#endif

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using STKMeshType = SimpleFluid::STKMesh<Pack>;
using Comm = Teuchos::Comm<int>;

struct Dimensions
{
    int nx = 0;
    int ny = 0;
    int nz = 0;
};

enum class CaseSelection
{
    All,
    Diffusion,
    PressureVelocity
};

struct Options
{
    std::string preset = "debug-small";
    CaseSelection selection = CaseSelection::All;
    Dimensions diffusion{6, 6, 6};
    Dimensions pressure_velocity{8, 8, 1};
    std::vector<double> shears{0.0, 0.2, 0.4, 0.6};
    int repetitions = 1;
    int warmups = 0;
    std::string configuration = "all";
    std::filesystem::path output{"simplefluid_benchmark.csv"};
};

struct SolverConfiguration
{
    SimpleFluid::FVM::NonOrthogonalTreatment treatment =
        SimpleFluid::FVM::NonOrthogonalTreatment::Implicit;
    SimpleFluid::LinearPreconditioner preconditioner =
        SimpleFluid::LinearPreconditioner::None;
    SimpleFluid::PressureVelocityCoupling coupling =
        SimpleFluid::PressureVelocityCoupling::PISO;
    int nonorthogonal_correctors = 0;
    std::string preconditioner_name;
};

struct AngleMetrics
{
    double mean_degrees = 0.0;
    double maximum_degrees = 0.0;
};

struct ErrorMetrics
{
    double l2 = 0.0;
    double linf = 0.0;
};

template<class T>
T global_sum(const Teuchos::RCP<const Comm>& comm, T local)
{
    T global{};
    Teuchos::reduceAll(
        *comm, Teuchos::REDUCE_SUM, 1, &local, &global);
    return global;
}

template<class T>
T global_max(const Teuchos::RCP<const Comm>& comm, T local)
{
    T global{};
    Teuchos::reduceAll(
        *comm, Teuchos::REDUCE_MAX, 1, &local, &global);
    return global;
}

int parse_int(std::string_view value, std::string_view option)
{
    size_t consumed = 0;
    const auto parsed = std::stoi(std::string(value), &consumed);
    if (consumed != value.size() || parsed <= 0)
    {
        throw std::invalid_argument(
            std::string(option) + " requires a positive integer.");
    }
    return parsed;
}

double parse_double(std::string_view value, std::string_view option)
{
    size_t consumed = 0;
    const auto parsed = std::stod(std::string(value), &consumed);
    if (consumed != value.size() || parsed < 0.0)
    {
        throw std::invalid_argument(
            std::string(option) + " requires a non-negative number.");
    }
    return parsed;
}

void apply_preset(Options& options,
                  std::string_view preset,
                  int mpi_ranks)
{
    options.preset = std::string(preset);
    if (preset == "debug-small")
    {
        options.selection = CaseSelection::All;
        options.diffusion = {6, 6, 6};
        options.pressure_velocity = {8, 8, 1};
        options.repetitions = 1;
        options.warmups = 0;
        return;
    }
    if (preset == "release-profile")
    {
        options.selection = CaseSelection::All;
        options.diffusion = {64, 64, 64};
        options.pressure_velocity = {64, 64, 8};
        options.repetitions = 3;
        options.warmups = 1;
        return;
    }
    if (preset == "mpi-strong")
    {
        options.selection = CaseSelection::PressureVelocity;
        options.pressure_velocity = {64, 64, 8};
        options.repetitions = 3;
        options.warmups = 1;
        return;
    }
    if (preset == "mpi-weak")
    {
        options.selection = CaseSelection::PressureVelocity;
        options.pressure_velocity = {32 * mpi_ranks, 32, 8};
        options.repetitions = 3;
        options.warmups = 1;
        return;
    }

    throw std::invalid_argument("Unknown benchmark preset.");
}

std::string_view require_value(
    int argc, char** argv, int& index, std::string_view option)
{
    if (index + 1 >= argc)
    {
        throw std::invalid_argument(
            std::string(option) + " requires a value.");
    }
    return argv[++index];
}

Options parse_options(int argc, char** argv, int mpi_ranks)
{
    Options options;
    std::string preset = "debug-small";
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view argument(argv[i]);
        if (argument == "--preset")
        {
            preset = require_value(argc, argv, i, argument);
        }
        else if (argument == "--scaling")
        {
            const auto scaling =
                require_value(argc, argv, i, argument);
            if (scaling == "strong")
            {
                preset = "mpi-strong";
            }
            else if (scaling == "weak")
            {
                preset = "mpi-weak";
            }
            else if (scaling != "none")
            {
                throw std::invalid_argument(
                    "--scaling expects none, strong, or weak.");
            }
        }
    }
    apply_preset(options, preset, mpi_ranks);

    for (int i = 1; i < argc; ++i)
    {
        const std::string_view argument(argv[i]);
        if (argument == "--preset" || argument == "--scaling")
        {
            ++i;
        }
        else if (argument == "--case")
        {
            const auto value = require_value(argc, argv, i, argument);
            if (value == "all")
            {
                options.selection = CaseSelection::All;
            }
            else if (value == "diffusion_nonorthogonal")
            {
                options.selection = CaseSelection::Diffusion;
            }
            else if (value == "pressure_velocity")
            {
                options.selection = CaseSelection::PressureVelocity;
            }
            else
            {
                throw std::invalid_argument("Unknown benchmark case.");
            }
        }
        else if (argument == "--nx")
        {
            const auto value =
                parse_int(require_value(argc, argv, i, argument), argument);
            options.diffusion.nx = value;
            options.pressure_velocity.nx = value;
        }
        else if (argument == "--ny")
        {
            const auto value =
                parse_int(require_value(argc, argv, i, argument), argument);
            options.diffusion.ny = value;
            options.pressure_velocity.ny = value;
        }
        else if (argument == "--nz")
        {
            const auto value =
                parse_int(require_value(argc, argv, i, argument), argument);
            options.diffusion.nz = value;
            options.pressure_velocity.nz = value;
        }
        else if (argument == "--shear")
        {
            options.shears = {
                parse_double(
                    require_value(argc, argv, i, argument), argument)};
        }
        else if (argument == "--repetitions")
        {
            options.repetitions =
                parse_int(require_value(argc, argv, i, argument), argument);
        }
        else if (argument == "--warmups")
        {
            const auto value = require_value(argc, argv, i, argument);
            size_t consumed = 0;
            options.warmups = std::stoi(std::string(value), &consumed);
            if (consumed != value.size() || options.warmups < 0)
            {
                throw std::invalid_argument(
                    "--warmups requires a non-negative integer.");
            }
        }
        else if (argument == "--output")
        {
            options.output =
                require_value(argc, argv, i, argument);
        }
        else if (argument == "--configuration")
        {
            options.configuration =
                require_value(argc, argv, i, argument);
            if (options.configuration != "all"
                && options.configuration != "explicit"
                && options.configuration != "implicit"
                && options.configuration != "hybrid"
                && options.configuration != "coupled")
            {
                throw std::invalid_argument(
                    "--configuration expects all, explicit, implicit, hybrid, or coupled.");
            }
        }
        else if (argument == "--help")
        {
            std::cout
                << "simplefluid_benchmark [--preset debug-small|release-profile|"
                   "mpi-strong|mpi-weak] [--case all|diffusion_nonorthogonal|"
                   "pressure_velocity] [--nx N --ny N --nz N] [--shear S] "
                   "[--repetitions N] [--warmups N] [--output PATH] "
                   "[--configuration all|explicit|implicit|hybrid|coupled] "
                   "[--scaling none|strong|weak]\n";
            std::exit(0);
        }
        else
        {
            throw std::invalid_argument(
                "Unknown benchmark option: " + std::string(argument));
        }
    }
    return options;
}

SimpleFluid::ArrReal unit_edges(int cells)
{
    SimpleFluid::ArrReal edges(static_cast<size_t>(cells) + 1);
    for (int edge = 0; edge <= cells; ++edge)
    {
        edges[static_cast<size_t>(edge)] =
            static_cast<double>(edge) / static_cast<double>(cells);
    }
    return edges;
}

std::shared_ptr<SimpleFluid::Database>
make_box_database(const Dimensions& dimensions)
{
    auto database = std::make_shared<SimpleFluid::Database>();
    database->set("dimension", 3);
    database->set("mesh_size", SimpleFluid::real_t{1.0});
    database->set(
        "domain_type",
        static_cast<int>(SimpleFluid::MeshFactory::DomainType::BOX));
    database->set("X", unit_edges(dimensions.nx));
    database->set("Y", unit_edges(dimensions.ny));
    database->set("Z", unit_edges(dimensions.nz));
    database->set(
        "domain_exterior_face_types",
        SimpleFluid::ArrString{
            "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});
    return database;
}

std::shared_ptr<MeshType>
make_sheared_box_mesh(const Dimensions& dimensions, double shear)
{
    auto mesh = std::make_shared<STKMeshType>();
    auto meta = mesh->meta();
    auto bulk = mesh->bulk();
    auto& coordinates =
        meta->declare_field<double>(
            stk::topology::NODE_RANK, "coordinates");
    stk::mesh::put_field_on_mesh(
        coordinates, meta->universal_part(), 3, nullptr);
    meta->set_coordinate_field(&coordinates);

    auto& hex_part =
        meta->declare_part_with_topology(
            "benchmark_hexes", stk::topology::HEX_8);
    stk::io::put_io_part_attribute(hex_part);

    std::array<stk::mesh::Part*, 6> boundary_parts{};
    const std::array<const char*, 6> boundary_names{
        "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
    for (size_t index = 0; index < boundary_parts.size(); ++index)
    {
        auto& part =
            meta->declare_part(boundary_names[index], meta->side_rank());
        stk::io::put_io_part_attribute(part);
        boundary_parts[index] = &part;
    }

    const auto nx = static_cast<size_t>(dimensions.nx);
    const auto ny = static_cast<size_t>(dimensions.ny);
    const auto nz = static_cast<size_t>(dimensions.nz);
    auto node_id =
        [=](size_t i, size_t j, size_t k)
        {
            return static_cast<stk::mesh::EntityId>(
                1 + i + (nx + 1) * (j + (ny + 1) * k));
        };
    auto element_id =
        [=](size_t i, size_t j, size_t k)
        {
            return static_cast<stk::mesh::EntityId>(
                1 + i + nx * (j + ny * k));
        };
    auto declare_boundary_side =
        [&](stk::mesh::Entity element,
            unsigned side,
            stk::mesh::Part* part)
        {
            stk::mesh::PartVector parts{part};
            bulk->declare_element_side(element, side, parts);
        };

    bulk->modification_begin();
    for (size_t k = 0; k < nz; ++k)
    {
        for (size_t j = 0; j < ny; ++j)
        {
            for (size_t i = 0; i < nx; ++i)
            {
                const stk::mesh::EntityIdVector nodes{
                    node_id(i, j, k),
                    node_id(i + 1, j, k),
                    node_id(i + 1, j + 1, k),
                    node_id(i, j + 1, k),
                    node_id(i, j, k + 1),
                    node_id(i + 1, j, k + 1),
                    node_id(i + 1, j + 1, k + 1),
                    node_id(i, j + 1, k + 1)};
                const auto element =
                    stk::mesh::declare_element(
                        *bulk, hex_part, element_id(i, j, k), nodes);
                if (i == 0) declare_boundary_side(element, 3, boundary_parts[0]);
                if (i + 1 == nx) declare_boundary_side(element, 1, boundary_parts[1]);
                if (j == 0) declare_boundary_side(element, 0, boundary_parts[2]);
                if (j + 1 == ny) declare_boundary_side(element, 2, boundary_parts[3]);
                if (k == 0) declare_boundary_side(element, 4, boundary_parts[4]);
                if (k + 1 == nz) declare_boundary_side(element, 5, boundary_parts[5]);
            }
        }
    }

    for (size_t k = 0; k <= nz; ++k)
    {
        for (size_t j = 0; j <= ny; ++j)
        {
            for (size_t i = 0; i <= nx; ++i)
            {
                const auto x =
                    static_cast<double>(i) / static_cast<double>(nx);
                const auto y =
                    static_cast<double>(j) / static_cast<double>(ny);
                const auto z =
                    static_cast<double>(k) / static_cast<double>(nz);
                const auto node =
                    bulk->get_entity(
                        stk::topology::NODE_RANK, node_id(i, j, k));
                auto* data = stk::mesh::field_data(coordinates, node);
                data[0] = x + shear * y;
                data[1] = y;
                data[2] = z;
            }
        }
    }
    bulk->modification_end();
    mesh->assemble();
    return mesh;
}

double manufactured_solution(const SimpleFluid::vec3<>& point)
{
    return 0.1 + point.x * point.x
         + 0.25 * point.y - 0.125 * point.z;
}

AngleMetrics nonorthogonality_metrics(const MeshType& mesh)
{
    double local_sum = 0.0;
    double local_maximum = 0.0;
    long long local_count = 0;
    for (size_t face = 0; face < mesh.num_faces(); ++face)
    {
        const auto face_lid =
            static_cast<MeshType::local_ordinal_type>(face);
        if (!mesh.is_owned_face(face_lid)
            || !mesh.is_interior_face(face_lid))
        {
            continue;
        }
        const auto owner = mesh.owner_cell(face_lid);
        const auto neighbor = mesh.neighbor_cell(face_lid);
        const auto center_vector =
            mesh.cell_centroid(neighbor) - mesh.cell_centroid(owner);
        const auto area_vector =
            mesh.face_area_vector_outward(face_lid, owner);
        const auto denominator =
            center_vector.norm() * area_vector.norm();
        if (denominator <= 0.0)
        {
            continue;
        }
        const auto cosine = std::clamp(
            std::abs(center_vector.dot(area_vector)) / denominator,
            0.0, 1.0);
        const auto angle =
            std::acos(cosine) * 180.0 / std::numbers::pi;
        local_sum += angle;
        local_maximum = std::max(local_maximum, angle);
        ++local_count;
    }

    const auto comm = mesh.owned_cell_map()->getComm();
    const auto count = global_sum(comm, local_count);
    return {
        count > 0 ? global_sum(comm, local_sum) / count : 0.0,
        global_max(comm, local_maximum)};
}

ErrorMetrics error_metrics(const FieldType& field)
{
    double local_squared_error = 0.0;
    double local_volume = 0.0;
    double local_maximum = 0.0;
    for (size_t owned = 0;
         owned < field.mesh().num_owned_cells();
         ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto error =
            field.value(cell_lid)
          - manufactured_solution(field.mesh().cell_centroid(cell_lid));
        const auto volume = field.mesh().cell_volume(cell_lid);
        local_squared_error += error * error * volume;
        local_volume += volume;
        local_maximum = std::max(local_maximum, std::abs(error));
    }
    const auto comm = field.mesh().owned_cell_map()->getComm();
    const auto squared_error = global_sum(comm, local_squared_error);
    const auto volume = global_sum(comm, local_volume);
    return {
        volume > 0.0 ? std::sqrt(squared_error / volume) : 0.0,
        global_max(comm, local_maximum)};
}

SimpleFluid::Benchmark::Record base_record(
    const Options& options,
    const std::string& run_id,
    std::string benchmark_case,
    const Dimensions& dimensions,
    int mpi_ranks)
{
    SimpleFluid::Benchmark::Record record;
    record.timestamp = SimpleFluid::Benchmark::utc_timestamp();
    record.run_id = run_id;
    record.git_commit = SIMPLEFLUID_GIT_COMMIT;
    record.git_dirty = SIMPLEFLUID_GIT_DIRTY;
    record.build_type = SIMPLEFLUID_BUILD_TYPE;
    record.compiler = SIMPLEFLUID_COMPILER;
    record.hostname = SimpleFluid::Benchmark::host_name();
    record.benchmark_case = std::move(benchmark_case);
    record.preset = options.preset;
    record.mesh_nx = dimensions.nx;
    record.mesh_ny = dimensions.ny;
    record.mesh_nz = dimensions.nz;
    record.mpi_ranks = mpi_ranks;
    return record;
}

void add_memory_metrics(
    SimpleFluid::Benchmark::Record& record,
    const Teuchos::RCP<const Comm>& comm)
{
    const auto local = SimpleFluid::Benchmark::sample_process_memory();
    record.rss_max_rank_kib =
        global_max(comm, local.resident_kib);
    record.rss_sum_ranks_kib =
        global_sum(comm, local.resident_kib);
    record.peak_rss_max_rank_kib =
        global_max(comm, local.peak_resident_kib);
    record.peak_rss_sum_ranks_kib =
        global_sum(comm, local.peak_resident_kib);
}

SimpleFluid::Benchmark::Record run_diffusion(
    const Options& options,
    const std::string& run_id,
    const SolverConfiguration& configuration,
    double shear,
    int repetition)
{
    const auto comm = Tpetra::getDefaultComm();
    comm->barrier();
    const auto total_start = MPI_Wtime();
    const auto setup_start = total_start;
    auto mesh = make_sheared_box_mesh(options.diffusion, shear);
    const auto angles = nonorthogonality_metrics(*mesh);
    FieldType solution(mesh, "benchmark_diffusion_solution");
    SimpleFluid::BelosLinearSolver<Pack> solver;
    comm->barrier();
    const auto solve_start = MPI_Wtime();

    constexpr double diffusivity = 1.0;
    auto boundary_condition =
        [&](int patch_id, size_t in_patch_id)
            -> SimpleFluid::BoundaryCondition
    {
        const auto face_lid =
            mesh->boundary_face_patch(patch_id).face_lids[in_patch_id];
        return {
            SimpleFluid::BoundaryConditionType::Dirichlet,
            manufactured_solution(mesh->face_centroid(face_lid))};
    };
    auto source =
        [](MeshType::local_ordinal_type)
        {
            return -2.0;
        };

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 500;
    linear_options.tolerance = 1.0e-10;
    linear_options.preconditioner = configuration.preconditioner;

    SimpleFluid::LinearSolveSummary summary;
    const auto solve_system =
        [&](const auto& system)
    {
        solution.owned_data().putScalar(0.0);
        Teuchos::RCP<const Pack::operator_type> matrix = system.matrix;
        const auto statistics =
            solver.solve_with_statistics(
                matrix, *system.rhs,
                solution.owned_data(), linear_options);
        summary.add(statistics);
        mesh->sync_periodic_boundaries(solution);
    };

    if (configuration.treatment
        == SimpleFluid::FVM::NonOrthogonalTreatment::Implicit)
    {
        solve_system(
            SimpleFluid::FVM::
                fully_implicit_non_orthogonal_diffusion_system<Pack>(
                    *mesh, diffusivity, boundary_condition, source));
    }
    else
    {
        for (int corrector = 0;
             corrector <= configuration.nonorthogonal_correctors;
             ++corrector)
        {
            if (configuration.treatment
                == SimpleFluid::FVM::NonOrthogonalTreatment::Explicit)
            {
                if (corrector == 0)
                {
                    solve_system(
                        SimpleFluid::FVM::diffusion_system<Pack>(
                            *mesh, diffusivity,
                            boundary_condition, source));
                }
                else
                {
                    solve_system(
                        SimpleFluid::FVM::
                            explicit_non_orthogonal_diffusion_system<Pack>(
                                *mesh, diffusivity,
                                boundary_condition, source, solution));
                }
            }
            else
            {
                const auto* correction =
                    corrector == 0 ? nullptr : &solution;
                solve_system(
                    SimpleFluid::FVM::
                        non_orthogonal_diffusion_system<Pack>(
                            *mesh, diffusivity,
                            boundary_condition, source,
                            SimpleFluid::FVM::NonOrthogonalTreatment::Hybrid,
                            correction));
            }
        }
    }

    const auto errors = error_metrics(solution);
    comm->barrier();
    const auto total_end = MPI_Wtime();

    auto record = base_record(
        options, run_id, "diffusion_nonorthogonal",
        options.diffusion, comm->getSize());
    record.global_cells =
        static_cast<long long>(
            mesh->owned_cell_map()->getGlobalNumElements());
    record.shear = shear;
    record.mean_nonorthogonality_deg = angles.mean_degrees;
    record.max_nonorthogonality_deg = angles.maximum_degrees;
    record.treatment =
        SimpleFluid::FVM::to_string(configuration.treatment);
    record.coupling = "none";
    record.preconditioner = configuration.preconditioner_name;
    record.linear_tolerance = linear_options.tolerance;
    record.max_linear_iterations = linear_options.max_iterations;
    record.nonorthogonal_correctors =
        configuration.nonorthogonal_correctors;
    record.repetition = repetition;
    record.linear_solves = summary.solves;
    record.krylov_iterations = summary.iterations;
    record.setup_seconds =
        global_max(comm, solve_start - setup_start);
    record.solve_seconds =
        global_max(comm, total_end - solve_start);
    record.total_seconds =
        global_max(comm, total_end - total_start);
    record.l2_error = errors.l2;
    record.linf_error = errors.linf;
    record.achieved_tolerance = summary.achieved_tolerance;
    record.converged = summary.converged;
    add_memory_metrics(record, comm);
    return record;
}

SimpleFluid::BoundaryConditionSet cavity_boundary_conditions()
{
    SimpleFluid::BoundaryConditionSet conditions;
    for (const auto* name :
         {"xmin", "xmax", "ymin", "zmin", "zmax"})
    {
        conditions.velocity[name] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }
    conditions.velocity["ymax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet,
        {1.0, 0.0, 0.0}};
    conditions.velocity["zmin"] = {
        SimpleFluid::BoundaryConditionType::Slip, {}};
    conditions.velocity["zmax"] = {
        SimpleFluid::BoundaryConditionType::Slip, {}};
    for (const auto* name :
         {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        conditions.temperature[name] = {
            SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    }
    return conditions;
}

SimpleFluid::Benchmark::Record run_pressure_velocity(
    const Options& options,
    const std::string& run_id,
    const SolverConfiguration& configuration,
    int repetition)
{
    const auto comm = Tpetra::getDefaultComm();
    comm->barrier();
    const auto total_start = MPI_Wtime();
    const auto setup_start = total_start;
    SimpleFluid::MeshFactory factory(
        make_box_database(options.pressure_velocity));
    auto mesh = factory.build<Pack>();
    long long local_lid_faces = 0;
    long long local_owned_lid_faces = 0;
    double local_lid_area = 0.0;
    for (const auto& [patch_id, patch] : mesh->boundary_patches())
    {
        if (mesh->boundary_patch_name(patch_id) == "ymax")
        {
            local_lid_faces +=
                static_cast<long long>(patch.face_lids.size());
            for (const auto face_lid : patch.face_lids)
            {
                local_owned_lid_faces +=
                    mesh->is_owned_face(face_lid) ? 1 : 0;
                if (mesh->is_owned_face(face_lid))
                {
                    local_lid_area += mesh->face_area(face_lid);
                }
            }
        }
    }
    if (global_sum(comm, local_lid_faces) == 0)
    {
        throw std::runtime_error(
            "Pressure-velocity benchmark mesh has no ymax boundary faces.");
    }
    if (global_sum(comm, local_owned_lid_faces) == 0)
    {
        throw std::runtime_error(
            "Pressure-velocity benchmark mesh has no owned ymax boundary faces.");
    }
    if (global_sum(comm, local_lid_area) <= 0.0)
    {
        throw std::runtime_error(
            "Pressure-velocity benchmark moving lid has zero area.");
    }

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 2.0e-3;
    time_options.steps = 1;
    time_options.thermal_diffusivity = 0.0;
    time_options.kinematic_viscosity = 1.0e-2;
    time_options.thermal_expansion = 0.0;
    time_options.gravity_x = 0.0;
    time_options.gravity_y = 0.0;
    time_options.gravity_z = 0.0;
    time_options.non_orthogonal_treatment =
        configuration.treatment;
    time_options.n_non_orthogonal_correctors =
        configuration.nonorthogonal_correctors;
    time_options.pressure_velocity_coupling =
        configuration.coupling;
    time_options.n_pressure_correctors = 2;
    time_options.n_outer_correctors = 1;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 300;
    linear_options.tolerance = 1.0e-10;
    linear_options.preconditioner = configuration.preconditioner;

    const auto boundary_conditions = cavity_boundary_conditions();
    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);
    double local_lid_velocity = 0.0;
    for (const auto& [patch_id, values] : boundary_cache.value)
    {
        if (mesh->boundary_patch_name(patch_id) != "ymax")
        {
            continue;
        }
        for (const auto& value : values)
        {
            local_lid_velocity += value.x;
        }
    }
    if (global_sum(comm, local_lid_velocity) <= 0.0)
    {
        throw std::runtime_error(
            "Pressure-velocity benchmark lost its moving-lid values.");
    }

    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, boundary_conditions,
        time_options, linear_options);
    solver.initialize_heated_box(0.0, 0.0);
    comm->barrier();
    const auto solve_start = MPI_Wtime();
    solver.step();
    comm->barrier();
    const auto total_end = MPI_Wtime();

    const auto statistics = solver.last_step_statistics();
    auto record = base_record(
        options, run_id, "pressure_velocity",
        options.pressure_velocity, comm->getSize());
    record.global_cells =
        static_cast<long long>(
            mesh->owned_cell_map()->getGlobalNumElements());
    record.treatment =
        SimpleFluid::FVM::to_string(configuration.treatment);
    record.coupling =
        SimpleFluid::to_string(configuration.coupling);
    record.preconditioner = configuration.preconditioner_name;
    record.linear_tolerance = linear_options.tolerance;
    record.max_linear_iterations = linear_options.max_iterations;
    record.nonorthogonal_correctors =
        configuration.nonorthogonal_correctors;
    record.pressure_correctors = time_options.n_pressure_correctors;
    record.outer_correctors = time_options.n_outer_correctors;
    record.repetition = repetition;
    record.nonlinear_iterations =
        global_max(comm, statistics.nonlinear_iterations);
    record.linear_solves =
        global_max(comm, statistics.linear_solves);
    record.krylov_iterations =
        global_max(comm, statistics.krylov_iterations);
    record.setup_seconds =
        global_max(comm, solve_start - setup_start);
    record.solve_seconds =
        global_max(comm, total_end - solve_start);
    record.total_seconds =
        global_max(comm, total_end - total_start);
    record.momentum_residual =
        global_max(comm, statistics.momentum);
    record.pressure_residual =
        global_max(comm, statistics.pressure);
    record.temperature_residual =
        global_max(comm, statistics.temperature);
    record.continuity_residual =
        global_max(comm, statistics.continuity);
    record.achieved_tolerance =
        global_max(comm, statistics.achieved_tolerance);
    const auto local_converged = statistics.converged ? 1 : 0;
    record.converged =
        global_sum(comm, local_converged) == comm->getSize();
    add_memory_metrics(record, comm);
    return record;
}

void emit_record(const SimpleFluid::Benchmark::Record& record,
                 SimpleFluid::Benchmark::CsvWriter* writer,
                 int rank)
{
    if (rank != 0)
    {
        return;
    }
    writer->append(record);
    std::cout
        << record.benchmark_case << ' '
        << record.treatment << ' '
        << record.coupling << " cells=" << record.global_cells
        << " ranks=" << record.mpi_ranks
        << " time=" << record.total_seconds
        << "s iterations=" << record.krylov_iterations
        << " converged=" << std::boolalpha << record.converged
        << '\n';
}

bool selected_configuration(
    const Options& options,
    const SolverConfiguration& configuration)
{
    if (options.configuration == "all")
    {
        return true;
    }
    if (options.configuration == "coupled")
    {
        return configuration.coupling
            == SimpleFluid::PressureVelocityCoupling::CoupledKrylov;
    }
    if (configuration.coupling
        == SimpleFluid::PressureVelocityCoupling::CoupledKrylov)
    {
        return false;
    }
    return options.configuration
        == SimpleFluid::FVM::to_string(configuration.treatment);
}

} // namespace

int main(int argc, char** argv)
{
    Tpetra::ScopeGuard scope(&argc, &argv);
    const auto comm = Tpetra::getDefaultComm();
    const auto rank = comm->getRank();

    try
    {
        const auto options =
            parse_options(argc, argv, comm->getSize());
        const auto run_id =
            SimpleFluid::Benchmark::utc_timestamp()
          + "-" + std::to_string(getpid());
        std::unique_ptr<SimpleFluid::Benchmark::CsvWriter> writer;
        if (rank == 0)
        {
            writer =
                std::make_unique<SimpleFluid::Benchmark::CsvWriter>(
                    options.output);
        }

        const std::array<SolverConfiguration, 3> diffusion_configurations{{
            {SimpleFluid::FVM::NonOrthogonalTreatment::Explicit,
             SimpleFluid::LinearPreconditioner::MueLu,
             SimpleFluid::PressureVelocityCoupling::PISO,
             4, "MueLu"},
            {SimpleFluid::FVM::NonOrthogonalTreatment::Implicit,
             SimpleFluid::LinearPreconditioner::None,
             SimpleFluid::PressureVelocityCoupling::PISO,
             0, "none"},
            {SimpleFluid::FVM::NonOrthogonalTreatment::Hybrid,
             SimpleFluid::LinearPreconditioner::None,
             SimpleFluid::PressureVelocityCoupling::PISO,
             4, "none"}
        }};
        const std::array<SolverConfiguration, 4> pressure_configurations{{
            diffusion_configurations[0],
            diffusion_configurations[1],
            diffusion_configurations[2],
            {SimpleFluid::FVM::NonOrthogonalTreatment::Implicit,
             SimpleFluid::LinearPreconditioner::None,
             SimpleFluid::PressureVelocityCoupling::CoupledKrylov,
             0, "Schur(Ifpack2+MueLu)"}
        }};

        if (options.selection == CaseSelection::All
            || options.selection == CaseSelection::Diffusion)
        {
            for (const auto shear : options.shears)
            {
                for (const auto& configuration :
                     diffusion_configurations)
                {
                    if (!selected_configuration(options, configuration))
                    {
                        continue;
                    }
                    for (int warmup = 0;
                         warmup < options.warmups;
                         ++warmup)
                    {
                        (void)run_diffusion(
                            options, run_id, configuration,
                            shear, -1);
                    }
                    for (int repetition = 0;
                         repetition < options.repetitions;
                         ++repetition)
                    {
                        emit_record(
                            run_diffusion(
                                options, run_id, configuration,
                                shear, repetition),
                            writer.get(), rank);
                    }
                }
            }
        }

        if (options.selection == CaseSelection::All
            || options.selection == CaseSelection::PressureVelocity)
        {
            for (const auto& configuration :
                 pressure_configurations)
            {
                if (!selected_configuration(options, configuration))
                {
                    continue;
                }
                for (int warmup = 0;
                     warmup < options.warmups;
                     ++warmup)
                {
                    (void)run_pressure_velocity(
                        options, run_id, configuration, -1);
                }
                for (int repetition = 0;
                     repetition < options.repetitions;
                     ++repetition)
                {
                    emit_record(
                        run_pressure_velocity(
                            options, run_id, configuration,
                            repetition),
                        writer.get(), rank);
                }
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "Rank " << rank
                  << " benchmark failure: " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    return 0;
}
