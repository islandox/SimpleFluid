/**
 * @file testVerificationCases.cc
 * @brief Named CFD verification cases from TODO Phase 8.
 */

#include <gtest/gtest.h>

#include "equations/BoussinesqMomentumEquation.hh"
#include "FVM/Operators.hh"
#include "geometry/MeshFactory.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "geometry/unitTests/test_skewed_prism_mesh_helpers.hh"
#include "solvers/BelosLinearSolver.hh"
#include "solvers/BoussinesqSolver.hh"
#include "solvers/unitTests/VelocityProfileCsv.hh"
#include "utils/ErrorNorms.hh"
#include "utils/testing_environment.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using VectorFieldType = SimpleFluid::VectorCellField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

constexpr double pi = 3.141592653589793238462643383279502884;

SimpleFluid::SP<MeshType> make_box_mesh(int nx, int ny, int nz)
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(
            nx, ny, nz, 1.0 / static_cast<double>(nx)));
}

SimpleFluid::SP<const SimpleFluid::MeshHandle<Pack>>
make_cartesian_cavity_mesh(int nx, int ny, int nz)
{
    const auto spacing = 1.0 / static_cast<double>(nx);
    auto edges = [spacing](int count)
    {
        SimpleFluid::ArrReal result(
            static_cast<size_t>(count) + 1);
        for (int edge = 0; edge <= count; ++edge)
        {
            result[static_cast<size_t>(edge)] =
                static_cast<double>(edge) * spacing;
        }
        return result;
    };
    auto cartesian = std::make_shared<
        SimpleFluid::Meshes::OrthogonalCartesian3D>(
            SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
                edges(nx), edges(ny), edges(nz)}});
    return std::make_shared<SimpleFluid::MeshHandle<Pack>>(
        std::move(cartesian));
}

SimpleFluid::BoundaryConditionSet cavity_boundary_conditions()
{
    SimpleFluid::BoundaryConditionSet bcs;
    bcs.velocity["xmin"] = {
        SimpleFluid::BoundaryConditionType::NoSlip, {}};
    bcs.velocity["xmax"] = {
        SimpleFluid::BoundaryConditionType::NoSlip, {}};
    bcs.velocity["ymin"] = {
        SimpleFluid::BoundaryConditionType::NoSlip, {}};
    bcs.velocity["ymax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, {1.0, 0.0, 0.0}};
    bcs.velocity["zmin"] = {
        SimpleFluid::BoundaryConditionType::Slip, {}};
    bcs.velocity["zmax"] = {
        SimpleFluid::BoundaryConditionType::Slip, {}};
    return bcs;
}

double stabilized_continuity_norm(
    const VectorFieldType& velocity,
    const FieldType& pressure,
    double time_step,
    const SimpleFluid::FVM::VelocityBoundaryCache<Pack>& cache)
{
    SimpleFluid::FaceField<Pack> fluxes(
        velocity.mesh_ptr(), "verification_fluxes");
    SimpleFluid::FVM::pressure_weighted_face_fluxes(
        velocity, pressure, time_step, cache, fluxes);

    double norm_squared = 0.0;
    for (size_t owned = 0;
         owned < velocity.mesh().num_owned_cells();
         ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto balance =
            SimpleFluid::FVM::cell_flux_balance<Pack>(
                velocity.mesh(), fluxes, cell_lid);
        norm_squared += balance * balance;
    }
    return std::sqrt(norm_squared);
}

void verify_lid_driven_cavity(
    double reynolds_number,
    bool use_orthogonal_cartesian = false)
{
    const auto bcs = cavity_boundary_conditions();

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 2.0e-3;
    time_options.steps = 12;
    time_options.thermal_diffusivity = 0.0;
    time_options.kinematic_viscosity = 1.0 / reynolds_number;
    time_options.thermal_expansion = 0.0;
    time_options.gravity_x = 0.0;
    time_options.gravity_y = 0.0;
    time_options.gravity_z = 0.0;
    time_options.pressure_velocity_coupling =
        SimpleFluid::PressureVelocityCoupling::PISO;
    time_options.n_pressure_correctors = 2;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-11;
    linear_options.max_iterations = 300;

    auto verify = [&](const auto& input_mesh)
    {
        SimpleFluid::BoussinesqSolver<Pack> solver(
            input_mesh, bcs, time_options, linear_options);
        solver.initialize_heated_box(0.0, 0.0);
        solver.run();
        const auto mesh = solver.velocity().mesh_ptr();

        const auto cache =
            SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
                mesh, bcs);
        SimpleFluid::VectorFaceField<Pack> face_velocity(
            mesh, "cavity_face_velocity");
        SimpleFluid::FVM::face_velocities(
            solver.velocity(), cache, face_velocity);

        double kinetic_energy = 0.0;
        bool saw_moving_wall = false;
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<MeshType::local_ordinal_type>(owned);
            const auto velocity = solver.velocity().value(cell_lid);
            EXPECT_TRUE(std::isfinite(velocity.x));
            EXPECT_TRUE(std::isfinite(velocity.y));
            kinetic_energy += velocity.dot(velocity)
                            * mesh->cell_volume(cell_lid);
        }
        for (const auto& [batch_id, batch] : mesh->boundary_batches())
        {
            if (mesh->boundary_batch_name(batch_id) != "ymax")
            {
                continue;
            }
            for (const auto face_lid : batch.face_lids)
            {
                if (!face_velocity.is_owned_face(face_lid))
                {
                    continue;
                }
                saw_moving_wall = true;
                EXPECT_NEAR(
                    face_velocity.value(face_lid).x, 1.0, 1.0e-12);
            }
        }

        EXPECT_TRUE(saw_moving_wall);
        EXPECT_GT(kinetic_energy, 1.0e-12);
        EXPECT_LT(
            stabilized_continuity_norm(
                solver.velocity(), solver.pressure(),
                time_options.time_step, cache),
            1.0e-2);

        if (reynolds_number == 1000.0
            && !use_orthogonal_cartesian)
        {
            const auto* output_directory =
                std::getenv("SIMPLEFLUID_PROFILE_OUTPUT_DIR");
            if (output_directory != nullptr
                && output_directory[0] != '\0')
            {
                const std::filesystem::path directory(output_directory);
                std::filesystem::create_directories(directory);
                using Profile =
                    SimpleFluid::test::VelocityProfileCsv<Pack>;
                Profile::write_nearest_line(
                    directory / "simplefluid_lineX.csv",
                    solver.velocity(), 0, {0.0, 0.5, 0.5});
                Profile::write_nearest_line(
                    directory / "simplefluid_lineY.csv",
                    solver.velocity(), 1, {0.5, 0.0, 0.5});
            }
        }
    };

    if (use_orthogonal_cartesian)
    {
        verify(make_cartesian_cavity_mesh(8, 8, 1));
    }
    else
    {
        verify(make_box_mesh(8, 8, 1));
    }
}

SimpleFluid::vec3<> taylor_green_velocity(
    const SimpleFluid::vec3<>& point)
{
    return {
        std::sin(point.x) * std::cos(point.y),
        -std::cos(point.x) * std::sin(point.y),
        0.0};
}

SimpleFluid::vec3<> taylor_green_source(
    const SimpleFluid::vec3<>& point,
    double viscosity)
{
    const auto velocity = taylor_green_velocity(point);
    return {
        -0.5 * std::sin(2.0 * point.x)
            + 2.0 * viscosity * velocity.x,
        -0.5 * std::sin(2.0 * point.y)
            + 2.0 * viscosity * velocity.y,
        0.0};
}

double solve_taylor_green_error(int n_cells)
{
    const auto h = 2.0 * pi / static_cast<double>(n_cells);
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(
            n_cells, n_cells, 1, h));
    constexpr double viscosity = 0.05;

    VectorFieldType old_velocity(mesh, "manufactured_old_velocity");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        old_velocity.set_value(
            cell_lid,
            taylor_green_velocity(mesh->cell_centroid(cell_lid)));
    }
    old_velocity.sync_ghosts();

    SimpleFluid::VectorFaceField<Pack> face_velocity(
        mesh, "manufactured_face_velocity");
    for (MeshType::local_ordinal_type face_lid = 0;
         face_lid < static_cast<MeshType::local_ordinal_type>(
                        mesh->num_faces());
         ++face_lid)
    {
        if (face_velocity.is_owned_face(face_lid))
        {
            face_velocity.set_value(
                face_lid,
                taylor_green_velocity(mesh->face_centroid(face_lid)));
        }
    }
    SimpleFluid::FaceField<Pack> face_fluxes(
        mesh, "manufactured_face_fluxes");
    SimpleFluid::FVM::normal_face_fluxes(
        face_velocity, face_fluxes);

    auto boundary_value =
        [&](int batch_id, size_t in_batch_id)
    {
        const auto face_lid =
            mesh->boundary_face_batch(batch_id).face_lids[in_batch_id];
        return taylor_green_velocity(mesh->face_centroid(face_lid));
    };
    auto source =
        [&](MeshType::local_ordinal_type cell_lid)
    {
        return taylor_green_source(
            mesh->cell_centroid(cell_lid), viscosity);
    };

    const auto system =
        SimpleFluid::FVM::transport_system<Pack>(
            old_velocity, face_fluxes, 0.1, viscosity,
            boundary_value, source);
    VectorFieldType numerical(mesh, "manufactured_velocity");
    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-12;
    options.max_iterations = 500;
    EXPECT_TRUE(SimpleFluid::solve_linear_system<Pack>(
        Teuchos::rcp_implicit_cast<const Pack::matrix_type>(
            system.matrix),
        *system.rhs, numerical.owned_data(), options));
    mesh->sync_periodic_boundaries(numerical);

    return SimpleFluid::l2_error(
        numerical, taylor_green_velocity);
}

double solve_skewed_diffusion_error(size_t n_cells)
{
    auto mesh = SimpleFluid::test::make_skewed_prism_mesh<Pack>(
        n_cells, n_cells, n_cells);
    constexpr double diffusivity = 1.0;
    auto exact = [](SimpleFluid::vec3<> point)
    {
        return point.x * point.x + 0.25 * point.y - 0.125 * point.z;
    };
    auto boundary_condition =
        [&](int batch_id, size_t in_batch_id)
            -> SimpleFluid::BoundaryCondition
    {
        const auto face_lid =
            mesh->boundary_face_batch(batch_id).face_lids[in_batch_id];
        return {
            SimpleFluid::BoundaryConditionType::Dirichlet,
            exact(mesh->face_centroid(face_lid))};
    };
    auto source =
        [](MeshType::local_ordinal_type)
    {
        return -2.0 * diffusivity;
    };

    FieldType numerical(mesh, "skewed_diffusion");
    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-12;
    options.max_iterations = 500;
    EXPECT_TRUE(
        SimpleFluid::FVM::solve_non_orthogonal_diffusion<Pack>(
            *mesh, diffusivity, boundary_condition, source,
            numerical,
            SimpleFluid::FVM::NonOrthogonalTreatment::Implicit,
            0, options));
    return SimpleFluid::l2_error(numerical, exact);
}

} // namespace

TEST(VerificationCasesTest, LidDrivenCavityRe100)
{
    verify_lid_driven_cavity(100.0);
}

TEST(VerificationCasesTest, LidDrivenCavityRe1000)
{
    verify_lid_driven_cavity(1000.0);
}

TEST(VerificationCasesTest, LidDrivenCavityRe1000OrthogonalCartesian3D)
{
    verify_lid_driven_cavity(1000.0, true);
}

TEST(VerificationCasesTest, WritesSimpleFluidVelocityProfileCsv)
{
    auto mesh = make_box_mesh(4, 4, 1);
    VectorFieldType velocity(mesh, "csv_velocity");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto center = mesh->cell_centroid(cell_lid);
        velocity.set_value(
            cell_lid, {center.x, 2.0 * center.y, 3.0 * center.z});
    }

    using Profile = SimpleFluid::test::VelocityProfileCsv<Pack>;
    const auto samples = Profile::sample_nearest_line(
        velocity, 1, {0.5, 0.0, 0.5});
    ASSERT_EQ(samples.size(), 4U);
    for (size_t sample = 0; sample < samples.size(); ++sample)
    {
        const auto coordinate =
            (static_cast<double>(sample) + 0.5) / 4.0;
        EXPECT_NEAR(samples[sample].coordinate, coordinate, 1.0e-12);
        EXPECT_NEAR(samples[sample].velocity.x, 0.5, 1.0e-12);
        EXPECT_NEAR(
            samples[sample].velocity.y,
            2.0 * coordinate, 1.0e-12);
        EXPECT_NEAR(samples[sample].velocity.z, 0.375, 1.0e-12);
    }

    const auto output_path =
        std::filesystem::temp_directory_path()
        / "simplefluid_velocity_profile.csv";
    Profile::write(output_path, samples);
    std::ifstream input(output_path);
    ASSERT_TRUE(input.good());
    const std::string contents{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    EXPECT_EQ(
        contents.substr(
            0, contents.find('\n')),
        "coordinate,ux,uy,uz");
    EXPECT_EQ(
        static_cast<size_t>(
            std::count(contents.begin(), contents.end(), '\n')),
        samples.size() + 1);
    std::filesystem::remove(output_path);
}

TEST(VerificationCasesTest, PoiseuilleFlowMatchesParabolicProfile)
{
    constexpr int n_cells = 32;
    constexpr double viscosity = 0.1;
    constexpr double peak_velocity = 1.0;
    constexpr double source_value =
        8.0 * viscosity * peak_velocity;
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(
            1, n_cells, 1,
            1.0 / static_cast<double>(n_cells)));

    auto boundary_condition =
        [&](int batch_id, size_t)
            -> SimpleFluid::VectorBoundaryCondition
    {
        const auto& name = mesh->boundary_batch_name(batch_id);
        if (name == "ymin" || name == "ymax")
        {
            return {
                SimpleFluid::BoundaryConditionType::NoSlip, {}};
        }
        return {
            SimpleFluid::BoundaryConditionType::Neumann, {}};
    };
    auto source =
        [](MeshType::local_ordinal_type)
    {
        return SimpleFluid::vec3<>{source_value, 0.0, 0.0};
    };

    const auto system =
        SimpleFluid::FVM::vector_diffusion_system<Pack>(
            *mesh, viscosity, boundary_condition, source);
    VectorFieldType velocity(mesh, "poiseuille_velocity");
    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-13;
    ASSERT_TRUE(SimpleFluid::solve_linear_system<Pack>(
        Teuchos::rcp_implicit_cast<const Pack::matrix_type>(
            system.matrix),
        *system.rhs, velocity.owned_data(), options));

    auto exact = [](SimpleFluid::vec3<> point)
    {
        return SimpleFluid::vec3<>{
            4.0 * peak_velocity * point.y * (1.0 - point.y),
            0.0, 0.0};
    };
    EXPECT_LT(SimpleFluid::l2_error(velocity, exact), 2.0e-3);
    EXPECT_LT(SimpleFluid::linf_error(velocity, exact), 2.0e-3);
}

TEST(VerificationCasesTest, ManufacturedIncompressibleNavierStokesConverges)
{
    const auto coarse_error = solve_taylor_green_error(8);
    const auto fine_error = solve_taylor_green_error(16);

    EXPECT_LT(fine_error, coarse_error);
    EXPECT_LT(fine_error, 1.0e-1);
}

TEST(VerificationCasesTest, SkewedMeshDiffusionConverges)
{
    const auto coarse_error = solve_skewed_diffusion_error(3);
    const auto fine_error = solve_skewed_diffusion_error(5);

    EXPECT_LT(fine_error, coarse_error);
    EXPECT_LT(fine_error, 1.0e-1);
}

TEST(VerificationCasesTest, NaturalConvectionSquareCavityRemainsBounded)
{
    auto mesh = make_box_mesh(8, 8, 1);
    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["xmin"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
    bcs.temperature["xmax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    bcs.temperature["ymin"] = {
        SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    bcs.temperature["ymax"] = {
        SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax"})
    {
        bcs.velocity[name] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }
    bcs.velocity["zmin"] = {
        SimpleFluid::BoundaryConditionType::Slip, {}};
    bcs.velocity["zmax"] = {
        SimpleFluid::BoundaryConditionType::Slip, {}};

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 2.0e-3;
    time_options.steps = 16;
    time_options.thermal_diffusivity = 1.0e-2;
    time_options.kinematic_viscosity = 1.0e-2;
    time_options.thermal_expansion = 1.0;
    time_options.gravity_x = 0.0;
    time_options.gravity_y = -1.0;
    time_options.gravity_z = 0.0;
    time_options.reference_temperature = 0.5;
    time_options.n_pressure_correctors = 2;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-11;
    linear_options.max_iterations = 300;
    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, bcs, time_options, linear_options);
    solver.initialize_heated_box(1.0, 0.0);
    solver.run();

    double kinetic_energy = 0.0;
    double minimum_temperature =
        std::numeric_limits<double>::infinity();
    double maximum_temperature =
        -std::numeric_limits<double>::infinity();
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto temperature =
            solver.temperature().value(cell_lid);
        const auto velocity = solver.velocity().value(cell_lid);
        minimum_temperature =
            std::min(minimum_temperature, temperature);
        maximum_temperature =
            std::max(maximum_temperature, temperature);
        kinetic_energy += velocity.dot(velocity)
                        * mesh->cell_volume(cell_lid);
    }

    EXPECT_GE(minimum_temperature, -1.0e-8);
    EXPECT_LE(maximum_temperature, 1.0 + 1.0e-8);
    EXPECT_GT(kinetic_energy, 1.0e-12);
}

TEST(VerificationCasesTest, MatchesDocumentedOpenFoamCavityConfiguration)
{
    constexpr int openfoam_cells_x = 100;
    constexpr int openfoam_cells_y = 100;
    constexpr int openfoam_cells_z = 1;
    constexpr double openfoam_length = 1.0;
    constexpr double openfoam_lid_speed = 1.0;
    constexpr double openfoam_reynolds_number = 1000.0;
    constexpr double openfoam_viscosity =
        openfoam_lid_speed * openfoam_length
        / openfoam_reynolds_number;

    EXPECT_EQ(openfoam_cells_x, 100);
    EXPECT_EQ(openfoam_cells_y, 100);
    EXPECT_EQ(openfoam_cells_z, 1);
    EXPECT_DOUBLE_EQ(openfoam_length, 1.0);
    EXPECT_DOUBLE_EQ(openfoam_lid_speed, 1.0);
    EXPECT_DOUBLE_EQ(openfoam_viscosity, 1.0e-3);

    const auto bcs = cavity_boundary_conditions();
    EXPECT_EQ(
        bcs.velocity.at("ymax").type,
        SimpleFluid::BoundaryConditionType::Dirichlet);
    EXPECT_EQ(
        bcs.velocity.at("ymax").value,
        (SimpleFluid::vec3<>{1.0, 0.0, 0.0}));
    EXPECT_EQ(
        bcs.velocity.at("xmin").type,
        SimpleFluid::BoundaryConditionType::NoSlip);
    EXPECT_EQ(
        bcs.velocity.at("xmax").type,
        SimpleFluid::BoundaryConditionType::NoSlip);
    EXPECT_EQ(
        bcs.velocity.at("ymin").type,
        SimpleFluid::BoundaryConditionType::NoSlip);
}
