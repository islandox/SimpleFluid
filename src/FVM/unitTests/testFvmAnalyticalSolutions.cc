/**
 * @file testFvmAnalyticalSolutions.cc
 * @brief Analytical and manufactured-solution tests for FVM operators.
 */

#include <gtest/gtest.h>

#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/VectorCellField.hh"
#include "fields/VectorFaceField.hh"
#include "FVM/FvmOperators.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "solvers/BelosLinearSolver.hh"
#include "utils/ErrorNorms.hh"
#include "utils/testing_environment.hh"

#include <cmath>
#include <stdexcept>
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
constexpr double burgers_amplitude = 0.2;
constexpr double burgers_background = 1.0;

SimpleFluid::SP<MeshType> make_unit_box_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(2, 2, 2, 0.5));
}

SimpleFluid::SP<MeshType> make_taylor_green_mesh()
{
    constexpr int n_cells = 8;
    const auto mesh_size = 2.0 * pi / static_cast<double>(n_cells);
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(n_cells, n_cells, 1, mesh_size));
}

MeshType::local_ordinal_type boundary_face_lid(const MeshType& mesh,
                                               const char* boundary_name)
{
    for (const auto& [patch_id, patch] : mesh.boundary_patches())
    {
        if (mesh.boundary_patch_name(patch_id) == boundary_name
            && !patch.face_lids.empty())
        {
            return patch.face_lids.front();
        }
    }

    throw std::runtime_error("Requested boundary face was not found.");
}

SimpleFluid::SP<MeshType> make_periodic_line_mesh(int n_cells)
{
    const auto dx = 1.0 / static_cast<double>(n_cells);
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(n_cells, 1, 1, dx));

    const auto xmin_face = boundary_face_lid(*mesh, "xmin");
    const auto xmax_face = boundary_face_lid(*mesh, "xmax");
    const auto xmin_owner = mesh->owner_cell(xmin_face);
    const auto xmax_owner = mesh->owner_cell(xmax_face);
    mesh->set_periodic_face(xmin_face, xmax_owner);
    mesh->set_periodic_face(xmax_face, xmin_owner);

    return mesh;
}

double affine_scalar(const SimpleFluid::vec3<>& point)
{
    return 1.0 + 2.0 * point.x - 3.0 * point.y + 4.0 * point.z;
}

double scalar_source(const SimpleFluid::vec3<>& point)
{
    return 0.25 + point.x - 0.5 * point.y + 0.75 * point.z;
}

double constant_source_diffusion_exact(double x,
                                       double mesh_size,
                                       double source,
                                       double diffusivity)
{
    const auto scaled_source = source / diffusivity;
    return 0.5 * scaled_source * x * (1.0 - x)
         + scaled_source * mesh_size * mesh_size / 8.0;
}

SimpleFluid::vec3<> affine_vector(const SimpleFluid::vec3<>& point)
{
    return {1.0 + point.x, 3.0 + 2.0 * point.y, 2.0 - point.z};
}

SimpleFluid::vec3<> vector_source(const SimpleFluid::vec3<>& point)
{
    return {1.0 + point.x, point.y - point.z, 2.0 - 0.5 * point.x};
}

SimpleFluid::vec3<> taylor_green_velocity(const SimpleFluid::vec3<>& point)
{
    return {
        std::sin(point.x) * std::cos(point.y),
       -std::cos(point.x) * std::sin(point.y),
        0.0
    };
}

double burgers_initial(double x)
{
    return burgers_background + burgers_amplitude * std::sin(2.0 * pi * x);
}

double viscous_burgers_exact(double x, double time, double viscosity)
{
    const auto heat_kernel = std::exp(-x * x / (4.0 * viscosity * time));
    const auto denominator =
        std::sqrt(4.0 * pi * viscosity * time) + heat_kernel;
    return heat_kernel * x / (time * denominator);
}

double burgers_exact(double x, double time)
{
    double foot = x - time * burgers_initial(x);
    for (int iteration = 0; iteration < 12; ++iteration)
    {
        const auto sine = std::sin(2.0 * pi * foot);
        const auto cosine = std::cos(2.0 * pi * foot);
        const auto residual =
            foot + time * (burgers_background + burgers_amplitude * sine) - x;
        const auto jacobian =
            1.0 + time * burgers_amplitude * 2.0 * pi * cosine;
        foot -= residual / jacobian;
    }

    return burgers_initial(foot);
}

FieldType make_viscous_burgers_solution(SimpleFluid::SP<MeshType> mesh,
                                        double time,
                                        double viscosity)
{
    FieldType solution(mesh, "viscous_burgers");
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        solution.set_value(
            lid, viscous_burgers_exact(mesh->cell_centroid(lid).x,
                                       time,
                                       viscosity));
    }
    solution.sync_ghosts();

    return solution;
}

FieldType make_burgers_initial_solution(SimpleFluid::SP<MeshType> mesh)
{
    FieldType solution(mesh, "burgers");
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        solution.set_value(lid, burgers_initial(mesh->cell_centroid(lid).x));
    }
    mesh->sync_periodic_boundaries(solution);

    return solution;
}

void overwrite_owned_values(FieldType& target, const FieldType& source)
{
    if (target.mesh_ptr().get() != source.mesh_ptr().get())
    {
        throw std::invalid_argument(
            "Cannot overwrite Burgers solution values across different meshes.");
    }

    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(
             target.mesh().num_owned_cells());
         ++lid)
    {
        target.set_value(lid, source.value(lid));
    }
    target.mesh().sync_periodic_boundaries(target);
}

void expect_burgers_near_exact(const FieldType& numerical,
                               double time,
                               double l2_tolerance,
                               double linf_tolerance)
{
    auto exact = [time](SimpleFluid::vec3<> position)
    {
        return burgers_exact(position.x, time);
    };

    EXPECT_LT(SimpleFluid::l2_error(numerical, exact), l2_tolerance)
        << "at time " << time;
    EXPECT_LT(SimpleFluid::linf_error(numerical, exact), linf_tolerance)
        << "at time " << time;
}

SimpleFluid::FaceField<Pack> burgers_fluxes(const FieldType& solution)
{
    const auto& mesh = solution.mesh();
    SimpleFluid::FaceField<Pack> fluxes(solution.mesh_ptr(), "burgers_flux");
    fluxes.put_scalar(0.0);

    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh.num_faces());
         ++fid)
    {
        if (!fluxes.is_owned_face(fid))
        {
            continue;
        }

        const auto normal_x = mesh.face_normal(fid).x;
        if (std::abs(normal_x) < 0.5)
        {
            continue;
        }

        const auto owner = mesh.owner_cell(fid);
        const auto left_cell =
            normal_x > 0.0
                ? owner
                : mesh.opposite_or_periodic_neighbor_cell(fid, owner);
        const auto left_value = solution.local_value(left_cell);
        fluxes.set_value(fid,
                         normal_x * 0.5 * left_value * left_value
                       * mesh.face_area(fid));
    }

    return fluxes;
}

SimpleFluid::FaceField<Pack> burgers_transport_fluxes(
    const FieldType& solution)
{
    const auto& mesh = solution.mesh();
    SimpleFluid::FaceField<Pack> fluxes(solution.mesh_ptr(),
                                        "burgers_transport_flux");
    fluxes.put_scalar(0.0);

    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh.num_faces());
         ++fid)
    {
        if (!fluxes.is_owned_face(fid))
        {
            continue;
        }

        const auto normal_x = mesh.face_normal(fid).x;
        if (std::abs(normal_x) < 0.5)
        {
            continue;
        }

        const auto owner = mesh.owner_cell(fid);
        const auto left_cell =
            normal_x > 0.0
                ? owner
                : mesh.opposite_or_periodic_neighbor_cell(fid, owner);
        const auto left_value = solution.local_value(left_cell);
        fluxes.set_value(fid,
                         normal_x * 0.5 * left_value * mesh.face_area(fid));
    }

    return fluxes;
}

SimpleFluid::FaceField<Pack> viscous_burgers_transport_fluxes(
    const FieldType& solution,
    double time,
    double viscosity)
{
    const auto& mesh = solution.mesh();
    SimpleFluid::FaceField<Pack> fluxes(solution.mesh_ptr(),
                                        "viscous_burgers_transport_flux");
    fluxes.put_scalar(0.0);

    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh.num_faces());
         ++fid)
    {
        if (!fluxes.is_owned_face(fid))
        {
            continue;
        }

        const auto normal_x = mesh.face_normal(fid).x;
        if (std::abs(normal_x) < 0.5)
        {
            continue;
        }

        const auto owner = mesh.owner_cell(fid);
        Pack::scalar_type left_value = 0.0;
        if (normal_x > 0.0)
        {
            left_value = solution.local_value(owner);
        }
        else if (mesh.is_interior_face(fid))
        {
            const auto left_cell =
                mesh.opposite_or_periodic_neighbor_cell(fid, owner);
            left_value = solution.local_value(left_cell);
        }
        else
        {
            left_value = viscous_burgers_exact(mesh.face_centroid(fid).x,
                                               time,
                                               viscosity);
        }

        fluxes.set_value(fid,
                         normal_x * 0.5 * left_value * mesh.face_area(fid));
    }

    return fluxes;
}

FieldType advance_burgers_explicit(const FieldType& old_solution,
                                   Pack::scalar_type time_step)
{
    const auto& mesh = old_solution.mesh();
    const auto fluxes = burgers_fluxes(old_solution);
    FieldType new_solution(old_solution.mesh_ptr(), "burgers_next");

    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh.num_owned_cells());
         ++lid)
    {
        const auto balance =
            SimpleFluid::FvmOperators::cell_flux_balance(mesh, fluxes, lid);
        new_solution.set_value(lid,
                               old_solution.value(lid)
                             - time_step * balance / mesh.cell_volume(lid));
    }
    mesh.sync_periodic_boundaries(new_solution);

    return new_solution;
}

FieldType solve_burgers_semi_implicit(const FieldType& old_solution,
                                      Pack::scalar_type time_step,
                                      bool& converged)
{
    const auto& mesh = old_solution.mesh();
    const auto fluxes = burgers_transport_fluxes(old_solution);
    auto boundary_value =
        [](int, std::size_t) -> Pack::scalar_type
    {
        return 0.0;
    };

    const auto system = SimpleFluid::FvmOperators::transport_system<Pack>(
        old_solution, fluxes, time_step, 0.0, boundary_value);

    FieldType new_solution(old_solution.mesh_ptr(), "burgers_semi_implicit");
    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-13;
    converged = SimpleFluid::solve_linear_system<Pack>(
        Teuchos::rcp_implicit_cast<const Pack::matrix_type>(system.matrix),
        *system.rhs,
        new_solution.owned_data(),
        options);
    mesh.sync_periodic_boundaries(new_solution);

    return new_solution;
}

FieldType solve_viscous_burgers_semi_implicit(
    const FieldType& old_solution,
    Pack::scalar_type old_time,
    Pack::scalar_type time_step,
    Pack::scalar_type viscosity,
    bool& converged)
{
    const auto& mesh = old_solution.mesh();
    const auto fluxes =
        viscous_burgers_transport_fluxes(old_solution, old_time, viscosity);
    const auto new_time = old_time + time_step;
    auto boundary_value =
        [&](int patch_id, std::size_t in_patch_id) -> Pack::scalar_type
    {
        const auto face_lid =
            mesh.boundary_face_patch(patch_id).face_lids[in_patch_id];
        return viscous_burgers_exact(mesh.face_centroid(face_lid).x,
                                     new_time,
                                     viscosity);
    };

    const auto system = SimpleFluid::FvmOperators::transport_system<Pack>(
        old_solution, fluxes, time_step, viscosity, boundary_value);

    FieldType new_solution(old_solution.mesh_ptr(), "viscous_burgers_next");
    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-13;
    converged = SimpleFluid::solve_linear_system<Pack>(
        Teuchos::rcp_implicit_cast<const Pack::matrix_type>(system.matrix),
        *system.rhs,
        new_solution.owned_data(),
        options);
    mesh.sync_periodic_boundaries(new_solution);

    return new_solution;
}

} // namespace

TEST(FvmAnalyticalSolutionsTest, FaceSampledAffineVelocityHasExactDivergence)
{
    auto mesh = make_unit_box_mesh();
    SimpleFluid::VectorFaceField<Pack> face_velocity(mesh, "face_velocity");

    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        if (!mesh->is_owned_face(fid))
        {
            continue;
        }

        const auto& center = mesh->face_centroid(fid);
        face_velocity.set_value(fid, {center.x, 2.0 * center.y, 3.0 * center.z});
    }

    SimpleFluid::FaceField<Pack> face_fluxes(mesh, "face_flux");
    SimpleFluid::FvmOperators::normal_face_fluxes(face_velocity, face_fluxes);
    const auto divergence =
        SimpleFluid::FvmOperators::cell_divergence_from_fluxes<Pack>(
            *mesh, face_fluxes);

    ASSERT_EQ(divergence.size(), mesh->num_owned_cells());
    for (const auto value : divergence)
    {
        EXPECT_NEAR(value, 6.0, 1.0e-12);
    }
}

TEST(FvmAnalyticalSolutionsTest, TaylorGreenVortexIsDiscreteDivergenceFree)
{
    auto mesh = make_taylor_green_mesh();
    SimpleFluid::VectorFaceField<Pack> face_velocity(mesh, "taylor_green");

    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        if (!mesh->is_owned_face(fid))
        {
            continue;
        }

        face_velocity.set_value(
            fid, taylor_green_velocity(mesh->face_centroid(fid)));
    }

    SimpleFluid::FaceField<Pack> face_fluxes(mesh, "taylor_green_flux");
    SimpleFluid::FvmOperators::normal_face_fluxes(face_velocity, face_fluxes);
    const auto divergence =
        SimpleFluid::FvmOperators::cell_divergence_from_fluxes<Pack>(
            *mesh, face_fluxes);

    ASSERT_EQ(divergence.size(), mesh->num_owned_cells());
    for (const auto value : divergence)
    {
        EXPECT_NEAR(value, 0.0, 1.0e-12);
    }
}

TEST(FvmAnalyticalSolutionsTest, SemiImplicitDiffusionPreservesAffineScalar)
{
    auto mesh = make_unit_box_mesh();
    FieldType old_values(mesh, "old_values");

    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        old_values.set_value(lid, affine_scalar(mesh->cell_centroid(lid)));
    }
    old_values.sync_ghosts();

    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    auto boundary_value =
        [&](int patch_id, std::size_t in_patch_id) -> Pack::scalar_type
    {
        const auto face_lid =
            mesh->boundary_face_patch(patch_id).face_lids[in_patch_id];
        return affine_scalar(mesh->face_centroid(face_lid));
    };

    auto system = SimpleFluid::FvmOperators::transport_system<Pack>(
        old_values, zero_fluxes, 0.25, 0.7, boundary_value);

    Pack::vector_type solution(mesh->owned_cell_map(), true);
    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-13;
    ASSERT_TRUE(SimpleFluid::solve_linear_system<Pack>(
        Teuchos::rcp_implicit_cast<const Pack::matrix_type>(system.matrix),
        *system.rhs,
        solution,
        options));

    const auto data = solution.getData();
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        EXPECT_NEAR(data[static_cast<std::size_t>(lid)],
                    affine_scalar(mesh->cell_centroid(lid)),
                    1.0e-10);
    }
}

TEST(FvmAnalyticalSolutionsTest, ScalarTransportSourceMatchesExactTransient)
{
    auto mesh = make_unit_box_mesh();
    FieldType old_values(mesh, "old_values");

    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        old_values.set_value(lid, affine_scalar(mesh->cell_centroid(lid)));
    }
    old_values.sync_ghosts();

    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    constexpr double time_step = 0.125;
    constexpr double diffusivity = 0.7;
    auto boundary_value =
        [&](int patch_id, std::size_t in_patch_id) -> Pack::scalar_type
    {
        const auto face_lid =
            mesh->boundary_face_patch(patch_id).face_lids[in_patch_id];
        const auto& point = mesh->face_centroid(face_lid);
        return affine_scalar(point) + time_step * scalar_source(point);
    };
    auto source =
        [&](MeshType::local_ordinal_type cell_lid) -> Pack::scalar_type
    {
        return scalar_source(mesh->cell_centroid(cell_lid));
    };

    auto system = SimpleFluid::FvmOperators::transport_system<Pack>(
        old_values, zero_fluxes, time_step, diffusivity,
        boundary_value, source);

    FieldType numerical(mesh, "forced_scalar");
    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-13;
    ASSERT_TRUE(SimpleFluid::solve_linear_system<Pack>(
        Teuchos::rcp_implicit_cast<const Pack::matrix_type>(system.matrix),
        *system.rhs,
        numerical.owned_data(),
        options));
    mesh->sync_periodic_boundaries(numerical);

    auto exact = [](SimpleFluid::vec3<> point)
    {
        return affine_scalar(point) + time_step * scalar_source(point);
    };

    EXPECT_LT(SimpleFluid::l2_error(numerical, exact), 1.0e-12);
    EXPECT_LT(SimpleFluid::linf_error(numerical, exact), 1.0e-12);
}

TEST(FvmAnalyticalSolutionsTest, OneDimensionalDiffusionWithConstantSourceMatchesQuadratic)
{
    constexpr int n_cells = 16;
    constexpr double mesh_size = 1.0 / static_cast<double>(n_cells);
    constexpr double diffusivity = 0.25;
    constexpr double source_value = 0.5;
    constexpr double time_step = 0.25;

    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(n_cells, 1, 1, mesh_size));
    FieldType old_values(mesh, "constant_source_diffusion");

    auto exact = [](SimpleFluid::vec3<> point)
    {
        return constant_source_diffusion_exact(point.x, mesh_size,
                                               source_value, diffusivity);
    };

    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        old_values.set_value(lid, exact(mesh->cell_centroid(lid)));
    }
    old_values.sync_ghosts();

    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    auto boundary_value =
        [&](int patch_id, std::size_t in_patch_id) -> Pack::scalar_type
    {
        const auto face_lid =
            mesh->boundary_face_patch(patch_id).face_lids[in_patch_id];
        const auto& normal = mesh->face_normal(face_lid);
        if (std::abs(normal.x) > 0.5)
        {
            return 0.0;
        }
        return exact(mesh->face_centroid(face_lid));
    };
    auto source =
        [](MeshType::local_ordinal_type) -> Pack::scalar_type
    {
        return source_value;
    };

    auto system = SimpleFluid::FvmOperators::transport_system<Pack>(
        old_values, zero_fluxes, time_step, diffusivity,
        boundary_value, source);

    FieldType numerical(mesh, "constant_source_solution");
    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-13;
    ASSERT_TRUE(SimpleFluid::solve_linear_system<Pack>(
        Teuchos::rcp_implicit_cast<const Pack::matrix_type>(system.matrix),
        *system.rhs,
        numerical.owned_data(),
        options));
    mesh->sync_periodic_boundaries(numerical);

    EXPECT_LT(SimpleFluid::l2_error(numerical, exact), 1.0e-11);
    EXPECT_LT(SimpleFluid::linf_error(numerical, exact), 1.0e-11);
}

TEST(FvmAnalyticalSolutionsTest, VectorTransportSourceMatchesExactTransient)
{
    auto mesh = make_unit_box_mesh();
    VectorFieldType old_values(mesh, "old_values");

    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        old_values.set_value(lid, affine_vector(mesh->cell_centroid(lid)));
    }
    old_values.sync_ghosts();

    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    constexpr double time_step = 0.125;
    constexpr double diffusivity = 0.7;
    auto boundary_value =
        [&](int patch_id, std::size_t in_patch_id)
            -> SimpleFluid::vec3<Pack::scalar_type>
    {
        const auto face_lid =
            mesh->boundary_face_patch(patch_id).face_lids[in_patch_id];
        const auto& point = mesh->face_centroid(face_lid);
        return affine_vector(point) + vector_source(point) * time_step;
    };
    auto source =
        [&](MeshType::local_ordinal_type cell_lid)
            -> SimpleFluid::vec3<Pack::scalar_type>
    {
        return vector_source(mesh->cell_centroid(cell_lid));
    };

    auto system = SimpleFluid::FvmOperators::transport_system<Pack>(
        old_values, zero_fluxes, time_step, diffusivity,
        boundary_value, source);

    VectorFieldType numerical(mesh, "forced_vector");
    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-13;
    ASSERT_TRUE(SimpleFluid::solve_linear_system<Pack>(
        Teuchos::rcp_implicit_cast<const Pack::matrix_type>(system.matrix),
        *system.rhs,
        numerical.owned_data(),
        options));
    mesh->sync_periodic_boundaries(numerical);

    auto exact = [](SimpleFluid::vec3<> point)
    {
        return affine_vector(point) + vector_source(point) * time_step;
    };

    EXPECT_LT(SimpleFluid::l2_error(numerical, exact), 1.0e-12);
    EXPECT_LT(SimpleFluid::linf_error(numerical, exact), 1.0e-12);
}

TEST(FvmAnalyticalSolutionsTest, OneDimensionalBurgersTracksPreShockSineWave)
{
    constexpr int n_cells = 64;
    constexpr double time_step = 2.0e-3;
    const std::vector<int> checkpoints{1, 2, 5, 10, 15, 20, 30, 40, 60};
    auto mesh = make_periodic_line_mesh(n_cells);
    auto solution = make_burgers_initial_solution(mesh);

    int step = 0;
    for (const auto checkpoint : checkpoints)
    {
        ASSERT_GT(checkpoint, step);
        while (step < checkpoint)
        {
            const auto next = advance_burgers_explicit(solution, time_step);
            overwrite_owned_values(solution, next);
            ++step;
        }

        expect_burgers_near_exact(solution,
                                  time_step * static_cast<double>(step),
                                  8.0e-3,
                                  1.6e-2);
    }
}

TEST(FvmAnalyticalSolutionsTest, SemiImplicitBurgersTracksPreShockSineWave)
{
    constexpr int n_cells = 64;
    constexpr double time_step = 2.0e-3;
    const std::vector<int> checkpoints{1, 2, 5, 10, 15, 20, 30, 40, 60};
    auto mesh = make_periodic_line_mesh(n_cells);
    auto solution = make_burgers_initial_solution(mesh);

    int step = 0;
    for (const auto checkpoint : checkpoints)
    {
        ASSERT_GT(checkpoint, step);
        while (step < checkpoint)
        {
            bool converged = false;
            const auto next =
                solve_burgers_semi_implicit(solution, time_step, converged);
            ASSERT_TRUE(converged) << "at step " << step + 1;
            overwrite_owned_values(solution, next);
            ++step;
        }

        expect_burgers_near_exact(solution,
                                  time_step * static_cast<double>(step),
                                  8.0e-3,
                                  1.6e-2);
    }
}

TEST(FvmAnalyticalSolutionsTest, SemiImplicitBurgersWithImplicitViscosityTracksExactSolution)
{
    constexpr int n_cells = 128;
    constexpr double mesh_size = 1.0 / static_cast<double>(n_cells);
    constexpr double initial_time = 0.5;
    constexpr double time_step = 5.0e-4;
    constexpr double viscosity = 0.05;
    const std::vector<int> checkpoints{1, 2, 5, 10, 20, 40, 80, 120, 160, 200};

    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(n_cells, 1, 1, mesh_size));
    auto solution =
        make_viscous_burgers_solution(mesh, initial_time, viscosity);

    int step = 0;
    auto time = initial_time;
    for (const auto checkpoint : checkpoints)
    {
        ASSERT_GT(checkpoint, step);
        while (step < checkpoint)
        {
            bool converged = false;
            const auto next =
                solve_viscous_burgers_semi_implicit(
                    solution, time, time_step, viscosity, converged);
            ASSERT_TRUE(converged) << "at step " << step + 1;
            overwrite_owned_values(solution, next);
            ++step;
            time += time_step;
        }

        auto exact = [time](SimpleFluid::vec3<> position)
        {
            return viscous_burgers_exact(position.x, time, viscosity);
        };

        EXPECT_LT(SimpleFluid::l2_error(solution, exact), 2.5e-3)
            << "at time " << time;
        EXPECT_LT(SimpleFluid::linf_error(solution, exact), 1.0e-2)
            << "at time " << time;
    }
}
