/**
 * @file testFvmAnalyticalSolutions.cc
 * @brief Analytical and manufactured-solution tests for FVM operators.
 */

#include <gtest/gtest.h>

#include "fields/CellField.hh"
#include "fields/FaceField.hh"
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

double burgers_initial(double x)
{
    return burgers_background + burgers_amplitude * std::sin(2.0 * pi * x);
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
    new_solution.sync_ghosts();

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
    new_solution.sync_ghosts();

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

TEST(FvmAnalyticalSolutionsTest, OneDimensionalBurgersTracksPreShockSineWave)
{
    constexpr int n_cells = 64;
    constexpr double time_step = 2.0e-3;
    auto mesh = make_periodic_line_mesh(n_cells);
    FieldType solution(mesh, "burgers");

    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        solution.set_value(lid, burgers_initial(mesh->cell_centroid(lid).x));
    }
    solution.sync_ghosts();

    const auto numerical = advance_burgers_explicit(solution, time_step);
    auto exact = [](SimpleFluid::vec3<> position)
    {
        return burgers_exact(position.x, time_step);
    };

    EXPECT_LT(SimpleFluid::l2_error(numerical, exact), 2.5e-3);
    EXPECT_LT(SimpleFluid::linf_error(numerical, exact), 5.0e-3);
}

TEST(FvmAnalyticalSolutionsTest, SemiImplicitBurgersTracksPreShockSineWave)
{
    constexpr int n_cells = 64;
    constexpr double time_step = 2.0e-3;
    auto mesh = make_periodic_line_mesh(n_cells);
    FieldType solution(mesh, "burgers");

    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        solution.set_value(lid, burgers_initial(mesh->cell_centroid(lid).x));
    }
    solution.sync_ghosts();

    bool converged = false;
    const auto numerical =
        solve_burgers_semi_implicit(solution, time_step, converged);
    ASSERT_TRUE(converged);
    auto exact = [](SimpleFluid::vec3<> position)
    {
        return burgers_exact(position.x, time_step);
    };

    EXPECT_LT(SimpleFluid::l2_error(numerical, exact), 2.5e-3);
    EXPECT_LT(SimpleFluid::linf_error(numerical, exact), 5.0e-3);
}
