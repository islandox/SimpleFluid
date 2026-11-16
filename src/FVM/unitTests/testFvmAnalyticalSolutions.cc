/**
 * @file testFvmAnalyticalSolutions.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Analytical and manufactured-solution tests for FVM operators.
 * @version 0.1
 * @date 2026-06-03
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "FVM/TransportSystemFactory.hh"
#include <gtest/gtest.h>
#include <type_traits>

#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/VectorCellField.hh"
#include "fields/VectorFaceField.hh"
#include "FVM/Operators.hh"
#include "geometry/mesh/UnstructuredMesh.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "geometry/unitTests/test_skewed_prism_mesh_helpers.hh"
#include "solvers/BelosLinearSolver.hh"
#include "solvers/NonOrthogonalDiffusionSolve.hh"
#include "utils/ErrorNorms.hh"
#include "utils/testing_environment.hh"


#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
template<class Field>
using TransportFactory = SimpleFluid::FVM::TransportSystemFactory<Pack, typename std::remove_cvref_t<Field>::mesh_type>;
using Mesh = SimpleFluid::MeshHandle<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using VectorFieldType = SimpleFluid::VectorCellField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double burgers_amplitude = 0.2;
constexpr double burgers_background = 1.0;

SimpleFluid::SP<Mesh> make_unit_box_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(2, 2, 2, 0.5));
}

SimpleFluid::SP<Mesh> make_taylor_green_mesh()
{
    constexpr int n_cells = 8;
    const auto mesh_size = 2.0 * pi / static_cast<double>(n_cells);
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(n_cells, n_cells, 1, mesh_size));
}

SimpleFluid::SP<Mesh> make_periodic_line_mesh(int n_cells)
{
    const auto dx = 1.0 / static_cast<double>(n_cells);
    SimpleFluid::ArrReal x(static_cast<size_t>(n_cells) + 1);
    for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<double>(i) * dx;
    return std::make_shared<Mesh>(std::make_shared<Mesh::Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{std::move(x), {0., dx}, {0., dx}}},
        SimpleFluid::Vec3D<bool>{{true, false, false}}));
}

SimpleFluid::SP<Mesh> make_sheared_box_mesh(size_t n_cells, double shear)
{
    using Native = SimpleFluid::Meshes::UnstructuredMesh;
    const auto node_id = [=](size_t i, size_t j, size_t k)
    { return static_cast<Native::NodeID>(i + (n_cells + 1) * (j + (n_cells + 1) * k)); };
    const auto h = 1.0 / static_cast<double>(n_cells);
    SimpleFluid::Arr<Native::Vec3> nodes;
    for (size_t k = 0; k <= n_cells; ++k)
        for (size_t j = 0; j <= n_cells; ++j)
            for (size_t i = 0; i <= n_cells; ++i)
            {
                const auto x = static_cast<double>(i) * h;
                const auto y = static_cast<double>(j) * h;
                const auto z = static_cast<double>(k) * h;
                nodes.push_back({x + shear * y, y, z});
            }
    SimpleFluid::Arr<Native::CellDefinition> cells;
    SimpleFluid::Arr<Native::BoundaryFaceDefinition> boundaries;
    for (size_t k = 0; k < n_cells; ++k)
        for (size_t j = 0; j < n_cells; ++j)
            for (size_t i = 0; i < n_cells; ++i)
            {
                const std::array ids{node_id(i,j,k), node_id(i+1,j,k), node_id(i+1,j+1,k), node_id(i,j+1,k),
                    node_id(i,j,k+1), node_id(i+1,j,k+1), node_id(i+1,j+1,k+1), node_id(i,j+1,k+1)};
                cells.push_back({Native::CellType::HEXAHEDRON, {ids.begin(), ids.end()}});
                if (i == 0) boundaries.push_back({{ids[3],ids[7],ids[4],ids[0]}, 1, "xmin"});
                if (i+1 == n_cells) boundaries.push_back({{ids[1],ids[5],ids[6],ids[2]}, 2, "xmax"});
                if (j == 0) boundaries.push_back({{ids[0],ids[4],ids[5],ids[1]}, 3, "ymin"});
                if (j+1 == n_cells) boundaries.push_back({{ids[2],ids[6],ids[7],ids[3]}, 4, "ymax"});
                if (k == 0) boundaries.push_back({{ids[0],ids[1],ids[2],ids[3]}, 5, "zmin"});
                if (k+1 == n_cells) boundaries.push_back({{ids[4],ids[7],ids[6],ids[5]}, 6, "zmax"});
            }
    return std::make_shared<Mesh>(std::make_shared<Native>(
        std::move(nodes), std::move(cells), std::move(boundaries)));
}

double affine_scalar(const SimpleFluid::vec3<>& point)
{
    return 1.0 + 2.0 * point.x - 3.0 * point.y + 4.0 * point.z;
}

double skewed_quadratic_scalar(const SimpleFluid::vec3<>& point)
{
    return 0.1 + point.x * point.x + 0.25 * point.y - 0.125 * point.z;
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

FieldType make_viscous_burgers_solution(SimpleFluid::SP<Mesh> mesh,
                                        double time,
                                        double viscosity)
{
    FieldType solution(mesh, "viscous_burgers");
    for (Mesh::local_ordinal_type lid = 0;
         lid < static_cast<Mesh::local_ordinal_type>(mesh->num_owned_cells());
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

FieldType make_burgers_initial_solution(SimpleFluid::SP<Mesh> mesh)
{
    FieldType solution(mesh, "burgers");
    for (Mesh::local_ordinal_type lid = 0;
         lid < static_cast<Mesh::local_ordinal_type>(mesh->num_owned_cells());
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

    for (Mesh::local_ordinal_type lid = 0;
         lid < static_cast<Mesh::local_ordinal_type>(
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

    for (Mesh::local_ordinal_type fid = 0;
         fid < static_cast<Mesh::local_ordinal_type>(mesh.num_faces());
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

    fluxes.sync_ghosts();
    return fluxes;
}

SimpleFluid::FaceField<Pack> burgers_transport_fluxes(
    const FieldType& solution)
{
    const auto& mesh = solution.mesh();
    SimpleFluid::FaceField<Pack> fluxes(solution.mesh_ptr(),
                                        "burgers_transport_flux");
    fluxes.put_scalar(0.0);

    for (Mesh::local_ordinal_type fid = 0;
         fid < static_cast<Mesh::local_ordinal_type>(mesh.num_faces());
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

    fluxes.sync_ghosts();
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

    for (Mesh::local_ordinal_type fid = 0;
         fid < static_cast<Mesh::local_ordinal_type>(mesh.num_faces());
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

    fluxes.sync_ghosts();
    return fluxes;
}

FieldType advance_burgers_explicit(const FieldType& old_solution,
                                   Pack::scalar_type time_step)
{
    const auto& mesh = old_solution.mesh();
    const auto fluxes = burgers_fluxes(old_solution);
    FieldType new_solution(old_solution.mesh_ptr(), "burgers_next");

    for (Mesh::local_ordinal_type lid = 0;
         lid < static_cast<Mesh::local_ordinal_type>(mesh.num_owned_cells());
         ++lid)
    {
        const auto balance =
            SimpleFluid::FVM::cell_flux_balance(mesh, fluxes, lid);
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
        [](int, size_t) -> Pack::scalar_type
    {
        return 0.0;
    };

    const auto system = TransportFactory<decltype(old_solution)>::assemble(
        typename TransportFactory<decltype(old_solution)>::ScalarTerms{
            .history = {.old_values = old_solution, .time_step = time_step},
            .face_fluxes = fluxes,
            .coefficients = {.diffusivity = 0.0}},
        {.condition = [](int, size_t)
            { return SimpleFluid::BoundaryCondition{SimpleFluid::BoundaryConditionType::Dirichlet, {}}; },
            .value = boundary_value},
        {}, {});

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
        [&](int batch_id, size_t in_batch_id) -> Pack::scalar_type
    {
        const auto face_lid =
            mesh.boundary_face_batch(batch_id).face_lids[in_batch_id];
        return viscous_burgers_exact(mesh.face_centroid(face_lid).x,
                                     new_time,
                                     viscosity);
    };

    const auto system = TransportFactory<decltype(old_solution)>::assemble(
        typename TransportFactory<decltype(old_solution)>::ScalarTerms{
            .history = {.old_values = old_solution, .time_step = time_step},
            .face_fluxes = fluxes,
            .coefficients = {.diffusivity = viscosity}},
        {.condition = [](int, size_t)
            { return SimpleFluid::BoundaryCondition{SimpleFluid::BoundaryConditionType::Dirichlet, {}}; },
            .value = boundary_value},
        {}, {});

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

bool cell_has_exterior_face(const Mesh& mesh,
                            Mesh::local_ordinal_type cell_lid)
{
    for (const auto face_lid : mesh.faces(cell_lid))
    {
        if (mesh.is_exterior_face(face_lid))
        {
            return true;
        }
    }

    return false;
}

} // namespace

/** @brief Verifies face-sampled affine velocity has the exact discrete divergence. */
TEST(FvmAnalyticalSolutionsTest, FaceSampledAffineVelocityHasExactDivergence)
{
    auto mesh = make_unit_box_mesh();
    SimpleFluid::VectorFaceField<Pack> face_velocity(mesh, "face_velocity");

    for (Mesh::local_ordinal_type fid = 0;
         fid < static_cast<Mesh::local_ordinal_type>(mesh->num_faces());
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
    SimpleFluid::FVM::normal_face_fluxes(face_velocity, face_fluxes);
    face_fluxes.sync_ghosts();
    const auto divergence =
        SimpleFluid::FVM::cell_divergence_from_fluxes<Pack>(
            *mesh, face_fluxes);

    ASSERT_EQ(divergence.size(), mesh->num_owned_cells());
    for (const auto value : divergence)
    {
        EXPECT_NEAR(value, 6.0, 1.0e-12);
    }
}

/** @brief Verifies exact affine-velocity divergence on a skewed prism mesh. */
TEST(FvmAnalyticalSolutionsTest, FaceSampledAffineVelocityHasExactDivergenceOnSkewedPrisms)
{
    auto mesh = SimpleFluid::test::make_skewed_prism_mesh<Pack>();
    ASSERT_TRUE(std::holds_alternative<Mesh::UnstructuredPtr>(mesh->variant()));
    const auto& geometry = *std::get<Mesh::UnstructuredPtr>(mesh->variant());
    // This fixture retains native input order for the handle's local ordinals.
    ASSERT_EQ(geometry.num_cells(), mesh->num_local_cells());
    ASSERT_EQ(geometry.num_faces(), mesh->num_faces());
    ASSERT_GE(mesh->num_owned_cells(), 3u * 3u * 3u);
    EXPECT_EQ(mesh->boundary_batches().size(), 6u);

    bool saw_triangular_face = false;
    size_t boundary_cells = 0;
    size_t interior_cells = 0;
    for (Mesh::local_ordinal_type lid = 0;
         lid < static_cast<Mesh::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        EXPECT_EQ(geometry.cell_type(geometry.cell_id(static_cast<size_t>(lid))),
                  SimpleFluid::MeshUtils::CellType::TRIPRISM);
        EXPECT_GT(mesh->cell_volume(lid), 0.0);
        if (cell_has_exterior_face(*mesh, lid))
        {
            ++boundary_cells;
        }
        else
        {
            ++interior_cells;
        }
    }

    SimpleFluid::VectorFaceField<Pack> face_velocity(mesh, "skewed_face_velocity");
    for (Mesh::local_ordinal_type fid = 0;
         fid < static_cast<Mesh::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        saw_triangular_face =
            saw_triangular_face
         || geometry.face_nodes(geometry.face_id(static_cast<size_t>(fid))).size() == 3;
        if (!mesh->is_owned_face(fid))
        {
            continue;
        }

        const auto center = mesh->face_centroid(fid);
        face_velocity.set_value(fid, {center.x, 2.0 * center.y, 3.0 * center.z});
    }
    EXPECT_TRUE(saw_triangular_face);
    EXPECT_GT(boundary_cells, 0u);
    EXPECT_GT(interior_cells, 0u);

    SimpleFluid::FaceField<Pack> face_fluxes(mesh, "skewed_face_flux");
    SimpleFluid::FVM::normal_face_fluxes(face_velocity, face_fluxes);
    face_fluxes.sync_ghosts();
    const auto divergence =
        SimpleFluid::FVM::cell_divergence_from_fluxes<Pack>(
            *mesh, face_fluxes);

    ASSERT_EQ(divergence.size(), mesh->num_owned_cells());
    for (const auto value : divergence)
    {
        EXPECT_NEAR(value, 6.0, 1.0e-12);
    }
}

/** @brief Verifies the sampled Taylor-Green vortex is discretely divergence-free. */
TEST(FvmAnalyticalSolutionsTest, TaylorGreenVortexIsDiscreteDivergenceFree)
{
    auto mesh = make_taylor_green_mesh();
    SimpleFluid::VectorFaceField<Pack> face_velocity(mesh, "taylor_green");

    for (Mesh::local_ordinal_type fid = 0;
         fid < static_cast<Mesh::local_ordinal_type>(mesh->num_faces());
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
    SimpleFluid::FVM::normal_face_fluxes(face_velocity, face_fluxes);
    face_fluxes.sync_ghosts();
    const auto divergence =
        SimpleFluid::FVM::cell_divergence_from_fluxes<Pack>(
            *mesh, face_fluxes);

    ASSERT_EQ(divergence.size(), mesh->num_owned_cells());
    for (const auto value : divergence)
    {
        EXPECT_NEAR(value, 0.0, 1.0e-12);
    }
}

/**
 * @brief Verifies semi-implicit diffusion preserves an affine scalar field (zero Laplacian).
 */
TEST(FvmAnalyticalSolutionsTest, SemiImplicitDiffusionPreservesAffineScalar)
{
    auto mesh = make_unit_box_mesh();
    FieldType old_values(mesh, "old_values");

    for (Mesh::local_ordinal_type lid = 0;
         lid < static_cast<Mesh::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        old_values.set_value(lid, affine_scalar(mesh->cell_centroid(lid)));
    }
    old_values.sync_ghosts();

    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "zero_fluxes");
    auto boundary_value =
        [&](int batch_id, size_t in_batch_id) -> Pack::scalar_type
    {
        const auto face_lid =
            mesh->boundary_face_batch(batch_id).face_lids[in_batch_id];
        return affine_scalar(mesh->face_centroid(face_lid));
    };

    auto system = TransportFactory<decltype(old_values)>::assemble(
        typename TransportFactory<decltype(old_values)>::ScalarTerms{
            .history = {.old_values = old_values, .time_step = 0.25},
            .face_fluxes = zero_fluxes,
            .coefficients = {.diffusivity = 0.7}},
        {.condition = [](int, size_t)
            { return SimpleFluid::BoundaryCondition{SimpleFluid::BoundaryConditionType::Dirichlet, {}}; },
            .value = boundary_value},
        {}, {});

    Pack::vector_type solution(mesh->owned_cell_map(), true);
    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-13;
    ASSERT_TRUE(SimpleFluid::solve_linear_system<Pack>(
        Teuchos::rcp_implicit_cast<const Pack::matrix_type>(system.matrix),
        *system.rhs,
        solution,
        options));

    const auto data = solution.getData();
    for (Mesh::local_ordinal_type lid = 0;
         lid < static_cast<Mesh::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        EXPECT_NEAR(data[static_cast<size_t>(lid)],
                    affine_scalar(mesh->cell_centroid(lid)),
                    1.0e-10);
    }
}

/**
 * @brief Verifies scalar transport with a source term reproduces the exact transient solution.
 */
TEST(FvmAnalyticalSolutionsTest, ScalarTransportSourceMatchesExactTransient)
{
    auto mesh = make_unit_box_mesh();
    FieldType old_values(mesh, "old_values");

    for (Mesh::local_ordinal_type lid = 0;
         lid < static_cast<Mesh::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        old_values.set_value(lid, affine_scalar(mesh->cell_centroid(lid)));
    }
    old_values.sync_ghosts();

    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "zero_fluxes");
    constexpr double time_step = 0.125;
    constexpr double diffusivity = 0.7;
    auto boundary_value =
        [&](int batch_id, size_t in_batch_id) -> Pack::scalar_type
    {
        const auto face_lid =
            mesh->boundary_face_batch(batch_id).face_lids[in_batch_id];
        const auto& point = mesh->face_centroid(face_lid);
        return affine_scalar(point) + time_step * scalar_source(point);
    };
    auto source =
        [&](Mesh::local_ordinal_type cell_lid) -> Pack::scalar_type
    {
        return scalar_source(mesh->cell_centroid(cell_lid));
    };

    auto system = TransportFactory<decltype(old_values)>::assemble(
        typename TransportFactory<decltype(old_values)>::ScalarTerms{
            .history = {.old_values = old_values, .time_step = time_step},
            .face_fluxes = zero_fluxes,
            .coefficients = {.diffusivity = diffusivity},
            .source = source},
        {.condition = [](int, size_t)
            { return SimpleFluid::BoundaryCondition{SimpleFluid::BoundaryConditionType::Dirichlet, {}}; },
            .value = boundary_value},
        {}, {});

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

/**
 * @brief Verifies 1-D diffusion with constant source converges to the exact quadratic steady-state profile.
 */
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

    for (Mesh::local_ordinal_type lid = 0;
         lid < static_cast<Mesh::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        old_values.set_value(lid, exact(mesh->cell_centroid(lid)));
    }
    old_values.sync_ghosts();

    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "zero_fluxes");
    auto boundary_value =
        [&](int batch_id, size_t in_batch_id) -> Pack::scalar_type
    {
        const auto face_lid =
            mesh->boundary_face_batch(batch_id).face_lids[in_batch_id];
        const auto& normal = mesh->face_normal(face_lid);
        if (std::abs(normal.x) > 0.5)
        {
            return 0.0;
        }
        return exact(mesh->face_centroid(face_lid));
    };
    auto source =
        [=](Mesh::local_ordinal_type) -> Pack::scalar_type
    {
        return source_value;
    };

    auto system = TransportFactory<decltype(old_values)>::assemble(
        typename TransportFactory<decltype(old_values)>::ScalarTerms{
            .history = {.old_values = old_values, .time_step = time_step},
            .face_fluxes = zero_fluxes,
            .coefficients = {.diffusivity = diffusivity},
            .source = source},
        {.condition = [](int, size_t)
            { return SimpleFluid::BoundaryCondition{SimpleFluid::BoundaryConditionType::Dirichlet, {}}; },
            .value = boundary_value},
        {}, {});

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

/**
 * @brief Solves an orthogonal Poisson equation with constant source and compares to the manufactured quadratic solution.
 */
TEST(FvmAnalyticalSolutionsTest, OrthogonalPoissonMatchesManufacturedQuadratic)
{
    constexpr int n_cells = 16;
    constexpr double mesh_size = 1.0 / static_cast<double>(n_cells);
    constexpr double diffusivity = 0.25;
    constexpr double source_value = 0.5;

    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(n_cells, 1, 1, mesh_size));

    auto boundary_condition =
        [&](int batch_id, size_t) -> SimpleFluid::BoundaryCondition
    {
        const auto& name = mesh->boundary_batch_name(batch_id);
        if (name == "xmin" || name == "xmax")
        {
            return {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
        }
        return {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    };
    auto source =
        [=](Mesh::local_ordinal_type) -> Pack::scalar_type
    {
        return source_value;
    };

    const auto system = SimpleFluid::FVM::diffusion_system<Pack>(
        *mesh, diffusivity, boundary_condition, source);

    FieldType numerical(mesh, "poisson_solution");
    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-13;
    ASSERT_TRUE(SimpleFluid::solve_linear_system<Pack>(
        Teuchos::rcp_implicit_cast<const Pack::matrix_type>(system.matrix),
        *system.rhs,
        numerical.owned_data(),
        options));
    mesh->sync_periodic_boundaries(numerical);

    auto exact = [=](SimpleFluid::vec3<> point)
    {
        return constant_source_diffusion_exact(point.x, mesh_size,
                                               source_value, diffusivity);
    };

    EXPECT_LT(SimpleFluid::l2_error(numerical, exact), 1.0e-11);
    EXPECT_LT(SimpleFluid::linf_error(numerical, exact), 1.0e-11);
}

/**
 * @brief Solves a vector-valued Poisson equation and compares each component to manufactured quadratic.
 */
TEST(FvmAnalyticalSolutionsTest, VectorOrthogonalPoissonMatchesManufacturedQuadratic)
{
    constexpr int n_cells = 16;
    constexpr double mesh_size = 1.0 / static_cast<double>(n_cells);
    constexpr double diffusivity = 0.25;
    const SimpleFluid::vec3<> source_value{0.5, 1.0, -0.25};

    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(n_cells, 1, 1, mesh_size));

    auto boundary_condition =
        [&](int batch_id, size_t) -> SimpleFluid::VectorBoundaryCondition
    {
        const auto& name = mesh->boundary_batch_name(batch_id);
        if (name == "xmin" || name == "xmax")
        {
            return {SimpleFluid::BoundaryConditionType::Dirichlet, {}};
        }
        return {SimpleFluid::BoundaryConditionType::Neumann, {}};
    };
    auto source =
        [&](Mesh::local_ordinal_type) -> SimpleFluid::vec3<Pack::scalar_type>
    {
        return source_value;
    };

    const auto system =
        SimpleFluid::FVM::vector_diffusion_system<Pack>(
            *mesh, diffusivity, boundary_condition, source);

    VectorFieldType numerical(mesh, "vector_poisson_solution");
    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-13;
    ASSERT_TRUE(SimpleFluid::solve_linear_system<Pack>(
        Teuchos::rcp_implicit_cast<const Pack::matrix_type>(system.matrix),
        *system.rhs,
        numerical.owned_data(),
        options));
    mesh->sync_periodic_boundaries(numerical);

    auto exact = [&](SimpleFluid::vec3<> point)
    {
        return SimpleFluid::vec3<>{
            constant_source_diffusion_exact(point.x, mesh_size,
                                           source_value.x, diffusivity),
            constant_source_diffusion_exact(point.x, mesh_size,
                                           source_value.y, diffusivity),
            constant_source_diffusion_exact(point.x, mesh_size,
                                           source_value.z, diffusivity)};
    };

    EXPECT_LT(SimpleFluid::l2_error(numerical, exact), 1.0e-11);
    EXPECT_LT(SimpleFluid::linf_error(numerical, exact), 1.0e-11);
}

/**
 * @brief Checks that explicit non-orthogonal correction for diffusion converges with mesh refinement on a sheared mesh.
 */
TEST(FvmAnalyticalSolutionsTest, ExplicitNonOrthogonalCorrectorsConvergeOnShearedQuadraticMesh)
{
    constexpr double diffusivity = 1.0;
    constexpr double source_value = -2.0 * diffusivity;
    auto source =
        [](Mesh::local_ordinal_type) -> Pack::scalar_type
    {
        return source_value;
    };
    auto exact = [](SimpleFluid::vec3<> point)
    {
        return skewed_quadratic_scalar(point);
    };
    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-13;

    auto solve_error = [&](size_t n_cells)
    {
        auto mesh = make_sheared_box_mesh(n_cells, 0.45);
        auto boundary_condition =
            [&](int batch_id, size_t in_batch_id)
                -> SimpleFluid::BoundaryCondition
        {
            const auto face_lid =
                mesh->boundary_face_batch(batch_id).face_lids[in_batch_id];
            return {SimpleFluid::BoundaryConditionType::Dirichlet,
                    skewed_quadratic_scalar(mesh->face_centroid(face_lid))};
        };

        FieldType candidate(mesh, "corrected_sheared_solution");
        const auto converged =
            SimpleFluid::FVM::solve_explicit_non_orthogonal_diffusion<Pack>(
                *mesh, diffusivity, boundary_condition, source, candidate,
                4, options);
        EXPECT_TRUE(converged);
        if (!converged)
        {
            return std::pair{std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::infinity()};
        }
        return std::pair{
            SimpleFluid::l2_error(candidate, exact),
            SimpleFluid::linf_error(candidate, exact)};
    };

    const auto coarse = solve_error(3);
    const auto medium = solve_error(4);
    const auto fine = solve_error(5);

    EXPECT_LT(medium.first, coarse.first);
    EXPECT_LT(fine.first, medium.first);
    EXPECT_LT(fine.second, medium.second);
}

/**
 * @brief Compares explicit, implicit, and hybrid non-orthogonal diffusion treatments on a sheared manufactured mesh.
 */
TEST(FvmAnalyticalSolutionsTest, NonOrthogonalTreatmentsConvergeOnShearedQuadraticMesh)
{
    constexpr double diffusivity = 1.0;
    constexpr double source_value = -2.0 * diffusivity;
    auto source =
        [](Mesh::local_ordinal_type) -> Pack::scalar_type
    {
        return source_value;
    };
    auto exact = [](SimpleFluid::vec3<> point)
    {
        return skewed_quadratic_scalar(point);
    };

    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-12;
    options.max_iterations = 500;

    /** @brief Non-orthogonal treatment and its explicit corrector count. */
    struct TreatmentCase
    {
        SimpleFluid::FVM::NonOrthogonalTreatment treatment;
        int correctors;
    };
    const std::array<TreatmentCase, 3> cases{{
        {SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, 4},
        {SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, 0},
        {SimpleFluid::FVM::NonOrthogonalTreatment::Hybrid, 4}
    }};

    auto solve_error =
        [&](SimpleFluid::FVM::NonOrthogonalTreatment treatment,
            int correctors,
            size_t n_cells)
    {
        auto mesh = make_sheared_box_mesh(n_cells, 0.45);
        auto boundary_condition =
            [&](int batch_id, size_t in_batch_id)
                -> SimpleFluid::BoundaryCondition
        {
            const auto face_lid =
                mesh->boundary_face_batch(batch_id).face_lids[in_batch_id];
            return {SimpleFluid::BoundaryConditionType::Dirichlet,
                    skewed_quadratic_scalar(mesh->face_centroid(face_lid))};
        };

        FieldType candidate(mesh, "non_orthogonal_solution");
        const auto converged =
            SimpleFluid::FVM::solve_non_orthogonal_diffusion<Pack>(
                *mesh, diffusivity, boundary_condition, source, candidate,
                treatment, correctors, options);
        EXPECT_TRUE(converged)
            << SimpleFluid::FVM::to_string(treatment);
        if (!converged)
        {
            return std::pair{std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::infinity()};
        }
        return std::pair{
            SimpleFluid::l2_error(candidate, exact),
            SimpleFluid::linf_error(candidate, exact)};
    };

    for (const auto& test_case : cases)
    {
        const auto coarse = solve_error(
            test_case.treatment, test_case.correctors, 3);
        const auto fine = solve_error(
            test_case.treatment, test_case.correctors, 5);

        EXPECT_LT(fine.first, coarse.first)
            << SimpleFluid::FVM::to_string(test_case.treatment);
        EXPECT_LT(fine.second, coarse.second)
            << SimpleFluid::FVM::to_string(test_case.treatment);
    }
}

/**
 * @brief Verifies vector-valued transport with a source term reproduces the exact transient solution.
 */
TEST(FvmAnalyticalSolutionsTest, VectorTransportSourceMatchesExactTransient)
{
    auto mesh = make_unit_box_mesh();
    VectorFieldType old_values(mesh, "old_values");

    for (Mesh::local_ordinal_type lid = 0;
         lid < static_cast<Mesh::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        old_values.set_value(lid, affine_vector(mesh->cell_centroid(lid)));
    }
    old_values.sync_ghosts();

    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "zero_fluxes");
    constexpr double time_step = 0.125;
    constexpr double diffusivity = 0.7;
    auto boundary_value =
        [&](int batch_id, size_t in_batch_id)
            -> SimpleFluid::vec3<Pack::scalar_type>
    {
        const auto face_lid =
            mesh->boundary_face_batch(batch_id).face_lids[in_batch_id];
        const auto& point = mesh->face_centroid(face_lid);
        return affine_vector(point) + vector_source(point) * time_step;
    };
    auto source =
        [&](Mesh::local_ordinal_type cell_lid)
            -> SimpleFluid::vec3<Pack::scalar_type>
    {
        return vector_source(mesh->cell_centroid(cell_lid));
    };

    auto system = TransportFactory<decltype(old_values)>::assemble(
        typename TransportFactory<decltype(old_values)>::VectorTerms{
            .history = {.old_values = old_values, .time_step = time_step},
            .face_fluxes = zero_fluxes,
            .coefficients = {.diffusivity = diffusivity},
            .source = source},
        typename TransportFactory<decltype(old_values)>::BasicVectorBoundary{
            .condition = [](int, size_t)
            { return SimpleFluid::VectorBoundaryCondition{SimpleFluid::BoundaryConditionType::Dirichlet, {}}; },
            .value = boundary_value},
        {}, {});

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

/**
 * @brief Advances inviscid Burgers equation explicitly and checks the solution tracks the pre-shock sine wave.
 */
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

/**
 * @brief Advances inviscid Burgers equation semi-implicitly and checks tracking of the pre-shock sine wave.
 */
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

/**
 * @brief Advances viscous Burgers equation semi-implicitly and checks against the exact analytical solution.
 */
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
