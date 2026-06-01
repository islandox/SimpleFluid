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
#include "utils/testing_environment.hh"

#include <cmath>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<MeshType> make_unit_box_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(2, 2, 2, 0.5));
}

double affine_scalar(const SimpleFluid::vec3<>& point)
{
    return 1.0 + 2.0 * point.x - 3.0 * point.y + 4.0 * point.z;
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

    const auto divergence =
        SimpleFluid::FvmOperators::cell_divergence_from_fluxes<Pack>(
            *mesh, face_velocity);

    ASSERT_EQ(divergence.size(), mesh->num_owned_cells());
    for (const auto value : divergence)
    {
        EXPECT_NEAR(value, 6.0, 1.0e-12);
    }
}

TEST(FvmAnalyticalSolutionsTest, SemiImplicitDiffusionPreservesAffineScalar)
{
    auto mesh = make_unit_box_mesh();
    SimpleFluid::CellField<Pack> old_values(mesh, "old_values");

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
