/**
 * @file testFvmOperators.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief unit tests for finite-volume helper operators
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "fields/CellField.hh"
#include "fields/VectorCellField.hh"
#include "FVM/FvmOperators.hh"
#include "equations/TemperatureDiffusionEquation.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <cmath>
#include <memory>
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

SimpleFluid::SP<MeshType> make_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_2x2x2_database());
}

std::vector<Pack::scalar_type> local_values(const FieldType& field)
{
    std::vector<Pack::scalar_type> values(field.num_local_cells());
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(field.num_local_cells());
         ++lid)
    {
        values[static_cast<std::size_t>(lid)] = field.local_value(lid);
    }
    return values;
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

Pack::scalar_type local_matrix_entry(
    const Pack::matrix_type& matrix,
    MeshType::local_ordinal_type row,
    MeshType::local_ordinal_type column)
{
    const auto row_entries = matrix.getNumEntriesInLocalRow(row);
    typename Pack::matrix_type::nonconst_local_inds_host_view_type columns(
        "columns", row_entries);
    typename Pack::matrix_type::nonconst_values_host_view_type values(
        "values", row_entries);
    std::size_t num_entries = 0;
    matrix.getLocalRowCopy(row, columns, values, num_entries);

    Pack::scalar_type entry = 0.0;
    for (std::size_t i = 0; i < num_entries; ++i)
    {
        if (columns(i) == column)
        {
            entry += values(i);
        }
    }

    return entry;
}

} // namespace

/**
 * @brief Sets a linear scalar field and verifies the cell_gradient operator recovers the exact gradient.
 */
TEST(FvmOperatorsTest, RecoversLinearCellGradientOnStructuredBox)
{
    auto mesh = make_mesh();
    FieldType phi(mesh, "phi");

    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        const auto& c = mesh->cell_centroid(lid);
        phi.set_value(lid, 2.0 * c.x + 3.0 * c.y + 4.0 * c.z);
    }
    phi.sync_ghosts();

    std::vector<MeshType::Vec3> gradients;
    SimpleFluid::FvmOperators::cell_gradient(phi, gradients);
    ASSERT_EQ(gradients.size(), mesh->num_owned_cells());
    for (const auto& gradient : gradients)
    {
        EXPECT_NEAR(gradient.x, 2.0, 1.0e-12);
        EXPECT_NEAR(gradient.y, 3.0, 1.0e-12);
        EXPECT_NEAR(gradient.z, 4.0, 1.0e-12);
    }
}

TEST(FvmOperatorsTest, BuildsIdentityAndDiffusionMatrices)
{
    auto mesh = make_mesh();

    auto identity = SimpleFluid::FvmOperators::identity_matrix<Pack>(
        mesh->owned_cell_map());
    auto diffusion = SimpleFluid::FvmOperators::diffusion_matrix<Pack>(*mesh, 1.0);

    EXPECT_EQ(identity->getGlobalNumRows(),
              mesh->owned_cell_map()->getGlobalNumElements());
    EXPECT_EQ(diffusion->getGlobalNumRows(),
              mesh->owned_cell_map()->getGlobalNumElements());
}

TEST(FvmOperatorsTest, FaceFluxesUseAllThreeVelocityComponents)
{
    auto mesh = make_mesh();
    VectorFieldType velocity(mesh, SimpleFluid::vec3{1.0, 2.0, 3.0}, "velocity");

    SimpleFluid::FaceField<Pack> fluxes(mesh, "face_flux");
    SimpleFluid::FvmOperators::face_fluxes(velocity, fluxes);

    bool saw_x_face = false;
    bool saw_y_face = false;
    bool saw_z_face = false;
    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        if (!mesh->is_interior_face(fid))
        {
            continue;
        }

        const auto& normal = mesh->face_normal(fid);
        const auto magnitude = fluxes.is_owned_face(fid) ? std::abs(fluxes.value(fid)) : 0.0;
        if (std::abs(normal.x) > 0.5)
        {
            saw_x_face = true;
            EXPECT_NEAR(magnitude, 1.0, 1.0e-12);
        }
        if (std::abs(normal.y) > 0.5)
        {
            saw_y_face = true;
            EXPECT_NEAR(magnitude, 2.0, 1.0e-12);
        }
        if (std::abs(normal.z) > 0.5)
        {
            saw_z_face = true;
            EXPECT_NEAR(magnitude, 3.0, 1.0e-12);
        }
    }

    EXPECT_TRUE(saw_x_face);
    EXPECT_TRUE(saw_y_face);
    EXPECT_TRUE(saw_z_face);
}

TEST(FvmOperatorsTest, FaceVelocitiesInterpolateInteriorFaces)
{
    auto mesh = make_mesh();
    VectorFieldType velocity(mesh, "velocity");

    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        const auto& center = mesh->cell_centroid(lid);
        velocity.set_value(lid,
                           {center.x + 1.0,
                            2.0 * center.y + 3.0,
                            3.0 * center.z + 5.0});
    }
    velocity.sync_ghosts();

    SimpleFluid::VectorFaceField<Pack> face_velocity(mesh, "face_velocity");
    SimpleFluid::FvmOperators::face_velocities(velocity, face_velocity);

    bool saw_interior_face = false;
    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        if (!mesh->is_interior_face(fid))
        {
            continue;
        }

        saw_interior_face = true;
        const auto owner = mesh->owner_cell(fid);
        const auto neighbor = mesh->neighbor_cell(fid);
        const auto expected =
            (velocity.local_value(owner) + velocity.local_value(neighbor)) / 2.0;
        const auto actual = face_velocity.value(fid);

        EXPECT_NEAR(actual.x, expected.x, 1.0e-12);
        EXPECT_NEAR(actual.y, expected.y, 1.0e-12);
        EXPECT_NEAR(actual.z, expected.z, 1.0e-12);
    }

    EXPECT_TRUE(saw_interior_face);
}

TEST(FvmOperatorsTest, NoSlipBoundaryProducesZeroFaceVelocity)
{
    auto mesh = make_mesh();
    VectorFieldType velocity(mesh, SimpleFluid::vec3{1.0, 2.0, 3.0}, "velocity");

    SimpleFluid::BoundaryConditionSet bcs;
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }

    const auto cache =
        SimpleFluid::FvmOperators::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    SimpleFluid::VectorFaceField<Pack> face_velocity(mesh, "face_velocity");
    SimpleFluid::FvmOperators::face_velocities(velocity, cache, face_velocity);

    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        if (mesh->is_boundary_face(fid))
        {
            EXPECT_EQ(face_velocity.value(fid), (SimpleFluid::vec3{}));
        }
    }
}

TEST(FvmOperatorsTest, NoSlipBoundaryProducesZeroExteriorFlux)
{
    auto mesh = make_mesh();
    VectorFieldType velocity(mesh, SimpleFluid::vec3{1.0, 2.0, 3.0}, "velocity");

    SimpleFluid::BoundaryConditionSet bcs;
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }

    const auto cache =
        SimpleFluid::FvmOperators::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    SimpleFluid::FaceField<Pack> fluxes(mesh, "face_flux");
    SimpleFluid::FvmOperators::face_fluxes(velocity, cache, fluxes);

    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        if (mesh->is_boundary_face(fid))
        {
            EXPECT_DOUBLE_EQ(fluxes.is_owned_face(fid) ? fluxes.value(fid) : 0.0, 0.0);
        }
    }
}

TEST(FvmOperatorsTest, BuildsUpwindAndPressurePoissonMatrices)
{
    auto mesh = make_mesh();
    VectorFieldType velocity(mesh, SimpleFluid::vec3{1.0, 0.0, 0.0}, "velocity");

    SimpleFluid::FaceField<Pack> fluxes(mesh, "face_flux");
    SimpleFluid::FvmOperators::face_fluxes(velocity, fluxes);
    auto convection =
        SimpleFluid::FvmOperators::upwind_convection_matrix<Pack>(*mesh, fluxes);
    auto pressure = SimpleFluid::FvmOperators::pressure_poisson_matrix<Pack>(
        *mesh, mesh->owned_cell_global_ids().front());

    EXPECT_EQ(convection->getGlobalNumRows(),
              mesh->owned_cell_map()->getGlobalNumElements());
    EXPECT_EQ(pressure->getGlobalNumRows(),
              mesh->owned_cell_map()->getGlobalNumElements());
}

// =========================================================================
//  Periodic Boundary Condition Tests
// =========================================================================

/**
 * @brief Create a 2×1×1 box mesh and pair the xmin face of cell 0 with
 *        the xmax face of cell 1 as a periodic boundary.
 *
 * @return Shared pointer to the configured mesh with periodic pairs set.
 */
SimpleFluid::SP<MeshType> make_periodic_box_mesh()
{
    constexpr int n = 2;
    auto db = SimpleFluid::test::make_box_database(n, 1, 1, 1.0 / n);
    auto mesh = SimpleFluid::test::build_mesh<Pack>(db);

    // Find xmin and xmax boundary faces.
    SimpleFluid::Mesh<Pack>::local_ordinal_type xmin_face = -1;
    SimpleFluid::Mesh<Pack>::local_ordinal_type xmax_face = -1;
    for (const auto& [patch_id, patch] : mesh->boundary_patches())
    {
        const auto& name = mesh->boundary_patch_name(patch_id);
        if (name == "xmin" && !patch.face_lids.empty()) xmin_face = patch.face_lids[0];
        if (name == "xmax" && !patch.face_lids.empty()) xmax_face = patch.face_lids[0];
    }

    // The cell adjacent to the xmin face is cell 0; its paired cell is the
    // cell adjacent to the xmax face (cell 1), and vice versa.
    const auto owner0 = mesh->owner_cell(xmin_face);
    const auto owner1 = mesh->owner_cell(xmax_face);
    mesh->set_periodic_face(xmin_face, owner1);
    mesh->set_periodic_face(xmax_face, owner0);

    return mesh;
}

/**
 * @brief Verifies that periodic faces are correctly identified and that
 *        face velocities average the owner and paired-cell values.
 */
TEST(FvmOperatorsTest, PeriodicBoundaryFaceVelocityIsAveraged)
{
    auto mesh = make_periodic_box_mesh();

    // Set velocity: cell 0 → (1,0,0), cell 1 → (3,0,0)
    VectorFieldType velocity(mesh, "velocity");
    velocity.set_value(0, {1.0, 0.0, 0.0});
    velocity.set_value(1, {3.0, 0.0, 0.0});
    velocity.sync_ghosts();

    // Assemble face velocities without boundary-condition treatment.
    SimpleFluid::VectorFaceField<Pack> face_vel(mesh, "face_vel");
    SimpleFluid::FvmOperators::face_velocities(velocity, face_vel);

    // Find periodic faces.
    SimpleFluid::Mesh<Pack>::local_ordinal_type xmin_face = -1;
    SimpleFluid::Mesh<Pack>::local_ordinal_type xmax_face = -1;
    for (const auto& [patch_id, patch] : mesh->boundary_patches())
    {
        const auto& name = mesh->boundary_patch_name(patch_id);
        if (name == "xmin" && !patch.face_lids.empty()) xmin_face = patch.face_lids[0];
        if (name == "xmax" && !patch.face_lids.empty()) xmax_face = patch.face_lids[0];
    }

    ASSERT_NE(xmin_face, -1);
    ASSERT_NE(xmax_face, -1);
    EXPECT_TRUE(mesh->is_periodic_boundary_face(xmin_face));
    EXPECT_TRUE(mesh->is_periodic_boundary_face(xmax_face));

    // Periodic faces should get the average of the owner and paired cell.
    const auto v_xmin = face_vel.value(xmin_face);
    const auto v_xmax = face_vel.value(xmax_face);
    EXPECT_NEAR(v_xmin.x, 2.0, 1.0e-12);  // (1 + 3) / 2
    EXPECT_NEAR(v_xmax.x, 2.0, 1.0e-12);  // (3 + 1) / 2
}

TEST(FvmOperatorsTest, PeriodicVelocityCacheDoesNotOverwritePairedFaceVelocity)
{
    auto mesh = make_periodic_box_mesh();

    VectorFieldType velocity(mesh, "velocity");
    velocity.set_value(0, {1.0, 0.0, 0.0});
    velocity.set_value(1, {3.0, 0.0, 0.0});
    velocity.sync_ghosts();

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.velocity["xmin"] = {SimpleFluid::BoundaryConditionType::Periodic, {}};
    bcs.velocity["xmax"] = {SimpleFluid::BoundaryConditionType::Periodic, {}};
    for (const auto* name : {"ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }

    const auto cache =
        SimpleFluid::FvmOperators::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    EXPECT_EQ(cache.value.size(), mesh->boundary_patches().size());

    SimpleFluid::VectorFaceField<Pack> face_vel(mesh, "face_vel");
    EXPECT_NO_THROW(SimpleFluid::FvmOperators::face_velocities(
        velocity, cache, face_vel));

    const auto xmin_face = boundary_face_lid(*mesh, "xmin");
    const auto xmax_face = boundary_face_lid(*mesh, "xmax");
    EXPECT_NEAR(face_vel.value(xmin_face).x, 2.0, 1.0e-12);
    EXPECT_NEAR(face_vel.value(xmax_face).x, 2.0, 1.0e-12);
}

/**
 * @brief Verifies that periodic face fluxes are computed correctly,
 *        treating the periodic boundary like an interior face.
 *
 * A 2×1×1 mesh with mesh_size = 0.5 gives face areas of 0.25 and
 * cell volumes of 0.125.  Cell 0 has velocity (2,0,0), cell 1 has (0,0,0).
 * The averaged face velocity at the periodic xmax face is (1,0,0).
 * With face normal (+1,0,0) and area 0.25: flux = 1 × 1 × 0.25 = 0.25.
 */
TEST(FvmOperatorsTest, PeriodicBoundaryFluxIsComputed)
{
    auto mesh = make_periodic_box_mesh();

    VectorFieldType velocity(mesh, "velocity");
    velocity.set_value(0, {2.0, 0.0, 0.0});
    velocity.set_value(1, {0.0, 0.0, 0.0});
    velocity.sync_ghosts();

    SimpleFluid::FaceField<Pack> fluxes(mesh, "face_flux");
    SimpleFluid::FvmOperators::face_fluxes(velocity, fluxes);

    SimpleFluid::Mesh<Pack>::local_ordinal_type xmax_face = -1;
    for (const auto& [patch_id, patch] : mesh->boundary_patches())
    {
        if (mesh->boundary_patch_name(patch_id) == "xmax" && !patch.face_lids.empty())
            xmax_face = patch.face_lids[0];
    }
    ASSERT_NE(xmax_face, -1);
    ASSERT_TRUE(fluxes.is_owned_face(xmax_face));

    // Averaged face velocity: (2 + 0)/2 = 1.0 along +x.
    // Face area = dy*dz = 0.5 * 0.5 = 0.25.
    // Normal on xmax = (+1, 0, 0).
    // Flux = v·n·A = 1.0 * 1.0 * 0.25 = 0.25.
    EXPECT_NEAR(fluxes.value(xmax_face), 0.25, 1.0e-12);
}

/**
 * @brief Verifies that explicit diffusion with periodic boundary
 *        conditions correctly exchanges heat between the paired cells.
 *
 * A 2×1×1 mesh with mesh_size = 0.5 has face area 0.25 and cell volume
 * 0.125.  Cell 0 (T=1) at x=0.25, cell 1 (T=0) at x=0.75.  The interior
 * face connects them (distance 0.5).  The periodic xmin face of cell 0
 * also connects to cell 1 (distance 0.5), doubling the laplacian.
 *
 * laplacian(cell 0) = -1.0,  laplacian/vol = -8.0
 * T0_new = 1.0 + 0.1 * (-8.0) = 0.2
 * T1_new = 0.0 + 0.1 * 8.0  = 0.8
 */
TEST(FvmOperatorsTest, PeriodicBoundaryExplicitDiffusionTransfersHeat)
{
    auto mesh = make_periodic_box_mesh();
    FieldType temperature(mesh, "temperature");
    temperature.set_value(0, 1.0);
    temperature.set_value(1, 0.0);
    temperature.sync_ghosts();

    // Insulated on Y and Z faces; periodic on X faces.
    SimpleFluid::BoundaryConditionSet bcs;
    for (const auto* name : {"ymin", "ymax", "zmin", "zmax"})
        bcs.temperature[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    bcs.temperature["xmin"] = {SimpleFluid::BoundaryConditionType::Periodic, 0.0};
    bcs.temperature["xmax"] = {SimpleFluid::BoundaryConditionType::Periodic, 0.0};

    SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);
    const auto old_t = local_values(temperature);
    equation.advance_explicit(old_t, 0.1, 1.0, temperature);

    // With doubled connectivity (interior + periodic), the heat transfer is
    // amplified: each cell exchanges with the other through two faces.
    // area = 0.25, dist = 0.5, vol = 0.125
    // laplacian magnitude = 2 * (1.0 - 0.0) * 0.25 / 0.5 = 1.0
    // laplacian/vol = 1.0 / 0.125 = 8.0
    // T0_new = 1.0 - 0.1 * 8.0 = 0.2
    // T1_new = 0.0 + 0.1 * 8.0 = 0.8
    EXPECT_NEAR(temperature.value(0), 0.2, 1.0e-12);
    EXPECT_NEAR(temperature.value(1), 0.8, 1.0e-12);
}

TEST(FvmOperatorsTest, PeriodicBoundaryScalarTransportMatrixUsesPairedCell)
{
    auto mesh = make_periodic_box_mesh();
    FieldType temperature(mesh, "temperature");
    temperature.set_value(0, 1.0);
    temperature.set_value(1, 0.0);
    temperature.sync_ghosts();

    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "face_flux");
    auto boundary_value =
        [&](int patch_id, MeshType::local_ordinal_type in_patch_id)
    {
        const auto face_lid =
            mesh->boundary_face_patch(patch_id).face_lids[
                static_cast<std::size_t>(in_patch_id)];
        EXPECT_FALSE(mesh->is_periodic_boundary_face(face_lid));
        return 0.0;
    };

    const auto system = SimpleFluid::FvmOperators::transport_system<Pack>(
        temperature, zero_fluxes, 1.0, 1.0, boundary_value);

    EXPECT_NEAR(local_matrix_entry(*system.matrix, 0, 0), 5.125, 1.0e-12);
    EXPECT_NEAR(local_matrix_entry(*system.matrix, 1, 1), 5.125, 1.0e-12);
    EXPECT_NEAR(local_matrix_entry(*system.matrix, 0, 1), -1.0, 1.0e-12);
    EXPECT_NEAR(local_matrix_entry(*system.matrix, 1, 0), -1.0, 1.0e-12);
}

TEST(FvmOperatorsTest, PeriodicBoundaryVectorTransportMatrixUsesPairedCell)
{
    auto mesh = make_periodic_box_mesh();
    VectorFieldType velocity(mesh, "velocity");
    velocity.set_value(0, {1.0, 2.0, 3.0});
    velocity.set_value(1, {0.0, 0.0, 0.0});
    velocity.sync_ghosts();

    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "face_flux");
    auto boundary_value =
        [&](int patch_id, MeshType::local_ordinal_type in_patch_id)
    {
        const auto face_lid =
            mesh->boundary_face_patch(patch_id).face_lids[
                static_cast<std::size_t>(in_patch_id)];
        EXPECT_FALSE(mesh->is_periodic_boundary_face(face_lid));
        return SimpleFluid::vec3<Pack::scalar_type>{};
    };

    const auto system = SimpleFluid::FvmOperators::transport_system<Pack>(
        velocity, zero_fluxes, 1.0, 1.0, boundary_value);

    EXPECT_NEAR(local_matrix_entry(*system.matrix, 0, 0), 5.125, 1.0e-12);
    EXPECT_NEAR(local_matrix_entry(*system.matrix, 1, 1), 5.125, 1.0e-12);
    EXPECT_NEAR(local_matrix_entry(*system.matrix, 0, 1), -1.0, 1.0e-12);
    EXPECT_NEAR(local_matrix_entry(*system.matrix, 1, 0), -1.0, 1.0e-12);
}
