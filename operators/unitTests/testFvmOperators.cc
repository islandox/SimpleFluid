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
#include "geometry/MeshFactory.hh"
#include "operators/FvmOperators.hh"
#include "utils/testing_environment.hh"

#include <cmath>
#include <memory>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<const SimpleFluid::Database> make_2x2x2_box_database()
{
    auto db = std::make_shared<SimpleFluid::Database>();

    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{1.0});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::BOX));
    db->set("X", SimpleFluid::ArrReal{0.0, 1.0, 2.0});
    db->set("Y", SimpleFluid::ArrReal{0.0, 1.0, 2.0});
    db->set("Z", SimpleFluid::ArrReal{0.0, 1.0, 2.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});

    return db;
}

SimpleFluid::SP<MeshType> make_mesh()
{
    auto db = make_2x2x2_box_database();
    SimpleFluid::MeshFactory factory(db);
    return factory.template build<Pack>();
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

    const auto gradients = SimpleFluid::FvmOperators::cell_gradient(phi);
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
    FieldType velocity_x(mesh, 1.0, "velocity_x");
    FieldType velocity_y(mesh, 2.0, "velocity_y");
    FieldType velocity_z(mesh, 3.0, "velocity_z");

    const auto fluxes = SimpleFluid::FvmOperators::face_fluxes(
        *mesh, velocity_x, velocity_y, velocity_z);

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
        const auto magnitude = std::abs(fluxes[static_cast<std::size_t>(fid)]);
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

TEST(FvmOperatorsTest, NoSlipBoundaryProducesZeroExteriorFlux)
{
    auto mesh = make_mesh();
    FieldType velocity_x(mesh, 1.0, "velocity_x");
    FieldType velocity_y(mesh, 2.0, "velocity_y");
    FieldType velocity_z(mesh, 3.0, "velocity_z");

    SimpleFluid::BoundaryConditionSet bcs;
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }

    const auto fluxes = SimpleFluid::FvmOperators::face_fluxes(
        *mesh, velocity_x, velocity_y, velocity_z, &bcs);

    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        if (mesh->is_boundary_face(fid))
        {
            EXPECT_DOUBLE_EQ(fluxes[static_cast<std::size_t>(fid)], 0.0);
        }
    }
}

TEST(FvmOperatorsTest, BuildsUpwindAndPressurePoissonMatrices)
{
    auto mesh = make_mesh();
    FieldType velocity_x(mesh, 1.0, "velocity_x");
    FieldType velocity_y(mesh, 0.0, "velocity_y");
    FieldType velocity_z(mesh, 0.0, "velocity_z");

    const auto fluxes = SimpleFluid::FvmOperators::face_fluxes(
        *mesh, velocity_x, velocity_y, velocity_z);
    auto convection =
        SimpleFluid::FvmOperators::upwind_convection_matrix<Pack>(*mesh, fluxes);
    auto pressure = SimpleFluid::FvmOperators::pressure_poisson_matrix<Pack>(
        *mesh, mesh->owned_cell_global_ids().front());

    EXPECT_EQ(convection->getGlobalNumRows(),
              mesh->owned_cell_map()->getGlobalNumElements());
    EXPECT_EQ(pressure->getGlobalNumRows(),
              mesh->owned_cell_map()->getGlobalNumElements());
}
