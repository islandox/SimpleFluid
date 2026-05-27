/**
 * @file testFvmOperators.cc
 * @brief unit tests for finite-volume helper operators
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
