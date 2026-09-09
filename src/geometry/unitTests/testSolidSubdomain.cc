/**
 * @file testSolidSubdomain.cc
 * @brief Tests for compact solid-cell mesh views.
 */

#include <gtest/gtest.h>

#include "geometry/SolidSubdomain.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using Cartesian = SimpleFluid::Meshes::OrthogonalCartesian3D;
using Handle = SimpleFluid::MeshHandle<Pack>;
using Subdomain = SimpleFluid::SolidSubdomain<Pack>;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<Handle> make_three_cell_handle()
{
    auto mesh = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0, 2.0, 3.0}, {0.0, 1.0}, {0.0, 1.0}}});
    return std::make_shared<Handle>(std::move(mesh));
}

const Subdomain::InterfaceFace& only_interface(const Subdomain& subdomain)
{
    EXPECT_EQ(subdomain.interface_faces().size(), 1U);
    return subdomain.interface_faces().front();
}

} // namespace

/** Selected cells receive compact ordinals and subset maps. */
TEST(SolidSubdomainTest, BuildsCompactOwnedFirstCellAndFaceMaps)
{
    auto parent = make_three_cell_handle();
    const auto first_selected_geometry_gid = parent->cell_geometry_global_id(1);
    const auto second_selected_geometry_gid = parent->cell_geometry_global_id(2);
    const auto second_selected_field_gid = parent->cell_global_id(2);
    const auto parent_comm = parent->owned_cell_map()->getComm();
    const Subdomain solid(std::move(parent), [](Pack::global_ordinal_type geometry_gid, const Subdomain::Vec3& centroid)
        { return geometry_gid >= 1 && centroid.x >= 1.0; });

    ASSERT_EQ(solid.num_owned_cells(), 2U);
    ASSERT_EQ(solid.num_local_cells(), 2U);
    EXPECT_FALSE(parent);
    EXPECT_EQ(solid.parent_cell_lid(0), 0);
    EXPECT_EQ(solid.parent_cell_lid(1), 1);
    EXPECT_EQ(solid.subdomain_cell_lid(0), 0);
    EXPECT_EQ(solid.subdomain_cell_lid(1), 1);
    EXPECT_EQ(solid.subdomain_cell_lid(2), Subdomain::invalid_local_id());
    EXPECT_EQ(solid.cell_geometry_global_id(0), first_selected_geometry_gid);
    EXPECT_EQ(solid.cell_geometry_global_id(1), second_selected_geometry_gid);
    EXPECT_EQ(solid.cell_global_id(1), second_selected_field_gid);

    EXPECT_EQ(solid.owned_cell_map()->getLocalNumElements(), 2U);
    EXPECT_EQ(solid.overlap_cell_map()->getLocalNumElements(), 2U);
    EXPECT_EQ(solid.owned_cell_map()->getComm(), parent_comm);
    for (size_t local = 0; local < solid.num_local_cells(); ++local)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(local);
        EXPECT_EQ(solid.overlap_cell_map()->getGlobalElement(lid), solid.cell_global_id(lid));
    }

    ASSERT_EQ(solid.num_owned_faces(), solid.num_faces());
    EXPECT_EQ(solid.owned_face_map()->getLocalNumElements(), solid.num_owned_faces());
    EXPECT_EQ(solid.overlap_face_map()->getLocalNumElements(), solid.num_faces());
    for (size_t local = 0; local < solid.num_faces(); ++local)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(local);
        EXPECT_TRUE(solid.is_owned_face(lid));
        EXPECT_EQ(solid.overlap_face_map()->getGlobalElement(lid), solid.face_global_id(lid));
        EXPECT_EQ(solid.subdomain_face_lid(solid.parent_face_lid(lid)), lid);
    }
}

/** A cut face is a named boundary owned and oriented from the solid side. */
TEST(SolidSubdomainTest, ReorientsCutFaceFromSelectedParentNeighbor)
{
    auto parent = make_three_cell_handle();
    const auto outside_geometry_gid = parent->cell_geometry_global_id(0);
    const Subdomain solid(
        std::move(parent), [](Pack::global_ordinal_type, const Subdomain::Vec3& centroid) { return centroid.x > 1.0; },
        "fluid_solid_interface");

    const auto& interface = only_interface(solid);
    EXPECT_TRUE(solid.is_interface_face(interface.face_lid));
    EXPECT_TRUE(solid.is_boundary_face(interface.face_lid));
    EXPECT_TRUE(solid.is_exterior_face(interface.face_lid));
    EXPECT_TRUE(solid.is_owned_face(interface.face_lid));
    EXPECT_EQ(solid.owner_cell(interface.face_lid), 0);
    EXPECT_EQ(solid.neighbor_cell(interface.face_lid), Subdomain::invalid_local_id());
    EXPECT_EQ(interface.solid_cell_lid, 0);
    EXPECT_EQ(interface.outside_parent_cell_lid, 2);
    EXPECT_EQ(interface.outside_cell_geometry_global_id, outside_geometry_gid);
    EXPECT_EQ(solid.parent_face_lid(interface.face_lid), interface.parent_face_lid);

    EXPECT_EQ(solid.boundary_id(interface.face_lid), solid.interface_boundary_id());
    EXPECT_EQ(solid.boundary_batch_name(solid.interface_boundary_id()), "fluid_solid_interface");
    const auto& batch = solid.boundary_face_batch(solid.interface_boundary_id());
    ASSERT_EQ(batch.face_lids.size(), 1U);
    EXPECT_EQ(batch.face_lids.front(), interface.face_lid);

    const auto parent_normal = solid.parent_mesh().face_normal(interface.parent_face_lid);
    const auto solid_normal = solid.face_normal(interface.face_lid);
    EXPECT_DOUBLE_EQ(parent_normal.x, 1.0);
    EXPECT_DOUBLE_EQ(solid_normal.x, -1.0);
    EXPECT_DOUBLE_EQ(solid_normal.y, 0.0);
    EXPECT_DOUBLE_EQ(solid_normal.z, 0.0);
    EXPECT_EQ(solid.face_normal_outward(interface.face_lid, interface.solid_cell_lid), solid_normal);
    EXPECT_DOUBLE_EQ(solid.face_centroid(interface.face_lid).x, 1.0);
    EXPECT_DOUBLE_EQ(solid.cell_to_face_distance(interface.face_lid, interface.solid_cell_lid), 0.5);
    EXPECT_DOUBLE_EQ(solid.face_cell_center_distance(interface.face_lid), 0.0);

    const auto cell_faces = solid.faces(interface.solid_cell_lid);
    EXPECT_NE(std::find(cell_faces.begin(), cell_faces.end(), interface.face_lid), cell_faces.end());
    EXPECT_EQ(solid.boundary_face_map()->getLocalNumElements(), 10U);

    bool retained_xmax = false;
    for (const auto& [batch_id, retained_batch] : solid.boundary_batches())
    {
        static_cast<void>(retained_batch);
        retained_xmax = retained_xmax || solid.boundary_batch_name(batch_id) == "xmax";
    }
    EXPECT_TRUE(retained_xmax);
}

/** A legacy STK mesh works through the established MeshHandle adapter. */
TEST(SolidSubdomainTest, SelectsCellsOnLegacyBackedMeshHandle)
{
    const auto legacy = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database());
    auto parent = std::make_shared<Handle>(legacy);
    ASSERT_TRUE(parent->is_stk());
    const auto selected_field_gid = parent->cell_global_id(1);

    const Subdomain solid(
        std::move(parent), [](Pack::global_ordinal_type, const Subdomain::Vec3& centroid) { return centroid.x > 1.0; });

    ASSERT_EQ(solid.num_owned_cells(), 1U);
    EXPECT_FALSE(parent);
    EXPECT_EQ(solid.parent_cell_lid(0), 0);
    EXPECT_EQ(solid.cell_geometry_global_id(0), legacy->cell_global_id(1));
    EXPECT_EQ(solid.cell_global_id(0), selected_field_gid);

    const auto& interface = only_interface(solid);
    EXPECT_EQ(solid.owner_cell(interface.face_lid), 0);
    EXPECT_DOUBLE_EQ(solid.face_normal(interface.face_lid).x, -1.0);
}

/** Untagged physical exteriors are not mistaken for missing partition ghosts. */
TEST(SolidSubdomainTest, AcceptsUntaggedUnstructuredExteriorFaces)
{
    using Unstructured = SimpleFluid::Meshes::UnstructuredMesh;
    const auto geometry = std::make_shared<Unstructured>(
        SimpleFluid::Arr<Unstructured::Vec3>{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {0.0, 1.0, 1.0}},
        SimpleFluid::Arr<Unstructured::CellDefinition>{{Unstructured::CellType::HEXAHEDRON, {0, 1, 2, 3, 4, 5, 6, 7}}},
        SimpleFluid::Arr<Unstructured::BoundaryFaceDefinition>{});
    const auto parent = std::make_shared<Handle>(geometry);
    const Subdomain solid(parent);

    ASSERT_EQ(solid.num_owned_cells(), 1U);
    EXPECT_TRUE(solid.boundary_batches().empty());
    EXPECT_EQ(solid.boundary_face_map()->getLocalNumElements(), 0U);
    for (size_t face = 0; face < solid.num_faces(); ++face)
    {
        const auto face_lid = static_cast<Pack::local_ordinal_type>(face);
        EXPECT_TRUE(parent->is_geometry_exterior_face(solid.parent_face_lid(face_lid)));
        EXPECT_TRUE(solid.is_exterior_face(face_lid));
        EXPECT_FALSE(solid.is_boundary_face(face_lid));
    }
}

/** Invalid selectors and ambiguous interface names fail before topology use. */
TEST(SolidSubdomainTest, CollectivelyValidatesSelectionInputs)
{
    auto null_selector_parent = make_three_cell_handle();
    EXPECT_THROW(
        static_cast<void>(Subdomain(std::move(null_selector_parent), {}, "solid_interface")), std::invalid_argument);

    auto throwing_selector_parent = make_three_cell_handle();
    EXPECT_THROW(static_cast<void>(Subdomain(std::move(throwing_selector_parent),
                     [](Pack::global_ordinal_type, const Subdomain::Vec3&)
                     {
                         throw std::runtime_error("selector failure");
                         return true;
                     })),
        std::runtime_error);

    auto empty_selector_parent = make_three_cell_handle();
    EXPECT_THROW(static_cast<void>(Subdomain(std::move(empty_selector_parent),
                     [](Pack::global_ordinal_type, const Subdomain::Vec3&) { return false; })),
        std::invalid_argument);

    auto colliding_name_parent = make_three_cell_handle();
    EXPECT_THROW(static_cast<void>(Subdomain(
                     std::move(colliding_name_parent),
                     [](Pack::global_ordinal_type, const Subdomain::Vec3&) { return true; }, "xmin")),
        std::invalid_argument);
}
