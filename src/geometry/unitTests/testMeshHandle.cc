/**
 * @file testMeshHandle.cc
 * @brief Tests for the variant-backed mesh runtime.
 */

#include <gtest/gtest.h>

#include "geometry/MeshHandle.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <algorithm>
#include <filesystem>
#include <numbers>
#include <span>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using Handle = SimpleFluid::MeshHandle<Pack>;
using Cartesian = SimpleFluid::Meshes::OrthogonalCartesian3D;
using Cylindrical = SimpleFluid::Meshes::OrthogonalCylindrial3D;
using SemiStructured = SimpleFluid::Meshes::SemiStructuredXY_Z;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

} // namespace

TEST(MeshHandleTest, VisitsEveryConcreteMeshAlternative)
{
    const auto cartesian = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {0.0, 1.0, 2.0},
            {0.0, 1.0},
            {0.0, 1.0}}});
    const auto cylindrical = std::make_shared<Cylindrical>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {1.0, 2.0},
            {0.0, std::numbers::pi},
            {0.0, 1.0}}});
    const auto semi_structured = std::make_shared<SemiStructured>(
        SimpleFluid::Arr<SemiStructured::Vec3>{
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0}},
        SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>{{0, 1, 2}},
        SimpleFluid::ArrReal{0.0, 1.0});

    const Handle cartesian_handle(cartesian);
    const Handle cylindrical_handle(cylindrical);
    const Handle semi_structured_handle(semi_structured);

    EXPECT_EQ(
        cartesian_handle.visit(
            [](const auto& mesh) { return mesh.num_cells(); }),
        cartesian->num_cells());
    EXPECT_EQ(cylindrical_handle.num_owned_cells(), 1U);
    EXPECT_EQ(semi_structured_handle.num_owned_cells(), 1U);

    for (const auto* handle :
         {&cartesian_handle, &cylindrical_handle,
          &semi_structured_handle})
    {
        EXPECT_EQ(
            handle->owned_cell_map()->getLocalNumElements(),
            handle->num_owned_cells());
        EXPECT_EQ(
            handle->overlap_face_map()->getLocalNumElements(),
            handle->num_faces());
        EXPECT_FALSE(handle->boundary_patches().empty());
    }
}

TEST(MeshHandleTest, PreservesLegacySTKMapsAndGeometry)
{
    auto legacy = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
    const Handle handle(legacy);

    EXPECT_TRUE(handle.is_stk());
    EXPECT_EQ(handle.legacy_mesh(), legacy);
    EXPECT_EQ(handle.owned_cell_map(), legacy->owned_cell_map());
    EXPECT_EQ(handle.overlap_cell_map(), legacy->overlap_cell_map());
    EXPECT_EQ(handle.owned_face_map(), legacy->owned_face_map());
    EXPECT_EQ(handle.cell_global_id(0),
              legacy->owned_cell_map()->getGlobalElement(0));
    EXPECT_EQ(handle.face_global_id(0),
              legacy->owned_face_map()->getGlobalElement(0));
    EXPECT_EQ(handle.num_owned_cells(), legacy->num_owned_cells());
    EXPECT_EQ(handle.cell_centroid(0), legacy->cell_centroid(0));
    EXPECT_EQ(handle.faces(0).size(), legacy->faces(0).size());
}

TEST(MeshHandleTest, ReusesMaterializedCellFaceConnectivity)
{
    const auto mesh = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {0.0, 1.0, 2.0},
            {0.0, 1.0},
            {0.0, 1.0}}});
    const Handle handle(mesh);

    static_assert(std::is_same_v<
        decltype(handle.faces(0)),
        std::span<const Handle::local_ordinal_type>>);

    const auto first = handle.faces(0);
    const auto second = handle.faces(0);
    EXPECT_EQ(first.data(), second.data());
    EXPECT_EQ(first.size(), 6U);
    EXPECT_TRUE(std::ranges::equal(first, second));
#ifdef CHECK_BOUNDS_ENABLED
    EXPECT_THROW(handle.faces(-1), std::out_of_range);
#endif
}

TEST(MeshHandleTest, BuildsOrthogonalSlabOwnershipAndGhostMaps)
{
    const auto mesh = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {0.0, 1.0, 2.0, 3.0, 4.0},
            {0.0, 1.0},
            {0.0, 1.0}}});
    const Handle handle(
        mesh,
        Handle::DistributionOptions{
            .ghost_layers = 1,
            .partition = 0,
            .partitions = 2});

    EXPECT_EQ(handle.num_owned_cells(), 2U);
    EXPECT_EQ(handle.num_local_cells(), 3U);
    EXPECT_EQ(handle.owned_cell_map()->getLocalNumElements(), 2U);
    EXPECT_EQ(handle.overlap_cell_map()->getLocalNumElements(), 3U);
    EXPECT_TRUE(handle.is_owned_cell(0));
    EXPECT_FALSE(handle.is_owned_cell(2));
}

TEST(MeshHandleTest, ExportsEveryCRTPMeshAlternative)
{
    const auto cartesian_file = "mesh_handle_cartesian.vtu";
    const auto cylindrical_file = "mesh_handle_cylindrical.vtu";
    const auto semi_structured_file =
        "mesh_handle_semi_structured.vtu";
    const auto cartesian = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {0.0, 1.0},
            {0.0, 1.0},
            {0.0, 1.0}}});
    const auto cylindrical = std::make_shared<Cylindrical>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {1.0, 2.0},
            {0.0,
             0.5 * std::numbers::pi,
             std::numbers::pi,
             1.5 * std::numbers::pi,
             2.0 * std::numbers::pi},
            {0.0, 1.0}}});
    const auto semi_structured = std::make_shared<SemiStructured>(
        SimpleFluid::Arr<SemiStructured::Vec3>{
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0}},
        SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>{{0, 1, 2}},
        SimpleFluid::ArrReal{0.0, 1.0});

    Handle(cartesian).export_vtu(cartesian_file);
    Handle(cylindrical).export_vtu(cylindrical_file);
    Handle(semi_structured).export_vtu(semi_structured_file);

    EXPECT_TRUE(std::filesystem::exists(cartesian_file));
    EXPECT_TRUE(std::filesystem::exists(cylindrical_file));
    EXPECT_TRUE(std::filesystem::exists(semi_structured_file));

    std::filesystem::remove(cartesian_file);
    std::filesystem::remove(cylindrical_file);
    std::filesystem::remove(semi_structured_file);
}
