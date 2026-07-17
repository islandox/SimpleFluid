/**
 * @file testBoundaryLayerMeshFactory.cc
 * @brief Tests for same-family in-place boundary-layer mesh refinement.
 */

#include <gtest/gtest.h>

#include "geometry/BoundaryLayerMeshFactory.hh"

#include <memory>
#include <numbers>
#include <ranges>
#include <stdexcept>
#include <type_traits>

namespace
{

using Factory = SimpleFluid::BoundaryLayerMeshFactory;
using Cartesian = SimpleFluid::Meshes::OrthogonalCartesian3D;
using Cylindrical = SimpleFluid::Meshes::OrthogonalCylindrial3D;
using SemiStructured = SimpleFluid::Meshes::SemiStructuredXY_Z;

SemiStructured make_semi_structured_mesh()
{
    return SemiStructured(
        {{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0},
         {2.0, 1.0, 0.0}, {0.0, 1.0, 0.0}},
        {{0, 1, 3}, {1, 2, 3}},
        {0.0, 1.0, 3.0},
        {{0, 1, "outer"}, {1, 2, "outer"},
         {2, 3, "outer"}, {3, 0, "outer"}});
}

void expect_edges_near(const SimpleFluid::ArrReal& actual,
                       const SimpleFluid::ArrReal& expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t edge = 0; edge < actual.size(); ++edge)
    {
        EXPECT_NEAR(actual[edge], expected[edge], 1.0e-14);
    }
}

} // namespace

static_assert(std::is_move_assignable_v<Cartesian>);
static_assert(std::is_move_assignable_v<Cylindrical>);
static_assert(std::is_move_assignable_v<SemiStructured>);

TEST(BoundaryLayerMeshFactoryTest,
     CartesianBuildsInPlaceFromDatabaseAndPreservesOrthogonality)
{
    auto database = std::make_shared<SimpleFluid::Database>();
    database->set("boundary_layer_boundary_names",
                  SimpleFluid::ArrString{"xmin", "xmax"});
    database->set("boundary_layer_counts", SimpleFluid::ArrInt{2, 1});
    database->set("boundary_layer_first_cell_heights",
                  SimpleFluid::ArrReal{0.1, 0.25});
    database->set("boundary_layer_growth_ratios",
                  SimpleFluid::ArrReal{2.0, 1.0});

    Factory factory(database);
    auto mesh = std::make_shared<Cartesian>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
        {0.0, 1.0, 2.0, 3.0},
        {-1.0, 1.0},
        {0.0, 2.0}}});
    const auto* original_address = mesh.get();

    factory.build(mesh);

    EXPECT_EQ(mesh.get(), original_address);
    expect_edges_near(
        mesh->cell_edges()[Cartesian::X],
        {0.0, 0.1, 0.3, 1.0, 2.0, 2.75, 3.0});
    EXPECT_EQ(mesh->cell_edges()[Cartesian::Y],
              (SimpleFluid::ArrReal{-1.0, 1.0}));
    EXPECT_EQ(mesh->cell_edges()[Cartesian::Z],
              (SimpleFluid::ArrReal{0.0, 2.0}));
    EXPECT_EQ(mesh->num_cells_per_dimension(),
              (SimpleFluid::Vec3D<unsigned>{6, 1, 1}));
    EXPECT_EQ(mesh->num_cells(), 6U);
    EXPECT_EQ(mesh->boundary_batch_name(0), "xmin");
    EXPECT_EQ(mesh->boundary_batch_name(1), "xmax");
    EXPECT_EQ(mesh->face_normal(
                  Cartesian::FaceID{1, 0, 0, Cartesian::X_FACE}),
              (Cartesian::Vec3{1.0, 0.0, 0.0}));

    const auto refined_edges = mesh->cell_edges();
    factory.build(mesh);
    EXPECT_EQ(mesh.get(), original_address);
    EXPECT_EQ(mesh->cell_edges(), refined_edges);
}

TEST(BoundaryLayerMeshFactoryTest,
     CylindricalPreservesPeriodicOrthogonalTopology)
{
    constexpr auto pi = std::numbers::pi_v<SimpleFluid::real_t>;
    Cylindrical mesh({{
        {1.0, 2.0, 4.0},
        {0.0, pi, 2.0 * pi},
        {0.0, 1.0, 3.0}}});
    Factory factory({
        {"rmax", 2, 0.25, 2.0},
        {"zmin", 2, 0.1, 2.0}});

    factory.build(mesh);

    EXPECT_TRUE(mesh.is_theta_periodic());
    EXPECT_EQ(mesh.cell_edges()[Cylindrical::R],
              (SimpleFluid::ArrReal{1.0, 2.0, 3.25, 3.75, 4.0}));
    EXPECT_EQ(mesh.cell_edges()[Cylindrical::THETA],
              (SimpleFluid::ArrReal{0.0, pi, 2.0 * pi}));
    expect_edges_near(
        mesh.cell_edges()[Cylindrical::AXIAL],
        {0.0, 0.1, 0.3, 1.0, 3.0});
    EXPECT_EQ(mesh.num_cells_per_dimension(),
              (SimpleFluid::Vec3D<unsigned>{4, 2, 4}));
    EXPECT_EQ(mesh.num_boundary_batches(), 4);
    EXPECT_EQ(mesh.boundary_batch_name(0), "rmin");
    EXPECT_EQ(mesh.boundary_batch_name(5), "zmax");
}

TEST(BoundaryLayerMeshFactoryTest,
     CylindricalSectorSupportsAngularBoundaryLayersInRadians)
{
    Cylindrical mesh({{
        {1.0, 2.0},
        {0.0, 1.0, 2.0},
        {0.0, 1.0}}});
    Factory factory({{"thetamin", 2, 0.1, 2.0}});

    factory.build(mesh);

    EXPECT_FALSE(mesh.is_theta_periodic());
    expect_edges_near(
        mesh.cell_edges()[Cylindrical::THETA],
        {0.0, 0.1, 0.3, 1.0, 2.0});
    EXPECT_EQ(mesh.num_boundary_batches(), 6);
    EXPECT_EQ(mesh.boundary_batch_name(2), "thetamin");
}

TEST(BoundaryLayerMeshFactoryTest,
     SemiStructuredPreservesXYTopologyWhileRefiningZ)
{
    auto mesh = make_semi_structured_mesh();
    const auto original_xy_nodes = mesh.xy_nodes();
    const auto original_xy_cells = mesh.xy_cell_nodes();
    Factory factory({
        {"zmin", 2, 0.1, 2.0},
        {"zmax", 1, 0.2, 1.0}});

    factory.build(mesh);

    EXPECT_EQ(mesh.xy_nodes(), original_xy_nodes);
    EXPECT_EQ(mesh.xy_cell_nodes(), original_xy_cells);
    expect_edges_near(
        mesh.z_edges(), {0.0, 0.1, 0.3, 1.0, 2.8, 3.0});
    EXPECT_EQ(mesh.num_cells(), 10U);
    EXPECT_EQ(mesh.topology().side_faces().size(), 5U);

    bool found_outer = false;
    for (const auto batch_id : mesh.boundary_batch_ids())
    {
        if (mesh.boundary_batch_name(batch_id) == "outer")
        {
            found_outer = true;
            EXPECT_EQ(mesh.boundary_face_batch(batch_id).face_lids.size(), 20U);
        }
    }
    EXPECT_TRUE(found_outer);
}

TEST(BoundaryLayerMeshFactoryTest,
     RejectsUnsupportedOrOverlappingLayersWithoutChangingInput)
{
    Cartesian mesh({{{0.0, 0.5, 1.0},
                     {0.0, 1.0},
                     {0.0, 1.0}}});
    const auto original_edges = mesh.cell_edges();

    Factory unknown_boundary({{"wall", 1, 0.1, 1.0}});
    EXPECT_THROW(unknown_boundary.build(mesh), std::invalid_argument);
    EXPECT_EQ(mesh.cell_edges(), original_edges);

    Factory overlapping({
        {"xmin", 1, 0.6, 1.0},
        {"xmax", 1, 0.5, 1.0}});
    EXPECT_THROW(overlapping.build(mesh), std::invalid_argument);
    EXPECT_EQ(mesh.cell_edges(), original_edges);

    auto semi_structured = make_semi_structured_mesh();
    const auto original_z = semi_structured.z_edges();
    Factory side_layer({{"outer", 1, 0.1, 1.0}});
    EXPECT_THROW(side_layer.build(semi_structured), std::invalid_argument);
    EXPECT_EQ(semi_structured.z_edges(), original_z);

    constexpr auto pi = std::numbers::pi_v<SimpleFluid::real_t>;
    Cylindrical periodic({{{1.0, 2.0},
                           {0.0, pi, 2.0 * pi},
                           {0.0, 1.0}}});
    const auto periodic_edges = periodic.cell_edges();
    Factory periodic_boundary({{"thetamin", 1, 0.1, 1.0}});
    EXPECT_THROW(periodic_boundary.build(periodic), std::invalid_argument);
    EXPECT_EQ(periodic.cell_edges(), periodic_edges);
}

TEST(BoundaryLayerMeshFactoryTest, ValidatesConfigurationAndNullInputs)
{
    EXPECT_THROW(
        Factory({{"xmin", 0, 0.1, 1.0}}), std::invalid_argument);
    EXPECT_THROW(
        Factory({{"xmin", 1, -0.1, 1.0}}), std::invalid_argument);
    EXPECT_THROW(
        Factory({{"xmin", 1, 0.1, 0.9}}), std::invalid_argument);
    EXPECT_THROW(
        Factory({{"xmin", 1, 0.1, 1.0},
                 {"xmin", 1, 0.2, 1.0}}),
        std::invalid_argument);

    Factory factory({{"xmin", 1, 0.1, 1.0}});
    SimpleFluid::SP<Cartesian> null_mesh;
    EXPECT_THROW(factory.build(null_mesh), std::invalid_argument);

    auto incomplete_database = std::make_shared<SimpleFluid::Database>();
    incomplete_database->set(
        "boundary_layer_boundary_names", SimpleFluid::ArrString{"xmin"});
    EXPECT_THROW((void)Factory{incomplete_database}, std::invalid_argument);

    auto empty_database = std::make_shared<SimpleFluid::Database>();
    Factory no_layers(empty_database);
    Cartesian unchanged({{{0.0, 1.0},
                          {0.0, 1.0},
                          {0.0, 1.0}}});
    const auto original_edges = unchanged.cell_edges();
    no_layers.build(unchanged);
    EXPECT_EQ(unchanged.cell_edges(), original_edges);
}
