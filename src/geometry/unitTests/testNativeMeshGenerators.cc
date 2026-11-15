#include "benchmarks/BenchmarkMesh.hh"
#include "examples/PitzDailyMesh.hh"
#include "geometry/details/NativeMeshGenerators.hh"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace
{
using namespace SimpleFluid;
using Data = Meshes::UnstructuredMeshData;
using Options = detail::NativeMeshGeneratorOptions;
constexpr double pi = std::numbers::pi;

double volume(const Data& data)
{
    return std::accumulate(
        data.cells.begin(), data.cells.end(), 0.0, [](double total, const auto& cell) { return total + cell.volume; });
}

std::map<std::string, size_t> boundary_counts(const Data& data)
{
    std::map<std::string, size_t> result;
    for (const auto& face : data.faces)
        if (face.boundary_id >= 0)
            ++result[data.boundary_names.at(face.boundary_id)];
    return result;
}

TEST(NativeMeshGeneratorsTest, BoxRetainsAxesSourceOrderSideIdentityAndMetrics)
{
    Options o;
    o.box_cell_edges = {ArrReal{2, 3, 5}, ArrReal{-1, 2}, ArrReal{4, 8}};
    o.domain_exterior_face_types = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "unused"};
    const auto data = detail::generate_native_mesh(o);
    ASSERT_EQ(data.cells.size(), 2);
    ASSERT_EQ(data.nodes.size(), 12);
    ASSERT_EQ(data.faces.size(), 11);
    EXPECT_DOUBLE_EQ(volume(data), 36);
    EXPECT_DOUBLE_EQ(data.cells[0].center.x, 2.5);
    EXPECT_DOUBLE_EQ(data.cells[1].center.x, 4);
    EXPECT_EQ(data.cells[0].node_ids, (std::vector<Data::ID>{1, 2, 5, 4, 7, 8, 11, 10}));
    EXPECT_EQ(data.cells[0].face_ids, (std::vector<Data::ID>{11, 12, 13, 14, 15, 16}));
    EXPECT_EQ(data.faces[0].node_ids, (std::vector<Data::ID>{1, 2, 8, 7}));
    EXPECT_EQ(data.faces[1].neighbor, 2);
    EXPECT_DOUBLE_EQ(data.faces[1].area, 12);
    EXPECT_DOUBLE_EQ(data.faces[1].normal_from_owner.x, 1);
    EXPECT_EQ(data.boundary_names.size(), 7);
    EXPECT_EQ(data.boundary_names.at(1), "xmin");
    EXPECT_EQ(data.boundary_names.at(2), "xmax");
    EXPECT_EQ(data.boundary_names.at(3), "ymin");
    EXPECT_EQ(data.boundary_names.at(4), "ymax");
    EXPECT_EQ(data.boundary_names.at(5), "zmin");
    EXPECT_EQ(data.boundary_names.at(6), "zmax");
    EXPECT_EQ(data.boundary_names.at(7), "unused");
    EXPECT_EQ(data.faces[0].boundary_id, 3);
    EXPECT_EQ(data.faces[3].boundary_id, 1);
}

TEST(NativeMeshGeneratorsTest, BoxBoundaryLayersRetainBaseCellCountAndWallWidths)
{
    Options o;
    o.box_cell_edges = {ArrReal{0, .25, .5, .75, 1}, ArrReal{0, 1}, ArrReal{0, 1}};
    o.domain_exterior_face_types = {"xmin", "xmax", "wall", "wall", "wall", "wall"};
    o.boundary_layer_specs = {{"xmin", 1, .1, 1}, {"xmax", 1, .2, 1}};
    const auto data = detail::generate_native_mesh(o);
    ASSERT_EQ(data.cells.size(), 4);
    EXPECT_NEAR(data.cells[0].volume, .1, 1e-15);
    EXPECT_NEAR(data.cells[1].volume, .35, 1e-15);
    EXPECT_NEAR(data.cells[2].volume, .35, 1e-15);
    EXPECT_NEAR(data.cells[3].volume, .2, 1e-15);
    EXPECT_NEAR(volume(data), 1, 1e-15);
    EXPECT_EQ(data.boundary_names.size(), 3);
    EXPECT_EQ(data.boundary_names.at(3), "wall");
}

TEST(NativeMeshGeneratorsTest, CylinderUsesPolygonalDiskAndPrismExtrusion)
{
    Options o;
    o.domain_type = 1;
    o.radius = 1;
    o.cylinder_height = 2;
    o.mesh_size = 1;
    o.cylinder_circumferential_mesh_size = 1;
    o.domain_exterior_face_types = {"radial", "bottom", "top"};
    const auto data = detail::generate_native_mesh(o);
    // Seven outer segments: extrusion must conserve their polygonal area,
    // without normalizing to the analytic disk area.
    const auto expected = 7.0 * std::sin(2 * pi / 7);
    EXPECT_NEAR(volume(data), expected, 2e-13);
    for (const auto& cell : data.cells)
        EXPECT_EQ(cell.type, MeshUtils::CellType::TRIPRISM);
    const auto counts = boundary_counts(data);
    EXPECT_EQ(counts.at("radial"), 14);
    EXPECT_EQ(counts.at("bottom"), counts.at("top"));
    EXPECT_NEAR(data.nodes[1].coordinates.x, std::cos(pi / 7), 1e-15);
    EXPECT_NEAR(data.nodes[1].coordinates.y, std::sin(pi / 7), 1e-15);
}

TEST(NativeMeshGeneratorsTest, AnnulusClosesSeamAndKeepsFacetedVolume)
{
    Options o;
    o.domain_type = 4;
    o.annulus_cell_edges = {ArrReal{1, 2}, ArrReal{0, pi / 2, pi, 3 * pi / 2, 2 * pi}, ArrReal{0, 3}};
    o.domain_exterior_face_types = {"inner", "outer", "bottom", "top"};
    const auto data = detail::generate_native_mesh(o);
    EXPECT_EQ(data.nodes.size(), 16);
    EXPECT_EQ(data.cells.size(), 4);
    EXPECT_NEAR(volume(data), 18, 2e-14);
    EXPECT_EQ(
        boundary_counts(data), (std::map<std::string, size_t>{{"inner", 4}, {"outer", 4}, {"bottom", 4}, {"top", 4}}));
    EXPECT_EQ(std::count_if(data.faces.begin(), data.faces.end(),
                  [](const auto& face) { return face.neighbor != Data::invalid_id; }),
        4);
    EXPECT_EQ(data.cells.back().node_ids[2], 2);
}

TEST(NativeMeshGeneratorsTest, AnnularSectorRetainsPhysicalThetaFaces)
{
    Options o;
    o.domain_type = 4;
    o.annulus_cell_edges = {ArrReal{1, 2}, ArrReal{0, pi / 3}, ArrReal{0, 2}};
    o.domain_exterior_face_types = {"inner", "outer", "theta0", "theta1", "bottom", "top"};
    const auto data = detail::generate_native_mesh(o);
    EXPECT_EQ(data.nodes.size(), 8);
    EXPECT_NEAR(volume(data), 3 * std::sin(pi / 3), 1e-14);
    EXPECT_EQ(boundary_counts(data).size(), 6);
    EXPECT_EQ(boundary_counts(data).at("theta0"), 1);
    EXPECT_EQ(boundary_counts(data).at("theta1"), 1);
}

TEST(NativeMeshGeneratorsTest, SphereUsesSpherifiedCubeCoordinatesAndFacetedMetrics)
{
    Options o;
    o.domain_type = 2;
    o.radius = 2;
    o.mesh_size = 4;
    o.domain_exterior_face_types = {"surface"};
    const auto data = detail::generate_native_mesh(o);
    ASSERT_EQ(data.cells.size(), 1);
    ASSERT_EQ(data.nodes.size(), 8);
    const auto a = 2 / std::sqrt(3.0);
    EXPECT_NEAR(data.nodes.front().coordinates.x, -a, 1e-15);
    EXPECT_NEAR(data.nodes.back().coordinates.z, a, 1e-15);
    EXPECT_NEAR(volume(data), 8 * a * a * a, 1e-13);
    for (const auto& face : data.faces)
        EXPECT_NEAR(face.area, 4 * a * a, 1e-14);
    o.mesh_size = 2;
    const auto refined = detail::generate_native_mesh(o);
    EXPECT_EQ(refined.nodes.size(), 27);
    EXPECT_DOUBLE_EQ(refined.nodes[13].coordinates.norm(), 0);
    EXPECT_DOUBLE_EQ(refined.nodes[4].coordinates.z, -2);
    EXPECT_NEAR(refined.nodes[1].coordinates.y, -std::sqrt(2.0), 1e-15);
}

TEST(NativeMeshGeneratorsTest, BuilderPreservesSparseIdsAndExactWedgeSideOrder)
{
    std::vector<Data::Node> nodes{
        {101, {0, 0, 0}}, {103, {1, 0, 0}}, {107, {0, 1, 0}}, {109, {0, 0, 2}}, {113, {1, 0, 2}}, {127, {0, 1, 2}}};
    Data::Cell cell;
    cell.id = 47;
    cell.type = MeshUtils::CellType::TRIPRISM;
    cell.node_ids = {101, 103, 107, 109, 113, 127};
    const auto data = detail::build_native_mesh_data(std::move(nodes), {cell}, {{47, 2, "side"}}, {"side"});
    EXPECT_EQ(data.cells[0].id, 47);
    EXPECT_DOUBLE_EQ(data.cells[0].volume, 1);
    EXPECT_NEAR(data.cells[0].center.x, 1.0 / 3, 1e-15);
    EXPECT_NEAR(data.cells[0].center.y, 1.0 / 3, 1e-15);
    EXPECT_DOUBLE_EQ(data.cells[0].center.z, 1);
    EXPECT_EQ(data.faces[2].id, 473);
    EXPECT_EQ(data.faces[2].node_ids, (std::vector<Data::ID>{101, 109, 127, 107}));
    EXPECT_DOUBLE_EQ(data.faces[2].normal_from_owner.x, -1);
}

TEST(NativeMeshGeneratorsTest, PitzDailyNativeBuilderKeepsOutlineInventoryAndGrading)
{
    const auto data = detail::pitz_daily_mesh::generate(18, 1);
    // One upstream x cell, ten main x cells, two outlet x cells;
    // two cells on either side of the step centerline.
    EXPECT_EQ(data.cells.size(), 50);
    const double area = .0206 * .0254 + .206 * .0508 + .084 * (.0508 + .0332) / 2;
    EXPECT_NEAR(volume(data), area * .001, 1e-18);
    EXPECT_EQ(boundary_counts(data), (std::map<std::string, size_t>{{"inlet", 2}, {"outlet", 4}, {"upperWall", 13},
                                         {"lowerWall", 15}, {"frontAndBack", 100}}));
    std::map<std::string, double> areas;
    for (const auto& face : data.faces)
        if (face.boundary_id >= 0)
            areas[data.boundary_names.at(face.boundary_id)] += face.area;
    EXPECT_NEAR(areas.at("inlet"), .0254 * .001, 1e-18);
    EXPECT_NEAR(areas.at("outlet"), .0332 * .001, 1e-18);
    EXPECT_NEAR(areas.at("frontAndBack"), 2 * area, 1e-16);
    const auto fine = detail::pitz_daily_mesh::generate(9, 1);
    EXPECT_EQ(fine.cells.size(), 169);
    EXPECT_NEAR(volume(fine), area * .001, 2e-18);
    // Nonuniform grading survives the native construction seam.
    EXPECT_NE(fine.cells[0].volume, fine.cells[1].volume);
}

TEST(NativeMeshGeneratorsTest, BenchmarkNativeBuilderPreservesAffineShearAndMetrics)
{
    const auto data = detail::benchmark_mesh::generate({2, 2, 1}, .6);
    ASSERT_EQ(data.cells.size(), 4);
    EXPECT_EQ(data.boundary_names.at(1), "ymin");
    EXPECT_EQ(data.faces[0].boundary_id, 1);
    EXPECT_NEAR(volume(data), 1, 1e-15);
    EXPECT_NEAR(data.cells[0].center.x, .4, 1e-15);
    EXPECT_NEAR(data.cells[0].center.y, .25, 1e-15);
    EXPECT_NEAR(data.nodes.back().coordinates.x, 1.6, 1e-15);
    EXPECT_EQ(boundary_counts(data),
        (std::map<std::string, size_t>{{"xmin", 2}, {"xmax", 2}, {"ymin", 2}, {"ymax", 2}, {"zmin", 4}, {"zmax", 4}}));
    const auto face = std::find_if(data.faces.begin(), data.faces.end(),
        [&](const auto& f) { return f.boundary_id >= 0 && data.boundary_names.at(f.boundary_id) == "xmin"; });
    ASSERT_NE(face, data.faces.end());
    EXPECT_NEAR(face->area, .5 * std::sqrt(1.36), 1e-15);
    EXPECT_NEAR(face->normal_from_owner.x, -1 / std::sqrt(1.36), 1e-15);
    EXPECT_NEAR(face->normal_from_owner.y, .6 / std::sqrt(1.36), 1e-15);
}

TEST(NativeMeshGeneratorsTest, InvalidCylinderLengthAndSpacingRejectBeforeCountConversion)
{
    Options base;
    base.domain_type = 1;
    base.domain_exterior_face_types = {"radial", "bottom", "top"};
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    for (const double invalid : {nan, infinity, -infinity, 0.0, -1.0})
    {
        auto options = base;
        options.radius = invalid;
        EXPECT_THROW(detail::generate_native_mesh(options), std::runtime_error);
        options = base;
        options.cylinder_height = invalid;
        EXPECT_THROW(detail::generate_native_mesh(options), std::runtime_error);
        options = base;
        options.mesh_size = invalid;
        EXPECT_THROW(detail::generate_native_mesh(options), std::runtime_error);
    }
}

TEST(NativeMeshGeneratorsTest, UnrepresentableCellCountsRejectBeforeAllocation)
{
    Options options;
    options.domain_type = 1;
    options.domain_exterior_face_types = {"radial", "bottom", "top"};
    // An exactly represented finite quotient at size_t's exclusive upper bound.
    options.radius = std::ldexp(1.0, std::numeric_limits<size_t>::digits);
    EXPECT_THROW(detail::generate_native_mesh(options), std::overflow_error);
    options.radius = 1;
    options.mesh_size = std::numeric_limits<double>::denorm_min();
    EXPECT_THROW(detail::generate_native_mesh(options), std::overflow_error);
    options.domain_type = 2;
    options.domain_exterior_face_types = {"surface"};
    options.mesh_size = 1;
    options.radius = std::numeric_limits<double>::max();
    EXPECT_THROW(detail::generate_native_mesh(options), std::runtime_error);
}
} // namespace
