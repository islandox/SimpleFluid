/** @file testAnnularSemiStructuredMeshFactory.cc
 * @brief Checks annular mesh dimensions, wall layers, and boundary geometry.
 */
#include <gtest/gtest.h>

#include "geometry/AnnularSemiStructuredMeshFactory.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/MeshQuality.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/mesh/FrontalDelaunay2D.hh"
#include "geometry/mesh/MultiRegionMesh.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <map>
#include <numbers>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
using Factory = SimpleFluid::AnnularSemiStructuredMeshFactory;
using Native = SimpleFluid::Meshes::SemiStructuredXY_Z;
using Composite = SimpleFluid::Meshes::MultiRegionMesh;
using Handle = SimpleFluid::MeshHandle<>;
testing::Environment* const environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

Factory::Options small_options()
{
    Factory::Options options;
    options.inner_radius = 0.04;
    options.outer_radius = 0.09;
    options.bottom = 0.2;
    options.top = 0.25;
    options.xy_spacing = 0.01;
    options.z_spacing = 0.02;
    options.wall_layers = 3;
    // Triangular cells at this wall width need centimetre tangential spacing
    // to meet the unchanged non-orthogonality gate. Production defaults use
    // eight layers with growth 1.25 and are checked separately below.
    options.first_layer_height = 0.002;
    options.growth_ratio = 1.5;
    // These compatibility checks exercise the original tensor-product wall
    // stacks. Dedicated tests below enable the unstructured corner patch.
    options.coarsen_bottom_corners = false;
    return options;
}

double segment_distance(const Native::Vec3& a, const Native::Vec3& b)
{
    const auto dx = b.x - a.x;
    const auto dy = b.y - a.y;
    const auto t = std::clamp(-(a.x * dx + a.y * dy) / (dx * dx + dy * dy), 0.0, 1.0);
    return std::hypot(a.x + t * dx, a.y + t * dy);
}

const Native& child(const Composite& composite, size_t region)
{
    return std::get<SimpleFluid::Meshes::NativeIsoRegion<Native>>(composite.regions()[region]).topology().native();
}

SimpleFluid::ArrReal axial_edges(const Composite& composite)
{
    SimpleFluid::ArrReal edges;
    for (size_t region = 0; region < composite.regions().size(); ++region)
    {
        const auto& z = child(composite, region).z_edges();
        edges.insert(edges.end(), z.begin(), z.end());
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    return edges;
}

/** Recover the two bulk interface loops without assuming factory node IDs. */
std::vector<SimpleFluid::Arr<Native::Vec3>> bulk_boundary_loops(const Native& native)
{
    std::map<std::pair<unsigned, unsigned>, unsigned> incidences;
    for (const auto& cell : native.xy_cell_nodes())
        if (cell.size() == 3)
            for (size_t j = 0; j < 3; ++j)
                ++incidences[std::minmax(cell[j], cell[(j + 1) % 3])];
    std::map<unsigned, std::vector<unsigned>> adjacency;
    for (const auto& [edge, count] : incidences)
    {
        if (count != 1 && count != 2)
            throw std::runtime_error("Bulk triangulation is non-manifold");
        if (count == 1)
        {
            adjacency[edge.first].push_back(edge.second);
            adjacency[edge.second].push_back(edge.first);
        }
    }
    for (const auto& [node, neighbors] : adjacency)
    {
        static_cast<void>(node);
        if (neighbors.size() != 2)
            throw std::runtime_error("Bulk interface is not a closed polygon loop");
    }
    std::set<unsigned> visited;
    std::vector<SimpleFluid::Arr<Native::Vec3>> loops;
    for (const auto& [start, neighbors] : adjacency)
    {
        static_cast<void>(neighbors);
        if (!visited.insert(start).second) continue;
        std::vector<unsigned> component{start};
        for (size_t i = 0; i < component.size(); ++i)
            for (const auto other : adjacency.at(component[i]))
                if (visited.insert(other).second) component.push_back(other);
        SimpleFluid::Arr<Native::Vec3> loop;
        for (const auto node : component) loop.push_back(native.xy_nodes()[node]);
        const auto angle = [](const Native::Vec3& p)
        {
            const auto value = std::atan2(p.y, p.x);
            return value < 0 ? value + 2 * std::numbers::pi : value;
        };
        std::sort(loop.begin(), loop.end(), [&](const auto& a, const auto& b) { return angle(a) < angle(b); });
        loops.push_back(std::move(loop));
    }
    if (loops.size() != 2)
        throw std::runtime_error("Expected exactly two annular bulk boundary loops");
    const auto mean_radius = [](const auto& loop)
    {
        double total = 0;
        for (const auto& node : loop) total += std::hypot(node.x, node.y);
        return total / loop.size();
    };
    std::sort(loops.begin(), loops.end(), [&](const auto& a, const auto& b) { return mean_radius(a) < mean_radius(b); });
    return loops; // inner first, outer second; each loop is CCW from +X.
}

using TriangleCoordinates = std::array<std::array<double, 2>, 3>;

template<class Indices>
TriangleCoordinates triangle_coordinates(const SimpleFluid::Arr<Native::Vec3>& nodes, const Indices& indices)
{
    TriangleCoordinates result;
    for (size_t j = 0; j < 3; ++j) result[j] = {nodes[indices[j]].x, nodes[indices[j]].y};
    std::sort(result.begin(), result.end());
    return result;
}

/** Check every quad's actual apothems and width, independently of node IDs. */
void expect_wall_quadrilaterals(const Native& native, const Factory::Result& result,
    const Factory::Options& options)
{
    const std::array<size_t, 2> angular{result.counts.inner_angular_cells, result.counts.outer_angular_cells};
    const std::array<size_t, 2> layers{
        options.refine_inner ? options.wall_layers : 0,
        options.refine_outer ? options.wall_layers : 0};
    std::array<std::vector<size_t>, 2> observed{
        std::vector<size_t>(layers[0]), std::vector<size_t>(layers[1])};
    for (const auto& cell : native.xy_cell_nodes())
    {
        if (cell.size() != 4) continue;
        double smallest = std::numeric_limits<double>::infinity(), largest = 0;
        for (const auto index : cell)
        {
            const auto& point = native.xy_nodes()[index];
            const auto radius = std::hypot(point.x, point.y);
            smallest = std::min(smallest, radius);
            largest = std::max(largest, radius);
        }
        bool matched = false;
        for (size_t side = 0; side < 2 && !matched; ++side)
        {
            const auto cosine = std::cos(std::numbers::pi / angular[side]);
            const auto boundary = side == 0 ? options.inner_radius : options.outer_radius * cosine;
            double offset = 0, width = options.first_layer_height;
            for (size_t layer = 0; layer < layers[side]; ++layer)
            {
                const auto actual_boundary = (side == 0 ? smallest : largest) * cosine;
                const auto expected_boundary = boundary + (side == 0 ? offset : -offset);
                if (std::abs(actual_boundary - expected_boundary) <= 1e-12)
                {
                    EXPECT_NEAR((largest - smallest) * cosine, width, 1e-12);
                    ++observed[side][layer];
                    matched = true;
                    break;
                }
                offset += width;
                width *= options.growth_ratio;
            }
        }
        EXPECT_TRUE(matched) << "Quadrilateral is outside the selected radial wall layers";
    }
    for (size_t side = 0; side < 2; ++side)
        for (const auto count : observed[side]) EXPECT_EQ(count, angular[side]);
}
} // namespace

TEST(AnnularSemiStructuredMeshFactoryTest, ContainsOnlyAnnularFluidAndPreservesNativeVolume)
{
    const auto options = small_options();
    const Factory factory(options);
    const auto result = factory.build();
    const auto& mesh = *result.mesh;
    EXPECT_EQ(result.counts, factory.planned_counts());
    EXPECT_EQ(result.counts.cells, mesh.num_cells());
    EXPECT_EQ(result.counts.nodes, mesh.num_nodes());
    EXPECT_EQ(result.counts.faces, mesh.num_faces());
    EXPECT_EQ(result.counts.regions, mesh.regions().size());
    EXPECT_EQ(result.counts.mixed_xy_cells, result.counts.xy_cells);
    EXPECT_EQ(result.counts.mixed_xy_cells, result.counts.wall_xy_cells + result.counts.bulk_xy_cells);
    EXPECT_EQ(result.angular_cells, result.counts.outer_angular_cells);
    EXPECT_EQ(result.counts.angular_cells, result.counts.outer_angular_cells);
    for (const auto count : {result.counts.inner_angular_cells, result.counts.outer_angular_cells})
    {
        EXPECT_EQ(count % 2, 0U);
        ASSERT_GE(count, 8U);
    }
    EXPECT_LE(2 * std::numbers::pi * options.outer_radius / result.counts.outer_angular_cells, options.xy_spacing);
    const auto inner_front_apothem = result.radial_apothems[options.wall_layers];
    EXPECT_LE(2 * std::numbers::pi * inner_front_apothem /
        (result.counts.inner_angular_cells * std::cos(std::numbers::pi / result.counts.inner_angular_cells)),
        options.xy_spacing + 1e-14);

    for (size_t region = 0; region < mesh.regions().size(); ++region)
    {
        const auto& native = child(mesh, region);
        size_t quads = 0, triangles = 0;
        EXPECT_EQ(result.counts.xy_nodes, native.xy_nodes().size());
        for (const auto& node : native.xy_nodes())
        {
            EXPECT_EQ(node.z, 0);
            EXPECT_GE(std::hypot(node.x, node.y), options.inner_radius - 1e-14);
            EXPECT_LE(std::hypot(node.x, node.y), options.outer_radius + 1e-14);
        }
        for (const auto& cell : native.xy_cell_nodes())
        {
            ASSERT_TRUE(cell.size() == 3 || cell.size() == 4);
            if (cell.size() == 4)
            {
                ++quads;
            }
            else ++triangles;
            double signed_area = 0;
            for (size_t j = 0; j < cell.size(); ++j)
            {
                const auto& a = native.xy_nodes()[cell[j]];
                const auto& b = native.xy_nodes()[cell[(j + 1) % cell.size()]];
                signed_area += a.x * b.y - a.y * b.x;
                EXPECT_GE(segment_distance(a, b), options.inner_radius - 1e-14);
            }
            EXPECT_GT(signed_area, 0);
        }
        EXPECT_EQ(quads, options.wall_layers *
            (result.counts.inner_angular_cells + result.counts.outer_angular_cells));
        EXPECT_EQ(quads, result.counts.wall_xy_cells);
        EXPECT_EQ(triangles, result.counts.bulk_xy_cells);
        expect_wall_quadrilaterals(native, result, options);
    }
    long double total_volume = 0;
    std::vector<unsigned> incidences(mesh.num_faces());
    size_t hex_cells = 0, prism_cells = 0;
    for (size_t c = 0; c < mesh.num_cells(); ++c)
    {
        EXPECT_GT(mesh.cell_volume(c), 0);
        total_volume += mesh.cell_volume(c);
        if (mesh.cell_type(c) == SimpleFluid::MeshUtils::CellType::HEXAHEDRON) ++hex_cells;
        else if (mesh.cell_type(c) == SimpleFluid::MeshUtils::CellType::TRIPRISM) ++prism_cells;
        else ADD_FAILURE() << "Unexpected transition volume cell type";
        for (const auto f : mesh.cell_faces(c)) ++incidences[f];
    }
    EXPECT_EQ(hex_cells, result.counts.hex_cells);
    EXPECT_EQ(prism_cells, result.counts.prism_cells);
    for (size_t f = 0; f < mesh.num_faces(); ++f)
    {
        EXPECT_EQ(incidences[f], mesh.neighbor_cell(f) == Composite::invalid_cell_id() ? 1U : 2U);
    }
    const auto polygon_area = result.counts.outer_angular_cells * std::tan(std::numbers::pi / result.counts.outer_angular_cells)
        * std::pow(result.radial_apothems.back(), 2)
        - result.counts.inner_angular_cells * std::tan(std::numbers::pi / result.counts.inner_angular_cells)
        * std::pow(options.inner_radius, 2);
    EXPECT_NEAR(result.cross_section_area, polygon_area, 1e-14);
    EXPECT_NEAR(static_cast<double>(total_volume), result.cross_section_area * (options.top - options.bottom), 1e-14);
    EXPECT_GT(result.relative_area_deficit, 0);
    EXPECT_LT(result.cross_section_area, result.analytic_cross_section_area);
    std::set<std::string> names;
    for (const auto batch : mesh.boundary_batch_ids()) names.insert(mesh.boundary_batch_name(batch));
    EXPECT_EQ(names, (std::set<std::string>{"rmin", "rmax", "zmin", "zmax"}));
}

TEST(AnnularSemiStructuredMeshFactoryTest, BulkTrianglesComeFromPublicFrontalDelaunayAnnulus)
{
    const auto options = small_options();
    const auto result = Factory(options).build();
    const auto& native = child(*result.mesh, 0);
    const auto loops = bulk_boundary_loops(native);
    ASSERT_EQ(loops[0].size(), result.counts.inner_angular_cells);
    ASSERT_EQ(loops[1].size(), result.counts.outer_angular_cells);
    const auto direct = SimpleFluid::Meshes::FrontalDelaunay2D::triangulate_annulus(
        loops[1], loops[0], options.xy_spacing);
    std::set<TriangleCoordinates> expected, actual;
    for (const auto& triangle : direct.triangles)
        EXPECT_TRUE(expected.insert(triangle_coordinates(direct.nodes, triangle)).second);
    for (const auto& cell : native.xy_cell_nodes())
        if (cell.size() == 3)
            EXPECT_TRUE(actual.insert(triangle_coordinates(native.xy_nodes(), cell)).second);
    ASSERT_EQ(actual.size(), result.counts.bulk_xy_cells);
    EXPECT_EQ(actual, expected);
}

TEST(AnnularSemiStructuredMeshFactoryTest, PreservesWallNormalGrowthAndDeterministicAxialFractions)
{
    auto options = small_options();
    options.refine_top = true;
    const Factory factory(options);
    const auto result = factory.build();
    const auto repeated = factory.build();
    ASSERT_EQ(result.mesh->regions().size(), 3U);
    ASSERT_EQ(result.mesh->regions().size(), repeated.mesh->regions().size());
    for (size_t r = 0; r < result.mesh->regions().size(); ++r)
    {
        EXPECT_EQ(child(*result.mesh, r).xy_nodes(), child(*repeated.mesh, r).xy_nodes());
        EXPECT_EQ(child(*result.mesh, r).xy_cell_nodes(), child(*repeated.mesh, r).xy_cell_nodes());
        EXPECT_EQ(child(*result.mesh, r).z_edges(), child(*repeated.mesh, r).z_edges());
        EXPECT_EQ(child(*result.mesh, r).xy_cell_nodes(), child(*result.mesh, 0).xy_cell_nodes());
        EXPECT_EQ(child(*result.mesh, r).xy_cell_nodes().size(), result.counts.mixed_xy_cells);
        expect_wall_quadrilaterals(child(*result.mesh, r), result, options);
    }
    ASSERT_EQ(result.mesh->interfaces().size(), 2U);
    for (const auto& declared : result.mesh->interfaces())
    {
        const auto& interface = std::get<SimpleFluid::Meshes::ExplicitConformingInterface>(declared);
        const auto& lower = child(*result.mesh, interface.first_region);
        const auto& upper = child(*result.mesh, interface.second_region);
        ASSERT_EQ(interface.faces.size(), result.counts.mixed_xy_cells);
        std::set<uint64_t> lower_faces, upper_faces;
        for (const auto& [first, second] : interface.faces)
        {
            const auto lower_face = lower.indexer().face_id(first);
            const auto upper_face = upper.indexer().face_id(second);
            EXPECT_TRUE(lower_faces.insert(first).second);
            EXPECT_TRUE(upper_faces.insert(second).second);
            EXPECT_EQ(lower_face.ij, upper_face.ij);
            EXPECT_LT(lower.face_normal(lower_face).dot(upper.face_normal(upper_face)), -0.999999);
            EXPECT_NEAR(lower.face_area(lower_face), upper.face_area(upper_face), 1e-14);
            EXPECT_EQ(lower.face_centroid(lower_face), upper.face_centroid(upper_face));
        }
        EXPECT_EQ(lower_faces.size(), upper_faces.size());
    }
    EXPECT_EQ(result.axial_fractions, repeated.axial_fractions);
    const auto& radial = result.radial_apothems;
    const auto z = axial_edges(*result.mesh);
    for (size_t i = 0; i < options.wall_layers; ++i)
    {
        const auto width = options.first_layer_height * std::pow(options.growth_ratio, i);
        EXPECT_NEAR(radial[i + 1] - radial[i], width, 1e-14);
        EXPECT_NEAR(radial[radial.size() - i - 1] - radial[radial.size() - i - 2], width, 1e-14);
        EXPECT_NEAR(z[i + 1] - z[i], width, 1e-14);
        EXPECT_NEAR(z[z.size() - i - 1] - z[z.size() - i - 2], width, 1e-14);
    }
    EXPECT_EQ(z.front(), options.bottom);
    EXPECT_EQ(z.back(), options.top);
    EXPECT_EQ(result.axial_fractions.front(), 0);
    EXPECT_EQ(result.axial_fractions.back(), 1);
    for (size_t i = 1; i < z.size(); ++i)
    {
        EXPECT_GT(result.axial_fractions[i], result.axial_fractions[i - 1]);
        EXPECT_LE(z[i] - z[i - 1], options.z_spacing + 1e-14);
    }
    // Interior radial coordinates are planning guides; the actual bulk
    // topology is supplied by frontal-Delaunay point placement above.
}

TEST(AnnularSemiStructuredMeshFactoryTest, CanDisableLayersAndPlanTheCentimetreR100Mesh)
{
    auto options = small_options();
    options.wall_layers = 0;
    const auto result = Factory(options).build();
    EXPECT_EQ(result.counts.axial_cells, 3U);
    EXPECT_EQ(result.counts.hex_cells, 0U);
    EXPECT_EQ(result.counts.wall_xy_cells, 0U);
    EXPECT_EQ(result.counts.bulk_xy_cells, result.counts.xy_cells);
    EXPECT_EQ(result.counts.prism_cells, result.counts.cells);
    EXPECT_EQ(result.mesh->regions().size(), 1U);
    EXPECT_TRUE(result.mesh->interfaces().empty());
    for (const auto& xy : child(*result.mesh, 0).xy_cell_nodes()) EXPECT_EQ(xy.size(), 3U);
    const auto loops = bulk_boundary_loops(child(*result.mesh, 0));
    EXPECT_EQ(loops[0].size(), result.counts.inner_angular_cells);
    EXPECT_EQ(loops[1].size(), result.counts.outer_angular_cells);
    Factory::Options r100;
    r100.inner_radius = 0.03815;
    r100.outer_radius = 0.25;
    r100.bottom = 0.1;
    r100.top = 0.60852;
    r100.coarsen_bottom_corners = false;
    const auto counts = Factory(r100).planned_counts();
    EXPECT_EQ(counts.angular_cells, 158U);
    EXPECT_EQ(counts.outer_angular_cells, 158U);
    EXPECT_EQ(counts.inner_angular_cells, 50U);
    EXPECT_EQ(counts.axial_cells, 55U);
    EXPECT_EQ(counts.wall_xy_cells, 1664U);
    EXPECT_GT(counts.bulk_xy_cells, 0U);
    EXPECT_EQ(counts.mixed_xy_cells, counts.wall_xy_cells + counts.bulk_xy_cells);
    EXPECT_EQ(counts.bottom_axial_cells, 8U);
    EXPECT_EQ(counts.bulk_axial_cells, 47U);
    EXPECT_EQ(counts.top_axial_cells, 0U);
    EXPECT_EQ(counts.hex_cells, 91520U);
    EXPECT_EQ(counts.prism_cells, counts.bulk_xy_cells * counts.axial_cells);
    EXPECT_EQ(counts.cells, counts.hex_cells + counts.prism_cells);
    EXPECT_EQ(counts.cells, counts.xy_cells * counts.axial_cells);
    EXPECT_GT(counts.axial_cells, r100.wall_layers);
}

TEST(AnnularSemiStructuredMeshFactoryTest, DisabledRadialWallsKeepOnlyRequestedQuadrilateralLayers)
{
    for (const auto [inner, outer] : {std::pair{false, true}, std::pair{true, false}, std::pair{false, false}})
    {
        SCOPED_TRACE(inner);
        SCOPED_TRACE(outer);
        auto options = small_options();
        options.refine_inner = inner;
        options.refine_outer = outer;
        const auto result = Factory(options).build();
        const auto expected_wall = options.wall_layers *
            ((inner ? result.counts.inner_angular_cells : 0) + (outer ? result.counts.outer_angular_cells : 0));
        EXPECT_EQ(result.counts.wall_xy_cells, expected_wall);
        EXPECT_EQ(result.counts.hex_cells, expected_wall * result.counts.axial_cells);
        for (size_t region = 0; region < result.mesh->regions().size(); ++region)
        {
            const auto& native = child(*result.mesh, region);
            expect_wall_quadrilaterals(native, result, options);
            const auto quads = std::count_if(native.xy_cell_nodes().begin(), native.xy_cell_nodes().end(),
                [](const auto& cell) { return cell.size() == 4; });
            EXPECT_EQ(static_cast<size_t>(quads), expected_wall);
        }
        const auto loops = bulk_boundary_loops(child(*result.mesh, 0));
        EXPECT_EQ(loops[0].size(), result.counts.inner_angular_cells);
        EXPECT_EQ(loops[1].size(), result.counts.outer_angular_cells);
    }
}

TEST(AnnularSemiStructuredMeshFactoryTest, ExplicitIndependentAngularCountsPreserveMatchingInterfaces)
{
    // Odd explicit counts are valid too; only automatic sizing rounds even.
    for (const auto [inner, outer] : {std::pair<size_t, size_t>{20, 40}, {21, 41}})
    {
        SCOPED_TRACE(inner);
        SCOPED_TRACE(outer);
        auto options = small_options();
        options.xy_spacing = 0.02;
        options.inner_angular_cells = inner;
        options.outer_angular_cells = outer;
        const auto result = Factory(options).build();
        EXPECT_EQ(result.counts.inner_angular_cells, inner);
        EXPECT_EQ(result.counts.outer_angular_cells, outer);
        EXPECT_EQ(result.counts.angular_cells, outer);
        EXPECT_EQ(result.angular_cells, outer);
        EXPECT_EQ(result.counts.wall_xy_cells, options.wall_layers * (inner + outer));
        const auto loops = bulk_boundary_loops(child(*result.mesh, 0));
        EXPECT_EQ(loops[0].size(), inner);
        EXPECT_EQ(loops[1].size(), outer);
        for (size_t region = 0; region < result.mesh->regions().size(); ++region)
            expect_wall_quadrilaterals(child(*result.mesh, region), result, options);
        ASSERT_EQ(result.mesh->interfaces().size(), 1U);
        const auto& interface = std::get<SimpleFluid::Meshes::ExplicitConformingInterface>(
            result.mesh->interfaces().front());
        const auto& lower = child(*result.mesh, interface.first_region);
        const auto& upper = child(*result.mesh, interface.second_region);
        ASSERT_EQ(interface.faces.size(), result.counts.xy_cells);
        EXPECT_EQ(lower.xy_cell_nodes(), upper.xy_cell_nodes());
        for (const auto& [first, second] : interface.faces)
        {
            const auto lower_face = lower.indexer().face_id(first);
            const auto upper_face = upper.indexer().face_id(second);
            EXPECT_EQ(lower_face.ij, upper_face.ij);
            EXPECT_EQ(lower.face_centroid(lower_face), upper.face_centroid(upper_face));
            EXPECT_NEAR(lower.face_area(lower_face), upper.face_area(upper_face), 1e-14);
        }
    }
}

TEST(AnnularSemiStructuredMeshFactoryTest, RejectsUnsupportedExplicitAngularCounts)
{
    for (auto member : {&Factory::Options::inner_angular_cells, &Factory::Options::outer_angular_cells})
    {
        for (size_t count = 1; count < 8; ++count)
        {
            SCOPED_TRACE(count);
            auto options = small_options();
            options.*member = count;
            EXPECT_THROW((Factory(options)), std::invalid_argument);
        }
        if constexpr (std::numeric_limits<size_t>::max() > std::numeric_limits<unsigned>::max())
        {
            auto options = small_options();
            options.*member = static_cast<size_t>(std::numeric_limits<unsigned>::max()) + 1;
            EXPECT_THROW((Factory(options)), std::overflow_error);
        }
    }
}

TEST(AnnularSemiStructuredMeshFactoryTest, CentimetreR100DefaultsMeetUnchangedALEQualityGate)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "The complete native default geometry is checked on one rank.";
    Factory::Options options;
    options.inner_radius = 0.03815;
    options.outer_radius = 0.25;
    options.bottom = 0.1;
    options.top = 0.60852;
    const auto result = Factory(options).build();
    EXPECT_EQ(result.counts.corner_layers, 5U);
    EXPECT_EQ(result.counts.corner_cells, 14U * (50U + 158U));
    EXPECT_EQ(result.counts.removed_corner_cells, 11U * (50U + 158U));
    EXPECT_EQ(result.counts.regions, 6U);
    EXPECT_EQ(result.counts.hex_cells, 89232U);
    EXPECT_EQ(result.counts.prism_cells, 178420U);
    EXPECT_EQ(result.counts.cells, 267652U);
    EXPECT_EQ(result.mesh->num_cells(), result.counts.cells);
    EXPECT_EQ(result.mesh->num_faces(), result.counts.faces);
    EXPECT_EQ(result.mesh->num_nodes(), result.counts.nodes);
    size_t swept_regions = 0, swept_cells = 0;
    for (const auto& region : result.mesh->regions())
        if (const auto* swept = std::get_if<SimpleFluid::Meshes::SweptRZRegion>(&region))
        {
            ++swept_regions;
            swept_cells += swept->layout().cells;
            EXPECT_EQ(swept->topology().indexer().num_cells_per_layer, 14U);
            EXPECT_NEAR(swept->geometry().axial_bounds()[0], options.bottom, 1e-14);
            EXPECT_NEAR(swept->geometry().axial_bounds()[1] - options.bottom, 0.0164140625, 1e-14);
            for (const auto& loop : swept->geometry().rz_cell_nodes()) EXPECT_EQ(loop.size(), 4U);
        }
    EXPECT_EQ(swept_regions, 2U);
    EXPECT_EQ(swept_cells, result.counts.corner_cells);
    long double volume = 0;
    for (size_t cell = 0; cell < result.mesh->num_cells(); ++cell)
        volume += result.mesh->cell_volume(cell);
    EXPECT_NEAR(static_cast<double>(volume), result.cross_section_area * (options.top - options.bottom), 2e-13);
    const Handle handle(result.mesh);
    const auto metrics = SimpleFluid::evaluate_mesh_quality(handle);
    const SimpleFluid::MeshQualityGate gate;
    const auto assessment = gate.assess(metrics);
    EXPECT_TRUE(assessment.accepted()) << assessment.report(metrics);
    EXPECT_EQ(metrics.global_cell_count, result.counts.cells);
    EXPECT_EQ(metrics.global_face_count, result.counts.faces);
    RecordProperty("maximum_growth_ratio", std::to_string(metrics.maximum_growth_ratio));
    RecordProperty("maximum_aspect_ratio", std::to_string(metrics.maximum_aspect_ratio));
}

TEST(AnnularSemiStructuredMeshFactoryTest, RejectsMalformedDimensionsAndOverlappingLayersBeforeXYAllocation)
{
    for (auto member : {&Factory::Options::inner_radius, &Factory::Options::outer_radius,
             &Factory::Options::xy_spacing, &Factory::Options::z_spacing, &Factory::Options::first_layer_height})
    {
        auto options = small_options();
        options.*member = 0;
        EXPECT_THROW((Factory(options)), std::invalid_argument);
        options.*member = std::numeric_limits<double>::quiet_NaN();
        EXPECT_THROW((Factory(options)), std::invalid_argument);
    }
    auto options = small_options();
    options.top = options.bottom;
    EXPECT_THROW((Factory(options)), std::invalid_argument);
    options = small_options();
    options.growth_ratio = 0.5;
    EXPECT_THROW((Factory(options)), std::invalid_argument);
    options = small_options();
    options.first_layer_height = 0.02;
    EXPECT_THROW(Factory(options).planned_counts(), std::invalid_argument);
    options = small_options();
    options.xy_spacing = std::numeric_limits<double>::min();
    EXPECT_THROW(Factory(options).planned_counts(), std::overflow_error);
    options = small_options();
    options.xy_spacing = 1e-7;
    EXPECT_THROW(Factory(options).planned_counts(), std::overflow_error);
    options = small_options();
    options.wall_layers = std::numeric_limits<size_t>::max();
    EXPECT_THROW((Factory(options)), std::overflow_error);
    options = small_options();
    options.inner_radius = options.outer_radius * 0.9999;
    options.wall_layers = 0;
    EXPECT_THROW(Factory(options).planned_counts(), std::invalid_argument);
}

TEST(AnnularSemiStructuredMeshFactoryTest, CompositeWrapperPartitionsAndMovesWithoutMutatingChild)
{
    for (const bool refine_top : {false, true})
    {
        auto options = small_options();
        options.refine_top = refine_top;
        SCOPED_TRACE(refine_top);
        const auto result = Factory(options).build();
        const auto original_z = axial_edges(*result.mesh);
        auto handle = std::make_shared<Handle>(result.mesh);
        const auto comm = Tpetra::getDefaultComm();
        EXPECT_EQ(handle->owned_cell_map()->getGlobalNumElements(), result.counts.cells);
        const auto vtu = handle->vtu_topology();
        size_t previous_offset = 0;
        long long local_hex = 0, local_prism = 0;
        for (size_t c = 0; c < vtu->cell_offsets.size(); ++c)
        {
            const auto type = vtu->cell_types[c];
            ASSERT_TRUE(type == 12 || type == 13);
            EXPECT_EQ(vtu->cell_offsets[c] - previous_offset, type == 12 ? 8U : 6U);
            previous_offset = vtu->cell_offsets[c];
            if (type == 12) ++local_hex;
            else ++local_prism;
        }
        long long global_hex = 0, global_prism = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_hex, &global_hex);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_prism, &global_prism);
        EXPECT_EQ(global_hex, result.counts.hex_cells);
        EXPECT_EQ(global_prism, result.counts.prism_cells);
        double local_volume = 0;
        for (size_t c = 0; c < handle->num_owned_cells(); ++c) local_volume += handle->cell_volume(c);
        double global_volume = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_volume, &global_volume);
        EXPECT_NEAR(global_volume, result.cross_section_area * (options.top - options.bottom), 1e-13);
        SimpleFluid::PlanarALEMeshMotion<SimpleFluid::DefaultTpetraTypes> motion(handle);
        motion.begin_trial(options.top + 0.01, 0.1);
        EXPECT_EQ(axial_edges(*result.mesh), original_z);
        local_volume = 0;
        for (size_t c = 0; c < handle->num_owned_cells(); ++c) local_volume += handle->cell_volume(c);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_volume, &global_volume);
        EXPECT_NEAR(global_volume, result.cross_section_area * (options.top + 0.01 - options.bottom), 1e-13);
        motion.rollback_trial();
        EXPECT_EQ(axial_edges(*result.mesh), original_z);
    }
}

TEST(AnnularSemiStructuredMeshFactoryTest, CornerSelectionUsesStrictHalfOfTheSmallerRequestedSpacing)
{
    auto options = small_options();
    options.coarsen_bottom_corners = true;
    options.first_layer_height = 0.00125;
    options.growth_ratio = 2;
    // 1.25 and 2.5 mm qualify; exactly 5 mm does not.
    auto counts = Factory(options).planned_counts();
    EXPECT_EQ(counts.corner_layers, 2U);
    EXPECT_GT(counts.removed_corner_cells, 0U);
    options.z_spacing = 0.005;
    // The 2.5 mm threshold now leaves only one layer, with no coarsening.
    counts = Factory(options).planned_counts();
    EXPECT_EQ(counts.corner_layers, 0U);
    EXPECT_EQ(counts.corner_cells, 0U);
    EXPECT_EQ(counts.removed_corner_cells, 0U);
    options.coarsen_bottom_corners = false;
    EXPECT_EQ(counts, Factory(options).planned_counts());
}

TEST(AnnularSemiStructuredMeshFactoryTest, CornerCoarseningDoesNothingWithoutAnEligibleWallIntersection)
{
    for (unsigned scenario = 0; scenario < 5; ++scenario)
    {
        SCOPED_TRACE(scenario);
        auto options = small_options();
        options.coarsen_bottom_corners = true;
        if (scenario == 0) options.refine_bottom = false;
        if (scenario == 1) options.refine_inner = options.refine_outer = false;
        if (scenario == 2) options.wall_layers = 0;
        if (scenario == 3) options.wall_layers = 1;
        if (scenario == 4) options.first_layer_height = 0.005; // None is strictly below 5 mm.
        const auto enabled = Factory(options).planned_counts();
        EXPECT_EQ(enabled.corner_layers, 0U);
        EXPECT_EQ(enabled.corner_cells, 0U);
        EXPECT_EQ(enabled.removed_corner_cells, 0U);
        options.coarsen_bottom_corners = false;
        EXPECT_EQ(enabled, Factory(options).planned_counts());
    }
}

TEST(AnnularSemiStructuredMeshFactoryTest, CornerTransitionsPreserveConformingFacesAndAffineMotion)
{
    auto options = small_options();
    options.coarsen_bottom_corners = true;
    options.z_spacing = 0.01;
    const auto result = Factory(options).build();
    EXPECT_EQ(result.counts.corner_layers, 3U);
    EXPECT_EQ(result.counts.corner_cells,
        5U * (result.counts.inner_angular_cells + result.counts.outer_angular_cells));
    EXPECT_EQ(result.counts.removed_corner_cells,
        4U * (result.counts.inner_angular_cells + result.counts.outer_angular_cells));
    EXPECT_EQ(result.counts.cells + result.counts.removed_corner_cells,
        result.counts.xy_cells * result.counts.axial_cells);
    EXPECT_EQ(result.counts.cells, result.mesh->num_cells());
    EXPECT_EQ(result.counts.faces, result.mesh->num_faces());
    EXPECT_EQ(result.counts.nodes, result.mesh->num_nodes());

    std::set<uint64_t> canonical_interfaces;
    for (const auto& declaration : result.mesh->interfaces())
    {
        const auto* interface = std::get_if<SimpleFluid::Meshes::ExplicitConformingInterface>(&declaration);
        ASSERT_NE(interface, nullptr);
        std::set<uint64_t> first_faces, second_faces;
        for (const auto& [first, second] : interface->faces)
        {
            EXPECT_TRUE(first_faces.insert(first).second);
            EXPECT_TRUE(second_faces.insert(second).second);
            const auto canonical = result.mesh->canonical_face({interface->first_region, first});
            EXPECT_EQ(canonical, result.mesh->canonical_face({interface->second_region, second}));
            EXPECT_TRUE(canonical_interfaces.insert(canonical).second);
            EXPECT_NE(result.mesh->neighbor_cell(canonical), Composite::invalid_cell_id());
            const auto metrics = [&](size_t region, uint64_t face)
            {
                return std::visit([&](const auto& child)
                {
                    return std::pair{child.geometry().face_centroid(face), child.geometry().face_area_vector(face)};
                }, result.mesh->regions()[region]);
            };
            const auto [first_center, first_vector] = metrics(interface->first_region, first);
            const auto [second_center, second_vector] = metrics(interface->second_region, second);
            EXPECT_NEAR((first_center - second_center).norm(), 0, 2e-13);
            EXPECT_NEAR((first_vector + second_vector).norm(), 0, 2e-13);
        }
    }
    std::set<std::string> boundaries;
    for (const auto id : result.mesh->boundary_batch_ids()) boundaries.insert(result.mesh->boundary_batch_name(id));
    EXPECT_EQ(boundaries, (std::set<std::string>{"rmin", "rmax", "zmin", "zmax"}));

    auto handle = std::make_shared<Handle>(result.mesh);
    const auto comm = Tpetra::getDefaultComm();
    const auto global_volume = [&]()
    {
        double local = 0, total = 0;
        for (size_t cell = 0; cell < handle->num_owned_cells(); ++cell) local += handle->cell_volume(cell);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local, &total);
        return total;
    };
    EXPECT_NEAR(global_volume(), result.cross_section_area * (options.top - options.bottom), 1e-13);
    const auto signature = result.mesh->configuration_signature();
    SimpleFluid::PlanarALEMeshMotion<> motion(handle);
    motion.begin_trial(options.top + 0.005, 0.1);
    EXPECT_NEAR(global_volume(), result.cross_section_area * (options.top + 0.005 - options.bottom), 1e-13);
    EXPECT_EQ(result.mesh->configuration_signature(), signature);
    motion.rollback_trial();
    EXPECT_EQ(result.mesh->axial_edges(), (SimpleFluid::ArrReal{options.bottom, options.top}));
    EXPECT_NEAR(global_volume(), result.cross_section_area * (options.top - options.bottom), 1e-13);
}

TEST(AnnularSemiStructuredMeshFactoryTest, CornerCoarseningTracksIndependentlySelectedRadialWalls)
{
    for (const bool inner : {false, true})
    {
        auto options = small_options();
        options.coarsen_bottom_corners = true;
        options.refine_inner = inner;
        options.refine_outer = !inner;
        const auto result = Factory(options).build();
        const auto angular = inner ? result.counts.inner_angular_cells : result.counts.outer_angular_cells;
        EXPECT_EQ(result.counts.corner_layers, 3U);
        EXPECT_EQ(result.counts.corner_cells, 5U * angular);
        EXPECT_EQ(result.counts.removed_corner_cells, 4U * angular);
        size_t swept_regions = 0;
        for (const auto& region : result.mesh->regions())
            swept_regions += std::holds_alternative<SimpleFluid::Meshes::SweptRZRegion>(region);
        EXPECT_EQ(swept_regions, 1U);
        long double volume = 0;
        for (size_t cell = 0; cell < result.mesh->num_cells(); ++cell)
            volume += result.mesh->cell_volume(cell);
        EXPECT_NEAR(static_cast<double>(volume), result.cross_section_area * (options.top - options.bottom), 1e-13);
    }
}
