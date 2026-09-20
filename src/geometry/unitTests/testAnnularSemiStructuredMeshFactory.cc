/** @file testAnnularSemiStructuredMeshFactory.cc */
#include <gtest/gtest.h>

#include "geometry/AnnularSemiStructuredMeshFactory.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/MeshQuality.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/mesh/MultiRegionMesh.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <set>
#include <stdexcept>
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
    options.xy_spacing = 0.025;
    options.z_spacing = 0.02;
    options.wall_layers = 3;
    // Three layers on this deliberately coarse fixture must still connect
    // smoothly to 2--2.5 cm bulk spacing. Production centimetre defaults use
    // eight layers starting at 1 mm and are checked separately below.
    options.first_layer_height = 0.002;
    options.growth_ratio = 1.5;
    return options;
}

double segment_distance(const Native::Vec3& a, const Native::Vec3& b)
{
    const auto dx = b.x - a.x;
    const auto dy = b.y - a.y;
    const auto t = std::clamp(-(a.x * dx + a.y * dy) / (dx * dx + dy * dy), 0.0, 1.0);
    return std::hypot(a.x + t * dx, a.y + t * dy);
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
    EXPECT_EQ(result.counts.xy_nodes, mesh.xy_nodes().size());
    EXPECT_EQ(result.angular_cells % 2, 0U);
    EXPECT_GE(result.angular_cells, 8U);
    EXPECT_LE(2 * std::numbers::pi * options.outer_radius / result.angular_cells, options.xy_spacing);

    for (const auto& node : mesh.xy_nodes())
    {
        EXPECT_EQ(node.z, 0);
        EXPECT_GE(std::hypot(node.x, node.y), options.inner_radius - 1e-14);
        EXPECT_LE(std::hypot(node.x, node.y), options.outer_radius + 1e-14);
    }
    for (const auto& cell : mesh.xy_cell_nodes())
    {
        ASSERT_EQ(cell.size(), 4U);
        double signed_area = 0;
        for (size_t j = 0; j < cell.size(); ++j)
        {
            const auto& a = mesh.xy_nodes()[cell[j]];
            const auto& b = mesh.xy_nodes()[cell[(j + 1) % cell.size()]];
            signed_area += a.x * b.y - a.y * b.x;
            EXPECT_GE(segment_distance(a, b), options.inner_radius - 1e-14);
        }
        EXPECT_GT(signed_area, 0);
    }
    long double total_volume = 0;
    std::vector<unsigned> incidences(mesh.num_faces());
    for (size_t c = 0; c < mesh.num_cells(); ++c)
    {
        const auto id = mesh.indexer().cell_id(c);
        EXPECT_GT(mesh.cell_volume(id), 0);
        total_volume += mesh.cell_volume(id);
        for (const auto f : mesh.faces(id)) ++incidences[mesh.indexer().face_ordinal(f)];
    }
    for (size_t f = 0; f < mesh.num_faces(); ++f)
    {
        const auto face = mesh.indexer().face_id(f);
        EXPECT_EQ(incidences[f], mesh.neighbor_cell(face) == Native::invalid_cell_id() ? 1U : 2U);
    }
    const auto polygon_area = result.angular_cells * std::tan(std::numbers::pi / result.angular_cells)
        * (std::pow(result.radial_apothems.back(), 2) - std::pow(result.radial_apothems.front(), 2));
    EXPECT_NEAR(result.cross_section_area, polygon_area, 1e-14);
    EXPECT_NEAR(static_cast<double>(total_volume), result.cross_section_area * (options.top - options.bottom), 1e-14);
    EXPECT_GT(result.relative_area_deficit, 0);
    EXPECT_LT(result.cross_section_area, result.analytic_cross_section_area);
    std::set<std::string> names;
    for (const auto batch : mesh.boundary_batch_ids()) names.insert(mesh.boundary_batch_name(batch));
    EXPECT_EQ(names, (std::set<std::string>{"rmin", "rmax", "zmin", "zmax"}));
}

TEST(AnnularSemiStructuredMeshFactoryTest, PreservesWallNormalGrowthAndDeterministicAxialFractions)
{
    auto options = small_options();
    options.refine_top = true;
    const Factory factory(options);
    const auto result = factory.build();
    const auto repeated = factory.build();
    EXPECT_EQ(result.mesh->xy_nodes(), repeated.mesh->xy_nodes());
    EXPECT_EQ(result.mesh->xy_cell_nodes(), repeated.mesh->xy_cell_nodes());
    EXPECT_EQ(result.mesh->z_edges(), repeated.mesh->z_edges());
    EXPECT_EQ(result.axial_fractions, repeated.axial_fractions);
    const auto& radial = result.radial_apothems;
    const auto& z = result.mesh->z_edges();
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
    // The bulk gap is evenly subdivided, not intersected with unrelated
    // uniform edges that can leave arbitrarily thin transition slivers.
    const auto radial_bulk_width = radial[options.wall_layers + 1] - radial[options.wall_layers];
    for (size_t i = options.wall_layers + 1; i + options.wall_layers + 1 < radial.size(); ++i)
        EXPECT_NEAR(radial[i + 1] - radial[i], radial_bulk_width, 1e-14);
}

TEST(AnnularSemiStructuredMeshFactoryTest, CanDisableLayersAndPlanTheCentimetreR100Mesh)
{
    auto options = small_options();
    options.wall_layers = 0;
    const auto result = Factory(options).build();
    EXPECT_EQ(result.counts.axial_cells, 3U);
    EXPECT_EQ(result.counts.radial_cells, 2U);
    Factory::Options r100;
    r100.inner_radius = 0.03815;
    r100.outer_radius = 0.25;
    r100.bottom = 0.1;
    r100.top = 0.60852;
    const auto counts = Factory(r100).planned_counts();
    EXPECT_EQ(counts.angular_cells, 158U);
    EXPECT_GT(counts.cells, 100000U);
    EXPECT_LT(counts.cells, 1000000U);
    EXPECT_GT(counts.radial_cells, 2 * r100.wall_layers);
    EXPECT_GT(counts.axial_cells, r100.wall_layers);
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
    const Handle handle(result.mesh);
    const auto metrics = SimpleFluid::evaluate_mesh_quality(handle);
    const SimpleFluid::MeshQualityGate gate;
    const auto assessment = gate.assess(metrics);
    EXPECT_TRUE(assessment.accepted()) << assessment.report(metrics);
    EXPECT_EQ(metrics.global_cell_count, result.counts.cells);
    EXPECT_EQ(metrics.global_face_count, result.counts.faces);
    EXPECT_LE(metrics.maximum_growth_ratio, 2.0);
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
    const auto options = small_options();
    const auto result = Factory(options).build();
    const auto original_z = result.mesh->z_edges();
    auto geometry = std::make_shared<Composite>(
        std::vector<Composite::Region>{SimpleFluid::Meshes::native_region("fluid", result.mesh)},
        std::vector<Composite::Interface>{}, SimpleFluid::Meshes::InterfaceTolerance{},
        Composite::BoundaryNamePolicy::MergeMatchingNames);
    auto handle = std::make_shared<Handle>(geometry);
    const auto comm = Tpetra::getDefaultComm();
    EXPECT_EQ(handle->owned_cell_map()->getGlobalNumElements(), result.counts.cells);
    double local_volume = 0;
    for (size_t c = 0; c < handle->num_owned_cells(); ++c) local_volume += handle->cell_volume(c);
    double global_volume = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_volume, &global_volume);
    EXPECT_NEAR(global_volume, result.cross_section_area * (options.top - options.bottom), 1e-13);
    SimpleFluid::PlanarALEMeshMotion<SimpleFluid::DefaultTpetraTypes> motion(handle);
    motion.begin_trial(options.top + 0.01, 0.1);
    EXPECT_EQ(result.mesh->z_edges(), original_z);
    local_volume = 0;
    for (size_t c = 0; c < handle->num_owned_cells(); ++c) local_volume += handle->cell_volume(c);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_volume, &global_volume);
    EXPECT_NEAR(global_volume, result.cross_section_area * (options.top + 0.01 - options.bottom), 1e-13);
    motion.rollback_trial();
    EXPECT_EQ(result.mesh->z_edges(), original_z);
}
