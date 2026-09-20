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
    options.xy_spacing = 0.01;
    options.z_spacing = 0.02;
    options.wall_layers = 3;
    // Triangular cells at this wall width need centimetre tangential spacing
    // to meet the unchanged non-orthogonality gate. Production defaults use
    // eight layers with growth 1.25 and are checked separately below.
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
    EXPECT_EQ(result.counts.coarse_xy_cells, result.counts.radial_cells * result.angular_cells);
    EXPECT_EQ(result.counts.mixed_xy_cells, result.counts.xy_cells);
    EXPECT_EQ(result.angular_cells % 2, 0U);
    EXPECT_GE(result.angular_cells, 8U);
    EXPECT_LE(2 * std::numbers::pi * options.outer_radius / result.angular_cells, options.xy_spacing);

    const auto middle = result.counts.bottom_axial_cells ? 1U : 0U;
    for (size_t region = 0; region < mesh.regions().size(); ++region)
    {
        const auto& native = child(mesh, region);
        EXPECT_EQ(result.counts.xy_nodes, native.xy_nodes().size());
        for (const auto& node : native.xy_nodes())
        {
            EXPECT_EQ(node.z, 0);
            EXPECT_GE(std::hypot(node.x, node.y), options.inner_radius - 1e-14);
            EXPECT_LE(std::hypot(node.x, node.y), options.outer_radius + 1e-14);
        }
        for (const auto& cell : native.xy_cell_nodes())
        {
            const auto ring = *std::min_element(cell.begin(), cell.end()) / result.angular_cells;
            const bool wall = region != middle || ring < options.wall_layers
                || ring >= result.counts.radial_cells - options.wall_layers;
            ASSERT_EQ(cell.size(), wall ? 4U : 3U);
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
    ASSERT_EQ(result.mesh->regions().size(), 3U);
    ASSERT_EQ(result.mesh->regions().size(), repeated.mesh->regions().size());
    for (size_t r = 0; r < result.mesh->regions().size(); ++r)
    {
        EXPECT_EQ(child(*result.mesh, r).xy_nodes(), child(*repeated.mesh, r).xy_nodes());
        EXPECT_EQ(child(*result.mesh, r).xy_cell_nodes(), child(*repeated.mesh, r).xy_cell_nodes());
        EXPECT_EQ(child(*result.mesh, r).z_edges(), child(*repeated.mesh, r).z_edges());
    }
    ASSERT_EQ(result.mesh->interfaces().size(), 2U);
    for (const auto& declared : result.mesh->interfaces())
    {
        const auto& interface = std::get<SimpleFluid::Meshes::NonconformingInterface>(declared);
        const auto& coarse = child(*result.mesh, interface.coarse_region);
        const auto& fine = child(*result.mesh, interface.fine_region);
        ASSERT_EQ(interface.faces.size(), result.counts.coarse_xy_cells);
        size_t fine_count = 0;
        for (const auto& xy : coarse.xy_cell_nodes()) EXPECT_EQ(xy.size(), 4U);
        for (const auto& mapping : interface.faces)
        {
            ASSERT_TRUE(mapping.fine_faces.size() == 1 || mapping.fine_faces.size() == 2);
            const auto coarse_face = coarse.indexer().face_id(mapping.coarse_face);
            double fine_area = 0;
            for (const auto native_face : mapping.fine_faces)
            {
                const auto fine_face = fine.indexer().face_id(native_face);
                fine_area += fine.face_area(fine_face);
                EXPECT_LT(coarse.face_normal(coarse_face).dot(fine.face_normal(fine_face)), -0.999999);
            }
            EXPECT_NEAR(coarse.face_area(coarse_face), fine_area, 1e-14);
            fine_count += mapping.fine_faces.size();
        }
        EXPECT_EQ(fine_count, result.counts.mixed_xy_cells);
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
    EXPECT_EQ(result.counts.radial_cells, 5U);
    EXPECT_EQ(result.counts.hex_cells, 0U);
    EXPECT_EQ(result.counts.prism_cells, result.counts.cells);
    EXPECT_EQ(result.mesh->regions().size(), 1U);
    EXPECT_TRUE(result.mesh->interfaces().empty());
    for (const auto& xy : child(*result.mesh, 0).xy_cell_nodes()) EXPECT_EQ(xy.size(), 3U);
    Factory::Options r100;
    r100.inner_radius = 0.03815;
    r100.outer_radius = 0.25;
    r100.bottom = 0.1;
    r100.top = 0.60852;
    const auto counts = Factory(r100).planned_counts();
    EXPECT_EQ(counts.angular_cells, 158U);
    EXPECT_EQ(counts.radial_cells, 30U);
    EXPECT_EQ(counts.axial_cells, 55U);
    EXPECT_EQ(counts.coarse_xy_cells, 4740U);
    EXPECT_EQ(counts.mixed_xy_cells, 6952U);
    EXPECT_EQ(counts.bottom_axial_cells, 8U);
    EXPECT_EQ(counts.bulk_axial_cells, 47U);
    EXPECT_EQ(counts.top_axial_cells, 0U);
    EXPECT_EQ(counts.hex_cells, 156736U);
    EXPECT_EQ(counts.prism_cells, 207928U);
    EXPECT_EQ(counts.cells, 364664U);
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
