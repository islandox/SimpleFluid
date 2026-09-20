/**
 * @file testMeshToMeshTransferPolygon.cc
 * @brief Exact annular-sector overlaps with straight polygonal extrusions.
 */

#include <gtest/gtest.h>

#include "fields/MeshToMeshTransfer.hh"
#include "geometry/MeshReorderingFactory.hh"
#include "geometry/mesh/MultiRegionMesh.hh"
#include "utils/testing_environment.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numbers>
#include <utility>
#include <vector>

namespace
{
using Pack = SimpleFluid::DefaultTpetraTypes;
using Handle = SimpleFluid::MeshHandle<Pack>;
using Transfer = SimpleFluid::MeshToMeshTransfer<Pack>;
using Options = SimpleFluid::MeshToMeshTransferOptions;
using Quantity = SimpleFluid::MeshToMeshQuantity;
using ScalarField = SimpleFluid::ScalarCellFieldStored<Pack>;
using LO = Pack::local_ordinal_type;
namespace Meshes = SimpleFluid::Meshes;

testing::Environment* const polygon_environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

Options partial_polygon_options()
{
    Options options;
    options.method = SimpleFluid::MeshToMeshTransferMethod::ConservativeCellAverage;
    options.coverage_mode = SimpleFluid::MeshToMeshCoverageMode::AllowPartial;
    return options;
}

SimpleFluid::SP<Handle> polygon_mesh(
    const SimpleFluid::Arr<Handle::Vec3>& points,
    const SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>& cells,
    bool composite = false)
{
    auto native = std::make_shared<Handle::SemiStructured>(
        points, cells, SimpleFluid::ArrReal{0.0, 1.0});
    if (!composite) return std::make_shared<Handle>(native);
    auto geometry = std::make_shared<Handle::MultiRegion>(
        std::vector<Handle::MultiRegion::Region>{Meshes::native_region("polygon", native)},
        std::vector<Handle::MultiRegion::Interface>{});
    return std::make_shared<Handle>(geometry);
}

SimpleFluid::SP<Handle> quarter_annulus(double inner = 0.5, double outer = 1.0,
                                     bool split_angle = false)
{
    const double pi = std::numbers::pi;
    SimpleFluid::ArrReal theta = split_angle
        ? SimpleFluid::ArrReal{0.0, pi / 4.0, pi / 2.0}
        : SimpleFluid::ArrReal{0.0, pi / 2.0};
    return std::make_shared<Handle>(std::make_shared<Handle::Cylindrical>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {inner, outer}, std::move(theta), {0.0, 1.0}}}));
}

void expect_polygon_report(const SimpleFluid::MeshToMeshConservationReport& report,
                           double source, double transferred, std::size_t component = 0)
{
    ASSERT_GT(report.source_integral.size(), component);
    ASSERT_GT(report.transferred_integral.size(), component);
    ASSERT_GT(report.target_integral.size(), component);
    ASSERT_GT(report.uncovered_source_integral.size(), component);
    ASSERT_GT(report.transfer_error.size(), component);
    ASSERT_GT(report.conservation_error.size(), component);
    const double tolerance = 3.0e-12 * std::max(1.0, std::abs(source));
    EXPECT_NEAR(report.source_integral[component], source, tolerance);
    EXPECT_NEAR(report.transferred_integral[component], transferred, tolerance);
    EXPECT_NEAR(report.target_integral[component], transferred, tolerance);
    EXPECT_NEAR(report.uncovered_source_integral[component], source - transferred, tolerance);
    EXPECT_NEAR(report.transfer_error[component], 0.0, tolerance);
    EXPECT_NEAR(report.conservation_error[component], 0.0, tolerance);
}
} // namespace

TEST(MeshToMeshTransferPolygonTest, TriangleQuarterRingUsesExactAreasInBothDirections)
{
    const double overlap = 3.0 * std::numbers::pi / 16.0;
    for (const bool composite : {false, true})
    {
        SCOPED_TRACE(composite);
        auto polygon = polygon_mesh({{0, 0, 0}, {2, 0, 0}, {0, 2, 0}}, {{0, 1, 2}}, composite);
        auto annulus = quarter_annulus();
        for (const bool reverse : {false, true})
        {
            SCOPED_TRACE(reverse);
            auto source_mesh = reverse ? polygon : annulus;
            auto target_mesh = reverse ? annulus : polygon;
            const double source_volume = reverse ? 2.0 : overlap;
            const double target_volume = reverse ? overlap : 2.0;
            auto full = partial_polygon_options();
            full.coverage_mode = SimpleFluid::MeshToMeshCoverageMode::RequireFull;
            EXPECT_THROW((Transfer(source_mesh, target_mesh, full)), std::invalid_argument);
            Transfer transfer(source_mesh, target_mesh, partial_polygon_options());
            const auto& coverage = transfer.coverage();
            EXPECT_NEAR(coverage.source_volume, source_volume, 1.0e-13);
            EXPECT_NEAR(coverage.target_volume, target_volume, 1.0e-13);
            EXPECT_NEAR(coverage.overlap_volume, overlap, 1.0e-13);
            EXPECT_NEAR(coverage.uncovered_source_volume, source_volume - overlap, 1.0e-13);
            EXPECT_NEAR(coverage.uncovered_target_volume, target_volume - overlap, 1.0e-13);

            ScalarField source(source_mesh, "source"), target(target_mesh, "target");
            source.put_scalar(4.0);
            const auto intensive = transfer.project(source, target, Quantity::Intensive);
            EXPECT_NEAR(target.value(0), 4.0 * overlap / target_volume, 1.0e-13);
            expect_polygon_report(intensive, 4.0 * source_volume, 4.0 * overlap);

            source.put_scalar(8.0);
            const auto extensive = transfer.project(source, target, Quantity::Extensive);
            EXPECT_NEAR(target.value(0), 8.0 * overlap / source_volume, 1.0e-13);
            expect_polygon_report(extensive, 8.0, 8.0 * overlap / source_volume);
            // Partial projection must not silently become normalized apply().
            EXPECT_THROW(transfer.apply(source, target), std::invalid_argument);
        }
    }
}

TEST(MeshToMeshTransferPolygonTest, QuadrilateralCircleCutRetainsCurvedSegment)
{
    // Quarter unit disk with x >= 1/2: integral sqrt(1-x*x) dx, x in [1/2,1].
    // The inner radius 1/10 does not intersect this quadrilateral.
    const double overlap = std::numbers::pi / 6.0 - std::sqrt(3.0) / 8.0;
    auto polygon = polygon_mesh(
        {{0.5, 0, 0}, {1.5, 0, 0}, {1.5, 1.5, 0}, {0.5, 1.5, 0}}, {{0, 1, 2, 3}});
    auto annulus = quarter_annulus(0.1, 1.0);
    for (const bool reverse : {false, true})
    {
        SCOPED_TRACE(reverse);
        auto source = reverse ? polygon : annulus;
        auto target = reverse ? annulus : polygon;
        Transfer transfer(source, target, partial_polygon_options());
        EXPECT_NEAR(transfer.coverage().overlap_volume, overlap, 2.0e-13);
        ScalarField density(source, "density"), result(target, "result");
        density.put_scalar(1.0);
        const auto report = transfer.project(density, result, Quantity::Intensive);
        EXPECT_NEAR(result.value(0), overlap / target->cell_volume(0), 2.0e-13);
        expect_polygon_report(report, source->cell_volume(0), overlap);
    }
}

TEST(MeshToMeshTransferPolygonTest, SignedMulticomponentInventoriesUseSameExactOverlap)
{
    const double overlap = 3.0 * std::numbers::pi / 16.0;
    const std::array<double, 3> values{2.0, -3.0, 0.25};
    auto polygon = polygon_mesh({{0, 0, 0}, {2, 0, 0}, {0, 2, 0}}, {{0, 1, 2}});
    auto annulus = quarter_annulus();
    for (const bool reverse : {false, true})
    {
        auto source_mesh = reverse ? polygon : annulus;
        auto target_mesh = reverse ? annulus : polygon;
        Transfer transfer(source_mesh, target_mesh, partial_polygon_options());
        Pack::multi_vector_type source(source_mesh->owned_cell_map(), values.size());
        Pack::multi_vector_type target(target_mesh->owned_cell_map(), values.size());
        {
            auto view = source.getLocalViewHost(Tpetra::Access::OverwriteAll);
            for (std::size_t j = 0; j < values.size(); ++j) view(0, j) = values[j];
        }
        for (const auto quantity : {Quantity::Intensive, Quantity::Extensive})
        {
            const auto report = transfer.project(source, target, quantity);
            const auto view = target.getLocalViewHost(Tpetra::Access::ReadOnly);
            for (std::size_t j = 0; j < values.size(); ++j)
            {
                const double density = values[j] / (quantity == Quantity::Extensive
                    ? source_mesh->cell_volume(0) : 1.0);
                EXPECT_NEAR(view(0, j), density * overlap / (quantity == Quantity::Intensive
                    ? target_mesh->cell_volume(0) : 1.0), 2.0e-13);
                expect_polygon_report(report, density * source_mesh->cell_volume(0), density * overlap, j);
            }
        }
    }
}

TEST(MeshToMeshTransferPolygonTest, ReorderedGeometryIdsKeepTrianglesOnCorrectAngularSides)
{
    using Factory = SimpleFluid::MeshReorderingFactory<Pack>;
    auto polygon = polygon_mesh({{0.5, 0.5, 0}, {1.5, 0.5, 0}, {1.5, 1.5, 0}, {0.5, 1.5, 0}},
                                {{0, 1, 2}, {0, 2, 3}});
    auto annulus = quarter_annulus(0.1, 3.0, true);
    auto selected = [](Pack::global_ordinal_type gid, const Handle::Vec3&) { return gid == 1; };
    // Compact composite meshes deliberately reject arbitrary reordering.
    // Exercise geometry-ID remapping through the supported direct extrusion.
    auto composite = polygon_mesh({{0.5, 0.5, 0}, {1.5, 0.5, 0}, {1.5, 1.5, 0}, {0.5, 1.5, 0}},
                                  {{0, 1, 2}, {0, 2, 3}}, true);
    EXPECT_THROW((void)Factory::selected_cells_first(std::move(composite), selected), std::invalid_argument);
    const auto source_layout = Factory::selected_cells_first(std::move(polygon), selected);
    const auto target_layout = Factory::selected_cells_first(std::move(annulus), selected);
    ASSERT_TRUE(source_layout.mesh->has_reordered_cells());
    ASSERT_TRUE(target_layout.mesh->has_reordered_cells());
    ScalarField source(source_layout.mesh, "source"), target(target_layout.mesh, "target");
    for (std::size_t c = 0; c < source.num_owned_cells(); ++c)
    {
        const auto lid = static_cast<LO>(c);
        source.set_owned_value(lid, source.mesh().cell_geometry_global_id(lid) == 0 ? 2.0 : 6.0);
    }
    Transfer transfer(source_layout.mesh, target_layout.mesh, partial_polygon_options());
    const auto report = transfer.project(source, target, Quantity::Intensive);
    expect_polygon_report(report, 4.0, 4.0);
    for (std::size_t c = 0; c < target.num_owned_cells(); ++c)
    {
        const auto lid = static_cast<LO>(c);
        const double inventory = target.mesh().cell_geometry_global_id(lid) == 0 ? 1.0 : 3.0;
        EXPECT_NEAR(target.value(lid), inventory / target.mesh().cell_volume(lid), 2.0e-13);
    }
}

TEST(MeshToMeshTransferPolygonTest, CurvedCompositeHexahedraAreNotPolygonalPrisms)
{
    auto native = std::make_shared<Handle::Cylindrical>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.5, 1.0}, {0.0, std::numbers::pi / 2.0}, {0.0, 1.0}}});
    auto composite = std::make_shared<Handle::MultiRegion>(
        std::vector<Handle::MultiRegion::Region>{Meshes::native_region("curved", native)},
        std::vector<Handle::MultiRegion::Interface>{});
    auto curved = std::make_shared<Handle>(composite);
    auto annulus = quarter_annulus();
    ASSERT_EQ(composite->cell_type(0), SimpleFluid::MeshUtils::CellType::HEXAHEDRON);
    EXPECT_THROW((Transfer(curved, annulus, partial_polygon_options())), std::invalid_argument);
    EXPECT_THROW((Transfer(annulus, curved, partial_polygon_options())), std::invalid_argument);
}

TEST(MeshToMeshTransferPolygonTest, TangentMidpointHasZeroOverlap)
{
    auto polygon = polygon_mesh({{-1, 1, 0}, {1, 1, 0}, {1, 2, 0}, {-1, 2, 0}}, {{0, 1, 2, 3}});
    auto annulus = std::make_shared<Handle>(std::make_shared<Handle::Cylindrical>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.1, 1}, {0, std::numbers::pi}, {0, 1}}}));
    for (const bool reverse : {false, true})
    {
        auto source = reverse ? polygon : annulus;
        auto target = reverse ? annulus : polygon;
        Transfer transfer(source, target, partial_polygon_options());
        EXPECT_NEAR(transfer.coverage().overlap_volume, 0, 1e-14);
        ScalarField density(source, "density"), result(target, "result");
        density.put_scalar(2);
        const auto report = transfer.project(density, result, Quantity::Intensive);
        expect_polygon_report(report, 2 * source->cell_volume(0), 0);
    }
}

TEST(MeshToMeshTransferPolygonTest, RepeatedCoordinateWindingIsRejected)
{
    auto annulus = quarter_annulus();
    EXPECT_THROW(([&] {
        // Distinct node IDs tracing the same triangle twice must not create
        // twice the physical overlap simply because its native area is doubled.
        auto polygon = polygon_mesh({{0, 0, 0}, {2, 0, 0}, {0, 2, 0},
                                     {0, 0, 0}, {2, 0, 0}, {0, 2, 0}}, {{0, 1, 2, 3, 4, 5}});
        Transfer transfer(polygon, annulus, partial_polygon_options());
    }()), std::invalid_argument);
}
