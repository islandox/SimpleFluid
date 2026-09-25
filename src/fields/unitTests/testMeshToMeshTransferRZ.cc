/** @file testMeshToMeshTransferRZ.cc
 * @brief Checks conservative transfer for swept RZ and moving composite meshes.
 */
#include <gtest/gtest.h>

#include "fields/MeshToMeshTransfer.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/mesh/SweptRZRegionProviders.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

namespace
{
using Pack = SimpleFluid::DefaultTpetraTypes;
using Handle = SimpleFluid::MeshHandle<Pack>;
using Composite = Handle::MultiRegion;
using Transfer = SimpleFluid::MeshToMeshTransfer<Pack>;
using ScalarField = SimpleFluid::ScalarCellFieldStored<Pack>;
using Quantity = SimpleFluid::MeshToMeshQuantity;
namespace Meshes = SimpleFluid::Meshes;
testing::Environment* const rz_environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

SimpleFluid::MeshToMeshTransferOptions partial_options()
{
    SimpleFluid::MeshToMeshTransferOptions options;
    options.method = SimpleFluid::MeshToMeshTransferMethod::ConservativeCellAverage;
    options.coverage_mode = SimpleFluid::MeshToMeshCoverageMode::AllowPartial;
    return options;
}

SimpleFluid::SP<Handle> cylinder(double top = 1.0)
{
    auto native = std::make_shared<Handle::Cylindrical>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
        {0.5, 2.5}, {0.0, 0.2, 0.4}, {0.0, top / 2, top}}});
    Handle::DistributionOptions distribution;
    distribution.partition = Tpetra::getDefaultComm()->getRank();
    distribution.partitions = Tpetra::getDefaultComm()->getSize();
    distribution.allow_empty_partitions = true;
    return std::make_shared<Handle>(native, distribution);
}

Meshes::SweptRZRegion swept_region(bool sloped = false)
{
    auto topology = std::make_shared<const Meshes::ExtrudedTopology>(4,
        SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>{{0, 1, 2, 3}}, 2,
        SimpleFluid::Arr<Meshes::SemiStructMeshTopo::BoundaryEdge>{
            {0, 1, "inner"}, {1, 2, "join"}, {2, 3, "outer"}, {3, 0, "bottom"}});
    return Meshes::swept_rz_region("corner", topology,
        {{1, 0, 0}, {1, 1, 0}, {2, sloped ? 0.6 : 1.0, 0}, {2, 0, 0}}, {0, 0.2, 0.4});
}

SimpleFluid::SP<Handle> swept(bool sloped = false)
{
    auto composite = std::make_shared<Composite>(std::vector<Composite::Region>{swept_region(sloped)},
        std::vector<Composite::Interface>{});
    return std::make_shared<Handle>(composite);
}

double total_volume(const Handle& mesh)
{
    double local = 0, global = 0;
    for (size_t c = 0; c < mesh.num_owned_cells(); ++c) local += mesh.cell_volume(c);
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
    return global;
}

void check_report(const SimpleFluid::MeshToMeshConservationReport& report, double source, double transferred,
                  size_t component = 0)
{
    const double tolerance = 3e-12 * std::max(1.0, std::abs(source));
    EXPECT_NEAR(report.source_integral[component], source, tolerance);
    EXPECT_NEAR(report.transferred_integral[component], transferred, tolerance);
    EXPECT_NEAR(report.target_integral[component], transferred, tolerance);
    EXPECT_NEAR(report.uncovered_source_integral[component], source - transferred, tolerance);
    EXPECT_NEAR(report.transfer_error[component], 0, tolerance);
    EXPECT_NEAR(report.conservation_error[component], 0, tolerance);
}
} // namespace

TEST(MeshToMeshTransferRZTest, NativeRectangleTransfersSignedMulticomponentFieldsBothDirections)
{
    const auto corner = swept();
    const auto annulus = cylinder();
    const double overlap = 3 * std::sin(0.2); // Two swept rectangles, integral rho d rho dz = 3/2 each.
    EXPECT_NEAR(total_volume(*corner), overlap, 2e-14);
    const std::array<double, 3> densities{2, -3, 0.25};
    for (const bool reverse : {false, true})
    {
        const auto source_mesh = reverse ? corner : annulus;
        const auto target_mesh = reverse ? annulus : corner;
        Transfer transfer(source_mesh, target_mesh, partial_options());
        EXPECT_NEAR(transfer.coverage().overlap_volume, overlap, 3e-13);
        for (const auto quantity : {Quantity::Intensive, Quantity::Extensive})
        {
            Pack::multi_vector_type source(source_mesh->owned_cell_map(), densities.size());
            Pack::multi_vector_type target(target_mesh->owned_cell_map(), densities.size());
            {
                const auto data = source.getLocalViewHost(Tpetra::Access::OverwriteAll);
                for (size_t c = 0; c < source_mesh->num_owned_cells(); ++c)
                    for (size_t j = 0; j < densities.size(); ++j)
                        data(c, j) = densities[j] * (quantity == Quantity::Extensive ? source_mesh->cell_volume(c) : 1);
            }
            const auto report = transfer.project(source, target, quantity);
            const auto data = target.getLocalViewHost(Tpetra::Access::ReadOnly);
            for (size_t j = 0; j < densities.size(); ++j)
            {
                check_report(report, densities[j] * total_volume(*source_mesh), densities[j] * overlap, j);
                for (size_t c = 0; c < target_mesh->num_owned_cells(); ++c)
                {
                    const double fraction = reverse ? 2.5 * std::sin(0.2) : 1.0;
                    const double expected = densities[j] * fraction *
                        (quantity == Quantity::Extensive ? target_mesh->cell_volume(c) : 1);
                    EXPECT_NEAR(data(c, j), expected, 3e-12);
                }
            }
        }
    }
}

TEST(MeshToMeshTransferRZTest, SlopingCornerUsesCurrentAffineZAndRejectsStaleTransfer)
{
    auto corner = swept(true);
    const auto annulus = cylinder(2);
    const double native = (7.0 / 3.0) * std::sin(0.2);
    EXPECT_NEAR(total_volume(*corner), native, 2e-14);
    ScalarField source(corner, "source"), target(annulus, "target");
    source.put_scalar(2);
    Transfer initial(corner, annulus, partial_options());
    check_report(initial.project(source, target, Quantity::Intensive), 2 * native, 2 * native);
    SimpleFluid::PlanarALEMeshMotion<Pack> motion(corner);
    motion.begin_trial(1.1, 0.1);
    EXPECT_THROW(initial.coverage(), std::runtime_error);
    Transfer expanded(corner, annulus, partial_options());
    EXPECT_NEAR(expanded.coverage().overlap_volume, 1.1 * native, 3e-13);
    check_report(expanded.project(source, target, Quantity::Intensive), 2.2 * native, 2.2 * native);
    motion.rollback_trial();
    EXPECT_THROW(expanded.coverage(), std::runtime_error);
    Transfer restored(corner, annulus, partial_options());
    EXPECT_NEAR(restored.coverage().overlap_volume, native, 3e-13);
}

TEST(MeshToMeshTransferRZTest, CompositeCanMixXYPrismsAndAngularRZCells)
{
    auto rz = swept_region();
    SimpleFluid::Arr<Handle::Vec3> points;
    for (const double radius : {1.0, 2.0})
        for (const double theta : {0.0, 0.2, 0.4})
            points.push_back({radius * std::cos(theta), radius * std::sin(theta), 0});
    auto xy = std::make_shared<Handle::SemiStructured>(points,
        SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>{{0, 3, 4, 1}, {1, 4, 5, 2}}, SimpleFluid::ArrReal{1, 2});
    Meshes::ExplicitConformingInterface interface;
    interface.first_region = 0;
    interface.second_region = 1;
    interface.second_boundary = xy->boundary_id({0, 0, Handle::SemiStructured::Z_FACE});
    for (size_t f = 0; f < rz.layout().faces; ++f)
    {
        const int boundary = rz.topology().boundary_id(f);
        if (boundary < 0 || rz.topology().boundary_batch_name(boundary) != "join") continue;
        interface.first_boundary = boundary;
        const auto face = rz.topology().indexer().face_id(f);
        interface.faces.emplace_back(f, xy->indexer().face_ordinal({face.k, 0, Handle::SemiStructured::Z_FACE}));
    }
    ASSERT_EQ(interface.faces.size(), 2U);
    auto composite = std::make_shared<Composite>(
        std::vector<Composite::Region>{rz, Meshes::native_region("upper", xy)},
        std::vector<Composite::Interface>{interface});
    auto mixed = std::make_shared<Handle>(composite);
    const auto annulus = cylinder(2);
    const double native = 6 * std::sin(0.2);
    EXPECT_NEAR(total_volume(*mixed), native, 3e-14);
    for (const bool reverse : {false, true})
    {
        auto source_mesh = reverse ? annulus : mixed;
        auto target_mesh = reverse ? mixed : annulus;
        ScalarField source(source_mesh, "source"), target(target_mesh, "target");
        source.put_scalar(1);
        Transfer transfer(source_mesh, target_mesh, partial_options());
        EXPECT_NEAR(transfer.coverage().overlap_volume, native, 4e-13);
        check_report(transfer.project(source, target, Quantity::Intensive), total_volume(*source_mesh), native);
        for (size_t c = 0; c < target_mesh->num_owned_cells(); ++c)
            EXPECT_NEAR(target.value(c), reverse ? 1 : 2.5 * std::sin(0.2), 3e-12);
    }
}
