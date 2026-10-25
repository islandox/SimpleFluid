/** @file testExactExplicitProvider.cc
 * @brief Ordinary native explicit packets, metric-only neighbors and record compatibility.
 */
#include "geometry/unitTests/polyhedral_amr_helpers.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_DefaultSerialComm.hpp>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <set>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
} // namespace
TEST(ExactExplicitProviderTest, SparsePacketRetainsOnlyVisibleCertificatesAndMetricNeighbors)
{
    ::SimpleFluid::Meshes::AMR::PolyhedralAMR parent(test::polyhedral_grid());
    const auto fine = parent.adapt({0, 1})->adapt({}, {parent.cell_key(0)});
    const std::vector<std::uint64_t> visible{0};
    const auto packet = fine->geometry()->serialize_partition(visible);
    const auto restored = MultiRegionMesh::deserialize_partition(packet, Tpetra::getDefaultComm());
    const auto& provider = std::get<DistributedExplicitRegion>(restored->regions().front()).topology();
    ASSERT_TRUE(provider.has_construction());
    // Native packet support includes owners incident to requested outer faces;
    // this is geometric support closure, not an expanded field halo. The coarse
    // left parent touches the four low-X octants of the right root (0,2,4,6).
    const std::vector<std::uint64_t> support{0, 1, 3, 5, 7};
    EXPECT_EQ(provider.construction()->resident_cell_ids(), support);
    EXPECT_LT(support.size(), provider.layout().cells);
    EXPECT_EQ(provider.layout().cells, fine->geometry()->num_cells());
    EXPECT_LT(provider.face_records().size(), provider.layout().faces);
    EXPECT_LT(provider.node_records().size(), provider.layout().nodes);
    std::size_t metrics = 0;
    std::set<std::uint64_t> metric_ids;
    for (const auto& [id, cell] : provider.cell_records())
        if (cell.faces.empty())
        {
            ++metrics;
            metric_ids.insert(id);
            EXPECT_TRUE(cell.nodes.empty());
            EXPECT_GT(cell.volume, 0);
            EXPECT_THROW((void) provider.certified_cell_geometry(id), std::invalid_argument);
            EXPECT_THROW((void) provider.atomic_cell_geometry(id), std::invalid_argument);
        }
    EXPECT_GT(metrics, 0u);
    EXPECT_EQ(metric_ids, (std::set<std::uint64_t>{2, 4, 6, 8}));
    for (const auto face : provider.construction()->face_ids())
    {
        EXPECT_EQ(provider.face_area(face), provider.construction()->face_area(face));
        EXPECT_EQ(restored->face_area(face), provider.construction()->face_area(face));
    }
    EXPECT_TRUE(provider.certified_cell_geometry(0)->equivalent(*fine->cell_geometry(0)));
    EXPECT_EQ(provider.atomic_cell_geometry(0)->num_faces(), provider.cell_faces(0).size());
    const auto empty =
        MultiRegionMesh::deserialize_partition(fine->geometry()->serialize_partition({}), Tpetra::getDefaultComm());
    const auto& empty_provider = std::get<DistributedExplicitRegion>(empty->regions().front()).topology();
    EXPECT_TRUE(empty_provider.cell_records().empty());
    EXPECT_TRUE(empty_provider.face_records().empty());
    EXPECT_TRUE(empty_provider.node_records().empty());
    EXPECT_TRUE(empty_provider.construction()->resident_cell_ids().empty());
    EXPECT_EQ(empty_provider.layout().cells, provider.layout().cells);
    EXPECT_EQ(empty_provider.boundary_names(), provider.boundary_names());
    EXPECT_THROW((void) ExplicitRegionProvider::from_complex(empty_provider.construction()), std::invalid_argument);
    const auto factory = ExplicitRegionProvider::from_complex(empty_provider.construction(), provider.boundary_names());
    EXPECT_EQ(factory->boundary_names(), provider.boundary_names());
    EXPECT_TRUE(factory->cell_records().empty());
}
TEST(ExactExplicitProviderTest, ExactRecordTamperingRejectsWhileLegacyConstructorRemainsAvailable)
{
    ::SimpleFluid::Meshes::AMR::PolyhedralAMR hierarchy(test::polyhedral_grid());
    const auto& source = std::get<DistributedExplicitRegion>(hierarchy.geometry()->regions().front()).topology();
    std::vector<ExplicitRegionProvider::Cell> cells;
    std::vector<ExplicitRegionProvider::Face> faces;
    std::vector<ExplicitRegionProvider::Node> nodes;
    for (const auto& [id, c] : source.cell_records())
        cells.push_back(c);
    for (const auto& [id, f] : source.face_records())
        faces.push_back(f);
    for (const auto& [id, p] : source.node_records())
        nodes.push_back({id, p});
    EXPECT_NO_THROW((void) ExplicitRegionProvider(source.layout(), cells, faces, nodes, source.boundary_names()));
    EXPECT_NO_THROW((void) ExplicitRegionProvider(
        source.layout(), cells, faces, nodes, source.boundary_names(), source.construction()));
    cells[0].volume = std::nextafter(cells[0].volume, std::numeric_limits<double>::infinity());
    EXPECT_THROW((void) ExplicitRegionProvider(
                     source.layout(), cells, faces, nodes, source.boundary_names(), source.construction()),
        std::invalid_argument);
    cells[0].volume = source.cell_volume(0);
    nodes[0].point.x += 0.125;
    EXPECT_THROW((void) ExplicitRegionProvider(
                     source.layout(), cells, faces, nodes, source.boundary_names(), source.construction()),
        std::invalid_argument);
}
TEST(ExactExplicitProviderTest, ExactReassemblyRejectsAndLegacySparseMergeStillWorks)
{
    ::SimpleFluid::Meshes::AMR::PolyhedralAMR hierarchy(test::polyhedral_grid());
    const std::vector<std::uint64_t> all{0, 1};
    const auto exact = MultiRegionMesh::deserialize_partition(
        hierarchy.geometry()->serialize_partition(all), Tpetra::getDefaultComm());
    EXPECT_THROW((void) MultiRegionMesh::merge_partitions({exact}, Teuchos::rcp(new Teuchos::SerialComm<int>())),
        std::invalid_argument);
    const auto legacy_source = test::polyhedral_pair();
    const auto legacy =
        MultiRegionMesh::deserialize_partition(legacy_source->serialize_partition(all), Tpetra::getDefaultComm());
    const auto merged = MultiRegionMesh::merge_partitions({legacy}, Teuchos::rcp(new Teuchos::SerialComm<int>()));
    EXPECT_FALSE(merged->has_construction());
    EXPECT_EQ(merged->num_cells(), 2u);
    auto old = hierarchy.geometry()->serialize_partition(all);
    const std::uint64_t old_magic = 0x5346524547490004ULL;
    std::memcpy(old.data(), &old_magic, sizeof(old_magic));
    EXPECT_THROW((void) MultiRegionMesh::deserialize_partition(old, Tpetra::getDefaultComm()), std::invalid_argument);
}

TEST(ExactExplicitProviderTest, UnlabelledExteriorRetainsAnExplicitEmptyGlobalSchema)
{
    ::SimpleFluid::Meshes::AMR::PolyhedralAMR parent(test::polyhedral_grid(test::polyhedral_pair(0, false)));
    const auto fine = parent.adapt({0, 1});
    const std::vector<std::uint64_t> visible{0};
    const auto restored = MultiRegionMesh::deserialize_partition(
        fine->geometry()->serialize_partition(visible), Tpetra::getDefaultComm());
    const auto& provider = std::get<DistributedExplicitRegion>(restored->regions().front()).topology();
    EXPECT_TRUE(provider.boundary_names().empty());
    for (const auto face : provider.construction()->face_ids())
        if (provider.construction()->face_surface(face).exterior)
        {
            EXPECT_EQ(provider.boundary_id(face), UnstructuredMesh::invalid_boundary_id);
            EXPECT_TRUE(provider.construction()->face_surface(face).boundary.empty());
        }
    const auto empty = SimpleFluid::PLIC::PolyhedralComplex<SimpleFluid::ArithmeticMode::Ordinary>::decode(
        fine->complex()->encode(std::span<const std::uint64_t>{}));
    EXPECT_THROW((void) ExplicitRegionProvider::from_complex(empty), std::invalid_argument);
    EXPECT_NO_THROW((void) ExplicitRegionProvider::from_complex(empty, std::map<int, std::string>{}));
}
