/** @file testPolyhedralAMR.cc
 * @brief Variable convex families, exact tiled neighbors and bounded atomic publication.
 */
#include "geometry/MeshHandle.hh"
#include "geometry/unitTests/polyhedral_amr_helpers.hh"
#include "parallel/MeshPartitioner.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <set>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

} // namespace

TEST(PolyhedralAMRTest, VariableOccupiedOctantsRestoreCanonicalParentAndPhysicalMetadata)
{
    ::SimpleFluid::Meshes::AMR::PolyhedralAMR parent(test::polyhedral_grid(test::polyhedral_tetrahedron()));
    const auto key = parent.cell_key(0);
    const auto observation = parent.observation();
    const auto fine = parent.adapt({0});
    EXPECT_EQ(fine->geometry()->num_cells(), 4u);
    std::set<std::uint64_t> octants;
    double volume = 0;
    for (std::uint64_t c = 0; c < fine->geometry()->num_cells(); ++c)
    {
        const auto child = fine->cell_key(c);
        octants.insert(std::get<::SimpleFluid::Meshes::AMR::ConvexCellAncestry>(child.ancestry).path.back());
        EXPECT_EQ(child.region, key.region);
        EXPECT_EQ(fine->cell_material(c), 17u);
        volume += fine->cell_geometry(c)->moments().volume.as_double();
    }
    EXPECT_EQ(octants, (std::set<std::uint64_t>{0, 1, 2, 4}));
    EXPECT_NEAR(volume, 1.0 / 6.0, 1e-15);
    const auto restored = fine->adapt({}, {key});
    EXPECT_EQ(restored->cell_key(0), key);
    EXPECT_TRUE(restored->cell_geometry(0)->equivalent(*parent.cell_geometry(0)));
    EXPECT_EQ(restored->geometry()->configuration_signature(), parent.geometry()->configuration_signature());
    EXPECT_EQ(parent.observation(), observation);
}

TEST(PolyhedralAMRTest, UnequalNeighborCutsRetainCanonicalTilesWhenOneParentCoarsens)
{
    ::SimpleFluid::Meshes::AMR::PolyhedralAMR parent(test::polyhedral_grid());
    const auto both = parent.adapt({0, 1});
    EXPECT_EQ(both->geometry()->num_cells(), 16u);
    std::size_t seam_tiles = 0;
    for (const auto face : both->complex()->face_ids())
    {
        const auto neighbor = both->complex()->neighbor_cell(face);
        if (neighbor == std::numeric_limits<std::uint64_t>::max())
            continue;
        const auto owner = both->complex()->owner_cell(face);
        if (std::get<::SimpleFluid::Meshes::AMR::ConvexCellAncestry>(both->cell_key(owner).ancestry).root !=
            std::get<::SimpleFluid::Meshes::AMR::ConvexCellAncestry>(both->cell_key(neighbor).ancestry).root)
            ++seam_tiles;
    }
    EXPECT_EQ(seam_tiles, 6u); // independent interface cuts at y=3/8 and y=3/4, z=1/2
    const auto restored = both->adapt({}, {parent.cell_key(0)});
    EXPECT_EQ(restored->geometry()->num_cells(), 9u);
    EXPECT_TRUE(restored->cell_geometry(0)->equivalent(*parent.cell_geometry(0)));
    EXPECT_GT(restored->complex()->cell_faces(0).size(), 6u);
    const auto& provider = std::get<DistributedExplicitRegion>(restored->geometry()->regions().front()).topology();
    EXPECT_EQ(provider.cell_type(0), MeshUtils::CellType::POLYHEDRON);
    for (std::uint64_t c = 0; c < restored->geometry()->num_cells(); ++c)
        EXPECT_EQ(restored->cell_material(c), c == 0 ? 17u : 23u);
}

TEST(PolyhedralAMRTest, SourceFamilyEligibilityBudgetsAndPlansRejectWithoutChangingSource)
{
    const auto grid = test::polyhedral_grid();
    ::SimpleFluid::Meshes::AMR::PolyhedralAMR parent(grid);
    const auto key = parent.cell_key(0);
    const auto observation = parent.observation();
    const auto planned = parent.plan_adaptation({0});
    const auto fine = parent.apply_adaptation(planned);
    EXPECT_THROW((void) fine->apply_adaptation(planned), std::invalid_argument);
    EXPECT_EQ(parent.adapt()->observation(), observation);
    EXPECT_THROW((void) fine->adapt({0}, {key}), std::invalid_argument);
    const auto grandchild = fine->cell_key(0);
    const auto deeper = fine->adapt({0});
    EXPECT_THROW((void) deeper->adapt({}, {key, grandchild}), std::invalid_argument);
    auto limited = grid;
    limited.maximum_cells = 2;
    ::SimpleFluid::Meshes::AMR::PolyhedralAMR budget(limited);
    const auto before = budget.observation();
    EXPECT_THROW((void) budget.adapt({0}), std::invalid_argument);
    EXPECT_EQ(budget.observation(), before);
    auto faces = grid;
    faces.maximum_face_tiles = grid.mesh->num_faces();
    ::SimpleFluid::Meshes::AMR::PolyhedralAMR face_budget(faces);
    EXPECT_THROW((void) face_budget.adapt({0, 1}), std::invalid_argument);
    EXPECT_EQ(parent.observation(), observation);
    std::vector<std::size_t> bad;
    if (Tpetra::getDefaultComm()->getRank() == 0)
        bad = {999};
    EXPECT_THROW((void) parent.adapt(bad), std::exception);
    EXPECT_EQ(parent.observation(), observation);
}

TEST(PolyhedralAMRTest, SameTopologyDifferentRootCoordinatesRejectCollectively)
{
    EXPECT_NO_THROW((void) ::SimpleFluid::Meshes::AMR::PolyhedralAMR(test::polyhedral_grid()));
    if (Tpetra::getDefaultComm()->getSize() < 2)
        GTEST_SKIP() << "Divergent roots require MPI.";
    const double shift = Tpetra::getDefaultComm()->getRank() == 0 ? 0.125 : 0;
    EXPECT_THROW((void) ::SimpleFluid::Meshes::AMR::PolyhedralAMR(test::polyhedral_grid(test::polyhedral_pair(shift))),
        std::invalid_argument);
}

TEST(PolyhedralAMRTest, ActualPartitionPinsResidentIdentityAndMixedRegionsReject)
{
    ::SimpleFluid::Meshes::AMR::PolyhedralAMR parent(test::polyhedral_grid());
    CompositePartitionOptions options;
    options.target_cells_per_fragment = 1;
    const auto partition = std::make_shared<const CompositePartition<>>(MeshPartitioner<DefaultTpetraTypes>::partition(
        CompositeMeshSource(parent.geometry()), options, Tpetra::getDefaultComm()));
    const auto bound = parent.with_partition(partition);
    EXPECT_EQ(bound->observation().topology_identity, parent.observation().topology_identity);
    EXPECT_NE(bound->observation().partition_identity, parent.observation().partition_identity);
    EXPECT_EQ(bound->observation().native_geometry, partition->geometry().get());
    EXPECT_EQ(bound->with_partition(partition)->observation(), bound->observation());
    auto mixed = test::polyhedral_grid();
    ++mixed.physical_regions[1];
    EXPECT_THROW((void) ::SimpleFluid::Meshes::AMR::PolyhedralAMR(mixed), std::invalid_argument);
}

TEST(PolyhedralAMRTest, SliverCandidateRejectsWithoutPublishingAnyChildren)
{
    const auto cube =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box({0, 0, 0}, {1, 1, 1});
    const auto root = cube->clip(SimpleFluid::PLIC::GeometryHalfspace{{1, 1, 1e-8}, {0, 0, 0}, 1 + 1e-8}).inside;
    ASSERT_TRUE(root);
    const ::SimpleFluid::Meshes::AMR::AdaptiveCellKey key{19, ::SimpleFluid::Meshes::AMR::ConvexCellAncestry{0, {}}};
    std::vector<SimpleFluid::PLIC::PolyhedralSurfaceUse> surfaces;
    for (std::uint64_t f = 0; f < root->num_faces(); ++f)
        surfaces.push_back({SimpleFluid::PLIC::PolyhedralSurfaceKey{SimpleFluid::PLIC::PolyhedralSurfaceKey::Kind::Root,
                                19, ::SimpleFluid::Meshes::AMR::ConvexCellAncestry{f, {}}, 0},
            1, true, "wall"});
    const auto complex = SimpleFluid::PLIC::PolyhedralComplex<SimpleFluid::ArithmeticMode::Ordinary>::make(
        {SimpleFluid::PLIC::PolyhedralComplexCell<SimpleFluid::ArithmeticMode::Ordinary>{key, 0, root, surfaces}});
    const auto provider = ExplicitRegionProvider::from_complex(complex);
    const auto native = std::make_shared<MultiRegionMesh>(
        std::vector<MultiRegionMesh::Region>{DistributedExplicitRegion("cut", provider, provider)},
        std::vector<MultiRegionMesh::Interface>{}, Teuchos::rcp(new Teuchos::SerialComm<int>()));
    auto grid = test::polyhedral_grid(native);
    grid.minimum_scaled_volume = 1e-6;
    ::SimpleFluid::Meshes::AMR::PolyhedralAMR parent(grid);
    const auto before = parent.observation();
    EXPECT_THROW((void) parent.adapt({0}), std::invalid_argument);
    EXPECT_EQ(parent.observation(), before);
    EXPECT_EQ(parent.geometry()->num_cells(), 1u);
}
