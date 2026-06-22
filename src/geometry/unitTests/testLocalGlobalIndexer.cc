/**
 * @file testLocalGlobalIndexer.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Tests for partition-local to mesh-global indexing.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "geometry/mesh/LocalGlobalIndexer.hh"
#include "geometry/mesh/LocalGlobalIndexer.tcc"
#include "geometry/mesh/OrthogonalIndexer.hh"
#include "geometry/mesh/OrthogonalLocalGlobalIndexer.tcc"

#include <compare>
#include <type_traits>

namespace
{

using IndexTypes = SimpleFluid::MeshIndexTypes<
    long long,
    long long,
    long long,
    int,
    long long>;
using Indexer = SimpleFluid::Meshes::LocalGlobalIndexer<IndexTypes>;
using OrthogonalIndexer = SimpleFluid::Meshes::OrthogonalIndexer;
using TypedIndexTypes = SimpleFluid::MeshIndexTypes<
    OrthogonalIndexer::CellID,
    OrthogonalIndexer::FaceID,
    OrthogonalIndexer::NodeID,
    int,
    long long>;
using TypedIndexer =
    SimpleFluid::Meshes::LocalGlobalIndexer<TypedIndexTypes>;
using OrthogonalBlockIndexTypes =
    SimpleFluid::Meshes::OrthogonalMeshIndexTypePack<int, long long>;
using OrthogonalBlockIndexer =
    SimpleFluid::Meshes::LocalGlobalIndexer<OrthogonalBlockIndexTypes>;

template<class Tag>
struct StrongID
{
    int value = 0;

    constexpr auto operator<=>(const StrongID&) const = default;
};

struct LocalCellTag;
struct LocalFaceTag;
struct LocalNodeTag;
struct GlobalCellTag;
struct GlobalFaceTag;
struct GlobalNodeTag;

using LocalCellID = StrongID<LocalCellTag>;
using LocalFaceID = StrongID<LocalFaceTag>;
using LocalNodeID = StrongID<LocalNodeTag>;
using GlobalCellID = StrongID<GlobalCellTag>;
using GlobalFaceID = StrongID<GlobalFaceTag>;
using GlobalNodeID = StrongID<GlobalNodeTag>;
using DistinctIndexTypes = SimpleFluid::MeshIndexTypes<
    GlobalCellID,
    GlobalFaceID,
    GlobalNodeID,
    int,
    long long,
    LocalCellID,
    LocalFaceID,
    LocalNodeID>;
using DistinctIndexer =
    SimpleFluid::Meshes::LocalGlobalIndexer<DistinctIndexTypes>;

static_assert(SimpleFluid::MeshIndexTypePack<
              typename Indexer::index_type_pack>);
static_assert(SimpleFluid::MeshIndexer<Indexer>);
static_assert(SimpleFluid::MeshIndexer<TypedIndexer>);
static_assert(SimpleFluid::MeshIndexer<DistinctIndexer>);
static_assert(SimpleFluid::MeshIndexer<OrthogonalBlockIndexer>);
static_assert(std::is_same_v<
              typename Indexer::global_ordinal_type,
              long long>);

} // namespace

TEST(LocalGlobalIndexerTest, MapsOwnedAndOverlapEntitiesBidirectionally)
{
    const Indexer indexer(
        {10, 12},
        {11},
        {20, 22, 24},
        {21, 23},
        {30, 31, 32, 33});

    EXPECT_EQ(indexer.num_owned_cells(), 2U);
    EXPECT_EQ(indexer.num_local_cells(), 3U);
    EXPECT_EQ(indexer.cell_global_id(0), 10);
    EXPECT_EQ(indexer.cell_global_id(2), 11);
    EXPECT_EQ(indexer.cell_id(1), 12);
    EXPECT_EQ(indexer.cell_ordinal(11), 2);
    EXPECT_EQ(indexer.cell_local_id(12), 1);
    EXPECT_EQ(indexer.cell_local_id(11), 2);
    EXPECT_EQ(indexer.local_to_global_cell(1), 12);
    EXPECT_EQ(indexer.global_to_local_cell(10), 0);
    EXPECT_EQ(indexer.local_to_global_cell_id(1), 12);
    EXPECT_EQ(indexer.global_to_local_cell_id(11), 2);
    EXPECT_EQ(indexer.local_to_global_cell_ordinal(1), 12);
    EXPECT_EQ(indexer.global_to_local_cell_ordinal(11), 2);
    EXPECT_TRUE(indexer.is_owned_cell(1));
    EXPECT_FALSE(indexer.is_owned_cell(2));

    EXPECT_EQ(indexer.num_owned_faces(), 3U);
    EXPECT_EQ(indexer.num_local_faces(), 5U);
    EXPECT_EQ(indexer.face_global_id(3), 21);
    EXPECT_EQ(indexer.face_local_id(23), 4);
    EXPECT_EQ(indexer.face_id(4), 23);
    EXPECT_EQ(indexer.face_ordinal(20), 0);
    EXPECT_EQ(indexer.local_to_global_face(2), 24);
    EXPECT_EQ(indexer.global_to_local_face(22), 1);
    EXPECT_EQ(indexer.local_to_global_face_id(2), 24);
    EXPECT_EQ(indexer.global_to_local_face_id(21), 3);
    EXPECT_EQ(indexer.local_to_global_face_ordinal(4), 23);
    EXPECT_EQ(indexer.global_to_local_face_ordinal(24), 2);
    EXPECT_EQ(indexer.node_global_id(2), 32);
    EXPECT_EQ(indexer.node_local_id(33), 3);
    EXPECT_EQ(indexer.node_id(0), 30);
    EXPECT_EQ(indexer.node_ordinal(32), 2);
    EXPECT_EQ(indexer.local_to_global_node(1), 31);
    EXPECT_EQ(indexer.global_to_local_node(30), 0);
    EXPECT_EQ(indexer.local_to_global_node_id(3), 33);
    EXPECT_EQ(indexer.global_to_local_node_id(32), 2);
    EXPECT_EQ(indexer.local_to_global_node_ordinal(0), 30);
    EXPECT_EQ(indexer.global_to_local_node_ordinal(31), 1);
    EXPECT_EQ(indexer.num_owned_nodes(), 4U);
    EXPECT_TRUE(indexer.is_owned_node(3));
}

TEST(LocalGlobalIndexerTest, ReportsMissingAndInvalidIdentifiers)
{
    const Indexer indexer({1}, {2}, {3}, {}, {4});

    EXPECT_EQ(indexer.cell_local_id(99), Indexer::invalid_local_id());
    EXPECT_EQ(indexer.face_local_id(99), Indexer::invalid_local_id());
    EXPECT_EQ(indexer.node_local_id(99), Indexer::invalid_local_id());
    EXPECT_EQ(
        indexer.global_to_local_cell(99), Indexer::invalid_local_id());
    EXPECT_EQ(
        indexer.global_to_local_face(99), Indexer::invalid_local_id());
    EXPECT_EQ(
        indexer.global_to_local_node(99), Indexer::invalid_local_id());
    EXPECT_EQ(
        indexer.global_to_local_cell_ordinal(99),
        Indexer::invalid_local_id());
    EXPECT_THROW(indexer.local_to_global_cell_id(9), std::out_of_range);
    EXPECT_THROW(indexer.global_to_local_face_id(99), std::out_of_range);
    EXPECT_THROW(indexer.cell_global_id(-1), std::out_of_range);
    EXPECT_THROW(indexer.face_global_id(1), std::out_of_range);
    EXPECT_THROW(indexer.local_to_global_node(1), std::out_of_range);
}

TEST(LocalGlobalIndexerTest, RejectsDuplicateGlobalIdentifiers)
{
    EXPECT_THROW(
        Indexer({1}, {1}, {}, {}, {}),
        std::invalid_argument);
    EXPECT_THROW(
        Indexer({}, {}, {2, 2}, {}, {}),
        std::invalid_argument);
    EXPECT_THROW(
        Indexer({}, {}, {}, {}, {3, 3}),
        std::invalid_argument);
}

TEST(LocalGlobalIndexerTest, AcceptsDistinctCRTPMeshIdentifierTypes)
{
    const OrthogonalIndexer::CellID owned_cell{1, 2, 3};
    const OrthogonalIndexer::CellID ghost_cell{2, 2, 3};
    const OrthogonalIndexer::FaceID owned_face{
        2, 2, 3, OrthogonalIndexer::I_FACE};
    const OrthogonalIndexer::FaceID overlap_face{
        3, 2, 3, OrthogonalIndexer::I_FACE};
    const OrthogonalIndexer::NodeID node{2, 3, 4};
    const OrthogonalIndexer::NodeID overlap_node{3, 3, 4};
    const TypedIndexer indexer(
        std::vector<TypedIndexer::CellMapping>{
            {owned_cell, owned_cell, 10}},
        std::vector<TypedIndexer::CellMapping>{
            {ghost_cell, ghost_cell, 11}},
        std::vector<TypedIndexer::FaceMapping>{
            {owned_face, owned_face, 20}},
        std::vector<TypedIndexer::FaceMapping>{
            {overlap_face, overlap_face, 21}},
        std::vector<TypedIndexer::NodeMapping>{
            {node, node, 30}},
        std::vector<TypedIndexer::NodeMapping>{
            {overlap_node, overlap_node, 31}});

    EXPECT_EQ(indexer.cell_global_id(0), owned_cell);
    EXPECT_EQ(indexer.cell_local_id(ghost_cell), 1);
    EXPECT_EQ(indexer.cell_id(1), ghost_cell);
    EXPECT_EQ(indexer.cell_ordinal(owned_cell), 0);
    EXPECT_EQ(indexer.local_to_global_cell(1), ghost_cell);
    EXPECT_EQ(indexer.global_to_local_cell(owned_cell), 0);
    EXPECT_EQ(indexer.face_global_id(1), overlap_face);
    EXPECT_EQ(indexer.face_local_id(owned_face), 0);
    EXPECT_EQ(indexer.face_id(0), owned_face);
    EXPECT_EQ(indexer.face_ordinal(overlap_face), 1);
    EXPECT_EQ(indexer.local_to_global_face(0), owned_face);
    EXPECT_EQ(indexer.global_to_local_face(overlap_face), 1);
    EXPECT_EQ(indexer.node_global_id(0), node);
    EXPECT_EQ(indexer.node_local_id(node), 0);
    EXPECT_EQ(indexer.node_local_id(overlap_node), 1);
    EXPECT_EQ(indexer.node_id(1), overlap_node);
    EXPECT_EQ(indexer.node_ordinal(node), 0);
    EXPECT_EQ(indexer.local_to_global_node(1), overlap_node);
    EXPECT_EQ(indexer.global_to_local_node(node), 0);
    EXPECT_TRUE(indexer.is_owned_node(0));
    EXPECT_FALSE(indexer.is_owned_node(1));
}

TEST(LocalGlobalIndexerTest, MapsDistinctLocalAndGlobalIdsAndOrdinals)
{
    const DistinctIndexer indexer(
        std::vector<DistinctIndexer::CellMapping>{
            {{4}, {40}, 400}},
        std::vector<DistinctIndexer::CellMapping>{
            {{7}, {70}, 700}},
        std::vector<DistinctIndexer::FaceMapping>{
            {{2}, {20}, 200}},
        std::vector<DistinctIndexer::FaceMapping>{
            {{5}, {50}, 500}},
        std::vector<DistinctIndexer::NodeMapping>{
            {{1}, {10}, 100}},
        std::vector<DistinctIndexer::NodeMapping>{
            {{3}, {30}, 300}});

    EXPECT_EQ(indexer.local_to_global_cell_id({7}), (GlobalCellID{70}));
    EXPECT_EQ(indexer.global_to_local_cell_id({40}), (LocalCellID{4}));
    EXPECT_EQ(indexer.local_to_global_cell_ordinal(1), 700);
    EXPECT_EQ(indexer.global_to_local_cell_ordinal(400), 0);

    EXPECT_EQ(indexer.local_to_global_face_id({2}), (GlobalFaceID{20}));
    EXPECT_EQ(indexer.global_to_local_face_id({50}), (LocalFaceID{5}));
    EXPECT_EQ(indexer.local_to_global_face_ordinal(0), 200);
    EXPECT_EQ(indexer.global_to_local_face_ordinal(500), 1);

    EXPECT_EQ(indexer.local_to_global_node_id({3}), (GlobalNodeID{30}));
    EXPECT_EQ(indexer.global_to_local_node_id({10}), (LocalNodeID{1}));
    EXPECT_EQ(indexer.local_to_global_node_ordinal(1), 300);
    EXPECT_EQ(indexer.global_to_local_node_ordinal(100), 0);

    EXPECT_EQ(indexer.cell_id(0), (GlobalCellID{40}));
    EXPECT_EQ(indexer.cell_ordinal({70}), 1);
}

TEST(LocalGlobalIndexerTest, RejectsAmbiguousExplicitMappings)
{
    EXPECT_THROW(
        DistinctIndexer(
            std::vector<DistinctIndexer::CellMapping>{
                {{1}, {10}, 100},
                {{1}, {20}, 200}},
            {}, {}, {}, {}),
        std::invalid_argument);
    EXPECT_THROW(
        DistinctIndexer(
            std::vector<DistinctIndexer::CellMapping>{
                {{1}, {10}, 100},
                {{2}, {20}, 100}},
            {}, {}, {}, {}),
        std::invalid_argument);
}

TEST(OrthogonalLocalGlobalIndexerTest, DividesAllDimensionsIntoBalancedBlocks)
{
    const OrthogonalIndexer global_indexer(10, 7, 5);
    const OrthogonalBlockIndexer indexer(
        global_indexer,
        {3, 2, 2},
        {1, 1, 0});

    EXPECT_EQ(indexer.block_begin(),
              (OrthogonalBlockIndexer::BlockOrigin{4, 4, 0}));
    EXPECT_EQ(indexer.local_indexer().num_cells_per_dim,
              (SimpleFluid::Vec3D<unsigned>{3, 3, 3}));
    EXPECT_EQ(indexer.num_owned_cells(), 27U);
    EXPECT_EQ(indexer.num_local_cells(), 27U);

    EXPECT_EQ(
        indexer.local_to_global_cell_id({0, 0, 0}),
        (OrthogonalIndexer::CellID{4, 4, 0}));
    EXPECT_EQ(
        indexer.local_to_global_cell_id({2, 2, 2}),
        (OrthogonalIndexer::CellID{6, 6, 2}));
    EXPECT_EQ(
        indexer.global_to_local_cell_id({5, 6, 1}),
        (OrthogonalIndexer::CellID{1, 2, 1}));
    EXPECT_THROW(
        indexer.global_to_local_cell_id({3, 4, 0}),
        std::out_of_range);

    const OrthogonalIndexer::CellID local_cell{1, 2, 1};
    const OrthogonalIndexer::CellID global_cell{5, 6, 1};
    const auto local_cell_ordinal =
        indexer.local_indexer().cell_ordinal(local_cell);
    const auto global_cell_ordinal =
        global_indexer.cell_ordinal(global_cell);
    EXPECT_EQ(
        indexer.local_to_global_cell_ordinal(local_cell_ordinal),
        global_cell_ordinal);
    EXPECT_EQ(
        indexer.global_to_local_cell_ordinal(global_cell_ordinal),
        local_cell_ordinal);
    EXPECT_EQ(
        indexer.global_to_local_cell_ordinal(
            global_indexer.cell_ordinal({0, 0, 0})),
        OrthogonalBlockIndexer::invalid_local_id());

    const OrthogonalIndexer::FaceID local_face{
        3, 1, 2, OrthogonalIndexer::I_FACE};
    const OrthogonalIndexer::FaceID global_face{
        7, 5, 2, OrthogonalIndexer::I_FACE};
    EXPECT_EQ(
        indexer.local_to_global_face_id(local_face), global_face);
    EXPECT_EQ(
        indexer.global_to_local_face_id(global_face), local_face);
    const auto local_face_ordinal =
        indexer.local_indexer().face_ordinal(local_face);
    const auto global_face_ordinal =
        global_indexer.face_ordinal(global_face);
    EXPECT_EQ(
        indexer.local_to_global_face_ordinal(local_face_ordinal),
        global_face_ordinal);
    EXPECT_EQ(
        indexer.global_to_local_face_ordinal(global_face_ordinal),
        local_face_ordinal);

    const OrthogonalIndexer::NodeID local_node{3, 3, 3};
    const OrthogonalIndexer::NodeID global_node{7, 7, 3};
    EXPECT_EQ(
        indexer.local_to_global_node_id(local_node), global_node);
    EXPECT_EQ(
        indexer.global_to_local_node_id(global_node), local_node);
    const auto local_node_ordinal =
        indexer.local_indexer().node_ordinal(local_node);
    const auto global_node_ordinal =
        global_indexer.node_ordinal(global_node);
    EXPECT_EQ(
        indexer.local_to_global_node_ordinal(local_node_ordinal),
        global_node_ordinal);
    EXPECT_EQ(
        indexer.global_to_local_node_ordinal(global_node_ordinal),
        local_node_ordinal);
}

TEST(OrthogonalLocalGlobalIndexerTest, MapsPeriodicUpperBlockSeam)
{
    const OrthogonalIndexer global_indexer(6, 4, 2, true, false, false);
    const OrthogonalBlockIndexer indexer(
        global_indexer,
        {2, 1, 1},
        {1, 0, 0});

    EXPECT_FALSE(indexer.local_indexer().periodic_dimensions[
        OrthogonalIndexer::I]);
    EXPECT_EQ(indexer.local_indexer().num_cells_per_dim[
                  OrthogonalIndexer::I],
              3U);
    EXPECT_EQ(indexer.local_indexer().num_nodes_per_dim[
                  OrthogonalIndexer::I],
              4U);

    const OrthogonalIndexer::FaceID local_face{
        3, 1, 0, OrthogonalIndexer::I_FACE};
    const OrthogonalIndexer::FaceID global_face{
        0, 1, 0, OrthogonalIndexer::I_FACE};
    EXPECT_EQ(
        indexer.local_to_global_face_id(local_face), global_face);
    EXPECT_EQ(
        indexer.global_to_local_face_id(global_face), local_face);

    const OrthogonalIndexer::NodeID local_node{3, 2, 1};
    const OrthogonalIndexer::NodeID global_node{0, 2, 1};
    EXPECT_EQ(
        indexer.local_to_global_node_id(local_node), global_node);
    EXPECT_EQ(
        indexer.global_to_local_node_id(global_node), local_node);

    const OrthogonalBlockIndexer full_indexer(global_indexer);
    EXPECT_TRUE(full_indexer.local_indexer().periodic_dimensions[
        OrthogonalIndexer::I]);
    EXPECT_EQ(full_indexer.local_indexer().num_nodes_per_dim[
                  OrthogonalIndexer::I],
              6U);
}

TEST(OrthogonalLocalGlobalIndexerTest, RejectsInvalidBlockDescriptions)
{
    const OrthogonalIndexer global_indexer(4, 3, 2);

    EXPECT_THROW(
        OrthogonalBlockIndexer(
            global_indexer, {0, 1, 1}, {0, 0, 0}),
        std::invalid_argument);
    EXPECT_THROW(
        OrthogonalBlockIndexer(
            global_indexer, {5, 1, 1}, {0, 0, 0}),
        std::invalid_argument);
    EXPECT_THROW(
        OrthogonalBlockIndexer(
            global_indexer, {2, 1, 1}, {2, 0, 0}),
        std::out_of_range);
}
