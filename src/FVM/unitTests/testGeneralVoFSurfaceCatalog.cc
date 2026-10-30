/** @file testGeneralVoFSurfaceCatalog.cc
 * @brief Private surface catalog aliasing, ordering, lifetime and preflight checks.
 */
#include "FVM/details/GeneralVoFInternal.hh"
#include <gtest/gtest.h>
namespace
{
using namespace SimpleFluid::VoF;
using Topology = GeneralVoF::Data::Topology;
using Tile = SimpleFluid::Meshes::AMR::AdaptiveTile;
using Owner = Topology::SurfaceTileOwner;
Tile tile(std::uint64_t face)
{
    Tile result;
    result.face_gid = face;
    result.first.region = face + 100;
    result.second = result.first;
    result.second->region += 100;
    result.node_gids = {face + 1, face + 2, face + 3};
    result.identity = {static_cast<unsigned char>(face)};
    return result;
}
TEST(GeneralVoFSurfaceCatalog, DraftOrderBorrowsExactOwnedTilesAndOwnsOnlyHaloExtras)
{
    Topology topology;
    // Caller-visible order includes an exterior face that is not a surface
    // candidate. The surface sequence deliberately interleaves halo support.
    topology.tiles = {tile(31), tile(12), tile(7)};
    topology.tiles[1].second.reset();
    topology.tiles[1].exterior = true;
    topology.extra_surface_tiles = {tile(44), tile(19)};
    topology.surface_tile_references = {
        {Owner::ExtraSurfaceTiles, 1}, {Owner::OwnedTiles, 2}, {Owner::ExtraSurfaceTiles, 0}, {Owner::OwnedTiles, 0}};
    ASSERT_NO_THROW(topology.validate_surface_tiles());
    const std::array<const Tile*, 4> expected = {
        &topology.extra_surface_tiles[1], &topology.tiles[2], &topology.extra_surface_tiles[0], &topology.tiles[0]};
    std::size_t index = 0;
    for (const auto& actual : topology.surface_tiles())
    {
        ASSERT_LT(index, expected.size());
        EXPECT_EQ(&actual, expected[index]);
        EXPECT_EQ(actual.identity.data(), expected[index]->identity.data());
        EXPECT_EQ(actual.node_gids.data(), expected[index]->node_gids.data());
        ++index;
    }
    EXPECT_EQ(index, expected.size());
    EXPECT_EQ(topology.tiles[0].face_gid, 31u);
    EXPECT_EQ(topology.tiles[1].face_gid, 12u);
    EXPECT_EQ(topology.tiles[2].face_gid, 7u);
    EXPECT_EQ(topology.extra_surface_tiles.size(), 2u);
    static_assert(std::is_same_v<decltype(*topology.surface_tiles().begin()), const Tile&>);
}
TEST(GeneralVoFSurfaceCatalog, ReferencesAndBorrowedViewSurviveOwnerAndCatalogGrowth)
{
    auto topology = std::make_shared<Topology>();
    topology->tiles.push_back(tile(23));
    topology->extra_surface_tiles.push_back(tile(61));
    topology->surface_tile_references = {{Owner::OwnedTiles, 0}, {Owner::ExtraSurfaceTiles, 0}};
    const auto view = topology->surface_tiles();
    const auto first = view.begin();
    // Force allocations in all three independent vectors after the view and
    // iterator have been obtained. Stored pointers would become invalid here.
    topology->tiles.reserve(topology->tiles.capacity() + 32);
    topology->extra_surface_tiles.reserve(topology->extra_surface_tiles.capacity() + 32);
    topology->surface_tile_references.reserve(topology->surface_tile_references.capacity() + 32);
    ASSERT_NO_THROW(topology->validate_surface_tiles());
    EXPECT_EQ(&*first, &topology->tiles[0]);
    auto second = first;
    ++second;
    EXPECT_EQ(&*second, &topology->extra_surface_tiles[0]);
    EXPECT_EQ(view.size(), 2u);
    // Accepted publications share the immutable owner for the entire borrow.
    GeneralVoF::Data accepted;
    accepted.topology = topology;
    auto next = accepted;
    topology.reset();
    accepted.topology.reset();
    EXPECT_EQ(&*view.begin(), &next.topology->tiles[0]);
    EXPECT_EQ((*second).face_gid, 61u);
}
TEST(GeneralVoFSurfaceCatalog, EmptyOwnedCatalogRetainsHaloOnlySupportAndEmptyView)
{
    Topology topology;
    EXPECT_TRUE(topology.surface_tiles().empty());
    EXPECT_EQ(topology.surface_tiles().begin(), topology.surface_tiles().end());
    ASSERT_NO_THROW(topology.validate_surface_tiles());
    topology.extra_surface_tiles.push_back(tile(83));
    topology.surface_tile_references.push_back({Owner::ExtraSurfaceTiles, 0});
    ASSERT_NO_THROW(topology.validate_surface_tiles());
    EXPECT_TRUE(topology.tiles.empty());
    EXPECT_EQ(&*topology.surface_tiles().begin(), &topology.extra_surface_tiles[0]);
}
TEST(GeneralVoFSurfaceCatalog, PreflightRejectsInvalidOwnerAndEitherOutOfRangeIndex)
{
    Topology topology;
    topology.tiles.push_back(tile(1));
    topology.extra_surface_tiles.push_back(tile(2));
    topology.surface_tile_references = {{Owner::OwnedTiles, 1}};
    EXPECT_THROW(topology.validate_surface_tiles(), std::invalid_argument);
    topology.surface_tile_references = {{Owner::ExtraSurfaceTiles, 1}};
    EXPECT_THROW(topology.validate_surface_tiles(), std::invalid_argument);
    topology.surface_tile_references = {{static_cast<Owner>(99), 0}};
    EXPECT_THROW(topology.validate_surface_tiles(), std::invalid_argument);
    topology.surface_tile_references = {{Owner::OwnedTiles, 0}, {Owner::ExtraSurfaceTiles, 0}};
    EXPECT_NO_THROW(topology.validate_surface_tiles());
}
TEST(GeneralVoFSurfaceCatalog, FrontierConstructionAndWallBorrowCanonicalAuthorityAcrossGrowthAndPublication)
{
    auto topology = std::make_shared<Topology>();
    topology->tiles = {tile(31)};
    topology->tiles[0].exterior = true;
    topology->tiles[0].second.reset();
    topology->tiles[0].domain = {'d'};
    topology->tiles[0].surface = {'s'};
    topology->extra_surface_tiles = {tile(44)};
    topology->frontier_tile_references = {{Owner::OwnedTiles, 0}, {Owner::ExtraSurfaceTiles, 0}};
    topology->construction_faces = {{31, {Owner::OwnedTiles, 0}}, {44, {Owner::ExtraSurfaceTiles, 0}}};
    topology->node_walls[32].push_back({std::uint64_t{31}, {}, {}, {}});
    const auto frontier = topology->frontier_tiles();
    topology->tiles.reserve(37);
    topology->extra_surface_tiles.reserve(29);
    ASSERT_NO_THROW(topology->validate_surface_tiles());
    EXPECT_EQ(&topology->construction_face(31), &topology->tiles[0]);
    EXPECT_EQ(&topology->construction_face(44), &topology->extra_surface_tiles[0]);
    EXPECT_EQ(&*frontier.begin(), &topology->tiles[0]);
    EXPECT_EQ(&topology->wall_identity(topology->node_walls.at(32)[0]), &topology->tiles[0].identity);
    EXPECT_EQ(topology->construction_face(31).domain, SimpleFluid::Meshes::AMR::AdaptiveBytes{'d'});
    EXPECT_EQ(topology->construction_face(31).surface, SimpleFluid::Meshes::AMR::AdaptiveBytes{'s'});
    GeneralVoF::Data accepted;
    accepted.topology = topology;
    auto next = accepted;
    topology.reset();
    accepted.topology.reset();
    EXPECT_EQ(&*frontier.begin(), &next.topology->construction_face(31));
    EXPECT_EQ(
        next.topology->wall_identity(next.topology->node_walls.at(32)[0]), SimpleFluid::Meshes::AMR::AdaptiveBytes{31});
}
TEST(GeneralVoFSurfaceCatalog, ConstructionAndWallPreflightRejectForeignIdentityAndUnresolvedOwner)
{
    Topology topology;
    topology.tiles = {tile(31), tile(44)};
    topology.tiles[0].exterior = true;
    topology.tiles[0].second.reset();
    topology.frontier_tile_references = {{Owner::ExtraSurfaceTiles, 0}};
    EXPECT_THROW(topology.validate_surface_tiles(), std::invalid_argument);
    topology.frontier_tile_references = {{Owner::OwnedTiles, 0}};
    topology.construction_faces = {{31, {Owner::OwnedTiles, 1}}};
    EXPECT_THROW(topology.validate_surface_tiles(), std::invalid_argument);
    topology.construction_faces = {{31, {Owner::OwnedTiles, 0}}, {44, {Owner::OwnedTiles, 1}}};
    const auto shell =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box({0, 0, 0}, {1, 1, 1});
    topology.tiles[0].geometry = shell->face_geometry(0);
    topology.tiles[1].geometry = shell->face_geometry(1);
    topology.node_walls[32].push_back({std::uint64_t{44}, topology.tiles[0].geometry, {}, {}});
    EXPECT_THROW(topology.validate_surface_tiles(), std::invalid_argument);
    topology.tiles[1].exterior = true;
    EXPECT_THROW(topology.validate_surface_tiles(), std::invalid_argument);
    topology.node_walls[32][0].authority = std::uint64_t{31};
    EXPECT_NO_THROW(topology.validate_surface_tiles());
    // Exact mode retains its original independent wall identity without
    // retaining the finite construction catalog or otherwise unused halo tiles.
    topology.node_walls[32][0].authority = SimpleFluid::Meshes::AMR::AdaptiveBytes{'e'};
    topology.construction_faces.clear();
    EXPECT_NO_THROW(topology.validate_surface_tiles());
    EXPECT_EQ(topology.wall_identity(topology.node_walls[32][0]), SimpleFluid::Meshes::AMR::AdaptiveBytes{'e'});
}
TEST(GeneralVoFKeyCatalog, OppositeInsertionOrdersShareCanonicalKeysAndRejectDuplicates)
{
    Topology topology;
    Key low, middle, high;
    low.region = 3;
    middle.region = 19;
    high.region = 71;
    auto& forward = topology.node_fans[4];
    auto& reverse = topology.adjacency[low];
    for (const auto& key : {low, middle, high})
        EXPECT_TRUE(forward.insert(key, topology.graph_keys).second);
    for (const auto& key : {high, middle, low})
        EXPECT_TRUE(reverse.insert(key, topology.graph_keys).second);
    const std::array<Key, 3> expected{low, middle, high};
    auto first = forward.begin();
    auto second = reverse.begin();
    for (const auto& key : expected)
    {
        ASSERT_NE(first, forward.end());
        ASSERT_NE(second, reverse.end());
        EXPECT_EQ(*first, key);
        EXPECT_EQ(*second, key);
        EXPECT_EQ(&*first, &*second);
        EXPECT_EQ(&*first, &*topology.graph_keys.find(key));
        ++first;
        ++second;
    }
    EXPECT_EQ(first, forward.end());
    EXPECT_EQ(second, reverse.end());
    const auto duplicate = forward.insert(middle, topology.graph_keys);
    EXPECT_FALSE(duplicate.second);
    EXPECT_EQ(&*duplicate.first, &*topology.graph_keys.find(middle));
    EXPECT_EQ(forward.size(), 3u);
    EXPECT_EQ(topology.graph_keys.size(), 3u);
    EXPECT_TRUE(forward.contains(high));
    high.region = 72;
    EXPECT_FALSE(forward.contains(high));
    static_assert(std::is_same_v<decltype(*forward.begin()), const Key&>);
    static_assert(!std::is_assignable_v<decltype(*forward.begin()), Key>);
    static_assert(std::forward_iterator<KeySet::Iterator>);
}
TEST(GeneralVoFKeyCatalog, FullAncestryDistinguishesCartesianConvexAndProductKeys)
{
    namespace AMR = SimpleFluid::Meshes::AMR;
    Topology topology;
    Key cartesian, shifted, convex, product, descendant;
    cartesian.region = shifted.region = convex.region = product.region = descendant.region = 9;
    cartesian.ancestry = AMR::CartesianCellAncestry{2, {1, 3, 0}};
    shifted.ancestry = AMR::CartesianCellAncestry{2, {1, 3, 1}};
    convex.ancestry = AMR::ConvexCellAncestry{12, {2, 5}};
    product.ancestry = AMR::ProductCellAncestry{{12, 4, {2, 5}}, {7, {1}}};
    descendant.ancestry = AMR::ProductCellAncestry{{12, 4, {2, 5}}, {7, {1, 0}}};
    auto& row = topology.node_fans[33];
    for (const auto& key : {descendant, convex, shifted, product, cartesian})
        EXPECT_TRUE(row.insert(key, topology.graph_keys).second);
    const std::array<Key, 5> expected{cartesian, shifted, convex, product, descendant};
    EXPECT_EQ(std::vector<Key>(row.begin(), row.end()), (std::vector<Key>(expected.begin(), expected.end())));
    for (const auto& key : expected)
    {
        EXPECT_TRUE(row.contains(key));
        EXPECT_EQ(topology_key_wire_bytes(key), AMR::encode_adaptive_key(key).size());
    }
    Key participant;
    participant.region = 9;
    participant.ancestry = AMR::StaticCellAncestry{12, 17};
    EXPECT_EQ(topology_key_wire_bytes(participant), AMR::encode_adaptive_key(participant).size());
    auto absent = product;
    std::get<AMR::ProductCellAncestry>(absent.ancestry).base.path.push_back(0);
    EXPECT_FALSE(row.contains(absent));
    EXPECT_TRUE(row.insert(absent, topology.graph_keys).second);
    EXPECT_EQ(topology.graph_keys.size(), 6u);
}
TEST(GeneralVoFKeyCatalog, BorrowedKeysSurvivePoolAndRowGrowthAndAcceptedPublicationLifetime)
{
    auto topology = std::make_shared<Topology>();
    Key original;
    original.region = 8;
    original.ancestry = SimpleFluid::Meshes::AMR::ConvexCellAncestry{7, {3, 1, 4}};
    auto& row = topology->node_fans[27];
    const auto* stable = &*row.insert(original, topology->graph_keys).first;
    const auto* ancestry_data = std::get<SimpleFluid::Meshes::AMR::ConvexCellAncestry>(stable->ancestry).path.data();
    for (std::uint64_t region = 100; region < 356; ++region)
    {
        Key key;
        key.region = region;
        row.insert(key, topology->graph_keys);
    }
    topology->adjacency[original].insert(original, topology->graph_keys);
    EXPECT_EQ(&*row.begin(), stable);
    EXPECT_EQ(&*topology->adjacency.at(original).begin(), stable);
    GeneralVoF::Data source;
    source.topology = topology;
    auto accepted = source;
    topology.reset();
    source.topology.reset();
    EXPECT_EQ(*stable, original);
    EXPECT_EQ(&*accepted.topology->node_fans.at(27).begin(), stable);
    EXPECT_EQ(std::get<SimpleFluid::Meshes::AMR::ConvexCellAncestry>(stable->ancestry).path.data(), ancestry_data);
    EXPECT_EQ(accepted.topology->graph_keys.size(), 257u);
}
TEST(GeneralVoFKeyCatalog, BytePreflightRejectsPoolAndRowGrowthBeforeMutation)
{
    Topology topology;
    Key key;
    key.region = 42;
    KeySet row;
    general_vof_detail::wire::Budget exhausted{0};
    EXPECT_THROW(row.insert(key, topology.graph_keys, &exhausted), SimpleFluid::PLIC::GeometryError);
    EXPECT_TRUE(row.empty());
    EXPECT_TRUE(topology.graph_keys.empty());
    topology.graph_keys.insert(key);
    // Even when the immutable key already exists, the new pointer row needs
    // an admitted allocation before it may mutate.
    EXPECT_THROW(row.insert(key, topology.graph_keys, &exhausted), SimpleFluid::PLIC::GeometryError);
    EXPECT_TRUE(row.empty());
    EXPECT_EQ(topology.graph_keys.size(), 1u);
    EXPECT_TRUE(row.insert(key, topology.graph_keys).second);
    EXPECT_FALSE(row.insert(key, topology.graph_keys, &exhausted).second);
    EXPECT_EQ(row.size(), 1u);
    Key another = key;
    another.region = 43;
    EXPECT_THROW(row.insert(another, topology.graph_keys, &exhausted), SimpleFluid::PLIC::GeometryError);
    EXPECT_EQ(row.size(), 1u);
    EXPECT_EQ(topology.graph_keys.size(), 1u);
}
} // namespace
