/** @file testDistributedAdaptiveAuthority.cc
 * @brief Exact crossed overlays, selective edge incidence and collective rejection.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveAuthority.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <bit>
#include <gtest/gtest.h>
#include <map>
#include <set>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
SimpleFluid::PLIC::PlanarFacePtr<SimpleFluid::ArithmeticMode::Ordinary> rectangle(
    double x0, double y0, double x1, double y1)
{
    const std::vector<SimpleFluid::PLIC::GeometryPoint> points{{x0, y0, 0}, {x1, y0, 0}, {x1, y1, 0}, {x0, y1, 0}};
    return SimpleFluid::PLIC::PlanarFace<SimpleFluid::ArithmeticMode::Ordinary>::make(points);
}
std::vector<::SimpleFluid::Meshes::AMR::AdaptiveLeaf> crossed(int permutation, bool missing = false)
{
    auto comm = Tpetra::getDefaultComm();
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveLeaf> leaves;
    for (int i = 0; i < 4; ++i)
    {
        if ((i + permutation) % comm->getSize() != comm->getRank() || (missing && i == 3))
            continue;
        ::SimpleFluid::Meshes::AMR::AdaptiveLeaf leaf;
        leaf.key = {std::uint64_t(i / 2), ::SimpleFluid::Meshes::AMR::ConvexCellAncestry{std::uint64_t(i), {}}};
        leaf.domain = {'d'};
        auto face =
            i < 2 ? rectangle(i * .5, 0, (i + 1) * .5, 1) : rectangle(0, (i - 2) * .5, 1, (i - 1) * .5)->reversed();
        leaf.surfaces.push_back({{'s'}, face, i < 2 ? 1 : -1, false, {}});
        leaves.push_back(std::move(leaf));
    }
    return leaves;
}
TEST(DistributedAdaptiveAuthority, CrossedOverlayIsOwnershipIndependent)
{
    auto comm = Tpetra::getDefaultComm();
    std::set<::SimpleFluid::Meshes::AMR::AdaptiveBytes> baseline;
    for (int permutation = 0; permutation < 2; ++permutation)
    {
        auto leaves = crossed(permutation);
        ::SimpleFluid::Meshes::AMR::AdaptiveAuthorityStats stats;
        std::vector<::SimpleFluid::Meshes::AMR::AdaptiveNode> nodes;
        auto tiles =
            ::SimpleFluid::Meshes::AMR::decide_adaptive_tiles(leaves, comm, 16 * 1024 * 1024, 1024, &stats, &nodes);
        EXPECT_EQ(stats.global_faces, 4);
        EXPECT_EQ(stats.global_nodes, 9);
        std::map<std::uint64_t, const ::SimpleFluid::Meshes::AMR::AdaptiveNode*> by_gid;
        for (const auto& node : nodes)
        {
            EXPECT_LT(node.gid, stats.global_nodes);
            EXPECT_TRUE(by_gid.emplace(node.gid, &node).second);
        }
        std::set<::SimpleFluid::Meshes::AMR::AdaptiveBytes> identities;
        for (const auto& tile : tiles)
        {
            EXPECT_TRUE(tile.second.has_value());
            EXPECT_FALSE(tile.exterior);
            EXPECT_TRUE(tile.boundary.empty());
            EXPECT_EQ(tile.domain, ::SimpleFluid::Meshes::AMR::AdaptiveBytes({'d'}));
            EXPECT_TRUE(identities.insert(tile.identity).second);
            EXPECT_NEAR(tile.geometry->area().value, 0.25, 1e-15);
            auto vertices = tile.geometry->vertex_identities();
            EXPECT_EQ(vertices.size(), 4);
            ASSERT_EQ(tile.node_gids.size(), vertices.size());
            EXPECT_LT(tile.face_gid, stats.global_faces);
            for (std::size_t v = 0; v < vertices.size(); ++v)
            {
                auto node = by_gid.find(tile.node_gids[v]);
                ASSERT_NE(node, by_gid.end());
                ::SimpleFluid::Meshes::AMR::adaptive_detail::Reader key{node->second->identity};
                EXPECT_EQ(key.bytes(), tile.domain);
                EXPECT_EQ(key.bytes(), SimpleFluid::PLIC::canonical_key(vertices[v]));
                key.done();
                EXPECT_EQ(node->second->coordinate,
                    SimpleFluid::PLIC::point_display<SimpleFluid::ArithmeticMode::Ordinary>(vertices[v]));
            }
            bool expected = false;
            for (int x = 0; x < 2; ++x)
                for (int y = 0; y < 2; ++y)
                    expected |= tile.geometry->equivalent(*rectangle(x * .5, y * .5, (x + 1) * .5, (y + 1) * .5));
            EXPECT_TRUE(expected);
        }
        // This four-tile fixture deliberately gathers bounded test evidence;
        // production authority routing remains sparse and never gathers leaves.
        ::SimpleFluid::Meshes::AMR::adaptive_detail::Writer encoded;
        int canonical_owned = 0;
        for (const auto& tile : tiles)
        {
            ::SimpleFluid::Meshes::AMR::adaptive_detail::Writer record;
            record.word(0);
            record.bytes(tile.identity);
            record.word(tile.face_gid);
            record.bytes(::SimpleFluid::Meshes::AMR::encode_adaptive_key(tile.first));
            if (tile.second)
                record.bytes(::SimpleFluid::Meshes::AMR::encode_adaptive_key(*tile.second));
            encoded.bytes(record.data);
            canonical_owned += tile.first_owner == comm->getRank();
        }
        for (const auto& node : nodes)
        {
            ::SimpleFluid::Meshes::AMR::adaptive_detail::Writer record;
            record.word(1);
            record.bytes(node.identity);
            record.word(node.gid);
            for (auto coordinate : node.coordinate)
                record.word(std::bit_cast<std::uint64_t>(coordinate));
            encoded.bytes(record.data);
        }
        int global_count = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &canonical_owned, &global_count);
        EXPECT_EQ(global_count, 4);
        constexpr int capacity = 16384;
        ASSERT_LE(encoded.data.size(), std::size_t(capacity));
        const int local_size = int(encoded.data.size());
        std::vector<int> sizes(comm->getSize());
        Teuchos::gatherAll(*comm, 1, &local_size, int(sizes.size()), sizes.data());
        encoded.data.resize(capacity);
        std::vector<char> gathered(std::size_t(capacity) * comm->getSize());
        Teuchos::gatherAll(*comm, capacity, encoded.data.data(), int(gathered.size()), gathered.data());
        std::set<::SimpleFluid::Meshes::AMR::AdaptiveBytes> global;
        for (int rank = 0; rank < comm->getSize(); ++rank)
        {
            ::SimpleFluid::Meshes::AMR::adaptive_detail::Reader reader{
                std::span(gathered).subspan(std::size_t(rank) * capacity, sizes[rank])};
            while (reader.offset < reader.data.size())
                global.insert(reader.bytes());
        }
        EXPECT_EQ(global.size(), 13);
        if (permutation == 0)
            baseline = std::move(global);
        else
            EXPECT_EQ(global, baseline);
    }
}
TEST(DistributedAdaptiveAuthority, MissingOppositeCoverageRejectsCollectively)
{
    auto leaves = crossed(0, true);
    EXPECT_THROW(
        ::SimpleFluid::Meshes::AMR::decide_adaptive_tiles(leaves, Tpetra::getDefaultComm(), 16 * 1024 * 1024, 1024),
        std::invalid_argument);
}
TEST(DistributedAdaptiveAuthority, RankLocalBudgetFailureRejectsCollectively)
{
    auto comm = Tpetra::getDefaultComm();
    auto leaves = crossed(0);
    EXPECT_THROW(::SimpleFluid::Meshes::AMR::decide_adaptive_tiles(
                     leaves, comm, comm->getRank() == 0 ? 1 : 16 * 1024 * 1024, 1024),
        std::invalid_argument);
}
TEST(DistributedAdaptiveAuthority, LongLineRepliesContainOnlyClosedSegment)
{
    std::vector<SimpleFluid::PLIC::PointIdentity<SimpleFluid::ArithmeticMode::Ordinary>> points;
    for (int i = 0; i <= 256; ++i)
    {
        auto vertices = rectangle(0, i, 1, i + 1)->vertex_identities();
        points.push_back(vertices[0]);
    }
    SimpleFluid::PLIC::EdgePointIndex<SimpleFluid::ArithmeticMode::Ordinary> index(points);
    for (int i = 0; i < 256; ++i)
    {
        auto selected = index.closed_segment(points[i], points[i + 1]);
        EXPECT_EQ(selected.size(), 2);
    }
}
TEST(DistributedAdaptiveAuthority, DistributedLongColumnHasLinearReplyPayload)
{
    auto comm = Tpetra::getDefaultComm();
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveLeaf> leaves;
    for (int i = 0; i < 256; ++i)
        if (i % comm->getSize() == comm->getRank())
        {
            ::SimpleFluid::Meshes::AMR::AdaptiveLeaf leaf;
            leaf.key = {0, ::SimpleFluid::Meshes::AMR::ConvexCellAncestry{std::uint64_t(i), {}}};
            leaf.domain = {'d'};
            leaf.surfaces.push_back({{'w'}, rectangle(0, i, 1, i + 1), 1, true, "wall"});
            leaves.push_back(std::move(leaf));
        }
    ::SimpleFluid::Meshes::AMR::AdaptiveAuthorityStats stats;
    auto tiles = ::SimpleFluid::Meshes::AMR::decide_adaptive_tiles(leaves, comm, 64 * 1024 * 1024, 1024, &stats);
    EXPECT_EQ(tiles.size(), leaves.size());
    EXPECT_EQ(stats.edge_reply_points, 8 * leaves.size());
    EXPECT_LE(stats.edge_reply_bytes, 4 * stats.edge_request_bytes);
    for (const auto& tile : tiles)
    {
        EXPECT_TRUE(tile.exterior);
        EXPECT_EQ(tile.boundary, "wall");
        EXPECT_EQ(tile.geometry->vertex_identities().size(), 4);
    }
}
TEST(DistributedAdaptiveAuthority, IncidenceWireRetainsInsertedCollinearNode)
{
    auto face = rectangle(0, 0, 1, 1);
    auto midpoint = rectangle(.5, 0, 1, 1)->vertex_identities().front();
    std::vector<SimpleFluid::PLIC::PointIdentity<SimpleFluid::ArithmeticMode::Ordinary>> inserted{midpoint};
    auto enriched = face->with_edge_vertices(inserted);
    ASSERT_EQ(enriched->vertex_identities().size(), 5);
    auto restored =
        SimpleFluid::PLIC::PlanarFace<SimpleFluid::ArithmeticMode::Ordinary>::decode(enriched->encode_incidence());
    EXPECT_EQ(restored->vertex_identities(), enriched->vertex_identities());
    EXPECT_EQ(restored->encode(), face->encode());
}

TEST(DistributedAdaptiveAuthority, TileCodecRejectsLegacyWrongKindAndRankLocalCorruption)
{
    namespace A = ::SimpleFluid::Meshes::AMR;
    const auto comm = Tpetra::getDefaultComm();
    A::AdaptiveTile tile;
    tile.face_gid = 7;
    tile.domain = {'d'};
    tile.surface = {'s'};
    tile.identity = {'i'};
    tile.geometry = rectangle(0, 0, 1, 1);
    tile.first = {0, A::ConvexCellAncestry{0, {}}};
    tile.first_owner = 0;
    tile.exterior = true;
    tile.node_gids = {0, 1, 2, 3};
    const auto bytes = A::encode_adaptive_tile(tile);
    EXPECT_EQ(A::decode_adaptive_tile(bytes).node_gids, tile.node_gids);
    EXPECT_EQ(A::encode_adaptive_tile(A::decode_adaptive_tile(bytes)), bytes);
    EXPECT_THROW(A::decode_adaptive_tile(bytes, bytes.size() - 1), std::invalid_argument);
    auto old = bytes;
    // The leading GID stays first for authority ordinal patching. Old packets
    // had no stamp following it, even when all optional records were absent.
    old.erase(old.begin() + 8, old.begin() + 8 + A::adaptive_detail::geometry_record_header_bytes);
    EXPECT_THROW(A::decode_adaptive_tile(old), std::invalid_argument);
    auto wrong = bytes;
    wrong[17] = static_cast<char>(A::adaptive_detail::GeometryRecordKind::SurfaceRequest);
    EXPECT_THROW(A::decode_adaptive_tile(wrong), std::invalid_argument);
    auto truncated = bytes;
    truncated.pop_back();
    EXPECT_THROW(A::decode_adaptive_tile(truncated), std::invalid_argument);
    auto trailing = bytes;
    trailing.push_back(0);
    EXPECT_THROW(A::decode_adaptive_tile(trailing), std::invalid_argument);
    std::optional<A::AdaptiveTile> published;
    const auto prepare = [&]
    {
        std::optional<A::AdaptiveTile> candidate;
        A::adaptive_detail::collective(*comm,
            [&]
            {
                auto incoming = bytes;
                if (comm->getRank() == 0)
                    incoming[16] = 3; // Unsupported outer record version on one rank.
                candidate.emplace(A::decode_adaptive_tile(incoming));
            });
        published = std::move(candidate);
    };
    EXPECT_THROW(prepare(), std::invalid_argument);
    EXPECT_FALSE(published);
    EXPECT_EQ(A::encode_adaptive_tile(tile), bytes);
}

} // namespace
