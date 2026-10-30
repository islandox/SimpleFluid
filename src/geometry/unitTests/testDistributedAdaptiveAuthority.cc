/** @file testDistributedAdaptiveAuthority.cc
 * @brief Exact crossed overlays, selective edge incidence and collective rejection.
 */
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveAuthority.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <algorithm>
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
// All support keys route to rank zero, so the MPI registrations exercise
// collective success/failure on ranks with no resident leaves as well.
std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes> plane_supports(std::size_t count)
{
    namespace AMR = ::SimpleFluid::Meshes::AMR;
    const auto comm = Tpetra::getDefaultComm();
    std::vector<AMR::AdaptiveBytes> keys;
    for (std::uint64_t ordinal = 0; keys.size() < count; ++ordinal)
    {
        AMR::adaptive_detail::Writer writer;
        writer.word(ordinal);
        if (AMR::adaptive_detail::authority(writer.data, comm->getSize()) == 0)
            keys.push_back(std::move(writer.data));
    }
    std::sort(keys.begin(), keys.end());
    return keys;
}
std::vector<::SimpleFluid::Meshes::AMR::AdaptiveLeaf> complete_planes(std::size_t planes, std::size_t patches,
    bool shared_support = false, bool missing_last = false, bool distribute_planes = false)
{
    namespace AMR = ::SimpleFluid::Meshes::AMR;
    const auto comm = Tpetra::getDefaultComm();
    std::vector<AMR::AdaptiveLeaf> leaves;
    const auto supports = plane_supports(planes);
    for (std::size_t plane = 0; plane < planes; ++plane)
        for (std::size_t patch = 0; patch < patches; ++patch)
            for (std::size_t side = 0; side < 2; ++side)
            {
                if (comm->getRank() != (distribute_planes ? int(plane % comm->getSize()) : 0))
                    continue;
                if (missing_last && plane + 1 == planes && patch + 1 == patches && side == 1)
                    continue;
                const auto x = double(shared_support ? plane * patches + patch : patch);
                const auto z = double(shared_support ? 0 : plane);
                const std::vector<PLIC::GeometryPoint> vertices{{x, 0, z}, {x + 1, 0, z}, {x + 1, 1, z}, {x, 1, z}};
                auto face = PLIC::PlanarFace<ArithmeticMode::Ordinary>::make(vertices);
                AMR::AdaptiveLeaf leaf;
                leaf.key = {0, AMR::ConvexCellAncestry{2 * (plane * patches + patch) + side, {}}};
                leaf.domain = {'p'};
                leaf.surfaces.push_back({shared_support ? supports[0] : supports[plane], side ? face->reversed() : face,
                    side ? -1 : 1, false, {}});
                leaves.push_back(std::move(leaf));
            }
    return leaves;
}
std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes> plane_input_snapshot(
    const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveLeaf>& leaves)
{
    namespace AMR = ::SimpleFluid::Meshes::AMR;
    std::vector<AMR::AdaptiveBytes> records;
    for (const auto& leaf : leaves)
    {
        AMR::adaptive_detail::Writer record;
        record.bytes(AMR::encode_adaptive_key(leaf.key));
        record.bytes(leaf.domain);
        for (const auto& surface : leaf.surfaces)
        {
            record.bytes(surface.key);
            record.bytes(surface.geometry->encode_incidence());
            record.word(surface.orientation == 1);
        }
        records.push_back(std::move(record.data));
    }
    return records;
}
TEST(DistributedAdaptiveAuthority, CompleteParallelSupportGroupsPreserveGeometryAndGlobalIds)
{
    namespace AMR = ::SimpleFluid::Meshes::AMR;
    const auto comm = Tpetra::getDefaultComm();
    auto leaves = complete_planes(8, 2);
    const auto original = plane_input_snapshot(leaves);
    std::map<AMR::AdaptiveBytes, std::uint64_t> expected_faces;
    for (std::size_t i = 0; i < leaves.size(); i += 2)
    {
        const auto& surface = leaves[i].surfaces.front();
        AMR::adaptive_detail::Writer identity;
        identity.bytes(surface.key);
        identity.bytes(std::min(
            PLIC::canonical_face_key(*surface.geometry), PLIC::canonical_face_key(*surface.geometry->reversed())));
        expected_faces.emplace(std::move(identity.data), 0);
    }
    std::uint64_t ordinal = 0;
    for (auto& [key, gid] : expected_faces)
        gid = ordinal++;
    AMR::AdaptiveAuthorityStats stats;
    std::vector<AMR::AdaptiveNode> nodes;
    auto tiles = AMR::decide_adaptive_tiles(leaves, comm, 4 * 1024 * 1024, 64, &stats, &nodes);
    EXPECT_EQ(stats.global_faces, 16);
    EXPECT_EQ(stats.global_nodes, 48);
    EXPECT_EQ(tiles.size(), leaves.size() / 2);
    std::map<AMR::AdaptiveBytes, std::uint64_t> baseline_nodes, baseline_faces;
    for (const auto& tile : tiles)
    {
        const auto expected_face = expected_faces.find(tile.identity);
        EXPECT_NE(expected_face, expected_faces.end());
        if (expected_face != expected_faces.end())
            EXPECT_EQ(tile.face_gid, expected_face->second);
        EXPECT_TRUE(tile.second.has_value());
        EXPECT_FALSE(tile.exterior);
        EXPECT_NEAR(tile.geometry->area().value, 1, 1e-15);
        bool expected = false;
        for (std::size_t i = 0; i < leaves.size(); i += 2)
            expected |=
                tile.surface == leaves[i].surfaces[0].key && tile.geometry->equivalent(*leaves[i].surfaces[0].geometry);
        EXPECT_TRUE(expected);
        baseline_faces.emplace(tile.identity, tile.face_gid);
    }
    for (const auto& node : nodes)
        baseline_nodes.emplace(node.identity, node.gid);
    EXPECT_EQ(plane_input_snapshot(leaves), original);
    std::reverse(leaves.begin(), leaves.end());
    const auto reversed = plane_input_snapshot(leaves);
    nodes.clear();
    tiles = AMR::decide_adaptive_tiles(leaves, comm, 4 * 1024 * 1024, 64, &stats, &nodes);
    std::map<AMR::AdaptiveBytes, std::uint64_t> permuted_nodes, permuted_faces;
    for (const auto& tile : tiles)
        permuted_faces.emplace(tile.identity, tile.face_gid);
    for (const auto& node : nodes)
        permuted_nodes.emplace(node.identity, node.gid);
    EXPECT_EQ(permuted_faces, baseline_faces);
    EXPECT_EQ(permuted_nodes, baseline_nodes);
    EXPECT_EQ(plane_input_snapshot(leaves), reversed);
}
TEST(DistributedAdaptiveAuthority, IndependentSupportsFitButOneCompleteSupportExceedsOverlapBudget)
{
    namespace AMR = ::SimpleFluid::Meshes::AMR;
    const auto comm = Tpetra::getDefaultComm();
    constexpr std::size_t budget = 4 * 1024 * 1024;
    auto independent = complete_planes(32, 8);
    AMR::AdaptiveAuthorityStats stats;
    const auto tiles = AMR::decide_adaptive_tiles(independent, comm, budget, 512, &stats);
    EXPECT_EQ(stats.global_faces, 256);
    EXPECT_EQ(tiles.size(), independent.size() / 2);
    auto one_support = complete_planes(32, 8, true);
    // The same 512 occurrences in one complete support require at least
    // 512^2 shared-pointer slots, before its decoded geometry and replies.
    EXPECT_THROW(AMR::decide_adaptive_tiles(one_support, comm, budget, 512), std::invalid_argument);
}
TEST(DistributedAdaptiveAuthority, TileLimitAccumulatesAcrossCompleteSupportGroups)
{
    namespace AMR = ::SimpleFluid::Meshes::AMR;
    const auto comm = Tpetra::getDefaultComm();
    const auto one = complete_planes(1, 1);
    EXPECT_NO_THROW(AMR::decide_adaptive_tiles(one, comm, 4 * 1024 * 1024, 1));
    // MPI owners each receive one tile, while authority zero decides both
    // groups: rejection must retain the authority-wide tile count.
    const auto two = complete_planes(2, 1, false, false, true);
    EXPECT_THROW(AMR::decide_adaptive_tiles(two, comm, 4 * 1024 * 1024, 1), std::invalid_argument);
    EXPECT_NO_THROW(AMR::decide_adaptive_tiles(two, comm, 4 * 1024 * 1024, 2));
}
TEST(DistributedAdaptiveAuthority, LateIncompleteGroupRejectsCollectivelyAndNextCallRecovers)
{
    namespace AMR = ::SimpleFluid::Meshes::AMR;
    const auto comm = Tpetra::getDefaultComm();
    const auto missing = complete_planes(8, 2, false, true);
    const auto original = plane_input_snapshot(missing);
    std::vector<AMR::AdaptiveNode> catalog{{91, {'x'}, {}, rectangle(0, 0, 1, 1)->vertex_identities().front()}};
    EXPECT_THROW(
        AMR::decide_adaptive_tiles(missing, comm, 4 * 1024 * 1024, 64, nullptr, &catalog), std::invalid_argument);
    EXPECT_EQ(plane_input_snapshot(missing), original);
    EXPECT_EQ(catalog.size(), 1);
    if (!catalog.empty())
    {
        EXPECT_EQ(catalog[0].gid, 91);
        EXPECT_EQ(catalog[0].identity, AMR::AdaptiveBytes({'x'}));
    }
    const auto complete = complete_planes(8, 2);
    AMR::AdaptiveAuthorityStats stats;
    EXPECT_NO_THROW(AMR::decide_adaptive_tiles(complete, comm, 4 * 1024 * 1024, 64, &stats, &catalog));
    EXPECT_EQ(stats.global_faces, 16);
    EXPECT_EQ(stats.global_nodes, 48);
    EXPECT_EQ(catalog.size(), comm->getRank() == 0 ? 48 : 0);
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
std::vector<::SimpleFluid::Meshes::AMR::AdaptiveLeaf> separate_support_column(std::size_t count, unsigned domains = 1)
{
    namespace AMR = ::SimpleFluid::Meshes::AMR;
    std::vector<AMR::AdaptiveLeaf> leaves;
    if (Tpetra::getDefaultComm()->getRank() != 0)
        return leaves;
    for (unsigned domain = 0; domain < domains; ++domain)
        for (std::size_t i = 0; i < count; ++i)
        {
            AMR::AdaptiveLeaf leaf;
            leaf.key = {domain, AMR::ConvexCellAncestry{i, {}}};
            leaf.domain = {char('a' + domain)};
            AMR::adaptive_detail::Writer support;
            support.word(domain);
            support.word(i);
            leaf.surfaces.push_back(
                {std::move(support.data), rectangle(0, double(i), 1, double(i + 1)), 1, true, "wall"});
            leaves.push_back(std::move(leaf));
        }
    return leaves;
}
TEST(DistributedAdaptiveAuthority, RepeatedExactLineKeysFitBoundedSubscriptionsAndRecoverFromTightPeak)
{
    namespace AMR = ::SimpleFluid::Meshes::AMR;
    const auto comm = Tpetra::getDefaultComm();
    constexpr std::size_t segments = 1024;
    const auto leaves = separate_support_column(segments);
    const auto original = plane_input_snapshot(leaves);
    // Distinct face supports avoid a quadratic face-overlay table. The two
    // vertical exact line keys are independently shared by all column segments.
    bool rejected = false;
    try
    {
        (void) AMR::decide_adaptive_tiles(leaves, comm, 2 * 1024 * 1024, segments);
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message = error.what();
        EXPECT_TRUE(message.find("budget") != std::string::npos ||
                    message == "Distributed adaptation rejected on another rank." ||
                    message == "Owner packet validation failed on another rank.")
            << message;
        rejected = true;
    }
    catch (const std::length_error& error)
    {
        const std::string message = error.what();
        EXPECT_TRUE(message == "Owner packet receive buffers exceed byte budget." ||
                    message == "Owner packet validation failed on another rank.")
            << message;
        rejected = true;
    }
    EXPECT_TRUE(rejected);
    AMR::AdaptiveAuthorityStats stats;
    std::vector<AMR::AdaptiveNode> nodes;
    // Both long lines require their complete N+1-point endpoint index. The
    // first finished index and per-query results must die before later lines;
    // retaining their cumulative admissions rejected this same 4 MiB case.
    const auto tiles = AMR::decide_adaptive_tiles(leaves, comm, 4 * 1024 * 1024, segments, &stats, &nodes);
    EXPECT_EQ(stats.global_faces, segments);
    EXPECT_EQ(stats.global_nodes, 2 * (segments + 1));
    EXPECT_EQ(tiles.size(), leaves.size());
    EXPECT_EQ(stats.edge_reply_points, 8 * leaves.size());
    EXPECT_EQ(nodes.size(), comm->getRank() == 0 ? 2 * (segments + 1) : 0);
    for (const auto& tile : tiles)
    {
        const auto segment = std::get<AMR::ConvexCellAncestry>(tile.first.ancestry).root;
        EXPECT_LT(segment, segments);
        EXPECT_EQ(tile.geometry->vertex_identities(),
            rectangle(0, double(segment), 1, double(segment + 1))->vertex_identities());
        EXPECT_EQ(tile.node_gids.size(), 4);
        EXPECT_EQ(tile.boundary, "wall");
    }
    EXPECT_EQ(plane_input_snapshot(leaves), original);
}
TEST(DistributedAdaptiveAuthority, EqualGeometricLinesInDifferentDomainsKeepDistinctCanonicalNodes)
{
    namespace AMR = ::SimpleFluid::Meshes::AMR;
    const auto comm = Tpetra::getDefaultComm();
    auto leaves = separate_support_column(2, 2);
    // Domain a has a midpoint along the two vertical exact lines; domain b
    // spans the whole height in one face. Domain a's point must never enrich b.
    std::erase_if(leaves, [](const AMR::AdaptiveLeaf& leaf)
        { return leaf.key.region == 1 && std::get<AMR::ConvexCellAncestry>(leaf.key.ancestry).root == 1; });
    for (auto& leaf : leaves)
        if (leaf.key.region == 1)
            leaf.surfaces[0].geometry = rectangle(0, 0, 1, 2);
    AMR::AdaptiveAuthorityStats stats;
    std::vector<AMR::AdaptiveNode> nodes;
    const auto tiles = AMR::decide_adaptive_tiles(leaves, comm, 4 * 1024 * 1024, 16, &stats, &nodes);
    EXPECT_EQ(stats.global_faces, 3);
    EXPECT_EQ(stats.global_nodes, 10);
    EXPECT_EQ(stats.edge_reply_points, 8 * leaves.size());
    EXPECT_EQ(tiles.size(), leaves.size());
    std::map<AMR::AdaptiveBytes, std::set<std::uint64_t>> by_domain;
    std::map<std::uint64_t, const AMR::AdaptiveNode*> by_gid;
    for (const auto& node : nodes)
    {
        AMR::adaptive_detail::Reader identity{node.identity};
        const auto domain = identity.bytes();
        const auto point = identity.bytes();
        identity.done();
        EXPECT_EQ(PLIC::canonical_key(node.coordinate_identity), point);
        EXPECT_EQ(node.coordinate, PLIC::point_display<ArithmeticMode::Ordinary>(node.coordinate_identity));
        by_domain[domain].insert(node.gid);
        by_gid.emplace(node.gid, &node);
    }
    if (comm->getRank() == 0)
    {
        EXPECT_EQ(by_domain[AMR::AdaptiveBytes{'a'}].size(), 6);
        EXPECT_EQ(by_domain[AMR::AdaptiveBytes{'b'}].size(), 4);
        for (const auto gid : by_domain[AMR::AdaptiveBytes{'a'}])
            EXPECT_FALSE(by_domain[AMR::AdaptiveBytes{'b'}].contains(gid));
    }
    for (const auto& tile : tiles)
    {
        const auto segment = std::get<AMR::ConvexCellAncestry>(tile.first.ancestry).root;
        const auto expected =
            (tile.first.region == 1 ? rectangle(0, 0, 1, 2) : rectangle(0, double(segment), 1, double(segment + 1)))
                ->vertex_identities();
        EXPECT_EQ(tile.geometry->vertex_identities(), expected);
        EXPECT_EQ(tile.node_gids.size(), expected.size());
        for (std::size_t i = 0; i < tile.node_gids.size() && i < expected.size(); ++i)
        {
            const auto found = by_gid.find(tile.node_gids[i]);
            EXPECT_NE(found, by_gid.end());
            if (found != by_gid.end())
            {
                AMR::adaptive_detail::Reader identity{found->second->identity};
                EXPECT_EQ(identity.bytes(), tile.domain);
                EXPECT_EQ(identity.bytes(), PLIC::canonical_key(expected[i]));
                identity.done();
            }
        }
    }
}
TEST(DistributedAdaptiveAuthority, ExactReservedTileEncodingMatchesIndependentWireReference)
{
    namespace AMR = ::SimpleFluid::Meshes::AMR;
    for (const bool interior : {false, true})
    {
        AMR::AdaptiveTile tile;
        tile.face_gid = 37;
        tile.surface.assign(72, 's');
        tile.domain = {'d'};
        tile.identity.assign(608, 'i');
        auto face = rectangle(0, 0, 1, 1);
        const std::vector<PLIC::PointIdentity<ArithmeticMode::Ordinary>> inserted{
            rectangle(.5, 0, 1, 1)->vertex_identities().front()};
        tile.geometry = face->with_edge_vertices(inserted);
        tile.first = {2, AMR::ConvexCellAncestry{7, {3, 4}}};
        tile.first_owner = 3;
        tile.first_native_face = 11;
        tile.exterior = !interior;
        tile.boundary = interior ? "" : "wall";
        tile.node_gids = {11, 13, 17, 19, 23};
        if (interior)
        {
            tile.second = AMR::AdaptiveCellKey{5, AMR::ConvexCellAncestry{9, {6}}};
            tile.second_owner = 4;
            tile.second_native_face = 29;
            tile.second_image_identity = PLIC::displacement<ArithmeticMode::Ordinary>({0, 0, 0}, {0, 0, 2});
            tile.second_node_gids = {31, 37, 41, 43, 47};
        }
        // Independent reference for the framed protocol: explicitly append each
        // little-endian word and length-prefixed field without the production
        // Writer, counting pass or reserve policy.
        AMR::AdaptiveBytes reference;
        auto word = [&](std::uint64_t value)
        {
            for (unsigned byte = 0; byte < 8; ++byte)
                reference.push_back(char(value >> (8 * byte)));
        };
        auto bytes = [&](std::span<const char> value)
        {
            word(value.size());
            reference.insert(reference.end(), value.begin(), value.end());
        };
        word(37);
        for (std::size_t index = 0; index < AMR::adaptive_detail::geometry_record_header_bytes; ++index)
        {
            std::uint8_t value = 0;
            if (index == offsetof(AMR::adaptive_detail::GeometryRecordHeader, family))
                value = static_cast<std::uint8_t>(AMR::adaptive_detail::PacketFamily::GeometryRecord);
            else if (index == offsetof(AMR::adaptive_detail::GeometryRecordHeader, version))
                value = AMR::adaptive_detail::geometry_record_version;
            else if (index == offsetof(AMR::adaptive_detail::GeometryRecordHeader, kind))
                value = static_cast<std::uint8_t>(AMR::adaptive_detail::GeometryRecordKind::Tile);
            reference.push_back(static_cast<char>(value));
        }
        bytes(tile.surface);
        bytes(tile.domain);
        bytes(tile.identity);
        bytes(tile.geometry->encode_incidence());
        bytes(PLIC::encode_identity(tile.second_image_identity));
        bytes(AMR::encode_adaptive_key(tile.first));
        const auto second_flag = reference.size();
        word(interior);
        if (interior)
            bytes(AMR::encode_adaptive_key(*tile.second));
        word(3);
        word(interior ? 5 : 0);
        word(11);
        word(interior ? 29 : std::numeric_limits<std::uint64_t>::max());
        word(!interior);
        bytes(std::span(tile.boundary));
        const auto first_incidence_count = reference.size();
        word(5);
        for (const auto node : {11, 13, 17, 19, 23})
            word(node);
        word(interior ? 5 : 0);
        if (interior)
            for (const auto node : {31, 37, 41, 43, 47})
                word(node);
        const auto encoded = AMR::encode_adaptive_tile(tile);
        EXPECT_EQ(encoded, reference);
        EXPECT_EQ(encoded.capacity(), encoded.size());
        EXPECT_LT(encoded.capacity(), reference.capacity());
        const auto restored = AMR::decode_adaptive_tile(encoded);
        EXPECT_EQ(restored.geometry->vertex_identities(), tile.geometry->vertex_identities());
        EXPECT_EQ(restored.node_gids, tile.node_gids);
        EXPECT_EQ(restored.second_node_gids, tile.second_node_gids);
        EXPECT_EQ(restored.second_image_identity, tile.second_image_identity);
        EXPECT_EQ(restored.first, tile.first);
        EXPECT_EQ(restored.second, tile.second);
        // Existing decoder rejection remains independent of output allocation.
        auto malformed = reference;
        malformed[second_flag] = 2;
        EXPECT_THROW(AMR::decode_adaptive_tile(malformed), std::invalid_argument);
        malformed = reference;
        malformed[first_incidence_count] = char(255);
        EXPECT_THROW(AMR::decode_adaptive_tile(malformed), std::invalid_argument);
        malformed = reference;
        malformed.pop_back();
        EXPECT_THROW(AMR::decode_adaptive_tile(malformed), std::invalid_argument);
        malformed = reference;
        malformed.push_back(0);
        EXPECT_THROW(AMR::decode_adaptive_tile(malformed), std::invalid_argument);
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
    wrong[sizeof(std::uint64_t) + offsetof(A::adaptive_detail::GeometryRecordHeader, kind)] =
        static_cast<char>(A::adaptive_detail::GeometryRecordKind::SurfaceRequest);
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
                    incoming[sizeof(std::uint64_t) + offsetof(A::adaptive_detail::GeometryRecordHeader, version)] =
                        4; // Unsupported outer record version on one rank.
                candidate.emplace(A::decode_adaptive_tile(incoming));
            });
        published = std::move(candidate);
    };
    EXPECT_THROW(prepare(), std::invalid_argument);
    EXPECT_FALSE(published);
    EXPECT_EQ(A::encode_adaptive_tile(tile), bytes);
}

} // namespace
