/** @file testDistributedAdaptivePackets.cc
 * @brief Shared active/ancestor procedural recipes and bounded lossless packets.
 */
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/DistributedAdaptivePackets.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/DistributedMixedKernel.hh"
#include "geometry/mesh/AMR/FrozenNativeDetails.hh"
#include "utils/testing_environment.hh"
#include <gtest/gtest.h>
#include <algorithm>
#include <map>
#include <numeric>
#include <set>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
constexpr std::size_t budget = 8 * 1024 * 1024;
TEST(DistributedAdaptivePackets, ActiveAndAncestorGraphsShareLosslessParts)
{
    ::SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.max_level = 2;
    ::SimpleFluid::Meshes::AMR::CartesianAMR source(grid);
    auto refined = source.adapt({0});
    std::vector<std::uint64_t> cells(64);
    std::iota(cells.begin(), cells.end(), 0);
    auto detached = refined->detach_owned(cells, {'d'});
    ASSERT_EQ(detached.owned.size(), 64);
    ASSERT_EQ(detached.families.size(), 8);
    // Independently allocated equal payloads must deduplicate by full bytes,
    // including the saved parent capsules, rather than pointer/hash equality.
    auto attach = [](::SimpleFluid::Meshes::AMR::AdaptiveLeaf& leaf)
    {
        auto factor = std::make_shared<::SimpleFluid::Meshes::AMR::AdaptiveFactorRecipe>();
        factor->root = std::make_shared<const ::SimpleFluid::Meshes::AMR::AdaptiveBytes>(32768, 'r');
        factor->base = std::make_shared<const ::SimpleFluid::Meshes::AMR::AdaptiveBytes>(8192, 'b');
        factor->axis = std::make_shared<const ::SimpleFluid::Meshes::AMR::AdaptiveBytes>(4096, 'z');
        leaf.factor_recipe = std::move(factor);
        leaf.join_recipe = std::make_shared<const ::SimpleFluid::Meshes::AMR::AdaptiveBytes>(2048, 'j');
    };
    for (auto& leaf : detached.owned)
        attach(leaf);
    for (auto& family : detached.families)
        attach(family.parent);
    const auto packet = ::SimpleFluid::Meshes::AMR::encode_adaptive_batch(detached.owned, detached.families, budget);
    auto result = ::SimpleFluid::Meshes::AMR::decode_adaptive_batch(packet, budget);
    ASSERT_EQ(result.leaves.size(), 64);
    ASSERT_EQ(result.families.size(), 8);
    EXPECT_LT(packet.size(), 200000);
    const auto factor = result.leaves.front().factor_recipe;
    for (std::size_t i = 0; i < result.leaves.size(); ++i)
    {
        EXPECT_EQ(result.leaves[i].key, detached.owned[i].key);
        EXPECT_EQ(result.leaves[i].factor_recipe, factor);
        EXPECT_EQ(result.leaves[i].join_recipe, result.families.front().parent.join_recipe);
        ASSERT_TRUE(result.leaves[i].join_recipe);
        EXPECT_EQ(*result.leaves[i].join_recipe, *detached.owned[i].join_recipe);
        EXPECT_EQ(result.leaves[i].native_recipe->parts[0], result.families.front().parent.native_recipe->parts[0]);
    }
    for (std::size_t i = 0; i < result.families.size(); ++i)
    {
        EXPECT_EQ(result.families[i].parent.factor_recipe, factor);
        EXPECT_EQ(result.families[i].key, detached.families[i].key);
        EXPECT_EQ(result.families[i].children, detached.families[i].children);
    }
    EXPECT_EQ(::SimpleFluid::Meshes::AMR::encode_adaptive_batch(result.leaves, result.families, budget), packet);
    NativeFragmentCache cache;
    auto region = materialize_native_fragment(*result.leaves.front().native_recipe, cache);
    EXPECT_TRUE(std::holds_alternative<CartesianRegion>(region));
}
TEST(DistributedAdaptivePackets, LongValidNativeNamesAreCatalogedOnceAcrossActiveAndSavedParents)
{
    ::SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.max_level = 2;
    ::SimpleFluid::Meshes::AMR::CartesianAMR source(grid);
    auto refined = source.adapt({0});
    std::vector<std::uint64_t> cells(64);
    std::iota(cells.begin(), cells.end(), 0);
    auto detached = refined->detach_owned(cells, {'d'});
    const auto short_packet =
        ::SimpleFluid::Meshes::AMR::encode_adaptive_batch(detached.owned, detached.families, budget);
    NativeFragmentCache cache;
    std::set<::SimpleFluid::Meshes::AMR::AdaptiveBytes> unique_views;
    std::map<const NativeFragmentDescriptor*, std::shared_ptr<const NativeFragmentDescriptor>> renamed;
    const std::string prefix(32768, 'v');
    const auto rename = [&](::SimpleFluid::Meshes::AMR::AdaptiveLeaf& leaf)
    {
        unique_views.insert(leaf.native_recipe->view);
        auto [entry, inserted] = renamed.try_emplace(leaf.native_recipe.get());
        if (inserted)
        {
            auto region = materialize_native_fragment(*leaf.native_recipe, cache);
            const auto& cart = std::get<CartesianRegion>(region);
            // Build the long name through the real native provider factory and
            // encoder. Retain the original topology/axis parts and cell ordinal.
            auto named = cartesian_region(prefix + cart.name(), cart.geometry().cell_edges());
            auto view = extract_native_fragment(MultiRegionMesh::Region{std::move(named)}, cache);
            auto descriptor = std::make_shared<NativeFragmentDescriptor>(*leaf.native_recipe);
            descriptor->view = view->view;
            entry->second = std::move(descriptor);
        }
        leaf.native_recipe = entry->second;
    };
    for (auto& leaf : detached.owned)
        rename(leaf);
    for (auto& family : detached.families)
        rename(family.parent);
    ASSERT_GT(detached.owned.size() + detached.families.size(), unique_views.size());
    const auto packet = ::SimpleFluid::Meshes::AMR::encode_adaptive_batch(detached.owned, detached.families, budget);
    // One per-peer batch: growth is exactly the unique native view payloads,
    // not the number of leaf and saved-parent references to those views.
    EXPECT_EQ(packet.size() - short_packet.size(), unique_views.size() * prefix.size());
    auto decoded = ::SimpleFluid::Meshes::AMR::decode_adaptive_batch(packet, budget);
    ASSERT_EQ(decoded.leaves.size(), detached.owned.size());
    ASSERT_EQ(decoded.families.size(), detached.families.size());
    std::map<::SimpleFluid::Meshes::AMR::AdaptiveBytes, const NativeFragmentDescriptor*> identities;
    const auto inspect = [&](const ::SimpleFluid::Meshes::AMR::AdaptiveLeaf& leaf, double volume)
    {
        auto packet = encode_native_fragment_descriptor(*leaf.native_recipe);
        auto [where, inserted] = identities.emplace(std::move(packet), leaf.native_recipe.get());
        EXPECT_EQ(where->second, leaf.native_recipe.get());
        auto region = materialize_native_fragment(*leaf.native_recipe, cache);
        const auto& cart = std::get<CartesianRegion>(region);
        EXPECT_TRUE(cart.name().starts_with(prefix));
        EXPECT_DOUBLE_EQ(cart.geometry().cell_volume(leaf.native_cell), volume);
    };
    for (std::size_t i = 0; i < decoded.leaves.size(); ++i)
    {
        EXPECT_EQ(decoded.leaves[i].key, detached.owned[i].key);
        inspect(decoded.leaves[i], 1. / 64);
    }
    for (std::size_t i = 0; i < decoded.families.size(); ++i)
    {
        EXPECT_EQ(decoded.families[i].key, detached.families[i].key);
        inspect(decoded.families[i].parent, 1. / 8);
    }
    EXPECT_EQ(::SimpleFluid::Meshes::AMR::encode_adaptive_batch(decoded.leaves, decoded.families, budget), packet);
}
TEST(DistributedAdaptivePackets, MalformedAndBudgetFailuresLeaveAcceptedBatchUntouched)
{
    const auto empty = ::SimpleFluid::Meshes::AMR::encode_adaptive_batch({}, {}, budget);
    const auto accepted = ::SimpleFluid::Meshes::AMR::decode_adaptive_batch(empty, budget);
    EXPECT_TRUE(accepted.leaves.empty());
    EXPECT_TRUE(accepted.families.empty());
    auto bad = empty;
    bad.push_back('x');
    EXPECT_THROW(::SimpleFluid::Meshes::AMR::decode_adaptive_batch(bad, budget), std::invalid_argument);
    bad = empty;
    bad.front() ^= 1;
    EXPECT_THROW(::SimpleFluid::Meshes::AMR::decode_adaptive_batch(bad, budget), std::invalid_argument);
    EXPECT_THROW(::SimpleFluid::Meshes::AMR::decode_adaptive_batch(empty, empty.size() - 1), std::invalid_argument);
    EXPECT_THROW(::SimpleFluid::Meshes::AMR::encode_adaptive_batch({}, {}, empty.size() - 1), std::invalid_argument);
    EXPECT_TRUE(accepted.leaves.empty());
}
TEST(DistributedAdaptivePackets, NativeBatchHeaderPreservesBoundsAndRejectsLegacyFrames)
{
    namespace A = ::SimpleFluid::Meshes::AMR;
    namespace D = A::adaptive_detail;
    using D::BatchHeader;
    A::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    A::CartesianAMR source(grid);
    const std::array<std::uint64_t, 1> selected{0};
    auto detached = source.detach_owned(selected, {'d'});
    ASSERT_EQ(detached.owned.size(), 1u);
    // Keep the exact wire-size budget above independent dictionary metadata charges.
    detached.owned.front().domain.assign(1024, 'd');
    const auto populated = A::encode_adaptive_batch(detached.owned, {}, budget);
    const auto empty = A::encode_adaptive_batch({}, {}, budget);
    for (const auto& packet : {empty, populated})
    {
        ASSERT_GE(packet.size(), sizeof(BatchHeader));
        EXPECT_EQ(static_cast<std::uint8_t>(packet[offsetof(BatchHeader, family)]), 5u);
        EXPECT_EQ(static_cast<std::uint8_t>(packet[offsetof(BatchHeader, version)]), 7u);
        D::Reader count_reader{packet};
        count_reader.offset = offsetof(BatchHeader, dictionary_count);
        const auto dictionary_count = count_reader.word();
        if (packet == empty)
            EXPECT_EQ(dictionary_count, 0u);
        else
            EXPECT_GT(dictionary_count, 0u);
        const auto decoded = A::decode_adaptive_batch(packet, packet.size());
        EXPECT_EQ(A::encode_adaptive_batch(decoded.leaves, decoded.families, packet.size()), packet);
        EXPECT_THROW(A::decode_adaptive_batch(packet, packet.size() - 1), std::invalid_argument);
        EXPECT_THROW(A::encode_adaptive_batch(decoded.leaves, decoded.families, packet.size() - 1),
            std::invalid_argument);
        A::AdaptiveBytes unaligned(packet.size() + 1);
        std::copy(packet.begin(), packet.end(), unaligned.begin() + 1);
        const auto shifted = A::decode_adaptive_batch(
            std::span<const char>{unaligned.data() + 1, packet.size()}, packet.size());
        EXPECT_EQ(A::encode_adaptive_batch(shifted.leaves, shifted.families, budget), packet);
        const auto reject = [](std::span<const char> bytes)
        {
            EXPECT_THROW(A::decode_adaptive_batch(bytes, budget), std::invalid_argument);
        };
        for (std::size_t i = 0; i < sizeof(BatchHeader); ++i)
        {
            const auto in_field = [i](std::size_t offset, std::size_t width)
            { return i >= offset && i - offset < width; };
            const bool field = in_field(offsetof(BatchHeader, family), sizeof(BatchHeader::family)) ||
                               in_field(offsetof(BatchHeader, version), sizeof(BatchHeader::version)) ||
                               in_field(offsetof(BatchHeader, dictionary_count), sizeof(BatchHeader::dictionary_count));
            if (!field)
            {
                EXPECT_EQ(packet[i], 0);
                auto bad = packet;
                bad[i] = 1;
                reject(bad);
            }
            reject(std::span<const char>{packet.data(), i});
        }
        auto bad = packet;
        bad[offsetof(BatchHeader, family)] = char(255);
        reject(bad);
        bad = packet;
        bad[offsetof(BatchHeader, version)] = 6;
        reject(bad);
        bad[offsetof(BatchHeader, version)] = 8;
        reject(bad);
        bad = packet;
        std::fill_n(bad.begin() + offsetof(BatchHeader, dictionary_count), sizeof(std::uint64_t), char(255));
        reject(bad);
        for (const std::uint64_t magic : {0x5346414442410006ULL, 0x5346414442410007ULL})
        {
            D::Writer legacy;
            legacy.word(magic);
            legacy.word(dictionary_count);
            legacy.data.insert(legacy.data.end(), packet.begin() + sizeof(BatchHeader), packet.end());
            reject(legacy.data);
        }
    }
}

TEST(DistributedAdaptivePackets, GeometryRecordHeadersRejectLegacyFramesAndMalformedEntries)
{
    namespace D = ::SimpleFluid::Meshes::AMR::adaptive_detail;
    using D::GeometryRecordHeader;
    for (const auto kind :
        {D::GeometryRecordKind::Tile, D::GeometryRecordKind::SurfaceRequest, D::GeometryRecordKind::EdgeRequest,
            D::GeometryRecordKind::EdgeReply, D::GeometryRecordKind::NodeRequest, D::GeometryRecordKind::NodeReply,
            D::GeometryRecordKind::ClosureContact, D::GeometryRecordKind::ClosureRequests,
            D::GeometryRecordKind::ClosureReply, D::GeometryRecordKind::GhostRequest, D::GeometryRecordKind::GhostReply,
            D::GeometryRecordKind::GhostNode, D::GeometryRecordKind::FrozenPart, D::GeometryRecordKind::FrozenBatch})
    {
        D::Writer current;
        D::write_geometry_record_header(current, kind);
        ASSERT_EQ(current.data.size(), sizeof(GeometryRecordHeader));
        EXPECT_EQ(static_cast<std::uint8_t>(current.data[offsetof(GeometryRecordHeader, family)]), 4u);
        EXPECT_EQ(static_cast<std::uint8_t>(current.data[offsetof(GeometryRecordHeader, version)]), 3u);
        EXPECT_EQ(static_cast<std::uint8_t>(current.data[offsetof(GeometryRecordHeader, kind)]),
            static_cast<std::uint8_t>(kind));
        D::Writer repeated;
        D::write_geometry_record_header(repeated, kind);
        EXPECT_EQ(repeated.data, current.data);
        D::Reader accepted{current.data};
        EXPECT_NO_THROW(D::read_geometry_record_header(accepted, kind));
        EXPECT_EQ(accepted.offset, sizeof(GeometryRecordHeader));
        const auto reject = [kind](std::span<const char> bytes)
        {
            D::Reader reader{bytes};
            EXPECT_THROW(D::read_geometry_record_header(reader, kind), std::invalid_argument);
        };
        for (std::size_t i = 0; i < sizeof(GeometryRecordHeader); ++i)
        {
            if (i != offsetof(GeometryRecordHeader, family) && i != offsetof(GeometryRecordHeader, version) &&
                i != offsetof(GeometryRecordHeader, kind))
            {
                EXPECT_EQ(current.data[i], 0);
                auto padding = current.data;
                padding[i] = 1;
                reject(padding);
            }
            reject(std::span<const char>{current.data.data(), i});
        }
        auto changed = current.data;
        changed[offsetof(GeometryRecordHeader, family)] = char(255);
        reject(changed);
        changed = current.data;
        changed[offsetof(GeometryRecordHeader, version)] = 2;
        reject(changed);
        changed[offsetof(GeometryRecordHeader, version)] = 4;
        reject(changed);
        changed = current.data;
        changed[offsetof(GeometryRecordHeader, kind)] = char(255);
        reject(changed);
        const auto other = kind == D::GeometryRecordKind::Tile ? D::GeometryRecordKind::FrozenPart :
                                                               D::GeometryRecordKind::Tile;
        D::Reader wrong_kind{current.data};
        EXPECT_THROW(D::read_geometry_record_header(wrong_kind, other), std::invalid_argument);
        for (const auto version : {std::uint8_t{2}, std::uint8_t{3}})
        {
            D::Writer legacy;
            legacy.word(0x5346414452454332ULL);
            legacy.byte(version);
            legacy.byte(static_cast<std::uint8_t>(kind));
            reject(legacy.data);
        }
    }
}

TEST(DistributedAdaptivePackets, NewContainingVersionsRejectLegacyEmptyRecordsAndKeepAncestryKeys)
{
    namespace A = ::SimpleFluid::Meshes::AMR;
    const A::AdaptiveCellKey key{7, A::ConvexCellAncestry{9, {1, 3}}};
    const auto key_bytes = A::encode_adaptive_key(key);
    A::adaptive_detail::Reader key_reader{key_bytes};
    EXPECT_EQ(key_reader.word(), 0x5346414450543631ULL);
    EXPECT_EQ(key_reader.word(), 1u);
    EXPECT_EQ(key_reader.word(), 0u);
    EXPECT_EQ(A::decode_adaptive_key(key_bytes), key);
    const auto batch = A::encode_adaptive_batch({}, {}, budget);
    EXPECT_TRUE(A::decode_adaptive_batch(batch, budget).leaves.empty());
    auto old_batch = batch;
    old_batch[offsetof(A::adaptive_detail::BatchHeader, version)] = 6;
    // Previous batch version, including a zero-record batch.
    EXPECT_THROW(A::decode_adaptive_batch(old_batch, budget), std::invalid_argument);
    A::AdaptiveJoinCapsule capsule;
    capsule.route = 2;
    const auto join = A::encode_join_capsule(capsule, budget);
    EXPECT_TRUE(A::decode_join_capsule(join, budget).rules.empty());
    auto old_join = join;
    old_join[8] = 4; // Old join schema, with no nested geometry to detect it.
    EXPECT_THROW(A::decode_join_capsule(old_join, budget), std::invalid_argument);
    A::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    A::CartesianAMR source(grid);
    const std::array<std::uint64_t, 1> selected{0};
    const auto detached = source.detach_owned(selected, {'d'});
    ASSERT_EQ(detached.owned.size(), 1u);
    const auto leaf = A::encode_adaptive_leaf(detached.owned.front());
    EXPECT_EQ(A::decode_adaptive_leaf(leaf).key, detached.owned.front().key);
    auto old_leaf = leaf;
    old_leaf[8] = 3;
    EXPECT_THROW(A::decode_adaptive_leaf(old_leaf), std::invalid_argument);
}

TEST(DistributedAdaptivePackets, FrozenContainersRejectUnstampedEmptyRecords)
{
    namespace A = ::SimpleFluid::Meshes::AMR;
    A::FrozenPart part;
    part.source = 7;
    part.layout.family = RegionLayout::Family::Explicit;
    const auto packet = A::encode_frozen_part(part);
    const auto restored = A::decode_frozen_part(packet);
    EXPECT_EQ(restored->source, part.source);
    EXPECT_EQ(A::encode_frozen_part(*restored), packet);
    auto previous_part = packet;
    previous_part[offsetof(A::adaptive_detail::GeometryRecordHeader, version)] = 2;
    EXPECT_THROW(A::decode_frozen_part(previous_part), std::invalid_argument);
    const A::AdaptiveBytes old_part(packet.begin() + A::adaptive_detail::geometry_record_header_bytes, packet.end());
    EXPECT_THROW(A::decode_frozen_part(old_part), std::invalid_argument);
    const auto empty = A::encode_frozen_batch({}, budget);
    EXPECT_EQ(empty.size(), A::adaptive_detail::geometry_record_header_bytes + 3 * sizeof(std::uint64_t));
    A::adaptive_detail::Reader reader{empty};
    EXPECT_NO_THROW(
        A::adaptive_detail::read_geometry_record_header(reader, A::adaptive_detail::GeometryRecordKind::FrozenBatch));
    EXPECT_EQ(reader.word(), 0u);
    EXPECT_EQ(reader.word(), 0u);
    EXPECT_EQ(reader.word(), 0u);
    EXPECT_NO_THROW(reader.done());
    auto previous_empty = empty;
    previous_empty[offsetof(A::adaptive_detail::GeometryRecordHeader, version)] = 2;
    A::adaptive_detail::Reader previous{previous_empty};
    EXPECT_THROW(
        A::adaptive_detail::read_geometry_record_header(previous, A::adaptive_detail::GeometryRecordKind::FrozenBatch),
        std::invalid_argument);
    for (const auto version : {std::uint8_t{2}, std::uint8_t{3}})
        for (const auto& [kind, current] :
            {std::pair{A::adaptive_detail::GeometryRecordKind::FrozenPart, packet},
                std::pair{A::adaptive_detail::GeometryRecordKind::FrozenBatch, empty}})
        {
            A::adaptive_detail::Writer old_frame;
            old_frame.word(0x5346414452454332ULL);
            old_frame.byte(version);
            old_frame.byte(static_cast<std::uint8_t>(kind));
            old_frame.data.insert(old_frame.data.end(),
                current.begin() + A::adaptive_detail::geometry_record_header_bytes, current.end());
            if (kind == A::adaptive_detail::GeometryRecordKind::FrozenPart)
                EXPECT_THROW(A::decode_frozen_part(old_frame.data), std::invalid_argument);
            else
            {
                A::adaptive_detail::Reader old_batch{old_frame.data};
                EXPECT_THROW(A::adaptive_detail::read_geometry_record_header(old_batch, kind), std::invalid_argument);
            }
        }
    const A::AdaptiveBytes old_empty(3 * sizeof(std::uint64_t), 0);
    A::adaptive_detail::Reader legacy{old_empty};
    EXPECT_THROW(
        A::adaptive_detail::read_geometry_record_header(legacy, A::adaptive_detail::GeometryRecordKind::FrozenBatch),
        std::invalid_argument);
    EXPECT_THROW(A::encode_frozen_batch({}, empty.size() - 1), std::invalid_argument);
}

} // namespace
