/** @file testDistributedAdaptivePackets.cc
 * @brief Shared active/ancestor procedural recipes and bounded lossless packets.
 */
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/DistributedAdaptivePackets.hh"
#include "utils/testing_environment.hh"
#include <gtest/gtest.h>
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
} // namespace
