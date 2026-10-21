/** @file testDistributedCartesianKernel.cc
 * @brief Detached Cartesian native block ownership and exact cell families.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveAuthority.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "geometry/mesh/NativeFragmentRecipe.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <gtest/gtest.h>
#include <map>
#include <set>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
::SimpleFluid::Meshes::AMR::AMRGrid grid()
{
    ::SimpleFluid::Meshes::AMR::AMRGrid result;
    result.block_cells = {2, 2, 2};
    result.max_level = 2;
    return result;
}
TEST(DistributedCartesianKernel, DetachedExportReleasesSourceAndKeepsNativeBlocks)
{
    auto source = std::make_shared<::SimpleFluid::Meshes::AMR::CartesianAMR>(grid());
    std::weak_ptr<const MultiRegionMesh> old = source->geometry();
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    for (std::uint64_t cell = 0; cell < 8; ++cell)
        if (cell % comm->getSize() == std::uint64_t(comm->getRank()))
            owned.push_back(cell);
    auto detached = source->detach_owned(owned, {'d'});
    for (std::size_t i = 0; i < owned.size(); ++i)
    {
        auto box = source->cell_box(owned[i]);
        EXPECT_TRUE(
            detached.kernel->geometry(detached.owned[i])->equivalent(*ConvexCellGeometry::box(box.lower, box.upper)));
        EXPECT_EQ(detached.owned[i].key, source->cell_key(owned[i]));
        EXPECT_EQ(detached.owned[i].native_cell, owned[i]);
        EXPECT_FALSE(detached.owned[i].geometry);
        EXPECT_TRUE(detached.owned[i].surfaces.empty());
    }
    source.reset();
    EXPECT_TRUE(old.expired());
    NativeFragmentCache cache;
    std::set<const NativeFragmentDescriptor*> recipes;
    for (const auto& leaf : detached.owned)
    {
        recipes.insert(leaf.native_recipe.get());
        auto region = materialize_native_fragment(*leaf.native_recipe, cache);
        EXPECT_TRUE(std::holds_alternative<CartesianRegion>(region));
    }
    EXPECT_LE(recipes.size(), 1);
}
TEST(DistributedCartesianKernel, DetachedAncestorCapsulesShareProceduralParts)
{
    ::SimpleFluid::Meshes::AMR::CartesianAMR source(grid());
    auto refined = source.adapt({0});
    auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    for (std::uint64_t cell = 0; cell < 64; ++cell)
        if (cell % comm->getSize() == std::uint64_t(comm->getRank()))
            owned.push_back(cell);
    auto detached = refined->detach_owned(owned, {'d'});
    std::set<const NativeFragmentDescriptor*> parent_blocks;
    std::set<const NativeFragmentBytes*> topologies;
    for (const auto& leaf : detached.owned)
        topologies.insert(leaf.native_recipe->parts.at(0).get());
    for (const auto& family : detached.families)
    {
        parent_blocks.insert(family.parent.native_recipe.get());
        topologies.insert(family.parent.native_recipe->parts.at(0).get());
    }
    EXPECT_LE(parent_blocks.size(), 1);
    EXPECT_LE(topologies.size(), 1);
    int local = int(detached.families.size()), total = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local, &total);
    EXPECT_EQ(total, 8);
}
TEST(DistributedCartesianKernel, SplitRepackagesEightActualNativeBlocksAndInverse)
{
    auto config = grid();
    auto kernel = ::SimpleFluid::Meshes::AMR::cartesian_adaptive_kernel(config);
    ::SimpleFluid::Meshes::AMR::AdaptiveCellKey first{
        0, ::SimpleFluid::Meshes::AMR::CartesianCellAncestry{0, {0, 0, 0}}};
    auto group = kernel->refinement_group(first, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic);
    ASSERT_EQ(group.size(), 8);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveLeaf> children;
    for (const auto& key : group)
    {
        auto parent = ::SimpleFluid::Meshes::AMR::cartesian_adaptive_leaf(config, key, {'d'});
        auto split = kernel->split(parent, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic);
        ASSERT_EQ(split.children.size(), 8);
        EXPECT_EQ(kernel->coarsening_group(split.family).size(), 8);
        auto restored = kernel->coarsen(split.family, split.children);
        EXPECT_EQ(restored.key, parent.key);
        EXPECT_TRUE(kernel->geometry(restored)->equivalent(*kernel->geometry(parent)));
        children.insert(children.end(), split.children.begin(), split.children.end());
    }
    children = kernel->repack(children);
    ASSERT_EQ(children.size(), 64);
    std::map<::SimpleFluid::Meshes::AMR::AdaptiveBytes, std::set<std::uint64_t>> blocks;
    for (const auto& leaf : children)
    {
        EXPECT_FALSE(leaf.geometry);
        EXPECT_TRUE(leaf.surfaces.empty());
        blocks[encode_native_fragment_descriptor(*leaf.native_recipe)].insert(leaf.native_cell);
        auto wire =
            ::SimpleFluid::Meshes::AMR::decode_adaptive_leaf(::SimpleFluid::Meshes::AMR::encode_adaptive_leaf(leaf));
        EXPECT_EQ(wire.key, leaf.key);
    }
    ASSERT_EQ(blocks.size(), 8);
    for (const auto& [recipe, cells] : blocks)
    {
        EXPECT_EQ(cells.size(), 8);
        EXPECT_EQ(*cells.begin(), 0);
        EXPECT_EQ(*cells.rbegin(), 7);
    }
}
TEST(DistributedCartesianKernel, RefinedNativeBlocksHaveClosedDistributedAuthorityShells)
{
    auto config = grid();
    auto kernel = ::SimpleFluid::Meshes::AMR::cartesian_adaptive_kernel(config);
    auto comm = Tpetra::getDefaultComm();
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveLeaf> owned;
    std::uint64_t ordinal = 0;
    for (const auto& parent_key :
        kernel->refinement_group({0, ::SimpleFluid::Meshes::AMR::CartesianCellAncestry{0, {0, 0, 0}}},
            ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic))
    {
        auto split = kernel->split(::SimpleFluid::Meshes::AMR::cartesian_adaptive_leaf(config, parent_key, {'d'}),
            ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic);
        for (auto& child : split.children)
            if (ordinal++ % comm->getSize() == std::uint64_t(comm->getRank()))
            {
                child.geometry = kernel->geometry(child);
                child.surfaces = kernel->surfaces(child);
                owned.push_back(std::move(child));
            }
    }
    auto tiles = ::SimpleFluid::Meshes::AMR::decide_adaptive_tiles(owned, comm, 64 * 1024 * 1024, 4096);
    int local_count = 0, global_count = 0;
    std::map<::SimpleFluid::Meshes::AMR::AdaptiveCellKey, std::vector<ExactPlanarFacePtr>> shells;
    for (const auto& tile : tiles)
    {
        local_count += tile.first_owner == comm->getRank();
        if (tile.first_owner == comm->getRank())
            shells[tile.first].push_back(tile.geometry);
        if (tile.second_owner == comm->getRank())
            shells[*tile.second].push_back(tile.geometry->reversed());
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_count, &global_count);
    EXPECT_EQ(global_count, 240);
    for (const auto& leaf : owned)
        EXPECT_TRUE(ExactPlanarFace::cell_shell(shells[leaf.key])->equivalent(*leaf.geometry));
}

std::set<::SimpleFluid::Meshes::AMR::AdaptiveBytes> gathered_keys(
    const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy& state)
{
    auto comm = Tpetra::getDefaultComm();
    ::SimpleFluid::Meshes::AMR::adaptive_detail::Writer wire;
    for (const auto& [key, leaf] : state.owned())
        wire.bytes(::SimpleFluid::Meshes::AMR::encode_adaptive_key(key));
    constexpr int bound = 32768;
    const int n = int(wire.data.size());
    if (n > bound)
        throw std::invalid_argument("Bounded Cartesian test gather exceeded fixture size.");
    std::vector<int> sizes(comm->getSize());
    Teuchos::gatherAll(*comm, 1, &n, int(sizes.size()), sizes.data());
    wire.data.resize(bound);
    std::vector<char> all(std::size_t(bound) * comm->getSize());
    Teuchos::gatherAll(*comm, bound, wire.data.data(), int(all.size()), all.data());
    std::set<::SimpleFluid::Meshes::AMR::AdaptiveBytes> result;
    for (int rank = 0; rank < comm->getSize(); ++rank)
    {
        ::SimpleFluid::Meshes::AMR::adaptive_detail::Reader reader{
            std::span(all).subspan(std::size_t(rank) * bound, sizes[rank])};
        while (reader.offset < reader.data.size())
            if (!result.insert(reader.bytes()).second)
                throw std::invalid_argument("Duplicate Cartesian test owner.");
    }
    return result;
}
std::shared_ptr<const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy> hierarchy(
    const ::SimpleFluid::Meshes::AMR::CartesianAMR& source)
{
    auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    const auto cells = source.blocks().size() * 8;
    for (std::uint64_t cell = 0; cell < cells; ++cell)
        if (cell % comm->getSize() == std::uint64_t(comm->getRank()))
            owned.push_back(cell);
    auto detached = source.detach_owned(owned, {'d'});
    return std::make_shared<const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy>(
        std::move(detached.owned), std::move(detached.families), std::move(detached.kernel), comm);
}
void compare_oracle(const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy& state,
    const ::SimpleFluid::Meshes::AMR::CartesianAMR& oracle)
{
    std::set<::SimpleFluid::Meshes::AMR::AdaptiveBytes> expected;
    for (std::uint64_t i = 0; i < oracle.blocks().size() * 8; ++i)
        expected.insert(::SimpleFluid::Meshes::AMR::encode_adaptive_key(oracle.cell_key(i)));
    EXPECT_EQ(gathered_keys(state), expected);
    auto kernel = ::SimpleFluid::Meshes::AMR::cartesian_adaptive_kernel(oracle.config());
    std::array<double, 4> moments{};
    std::map<::SimpleFluid::Meshes::AMR::AdaptiveBytes, std::set<std::uint64_t>> blocks;
    for (const auto& [key, leaf] : state.owned())
    {
        EXPECT_EQ(leaf.provider, ::SimpleFluid::Meshes::AMR::AdaptiveProviderKind::Cartesian);
        EXPECT_FALSE(leaf.geometry);
        ASSERT_TRUE(leaf.native_recipe);
        EXPECT_EQ(leaf.native_recipe->kind, NativeFragmentDescriptor::Kind::Cartesian);
        blocks[encode_native_fragment_descriptor(*leaf.native_recipe)].insert(leaf.native_cell);
        auto shape = kernel->geometry(leaf)->moments();
        const double volume = shape.volume.as_double();
        moments[0] += volume;
        for (std::size_t a = 0; a < 3; ++a)
            moments[a + 1] += shape.first_moment[a].as_double() + shape.origin[a] * volume;
    }
    for (const auto& [recipe, cells] : blocks)
        EXPECT_LE(cells.size(), 8);
    std::array<double, 4> totals{};
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 4, moments.data(), totals.data());
    EXPECT_NEAR(totals[0], 1, 1e-14);
    for (std::size_t a = 1; a < 4; ++a)
        EXPECT_NEAR(totals[a], .5, 1e-14);
}
TEST(DistributedCartesianKernel, DistributedBlockRefineSplitFamiliesCoarsenAndEmptyOwners)
{
    auto comm = Tpetra::getDefaultComm();
    ::SimpleFluid::Meshes::AMR::CartesianAMR source(grid());
    auto state = hierarchy(source);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveMark> marks;
    if (comm->getRank() == 0)
        marks.push_back({source.cell_key(0), ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic});
    auto refine = state->plan_adaptation(marks);
    auto refined = state->apply(refine);
    auto oracle = source.adapt({0});
    compare_oracle(*refined, *oracle);
    EXPECT_EQ(state->topology_generation(), 0);
    EXPECT_EQ(refined->topology_generation(), 1);
    // Deliberately split every immediate eight-child cell family across owners.
    std::vector<int> destinations;
    for (const auto& [key, leaf] : refined->owned())
    {
        const auto& cell = std::get<::SimpleFluid::Meshes::AMR::CartesianCellAncestry>(key.ancestry);
        destinations.push_back(int((cell.index[0] + 4 * (cell.index[1] + 4 * cell.index[2])) % comm->getSize()));
    }
    auto migrate = refined->plan_migration(destinations);
    auto scattered = refined->apply(migrate);
    compare_oracle(*scattered, *oracle);
    EXPECT_THROW(scattered->apply(refine), std::invalid_argument);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes> coarsen;
    if (comm->getRank() == 0)
        coarsen.push_back(::SimpleFluid::Meshes::AMR::adaptive_family_key(
            source.cell_key(0), ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic));
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveMark> conflict;
    for (const auto& [key, leaf] : scattered->owned())
        if (std::get<::SimpleFluid::Meshes::AMR::CartesianCellAncestry>(key.ancestry).index ==
            std::array<std::uint64_t, 3>{0, 0, 0})
            conflict.push_back({key, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic});
    EXPECT_THROW(scattered->plan_adaptation(conflict, coarsen), std::invalid_argument);
    compare_oracle(*scattered, *oracle);
    auto collapse = scattered->plan_adaptation({}, coarsen);
    auto restored = scattered->apply(collapse);
    compare_oracle(*restored, source);
    std::vector<int> all_to_zero(restored->owned().size(), 0);
    auto empty_plan = restored->plan_migration(all_to_zero);
    auto empty = restored->apply(empty_plan);
    EXPECT_EQ(empty->owned().size(), comm->getRank() == 0 ? 8 : 0);
    compare_oracle(*empty, source);
    auto draft = empty->native_draft();
    EXPECT_EQ(draft.owned.size(), empty->owned().size());
    if (comm->getRank() != 0)
        EXPECT_TRUE(draft.tiles.empty());
    auto no_op = empty->plan_adaptation({});
    EXPECT_EQ(empty->apply(no_op)->identity(), empty->identity());
    if (comm->getSize() > 1)
    {
        auto another = empty->plan_adaptation({});
        EXPECT_THROW(empty->apply(comm->getRank() % 2 ? no_op : another), std::invalid_argument);
    }
}
TEST(DistributedConvexHierarchy, SplitSiblingOwnersCoarsenExactParent)
{
    auto comm = Tpetra::getDefaultComm();
    auto kernel = ::SimpleFluid::Meshes::AMR::convex_adaptive_kernel(3, 0);
    ::SimpleFluid::Meshes::AMR::AdaptiveLeaf parent;
    parent.key = {7, ::SimpleFluid::Meshes::AMR::ConvexCellAncestry{3, {}}};
    parent.domain = {'p'};
    parent.material = 9;
    parent.geometry = ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1});
    for (std::size_t f = 0; f < parent.geometry->num_faces(); ++f)
        parent.surfaces.push_back({::SimpleFluid::Meshes::AMR::AdaptiveBytes{char('a' + f)},
            parent.geometry->face_geometry(f), 1, true, "wall"});
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveLeaf> owned;
    if (comm->getRank() == 0)
        owned.push_back(parent);
    ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy state(std::move(owned), {}, kernel, comm);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveMark> marks;
    if (comm->getRank() == 0)
        marks.push_back({parent.key, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic});
    auto split = state.plan_adaptation(marks);
    auto refined = state.apply(split);
    std::vector<int> destinations;
    for (const auto& [key, leaf] : refined->owned())
        destinations.push_back(
            int(std::get<::SimpleFluid::Meshes::AMR::ConvexCellAncestry>(key.ancestry).path.back() % comm->getSize()));
    auto migration = refined->plan_migration(destinations);
    auto scattered = refined->apply(migration);
    auto draft = scattered->native_draft();
    EXPECT_EQ(draft.owned.size(), scattered->owned().size());
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes> requests;
    if (comm->getRank() == 0)
        requests.push_back(::SimpleFluid::Meshes::AMR::adaptive_family_key(
            parent.key, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic));
    auto inverse = scattered->plan_adaptation({}, requests);
    auto restored = scattered->apply(inverse);
    EXPECT_EQ(gathered_keys(*restored), std::set<::SimpleFluid::Meshes::AMR::AdaptiveBytes>{
                                            ::SimpleFluid::Meshes::AMR::encode_adaptive_key(parent.key)});
    if (comm->getRank() == 0)
    {
        ASSERT_EQ(restored->owned().size(), 1);
        EXPECT_TRUE(restored->owned().begin()->second.geometry->equivalent(*parent.geometry));
    }
}
TEST(DistributedCartesianKernel, WrongBlockRecipeOrNativeOrdinalRejectsCollectively)
{
    auto config = grid();
    config.root_blocks[0] = 2;
    auto kernel = ::SimpleFluid::Meshes::AMR::cartesian_adaptive_kernel(config);
    auto comm = Tpetra::getDefaultComm();
    for (bool wrong_recipe : {false, true})
    {
        std::vector<::SimpleFluid::Meshes::AMR::AdaptiveLeaf> owned;
        if (comm->getRank() == 0)
        {
            auto leaf = ::SimpleFluid::Meshes::AMR::cartesian_adaptive_leaf(
                config, {0, ::SimpleFluid::Meshes::AMR::CartesianCellAncestry{0, {0, 0, 0}}}, {'d'});
            if (wrong_recipe)
                leaf.native_recipe = ::SimpleFluid::Meshes::AMR::cartesian_adaptive_leaf(
                    config, {0, ::SimpleFluid::Meshes::AMR::CartesianCellAncestry{0, {2, 0, 0}}}, {'d'})
                                         .native_recipe;
            else
                leaf.native_cell = 1;
            owned.push_back(std::move(leaf));
        }
        EXPECT_THROW(::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy(std::move(owned), {}, kernel, comm),
            std::invalid_argument);
    }
}
TEST(DistributedCartesianKernel, CoarsenRejectsChangedDomainOrProvider)
{
    auto config = grid();
    auto kernel = ::SimpleFluid::Meshes::AMR::cartesian_adaptive_kernel(config);
    auto parent = ::SimpleFluid::Meshes::AMR::cartesian_adaptive_leaf(
        config, {0, ::SimpleFluid::Meshes::AMR::CartesianCellAncestry{0, {0, 0, 0}}}, {'d'});
    auto split = kernel->split(parent, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic);
    auto changed = split.children;
    changed[0].domain = {'x'};
    EXPECT_THROW(kernel->coarsen(split.family, changed), std::invalid_argument);
    changed = split.children;
    changed[0].provider = ::SimpleFluid::Meshes::AMR::AdaptiveProviderKind::Product;
    EXPECT_THROW(kernel->coarsen(split.family, changed), std::invalid_argument);
}
TEST(DistributedConvexHierarchy, CoarsenRejectsChangedDomainOrProvider)
{
    auto kernel = ::SimpleFluid::Meshes::AMR::convex_adaptive_kernel(3, 0);
    ::SimpleFluid::Meshes::AMR::AdaptiveLeaf parent;
    parent.key = {7, ::SimpleFluid::Meshes::AMR::ConvexCellAncestry{3, {}}};
    parent.domain = {'p'};
    parent.geometry = ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1});
    for (std::size_t f = 0; f < parent.geometry->num_faces(); ++f)
        parent.surfaces.push_back({::SimpleFluid::Meshes::AMR::AdaptiveBytes{char('a' + f)},
            parent.geometry->face_geometry(f), 1, true, "wall"});
    auto split = kernel->split(parent, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic);
    auto changed = split.children;
    changed[0].domain = {'x'};
    EXPECT_THROW(kernel->coarsen(split.family, changed), std::invalid_argument);
    changed = split.children;
    changed[0].provider = ::SimpleFluid::Meshes::AMR::AdaptiveProviderKind::Static;
    EXPECT_THROW(kernel->coarsen(split.family, changed), std::invalid_argument);
}
TEST(DistributedCartesianKernel, SimultaneousSubtreeInverseRetainsAncestorAnchors)
{
    auto comm = Tpetra::getDefaultComm();
    ::SimpleFluid::Meshes::AMR::CartesianAMR root(grid());
    auto level_one = root.adapt({0});
    const auto first = level_one->cell_key(0), last = level_one->cell_key(56);
    auto nested = level_one->adapt({0, 7});
    auto state = hierarchy(*nested);
    compare_oracle(*state, *nested);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes> requests;
    if (comm->getRank() == 0)
    {
        requests.push_back(::SimpleFluid::Meshes::AMR::adaptive_family_key(
            first, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic));
        requests.push_back(::SimpleFluid::Meshes::AMR::adaptive_family_key(
            last, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic));
    }
    auto inverse = state->plan_adaptation({}, requests);
    auto restored = state->apply(inverse);
    compare_oracle(*restored, *level_one);
    std::vector<int> destinations;
    for (const auto& [key, leaf] : restored->owned())
    {
        const auto& cell = std::get<::SimpleFluid::Meshes::AMR::CartesianCellAncestry>(key.ancestry);
        destinations.push_back(int((cell.index[0] + 4 * (cell.index[1] + 4 * cell.index[2]) + 1) % comm->getSize()));
    }
    auto migration = restored->plan_migration(destinations);
    auto moved = restored->apply(migration);
    compare_oracle(*moved, *level_one);
    for (const auto& [key, family] : moved->families())
        EXPECT_TRUE(moved->owned().contains(::SimpleFluid::Meshes::AMR::decode_adaptive_key(family.anchor)));
    requests.clear();
    if (comm->getRank() == 0)
        requests.push_back(::SimpleFluid::Meshes::AMR::adaptive_family_key(
            root.cell_key(0), ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic));
    auto final_inverse = moved->plan_adaptation({}, requests);
    auto final = moved->apply(final_inverse);
    compare_oracle(*final, root);
    EXPECT_TRUE(final->families().empty());
}
TEST(DistributedCartesianKernel, RankDivergentKernelSemanticsRejectCollectively)
{
    auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() == 1)
        GTEST_SKIP();
    auto config = grid();
    if (comm->getRank() == 0)
        config.extent[0] = 2;
    auto kernel = ::SimpleFluid::Meshes::AMR::cartesian_adaptive_kernel(config);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveLeaf> leaves;
    if (comm->getRank() == 0)
        leaves.push_back(::SimpleFluid::Meshes::AMR::cartesian_adaptive_leaf(
            config, {0, ::SimpleFluid::Meshes::AMR::CartesianCellAncestry{0, {0, 0, 0}}}, {'d'}));
    EXPECT_THROW(::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy(std::move(leaves), {}, kernel, comm),
        std::invalid_argument);
}
std::set<::SimpleFluid::Meshes::AMR::AdaptiveBytes> periodic_signature(
    const ::SimpleFluid::Meshes::AMR::AdaptiveNativeDraft& draft)
{
    auto comm = Tpetra::getDefaultComm();
    ::SimpleFluid::Meshes::AMR::adaptive_detail::Writer wire;
    for (const auto& tile : draft.tiles)
    {
        ::SimpleFluid::Meshes::AMR::adaptive_detail::Writer record;
        record.word(0);
        record.bytes(tile.identity);
        record.word(tile.face_gid);
        wire.bytes(record.data);
    }
    for (const auto& node : draft.nodes)
    {
        ::SimpleFluid::Meshes::AMR::adaptive_detail::Writer record;
        record.word(1);
        record.bytes(node.identity);
        record.word(node.gid);
        wire.bytes(record.data);
    }
    constexpr int bound = 131072;
    const int n = int(wire.data.size());
    if (n > bound)
        throw std::invalid_argument("Periodic fixture evidence exceeds test bound.");
    std::vector<int> sizes(comm->getSize());
    Teuchos::gatherAll(*comm, 1, &n, int(sizes.size()), sizes.data());
    wire.data.resize(bound);
    std::vector<char> all(std::size_t(bound) * comm->getSize());
    Teuchos::gatherAll(*comm, bound, wire.data.data(), int(all.size()), all.data());
    std::set<::SimpleFluid::Meshes::AMR::AdaptiveBytes> result;
    for (int rank = 0; rank < comm->getSize(); ++rank)
    {
        ::SimpleFluid::Meshes::AMR::adaptive_detail::Reader reader{
            std::span(all).subspan(std::size_t(rank) * bound, sizes[rank])};
        while (reader.offset < reader.data.size())
            result.insert(reader.bytes());
    }
    return result;
}
TEST(DistributedCartesianKernel, PeriodicPhysicalShellsRetainDistinctNodeImages)
{
    auto config = grid();
    config.origin = {.1, -.2, .3};
    config.extent = {.7, .8, .9};
    config.periodic = {true, true, true};
    ::SimpleFluid::Meshes::AMR::CartesianAMR source(config);
    auto state = hierarchy(source);
    auto comm = Tpetra::getDefaultComm();
    auto kernel = ::SimpleFluid::Meshes::AMR::cartesian_adaptive_kernel(config);
    auto draft = state->native_draft();
    EXPECT_EQ(draft.global_faces, 24);
    EXPECT_EQ(draft.global_nodes, 27);
    std::map<std::uint64_t, const ::SimpleFluid::Meshes::AMR::AdaptiveNode*> nodes;
    for (const auto& node : draft.nodes)
        nodes.emplace(node.gid, &node);
    std::map<::SimpleFluid::Meshes::AMR::AdaptiveCellKey, std::vector<ExactPlanarFacePtr>> shells;
    int local_periodic = 0, global_periodic = 0;
    for (const auto& tile : draft.tiles)
    {
        EXPECT_FALSE(tile.exterior);
        ASSERT_TRUE(tile.second);
        ASSERT_EQ(tile.node_gids.size(), tile.geometry->vertex_identities().size());
        ASSERT_EQ(tile.second_node_gids.size(), tile.node_gids.size());
        auto second =
            tile.geometry->reversed()->translated_exact(exact_displacement_difference({}, tile.exact_second_image));
        if (tile.first_owner == comm->getRank())
            shells[tile.first].push_back(tile.geometry);
        if (tile.second_owner == comm->getRank())
            shells[*tile.second].push_back(second);
        if (!tile.exact_second_image.empty())
        {
            local_periodic += tile.first_owner == comm->getRank();
            EXPECT_NE(tile.node_gids, tile.second_node_gids);
            auto first_metric =
                kernel->metrics(::SimpleFluid::Meshes::AMR::cartesian_adaptive_leaf(config, tile.first, {'d'}));
            auto second_metric =
                kernel->metrics(::SimpleFluid::Meshes::AMR::cartesian_adaptive_leaf(config, *tile.second, {'d'}));
            auto normal = tile.geometry->area_vector();
            double dot = 0;
            for (std::size_t axis = 0; axis < 3; ++axis)
                dot += (second_metric.centroid[axis] + tile.second_image[axis] - first_metric.centroid[axis]) *
                       normal.component(axis);
            EXPECT_GT(dot, 0);
        }
        auto vertices = tile.geometry->vertex_identities();
        for (std::size_t v = 0; v < vertices.size(); ++v)
        {
            auto expected =
                exact_point_translated(vertices[v], exact_displacement_difference({}, tile.exact_second_image));
            ASSERT_TRUE(nodes.contains(tile.second_node_gids[v]));
            EXPECT_EQ(nodes.at(tile.second_node_gids[v])->exact_identity, expected);
            EXPECT_EQ(nodes.at(tile.second_node_gids[v])->coordinate, exact_point_display(expected));
        }
        auto restored =
            ::SimpleFluid::Meshes::AMR::decode_adaptive_tile(::SimpleFluid::Meshes::AMR::encode_adaptive_tile(tile));
        EXPECT_EQ(restored.node_gids, tile.node_gids);
        EXPECT_EQ(restored.second_node_gids, tile.second_node_gids);
        EXPECT_EQ(restored.exact_second_image, tile.exact_second_image);
        EXPECT_EQ(restored.geometry->vertex_identities(), tile.geometry->vertex_identities());
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_periodic, &global_periodic);
    EXPECT_EQ(global_periodic, 12);
    for (const auto& [key, leaf] : state->owned())
        EXPECT_TRUE(ExactPlanarFace::cell_shell(shells[key])->equivalent(*kernel->geometry(leaf)));
    auto baseline = periodic_signature(draft);
    EXPECT_EQ(baseline.size(), 51);
    std::vector<std::uint64_t> permuted;
    for (std::uint64_t cell = 0; cell < 8; ++cell)
        if ((cell + 1) % comm->getSize() == std::uint64_t(comm->getRank()))
            permuted.push_back(cell);
    auto detached = source.detach_owned(permuted, {'d'});
    ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy changed(
        std::move(detached.owned), std::move(detached.families), std::move(detached.kernel), comm);
    auto changed_draft = changed.native_draft();
    EXPECT_EQ(changed_draft.global_faces, 24);
    EXPECT_EQ(changed_draft.global_nodes, 27);
    EXPECT_EQ(periodic_signature(changed_draft), baseline);
}
TEST(DistributedCartesianKernel, PeriodicPartnerBlocksRefineMigrateAndCoarsenTogether)
{
    auto config = grid();
    config.root_blocks = {2, 1, 1};
    config.periodic[0] = true;
    ::SimpleFluid::Meshes::AMR::CartesianAMR source(config);
    auto state = hierarchy(source);
    auto comm = Tpetra::getDefaultComm();
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveMark> marks;
    if (comm->getRank() == 0)
        marks.push_back({source.cell_key(0), ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic});
    auto plan = state->plan_adaptation(marks);
    auto refined = state->apply(plan);
    auto oracle = source.adapt({0});
    compare_oracle(*refined, *oracle);
    std::vector<int> destinations;
    for (const auto& [key, leaf] : refined->owned())
    {
        const auto& cell = std::get<::SimpleFluid::Meshes::AMR::CartesianCellAncestry>(key.ancestry);
        destinations.push_back(int((cell.index[0] + 8 * (cell.index[1] + 4 * cell.index[2])) % comm->getSize()));
    }
    auto migration = refined->plan_migration(destinations);
    auto moved = refined->apply(migration);
    compare_oracle(*moved, *oracle);
    auto draft = moved->native_draft();
    EXPECT_EQ(draft.owned.size(), moved->owned().size());
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes> inverse;
    if (comm->getRank() == 0)
        inverse.push_back(::SimpleFluid::Meshes::AMR::adaptive_family_key(
            source.cell_key(0), ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic));
    auto collapse = moved->plan_adaptation({}, inverse);
    auto restored = moved->apply(collapse);
    compare_oracle(*restored, source);
}
TEST(DistributedCartesianKernel, PeriodicUnmatchedBlockGenerationRejectsCollectively)
{
    auto config = grid();
    config.root_blocks = {2, 1, 1};
    config.periodic[0] = true;
    auto kernel = ::SimpleFluid::Meshes::AMR::cartesian_adaptive_kernel(config);
    auto comm = Tpetra::getDefaultComm();
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveLeaf> records;
    for (std::uint64_t k = 0; k < 2; ++k)
        for (std::uint64_t j = 0; j < 2; ++j)
            for (std::uint64_t i = 0; i < 4; ++i)
            {
                auto parent = ::SimpleFluid::Meshes::AMR::cartesian_adaptive_leaf(
                    config, {0, ::SimpleFluid::Meshes::AMR::CartesianCellAncestry{0, {i, j, k}}}, {'d'});
                if (i < 2)
                {
                    auto split = kernel->split(parent, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic);
                    for (auto& child : split.children)
                        if (::SimpleFluid::Meshes::AMR::adaptive_route_hash(
                                ::SimpleFluid::Meshes::AMR::encode_adaptive_key(child.key)) %
                                comm->getSize() ==
                            std::uint64_t(comm->getRank()))
                            records.push_back(std::move(child));
                }
                else if (::SimpleFluid::Meshes::AMR::adaptive_route_hash(
                             ::SimpleFluid::Meshes::AMR::encode_adaptive_key(parent.key)) %
                             comm->getSize() ==
                         std::uint64_t(comm->getRank()))
                    records.push_back(std::move(parent));
            }
    records = kernel->repack(records);
    EXPECT_THROW(::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy(std::move(records), {}, kernel, comm),
        std::invalid_argument);
}
TEST(DistributedCartesianKernel, PeriodicCornerContactsNormalizeAllAxesWithoutPhysicalWelding)
{
    auto config = grid();
    config.periodic = {true, true, true};
    config.origin = {.1, .2, .3};
    config.extent = {.7, .8, .9};
    auto kernel = ::SimpleFluid::Meshes::AMR::cartesian_adaptive_kernel(config);
    auto value = ::SimpleFluid::Meshes::AMR::cartesian_adaptive_leaf(
        config, {0, ::SimpleFluid::Meshes::AMR::CartesianCellAncestry{0, {1, 1, 1}}}, {'d'});
    auto shape = kernel->geometry(value);
    GeometryPoint lower = config.origin, upper{};
    for (std::size_t a = 0; a < 3; ++a)
        upper[a] = config.origin[a] + config.extent[a];
    auto corner = ExactPlanarFace::make(
        std::vector<GeometryPoint>{upper, {lower[0], upper[1], upper[2]}, {upper[0], lower[1], upper[2]}})
                      ->vertex_identities()
                      .front();
    auto contact = kernel->contact_point(value, corner);
    EXPECT_EQ(exact_point_display(contact), lower);
    EXPECT_EQ(exact_point_display(corner), upper);
    EXPECT_NE(contact, corner);
    auto group = kernel->refinement_group({0, ::SimpleFluid::Meshes::AMR::CartesianCellAncestry{1, {0, 0, 0}}},
        ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic);
    EXPECT_EQ(group.size(), 64);
}

} // namespace
