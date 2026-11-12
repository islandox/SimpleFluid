/** @file testDirectoryPersistence.cc
 * @brief Actual hierarchy projections and independent controller lifetimes.
 */
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/DirectoryFactPages.hh"
#include "geometry/mesh/AMR/DirectoryRegistrationAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <array>
#include <gtest/gtest.h>
#include <map>

namespace
{
namespace AMR = SimpleFluid::Meshes::AMR;
namespace D = AMR::adaptive_detail;
using Access = D::DirectoryRegistrationAccess;
using Hierarchy = AMR::DistributedAdaptiveHierarchy;
using State = std::shared_ptr<const Hierarchy>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

State refine(const State& source, const AMR::AdaptiveCellKey& key, AMR::AdaptiveSplit split)
{
    const std::vector<AMR::AdaptiveMark> marks = source->owned_leaves().contains(key)
                                                     ? std::vector{AMR::AdaptiveMark{key, split}}
                                                     : std::vector<AMR::AdaptiveMark>{};
    return source->apply(source->plan_adaptation(marks));
}
std::uint64_t sum(std::uint64_t local)
{
    std::uint64_t result = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local, &result);
    return result;
}

TEST(DirectoryPersistence, ShiftedObservedGIDsDoNotRewriteUneditedOwnerRevisions)
{
    const auto comm = Tpetra::getDefaultComm();
    AMR::AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {2, 1, 1};
    grid.max_level = 1;
    AMR::CartesianAMR mesh(grid);
    const auto left = mesh.cell_key(0);
    std::vector<AMR::AdaptiveCellKey> right;
    for (std::uint64_t i = 8; i < 16; ++i)
        right.push_back(mesh.cell_key(i));
    std::sort(right.begin(), right.end());
    std::vector<std::uint64_t> selected;
    const int second_owner = comm->getSize() > 1 ? 1 : 0;
    for (std::uint64_t i = 0; i < 16; ++i)
        if (comm->getRank() == (i < 8 ? 0 : second_owner))
            selected.push_back(i);
    auto detached = mesh.detach_owned(selected, {'d'});
    State source = std::make_shared<const Hierarchy>(
        std::move(detached.owned), std::move(detached.families), std::move(detached.kernel), comm);
    const auto old = source->lookup_cells(right);
    std::weak_ptr<const AMR::DistributedAncestryDirectory> old_controller = Access::cell_owner(*source);
    State second = refine(source, left, AMR::AdaptiveSplit::Isotropic);
    const auto current = second->lookup_cells(right);
    EXPECT_EQ(current.size(), right.size());
    for (std::size_t i = 0; i < current.size(); ++i)
    {
        EXPECT_EQ(current[i].key, AMR::encode_adaptive_key(right[i]));
        EXPECT_EQ(current[i].owner, second_owner);
        EXPECT_EQ(current[i].revision, old[i].revision);
        EXPECT_EQ(current[i].address, (comm->getSize() > 1 ? 64 : 0) + i);
    }
    EXPECT_EQ(sum(Access::statistics(Access::cells(*second)).fact_edits), 72u);
    source.reset();
    EXPECT_TRUE(old_controller.expired());
    auto native = AMR::assemble_adaptive_native(*second, second->configuration_descriptor());
    auto third = second->with_partition(native.partition, native.mesh->owned_cell_map());
    EXPECT_EQ(&Access::cells(*third), &Access::cells(*second));
    EXPECT_EQ(&Access::families(*third), &Access::families(*second));
    const auto* second_directory = &Access::cells(*second);
    const std::weak_ptr<const Hierarchy> old_second = second;
    second.reset();
    EXPECT_TRUE(old_second.expired());
    const auto third_values = third->lookup_cells(right);
    for (std::size_t i = 0; i < current.size(); ++i)
        EXPECT_EQ(third_values[i].address, current[i].address);
    EXPECT_EQ(&Access::cells(*third), second_directory);
}

AMR::AdaptiveCellKey product_key(std::uint64_t interval)
{
    return {0, AMR::ProductCellAncestry{{0, 1, {}}, {interval, {}}}};
}

TEST(DirectoryPersistence, ProductEpochProjectsWhileMigrationPreservesTopologyAndCurrentGIDs)
{
    const auto comm = Tpetra::getDefaultComm();
    AMR::ExtrudedAMRGrid grid;
    grid.xy_nodes = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
    grid.base_cells = {{0, 1, 2, 3}};
    grid.z_edges = {0, 1, 2};
    AMR::ExtrudedAMR mesh(grid);
    std::vector<std::uint64_t> selected;
    const int second_owner = comm->getSize() > 1 ? 1 : 0;
    for (std::uint64_t i = 0; i < 2; ++i)
        if (comm->getRank() == (i ? second_owner : 0))
            selected.push_back(i);
    auto detached = mesh.detach_owned(selected, {'p'});
    State source = std::make_shared<const Hierarchy>(
        std::move(detached.owned), std::move(detached.families), std::move(detached.kernel), comm);
    auto first = refine(source, product_key(0), AMR::AdaptiveSplit::Base);
    const auto family_key = AMR::adaptive_family_key(product_key(0), AMR::AdaptiveSplit::Base);
    const auto before = first->lookup_families(std::span(&family_key, 1));
    auto second = refine(first, product_key(1), AMR::AdaptiveSplit::Axial);
    const auto after = second->lookup_families(std::span(&family_key, 1));
    EXPECT_EQ(after.front().revision, before.front().revision + 1);
    EXPECT_EQ(after.front().revision, second->topology_generation());
    EXPECT_EQ(sum(Access::statistics(Access::families(*second)).fact_edits), 1u);
    if (const auto family = second->owned_families().find(family_key))
        EXPECT_NE(family->parent.revision, family->revision);

    std::vector<AMR::AdaptiveCellKey> all;
    for (std::uint64_t base = 0; base < 4; ++base)
        all.push_back({0, AMR::ProductCellAncestry{{0, 1, {base}}, {0, {}}}});
    for (std::uint64_t axial = 0; axial < 2; ++axial)
        all.push_back({0, AMR::ProductCellAncestry{{0, 1, {}}, {1, {axial}}}});
    std::sort(all.begin(), all.end());
    const auto destination = [&](const AMR::AdaptiveCellKey& key)
    {
        const auto at = std::lower_bound(all.begin(), all.end(), key) - all.begin();
        return comm->getSize() == 1 ? 0 : int(at % 2);
    };
    std::vector<int> destinations;
    for (const auto& leaf : second->owned_leaves())
        destinations.push_back(destination(leaf.key));
    auto migrated = second->apply(second->plan_migration(destinations));
    EXPECT_EQ(migrated->topology_generation(), second->topology_generation());
    EXPECT_EQ(migrated->lookup_families(std::span(&family_key, 1)).front().revision, second->topology_generation());
    const auto actual = migrated->lookup_cells(all);
    std::map<AMR::AdaptiveCellKey, std::uint64_t> expected;
    std::uint64_t gid = 0;
    for (int owner = 0; owner < comm->getSize(); ++owner)
        for (const auto& key : all)
            if (destination(key) == owner)
                expected.emplace(key, gid++);
    for (std::size_t i = 0; i < all.size(); ++i)
    {
        EXPECT_EQ(actual[i].owner, destination(all[i]));
        EXPECT_EQ(actual[i].address, expected.at(all[i]));
    }
    const auto values = first->lookup_families(std::span(&family_key, 1));
    EXPECT_EQ(values.front().revision, before.front().revision);
}
TEST(DirectoryPersistence, PartialShardReconciliationRejectsAndDiscardsPrivateRoots)
{
    const auto comm = Tpetra::getDefaultComm();
    constexpr std::size_t cap = 16 * 1024 * 1024;
    std::vector<AMR::AdaptiveBytes> keys;
    std::vector<AMR::AdaptiveOwner> owned;
    for (std::size_t i = 0; i < 1024; ++i)
    {
        keys.push_back({char(i >> 8), char(i & 255), 'd'});
        if (comm->getRank() == 0)
            owned.push_back({keys.back(), 0, 100 + i, 700 + i});
    }
    const AMR::DistributedAncestryDirectory accepted(owned, comm, cap);
    const auto before = accepted.lookup(keys);
    std::vector<D::DirectoryRegistrationView> rows;
    for (const auto& value : before)
        if (D::authority(value.key, comm->getSize()) == comm->getRank())
            rows.push_back({value.key, value.owner, value.revision, value.address});
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b)
        { return std::lexicographical_compare(a.key.begin(), a.key.end(), b.key.begin(), b.key.end()); });
    const auto* source = Access::facts(accepted);
    const auto root = source ? source->root_owner() : nullptr;
    const bool tight_rank = comm->getRank() == 0;
    auto limit = cap;
    D::collective(*comm,
        [&]
        {
            if (!tight_rank)
                return;
            if (!source || rows.size() < 2)
                throw std::logic_error("Directory shard fixture lacks two actual registration facts.");
            rows.front().revision += 10000;
            rows.back().revision += 20000;
            auto measured = std::make_shared<D::Budget>(D::Budget{cap});
            const auto probe = D::DirectoryFactPages::reconcile(source, rows, measured, 0);
            limit = measured->peak - 1;
        });
    auto memory = std::make_shared<D::Budget>(D::Budget{limit});
    std::unique_ptr<D::DirectoryFactPages> candidate;
    std::size_t completed = 0;
    bool rejected = false;
    EXPECT_THROW(D::collective(*comm,
                     [&]
                     {
                         try
                         {
                             candidate = std::make_unique<D::DirectoryFactPages>(
                                 D::DirectoryFactPages::reconcile(source, rows, memory, 0, &completed));
                         }
                         catch (const std::invalid_argument&)
                         {
                             rejected = true;
                             throw;
                         }
                     }),
        std::invalid_argument);
    EXPECT_EQ(rejected, tight_rank);
    if (tight_rank)
        EXPECT_GT(completed, 0u);
    candidate.reset();
    EXPECT_EQ(memory->used, 0u);
    if (source)
        EXPECT_EQ(source->root_owner(), root);
    const auto after = accepted.lookup(keys, true, cap);
    EXPECT_EQ(after.size(), before.size());
    for (std::size_t i = 0; i < std::min(after.size(), before.size()); ++i)
    {
        EXPECT_EQ(after[i].key, before[i].key);
        EXPECT_EQ(after[i].owner, before[i].owner);
        EXPECT_EQ(after[i].revision, before[i].revision);
        EXPECT_EQ(after[i].address, before[i].address);
    }
}

} // namespace
