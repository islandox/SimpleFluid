/** Immutable closed-contact witnesses and copy-on-write routing pages. */
#include "cartesian_incidence_test_helpers.hh"
#include "geometry/mesh/AMR/AdaptiveContactSnapshot.hh"
#include "geometry/mesh/AMR/ContactSlotIndex.hh"
#include "geometry/mesh/AMR/ContactSlotPages.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include "geometry/mesh/AMR/DistributedMixedKernel.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "geometry/unitTests/mixed_amr_helpers.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_DefaultSerialComm.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <map>
#include <numeric>
#include <tuple>

namespace
{
using namespace cartesian_incidence_test;
using Hierarchy = DistributedAdaptiveHierarchy;
using Owner = std::shared_ptr<const Hierarchy>;
using Snapshot = std::shared_ptr<const AdaptiveContactSnapshot>;
using Witness = std::tuple<AdaptiveCellKey, int, std::uint64_t, AdaptiveBytes, ContactPointIdentity,
    ContactPointIdentity, ContactPointIdentity>;
using Contacts = std::map<AdaptiveCellKey, std::vector<Witness>>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

int initial_owner(const AdaptiveCellKey& key, int ranks)
{
    return ranks == 1 ? 0 : int(std::get<CartesianCellAncestry>(key.ancestry).index[0]);
}
Owner source(AMRGrid config)
{
    const auto comm = Tpetra::getDefaultComm();
    std::vector<AdaptiveLeaf> local;
    for (auto& leaf : leaves(config))
        if (initial_owner(leaf.key, comm->getSize()) == comm->getRank())
            local.push_back(std::move(leaf));
    return std::make_shared<const Hierarchy>(std::move(local), std::vector<AdaptiveFamily>{},
        cartesian_adaptive_kernel(config), comm);
}
std::vector<Witness> contacts(const AdaptiveContactSnapshot& snapshot, const AdaptiveCellKey& key)
{
    std::vector<Witness> result;
    snapshot.visit_closed_contacts(key, [&](const auto& witness)
        {
            result.emplace_back(witness.neighbor, witness.owner, witness.revision,
                AdaptiveBytes(witness.domain.begin(), witness.domain.end()),
                witness.normalized, witness.local_physical, witness.neighbor_physical);
        });
    std::sort(result.begin(), result.end());
    return result;
}
Contacts all_contacts(const Hierarchy& hierarchy, const AdaptiveContactSnapshot& snapshot)
{
    Contacts result;
    for (const auto& leaf : hierarchy.owned_leaves())
        result.emplace(leaf.key, contacts(snapshot, leaf.key));
    return result;
}
void expect_contacts(const AdaptiveContactSnapshot& snapshot, const Contacts& expected)
{
    for (const auto& [key, witnesses] : expected)
        EXPECT_EQ(contacts(snapshot, key), witnesses);
}
std::array<long double, 3> corner(const AdaptiveCellKey& key, unsigned mask)
{
    const auto& cell = std::get<CartesianCellAncestry>(key.ancestry);
    std::array<long double, 3> result;
    for (unsigned axis = 0; axis < result.size(); ++axis)
        result[axis] = (cell.index[axis] + ((mask >> axis) & 1)) / 2.0L;
    return result;
}
ContactPointIdentity physical(const std::array<long double, 3>& coordinates)
{
    return PLIC::details::GeometryIdentityAccess::point(coordinates);
}
ContactPointIdentity normalized(std::array<long double, 3> coordinates)
{
    if (coordinates[0] == 1)
        coordinates[0] = 0;
    return physical(coordinates);
}
Owner refine(const Owner& state, const AdaptiveCellKey& key)
{
    const std::array marks{AdaptiveMark{key, AdaptiveSplit::Isotropic}};
    const auto requested = std::span(marks).first(state->owned_leaves().contains(key) ? 1 : 0);
    return state->apply(state->plan_adaptation(requested));
}
std::vector<std::uint64_t> all_indices(std::size_t count)
{
    std::vector<std::uint64_t> result(count);
    std::iota(result.begin(), result.end(), 0);
    return result;
}
AdaptiveBootstrap product()
{
    const ExtrudedAMR mesh(SimpleFluid::test::mixed_product_grid(true));
    return mesh.detach_owned(all_indices(mesh.geometry()->num_cells()), {'p'});
}
AdaptiveBootstrap convex()
{
    const auto mesh = SimpleFluid::test::mixed_polyhedron();
    return mesh->detach_owned(all_indices(mesh->geometry()->num_cells()), {'c'});
}
AdaptiveBootstrap stationary()
{
    const CartesianAMR mesh(grid());
    return detach_static_participant(mesh.geometry()->regions().front(), 71, 9, 5,
        all_indices(mesh.geometry()->num_cells()), {'s'});
}
AdaptiveBootstrap mixed()
{
    const MixedAMR mesh(SimpleFluid::test::mixed_joined_grid());
    return mesh.detach_owned(all_indices(mesh.geometry()->num_cells()), {'m'});
}
} // namespace

TEST(AdaptiveContactSnapshot, PeriodicContactsPreserveEveryExactPhysicalWitness)
{
    auto config = grid();
    config.periodic[0] = true;
    auto hierarchy = source(config);
    const auto snapshot = contact_snapshot(*hierarchy);
    ASSERT_TRUE(snapshot);
    const auto comm = hierarchy->communicator();
    auto all = leaves(config);
    std::size_t mismatched_images = 0;
    for (const auto& leaf : hierarchy->owned_leaves())
    {
        std::vector<Witness> expected;
        for (const auto& other : all)
        {
            if (other.key == leaf.key)
                continue;
            for (unsigned local = 0; local < 8; ++local)
                for (unsigned remote = 0; remote < 8; ++remote)
                {
                    const auto first = corner(leaf.key, local), second = corner(other.key, remote);
                    const auto common = normalized(first);
                    if (common != normalized(second))
                        continue;
                    mismatched_images += first != second;
                    expected.emplace_back(other.key, initial_owner(other.key, comm->getSize()), other.revision,
                        other.domain, common, physical(first), physical(second));
                }
        }
        std::sort(expected.begin(), expected.end());
        EXPECT_EQ(contacts(*snapshot, leaf.key), expected);
        snapshot->visit_closed_contacts(leaf.key,
            [&](const auto& witness)
            {
                if (const auto* owned = hierarchy->owned_leaves().find(witness.neighbor))
                {
                    EXPECT_EQ(&witness.neighbor, &owned->key);
                    EXPECT_EQ(witness.domain.data(), owned->domain.data());
                }
            });
    }
    if (comm->getRank() < std::min(2, comm->getSize()))
        EXPECT_GT(mismatched_images, 0u);
    else
    {
        EXPECT_TRUE(hierarchy->owned_leaves().empty());
        EXPECT_THROW(snapshot->visit_closed_contacts(all.front().key, [](const auto&) {}), std::out_of_range);
    }
}

TEST(AdaptiveContactSnapshot, ThreeGenerationsEscapeWithoutRetainingAnAncestorLayout)
{
    auto config = grid();
    config.periodic[0] = true;
    auto first = source(config);
    auto original = contact_snapshot(*first);
    ASSERT_TRUE(original);
    const auto before = all_contacts(*first, *original);
    std::weak_ptr first_layout(first->leaf_layout());
    std::weak_ptr first_hierarchy(first);

    const AdaptiveCellKey low{config.region_id, CartesianCellAncestry{0, {0, 0, 0}}};
    auto second = refine(first, low);
    EXPECT_NE(first->leaf_layout(), second->leaf_layout());
    EXPECT_EQ(second->topology_generation(), first->topology_generation() + 1);
    auto middle = contact_snapshot(*second);
    ASSERT_TRUE(middle);
    const auto during = all_contacts(*second, *middle);
    std::weak_ptr second_layout(second->leaf_layout());
    const std::array families{adaptive_family_key(low, AdaptiveSplit::Isotropic)};
    const auto requested = std::span(families).first(second->families().contains(families.front()) ? 1 : 0);
    auto third = second->apply(second->plan_adaptation({}, requested));
    EXPECT_NE(second->leaf_layout(), third->leaf_layout());
    EXPECT_EQ(third->topology_generation(), second->topology_generation() + 1);
    auto current = contact_snapshot(*third);
    ASSERT_TRUE(current);
    const auto after = all_contacts(*third, *current);

    first.reset();
    second.reset();
    EXPECT_TRUE(first_hierarchy.expired());
    EXPECT_FALSE(first_layout.expired());
    EXPECT_FALSE(second_layout.expired());
    expect_contacts(*original, before);
    expect_contacts(*middle, during);
    expect_contacts(*current, after);
    original.reset();
    middle.reset();
    EXPECT_TRUE(first_layout.expired());
    EXPECT_TRUE(second_layout.expired());
    third.reset();
    expect_contacts(*current, after);
}

TEST(AdaptiveContactSnapshot, ExactForkPageCreditAndOneByteShortLeaveSourceUnchanged)
{
    auto source_budget = std::make_shared<Budget>(Budget{ample});
    ContactSlotPages<std::uint64_t> pages(source_budget);
    for (std::size_t i = 0; i < 512; ++i)
        pages.edit(i) = i + 1;
    const auto* original = &pages[0];
    const auto* untouched = &pages[257];
    const auto source_used = source_budget->used;
    auto perform = [&](std::size_t limit)
    {
        auto budget = std::make_shared<Budget>(Budget{limit});
        ContactSlotPages<std::uint64_t> candidate(pages, budget);
        candidate.edit(0) = 1000;
        EXPECT_EQ(candidate[0], 1000u);
        EXPECT_NE(&candidate[0], original);
        EXPECT_EQ(&candidate[257], untouched);
        EXPECT_LT(candidate.copied_pages(), 6u);
        const auto peak = budget->peak;
        candidate.finish_overlap();
        EXPECT_EQ(budget->used, candidate.storage_bytes());
        return peak;
    };
    const auto exact = perform(ample);
    EXPECT_EQ(perform(exact), exact);
    EXPECT_THROW((void) perform(exact - 1), std::invalid_argument);
    EXPECT_EQ(source_budget->used, source_used);
    EXPECT_EQ(&pages[0], original);
    EXPECT_EQ(pages[0], 1u);
}

TEST(AdaptiveContactSnapshot, ExactWitnessesMatchTheIndependentCanonicalRowsForEveryProvider)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto serial = Teuchos::rcp(new Teuchos::SerialComm<int>);
    const std::array fixtures{std::pair{"Product", &product}, std::pair{"Convex", &convex},
        std::pair{"Static", &stationary}, std::pair{"Mixed", &mixed}};
    for (const auto& [name, fixture] : fixtures)
    {
        SCOPED_TRACE(name);
        auto input = fixture();
        const auto rows = bootstrap_adaptive_contact_rows(input.owned, *input.kernel, serial);
        ASSERT_EQ(rows.size(), input.owned.size());
        std::map<AdaptiveCellKey, int> owners;
        std::vector<AdaptiveLeaf> local;
        for (std::size_t i = 0; i < input.owned.size(); ++i)
        {
            // Leave ranks above two empty while still exercising distributed
            // owner facts and periodic normalization on the other ranks.
            const auto owner = int(i % std::min(2, comm->getSize()));
            owners.emplace(input.owned[i].key, owner);
            if (owner == comm->getRank())
                local.push_back(std::move(input.owned[i]));
        }
        ASSERT_TRUE(input.families.empty());
        const Hierarchy hierarchy(std::move(local), {}, input.kernel, comm);
        const auto snapshot = contact_snapshot(hierarchy);
        ASSERT_TRUE(snapshot);
        for (const auto& row : rows)
        {
            if (owners.at(row.key) != comm->getRank())
                continue;
            std::vector<Witness> expected;
            for (const auto& other : rows)
                if (other.key != row.key && other.domain == row.domain)
                    for (const auto left : row.points)
                        for (const auto right : other.points)
                            if (left.normalized == right.normalized)
                                expected.emplace_back(other.key, owners.at(other.key), other.revision,
                                    other.domain, left.normalized, left.physical, right.physical);
            std::sort(expected.begin(), expected.end());
            EXPECT_EQ(contacts(*snapshot, row.key), expected);
        }
    }
}

TEST(AdaptiveContactSnapshot, ExactSealCreditDoesNotChangeQueriesOnRejection)
{
    auto config = grid();
    config.periodic[0] = true;
    const auto hierarchy = source(config);
    const auto layout = hierarchy->leaf_layout();
    const std::vector<AdaptiveLeaf> input(layout->begin(), layout->end());
    const auto kernel = cartesian_adaptive_kernel(config);
    DistributedAdaptiveContactHalo halo(*kernel, hierarchy->communicator(), {});
    halo.bootstrap_canonical(input, [&](const auto& key) -> const AdaptiveLeaf& { return layout->at(key); }, layout);
    EXPECT_TRUE(halo.required_refinements().empty());
    const auto used = halo.allocation().used;
    const auto independent = source(config);
    ASSERT_EQ(independent->leaf_layout()->size(), layout->size());
    const std::shared_ptr<const AdaptiveLeafLayout> nonowner(layout.get(), [](const auto*) {});
    EXPECT_THROW((void) halo.seal_contacts(independent->leaf_layout()), std::invalid_argument);
    EXPECT_THROW((void) halo.seal_contacts(nonowner), std::invalid_argument);
    EXPECT_EQ(halo.allocation().used, used);
    const auto seal_bytes = sizeof(AdaptiveContactSnapshot) + 128;
    halo.allocation().limit = used + seal_bytes - 1;
    EXPECT_THROW((void) halo.seal_contacts(layout), std::invalid_argument);
    EXPECT_EQ(halo.allocation().used, used);
    const auto source_view = contact_snapshot(*hierarchy);
    ASSERT_TRUE(source_view);
    const auto expected = all_contacts(*hierarchy, *source_view);
    for (const auto& [key, witnesses] : expected)
    {
        std::size_t count = 0;
        halo.visit_closed_contacts(key, [&](const auto&) { ++count; });
        EXPECT_EQ(count, witnesses.size());
    }
    halo.allocation().limit = used + seal_bytes;
    const auto sealed = halo.seal_contacts(layout);
    EXPECT_EQ(halo.allocation().used, used + seal_bytes);
    expect_contacts(*sealed, expected);
}


TEST(AdaptiveContactSnapshot, PersistentIndexSplitsOnlyOneCollisionPath)
{
    auto first_budget = std::make_shared<Budget>(Budget{ample});
    ContactSlotIndex first(first_budget);
    const auto digest = [](std::size_t id) -> std::uint64_t { return id << 4; };
    // Sixteen cells share the low bucket and untouched groups spread across
    // the other buckets. Add a seventeenth cell to just that full bucket.
    for (std::size_t i = 0; i < 256; ++i)
        first.insert(i, [&](std::size_t id) { return digest(id / 16) + id % 16; });
    const auto before = first_budget->used;
    auto budget = std::make_shared<Budget>(Budget{ample});
    ContactSlotIndex candidate(first, budget);
    const auto keys = [&](std::size_t id)
    { return id == 256 ? digest(16) : digest(id / 16) + id % 16; };
    candidate.insert(256, keys);
    EXPECT_EQ(candidate.split_records(), 16u);
    EXPECT_LT(candidate.copied_pages(), 8u);
    EXPECT_EQ(candidate.find(keys(256), [](auto id) { return id == 256; }), 256u);
    EXPECT_FALSE(first.find(keys(256), [](auto id) { return id == 256; }));
    EXPECT_EQ(first_budget->used, before);
    candidate.erase(keys(0), 0);
    EXPECT_FALSE(candidate.find(keys(0), [](auto id) { return id == 0; }));
    EXPECT_EQ(first.find(keys(0), [](auto id) { return id == 0; }), 0u);
    candidate.finish_overlap();
    EXPECT_EQ(budget->used, candidate.storage_bytes());
}

TEST(AdaptiveContactSnapshot, PersistentIndexConfirmsCompleteEqualityAfterFullDigestCollision)
{
    auto budget = std::make_shared<Budget>(Budget{ample});
    ContactSlotIndex index(budget);
    const auto same_digest = [](std::size_t) { return std::uint64_t(0); };
    for (std::size_t i = 0; i < 49; ++i)
        index.insert(i, same_digest);
    for (std::size_t i = 16; i < 32; ++i)
        index.erase(0, i);
    for (std::size_t i = 0; i < 49; ++i)
    {
        const auto found = index.find(0, [&](auto candidate) { return candidate == i; });
        if (i >= 16 && i < 32)
            EXPECT_FALSE(found);
        else
            EXPECT_EQ(found, i);
    }
    EXPECT_FALSE(index.find(0, [](auto candidate) { return candidate == 49; }));
}
