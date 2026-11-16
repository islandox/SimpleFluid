/** Fixed affected contact work under unrelated native provider growth. */
#include "adaptive_storage_test_helpers.hh"
#include "geometry/mesh/AMR/AdaptiveContactSnapshot.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "utils/testing_environment.hh"
#include <gtest/gtest.h>
#include <map>

namespace
{
namespace A = SimpleFluid::Meshes::AMR;
namespace D = A::adaptive_detail;
namespace Fixtures = SimpleFluid::test::adaptive_storage;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using Key = std::pair<std::uintptr_t, D::ContactPayloadRole>;
using Census = std::map<Key, std::size_t>;

Census metadata(const D::DistributedAdaptiveContactHalo& halo)
{
    Census result;
    halo.for_each_borrowed_contact_payload([&](const D::ContactPayloadAllocation& allocation)
        {
            const auto role = allocation.identity.role;
            if (role == D::GeometryContactPayloadRole::ContactSlotBranch ||
                role == D::GeometryContactPayloadRole::ContactSlotLeaf ||
                role == D::GeometryContactPayloadRole::ContactOwnerUsageNode)
                result.emplace(Key{reinterpret_cast<std::uintptr_t>(allocation.identity.address), role},
                    allocation.payload_bytes);
        });
    return result;
}
struct Work
{
    std::size_t shared_bytes = 0, new_bytes = 0, new_pages = 0, owner_work = 0;
    std::size_t catalog_copies = 0, record_copies = 0, index_copies = 0, cover_copies = 0, trace_copies = 0;
};

Work fixed_edit(Fixtures::Provider provider, std::size_t count)
{
    auto input = Fixtures::make_disconnected(provider, count);
    auto comm = Teuchos::rcp(new Teuchos::SerialComm<int>);
    const A::DistributedAdaptiveHierarchy hierarchy(input.owned, {}, input.kernel, comm);
    const auto layout = hierarchy.leaf_layout();
    const auto& source = D::contact_cache(hierarchy);
    const auto original = D::contact_snapshot(hierarchy);
    const auto before = metadata(source);
    const auto used = source.statistics().retained_bytes;
    auto context = original->context();
    ++context.transaction;
    auto trial = source.fork_candidate(context);
    ++context.round;
    auto changed = layout->at(input.changed);
    const std::array retired{D::ContactRetiredLeaf{changed.key, changed.revision}};
    const std::array inserted{changed.key};
    ++changed.revision;
    trial->apply_delta(retired, inserted, context,
        [&](const auto& key) -> const A::AdaptiveLeaf& { return key == changed.key ? changed : layout->at(key); });
    EXPECT_TRUE(trial->required_refinements().empty());
    const auto& stats = trial->statistics();
    EXPECT_EQ(stats.delta_owned_visits, retired.size() + inserted.size());
    EXPECT_EQ(stats.full_owned_scans_after_bootstrap, 0u);
    EXPECT_EQ(source.statistics().retained_bytes, used);
    EXPECT_EQ(metadata(source), before);

    Work result;
    result.owner_work = stats.contact_owner_admission_work;
    result.catalog_copies = stats.contact_catalog_pages_copied;
    result.record_copies = stats.contact_record_pages_copied + stats.contact_row_pages_copied;
    result.index_copies = stats.contact_index_pages_copied;
    result.cover_copies = stats.contact_cover_pages_copied;
    result.trace_copies = stats.contact_trace_pages_copied;
    for (const auto& [key, bytes] : metadata(*trial))
        if (const auto prior = before.find(key); prior != before.end())
        {
            EXPECT_EQ(bytes, prior->second);
            result.shared_bytes += bytes;
        }
        else
        {
            result.new_bytes += bytes;
            ++result.new_pages;
        }
    EXPECT_GT(result.owner_work, 0u);
    EXPECT_GT(result.shared_bytes, result.new_bytes);

    const auto& unaffected = layout->at_ordinal(layout->size() - 1);
    std::vector<A::AdaptiveCellKey> old_neighbors, trial_neighbors;
    original->visit_closed_contacts(unaffected.key, [&](const auto& contact) { old_neighbors.push_back(contact.neighbor); });
    trial->visit_closed_contacts(unaffected.key, [&](const auto& contact) { trial_neighbors.push_back(contact.neighbor); });
    EXPECT_EQ(old_neighbors, trial_neighbors);
    return result;
}
} // namespace

TEST(AdaptiveContactStorageScaling, FixedNativeEditSharesDisconnectedMetadataAndOwnerCredits)
{
    for (const auto provider : Fixtures::providers)
    {
        SCOPED_TRACE(Fixtures::name(provider));
        const auto small = fixed_edit(provider, 48);
        const auto large = fixed_edit(provider, 96);
        EXPECT_GT(large.shared_bytes, small.shared_bytes);
        EXPECT_LE(large.new_pages, small.new_pages + 20);
        EXPECT_LE(large.new_bytes, small.new_bytes + 16 * 1024);
        EXPECT_LE(large.owner_work, small.owner_work + small.owner_work / 2 + 64);
        EXPECT_LE(large.catalog_copies, small.catalog_copies + 8);
        EXPECT_LE(large.record_copies, small.record_copies + 4);
        EXPECT_LE(large.index_copies, small.index_copies + 4);
        EXPECT_LE(large.cover_copies, small.cover_copies + 4);
        EXPECT_LE(large.trace_copies, small.trace_copies + 4);
    }
}
