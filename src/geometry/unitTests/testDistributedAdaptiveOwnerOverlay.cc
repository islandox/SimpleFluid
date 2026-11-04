/** Independent owner-delta maps, tombstones, leases and collective rollback. */
#include "geometry/mesh/AMR/DistributedAdaptiveOwnerOverlay.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <gtest/gtest.h>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace
{
namespace AMR = SimpleFluid::Meshes::AMR;
namespace D = AMR::adaptive_detail;
constexpr std::size_t ample = 8 * 1024 * 1024;
constexpr auto absent_address = std::numeric_limits<std::uint64_t>::max();
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using Observation = std::tuple<AMR::AdaptiveBytes, int, std::uint64_t, std::uint64_t>;

AMR::AdaptiveBytes key(std::uint64_t value)
{
    // Directory keys are opaque. This deliberately does not use the production
    // ancestry codec to derive an expected owner or revision.
    AMR::AdaptiveBytes result{'k'};
    for (unsigned byte = 0; byte < 8; ++byte)
        result.push_back(char(value >> (8 * byte)));
    return result;
}
std::vector<AMR::AdaptiveOwner> local(std::initializer_list<AMR::AdaptiveOwner> entries)
{
    std::vector<AMR::AdaptiveOwner> result;
    for (const auto& entry : entries)
        if (entry.owner == Tpetra::getDefaultComm()->getRank())
            result.push_back(entry);
    return result;
}
std::vector<Observation> observe(std::span<const AMR::AdaptiveOwner> entries)
{
    std::vector<Observation> result;
    for (const auto& entry : entries)
        result.emplace_back(entry.key, entry.owner, entry.revision, entry.address);
    return result;
}
std::vector<Observation> lookup(
    const D::DistributedAdaptiveOwnerOverlay& overlay, std::span<const AMR::AdaptiveBytes> keys, bool required = true)
{
    const auto lease = overlay.lookup(keys, required);
    return observe(lease.values());
}
template<class Operation> void reject_collectively(Operation&& operation, std::string_view reason)
{
    int rejected = 0, matching_reason = 0;
    try
    {
        operation();
    }
    catch (const std::invalid_argument& error)
    {
        rejected = 1;
        matching_reason = std::string_view(error.what()).find(reason) != std::string_view::npos;
    }
    catch (const std::exception& error)
    {
        ADD_FAILURE() << "Unexpected exception type: " << error.what();
    }
    int all_rejected = 0, any_matching_reason = 0;
    const auto comm = Tpetra::getDefaultComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &rejected, &all_rejected);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &matching_reason, &any_matching_reason);
    EXPECT_EQ(all_rejected, 1);
    EXPECT_EQ(any_matching_reason, 1) << "No rejecting rank reported the expected cause: " << reason;
}

TEST(DistributedAdaptiveOwnerOverlay, TombstonesSuppressBaseFallbackAndSameRevisionCanChangeOwner)
{
    const auto comm = Tpetra::getDefaultComm();
    const int other = std::min(1, comm->getSize() - 1);
    const auto owned = local({{key(1), 0, 7, 101}, {key(2), other, 9, 202}});
    const AMR::DistributedAncestryDirectory base(owned, comm, ample);
    const std::array keys{key(2), key(1), key(2)};
    const std::vector<Observation> accepted{{key(2), other, 9, 202}, {key(1), 0, 7, 101}, {key(2), other, 9, 202}};
    EXPECT_EQ(observe(base.lookup(keys)), accepted);
    const auto base_entries = base.local_entries(), base_bytes = base.storage_bytes();
    D::Budget budget{ample};
    constexpr std::size_t caller_live = 4093;
    budget.add(caller_live);
    {
        D::DistributedAdaptiveOwnerOverlay candidate(base, comm, budget);
        EXPECT_EQ(lookup(candidate, keys), accepted);
        candidate.apply(local({{key(1), 0, 7}}), {});
        const std::vector<Observation> retired{
            {key(2), other, 9, 202}, {key(1), -1, 7, absent_address}, {key(2), other, 9, 202}};
        EXPECT_EQ(lookup(candidate, keys, false), retired);
        const auto retained = budget.used;
        reject_collectively([&] { (void) candidate.lookup(keys); }, "Missing adaptive overlay owner");
        EXPECT_EQ(budget.used, retained);
        // Reappearance deliberately preserves the revision. Ownership is an
        // independent routing fact, not a geometry-revision invalidation.
        candidate.apply({}, local({{key(1), other, 7}}));
        const std::vector<Observation> moved{
            {key(2), other, 9, 202}, {key(1), other, 7, absent_address}, {key(2), other, 9, 202}};
        EXPECT_EQ(lookup(candidate, keys), moved);
        candidate.apply(local({{key(2), other, 9}}), local({{key(2), 0, 9}}));
        const std::vector<Observation> moved_both{
            {key(2), 0, 9, absent_address}, {key(1), other, 7, absent_address}, {key(2), 0, 9, absent_address}};
        EXPECT_EQ(lookup(candidate, keys), moved_both);
        EXPECT_EQ(observe(base.lookup(keys)), accepted);
        EXPECT_EQ(base.local_entries(), base_entries);
        EXPECT_EQ(base.storage_bytes(), base_bytes);
    }
    EXPECT_EQ(budget.used, caller_live);
    EXPECT_EQ(observe(base.lookup(keys)), accepted);
}

TEST(DistributedAdaptiveOwnerOverlay, DuplicateAndStaleDeltasRejectAtomicallyAndNextDeltaRecovers)
{
    const auto comm = Tpetra::getDefaultComm();
    const int other = std::min(1, comm->getSize() - 1);
    const auto owned = local({{key(1), 0, 7, 101}, {key(2), other, 9, 202}});
    const AMR::DistributedAncestryDirectory base(owned, comm, ample);
    const std::array keys{key(1), key(2), key(3)};
    const auto accepted = observe(base.lookup(keys, false));
    D::Budget budget{ample};
    {
        D::DistributedAdaptiveOwnerOverlay candidate(base, comm, budget);
        candidate.apply(local({{key(1), 0, 7}}), local({{key(1), other, 7}}));
        const auto before = lookup(candidate, keys, false);
        for (unsigned defect = 0; defect < 5; ++defect)
        {
            SCOPED_TRACE(defect);
            std::vector<AMR::AdaptiveOwner> retired, inserted;
            std::string_view reason = "Duplicate";
            if (defect == 0)
            {
                retired = local({{key(1), other, 8}});
                inserted = local({{key(3), 0, 11}});
                reason = "Stale";
            }
            else if (defect == 1)
            {
                // In serial the stale revision retains the same failure oracle.
                retired = local({{key(1), 0, comm->getSize() == 1 ? 8u : 7u}});
                inserted = local({{key(3), other, 11}});
                reason = "Stale";
            }
            else if (defect == 2)
                inserted = local({{key(2), other, 9}, {key(3), 0, 11}});
            else if (defect == 3)
            {
                retired = local({{key(1), other, 7}, {key(1), other, 7}});
                inserted = local({{key(3), 0, 11}});
            }
            else
                inserted = local({{key(3), 0, 11}, {key(3), other, 11}});
            const auto retained = budget.used;
            reject_collectively([&] { candidate.apply(retired, inserted); }, reason);
            EXPECT_EQ(budget.used, retained);
            EXPECT_EQ(lookup(candidate, keys, false), before);
            EXPECT_EQ(observe(base.lookup(keys, false)), accepted);
        }
        candidate.apply({}, local({{key(3), other, 11}}));
        const std::vector<Observation> expected{
            {key(1), other, 7, absent_address}, {key(2), other, 9, 202}, {key(3), other, 11, absent_address}};
        EXPECT_EQ(lookup(candidate, keys), expected);
    }
    EXPECT_EQ(budget.used, 0u);
    EXPECT_EQ(observe(base.lookup(keys, false)), accepted);
}

TEST(DistributedAdaptiveOwnerOverlay, RankLocalSharedBudgetFailurePreservesPublishedDeltaAndBase)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto owned = local({{key(1), 0, 7, 101}});
    const AMR::DistributedAncestryDirectory base(owned, comm, ample);
    const std::array keys{key(1), key(2), key(3)};
    const auto accepted = observe(base.lookup(keys, false));
    D::Budget budget{ample};
    constexpr std::size_t caller_live = 1223;
    budget.add(caller_live);
    {
        D::DistributedAdaptiveOwnerOverlay candidate(base, comm, budget);
        candidate.apply({}, local({{key(2), 0, 3}}));
        const auto before = lookup(candidate, keys, false);
        const auto retained = budget.used;
        if (comm->getRank() == 0)
            budget.limit = retained + 1;
        const auto retire = local({{key(1), 0, 7}});
        const auto insert = local({{key(3), 0, 11}});
        reject_collectively([&] { candidate.apply(retire, insert); }, "budget exhausted");
        budget.limit = ample;
        EXPECT_EQ(budget.used, retained);
        EXPECT_EQ(lookup(candidate, keys, false), before);
        EXPECT_EQ(observe(base.lookup(keys, false)), accepted);
        candidate.apply(retire, insert);
        const std::vector<Observation> expected{
            {key(1), -1, 7, absent_address}, {key(2), 0, 3, absent_address}, {key(3), 0, 11, absent_address}};
        EXPECT_EQ(lookup(candidate, keys, false), expected);
    }
    EXPECT_EQ(budget.used, caller_live);
    EXPECT_EQ(observe(base.lookup(keys, false)), accepted);
}

TEST(DistributedAdaptiveOwnerOverlay, LookupLeaseChargesDuplicateResultsUntilItsMovedOwnerDies)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto owned = local({{key(1), 0, 7, 101}});
    const AMR::DistributedAncestryDirectory base(owned, comm, ample);
    D::Budget budget{ample};
    budget.add(233);
    {
        D::DistributedAdaptiveOwnerOverlay candidate(base, comm, budget);
        const std::array keys{key(1), key(1), key(1)};
        const auto retained = budget.used;
        {
            auto first = candidate.lookup(keys);
            std::size_t expected_charge = first.values().capacity() * sizeof(AMR::AdaptiveOwner);
            for (const auto& value : first.values())
                expected_charge += value.key.capacity();
            EXPECT_EQ(budget.used, retained + expected_charge);
            auto moved = std::move(first);
            EXPECT_EQ(budget.used, retained + expected_charge);
            EXPECT_EQ(observe(moved.values()),
                (std::vector<Observation>{{key(1), 0, 7, 101}, {key(1), 0, 7, 101}, {key(1), 0, 7, 101}}));
        }
        EXPECT_EQ(budget.used, retained);
        candidate.apply({}, {});
        EXPECT_EQ(budget.used, retained);
    }
    EXPECT_EQ(budget.used, 233u);
}

TEST(DistributedAncestryRegistrationLease, EmptyDirectoryAdmitsExactlyItsObjectAndRejectsOneByteShort)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto exact = sizeof(AMR::DistributedAncestryDirectory);
    const AMR::DistributedAncestryDirectory accepted({}, comm, ample, exact);
    EXPECT_EQ(accepted.local_entries(), 0u);
    EXPECT_EQ(accepted.storage_bytes(), exact);
    reject_collectively(
        [&] { AMR::DistributedAncestryDirectory rejected({}, comm, ample, exact - 1); }, "budget exhausted");
    EXPECT_EQ(accepted.local_entries(), 0u);
    EXPECT_EQ(accepted.storage_bytes(), exact);
    EXPECT_TRUE(accepted.lookup({}).empty());
    EXPECT_TRUE(accepted.lookup({}, true, ample).empty());
}

TEST(DistributedAncestryRegistrationLease, RegistrationLeaseDoesNotBecomeFutureLookupLimit)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto owned = local({{key(1), 0, 7, 101}});
    constexpr std::size_t registration_cap = 16 * 1024;
    const AMR::DistributedAncestryDirectory leased(owned, comm, ample, registration_cap);
    const AMR::DistributedAncestryDirectory legacy(owned, comm, ample);
    const AMR::DistributedAncestryDirectory explicit_default(owned, comm, ample, std::nullopt);
    // The result records alone exceed the earlier registration lease. No
    // transport-size estimate is needed for this independent lower bound.
    const std::size_t count = registration_cap / sizeof(AMR::AdaptiveOwner) + 1;
    const std::vector<AMR::AdaptiveBytes> keys(count, key(1));
    const std::vector<Observation> expected(count, Observation{key(1), 0, 7, 101});
    EXPECT_GT(count * sizeof(AMR::AdaptiveOwner), registration_cap);
    EXPECT_EQ(observe(leased.lookup(keys)), expected);
    EXPECT_EQ(observe(leased.lookup(keys, true, ample)), expected);
    EXPECT_EQ(observe(legacy.lookup(keys)), expected);
    EXPECT_EQ(observe(explicit_default.lookup(keys)), expected);
    const auto entries = leased.local_entries(), bytes = leased.storage_bytes();
    reject_collectively([&] { (void) leased.lookup(keys, true, registration_cap); }, "transient lease");
    EXPECT_EQ(leased.local_entries(), entries);
    EXPECT_EQ(leased.storage_bytes(), bytes);
    EXPECT_EQ(observe(leased.lookup(keys)), expected);
}

TEST(DistributedAncestryRegistrationLease, DifferentRankCapsAreAllowedButPresenceMustAgree)
{
    const auto comm = Tpetra::getDefaultComm();
    const AMR::AdaptiveOwner own{key(std::uint64_t(comm->getRank())), comm->getRank(),
        std::uint64_t(17 + comm->getRank()), std::uint64_t(101 + comm->getRank())};
    const std::array owned{own};
    const auto cap = std::size_t(16 * 1024 + 1024 * comm->getRank());
    const AMR::DistributedAncestryDirectory accepted(owned, comm, ample, cap);
    std::vector<AMR::AdaptiveBytes> keys;
    std::vector<Observation> expected;
    for (int rank = comm->getSize() - 1; rank >= 0; --rank)
    {
        keys.push_back(key(std::uint64_t(rank)));
        expected.emplace_back(key(std::uint64_t(rank)), rank, std::uint64_t(17 + rank), std::uint64_t(101 + rank));
    }
    EXPECT_EQ(observe(accepted.lookup(keys)), expected);
    const auto entries = accepted.local_entries(), bytes = accepted.storage_bytes();
    if (comm->getSize() > 1)
    {
        const std::optional<std::size_t> inconsistent = comm->getRank() == 0 ? std::optional(cap) : std::nullopt;
        reject_collectively(
            [&] { AMR::DistributedAncestryDirectory rejected(owned, comm, ample, inconsistent); }, "presence differs");
    }
    for (const auto invalid : {std::size_t(0), ample + 1})
    {
        const auto local_cap = comm->getRank() == 0 ? invalid : cap;
        reject_collectively(
            [&] { AMR::DistributedAncestryDirectory rejected(owned, comm, ample, local_cap); }, "Invalid directory");
    }
    EXPECT_EQ(accepted.local_entries(), entries);
    EXPECT_EQ(accepted.storage_bytes(), bytes);
    EXPECT_EQ(observe(accepted.lookup(keys)), expected);
}
} // namespace
