/** Independent concrete allocation and map-profile tests for the private output route. */
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/NativePublicationViewData.hh"
#include "parallel/details/NativeTpetraMapAdmission.hh"
#include "pipeline_allocation_operators.inc"
#include "utils/testing_environment.hh"
#include <Kokkos_Core.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_Details_Hash.hpp>
#include <gtest/gtest.h>
namespace
{
using namespace SimpleFluid;
namespace A = Meshes::AMR;
namespace D = A::adaptive_detail;
namespace N = geometry_detail;
using Maps = N::NativeTpetraMapAdmission;
using GO = DefaultTpetraTypes::global_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
struct Trace
{
    Trace() { pipeline_trace::begin(); }
    ~Trace() { pipeline_trace::recording.store(false); }
};
A::AdaptiveCellKey key(std::uint64_t ordinal)
{
    return {7, A::ConvexCellAncestry{ordinal, {1, 2, 3, 4, 5, 6}}};
}
TEST(NativePublicationOutputAccounting, KeyOnlyCatalogCopiesActualBuffersAndKeepsUnchangedReferences)
{
    ASSERT_TRUE(N::native_constructor_capacity_profile_supported());
    N::NativeConstructionAdmission admission(N::NativeConstructionCounterAccess::create<D::Budget>(32 * 1024 * 1024));
    D::NativePublicationKeyCatalog first;
    const auto before = admission.statistics().actual_live;
    Trace trace;
    for (std::uint64_t i = 0; i < 193; ++i)
        first = first.with_key(key(i), admission);
    const auto* retained = &first.at_ordinal(100);
    auto second = first.with_key(key(500), admission);
    auto third = second.without_key(key(20), admission);
    const auto raw = pipeline_trace::summary();
    const auto exact = admission.statistics().actual_live - before;
    pipeline_trace::Pause pause;
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(raw.all_live, exact); // independent global-new census, no production visitor.
    EXPECT_EQ(&second.at_ordinal(*second.ordinal_of(key(100))), retained);
    EXPECT_EQ(&third.at_ordinal(*third.ordinal_of(key(100))), retained);
    EXPECT_EQ(first.size(), 193u);
    EXPECT_EQ(second.size(), 194u);
    EXPECT_EQ(third.size(), 193u);
    EXPECT_FALSE(third.ordinal_of(key(20)));
    EXPECT_EQ(first.at_ordinal(100), key(100));
}
TEST(NativePublicationOutputAccounting, DetachedKeyOwnerExpiresAtItsLastRealCatalogEdge)
{
    ASSERT_TRUE(N::native_constructor_capacity_profile_supported());
    N::NativeConstructionAdmission admission(N::NativeConstructionCounterAccess::create<D::Budget>(1024 * 1024));
    auto original = key(17);
    const auto expected = original;
    auto capsule = D::NativePublicationKeyCatalog::capture(original, admission);
    std::weak_ptr<const D::NativeRemoteKey> weak = capsule;
    original = {};
    EXPECT_EQ(capsule->key(), expected);
    capsule.reset();
    EXPECT_TRUE(weak.expired());
    // The weak target control stays paid until its actual weak lifetime ends.
    const auto held = admission.statistics().actual_live;
    weak.reset();
    EXPECT_LT(admission.statistics().actual_live, held);
}
TEST(NativePublicationOutputAccounting, InstalledMapProfileIncludesInclusiveSecondRunAndActualLibraryPeak)
{
    ASSERT_NO_THROW(Maps::require_profile()); // a required profile cannot pass by skipping.
    const auto comm = Tpetra::getDefaultComm();
    const GO base = comm->getRank() * 100;
    const std::array<GO, 5> gids{base, base + 1, base + 5, base + 6, base + 12};
    const auto plan = Maps::plan(Maps::Kind::Noncontiguous, gids);
    EXPECT_EQ(plan.n, 5u);
    EXPECT_EQ(plan.m, 3u);
    EXPECT_EQ(plan.k, 2u); // run [5,6] retains its inclusive last endpoint.
    using Hash = Tpetra::Details::Hash<GO, DefaultTpetraTypes::device_type>;
    EXPECT_EQ(plan.b, std::size_t(Hash::getRecommendedSize(2)));
    EXPECT_GT(plan.directory_object, 0u);
    EXPECT_GT(plan.rcp_node, 0u);
    EXPECT_GT(plan.retained_kokkos_headers, 0u);
    EXPECT_GT(plan.retained_kokkos_records, 0u);
    EXPECT_GT(plan.temporary_minmax, 0u);
    EXPECT_GT(plan.temporary_comm_helpers, 0u);
    N::NativeConstructionAdmission admission(N::NativeConstructionCounterAccess::create<D::Budget>(64 * 1024 * 1024));
    const auto before = admission.statistics();
    const auto initial_used = admission.counters().used;
    Trace trace;
    std::shared_ptr<Maps::Prepared> prepared;
    D::collective(*comm, [&] { prepared = Maps::prepare(plan, admission); });
    const auto actual = Maps::construct(*prepared, gids, 5 * comm->getSize(), comm);
    const auto raw = pipeline_trace::summary();
    const auto after = admission.statistics();
    const auto process_upper = Maps::library_context_bytes();
    {
        pipeline_trace::Pause pause;
        EXPECT_FALSE(pipeline_trace::overflow.load());
        EXPECT_LE(raw.all_peak, admission.counters().peak - initial_used);
        EXPECT_GT(after.profiled_cumulative_upper, before.profiled_cumulative_upper);
        EXPECT_GE(process_upper, Kokkos::Impl::HostThreadTeamData::scratch_size(512, 512, 0, 0));
        EXPECT_EQ(actual.map->getLocalNumElements(), 5u);
        EXPECT_EQ(actual.map->getGlobalNumElements(), std::size_t(5 * comm->getSize()));
        for (std::size_t i = 0; i < gids.size(); ++i)
        {
            EXPECT_EQ(actual.map->getLocalElement(gids[i]), static_cast<int>(i));
            EXPECT_EQ(actual.map->getGlobalElement(i), gids[i]);
        }
    }
    N::NativeConstructionAdmission warm_admission(
        N::NativeConstructionCounterAccess::create<D::Budget>(64 * 1024 * 1024));
    const auto warm_initial = warm_admission.counters().used;
    pipeline_trace::next_epoch();
    std::shared_ptr<Maps::Prepared> warm_prepared;
    D::collective(*comm, [&] { warm_prepared = Maps::prepare(plan, warm_admission); });
    const auto warm = Maps::construct(*warm_prepared, gids, 5 * comm->getSize(), comm);
    const auto warm_raw = pipeline_trace::summary();
    const pipeline_trace::Pause pause;
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_LE(warm_raw.incremental_peak, warm_admission.counters().peak - warm_initial);
    EXPECT_EQ(warm_admission.statistics().profiled_live, Maps::retained_upper(warm.map));
    EXPECT_EQ(Maps::library_context_bytes(), process_upper);
    EXPECT_FALSE(warm.map.shares_resource(actual.map));
}
TEST(NativePublicationOutputAccounting, ActualMapWeakTailKeepsControlButReleasesItsBody)
{
    ASSERT_NO_THROW(Maps::require_profile());
    const auto comm = Tpetra::getDefaultComm();
    const std::array<GO, 1> gids{GO(comm->getRank())};
    const auto plan = Maps::plan(Maps::Kind::ContiguousOwned, gids);
    N::NativeConstructionAdmission admission(N::NativeConstructionCounterAccess::create<D::Budget>(8 * 1024 * 1024));
    std::shared_ptr<Maps::Prepared> prepared;
    D::collective(*comm, [&] { prepared = Maps::prepare(plan, admission); });
    auto owner = Maps::construct(*prepared, gids, comm->getSize(), comm);
    auto weak = owner.map.create_weak();
    prepared.reset();
    const auto full = admission.statistics();
    owner.map = Teuchos::null;
    const auto tail = admission.statistics();
    EXPECT_LT(tail.actual_live, full.actual_live);
    EXPECT_LT(tail.profiled_live, full.profiled_live);
    EXPECT_GE(tail.profiled_live, plan.rcp_node);
    weak = Teuchos::null;
    EXPECT_LT(admission.statistics().actual_live, tail.actual_live);
    EXPECT_LT(admission.statistics().profiled_live, tail.profiled_live);
}
TEST(NativePublicationOutputAccounting, ProfileBudgetFailurePrecedesTheFirstLibraryCollective)
{
    ASSERT_NO_THROW(Maps::require_profile());
    const auto comm = Tpetra::getDefaultComm();
    const std::array<GO, 3> gids{
        GO(comm->getRank() * 100), GO(comm->getRank() * 100 + 7), GO(comm->getRank() * 100 + 13)};
    const auto plan = Maps::plan(Maps::Kind::Noncontiguous, gids);
    N::NativeConstructionAdmission admission(N::NativeConstructionCounterAccess::create<D::Budget>(4096));
    EXPECT_THROW(D::collective(*comm, [&] { (void) Maps::prepare(plan, admission); }), std::invalid_argument);
    // Every rank reaches the next established collective after local rejection.
    int one = 1, total = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &one, &total);
    EXPECT_EQ(total, comm->getSize());
}
TEST(NativePublicationOutputAccounting, ScalarAndExplicitLibraryUpperRemainVisibleInCompleteReport)
{
    D::NativePublicationStorageSummary summary;
    summary.metadata = 101;
    summary.physical_geometry = 23;
    summary.scalar_numeric = 61;
    summary.library_upper = 79;
    summary.fixed = 37;
    const auto report = summary.report();
    EXPECT_EQ(report.measured_bytes(), 161u);
    EXPECT_EQ(report.output_only, 61u);
    EXPECT_EQ(report.required_maps_estimate, 79u);
    EXPECT_EQ(report.measured_bytes() + report.output_only + report.required_maps_estimate, summary.retained_bytes());
}
} // namespace
