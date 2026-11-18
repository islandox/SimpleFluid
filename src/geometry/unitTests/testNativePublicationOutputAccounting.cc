/** Independent concrete allocation and map-profile tests for the private output route. */
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
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
} // namespace
