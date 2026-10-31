/** @file testGeneralVoFFaceWorkspace.cc
 * @brief Independent lifetime and overlap checks for preparation storage.
 */
#include "FVM/details/GeneralVoF/DonorSlots.hh"
#include "FVM/details/GeneralVoF/FaceWorkspaceBudget.hh"
#include "FVM/details/GeneralVoF/PendingFaceGeometry.hh"
#include "geometry/PLIC/details/GeometryKernels.hh"
#include <exception>
#include <gtest/gtest.h>
#include <string>

namespace
{
namespace local = SimpleFluid::VoF::general_vof_detail;
namespace sweep = SimpleFluid::geometry::swept_volume;
namespace PLIC = SimpleFluid::PLIC;
constexpr auto mode = SimpleFluid::ArithmeticMode::Ordinary;

sweep::Work retained_work(std::size_t limit, std::size_t retained)
{
    sweep::Work work;
    work.arithmetic_mode = mode;
    work.limits.byte_budget = limit;
    work.charge(retained);
    return work;
}

TEST(GeneralVoFFaceWorkspace, RepeatedFacesReleaseScratchAndKeepTrafficAndWorkMonotone)
{
    auto retained = retained_work(100, 10);
    local::FaceWorkspaceTotals totals;
    std::size_t previous_traffic = 10;
    for (unsigned i = 0; i < 10; ++i)
    {
        sweep::OrdinaryAggregateContext progress;
        {
            local::FaceWorkspaceBudget face(retained, progress, totals);
            // Model a complete conservative aggregate reservation. Each face
            // remains bounded even though total traffic exceeds the allowance.
            progress.bytes = 60;
            progress.operations = 7;
            progress.pairs = 1;
            face.charge_retained(2);
            face.validate_retained_commit(2);
            face.commit_retained(2);
            EXPECT_GT(face.construction_bytes(), previous_traffic);
            previous_traffic = face.construction_bytes();
            EXPECT_EQ(retained.bytes, 12u + 2 * i);
            EXPECT_LE(face.peak_bytes(), 100u);
        }
        EXPECT_EQ(retained.bytes, 12u + 2 * i);
        EXPECT_EQ(retained.pair_tests, i + 1u);
        EXPECT_EQ(totals.scratch_construction_bytes, 60u * (i + 1));
        EXPECT_EQ(totals.operations, 7u * (i + 1));
    }
    EXPECT_EQ(retained.bytes, 30u);
    EXPECT_EQ(totals.peak_bytes, 90u);
    EXPECT_EQ(previous_traffic, 630u);
}

TEST(GeneralVoFFaceWorkspace, RetainedOutputMustFitWhileAggregateScratchStillExists)
{
    auto retained = retained_work(100, 40);
    local::FaceWorkspaceTotals totals;
    sweep::OrdinaryAggregateContext progress;
    {
        local::FaceWorkspaceBudget face(retained, progress, totals);
        progress.bytes = 60;
        EXPECT_EQ(face.remaining_bytes(), 0u);
        EXPECT_THROW(face.charge_retained(1), PLIC::GeometryError);
        EXPECT_EQ(retained.bytes, 40u);
        EXPECT_EQ(face.scratch_bytes(), 60u);
    }
    EXPECT_EQ(retained.bytes, 40u);
    EXPECT_EQ(totals.scratch_construction_bytes, 60u);
    EXPECT_EQ(totals.peak_bytes, 100u);
}

TEST(GeneralVoFFaceWorkspace, OversizedFaceReportsStorageAndPreservesOriginalCause)
{
    auto retained = retained_work(100, 60);
    local::FaceWorkspaceTotals totals;
    sweep::OrdinaryAggregateContext progress;
    {
        local::FaceWorkspaceBudget face(retained, progress, totals);
        try
        {
            face.checked_action([&] { face.setup_work().charge(41); });
            FAIL() << "An oversized face must reject before admission.";
        }
        catch (const PLIC::GeometryError& error)
        {
            EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::ResourceLimit);
            EXPECT_NE(std::string(error.what()).find("retained_bytes=60"), std::string::npos);
            EXPECT_NE(std::string(error.what()).find("storage_limit=100"), std::string::npos);
            bool nested = false;
            try
            {
                std::rethrow_if_nested(error);
            }
            catch (const PLIC::GeometryError& cause)
            {
                nested = true;
                EXPECT_EQ(cause.code(), PLIC::GeometryErrorCode::ResourceLimit);
            }
            EXPECT_TRUE(nested);
        }
    }
    EXPECT_EQ(retained.bytes, 60u);
    EXPECT_EQ(totals.scratch_construction_bytes, 0u);
}

TEST(GeneralVoFFaceWorkspace, ExceptionDestroysScratchBeforeReleasingItsReservation)
{
    auto retained = retained_work(200, 16);
    local::FaceWorkspaceTotals totals;
    sweep::OrdinaryAggregateContext progress;
    const std::vector<int> previous_output{17, 19};
    bool destroyed = false;
    struct Owner
    {
        local::FaceWorkspaceBudget& face;
        bool& destroyed;
        std::vector<int> pending;
        ~Owner()
        {
            EXPECT_EQ(face.scratch_bytes(), 40u);
            EXPECT_GT(face.retained_bytes(), face.retained_entry());
            destroyed = true;
        }
    };
    const auto reject = [&]
    {
        local::FaceWorkspaceBudget face(retained, progress, totals);
        Owner owner{face, destroyed, {}};
        progress.bytes = 40;
        face.push_retained(owner.pending, 1);
        face.push_retained(owner.pending, 2);
        face.push_retained(owner.pending, 3);
        throw std::runtime_error("candidate rejected");
    };
    EXPECT_THROW(reject(), std::runtime_error);
    EXPECT_TRUE(destroyed);
    EXPECT_EQ(retained.bytes, 16u);
    EXPECT_EQ(previous_output, (std::vector<int>{17, 19}));
    EXPECT_EQ(totals.retired_retained_bytes, 7 * sizeof(int));
    EXPECT_EQ(totals.scratch_construction_bytes, 40u);
    // The same retained caller remains usable after the rejected face.
    progress = {};
    {
        local::FaceWorkspaceBudget face(retained, progress, totals);
        face.setup_work().charge(30);
        face.charge_retained(4);
        face.validate_retained_commit(4);
        face.commit_retained(4);
    }
    EXPECT_EQ(retained.bytes, 20u);
}

TEST(GeneralVoFFaceWorkspace, SuccessfulTransferKeepsOnlySurvivingOutputCapacities)
{
    auto retained = retained_work(512, 10);
    local::FaceWorkspaceTotals totals;
    std::vector<std::vector<int>> output;
    for (unsigned i = 0; i < 3; ++i)
    {
        sweep::OrdinaryAggregateContext progress;
        {
            local::FaceWorkspaceBudget face(retained, progress, totals);
            std::vector<int> pending;
            progress.bytes = 60;
            face.push_retained(pending, int(i));
            face.push_retained(pending, int(i + 100));
            const auto prior_capacity = output.capacity();
            face.reserve_retained_slot(output);
            const auto surviving =
                pending.capacity() * sizeof(int) + (output.capacity() - prior_capacity) * sizeof(std::vector<int>);
            face.validate_retained_commit(surviving);
            output.push_back(std::move(pending));
            face.commit_retained(surviving);
        }
        std::size_t actual = 10 + output.capacity() * sizeof(std::vector<int>);
        for (const auto& group : output)
            actual += group.capacity() * sizeof(int);
        EXPECT_EQ(retained.bytes, actual);
        EXPECT_EQ(output.back(), (std::vector<int>{int(i), int(i + 100)}));
    }
    EXPECT_GT(totals.retired_retained_bytes, 0u);
    EXPECT_EQ(totals.scratch_construction_bytes, 180u);
}

TEST(GeneralVoFFaceWorkspace, AggregateResultOutlivesSourceAndFaceWorkspace)
{
    using Geometry = PLIC::ConvexCellGeometry<mode>;
    using Point = PLIC::Point<mode>;
    auto retained = retained_work(1024 * 1024, 64);
    local::FaceWorkspaceTotals totals;
    std::vector<sweep::OrdinaryAggregateGroup> output;
    std::weak_ptr<const Geometry> donor_lifetime;
    sweep::OrdinaryAggregateContext progress;
    {
        local::FaceWorkspaceBudget face(retained, progress, totals);
        const auto geometry = Geometry::box({0, 0, 0}, {1, 1, 1});
        donor_lifetime = geometry;
        face.setup_work().charge(geometry->storage_bytes());
        sweep::OrdinaryAggregateSourceBatch batch{
            {Point({0, 0, 0}), Point({1, 0, 0}), Point({0, 1, 0}), Point({0, 0, 1})}, {{{{{0}, {1}, {2}, {3}}}, 1}}};
        sweep::OrdinaryAggregateDonor donor;
        donor.id = 9;
        donor.geometry = geometry;
        donor.gas = PLIC::GeometryHalfspace{{1, 0, 0}, {}, .5};
        donor.bulk_budget = (1.L / 6) * 8e-15L;
        donor.gas_budget = (7.L / 48) * 8e-15L;
        donor.carrier_budget = (1.L / 48) * 8e-15L;
        PLIC::GeometryLimits limits;
        limits.arithmetic_mode = mode;
        limits.byte_budget = face.remaining_bytes();
        limits.maximum_operations = 64 * 1024 * 1024;
        const sweep::OrdinaryAggregateSource source(std::move(batch), {}, limits, &progress);
        const std::array donors{donor};
        const auto result = source.measure(donors, &progress);
        ASSERT_EQ(result.groups.size(), 1u);
        EXPECT_GE(result.statistics.storage_bytes, result.storage_bytes());
        face.reserve_retained_slot(output);
        const auto surviving = output.capacity() * sizeof(sweep::OrdinaryAggregateGroup);
        face.validate_retained_commit(surviving);
        static_assert(std::is_nothrow_copy_constructible_v<sweep::OrdinaryAggregateGroup>);
        output.push_back(result.groups.front());
        face.commit_retained(surviving);
    }
    EXPECT_TRUE(donor_lifetime.expired());
    ASSERT_EQ(output.size(), 1u);
    EXPECT_EQ(output[0].donor, 9u);
    EXPECT_NEAR(output[0].nominal.bulk, 1.L / 6, 1e-15L);
    EXPECT_NEAR(output[0].nominal.gas, 7.L / 48, 1e-15L);
    EXPECT_NEAR(output[0].nominal.carrier, 1.L / 48, 1e-15L);
    EXPECT_EQ(retained.bytes, 64 + output.capacity() * sizeof(sweep::OrdinaryAggregateGroup));
    EXPECT_GT(totals.scratch_construction_bytes, retained.bytes);
}

TEST(GeneralVoFFaceWorkspace, WorkRejectionRemainsDistinctWithAmpleStorage)
{
    using Point = PLIC::Point<mode>;
    auto retained = retained_work(1024 * 1024, 64);
    local::FaceWorkspaceTotals totals;
    sweep::OrdinaryAggregateContext progress;
    sweep::OrdinaryAggregateSourceBatch batch{
        {Point({0, 0, 0}), Point({1, 0, 0}), Point({0, 1, 0}), Point({0, 0, 1})}, {{{{{0}, {1}, {2}, {3}}}, 1}}};
    {
        local::FaceWorkspaceBudget face(retained, progress, totals);
        PLIC::GeometryLimits limits;
        limits.arithmetic_mode = mode;
        limits.byte_budget = face.remaining_bytes();
        limits.maximum_operations = 1;
        try
        {
            face.checked_action(
                [&] { const sweep::OrdinaryAggregateSource source(std::move(batch), {}, limits, &progress); });
            FAIL();
        }
        catch (const PLIC::GeometryError& error)
        {
            EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::ResourceLimit);
            EXPECT_NE(std::string(error.what()).find("work limit"), std::string::npos);
            EXPECT_LT(progress.bytes, limits.byte_budget);
        }
    }
    EXPECT_EQ(retained.bytes, 64u);
}

TEST(GeneralVoFFaceWorkspace, PendingScalarFacesReleaseOwnersAndPreserveInterleavedOrder)
{
    using Point = PLIC::Point<mode>;
    auto work = retained_work(1024 * 1024, 0);
    local::PendingFaceGeometry pending;
    std::array<std::weak_ptr<const PLIC::DynamicPointProvenance>, 4> owners;
    for (unsigned i = 0; i < owners.size(); ++i)
    {
        sweep::Sweep<mode> recipe;
        recipe.center = PLIC::point_average<mode>(std::array{Point({static_cast<long double>(i), 0, 0})});
        owners[i] = PLIC::DynamicGeometryAccess::identity(recipe.center.dynamic());
        recipe.control.value = i + 1;
        recipe.ordinary_volume_budget = (i + 1) / 1024.L;
        recipe.current = {Point({0, 0, 0}), Point({1, 0, 0})};
        recipe.departure = {Point({0, 0, 1}), Point({1, 0, 1})};
        pending.push(nullptr, std::move(recipe), {i, i + 10}, i % 2 == 1, work);
        EXPECT_EQ(owners[i].expired(), i % 2 == 0);
    }
    const auto charged = work.bytes;
    ASSERT_EQ(pending.faces().size(), 4u);
    for (unsigned i = 0; i < pending.faces().size(); ++i)
    {
        const auto& scalar = pending.faces()[i];
        EXPECT_EQ(scalar.control.value, i + 1);
        EXPECT_EQ(scalar.ordinary_volume_budget, (i + 1) / 1024.L);
        EXPECT_EQ(scalar.geometry.has_value(), i % 2 == 1);
        if (scalar.geometry)
        {
            EXPECT_EQ(*scalar.geometry, i / 2);
            {
                auto geometric = pending.take_geometry(*scalar.geometry);
                EXPECT_EQ(geometric.candidates, (std::vector<std::size_t>{i, i + 10}));
                EXPECT_FALSE(owners[i].expired());
            }
            EXPECT_TRUE(owners[i].expired());
            EXPECT_THROW((void) pending.take_geometry(*scalar.geometry), std::logic_error);
        }
    }
    EXPECT_EQ(work.bytes, charged);
}

TEST(GeneralVoFFaceWorkspace, PendingPointTransferReleasesRecipeAndExceptionDoesNotRefundCharges)
{
    using Point = PLIC::Point<mode>;
    auto work = retained_work(1024 * 1024, 0);
    local::PendingFaceGeometry pending;
    std::weak_ptr<const PLIC::DynamicPointProvenance> owner;
    {
        sweep::Sweep<mode> recipe;
        recipe.center = PLIC::point_average<mode>(std::array{Point({.5L, .5L, 0})});
        owner = PLIC::DynamicGeometryAccess::identity(recipe.center.dynamic());
        recipe.backcenter = Point({.5L, .5L, 1});
        recipe.current = {Point({0, 0, 0}), Point({1, 0, 0}), Point({1, 1, 0}), Point({0, 1, 0})};
        recipe.departure = {Point({0, 0, 1}), Point({1, 0, 1}), Point({1, 1, 1}), Point({0, 1, 1})};
        pending.push(nullptr, std::move(recipe), {0, 1}, true, work);
    }
    const auto entry_charge = work.bytes;
    std::size_t transferred_charge = 0;
    EXPECT_THROW(
        [&]
        {
            auto geometric = pending.take_geometry(0);
            const auto points = geometric.take_points(work);
            ASSERT_EQ(points.size(), 10u);
            EXPECT_EQ(points[0].value, (std::array<long double, 3>{.5L, .5L, 0}));
            EXPECT_EQ(points[1].value, (std::array<long double, 3>{.5L, .5L, 1}));
            EXPECT_EQ(points[2].value, (std::array<long double, 3>{0, 0, 0}));
            EXPECT_EQ(points[9].value, (std::array<long double, 3>{0, 1, 1}));
            EXPECT_EQ(geometric.recipe.current.capacity(), 0u);
            EXPECT_EQ(geometric.recipe.departure.capacity(), 0u);
            EXPECT_FALSE(owner.expired());
            transferred_charge = work.bytes;
            EXPECT_EQ(transferred_charge - entry_charge, points.capacity() * sizeof(Point));
            throw std::runtime_error("Reject after consuming pending geometry.");
        }(),
        std::runtime_error);
    EXPECT_TRUE(owner.expired());
    EXPECT_EQ(work.bytes, transferred_charge);
    EXPECT_THROW((void) pending.take_geometry(0), std::logic_error);
}
TEST(GeneralVoFFaceWorkspace, SparseDonorIdsResolveUnchangedQuotaSlots)
{
    std::array<sweep::OrdinaryAggregateDonor, 3> donors{};
    donors[0].id = std::numeric_limits<std::uint64_t>::max() - 1;
    donors[1].id = 7;
    donors[2].id = std::uint64_t{1} << 40;
    const std::array ids{donors[0].id, donors[1].id, donors[2].id};
    const std::array<std::array<long double, 2>, 3> quotas{{{1.25L, 2.5L}, {3.75L, 4.5L}, {5.25L, 6.L}}};
    auto work = retained_work(4096, 0);
    const auto slots = local::make_donor_slot_order(donors, work);
    EXPECT_EQ(slots, (std::vector<std::size_t>{1, 2, 0}));
    EXPECT_EQ(work.bytes, slots.capacity() * sizeof(std::size_t));
    EXPECT_EQ(work.pair_tests, 0u);
    EXPECT_EQ(quotas[local::donor_slot(donors, slots, 7)][0], 3.75L);
    EXPECT_EQ(quotas[local::donor_slot(donors, slots, std::uint64_t{1} << 40)][1], 6.L);
    EXPECT_EQ(quotas[local::donor_slot(donors, slots, std::numeric_limits<std::uint64_t>::max() - 1)][0], 1.25L);
    EXPECT_THROW((void) local::donor_slot(donors, slots, 9), std::out_of_range);
    for (std::size_t i = 0; i < donors.size(); ++i)
        EXPECT_EQ(donors[i].id, ids[i]);

    donors[2].id = 7;
    auto duplicate_work = retained_work(4096, 0);
    const auto duplicate_slots = local::make_donor_slot_order(donors, duplicate_work);
    EXPECT_EQ(duplicate_slots, (std::vector<std::size_t>{1, 2, 0}));
    EXPECT_EQ(local::donor_slot(donors, duplicate_slots, 7), 1u);
    // The aggregate's existing donor-validation stage rejects duplicates.
}

TEST(GeneralVoFFaceWorkspace, DonorSlotAdmissionRetainsExactCapsAndInputOnRejection)
{
    std::array<sweep::OrdinaryAggregateDonor, 3> donors{};
    donors[0].id = 90;
    donors[1].id = 4;
    donors[2].id = 17;
    auto measured = retained_work(4096, 11);
    const auto slots = local::make_donor_slot_order(donors, measured);
    auto exact = retained_work(measured.bytes, 11);
    EXPECT_EQ(local::make_donor_slot_order(donors, exact), slots);
    EXPECT_EQ(exact.bytes, measured.bytes);
    auto short_cap = retained_work(measured.bytes - 1, 11);
    EXPECT_THROW((void) local::make_donor_slot_order(donors, short_cap), PLIC::GeometryError);
    EXPECT_EQ(short_cap.bytes, 11u);
    EXPECT_EQ(donors[0].id, 90u);
    EXPECT_EQ(donors[1].id, 4u);
    EXPECT_EQ(donors[2].id, 17u);
    auto empty_work = retained_work(1, 0);
    EXPECT_TRUE(local::make_donor_slot_order({}, empty_work).empty());
    EXPECT_EQ(empty_work.bytes, 0u);
    EXPECT_THROW((void) local::donor_slot({}, {}, 0), std::out_of_range);
}
} // namespace
