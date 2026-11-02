/** @file testOrdinaryVoFSweep.cc
 * @brief Ordinary signed sweep defects use independent physical cell budgets.
 */
#include "geometry/swept_volume/FiniteVoFTransport.tcc"
#include <array>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <new>

namespace ordinary_sweep_allocation_probe
{
thread_local std::size_t watched_bytes = 0, allocations = 0;
}

// Observe only the trajectory-entry array while a rejection call is active.
// Exception-message and test-framework allocations are outside this size gate.
void* operator new(std::size_t bytes)
{
    auto* result = std::malloc(bytes ? bytes : 1);
    if (!result)
        throw std::bad_alloc();
    if (bytes == ordinary_sweep_allocation_probe::watched_bytes)
        ++ordinary_sweep_allocation_probe::allocations;
    return result;
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }

namespace
{
using SimpleFluid::ArithmeticMode;
namespace fp = SimpleFluid::VoF::finite_detail;
using Point = SimpleFluid::PLIC::Point<ArithmeticMode::Ordinary>;

struct StationarySquare
{
    static constexpr long double height = 1e12L;
    const std::array<std::uint64_t, 4> nodes{0, 1, 2, 3};
    std::map<std::uint64_t, std::pair<Point, Point>> traces;

    StationarySquare()
    {
        const std::array<std::array<long double, 3>, 4> points{
            {{0, 0, height}, {1, 0, height}, {1, 1, height}, {0, 1, height}}};
        for (std::size_t i = 0; i < nodes.size(); ++i)
        {
            const Point point(points[i]);
            traces.emplace(nodes[i], std::pair{point, point});
        }
    }
};

SimpleFluid::geometry::swept_volume::Work ordinary_work()
{
    SimpleFluid::geometry::swept_volume::Work work;
    work.arithmetic_mode = ArithmeticMode::Ordinary;
    return work;
}

// Allocate the unchanged physical volume tolerance equally to the six faces
// of a unit-area box. The budget depends on cell volume, never observed defect.
long double face_budget(long double cell_volume, const SimpleFluid::geometry::swept_volume::Work& work)
{
    return cell_volume * static_cast<long double>(work.volume_tolerance) / 6;
}

TEST(OrdinaryVoFSweep, CellBudgetRetainsQuantizedSignedDefectAndCanonicalControl)
{
    const StationarySquare face;
    for (const long double sign : {-1.L, 1.L})
    {
        auto work = ordinary_work();
        const auto budget = face_budget(1, work);
        const auto control = fp::make_value(ArithmeticMode::Ordinary, sign * budget / 2);
        ASSERT_LT(std::abs(control.value),
            std::nextafter(StationarySquare::height, std::numeric_limits<long double>::infinity()) -
                StationarySquare::height);
        const auto sweep = SimpleFluid::geometry::swept_volume::make_sweep(7, face.nodes, face.traces, control, work, budget);
        EXPECT_EQ(sweep.face, 7u);
        EXPECT_EQ(sweep.backcenter.value, sweep.center.value);
        EXPECT_EQ(sweep.backcenter.value[2], StationarySquare::height);
        ASSERT_TRUE(sweep.pieces.empty());
        EXPECT_EQ(sweep.constructed_volume.value, 0);
        EXPECT_EQ(sweep.constructed_volume.absolute_error, 0);
        EXPECT_EQ(sweep.constructed_volume.arithmetic_mode, ArithmeticMode::Ordinary);
        EXPECT_EQ(sweep.control.value, control.value);
        EXPECT_EQ(sweep.control.absolute_error, control.absolute_error);
        EXPECT_EQ(sweep.control.arithmetic_mode, ArithmeticMode::Ordinary);
        EXPECT_EQ(sweep.signed_control_difference.value, -control.value);
        EXPECT_EQ(sweep.signed_control_difference.absolute_error, 0);
        EXPECT_EQ(sweep.signed_control_difference.arithmetic_mode, ArithmeticMode::Ordinary);
        EXPECT_NE(sweep.signed_control_difference.value, 0);
        EXPECT_EQ(sweep.ordinary_volume_budget, budget);
        EXPECT_LE(std::abs(sweep.constructed_volume.value - sweep.control.value), budget);
        for (auto node : face.nodes)
        {
            EXPECT_EQ(face.traces.at(node).first.value, face.traces.at(node).second.value);
            EXPECT_EQ(face.traces.at(node).first.value[2], StationarySquare::height);
        }
    }
}

TEST(OrdinaryVoFSweep, SmallerIndependentCellBudgetRejectsTheSameSignedDefect)
{
    const StationarySquare face;
    for (const long double sign : {-1.L, 1.L})
    {
        auto work = ordinary_work();
        const auto control = fp::make_value(ArithmeticMode::Ordinary, sign * face_budget(1, work) / 2);
        // A unit-area box one quarter as deep has a smaller independent quota;
        // its stationary face and requested canonical control stay identical.
        const auto smaller_budget = face_budget(.25L, work);
        ASSERT_GT(std::abs(control.value), smaller_budget);
        EXPECT_THROW(
            (void) SimpleFluid::geometry::swept_volume::make_sweep(7, face.nodes, face.traces, control, work, smaller_budget), std::invalid_argument);
        EXPECT_EQ(control.value, sign * face_budget(1, work) / 2);
    }
}

TEST(OrdinaryVoFSweep, InvalidIndependentBudgetsReject)
{
    const StationarySquare face;
    const auto control = fp::make_value(ArithmeticMode::Ordinary, 0);
    for (const auto budget :
        {0.L, -1.L, std::numeric_limits<long double>::infinity(), std::numeric_limits<long double>::quiet_NaN()})
    {
        auto work = ordinary_work();
        EXPECT_THROW((void) SimpleFluid::geometry::swept_volume::make_sweep(7, face.nodes, face.traces, control, work, budget), std::invalid_argument);
    }
    auto work = ordinary_work();
    EXPECT_THROW((void) SimpleFluid::geometry::swept_volume::make_sweep(7, face.nodes, face.traces, control, work), std::invalid_argument);
}

TEST(OrdinaryVoFSweep, CanonicalTriangleTraversalBorrowsRecipeAndPreservesMixedNodeOrder)
{
    const std::array<std::uint64_t, 4> nodes{9, 3, 7, 2};
    const std::vector<Point> current{Point({0, 0, 0}), Point({1, 0, 0}), Point({1, 1, 0}), Point({0, 1, 0})};
    const std::vector<Point> departure{
        Point({0, 0, -.25L}), Point({1, 0, -.25L}), Point({1, 1, -.25L}), Point({0, 1, -.25L})};
    const Point backcenter({.5L, .5L, -.25L});
    namespace sweep = SimpleFluid::geometry::swept_volume;
    sweep::Sweep<ArithmeticMode::Ordinary> recipe;
    std::vector<sweep::OrdinaryTrajectorySnapshot::Entry> entries;
    for (std::size_t i = 0; i < nodes.size(); ++i)
        entries.push_back({nodes[i], current[i], departure[i]});
    recipe.trajectories = sweep::OrdinaryTrajectorySnapshot::create(std::move(entries));
    for (const auto node : nodes)
        recipe.node_slots.push_back(recipe.trajectories->slot(node));
    const std::array<const Point*, 9> points{&recipe.current(0), &recipe.current(1), &recipe.current(2),
        &recipe.current(3), &recipe.departure(0), &recipe.departure(1), &recipe.departure(2),
        &recipe.departure(3), &backcenter};
    // Frozen canonical recipe for this mixed ascending/descending node sequence.
    const std::array<std::array<std::size_t, 3>, 12> expected{{
        {8, 5, 4}, {1, 0, 4}, {1, 4, 5}, {8, 6, 5}, {1, 5, 6}, {1, 6, 2},
        {8, 7, 6}, {3, 2, 6}, {3, 6, 7}, {8, 4, 7}, {3, 7, 4}, {3, 4, 0}}};
    std::size_t ordinal = 0;
    SimpleFluid::geometry::swept_volume::visit_sweep_triangles(nodes, recipe, backcenter,
        [&](const Point& a, const Point& b, const Point& c)
        {
            ASSERT_LT(ordinal, expected.size());
            EXPECT_EQ(&a, points[expected[ordinal][0]]);
            EXPECT_EQ(&b, points[expected[ordinal][1]]);
            EXPECT_EQ(&c, points[expected[ordinal][2]]);
            ++ordinal;
        });
    EXPECT_EQ(ordinal, expected.size());
}

TEST(OrdinaryVoFSweep, TranslatedUnitSquareRetainsAnalyticSignedPrismVolume)
{
    const std::array<std::uint64_t, 4> nodes{9, 3, 7, 2};
    for (const long double distance : {-.25L, 0.L, .25L})
    {
        auto work = ordinary_work();
        std::map<std::uint64_t, std::pair<Point, Point>> traces;
        const std::array<std::array<long double, 3>, 4> coordinates{{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}};
        for (std::size_t i = 0; i < nodes.size(); ++i)
        {
            auto displaced = coordinates[i];
            displaced[2] -= distance;
            traces.emplace(nodes[i], std::pair{Point(coordinates[i]), Point(displaced)});
        }
        const auto sweep = SimpleFluid::geometry::swept_volume::make_sweep(
            17, nodes, traces, fp::make_value(ArithmeticMode::Ordinary, distance), work, face_budget(1, work));
        EXPECT_EQ(sweep.control.value, distance);
        EXPECT_LE(std::abs(sweep.constructed_volume.value - distance), face_budget(1, work));
        if (distance == 0)
            EXPECT_TRUE(sweep.pieces.empty());
        else
        {
            ASSERT_FALSE(sweep.pieces.empty());
            long double volume = 0;
            for (const auto& piece : sweep.pieces)
                volume += piece.sign * piece.geometry->volume().value;
            EXPECT_LE(std::abs(volume - distance), face_budget(1, work));
        }
    }
}

TEST(OrdinaryVoFSweep, AdjacentFacesShareCanonicalNodePointsAndKeepPrivateCenters)
{
    namespace sweep = SimpleFluid::geometry::swept_volume;
    using Snapshot = sweep::OrdinaryTrajectorySnapshot;
    // Nodes 90 and 91 intentionally have equal represented coordinates.
    const auto snapshot = Snapshot::create({{90, Point({0, 0, 0}), Point({0, 0, -.25L})},
        {91, Point({0, 0, 0}), Point({0, 0, -.25L})},
        {7, Point({1, 0, 0}), Point({1, 0, -.25L})},
        {3, Point({1, 1, 0}), Point({1, 1, -.25L})},
        {15, Point({0, 1, 0}), Point({0, 1, -.25L})}});
    ASSERT_EQ(snapshot->size(), 5u);
    EXPECT_EQ(snapshot->entries()[0].node, 3u);
    EXPECT_NE(snapshot->slot(90), snapshot->slot(91));
    EXPECT_EQ(snapshot->at(snapshot->slot(90)).current.value, snapshot->at(snapshot->slot(91)).current.value);
    EXPECT_NE(&snapshot->at(snapshot->slot(90)).current, &snapshot->at(snapshot->slot(91)).current);
    auto work = ordinary_work();
    const std::array<std::uint64_t, 3> a{90, 7, 3}, b{90, 3, 15};
    const auto first = sweep::prepare_sweep<ArithmeticMode::Ordinary>(
        17, a, snapshot, fp::make_value(ArithmeticMode::Ordinary, .125L), work, work, 1e-12L);
    const auto second = sweep::prepare_sweep<ArithmeticMode::Ordinary>(
        18, b, snapshot, fp::make_value(ArithmeticMode::Ordinary, .125L), work, work, 1e-12L);
    EXPECT_EQ(first.trajectories, second.trajectories);
    EXPECT_EQ(&first.current(0), &second.current(0));
    EXPECT_EQ(&first.current(2), &second.current(1));
    EXPECT_EQ(&first.departure(2), &second.departure(1));
    EXPECT_NE(&first.center, &second.center);
    EXPECT_NE(&first.backcenter, &second.backcenter);
    EXPECT_EQ(first.node_id(0), 90u);
    EXPECT_EQ(first.node_id(2), 3u);
    const auto materialized = sweep::materialize_sweep(first, a, work);
    EXPECT_NEAR(materialized.constructed_volume.value, .125L, 1e-12L);
    EXPECT_EQ(snapshot->at(snapshot->slot(90)).departure.value[2], -.25L);
}

TEST(OrdinaryVoFSweep, MapAdapterOwnsOnlyRequestedNodesAfterInputDestruction)
{
    namespace sweep = SimpleFluid::geometry::swept_volume;
    auto work = ordinary_work();
    const auto result = [&]
    {
        StationarySquare input;
        input.traces.emplace(999, std::pair{Point({8, 9, 10}), Point({8, 9, 11})});
        return sweep::make_sweep(17, input.nodes, input.traces,
            fp::make_value(ArithmeticMode::Ordinary, 0), work, 1e-12L);
    }();
    ASSERT_EQ(result.node_count(), 4u);
    ASSERT_EQ(result.trajectories->size(), 4u);
    EXPECT_THROW((void) result.trajectories->slot(999), std::out_of_range);
    EXPECT_EQ(result.current(2).value, (std::array<long double, 3>{1, 1, StationarySquare::height}));
    EXPECT_EQ(result.departure(2).value, result.current(2).value);
}

TEST(OrdinaryVoFSweep, SnapshotAdmissionDeduplicatesWithinEachBudgetAndDoesNotRefund)
{
    namespace sweep = SimpleFluid::geometry::swept_volume;
    using Snapshot = sweep::OrdinaryTrajectorySnapshot;
    auto snapshot = Snapshot::create({{7, Point({0, 0, 0}), Point({0, 0, 1})}});
    const auto expected = snapshot->storage_bytes() + sizeof(std::shared_ptr<const Snapshot>);
    auto exact = ordinary_work();
    exact.limits.byte_budget = expected;
    EXPECT_NO_THROW(exact.retain_trajectories(snapshot));
    EXPECT_EQ(exact.bytes, expected);
    EXPECT_NO_THROW(exact.retain_trajectories(snapshot));
    EXPECT_EQ(exact.bytes, expected);
    auto independent = ordinary_work();
    independent.limits.byte_budget = expected;
    EXPECT_NO_THROW(independent.retain_trajectories(snapshot));
    EXPECT_EQ(independent.bytes, expected);
    auto short_work = ordinary_work();
    short_work.limits.byte_budget = expected - 1;
    EXPECT_THROW(short_work.retain_trajectories(snapshot), SimpleFluid::PLIC::GeometryError);
    std::weak_ptr<const Snapshot> pin = snapshot;
    snapshot.reset();
    EXPECT_FALSE(pin.expired());
    EXPECT_EQ(exact.bytes, expected);
}

TEST(OrdinaryVoFSweep, FreezeCountsOldMapAndNewTableAtExactCapBeforeMovingPoints)
{
    namespace sweep = SimpleFluid::geometry::swept_volume;
    using Snapshot = sweep::OrdinaryTrajectorySnapshot;
    const Snapshot::TraceMap input{{91, {Point({0, 0, 0}), Point({0, 0, -.5L})}},
        {7, {Point({1, 0, 0}), Point({1, 0, -.5L})}}, {3, {Point({0, 1, 0}), Point({0, 1, -.5L})}}};
    auto probe = input;
    std::size_t probe_bytes = 0;
    const auto frozen = Snapshot::freeze(probe, probe_bytes, 1024 * 1024);
    const auto expected = input.size() * (sizeof(Snapshot::TraceMap::value_type) + 4 * sizeof(void*)) +
                          frozen->storage_bytes();
    EXPECT_TRUE(probe.empty());
    EXPECT_EQ(probe_bytes, expected);
    constexpr std::size_t existing = 17;
    for (const auto limit : {existing + expected, existing + expected - 1})
    {
        auto traces = input;
        auto used = existing;
        if (limit == existing + expected)
        {
            const auto snapshot = Snapshot::freeze(traces, used, limit);
            EXPECT_TRUE(traces.empty());
            EXPECT_EQ(used, existing + expected);
            ASSERT_EQ(snapshot->size(), input.size());
            EXPECT_EQ(snapshot->entries()[0].node, 3u);
            EXPECT_EQ(snapshot->at(snapshot->slot(91)).departure.value[2], -.5L);
        }
        else
        {
            EXPECT_THROW((void) Snapshot::freeze(traces, used, limit), SimpleFluid::PLIC::GeometryError);
            EXPECT_EQ(traces.size(), input.size());
            for (const auto& [node, trace] : traces)
            {
                EXPECT_EQ(trace.first.value, input.at(node).first.value);
                EXPECT_EQ(trace.second.value, input.at(node).second.value);
            }
            EXPECT_GT(used, existing);
        }
    }
}

TEST(OrdinaryVoFSweep, MissingNodesAndModeMismatchRejectWithoutChangingSnapshot)
{
    namespace sweep = SimpleFluid::geometry::swept_volume;
    using Snapshot = sweep::OrdinaryTrajectorySnapshot;
    const auto snapshot = Snapshot::create({{1, Point({0, 0, 0}), Point({0, 0, 0})},
        {2, Point({1, 0, 0}), Point({1, 0, 0})}, {3, Point({0, 1, 0}), Point({0, 1, 0})}});
    const std::array<std::uint64_t, 3> missing{1, 2, 77}, present{1, 2, 3};
    auto work = ordinary_work();
    EXPECT_THROW((void) sweep::prepare_sweep<ArithmeticMode::Ordinary>(17, missing, snapshot,
        fp::make_value(ArithmeticMode::Ordinary, 0), work, work, 1e-12L), std::out_of_range);
    auto retry = ordinary_work();
    EXPECT_THROW((void) sweep::prepare_sweep<ArithmeticMode::Ordinary>(17, present, snapshot,
        fp::make_value(ArithmeticMode::BoundedCertification, 0), retry, retry, 1e-12L), std::invalid_argument);
    EXPECT_EQ(snapshot->at(snapshot->slot(1)).current.value, (std::array<long double, 3>{0, 0, 0}));
    EXPECT_NO_THROW((void) sweep::prepare_sweep<ArithmeticMode::Ordinary>(17, present, snapshot,
        fp::make_value(ArithmeticMode::Ordinary, 0), retry, retry, 1e-12L));
}

TEST(OrdinaryVoFSweep, MapAdapterRejectsOversizedAndUnadmittedInputsBeforeEntryAllocation)
{
    namespace sweep = SimpleFluid::geometry::swept_volume;
    const StationarySquare input;
    for (const bool oversized : {false, true})
    {
        auto work = ordinary_work();
        if (oversized)
            work.limits.maximum_simplices = 4;
        else
            work.limits.byte_budget = sizeof(sweep::OrdinaryTrajectorySnapshot) - 1;
        ordinary_sweep_allocation_probe::allocations = 0;
        ordinary_sweep_allocation_probe::watched_bytes =
            input.nodes.size() * sizeof(sweep::OrdinaryTrajectorySnapshot::Entry);
        bool resource_rejection = false, unexpected = false;
        try
        {
            (void) sweep::make_sweep(17, input.nodes, input.traces,
                fp::make_value(ArithmeticMode::Ordinary, 0), work, 1e-12L);
        }
        catch (const SimpleFluid::PLIC::GeometryError& error)
        {
            resource_rejection = error.code() == SimpleFluid::PLIC::GeometryErrorCode::ResourceLimit;
        }
        catch (...)
        {
            unexpected = true;
        }
        ordinary_sweep_allocation_probe::watched_bytes = 0;
        EXPECT_TRUE(resource_rejection);
        EXPECT_FALSE(unexpected);
        EXPECT_EQ(ordinary_sweep_allocation_probe::allocations, 0u);
        EXPECT_EQ(work.bytes, 0u);
        EXPECT_TRUE(work.trajectory_owners.empty());
        EXPECT_EQ(input.traces.size(), input.nodes.size());
        for (const auto node : input.nodes)
        {
            EXPECT_EQ(input.traces.at(node).first.value[2], StationarySquare::height);
            EXPECT_EQ(input.traces.at(node).second.value, input.traces.at(node).first.value);
        }
    }
}

TEST(OrdinaryVoFSweep, PrivateCenterAdmissionReservesTheLiveFaceGatherAtExactCombinedCap)
{
    namespace sweep = SimpleFluid::geometry::swept_volume;
    const std::array<std::uint64_t, 4> nodes{0, 1, 2, 3};
    const auto snapshot = sweep::OrdinaryTrajectorySnapshot::create(
        {{0, Point({0, 0, 0}), Point({0, 0, -.25L})}, {1, Point({1, 0, 0}), Point({1, 0, -.25L})},
            {2, Point({1, 1, 0}), Point({1, 1, -.25L})}, {3, Point({0, 1, 0}), Point({0, 1, -.25L})}});
    auto probe_retained = ordinary_work(), probe_scratch = ordinary_work();
    probe_retained.retain_trajectories(snapshot);
    probe_scratch.limits.byte_budget -= probe_retained.bytes;
    const auto control = fp::make_value(ArithmeticMode::Ordinary, .25L);
    const auto probe = sweep::prepare_sweep<ArithmeticMode::Ordinary>(
        17, nodes, snapshot, control, probe_retained, probe_scratch, 1e-12L);
    const auto expected = probe_retained.bytes + probe_scratch.bytes;
    ASSERT_GT(probe_scratch.bytes, 0u);
    for (const auto limit : {expected, expected - 1})
    {
        auto retained = ordinary_work(), scratch = ordinary_work();
        retained.limits.byte_budget = limit;
        retained.retain_trajectories(snapshot);
        scratch.limits.byte_budget = limit - retained.bytes;
        if (limit == expected)
        {
            const auto result = sweep::prepare_sweep<ArithmeticMode::Ordinary>(
                17, nodes, snapshot, control, retained, scratch, 1e-12L);
            EXPECT_EQ(result.center.value, probe.center.value);
            EXPECT_EQ(result.backcenter.value, probe.backcenter.value);
            EXPECT_EQ(retained.bytes + scratch.bytes, expected);
        }
        else
        {
            EXPECT_THROW((void) sweep::prepare_sweep<ArithmeticMode::Ordinary>(
                17, nodes, snapshot, control, retained, scratch, 1e-12L), SimpleFluid::PLIC::GeometryError);
            EXPECT_GT(scratch.bytes, 0u);
            EXPECT_LE(retained.bytes + scratch.bytes, limit);
        }
        EXPECT_EQ(snapshot->at(0).departure.value[2], -.25L);
    }
}
} // namespace
