/** @file testOrdinaryPredicateAccounting.cc
 * @brief Scalar and allocation failure boundaries of private Ordinary predicates.
 */
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include "geometry/PLIC/details/ordinary/Predicates.hh"
#include <array>
#include <cfenv>
#include <cmath>
#include <exception>
#include <gtest/gtest.h>
#include <iomanip>
#include <latch>
#include <limits>
#include <sstream>
#include <string>
#include <thread>

namespace
{
namespace PLIC = SimpleFluid::PLIC;
namespace detail = PLIC::details;
namespace predicate = detail::ordinary_predicates;
using Point = predicate::Point;

PLIC::GeometryLimits limits(std::size_t operations, std::size_t bytes = 1024 * 1024)
{
    PLIC::GeometryLimits result;
    result.arithmetic_mode = SimpleFluid::ArithmeticMode::Ordinary;
    result.maximum_operations = operations;
    result.byte_budget = bytes;
    return result;
}

// The private state is thread local. Restore it even after a fatal assertion,
// and isolate these observations from any preceding test's counter values.
class OrdinaryPredicateAccounting : public ::testing::Test
{
    detail::Work* saved_work_ = detail::active_work;
    PLIC::GeometryStatistics saved_statistics_ = detail::statistics;
    std::fenv_t saved_environment_{};

protected:
    void SetUp() override
    {
        ASSERT_EQ(std::fegetenv(&saved_environment_), 0);
        detail::active_work = nullptr;
        PLIC::reset_geometry_statistics();
    }
    void TearDown() override
    {
        detail::active_work = saved_work_;
        detail::statistics = saved_statistics_;
        (void) std::fesetenv(&saved_environment_);
    }
};

template<class F> void expect_error(F&& action, PLIC::GeometryErrorCode code)
{
    try
    {
        action();
        ADD_FAILURE() << "Expected typed geometry failure.";
    }
    catch (const PLIC::GeometryError& error)
    {
        EXPECT_EQ(error.code(), code);
        EXPECT_FALSE(error.recoverable_trial());
    }
}

void expect_counters(std::size_t operations, std::uint64_t floating, std::uint64_t rejections, const detail::Work& work)
{
    EXPECT_EQ(work.operations, operations);
    const auto statistics = PLIC::geometry_statistics();
    EXPECT_EQ(statistics.floating_operations, floating);
    EXPECT_EQ(statistics.rejections, rejections);
    EXPECT_EQ(statistics.interval_operations, 0u);
    EXPECT_EQ(statistics.exact_predicates, 0u);
    EXPECT_EQ(statistics.rational_constructions, 0u);
    EXPECT_EQ(statistics.rational_measures, 0u);
    EXPECT_EQ(statistics.rational_ledgers, 0u);
}

void record_work(const std::string& name, const detail::Work& work)
{
    const auto statistics = PLIC::geometry_statistics();
    ::testing::Test::RecordProperty(name + "_operations", std::to_string(work.operations));
    ::testing::Test::RecordProperty(name + "_bytes", std::to_string(work.bytes));
    ::testing::Test::RecordProperty(name + "_floating", std::to_string(statistics.floating_operations));
    ::testing::Test::RecordProperty(name + "_rejections", std::to_string(statistics.rejections));
}

void record_result(const std::string& name, const predicate::Result& result)
{
    // Compare represented values without inspecting long-double padding bytes.
    const auto hexadecimal = [](long double value)
    {
        std::ostringstream stream;
        stream << std::hexfloat << std::setprecision(std::numeric_limits<long double>::max_digits10) << value;
        return stream.str();
    };
    ::testing::Test::RecordProperty(name + "_value", hexadecimal(result.value));
    ::testing::Test::RecordProperty(name + "_error", hexadecimal(result.absolute_error));
    ::testing::Test::RecordProperty(name + "_sign", static_cast<int>(result.sign));
}

predicate::Result duplicate_zero()
{
    return predicate::orientation({Point{}, Point{}, Point{}, Point{}});
}

void expect_exact(const predicate::Result& result, long double value)
{
    EXPECT_EQ(result.value, value);
    EXPECT_EQ(result.absolute_error, 0);
    EXPECT_EQ(result.sign, value > 0   ? predicate::Sign::Positive
                           : value < 0 ? predicate::Sign::Negative
                                       : predicate::Sign::Zero);
}

std::array<predicate::FramedPoint, 4> unit_source()
{
    return {{{{0, 0, 0}, {}, 0}, {{1, 0, 0}, {}, 0}, {{0, 1, 0}, {}, 0}, {{0, 0, 1}, {}, 0}}};
}

std::array<predicate::SupportRow, 3> source_face_basis()
{
    std::array<predicate::SupportRow, 3> basis;
    for (unsigned i = 0; i < basis.size(); ++i)
        basis[i].source_face = i + 1;
    return basis;
}

TEST_F(OrdinaryPredicateAccounting, DuplicateZeroRetainsAtomicProbeAndFrexpBoundaries)
{
    // Four coincident origins have determinant zero independently of evaluation.
    // The IEEE probe is one indivisible two-operation charge, followed by frexp.
    for (const std::size_t cap : {1u, 2u, 3u})
    {
        SCOPED_TRACE(cap);
        PLIC::reset_geometry_statistics();
        detail::Operation operation(limits(cap));
        if (cap < 3)
            expect_error([] { (void) duplicate_zero(); }, PLIC::GeometryErrorCode::ResourceLimit);
        else
        {
            const auto result = duplicate_zero();
            expect_exact(result, 0);
            record_result("duplicate", result);
        }
        const auto charged = cap == 1 ? 0u : cap;
        expect_counters(charged, charged, cap < 3 ? 1 : 0, operation.work);
        record_work("cap" + std::to_string(cap), operation.work);
    }
}

TEST_F(OrdinaryPredicateAccounting, AccumulatorZeroAndUnitHaveIndependentKnownCosts)
{
    predicate::ResultAccumulator sum;
    detail::Operation operation(limits(1024));
    auto result = sum.result();
    expect_exact(result, 0);
    expect_counters(8, 8, 0, operation.work);
    record_result("empty", result);
    sum.add({});
    expect_counters(46, 46, 0, operation.work);
    result = sum.result();
    expect_exact(result, 0);
    expect_counters(54, 54, 0, operation.work);
    record_result("zero", result);
    sum.add({1, 0, predicate::Sign::Positive});
    expect_counters(92, 92, 0, operation.work);
    result = sum.result();
    expect_exact(result, 1);
    expect_counters(120, 120, 0, operation.work);
    record_result("unit", result);
    record_work("total", operation.work);
}

TEST_F(OrdinaryPredicateAccounting, InvalidInputsRetainTheirPreAndPostProbeFailureBoundaries)
{
    detail::Operation operation(limits(128));
    expect_error([] { (void) predicate::projected_orientation({}, {}, {}, 3); }, PLIC::GeometryErrorCode::InvalidInput);
    expect_counters(0, 0, 0, operation.work);
    record_work("projection", operation.work);
    expect_error(
        []
        {
            (void) predicate::orientation(
                {Point{}, Point{}, Point{}, Point{std::numeric_limits<long double>::infinity(), 0, 0}});
        },
        PLIC::GeometryErrorCode::InvalidInput);
    expect_counters(2, 2, 0, operation.work);
    record_work("infinite", operation.work);
    predicate::ResultAccumulator sum;
    expect_error([&] { sum.add({1, -1, predicate::Sign::Positive}); }, PLIC::GeometryErrorCode::InvalidInput);
    expect_counters(4, 4, 0, operation.work);
    record_work("negative_error", operation.work);
}

TEST_F(OrdinaryPredicateAccounting, FailedAccumulatorAddPublishesNoPartialState)
{
    predicate::ResultAccumulator sum;
    sum.add({1, 0, predicate::Sign::Positive});
    PLIC::reset_geometry_statistics();
    {
        detail::Operation operation(limits(37));
        expect_error([&] { sum.add({2, 0, predicate::Sign::Positive}); }, PLIC::GeometryErrorCode::ResourceLimit);
        expect_counters(37, 37, 1, operation.work);
        record_work("failed_add", operation.work);
    }
    const auto result = sum.result();
    expect_exact(result, 1);
    record_result("unchanged", result);
    EXPECT_EQ(PLIC::geometry_statistics().floating_operations, 65u);
}

TEST_F(OrdinaryPredicateAccounting, SupportAllocationRejectionRetainsScalarAndByteBoundaries)
{
    predicate::SupportRow query;
    query.source_face = 0;
    for (const auto budget : {std::size_t{1}, sizeof(long double)})
    {
        SCOPED_TRACE(budget);
        PLIC::reset_geometry_statistics();
        detail::Operation operation(limits(1024, budget));
        expect_error([&]
            { (void) predicate::support_intersection_distance(unit_source(), source_face_basis(), query); },
            PLIC::GeometryErrorCode::ResourceLimit);
        const bool first_allocation = budget == 1;
        const auto charged = first_allocation ? 2u : 5u;
        expect_counters(charged, charged, 1, operation.work);
        EXPECT_EQ(operation.work.bytes, first_allocation ? 0 : sizeof(long double));
        record_work(first_allocation ? "first_allocation" : "normalized_replacement", operation.work);
    }
}

TEST_F(OrdinaryPredicateAccounting, NormalNestedOperationContinuesChargingOuterWork)
{
    detail::Operation outer(limits(6));
    {
        detail::Operation nested(limits(1));
        EXPECT_EQ(detail::active_work, &outer.work);
        expect_exact(duplicate_zero(), 0);
        expect_counters(3, 3, 0, outer.work);
        EXPECT_EQ(nested.work.operations, 0u);
    }
    EXPECT_EQ(detail::active_work, &outer.work);
    expect_exact(duplicate_zero(), 0);
    expect_counters(6, 6, 0, outer.work);
    record_work("outer", outer.work);
}

TEST_F(OrdinaryPredicateAccounting, ProviderSubstitutionBindsFreshWorkForEveryAccumulatorCall)
{
    // OrdinaryAggregate substitutes its provider's Work after constructing a
    // nested Operation. An accumulator can outlive that temporary substitution.
    predicate::ResultAccumulator sum;
    detail::Operation outer(limits(64));
    {
        detail::Operation provider(limits(128, 1));
        detail::active_work = &provider.work;
        sum.add({1, 0, predicate::Sign::Positive});
        expect_exact(sum.result(), 1);
        expect_counters(66, 66, 0, provider.work);
        EXPECT_EQ(outer.work.operations, 0u);
        predicate::SupportRow query;
        query.source_face = 0;
        expect_error([&]
            { (void) predicate::support_intersection_distance(unit_source(), source_face_basis(), query); },
            PLIC::GeometryErrorCode::ResourceLimit);
        expect_counters(68, 68, 1, provider.work);
        EXPECT_EQ(provider.work.bytes, 0u);
        EXPECT_EQ(outer.work.bytes, 0u);
        record_work("provider", provider.work);
    }
    EXPECT_EQ(detail::active_work, &outer.work);
    expect_exact(sum.result(), 1);
    EXPECT_EQ(outer.work.operations, 28u);
    expect_exact(duplicate_zero(), 0);
    expect_counters(31, 99, 1, outer.work);
    record_work("restored_outer", outer.work);
}

TEST_F(OrdinaryPredicateAccounting, SequentialOperationsAndUncappedCallsAdvanceTheSameStatistics)
{
    expect_exact(duplicate_zero(), 0);
    EXPECT_EQ(PLIC::geometry_statistics().floating_operations, 3u);
    {
        detail::Operation first(limits(3));
        expect_exact(duplicate_zero(), 0);
        expect_counters(3, 6, 0, first.work);
        record_work("first", first.work);
    }
    EXPECT_EQ(detail::active_work, nullptr);
    {
        detail::Operation second(limits(2));
        expect_error([] { (void) duplicate_zero(); }, PLIC::GeometryErrorCode::ResourceLimit);
        expect_counters(2, 8, 1, second.work);
        record_work("second", second.work);
    }
    EXPECT_EQ(detail::active_work, nullptr);
    expect_exact(duplicate_zero(), 0);
    EXPECT_EQ(PLIC::geometry_statistics().floating_operations, 11u);
    EXPECT_EQ(PLIC::geometry_statistics().rejections, 1u);
}

TEST_F(OrdinaryPredicateAccounting, WorkerThreadsKeepCapsCountersAndActiveWorkIndependent)
{
    struct Observation
    {
        std::size_t operations = 0;
        PLIC::GeometryStatistics statistics;
        predicate::Result result;
        bool initial_work_empty = false, restored_work_empty = false, resource_error = false;
        std::exception_ptr unexpected;
    };
    std::array<Observation, 2> observations;
    detail::Operation sentinel(limits(3));
    expect_exact(duplicate_zero(), 0);
    std::latch ready(2), start(1);
    std::array<std::thread, 2> workers;
    for (std::size_t i = 0; i < workers.size(); ++i)
        workers[i] = std::thread(
            [&, i]
            {
                auto& observation = observations[i];
                observation.initial_work_empty = detail::active_work == nullptr;
                PLIC::reset_geometry_statistics();
                ready.count_down();
                start.wait();
                try
                {
                    detail::Operation operation(limits(i + 2));
                    try
                    {
                        observation.result = duplicate_zero();
                    }
                    catch (const PLIC::GeometryError& error)
                    {
                        observation.resource_error =
                            error.code() == PLIC::GeometryErrorCode::ResourceLimit && !error.recoverable_trial();
                        if (!observation.resource_error)
                            observation.unexpected = std::current_exception();
                    }
                    observation.operations = operation.work.operations;
                    observation.statistics = PLIC::geometry_statistics();
                }
                catch (...)
                {
                    observation.unexpected = std::current_exception();
                }
                observation.restored_work_empty = detail::active_work == nullptr;
            });
    ready.wait();
    start.count_down();
    for (auto& worker : workers)
        worker.join();
    // All GTest assertions about worker observations occur after joining.
    for (std::size_t i = 0; i < observations.size(); ++i)
    {
        SCOPED_TRACE(i);
        const auto& observation = observations[i];
        EXPECT_FALSE(observation.unexpected);
        EXPECT_TRUE(observation.initial_work_empty);
        EXPECT_TRUE(observation.restored_work_empty);
        EXPECT_EQ(observation.operations, i + 2);
        EXPECT_EQ(observation.statistics.floating_operations, i + 2);
        EXPECT_EQ(observation.statistics.rejections, i == 0 ? 1u : 0u);
        EXPECT_EQ(observation.resource_error, i == 0);
        RecordProperty(
            "worker" + std::to_string(i) + "_floating", std::to_string(observation.statistics.floating_operations));
    }
    expect_exact(observations[1].result, 0);
    EXPECT_EQ(detail::active_work, &sentinel.work);
    expect_counters(3, 3, 0, sentinel.work);
    record_work("sentinel", sentinel.work);
}

TEST_F(OrdinaryPredicateAccounting, AnalyticDyadicResultsAndCallerEnvironmentRemainStable)
{
    ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);
    ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
    ASSERT_EQ(std::feraiseexcept(FE_INVALID), 0);
    detail::Operation operation(limits(100000));
    // det(diag(1,2,4)) = 8; (1,2,3)x(2,1,4) has x component 5;
    // (1,2,3).((2,3,4)-(1,1,1))-11 = 3.
    const auto tetra = predicate::orientation({Point{}, Point{1, 0, 0}, Point{0, 2, 0}, Point{0, 0, 4}});
    const auto triangle = predicate::projected_orientation({}, {1, 2, 3}, {2, 1, 4}, 0);
    const auto distance = predicate::plane_distance(Point{2, 3, 4}, Point{1, 2, 3}, Point{1, 1, 1}, 11);
    expect_exact(tetra, 8);
    expect_exact(triangle, 5);
    expect_exact(distance, 3);
    record_result("tetra", tetra);
    record_result("triangle", triangle);
    record_result("distance", distance);
    record_work("analytic", operation.work);
    EXPECT_EQ(std::fegetround(), FE_DOWNWARD);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INVALID);
    expect_error([] { (void) predicate::projected_orientation({}, {}, {}, 3); }, PLIC::GeometryErrorCode::InvalidInput);
    EXPECT_EQ(std::fegetround(), FE_DOWNWARD);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INVALID);
}

TEST_F(OrdinaryPredicateAccounting, SupportQuotientRecordsCopyPathAndTightCapacityFailures)
{
    predicate::SupportRow query;
    query.source_face = 0;
    const auto evaluate = [&]
    { return predicate::support_intersection_distance(unit_source(), source_face_basis(), query); };
    std::size_t scalar_work = 0, storage = 0;
    {
        detail::Operation operation(limits(1000000));
        const auto result = evaluate();
        // lambda1=lambda2=lambda3=0 intersects at vertex0, where lambda0=1.
        EXPECT_FALSE(result.singular);
        expect_exact(result.distance, 1);
        record_result("support", result.distance);
        record_work("support_success", operation.work);
        scalar_work = operation.work.operations;
        storage = operation.work.bytes;
    }
    ASSERT_GT(scalar_work, 1u);
    ASSERT_GT(storage, 1u);
    for (bool limit_storage : {false, true})
    {
        PLIC::reset_geometry_statistics();
        detail::Operation operation(limits(limit_storage ? 1000000 : scalar_work - 1,
            limit_storage ? storage - 1 : 1024 * 1024));
        expect_error([&] { (void) evaluate(); }, PLIC::GeometryErrorCode::ResourceLimit);
        EXPECT_LE(operation.work.operations, operation.work.operation_limit);
        EXPECT_LE(operation.work.bytes, operation.work.limits.byte_budget);
        EXPECT_EQ(PLIC::geometry_statistics().rejections, 1u);
        // The raw partial tuple is compared with the pre-refactor compiler run.
        // C++ does not prescribe evaluation order between function arguments.
        record_work(limit_storage ? "support_byte_boundary" : "support_scalar_boundary", operation.work);
    }
}
} // namespace
