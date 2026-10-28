/** @file testOrdinaryPredicateBatch.cc
 * @brief Closed Ordinary accumulation preserves ordered arithmetic and caller FP state.
 */
#if defined(__clang__)
#pragma STDC FENV_ACCESS ON
#endif
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include "geometry/PLIC/details/ordinary/Predicates.hh"
#include <array>
#include <cfenv>
#include <cmath>
#include <exception>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__) && defined(__SSE2__)
#include <xmmintrin.h>
#define SIMPLEFLUID_BATCH_TEST_X86_ENVIRONMENT 1
#endif

namespace
{
namespace PLIC = SimpleFluid::PLIC;
namespace detail = PLIC::details;
namespace predicate = detail::ordinary_predicates;
using Result = predicate::Result;
using Sign = predicate::Sign;

struct Environment
{
    std::fenv_t saved{};
    int rounding = 0, flags = 0;
#if defined(SIMPLEFLUID_BATCH_TEST_X86_ENVIRONMENT)
    unsigned mxcsr = 0;
    unsigned short control = 0;
#endif
    static Environment capture()
    {
        Environment result;
        (void) std::fegetenv(&result.saved);
        result.rounding = std::fegetround();
        result.flags = std::fetestexcept(FE_ALL_EXCEPT);
#if defined(SIMPLEFLUID_BATCH_TEST_X86_ENVIRONMENT)
        result.mxcsr = _mm_getcsr();
        __asm__ __volatile__("fnstcw %0" : "=m"(result.control));
#endif
        return result;
    }
    void restore() const
    {
        (void) std::fesetenv(&saved);
#if defined(SIMPLEFLUID_BATCH_TEST_X86_ENVIRONMENT)
        _mm_setcsr(mxcsr);
#endif
    }
};

class OrdinaryPredicateBatch : public ::testing::Test
{
    detail::Work* saved_work_ = detail::active_work;
    PLIC::GeometryStatistics saved_statistics_ = detail::statistics;
    Environment original_ = Environment::capture();

protected:
    Environment normal_;
    void SetUp() override
    {
        detail::active_work = nullptr;
        PLIC::reset_geometry_statistics();
        ASSERT_EQ(std::fesetenv(FE_DFL_ENV), 0);
        normal_ = Environment::capture();
    }
    void TearDown() override
    {
        original_.restore();
        detail::active_work = saved_work_;
        detail::statistics = saved_statistics_;
    }
};

struct Observation
{
    Result result;
    Environment before, after;
    PLIC::GeometryStatistics statistics;
    std::size_t operations = 0;
    std::optional<PLIC::GeometryErrorCode> error;
    bool recoverable = false;
    std::exception_ptr unexpected;
};

// No GTest, formatting, or floating-point diagnostics execute between taking
// the incoming snapshot and restoring a normal, masked floating environment.
Observation observe(bool batch, std::span<const Result> terms, int binary_scale, std::size_t cap,
    const Environment& normal, predicate::ResultAccumulator* reused_accumulator = nullptr)
{
    Observation observation;
    observation.before = Environment::capture();
    PLIC::reset_geometry_statistics();
    PLIC::GeometryLimits limits;
    limits.arithmetic_mode = SimpleFluid::ArithmeticMode::Ordinary;
    limits.maximum_operations = cap;
    {
        detail::Operation operation(limits);
        try
        {
            if (batch)
                observation.result = predicate::sum_results(terms, binary_scale);
            else
            {
                predicate::ResultAccumulator local_accumulator;
                auto& accumulator = reused_accumulator ? *reused_accumulator : local_accumulator;
                for (const auto& term : terms)
                    accumulator.add(term);
                observation.result = accumulator.result(binary_scale);
            }
        }
        catch (const PLIC::GeometryError& error)
        {
            observation.error = error.code();
            observation.recoverable = error.recoverable_trial();
        }
        catch (...)
        {
            observation.unexpected = std::current_exception();
        }
        observation.operations = operation.work.operations;
        observation.statistics = PLIC::geometry_statistics();
    }
    observation.after = Environment::capture();
    normal.restore();
    return observation;
}

void expect_environment(const Observation& observation)
{
    EXPECT_EQ(observation.after.rounding, observation.before.rounding);
    EXPECT_EQ(observation.after.flags, observation.before.flags);
#if defined(SIMPLEFLUID_BATCH_TEST_X86_ENVIRONMENT)
    EXPECT_EQ(observation.after.mxcsr, observation.before.mxcsr);
    EXPECT_EQ(observation.after.control, observation.before.control);
#endif
    EXPECT_FALSE(observation.unexpected);
}

void expect_equal(const Result& actual, const Result& expected)
{
    EXPECT_EQ(actual.value, expected.value);
    EXPECT_EQ(actual.absolute_error, expected.absolute_error);
    EXPECT_EQ(actual.sign, expected.sign);
    EXPECT_EQ(std::signbit(actual.value), std::signbit(expected.value));
    EXPECT_EQ(std::signbit(actual.absolute_error), std::signbit(expected.absolute_error));
}

void expect_unchanged(std::span<const Result> actual, std::span<const Result> original)
{
    ASSERT_EQ(actual.size(), original.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        SCOPED_TRACE(i);
        expect_equal(actual[i], original[i]);
    }
}

void expect_success(const Observation& observation)
{
    expect_environment(observation);
    EXPECT_FALSE(observation.error);
    EXPECT_EQ(observation.statistics.rejections, 0u);
    EXPECT_EQ(observation.statistics.floating_operations, observation.operations);
    EXPECT_EQ(observation.statistics.interval_operations, 0u);
    EXPECT_EQ(observation.statistics.exact_predicates, 0u);
    EXPECT_EQ(observation.statistics.rational_constructions, 0u);
    EXPECT_EQ(observation.statistics.rational_measures, 0u);
    EXPECT_EQ(observation.statistics.rational_ledgers, 0u);
}

void expect_failure(const Observation& observation, PLIC::GeometryErrorCode code)
{
    expect_environment(observation);
    ASSERT_TRUE(observation.error);
    EXPECT_EQ(*observation.error, code);
    EXPECT_EQ(observation.recoverable, code == PLIC::GeometryErrorCode::UnresolvedArithmetic);
    EXPECT_EQ(observation.statistics.floating_operations, observation.operations);
}

TEST_F(OrdinaryPredicateBatch, AnalyticDyadicSequencesMatchStandaloneIncludingSignedZeros)
{
    const auto e = std::ldexp(1.L, -std::numeric_limits<long double>::digits - 4);
    const std::array<std::vector<Result>, 6> cases{
        {{}, {{1, 0, Sign::Positive}, {e, 0, Sign::Positive}, {-1, 0, Sign::Negative}},
            {{1, 0, Sign::Positive}, {e, e, Sign::Unresolved}, {-e, e, Sign::Unresolved}},
            {{1, e, Sign::Positive}, {-1, e, Sign::Negative}}, {{-0.L, 0, Sign::Zero}, {0.L, -0.L, Sign::Zero}},
            {{-2, 0, Sign::Negative}, {.5L, 0, Sign::Positive}}}};
    for (std::size_t i = 0; i < cases.size(); ++i)
    {
        SCOPED_TRACE(i);
        const auto input = cases[i];
        const int scale = i == 2 ? 3 : 0;
        const auto standalone = observe(false, cases[i], scale, 100000, normal_);
        const auto batch = observe(true, cases[i], scale, 100000, normal_);
        expect_success(standalone);
        expect_success(batch);
        expect_equal(batch.result, standalone.result);
        expect_unchanged(cases[i], input);
        EXPECT_EQ(standalone.operations - batch.operations, 2 * cases[i].size());
        if (i == 0)
            expect_equal(batch.result, {});
        else if (i == 1)
            expect_equal(batch.result, {e, 0, Sign::Positive});
        else if (i == 2)
        {
            EXPECT_EQ(batch.result.value, 8);
            EXPECT_EQ(batch.result.sign, Sign::Positive);
            EXPECT_GE(batch.result.absolute_error, 16 * e);
        }
        else if (i == 3)
        {
            EXPECT_EQ(batch.result.value, 0);
            EXPECT_GT(batch.result.absolute_error, 0);
            EXPECT_EQ(batch.result.sign, Sign::Unresolved);
        }
        else if (i == 5)
            expect_equal(batch.result, {-1.5L, 0, Sign::Negative});
    }
}

TEST_F(OrdinaryPredicateBatch, ZeroSequencesRemoveOnlyRepeatedProbes)
{
    for (const std::size_t count : {2u, 4u})
    {
        SCOPED_TRACE(count);
        const std::vector<Result> terms(count);
        const auto standalone = observe(false, terms, 0, 1000, normal_);
        const auto batch = observe(true, terms, 0, 1000, normal_);
        expect_success(standalone);
        expect_success(batch);
        expect_equal(batch.result, {});
        EXPECT_EQ(standalone.operations, count == 2 ? 84u : 160u);
        EXPECT_EQ(batch.operations, count == 2 ? 80u : 152u);
        RecordProperty("standalone" + std::to_string(count), std::to_string(standalone.operations));
        RecordProperty("batch" + std::to_string(count), std::to_string(batch.operations));
    }
}

TEST_F(OrdinaryPredicateBatch, AllRoundingModesRestoreSeededFlagsOnEveryExit)
{
    const std::array<Result, 2> zeros{};
    const auto e = std::ldexp(1.L, -std::numeric_limits<long double>::digits - 4);
    const std::array<Result, 3> cancellation{{{1, 0, Sign::Positive}, {e, 0, Sign::Positive}, {-1, 0, Sign::Negative}}};
    const std::array<Result, 1> invalid{{{1, -1, Sign::Positive}}};
    const auto largest = std::numeric_limits<long double>::max();
    const std::array<Result, 2> overflow{{{largest, 0, Sign::Positive}, {largest, 0, Sign::Positive}}};
    for (const int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO})
        for (unsigned action = 0; action < 5; ++action)
        {
            SCOPED_TRACE(::testing::Message() << rounding << ',' << action);
            normal_.restore();
            const auto terms = action == 0   ? std::span<const Result>(cancellation)
                               : action == 1 ? std::span<const Result>(invalid)
                               : action == 3 ? std::span<const Result>(overflow)
                                             : std::span<const Result>(zeros);
            const auto input = std::vector<Result>(terms.begin(), terms.end());
            const int set_rounding = std::fesetround(rounding);
            const int clear_flags = std::feclearexcept(FE_ALL_EXCEPT);
            const int seed_flags = std::feraiseexcept(FE_INVALID | FE_DIVBYZERO);
            const auto observation = observe(
                true, terms, action == 2 ? std::numeric_limits<int>::min() : 0, action == 4 ? 39 : 100000, normal_);
            ASSERT_EQ(set_rounding, 0);
            ASSERT_EQ(clear_flags, 0);
            ASSERT_EQ(seed_flags, 0);
            if (action == 0)
            {
                expect_success(observation);
                expect_equal(observation.result, {e, 0, Sign::Positive});
            }
            else
                expect_failure(observation, action == 3   ? PLIC::GeometryErrorCode::UnresolvedArithmetic
                                            : action == 4 ? PLIC::GeometryErrorCode::ResourceLimit
                                                          : PLIC::GeometryErrorCode::InvalidInput);
            expect_unchanged(terms, input);
            if (action == 4)
            {
                // Probe2, first zero's36 scalar operations, then one scalar
                // operation in the second add before the next charge rejects.
                EXPECT_EQ(observation.operations, 39u);
                EXPECT_EQ(observation.statistics.rejections, 1u);
            }
        }
}

TEST_F(OrdinaryPredicateBatch, OverflowPrecedesLaterInvalidInputAndNeverMutatesTheSpan)
{
    const auto largest = std::numeric_limits<long double>::max();
    const std::array<Result, 3> terms{
        {{largest, 0, Sign::Positive}, {largest, 0, Sign::Positive}, {0, -1, Sign::Zero}}};
    const auto input = terms;
    const auto standalone = observe(false, terms, 0, 100000, normal_);
    const auto batch = observe(true, terms, 0, 100000, normal_);
    expect_failure(standalone, PLIC::GeometryErrorCode::UnresolvedArithmetic);
    expect_failure(batch, PLIC::GeometryErrorCode::UnresolvedArithmetic);
    EXPECT_EQ(batch.statistics.rejections, 1u);
    expect_unchanged(terms, input);
}

#if defined(SIMPLEFLUID_BATCH_TEST_X86_ENVIRONMENT)
enum class UnsupportedMode
{
    FlushToZero,
    DenormalsAreZero,
    SseInvalidTrap,
    X87InvalidTrap,
    X87ReducedPrecision
};

void select_unsupported(UnsupportedMode mode)
{
    // Clear pending exceptions before unmasking anything. The seeded divide
    // flag stays masked in every selected mode, including the invalid traps.
    (void) std::feclearexcept(FE_ALL_EXCEPT);
    (void) std::feraiseexcept(FE_DIVBYZERO);
    auto mxcsr = _mm_getcsr();
    unsigned short control;
    __asm__ __volatile__("fnstcw %0" : "=m"(control));
    switch (mode)
    {
        case UnsupportedMode::FlushToZero:
            mxcsr |= 1u << 15;
            break;
        case UnsupportedMode::DenormalsAreZero:
            mxcsr |= 1u << 6;
            break;
        case UnsupportedMode::SseInvalidTrap:
            mxcsr &= ~(1u << 7);
            break;
        case UnsupportedMode::X87InvalidTrap:
            control &= ~1u;
            break;
        case UnsupportedMode::X87ReducedPrecision:
            control &= ~(3u << 8);
            break;
    }
    _mm_setcsr(mxcsr);
    __asm__ __volatile__("fldcw %0" : : "m"(control));
}
#endif

TEST_F(OrdinaryPredicateBatch, UnsupportedX86ModesRejectBeforeArithmeticAndRestoreTheIncomingMode)
{
#if defined(SIMPLEFLUID_BATCH_TEST_X86_ENVIRONMENT)
    const std::array<Result, 2> terms{};
    for (const auto mode : {UnsupportedMode::FlushToZero, UnsupportedMode::DenormalsAreZero,
             UnsupportedMode::SseInvalidTrap, UnsupportedMode::X87InvalidTrap, UnsupportedMode::X87ReducedPrecision})
    {
        // x87 controls matter to this production kernel only for the x87
        // 64-bit significand long-double format; other cases remain covered.
        if ((mode == UnsupportedMode::X87InvalidTrap || mode == UnsupportedMode::X87ReducedPrecision) &&
            std::numeric_limits<long double>::digits != 64)
            continue;
        for (const bool batch : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << static_cast<int>(mode) << ',' << batch);
            normal_.restore();
            select_unsupported(mode);
            const auto observation = observe(batch, terms, 0, 100000, normal_);
            expect_failure(observation, PLIC::GeometryErrorCode::InvalidInput);
            EXPECT_EQ(observation.operations, 0u);
            EXPECT_EQ(observation.statistics.rejections, 1u);
        }
    }
#else
    GTEST_SKIP() << "Explicit MXCSR/x87 mode checks require GNU/Clang x86-64 SSE2.";
#endif
}

TEST_F(OrdinaryPredicateBatch, ExternalEnvironmentChangesAreRevalidatedForStandaloneAndBatch)
{
#if defined(SIMPLEFLUID_BATCH_TEST_X86_ENVIRONMENT)
    const std::array<Result, 2> terms{};
    for (const bool batch : {false, true})
    {
        SCOPED_TRACE(batch);
        predicate::ResultAccumulator accumulator;
        const auto first = observe(batch, terms, 0, 100000, normal_, &accumulator);
        expect_success(first);
        select_unsupported(UnsupportedMode::FlushToZero);
        const auto changed = observe(batch, terms, 0, 100000, normal_, &accumulator);
        expect_failure(changed, PLIC::GeometryErrorCode::InvalidInput);
        EXPECT_EQ(changed.operations, 0u);
        const auto restored = observe(batch, terms, 0, 100000, normal_, &accumulator);
        expect_success(restored);
        expect_equal(restored.result, first.result);
        EXPECT_EQ(restored.operations, first.operations);
    }
#else
    GTEST_SKIP() << "The external unsupported-mode transition requires x86-64 SSE2 controls.";
#endif
}
} // namespace
