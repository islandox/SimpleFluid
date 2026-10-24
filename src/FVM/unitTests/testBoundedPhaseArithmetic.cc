/** @file testBoundedPhaseArithmetic.cc
 * @brief Independent high-precision checks of finite phase arithmetic enclosures.
 */
#include "FVM/details/BoundedPhaseArithmetic.hh"
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <gtest/gtest.h>
namespace
{
namespace fp = SimpleFluid::VoF::finite_detail;
using SimpleFluid::ArithmeticMode;

TEST(BoundedPhaseArithmetic, OrdinaryNominalArithmeticUsesExplicitModeAndNonzeroDenominators)
{
    const auto numerator = fp::make_value(ArithmeticMode::Ordinary, 6, 100);
    const auto denominator = fp::make_value(ArithmeticMode::Ordinary, 2, 100);
    const auto quotient = fp::divide(numerator, denominator);
    EXPECT_EQ(quotient.value, 3);
    EXPECT_EQ(quotient.absolute_error, 0);
    EXPECT_EQ(quotient.arithmetic_mode, ArithmeticMode::Ordinary);
    EXPECT_EQ(fp::lower(quotient), 3);
    EXPECT_EQ(fp::upper(quotient), 3);
    EXPECT_THROW((void) quotient.lower(), std::invalid_argument);
    EXPECT_THROW((void) quotient.upper(), std::invalid_argument);
    EXPECT_THROW((void) quotient.relative_error_at_most(3, 8e-15), std::invalid_argument);
    EXPECT_THROW((void) fp::divide(numerator, fp::make_value(ArithmeticMode::Ordinary, 0)), std::invalid_argument);
    EXPECT_THROW((void) fp::divide(fp::Value{6, 0}, fp::Value{2, 100}), std::invalid_argument);
    const fp::Value uncertain{2, 1};
    EXPECT_THROW((void) fp::add(numerator, uncertain), std::invalid_argument);
    EXPECT_THROW((void) fp::multiply(numerator, uncertain), std::invalid_argument);
    EXPECT_THROW((void) fp::divide(numerator, uncertain), std::invalid_argument);
    EXPECT_EQ(fp::add(numerator, fp::Value{2, 0}).value, 8);
    EXPECT_THROW((void) fp::make_value(ArithmeticMode::Ordinary, 1, -1), std::invalid_argument);
}

TEST(BoundedPhaseArithmetic, OrdinaryCompensatedSumsPinTheirModeWithoutClaimingBounds)
{
    fp::Sum sum(ArithmeticMode::Ordinary);
    EXPECT_EQ(sum.value().arithmetic_mode, ArithmeticMode::Ordinary);
    const auto large = std::scalbn(1.L, std::numeric_limits<long double>::digits);
    for (const auto term : {large, 1.L, -large})
        sum.add(fp::make_value(ArithmeticMode::Ordinary, term));
    EXPECT_EQ(sum.value().value, 1);
    EXPECT_EQ(sum.value().absolute_error, 0);
    EXPECT_EQ(sum.value().arithmetic_mode, ArithmeticMode::Ordinary);
    sum.add({2, 0}); // A represented singleton is a permissible input constant.
    EXPECT_EQ(sum.value().value, 3);
    EXPECT_THROW(sum.add(fp::Value{1, 1}), std::invalid_argument);
    EXPECT_EQ(sum.value().value, 3);
    fp::Sum certified(ArithmeticMode::BoundedCertification);
    EXPECT_THROW(certified.add(fp::make_value(ArithmeticMode::Ordinary, 1)), std::invalid_argument);
}

TEST(BoundedPhaseArithmetic, OrdinaryPublicationRetainsActualNarrowingAndPhysicalRangeChecks)
{
    const auto positive = fp::make_value(ArithmeticMode::Ordinary, 1, 100);
    EXPECT_NO_THROW(fp::admit_receipt(positive, 8e-15, "gas"));
    EXPECT_THROW(fp::admit_receipt(fp::make_value(ArithmeticMode::Ordinary, -1), 8e-15, "gas"), std::invalid_argument);
    EXPECT_THROW((void) fp::publish(fp::make_value(ArithmeticMode::Ordinary, -1)), std::invalid_argument);
    EXPECT_EQ(fp::publish(fp::make_value(ArithmeticMode::Ordinary, 0)).first, 0);
    const auto value = fp::make_value(
        ArithmeticMode::Ordinary, 1.L + static_cast<long double>(std::numeric_limits<double>::epsilon()) / 4);
    const auto publication = fp::publish(value);
    EXPECT_EQ(publication.first, 1);
    EXPECT_EQ(publication.second, static_cast<double>(std::abs(value.value - publication.first)));
    if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits)
        EXPECT_GT(publication.second, 0);
    const auto too_large = std::scalbn(static_cast<long double>(std::numeric_limits<double>::max()), 1);
    if (std::isfinite(too_large))
        EXPECT_THROW((void) fp::publish(fp::make_value(ArithmeticMode::Ordinary, too_large)), std::invalid_argument);
    const auto too_small = static_cast<long double>(std::numeric_limits<double>::denorm_min()) / 2;
    if (too_small > 0)
        EXPECT_THROW((void) fp::publish(fp::make_value(ArithmeticMode::Ordinary, too_small)), std::invalid_argument);
    EXPECT_THROW((void) fp::make_value(ArithmeticMode::Ordinary, std::numeric_limits<long double>::infinity()),
        std::invalid_argument);
}

using Oracle = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<256>>;
void contains(fp::Value value, const Oracle& oracle)
{
    EXPECT_LE(Oracle(value.value) - Oracle(value.absolute_error), oracle);
    EXPECT_GE(Oracle(value.value) + Oracle(value.absolute_error), oracle);
}
TEST(BoundedPhaseArithmetic, DerivedBoundsContainIndependentEndpointOracles)
{
    const long double epsilon = std::numeric_limits<long double>::epsilon();
    for (int exponent : {-15000, -4000, -1000, -10, 0, 70, 1000, 4000, 15000})
        for (long double significand : {1.L, std::nextafter(1.L, 2.L), 1.234567890123456789L, 1.999999999999999999L})
        {
            const auto a = std::scalbn(significand, exponent);
            const auto b = std::scalbn(1.333333333333333333L, -exponent / 2);
            if (!std::isfinite(a) || !std::isfinite(b) || a == 0 || b == 0) continue;
            const fp::Value left{a, std::abs(a) * 3 * epsilon}, right{b, std::abs(b) * 5 * epsilon};
            for (int i : {-1, 1})
                for (int j : {-1, 1})
                {
                    const auto x = Oracle(left.value) + i * Oracle(left.absolute_error);
                    const auto y = Oracle(right.value) + j * Oracle(right.absolute_error);
                    if (std::isfinite(left.value + right.value)) contains(fp::add(left, right), x + y);
                    if (std::isfinite(left.value * right.value) && left.value * right.value != 0)
                        contains(fp::multiply(left, right), x * y);
                    if (std::isfinite(left.value / right.value) && left.value / right.value != 0)
                        contains(fp::divide(left, right), x / y);
                }
        }
}
TEST(BoundedPhaseArithmetic, AggregateRetainsIndividuallyUnpublishablePositiveContributions)
{
    const auto quantum = std::numeric_limits<double>::denorm_min();
    const long double piece = static_cast<long double>(quantum) / 8;
    ASSERT_GT(piece, 0);
    EXPECT_EQ(static_cast<double>(piece), 0);
    fp::Sum sum;
    for (unsigned i = 0; i < 1024; ++i) sum.add({piece, 0});
    const auto result = sum.value();
    contains(result, Oracle(quantum) * 128);
    const auto published = fp::publish(result);
    EXPECT_EQ(published.first, quantum * 128);
    EXPECT_GE(published.second, result.absolute_error);
}
TEST(BoundedPhaseArithmetic, SmallProductsAndZeroReferenceRemainDistinctFromUnknownSign)
{
    const auto product = fp::multiply({1e-300, 0}, {1e-300, 0});
    EXPECT_GT(product.value, product.absolute_error);
    EXPECT_EQ(static_cast<double>(product.value), 0);
    contains(product, Oracle(1e-300) * Oracle(1e-300));
    const auto recovered = fp::divide(product, {1e-300, 0});
    contains(recovered, Oracle(1e-300));
    fp::Sum zero;
    zero.add({0, 0});
    EXPECT_EQ(zero.value().value, 0);
    EXPECT_EQ(zero.value().absolute_error, 0);
    EXPECT_EQ(fp::publish(zero.value()).first, 0);
    EXPECT_THROW(fp::publish({0, std::numeric_limits<long double>::denorm_min()}), std::invalid_argument);
    EXPECT_THROW(fp::divide({1, 0}, {1, 1}), std::invalid_argument);
}
TEST(BoundedPhaseArithmetic, ReceiptAdmissionUsesItsOwnPhaseAndExcludesFinalNarrowing)
{
    constexpr double tolerance = 8e-15;
    EXPECT_NO_THROW(fp::admit_receipt({0, 0}, tolerance, "gas"));
    EXPECT_NO_THROW(fp::admit_receipt({1e-300L, 0}, tolerance, "gas"));
    // The donor-sized gate would accept this 1e-19 error against a unit donor.
    // A newly tiny receipt still has its independent relative accuracy target.
    EXPECT_NO_THROW(fp::admit_receipt({1, 1e-19L}, tolerance, "gas"));
    EXPECT_THROW(fp::admit_receipt({1e-18L, 1e-19L}, tolerance, "gas"), std::invalid_argument);
    EXPECT_THROW(fp::admit_receipt({0, 1e-300L}, tolerance, "gas"), std::invalid_argument);
    const auto quantum = static_cast<long double>(std::numeric_limits<double>::denorm_min());
    EXPECT_NO_THROW(fp::admit_receipt({quantum * 1.25L, 0}, tolerance, "gas"));
    // Publication has an independent subnormal narrowing allowance. It must
    // not be charged as current-step geometric approximation or phase removal.
    const auto narrowed = fp::publish({quantum * 1.25L, 0});
    EXPECT_EQ(narrowed.first, std::numeric_limits<double>::denorm_min());
    EXPECT_GT(narrowed.second, 0);
}

template<ArithmeticMode Mode> void static_arithmetic_parity()
{
    using Math = fp::Arithmetic<Mode>;
    static_assert(sizeof(fp::SumFor<Mode>) == sizeof(fp::Sum));
    const auto a = Math::make_value(1.25L, 1e-20L);
    const auto b = Math::make_value(.75L, 2e-20L);
    const auto compare = [](fp::Value actual, fp::Value expected)
    {
        EXPECT_EQ(actual.value, expected.value);
        EXPECT_EQ(actual.absolute_error, expected.absolute_error);
        EXPECT_EQ(actual.arithmetic_mode, expected.arithmetic_mode);
    };
    SimpleFluid::Meshes::reset_bounded_geometry_statistics();
    const auto expected = fp::subtract(fp::multiply(a, b), fp::divide(a, b));
    const auto expected_operations = SimpleFluid::Meshes::bounded_geometry_statistics().ledger_operations;
    SimpleFluid::Meshes::reset_bounded_geometry_statistics();
    const auto actual = Math::subtract(Math::multiply(a, b), Math::divide(a, b));
    compare(actual, expected);
    EXPECT_EQ(SimpleFluid::Meshes::bounded_geometry_statistics().ledger_operations, expected_operations);
    if constexpr (Mode == ArithmeticMode::Ordinary)
    {
        fp::SumFor<Mode> unpinned;
        EXPECT_THROW(unpinned.add({1, 0}), std::invalid_argument);
        const fp::Value singleton{.75L, 0};
        SimpleFluid::Meshes::reset_bounded_geometry_statistics();
        const auto dynamic_negative = fp::negate(singleton);
        const auto dynamic_negate_events = SimpleFluid::Meshes::bounded_geometry_statistics().ledger_operations;
        SimpleFluid::Meshes::reset_bounded_geometry_statistics();
        compare(Math::negate(singleton), dynamic_negative);
        EXPECT_EQ(SimpleFluid::Meshes::bounded_geometry_statistics().ledger_operations, dynamic_negate_events);
        const fp::Value signed_zero{-0.L, -0.L};
        const auto dynamic_zero = fp::negate(signed_zero);
        const auto fixed_zero = Math::negate(signed_zero);
        compare(fixed_zero, dynamic_zero);
        EXPECT_EQ(std::signbit(fixed_zero.value), std::signbit(dynamic_zero.value));
        EXPECT_EQ(std::signbit(fixed_zero.absolute_error), std::signbit(dynamic_zero.absolute_error));
        SimpleFluid::Meshes::reset_bounded_geometry_statistics();
        const auto dynamic_difference = fp::subtract(a, singleton);
        const auto dynamic_subtract_events = SimpleFluid::Meshes::bounded_geometry_statistics().ledger_operations;
        SimpleFluid::Meshes::reset_bounded_geometry_statistics();
        compare(Math::subtract(a, singleton), dynamic_difference);
        EXPECT_EQ(SimpleFluid::Meshes::bounded_geometry_statistics().ledger_operations, dynamic_subtract_events);
    }
    SimpleFluid::Meshes::reset_bounded_geometry_statistics();
    const auto dynamic_publication = a.as_double();
    const auto dynamic_query_events = SimpleFluid::Meshes::bounded_geometry_statistics();
    SimpleFluid::Meshes::reset_bounded_geometry_statistics();
    EXPECT_EQ(Math::as_double(a), dynamic_publication);
    const auto static_query_events = SimpleFluid::Meshes::bounded_geometry_statistics();
    EXPECT_EQ(static_query_events.floating_operations, dynamic_query_events.floating_operations);
    EXPECT_EQ(static_query_events.interval_operations, dynamic_query_events.interval_operations);
    const auto foreign = Mode == ArithmeticMode::Ordinary ? ArithmeticMode::BoundedCertification :
                                                           ArithmeticMode::Ordinary;
    EXPECT_THROW((void) Math::as_double(fp::make_value(foreign, 1)), std::invalid_argument);
    fp::Sum inferred;
    fp::SumFor<Mode> inferred_fixed;
    compare(inferred_fixed.value(), inferred.value());
    inferred.add({0, 0});
    inferred_fixed.add({0, 0});
    compare(inferred_fixed.value(), inferred.value());
    inferred.add(Math::make_value(0));
    inferred_fixed.add(Math::make_value(0));
    compare(inferred_fixed.value(), inferred.value());
    fp::Sum pinned(Mode);
    fp::SumFor<Mode> pinned_fixed(Mode);
    compare(pinned_fixed.value(), pinned.value());
    pinned.add({0, 0});
    pinned_fixed.add({0, 0});
    compare(pinned_fixed.value(), pinned.value());
    fp::Sum dynamic(Mode);
    fp::SumFor<Mode> fixed(Mode);
    for (const auto value : {Math::make_value(1e20L), a, Math::make_value(-1e20L), b, Math::make_value(0)})
    {
        dynamic.add(value);
        fixed.add(value);
    }
    compare(fixed.value(), dynamic.value());
    compare(Math::divide(a, {2, 0}), fp::divide(a, {2, 0}));
    EXPECT_THROW((void) Math::divide(a, Math::make_value(0)), std::invalid_argument);
    EXPECT_THROW((void) fp::SumFor<Mode>(ArithmeticMode::Exact), std::invalid_argument);
    EXPECT_THROW((void) Math::make_value(ArithmeticMode::Exact, 1), std::invalid_argument);
    if constexpr (Mode == ArithmeticMode::Ordinary)
        EXPECT_THROW((void) Math::add(a, fp::make_value(ArithmeticMode::BoundedCertification, 1, 1e-20L)),
            std::invalid_argument);
    else
        EXPECT_THROW((void) Math::add(a, fp::make_value(ArithmeticMode::Ordinary, 1)), std::invalid_argument);
}
TEST(BoundedPhaseArithmetic, StaticOrdinaryKernelPreservesValuesCountersAndMixedTagRejection)
{
    static_arithmetic_parity<ArithmeticMode::Ordinary>();
}
TEST(BoundedPhaseArithmetic, StaticCertifiedKernelPreservesBoundsCountersAndMixedTagRejection)
{
    static_arithmetic_parity<ArithmeticMode::BoundedCertification>();
}
} // namespace
