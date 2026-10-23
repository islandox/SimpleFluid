/** @file testBoundedPhaseArithmetic.cc
 * @brief Independent high-precision checks of finite phase arithmetic enclosures.
 */
#include "FVM/details/BoundedPhaseArithmetic.hh"
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <gtest/gtest.h>
namespace
{
namespace fp = SimpleFluid::VoF::finite_detail;
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
} // namespace
