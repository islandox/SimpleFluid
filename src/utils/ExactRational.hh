/** @file ExactRational.hh
 * @brief Exact rational lifting of represented floating-point scalars.
 */
#pragma once
#include <bit>
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace SimpleFluid::detail::exact_rational
{
using Q = boost::multiprecision::cpp_rational;
using I = boost::multiprecision::cpp_int;

inline Q exact(double x)
{
    if (!std::isfinite(x))
        throw std::invalid_argument("Convex geometry requires finite coordinates.");
    const auto bits = std::bit_cast<std::uint64_t>(x);
    const auto exponent = (bits >> 52) & 2047;
    I mantissa = bits & ((std::uint64_t{1} << 52) - 1);
    if (exponent)
        mantissa += I{1} << 52;
    if (bits >> 63)
        mantissa = -mantissa;
    const int shift = exponent ? static_cast<int>(exponent) - 1023 - 52 : -1074;
    return shift >= 0 ? Q(mantissa << shift) : Q(mantissa, I{1} << -shift);
}

inline Q exact_wide(long double x)
{
    if (!std::isfinite(x))
        throw std::runtime_error("Unrepresentable geometry measure.");
    if (x == 0)
        return 0;
    int exponent;
    auto fraction = std::frexp(std::abs(x), &exponent);
    I mantissa = 0;
    for (int bit = 0; bit < std::numeric_limits<long double>::digits; ++bit)
    {
        fraction *= 2;
        mantissa <<= 1;
        if (fraction >= 1)
        {
            ++mantissa;
            fraction -= 1;
        }
    }
    if (x < 0)
        mantissa = -mantissa;
    const int shift = exponent - std::numeric_limits<long double>::digits;
    return shift >= 0 ? Q(mantissa << shift) : Q(mantissa, I{1} << -shift);
}
} // namespace SimpleFluid::detail::exact_rational
