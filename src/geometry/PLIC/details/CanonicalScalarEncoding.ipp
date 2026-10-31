/** @file CanonicalScalarEncoding.ipp
 * @brief Sink-dependent implementation of the frozen canonical scalar representation.
 */
#pragma once
#include <cmath>
#include <limits>

namespace SimpleFluid::PLIC::details
{
template<class WordSink>
void emit_canonical_scalar(long double value, WordSink& sink) noexcept(noexcept(sink.word(std::uint64_t{})))
{
    if (value == 0)
    {
        sink.word(0);
        sink.word(0);
        return;
    }
    int exponent = 0;
    auto mantissa = std::frexp(std::abs(value), &exponent);
    sink.word((std::uint64_t(std::uint32_t(exponent)) << 1) | (std::signbit(value) ? 1 : 0));
    sink.word(std::numeric_limits<long double>::digits);
    for (int written = 0; written < std::numeric_limits<long double>::digits; written += 32)
    {
        mantissa = std::ldexp(mantissa, 32);
        const auto limb = static_cast<std::uint32_t>(mantissa);
        sink.word(limb);
        mantissa -= limb;
    }
}
} // namespace SimpleFluid::PLIC::details
