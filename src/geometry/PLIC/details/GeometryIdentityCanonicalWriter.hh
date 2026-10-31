/** @file GeometryIdentityCanonicalWriter.hh
 * @brief Stack-only emission of the frozen canonical scalar representation.
 */
#pragma once
#include "geometry/PLIC/GeometryIdentity.hh"
#include <cmath>

namespace SimpleFluid::PLIC::details
{
struct GeometryIdentityCanonicalWriter
{
    void* context;
    void (*append)(void*, std::span<const char>);
    void byte(std::uint8_t value) const
    {
        const char encoded = static_cast<char>(value);
        append(context, {&encoded, 1});
    }
    void tag(GeometryTag value) const { byte(static_cast<std::uint8_t>(value)); }
    void word(std::uint64_t value) const
    {
        std::array<char, 8> encoded;
        for (unsigned i = 0; i < encoded.size(); ++i)
            encoded[i] = static_cast<char>(value >> (8 * i));
        append(context, encoded);
    }
    void real(long double value) const
    {
        if (value == 0)
        {
            word(0);
            word(0);
            return;
        }
        int exponent = 0;
        auto mantissa = std::frexp(std::abs(value), &exponent);
        word((std::uint64_t(std::uint32_t(exponent)) << 1) | (std::signbit(value) ? 1 : 0));
        word(std::numeric_limits<long double>::digits);
        for (int written = 0; written < std::numeric_limits<long double>::digits; written += 32)
        {
            mantissa = std::ldexp(mantissa, 32);
            const auto limb = static_cast<std::uint32_t>(mantissa);
            word(limb);
            mantissa -= limb;
        }
    }
    void point(const OrdinaryIdentityPoint& value) const
    {
        for (auto coordinate : value)
            real(coordinate);
    }
};
} // namespace SimpleFluid::PLIC::details
