/** @file GeometryIdentityCanonicalWriter.hh
 * @brief Stack-only emission of the frozen canonical scalar representation.
 */
#pragma once
#include "geometry/PLIC/GeometryIdentity.hh"
#include "geometry/PLIC/details/CanonicalScalarEncoding.hh"

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
    void real(long double value) const { emit_canonical_scalar(value, *this); }
    void point(const OrdinaryIdentityPoint& value) const
    {
        for (auto coordinate : value)
            real(coordinate);
    }
};
} // namespace SimpleFluid::PLIC::details
