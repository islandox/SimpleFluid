#include "geometry/PLIC/GeometryIdentity.hh"
#include "geometry/PLIC/details/GeometryIdentityBacking.hh"
#include "geometry/PLIC/details/OrdinaryIdentitySupport.hh"
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace SimpleFluid::PLIC::details
{

namespace
{
struct OrdinaryKeyWriter
{
    OrdinaryIdentityKey result;
    void tag(GeometryTag value) noexcept { result.bytes[result.size++] = static_cast<char>(value); }
    void word(std::uint64_t value) noexcept
    {
        for (unsigned i = 0; i < 8; ++i)
            result.bytes[result.size++] = static_cast<char>(value >> (8 * i));
    }
    void real(long double value) noexcept
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
    void point(const OrdinaryIdentityPoint& point) noexcept
    {
        for (auto value : point)
            real(value);
    }
};
std::size_t scalar_key_size(long double value) noexcept
{
    return value == 0 ? 16 : ordinary_identity_scalar_bytes;
}
} // namespace
OrdinaryIdentityPoint normalize_identity_point(OrdinaryIdentityPoint point)
{
    for (auto& value : point)
    {
        if (!std::isfinite(value))
            throw std::invalid_argument("Ordinary FP geometry: nonfinite native geometry coordinate.");
        if (value == 0)
            value = 0;
    }
    return point;
}
OrdinaryIdentityLine normalize_identity_line(OrdinaryIdentityLine line)
{
    line.direction = normalize_identity_point(std::move(line.direction));
    line.anchor = normalize_identity_point(std::move(line.anchor));
    if (line.axis > 2 || !(line.direction[line.axis] > 0))
        throw std::invalid_argument("Ordinary FP geometry: invalid native line axis or direction.");
    return line;
}
OrdinaryIdentityKey ordinary_identity_key(GeometryIdentityKind kind, const OrdinaryIdentityPoint& point) noexcept
{
    OrdinaryKeyWriter writer;
    if (kind == GeometryIdentityKind::Displacement && point == OrdinaryIdentityPoint{})
        return writer.result;
    writer.tag(kind == GeometryIdentityKind::Point ? GeometryTag::OrdinaryPoint : GeometryTag::OrdinaryDisplacement);
    writer.point(point);
    return writer.result;
}
OrdinaryIdentityKey ordinary_identity_key(const OrdinaryIdentityLine& line) noexcept
{
    OrdinaryKeyWriter writer;
    writer.tag(GeometryTag::OrdinaryLine);
    writer.word(line.axis);
    writer.point(line.direction);
    writer.point(line.anchor);
    return writer.result;
}
std::size_t ordinary_identity_key_size(GeometryIdentityKind kind, const OrdinaryIdentityPoint& point) noexcept
{
    if (kind == GeometryIdentityKind::Displacement && point == OrdinaryIdentityPoint{})
        return 0;
    std::size_t bytes = 1;
    for (auto value : point)
        bytes += scalar_key_size(value);
    return bytes;
}
std::size_t ordinary_identity_key_size(const OrdinaryIdentityLine& line) noexcept
{
    std::size_t bytes = 9;
    for (auto value : line.direction)
        bytes += scalar_key_size(value);
    for (auto value : line.anchor)
        bytes += scalar_key_size(value);
    return bytes;
}

std::strong_ordering compare_geometry_identity_native(GeometryIdentityKind kind, ArithmeticMode mode,
    const GeometryIdentityBackingPtr& first, const GeometryIdentityBackingPtr& second)
{
    if (first == second)
        return std::strong_ordering::equal;
    if (mode == ArithmeticMode::Exact)
        return compare_exact_identity_native(kind, first, second);
    if (mode == ArithmeticMode::Ordinary && kind == GeometryIdentityKind::PlaneSupport)
        return compare_ordinary_support_native(first, second);
    throw std::invalid_argument("Unsupported native identity ordering mode or kind.");
}

GeometryIdentityBackingPtr make_geometry_identity_backing(std::vector<char> key,
    std::unique_ptr<const GeometryIdentityNativePayload> native, GeometryIdentityBackingPtr parent)
{
    if (!native)
        throw std::invalid_argument("A retained identity requires a native payload.");
    return std::make_shared<const GeometryIdentityBacking>(std::move(key), std::move(parent), std::move(native));
}
const GeometryIdentityNativePayload& geometry_identity_native(const GeometryIdentityBackingPtr& owner)
{
    if (!owner || !owner->native)
        throw std::invalid_argument("Missing native geometry identity payload.");
    return *owner->native;
}
const std::vector<char>& geometry_identity_key(const GeometryIdentityBackingPtr& owner) noexcept
{
    static const std::vector<char> empty;
    return owner ? owner->key : empty;
}
std::size_t geometry_identity_backing_bytes() noexcept
{
    return sizeof(GeometryIdentityBacking);
}
std::size_t geometry_identity_payload_bytes(const GeometryIdentityBackingPtr& owner) noexcept
{
    // Match retained-construction accounting: backing structs and buffer
    // capacities, excluding allocator/control-block overhead and RSS.
    return owner ? sizeof(GeometryIdentityBacking) + owner->key.capacity() +
                       (owner->native ? owner->native->storage_bytes() : 0)
                 : 0;
}
GeometryIdentityBackingPtr geometry_identity_parent(const GeometryIdentityBackingPtr& owner) noexcept
{
    return owner ? owner->parent : GeometryIdentityBackingPtr{};
}
} // namespace SimpleFluid::PLIC::details
