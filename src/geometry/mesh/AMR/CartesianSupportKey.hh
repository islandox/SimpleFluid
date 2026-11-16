#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Tag of the canonical Cartesian provider support-key frame. */
inline constexpr std::uint64_t cartesian_support_key_tag = 0x5346434152543631ULL;
/** Typed identity of one Cartesian provider support key.
 *
 * The encoded key is the provider's own frame: tag, framed domain, region, axis,
 * level, plane and the two transverse indices. Equality and ordering use these
 * typed fields only; the encoded bytes are the wire form and can be produced on
 * demand. `domain` borrows the caller's bytes and is never owned here. */
struct CartesianSupportKey
{
    std::span<const char> domain;
    std::uint64_t region = 0, axis = 0, level = 0, plane = 0;
    std::array<std::uint64_t, 3> transverse{};
};
namespace cartesian_support_key_detail
{
inline constexpr std::size_t frame_words = 8; // tag, domain size, region, axis, level, plane, 2 transverse
inline constexpr std::size_t frame_bytes = frame_words * sizeof(std::uint64_t);
inline void require_valid(const CartesianSupportKey& key)
{
    if (key.axis >= 3)
        throw std::invalid_argument("Cartesian support key axis out of range.");
    if (key.level >= 63)
        throw std::invalid_argument("Cartesian support key level out of range.");
    if (key.domain.size() > std::numeric_limits<std::size_t>::max() - frame_bytes)
        throw std::invalid_argument("Cartesian support key domain size overflow.");
}
} // namespace cartesian_support_key_detail
[[nodiscard]] inline bool operator==(const CartesianSupportKey& a, const CartesianSupportKey& b) noexcept
{
    if (a.region != b.region || a.axis != b.axis || a.level != b.level || a.plane != b.plane ||
        !std::ranges::equal(a.domain, b.domain))
        return false;
    // The frame carries only the two transverse components; the axis component is
    // not part of the encoded identity and is not compared.
    for (unsigned i = 0; i != 3; ++i)
        if (i != a.axis && a.transverse[i] != b.transverse[i])
            return false;
    return true;
}
[[nodiscard]] inline bool operator!=(const CartesianSupportKey& a, const CartesianSupportKey& b) noexcept
{
    return !(a == b);
}
/** Exact encoded extent for the supplied domain: 64 bytes of words plus the domain. */
[[nodiscard]] inline std::size_t cartesian_support_key_size(std::span<const char> domain) noexcept
{
    return cartesian_support_key_detail::frame_bytes + domain.size();
}
/** Decode the canonical frame. Malformed, truncated, inconsistent or future
 * grammars return an empty optional; nothing is repaired or normalized. */
[[nodiscard]] inline std::optional<CartesianSupportKey> decode_cartesian_support_key(std::span<const char> key)
{
    using namespace cartesian_support_key_detail;
    if (key.size() < frame_bytes)
        return {};
    Reader reader{key};
    if (reader.word() != cartesian_support_key_tag)
        return {};
    const auto domain_size = reader.word();
    if (domain_size != key.size() - frame_bytes)
        return {};
    CartesianSupportKey result;
    result.domain = key.subspan(reader.offset, static_cast<std::size_t>(domain_size));
    reader.offset += static_cast<std::size_t>(domain_size);
    result.region = reader.word();
    result.axis = reader.word();
    result.level = reader.word();
    result.plane = reader.word();
    if (result.axis >= 3 || result.level >= 63)
        return {};
    for (unsigned a = 0; a != 3; ++a)
        if (a != result.axis)
            result.transverse[a] = reader.word();
    if (reader.offset != key.size())
        return {};
    return result;
}
/** Append the canonical frame. Throws on an out-of-range axis or level so a
 * non-Cartesian or corrupt key can never be re-emitted as a Cartesian one. */
inline void append_cartesian_support_key(Writer& w, const CartesianSupportKey& key)
{
    using namespace cartesian_support_key_detail;
    require_valid(key);
    w.word(cartesian_support_key_tag);
    w.bytes(key.domain);
    w.word(key.region);
    w.word(key.axis);
    w.word(key.level);
    w.word(key.plane);
    for (unsigned a = 0; a != 3; ++a)
        if (a != key.axis)
            w.word(key.transverse[a]);
}
[[nodiscard]] inline AdaptiveBytes encode_cartesian_support_key(const CartesianSupportKey& key)
{
    using namespace cartesian_support_key_detail;
    require_valid(key);
    Writer w;
    w.data.reserve(cartesian_support_key_size(key.domain));
    append_cartesian_support_key(w, key);
    return std::move(w.data);
}
/** Typed comparison against an encoded key: false for any malformed frame. */
[[nodiscard]] inline bool cartesian_support_key_matches(const CartesianSupportKey& key, std::span<const char> encoded)
{
    const auto decoded = decode_cartesian_support_key(encoded);
    return decoded && *decoded == key;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
