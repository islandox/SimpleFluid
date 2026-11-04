#pragma once
#include "geometry/mesh/AMR/ContactPointPageView.hh"
#include <cstdint>
#include <span>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Borrowed full key; neither the framing bytes nor a canonical-key owner is retained. */
struct ContactPointKeyView
{
    std::span<const char> domain;
    const ContactPointIdentity* point;
};
/** The legacy key is LE64(domain.size), domain, LE64(point-key.size), point-key.
 * Comparison deliberately uses char, including its platform signedness. */
struct ContactPointWireLess
{
    using is_transparent = void;
    bool operator()(ContactPointKeyView, ContactPointKeyView) const;
    bool operator()(ContactPointKeyView, std::span<const char>) const;
    bool operator()(std::span<const char>, ContactPointKeyView) const;
};
struct ContactPointWireHasher
{
    using is_transparent = void;
    std::uint64_t operator()(ContactPointKeyView) const;
    std::uint64_t operator()(std::span<const char>) const noexcept;
};
[[nodiscard]] std::size_t contact_point_key_size(ContactPointKeyView);
/** Size/emission of the complete outer Writer::bytes frame used in packets.
 * No serialized key owner or decoder is needed at the outgoing boundary. */
[[nodiscard]] std::size_t framed_contact_point_key_size(ContactPointKeyView);
void emit_framed_contact_point_key(ContactPointKeyView, void*, void (*)(void*, std::span<const char>));
void append_framed_contact_point_key(Writer&, ContactPointKeyView);
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
