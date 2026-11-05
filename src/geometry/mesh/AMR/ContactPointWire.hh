#pragma once
#include "ContactPointPageView.hh"
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Inline point value with borrowed domain bytes; no serialized key is retained. */
struct ContactPointKeyView
{
    std::span<const char> domain;
    ContactPointIdentity point;
    ContactPointKeyView(std::span<const char> domain_, ContactPointIdentity point_)
        : domain(domain_), point(std::move(point_))
    {
    }
    /** Compatibility probes copy the pointed value immediately. */
    ContactPointKeyView(std::span<const char> domain_, const ContactPointIdentity* point_)
        : ContactPointKeyView(
              domain_, point_ ? *point_ : throw std::invalid_argument("Invalid contact-point key view."))
    {
    }
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
/** Append the legacy Writer::bytes(encode_identity(point)) without heap scratch.
 * The caller must have admitted and reserved the entire destination capacity.
 * Canonical scalar emission retains the legacy caller floating-point environment. */
void append_framed_contact_point_identity(Writer&, const ContactPointIdentity&);
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
