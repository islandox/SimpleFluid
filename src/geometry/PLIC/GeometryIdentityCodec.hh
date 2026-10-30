/** @file GeometryIdentityCodec.hh
 * @brief Canonical versioned identity wire import and explicit promotion.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/PLIC/GeometryIdentity.hh"
#include "geometry/PLIC/GeometryPacketCodec.hh"
#include <span>
#include <stdexcept>
namespace SimpleFluid::PLIC
{
namespace details
{
[[nodiscard]] SIMPLEFLUID_LOCAL std::vector<char> checked_identity_key(
    GeometryIdentityKind, ArithmeticMode, std::span<const char>);
}
[[nodiscard]] constexpr GeometryPacketKind identity_packet_kind(GeometryIdentityKind kind)
{
    switch (kind)
    {
        case GeometryIdentityKind::Point:
            return GeometryPacketKind::PointIdentity;
        case GeometryIdentityKind::Line:
            return GeometryPacketKind::LineIdentity;
        case GeometryIdentityKind::PlaneSupport:
            return GeometryPacketKind::PlaneSupportIdentity;
        case GeometryIdentityKind::Displacement:
            return GeometryPacketKind::DisplacementIdentity;
    }
    throw std::invalid_argument("Unknown geometry identity kind.");
}
/** Complete wire size, including the independently versioned envelope. */
template<GeometryIdentityKind Kind, ArithmeticMode Mode>
[[nodiscard]] std::size_t encoded_identity_size(const GeometryIdentity<Kind, Mode>& identity) noexcept
{
    return geometry_packet_header_bytes + canonical_key_size(identity);
}
template<GeometryIdentityKind Kind, ArithmeticMode Mode>
[[nodiscard]] std::vector<char> encode_identity(const GeometryIdentity<Kind, Mode>& identity)
{
    return encode_geometry_packet(identity_packet_kind(Kind), Mode, canonical_key(identity));
}
template<class Identity> [[nodiscard]] Identity decode_identity(std::span<const char> packet, std::size_t byte_budget)
{
    const auto envelope = decode_geometry_packet(packet, Identity::mode, byte_budget);
    if (envelope.kind != identity_packet_kind(Identity::kind))
        throw std::invalid_argument("Foreign geometry identity kind.");
    return Identity(details::checked_identity_key(Identity::kind, Identity::mode, envelope.body), byte_budget);
}
template<class Identity> [[nodiscard]] Identity decode_identity(std::span<const char> packet)
{
    return decode_identity<Identity>(packet, geometry_packet_default_budget);
}
[[nodiscard]] SIMPLEFLUID_LOCAL PointIdentity<ArithmeticMode::Exact> promote_identity(
    const PointIdentity<ArithmeticMode::Ordinary>&);
[[nodiscard]] SIMPLEFLUID_LOCAL DisplacementIdentity<ArithmeticMode::Exact> promote_identity(
    const DisplacementIdentity<ArithmeticMode::Ordinary>&);
} // namespace SimpleFluid::PLIC
