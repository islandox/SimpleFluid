/** @file GeometryPacketHeader.hh
 * @brief Private checked envelope emission shared by owning and streaming codecs.
 */
#pragma once
#include "geometry/PLIC/GeometryPacketCodec.hh"
#include <array>

namespace SimpleFluid::PLIC::details
{
[[nodiscard]] SIMPLEFLUID_LOCAL std::array<char, geometry_packet_header_bytes> geometry_packet_header(
    GeometryPacketKind, ArithmeticMode, std::size_t body_bytes,
    std::size_t budget = geometry_packet_default_budget);
} // namespace SimpleFluid::PLIC::details
