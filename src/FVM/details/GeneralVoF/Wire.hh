/** @file Wire.hh
 * @brief Shared scalar, interface-plane and phase record codec declarations.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/PLIC/GeometryPacketCodec.hh"
#include "geometry/PLIC/PolyhedralReconstruction.hh"
#include "geometry/PLIC/ScalarPacket.hh"
#include "geometry/VoFWireProtocol.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <optional>
namespace SimpleFluid::VoF::general_vof_detail
{
namespace wire
{
using Writer = Meshes::AMR::adaptive_detail::Writer;
using Reader = Meshes::AMR::adaptive_detail::Reader;
} // namespace wire
SIMPLEFLUID_FVM_LOCAL void real(wire::Writer& w, double value);
SIMPLEFLUID_FVM_LOCAL double real(wire::Reader& r);
SIMPLEFLUID_FVM_LOCAL void finite_real(wire::Writer& w, long double value);
SIMPLEFLUID_FVM_LOCAL long double finite_real(wire::Reader& r);
SIMPLEFLUID_FVM_LOCAL void finite_value(wire::Writer& w, PLIC::DynamicGeometryValue v);
SIMPLEFLUID_FVM_LOCAL PLIC::DynamicGeometryValue finite_value(
    wire::Reader& r, std::optional<ArithmeticMode> expected = {});
/** Complete reconstruction records use natural header layout with zeroed
 * inter-member and tail padding. Body lengths and scalar words are little endian. */
struct ReconstructionPacketHeader
{
    packet_codec::Family family;
    uint8_t version;
    packet_codec::Record kind;
    ArithmeticMode mode;
    PLIC::GeometryNumericProfile profile;
    std::uint64_t body_length;
};
static_assert(
    std::is_standard_layout_v<ReconstructionPacketHeader> && std::is_trivially_copyable_v<ReconstructionPacketHeader>);
static_assert(sizeof(packet_codec::Family) == 1 && sizeof(packet_codec::Record) == 1 && sizeof(ArithmeticMode) == 1 &&
              sizeof(PLIC::GeometryNumericProfile) == 1);
inline constexpr uint8_t reconstruction_packet_version = 3;
inline constexpr std::size_t reconstruction_packet_header_bytes = sizeof(ReconstructionPacketHeader);
inline constexpr std::size_t reconstruction_scalar_wire_bytes = sizeof(PLIC::ScalarPacketWords);
[[nodiscard]] constexpr PLIC::GeometryNumericProfile reconstruction_numeric_profile() noexcept
{
    using Profile = std::numeric_limits<long double>;
    return PLIC::geometry_numeric_profile(
        Profile::radix, Profile::digits, Profile::min_exponent, Profile::max_exponent);
}
[[nodiscard]] constexpr std::size_t reconstruction_body_wire_bytes(ArithmeticMode mode)
{
    // Nineteen binary64 values, three counts, nine boolean/category bytes,
    // and one native scalar for the independent numerical-error indicator.
    constexpr auto common =
        19 * sizeof(std::uint64_t) + 3 * sizeof(std::uint64_t) + 9 + reconstruction_scalar_wire_bytes;
    switch (mode)
    {
        case ArithmeticMode::Ordinary:
            return common + reconstruction_scalar_wire_bytes;
        case ArithmeticMode::BoundedCertification:
            return common + 2 * reconstruction_scalar_wire_bytes;
        case ArithmeticMode::Exact:
            return common + 2 * reconstruction_scalar_wire_bytes + 1;
    }
    throw std::invalid_argument("Unknown PLIC reconstruction arithmetic mode.");
}
[[nodiscard]] constexpr std::size_t reconstruction_wire_bytes(ArithmeticMode mode)
{
    return reconstruction_packet_header_bytes + reconstruction_body_wire_bytes(mode);
}
inline constexpr std::size_t ordinary_plane_wire_bytes = reconstruction_wire_bytes(ArithmeticMode::Ordinary);
SIMPLEFLUID_FVM_LOCAL void validate_plane_metadata(
    const PLIC::DynamicPolyhedralReconstructionResult&, const PLIC::PolyhedralReconstructionOptions&, ArithmeticMode);
SIMPLEFLUID_FVM_LOCAL void plane(wire::Writer& w, const PLIC::DynamicPolyhedralReconstructionResult& p);
SIMPLEFLUID_FVM_LOCAL PLIC::DynamicPolyhedralReconstructionResult plane(
    wire::Reader& r, std::optional<ArithmeticMode> mode = {});
/** Exact transfer retains its finite common-plane metadata admission. The
 * independent resolution assessment keeps its original binary64 values. */
SIMPLEFLUID_FVM_LOCAL PLIC::DynamicPolyhedralReconstructionResult plane_for_exact_transfer(wire::Reader&);
SIMPLEFLUID_FVM_LOCAL void finite_phases(wire::Writer&, const PLIC::DynamicPhaseMeasures&);
SIMPLEFLUID_FVM_LOCAL PLIC::DynamicPhaseMeasures finite_phases(wire::Reader&, ArithmeticMode);
} // namespace SimpleFluid::VoF::general_vof_detail
