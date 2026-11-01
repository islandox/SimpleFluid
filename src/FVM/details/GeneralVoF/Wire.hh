/** @file Wire.hh
 * @brief Shared scalar, interface-plane and phase record codec declarations.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/PLIC/PolyhedralReconstruction.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/VoFWireProtocol.hh"
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
/** Versioned finite Ordinary plane, including its finite minority difference.
 * Budgets reserve the compact header, two model bytes and 27 eight-byte words. */
inline constexpr std::size_t ordinary_plane_wire_bytes = 218 + packet_codec::header_bytes<packet_codec::Record>;
SIMPLEFLUID_FVM_LOCAL void validate_plane_metadata(
    const PLIC::DynamicPolyhedralReconstructionResult&, const PLIC::PolyhedralReconstructionOptions&, ArithmeticMode);
SIMPLEFLUID_FVM_LOCAL void plane(wire::Writer& w, const PLIC::DynamicPolyhedralReconstructionResult& p);
SIMPLEFLUID_FVM_LOCAL PLIC::DynamicPolyhedralReconstructionResult plane(
    wire::Reader& r, std::optional<ArithmeticMode> mode = {});
SIMPLEFLUID_FVM_LOCAL void finite_phases(wire::Writer&, const PLIC::DynamicPhaseMeasures&);
SIMPLEFLUID_FVM_LOCAL PLIC::DynamicPhaseMeasures finite_phases(wire::Reader&, ArithmeticMode);
} // namespace SimpleFluid::VoF::general_vof_detail
