/** @file CapillaryNodeFans.hh
 * @brief Detached owner-authenticated fans for retained halo control nodes.
 */
#pragma once
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
struct CapillaryNodeFans
{
    std::map<std::uint64_t, std::set<Key>> fans;
    std::size_t allocation_bytes = 0;
    std::uint64_t source_control = 0;
};
/** Collective sparse import of missing finite-node fans. No geometry, source
 * topology, transport controls, or accepted state is modified. Members outside
 * the retained cell catalog stay as keys, so support admission can reject them.
 * A nonzero byte_budget caps all working/exchange storage below the source
 * option budget; zero uses the source option budget. */
SIMPLEFLUID_FVM_LOCAL CapillaryNodeFans import_capillary_node_fans(
    const GeneralVoF::Data&, std::size_t byte_budget = 0);
} // namespace SimpleFluid::VoF::general_vof_detail
