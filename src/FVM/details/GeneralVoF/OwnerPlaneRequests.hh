/** @file OwnerPlaneRequests.hh
 * @brief Invocation-local owner plane reuse and overlapping import storage.
 */
#pragma once
#include "FVM/GeneralVoF.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/swept_volume/FiniteVoFTransport.hh"
#include <map>
#include <span>

namespace SimpleFluid::VoF::general_vof_detail
{
using RequestedPlaneMap = std::map<Meshes::AMR::AdaptiveCellKey, PLIC::PolyhedralInterface>;
/** Temporary reservation on the caller's existing operation ledger. Declare
 * before import containers; destroy those containers before this reservation.
 * Returned plane nodes must be reserved by the caller before entering it. */
class PlaneImportStorage
{
public:
    SIMPLEFLUID_FVM_LOCAL PlaneImportStorage(geometry::swept_volume::Work&, GeneralVoFOwnerPlaneRequestDiagnostics&);
    SIMPLEFLUID_FVM_LOCAL ~PlaneImportStorage() noexcept;
    PlaneImportStorage(const PlaneImportStorage&) = delete;
    PlaneImportStorage& operator=(const PlaneImportStorage&) = delete;
    SIMPLEFLUID_FVM_LOCAL void charge(std::size_t);
    [[nodiscard]] SIMPLEFLUID_FVM_LOCAL bool try_charge(std::size_t) noexcept;
    SIMPLEFLUID_FVM_LOCAL void release(std::size_t) noexcept;
    [[nodiscard]] SIMPLEFLUID_FVM_LOCAL std::size_t remaining_bytes() const noexcept;
    [[nodiscard]] std::size_t charged_bytes() const noexcept { return work_.bytes - entry_; }
    SIMPLEFLUID_FVM_LOCAL void observe_exchange_peak(std::size_t) noexcept;

private:
    geometry::swept_volume::Work& work_;
    GeneralVoFOwnerPlaneRequestDiagnostics& diagnostics_;
    std::size_t entry_;
};
[[nodiscard]] SIMPLEFLUID_FVM_LOCAL std::size_t owner_plane_key_storage_bytes(const Meshes::AMR::AdaptiveCellKey&);
[[nodiscard]] SIMPLEFLUID_FVM_LOCAL std::size_t owner_plane_key_wire_bytes(const Meshes::AMR::AdaptiveCellKey&);
[[nodiscard]] SIMPLEFLUID_FVM_LOCAL std::size_t owner_plane_cache_entry_bytes(const Meshes::AMR::AdaptiveCellKey&);
[[nodiscard]] SIMPLEFLUID_FVM_LOCAL std::size_t owner_plane_packet_storage_bytes(std::span<const OwnerPacket>);
[[nodiscard]] SIMPLEFLUID_FVM_LOCAL std::size_t owner_plane_request_workspace_bytes(std::span<const OwnerPacket>);
/** Full context, packet completeness and binding validation precede every
 * lookup, including hits. validate(key, revision, peer) authenticates owner and
 * revision against the fixed source. reconstruct(key) runs only on a miss;
 * serialize(writer, plane) uses the existing Ordinary wire codec (<=216 bytes).
 * Reply/key scratch is mandatory; cache admission is optional and never uses
 * storage reserved for later replies. The peer map dies before return. */
template<class Validate, class Reconstruct, class Serialize>
std::vector<OwnerPacket> owner_plane_replies(const RequestedPlaneMap& local_results,
    std::span<const OwnerPacket> incoming, const PLIC::GeometryPacketContext&, int rank, int ranks, PlaneImportStorage&,
    GeneralVoFOwnerPlaneRequestDiagnostics&, Validate&&, Reconstruct&&, Serialize&&);
} // namespace SimpleFluid::VoF::general_vof_detail
#include "FVM/details/GeneralVoF/OwnerPlaneRequests.ipp"
