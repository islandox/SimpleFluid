/** @file OwnerPlaneRequests.cc
 * @brief Shared-budget reservations for sparse owner plane requests.
 */
#include "FVM/details/GeneralVoF/OwnerPlaneRequests.hh"
#include <cassert>
#include <limits>
#include <type_traits>

namespace SimpleFluid::VoF::general_vof_detail
{
namespace
{
std::size_t add(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, "Owner plane storage size overflows.");
    return a + b;
}
} // namespace
PlaneImportStorage::PlaneImportStorage(
    geometry::swept_volume::Work& work, GeneralVoFOwnerPlaneRequestDiagnostics& diagnostics)
    : work_(work), diagnostics_(diagnostics), entry_(work.bytes)
{
    observe_exchange_peak(0);
}
PlaneImportStorage::~PlaneImportStorage() noexcept
{
    work_.bytes = entry_;
}
void PlaneImportStorage::charge(std::size_t bytes)
{
    work_.charge(bytes);
    observe_exchange_peak(0);
}
bool PlaneImportStorage::try_charge(std::size_t bytes) noexcept
{
    if (bytes > remaining_bytes())
        return false;
    work_.bytes += bytes;
    observe_exchange_peak(0);
    return true;
}
void PlaneImportStorage::release(std::size_t bytes) noexcept
{
    assert(bytes <= work_.bytes - entry_);
    work_.bytes -= bytes;
}
std::size_t PlaneImportStorage::remaining_bytes() const noexcept
{
    return work_.limits.byte_budget - std::min(work_.bytes, work_.limits.byte_budget);
}
void PlaneImportStorage::observe_exchange_peak(std::size_t bytes) noexcept
{
    const auto peak = work_.bytes + std::min(bytes, std::numeric_limits<std::size_t>::max() - work_.bytes);
    diagnostics_.workspace_peak_bytes = std::max(diagnostics_.workspace_peak_bytes, peak);
}
std::size_t owner_plane_key_storage_bytes(const Meshes::AMR::AdaptiveCellKey& key)
{
    return std::visit(
        [](const auto& value) -> std::size_t
        {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, Meshes::AMR::ConvexCellAncestry>)
                return value.path.capacity() * sizeof(std::uint64_t);
            else if constexpr (std::is_same_v<T, Meshes::AMR::ProductCellAncestry>)
                return add(value.base.path.capacity() * sizeof(std::uint64_t),
                    value.axial.path.capacity() * sizeof(std::uint64_t));
            else
                return 0;
        },
        key.ancestry);
}
std::size_t owner_plane_cache_entry_bytes(const Meshes::AMR::AdaptiveCellKey& key)
{
    return add(sizeof(RequestedPlaneMap::value_type) + 4 * sizeof(void*), owner_plane_key_storage_bytes(key));
}
std::size_t owner_plane_key_wire_bytes(const Meshes::AMR::AdaptiveCellKey& key)
{
    const auto words = std::visit(
        [](const auto& value) -> std::size_t
        {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, Meshes::AMR::CartesianCellAncestry>)
                return 9;
            else if constexpr (std::is_same_v<T, Meshes::AMR::ConvexCellAncestry>)
                return add(7, value.path.size());
            else if constexpr (std::is_same_v<T, Meshes::AMR::ProductCellAncestry>)
                return add(10, add(value.base.path.size(), value.axial.path.size()));
            else
                return 7;
        },
        key.ancestry);
    if (words > std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t))
        throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, "Owner plane key size overflows.");
    return words * sizeof(std::uint64_t);
}
std::size_t owner_plane_packet_storage_bytes(std::span<const OwnerPacket> packets)
{
    if (packets.size() > std::numeric_limits<std::size_t>::max() / sizeof(OwnerPacket))
        throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, "Owner plane packet count overflows.");
    std::size_t bytes = packets.size() * sizeof(OwnerPacket);
    for (const auto& packet : packets)
        bytes = add(bytes, packet.bytes.capacity());
    return bytes;
}
std::size_t owner_plane_request_workspace_bytes(std::span<const OwnerPacket> incoming)
{
    std::size_t bytes = 0, key_scratch = 0;
    for (const auto& packet : incoming)
    {
        bytes = add(bytes, add(sizeof(OwnerPacket), add(packet.bytes.size(), 216)));
        // Borrow encoded bytes. Geometric path-vector growth can overlap old
        // and new allocations; three times the wire size covers that transient.
        key_scratch =
            std::max(key_scratch, add(sizeof(Meshes::AMR::AdaptiveCellKey),
                                      add(packet.bytes.size(), add(packet.bytes.size(), packet.bytes.size()))));
    }
    return add(bytes, key_scratch);
}
} // namespace SimpleFluid::VoF::general_vof_detail
