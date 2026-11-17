/** Shared-domain storage adapter for the existing owner-packet wire protocol. */
#pragma once
#include "geometry/details/NativeConstructorStorage.hh"
#include "parallel/SparseOwnerExchange.hh"
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_DefaultSerialComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <array>
#include <limits>
#include <utility>
namespace SimpleFluid
{
/** One allocation-free failure reduction for the qualified MPI/serial path.
 * Compatibility custom communicators keep their existing virtual reduction. */
inline int owner_packet_failure_agreement(const Teuchos::Comm<int>& comm, int failed)
{
#ifdef HAVE_TEUCHOS_MPI
    if (const auto* mpi = dynamic_cast<const Teuchos::MpiComm<int>*>(&comm))
    {
        int any = 0;
        const MPI_Comm raw = *mpi->getRawMpiComm();
        if (MPI_Allreduce(&failed, &any, 1, MPI_INT, MPI_MAX, raw) != MPI_SUCCESS)
            throw std::invalid_argument("Native collective failure agreement failed.");
        return any;
    }
#endif
    if (dynamic_cast<const Teuchos::SerialComm<int>*>(&comm))
        return failed;
    int any = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &failed, &any);
    return any;
}
/** Fixed scalar broadcasts use no CommHelpers timer/serialization owners. */
inline void owner_packet_broadcast_words(const Teuchos::Comm<int>& comm, int root,
    std::span<std::uint64_t> values)
{
    if (values.size() > std::size_t(std::numeric_limits<int>::max()) / sizeof(std::uint64_t))
        throw std::invalid_argument("Native scalar broadcast exceeds its fixed byte count.");
#ifdef HAVE_TEUCHOS_MPI
    if (const auto* mpi = dynamic_cast<const Teuchos::MpiComm<int>*>(&comm))
    {
        const MPI_Comm raw = *mpi->getRawMpiComm();
        if (MPI_Bcast(values.data(), static_cast<int>(values.size()), MPI_UINT64_T, root, raw) != MPI_SUCCESS)
            throw std::invalid_argument("Native scalar broadcast failed.");
        return;
    }
#endif
    if (dynamic_cast<const Teuchos::SerialComm<int>*>(&comm)) return;
    comm.broadcast(root, static_cast<int>(values.size_bytes()), reinterpret_cast<char*>(values.data()));
}
/** Integer numbering agreement; no floating-point reduction order changes. */
inline std::pair<std::uint64_t, std::uint64_t> owner_packet_entity_range(
    const Teuchos::Comm<int>& comm, std::uint64_t local)
{
    const int overflow = local > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(comm.getSize());
    if (owner_packet_failure_agreement(comm, overflow))
        throw std::invalid_argument("Native owner numbering exceeds safe prefix range on a rank.");
#ifdef HAVE_TEUCHOS_MPI
    if (const auto* mpi = dynamic_cast<const Teuchos::MpiComm<int>*>(&comm))
    {
        const MPI_Comm raw = *mpi->getRawMpiComm();
        std::uint64_t before = 0, global = 0;
        if (MPI_Exscan(&local, &before, 1, MPI_UINT64_T, MPI_SUM, raw) != MPI_SUCCESS ||
            MPI_Allreduce(&local, &global, 1, MPI_UINT64_T, MPI_SUM, raw) != MPI_SUCCESS)
            throw std::invalid_argument("Native owner numbering collective failed.");
        if (!comm.getRank())
            before = 0;
        return {before, global};
    }
#endif
    if (dynamic_cast<const Teuchos::SerialComm<int>*>(&comm))
        return {0, local};
    std::uint64_t inclusive = 0, global = 0;
    Teuchos::scan(comm, Teuchos::REDUCE_SUM, 1, &local, &inclusive);
    Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, 1, &local, &global);
    return {inclusive - local, global};
}
/** Private partition counts use the same supplied MPI subcommunicator and
 * original sum/prefix/max order, without Teuchos reduction-wrapper C++ births.
 * Library MPI-internal storage is separate from requested C++ extents.
 */
inline void owner_packet_partition_counts(const Teuchos::Comm<int>& comm,
    const std::array<std::uint64_t, 5>& local_counts, std::array<std::uint64_t, 5>& global_counts,
    std::uint64_t& inclusive, std::uint64_t& maximum)
{
#ifdef HAVE_TEUCHOS_MPI
    if (const auto* mpi = dynamic_cast<const Teuchos::MpiComm<int>*>(&comm))
    {
        const MPI_Comm raw = *mpi->getRawMpiComm();
        if (MPI_Allreduce(local_counts.data(), global_counts.data(), 5, MPI_UINT64_T, MPI_SUM, raw) != MPI_SUCCESS ||
            MPI_Scan(local_counts.data(), &inclusive, 1, MPI_UINT64_T, MPI_SUM, raw) != MPI_SUCCESS ||
            MPI_Allreduce(local_counts.data(), &maximum, 1, MPI_UINT64_T, MPI_MAX, raw) != MPI_SUCCESS)
            throw std::invalid_argument("Native partition count collective failed.");
        return;
    }
#endif
    if (!dynamic_cast<const Teuchos::SerialComm<int>*>(&comm))
        throw std::invalid_argument("Native partition profile requires the actual MPI or serial communicator.");
    global_counts = local_counts;
    inclusive = maximum = local_counts[0];
}
inline void owner_packet_broadcast_identity(const Teuchos::Comm<int>& comm, std::uint64_t& identity)
{
#ifdef HAVE_TEUCHOS_MPI
    if (const auto* mpi = dynamic_cast<const Teuchos::MpiComm<int>*>(&comm))
    {
        const MPI_Comm raw = *mpi->getRawMpiComm();
        if (MPI_Bcast(&identity, 1, MPI_UINT64_T, 0, raw) != MPI_SUCCESS)
            throw std::invalid_argument("Native partition identity collective failed.");
        return;
    }
#endif
    if (!dynamic_cast<const Teuchos::SerialComm<int>*>(&comm))
        throw std::invalid_argument("Native partition profile requires the actual MPI or serial communicator.");
}
namespace native_packet_detail
{
struct Storage
{
    geometry_detail::NativeConstructorStorageOwner owner;
};
} // namespace native_packet_detail
/** Base destruction order frees actual packets before retiring their storage receipts. */
class SIMPLEFLUID_LOCAL NativeOwnerExchange : private native_packet_detail::Storage, public OwnerExchange
{
public:
    NativeOwnerExchange() = default;
    NativeOwnerExchange(NativeOwnerExchange&&) noexcept = default;
    NativeOwnerExchange& operator=(NativeOwnerExchange&& other) noexcept
    {
        if (this != &other)
        {
            std::vector<OwnerPacket>().swap(packets);
            owner = std::move(other.owner);
            static_cast<OwnerExchange&>(*this) = std::move(static_cast<OwnerExchange&>(other));
        }
        return *this;
    }
    NativeOwnerExchange(const NativeOwnerExchange&) = delete;
    NativeOwnerExchange& operator=(const NativeOwnerExchange&) = delete;
    [[nodiscard]] geometry_detail::NativeConstructorStorageOwner& storage() noexcept { return owner; }
    [[nodiscard]] const geometry_detail::NativeConstructorStorageOwner& storage() const noexcept { return owner; }
};
/** This adapter uses the actual supplied communicator and the same peer-coalesced
 * length-word streams and cyclic chunk rounds as the compatibility transport.
 * All project C++ storage uses the supplied native domain. Blocking MPI C entry
 * points allocate no Distributor/Teuchos C++ request/plan objects; MPI-internal
 * storage remains outside the raw C++ extent measurement. */
[[nodiscard]] SIMPLEFLUID_LOCAL NativeOwnerExchange exchange_owner_packets(std::span<const OwnerPacket>,
    Teuchos::RCP<const Teuchos::Comm<int>>, const geometry_detail::NativeConstructionAdmission&,
    std::size_t round_limit = 16 * 1024 * 1024);
} // namespace SimpleFluid
