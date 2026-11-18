/** @file NativePartitionAccess.hh
 * @brief Closed owner-confirmed sparse partition construction.
 */
#pragma once
#include "parallel/CompositePartition.hh"
#include "parallel/details/NativeTpetraMapAdmission.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail { class NativePublicationSession; }
namespace SimpleFluid::geometry_detail
{
class NativeMeshHandleAccess;
class SIMPLEFLUID_LOCAL NativePartitionAccess final
{
public:
    struct Result
    {
        std::shared_ptr<const CompositePartition<>> partition;
        Teuchos::RCP<const DefaultTpetraTypes::map_type> actual_owned_map;
        std::shared_ptr<const NativeAllocationEnvelope> allocation;
        std::array<Tpetra::global_size_t, 5> global_map_counts{};
    };
    /** Session itself is the closed issuer: ownership is rechecked against
     * current owner-confirmed seals and exact rank-contiguous scalar routes. */
    [[nodiscard]] static Result make(Meshes::AMR::adaptive_detail::NativePublicationSession&,
        std::shared_ptr<Meshes::MultiRegionMesh>, std::span<const std::uint64_t>, std::span<const std::uint64_t>);
    [[nodiscard]] static std::shared_ptr<const NativeAllocationEnvelope> allocation(
        const std::shared_ptr<const CompositePartition<>>&) noexcept;
    /** Compatibility wrappers have no envelope. An installed envelope must
     * authenticate the partition's actual shared control. */
    [[nodiscard]] static std::shared_ptr<const NativeAllocationEnvelope> checked_allocation(
        const std::shared_ptr<const CompositePartition<>>&);
    static void visit(const std::shared_ptr<const CompositePartition<>>&, const NativeConstructorOwnerSink&,
        PLIC::details::GeometryOwnerGraphEditor&, bool, PLIC::details::NativeGeometryAccounting*);
};
} // namespace SimpleFluid::geometry_detail
