/** Private current-owner confirmation protocol; tickets are never decoder authority. */
#pragma once
#include "geometry/mesh/AMR/NativePublicationSession.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct NativeRemoteSource
{
    std::shared_ptr<const NativeCellSourceProof> source;
    std::uint64_t gid = 0;
    int owner = -1;
};
struct NativeRemoteNumericPiece
{
    std::uint64_t face_gid = 0, first_gid = 0, second_gid = std::numeric_limits<std::uint64_t>::max();
    std::vector<std::uint64_t> first_nodes, second_nodes;
};
struct NativeRemoteIncidence
{
    std::shared_ptr<const NativeCellIncidenceProof> proof;
    geometry_detail::NativeConstructorStorageOwner storage;
    std::vector<NativeRemoteNumericPiece> numeric;
};
struct NativePublicationExchangeStatistics
{
    std::size_t confirmed_reuse = 0, full_replies = 0;
    std::size_t incidence_reuse = 0, incidence_full = 0;
    OwnerExchangeStats requests, replies;
};
class SIMPLEFLUID_LOCAL NativePublicationExchange final
{
public:
    /** Every rank participates, including an empty request selection. Keys are
     * borrowed from current immutable incidence or explicit caller input. */
    [[nodiscard]] static geometry_detail::NativeConstructorVector<NativeRemoteSource> fetch(NativePublicationSession&,
        const DistributedAdaptiveHierarchy&, std::span<const AdaptiveCellKey* const>,
        NativePublicationExchangeStatistics* = nullptr);
    [[nodiscard]] static geometry_detail::NativeConstructorVector<NativeRemoteIncidence> fetch_incidence(
        NativePublicationSession&, const DistributedAdaptiveHierarchy&, std::span<const NativeRemoteSource>,
        const NativePublicationNumbering&, NativePublicationExchangeStatistics* = nullptr);
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
