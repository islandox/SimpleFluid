/** @file NativePublicationNumbering.hh
 * @brief Persistent physical-numbering records with current scalar GID routes.
 */
#pragma once
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/mesh/AMR/NativePublicationSession.hh"
#include "geometry/mesh/NativeSparseFragments.hh"
#include "geometry/mesh/SparseNativeMesh.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class NativeSparseAssembler;
struct NativeNumberingPoint
{
    explicit NativeNumberingPoint(const geometry_detail::NativeConstructionAdmission& a) : key(a.allocator<char>()) {}
    geometry_detail::NativeConstructorVector<char> key;
    PLIC::PointIdentity<ArithmeticMode::Ordinary> point =
        PLIC::details::GeometryIdentityAccess::point(PLIC::details::OrdinaryIdentityPoint{0, 0, 0});
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation;
};
struct NativeNumberingOccurrence
{
    const void* address = nullptr;
    std::weak_ptr<const void> control;
    std::size_t slot = 0;
    bool side = false;
};
struct NativeNumberingPiece
{
    explicit NativeNumberingPiece(const geometry_detail::NativeConstructionAdmission& a)
        : points(a.allocator<std::shared_ptr<const NativeNumberingPoint>>()),
          second_points(a.allocator<std::shared_ptr<const NativeNumberingPoint>>()),
          first_native_nodes(a.allocator<std::uint64_t>()), second_native_nodes(a.allocator<std::uint64_t>())
    {
    }
    NativeNumberingOccurrence occurrence;
    PLIC::details::NativeWeakOwnerObservation page_observation;
    NativeSourceTicket first_ticket, second_ticket;
    NativeIncidenceTicket first_incidence, second_incidence;
    std::shared_ptr<const NativeFragmentRecord> first_fragment, second_fragment;
    NativeSparsePhysicalFace physical;
    geometry_detail::NativeConstructorVector<std::shared_ptr<const NativeNumberingPoint>> points, second_points;
    geometry_detail::NativeConstructorVector<std::uint64_t> first_native_nodes, second_native_nodes;
    std::uint64_t first_native_face = UINT64_MAX, second_native_face = UINT64_MAX;
    int first_sign = 1, second_sign = -1;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation;
};
struct NativeNumberingPieceRow
{
    std::shared_ptr<const NativeNumberingPiece> record;
};
struct NativeNumberingPieceLess
{
    bool operator()(const NativeNumberingPieceRow&, const NativeNumberingPieceRow&) const;
};
struct NativeNumberingPointRow
{
    std::shared_ptr<const NativeNumberingPoint> record;
};
struct NativeNumberingPointLess
{
    bool operator()(const NativeNumberingPointRow& a, const NativeNumberingPointRow& b) const
    {
        return std::lexicographical_compare(
            a.record->key.begin(), a.record->key.end(), b.record->key.begin(), b.record->key.end());
    }
};
struct NativeNumberingBoundary
{
    geometry_detail::NativeConstructorStorageOwner metadata;
    std::string name;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation;
};
struct NativeNumberingBoundaryRow
{
    std::shared_ptr<const NativeNumberingBoundary> record;
};
struct NativeNumberingBoundaryLess
{
    bool operator()(const NativeNumberingBoundaryRow& a, const NativeNumberingBoundaryRow& b) const
    {
        return a.record->name < b.record->name;
    }
};
struct NativeNumberingRemotePiece
{
    explicit NativeNumberingRemotePiece(const geometry_detail::NativeConstructionAdmission& a)
        : native_nodes(a.allocator<std::uint64_t>())
    {
    }
    NativeIncidenceTicket ticket;
    NativeSourceTicket source_ticket;
    std::shared_ptr<const NativeRemoteIncidenceRecord> physical_record;
    std::shared_ptr<const NativeFragmentRecord> fragment;
    std::size_t slot = 0;
    NativeSparsePhysicalFace physical;
    std::uint64_t native_face = 0;
    int sign = 1;
    bool native_is_second = false;
    geometry_detail::NativeConstructorVector<std::uint64_t> native_nodes;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation;
};
struct NativeNumberingRemoteRow
{
    std::shared_ptr<const NativeNumberingRemotePiece> record;
};
struct NativeNumberingRemoteLess
{
    bool operator()(const NativeNumberingRemoteRow& a, const NativeNumberingRemoteRow& b) const
    {
        if (a.record->ticket != b.record->ticket)
            return a.record->ticket < b.record->ticket;
        return a.record->slot < b.record->slot;
    }
};
struct NativeNumberingPieceRoute
{
    const NativeNumberingPiece* record = nullptr;
    std::uint64_t face_gid = 0, first_gid = 0, second_gid = MultiRegionMesh::invalid_cell_id();
    std::size_t nodes_begin = 0, nodes_size = 0, second_begin = 0, second_size = 0;
};
struct NativeNumberingPointRoute
{
    const NativeNumberingPoint* record = nullptr;
    std::uint64_t gid = 0;
};
using NativeNumberingPieceEmit =
    CartesianBorrowedCallback<const CanonicalSubfaceView&, const NativeNumberingPieceRoute&, std::size_t>;
/** Private admitted shared adapter. Current arrays are scalar routes; immutable
 * point facts, images and physical overrides remain in shared output pages.
 * Source observations retain only fixed control lifetimes. Wire facts and
 * boundary names are borrowed from the current operation's shared snapshot.
 */
class SIMPLEFLUID_LOCAL NativePublicationNumbering final
{
public:
    using PiecePages = persistent_ordered_detail::OrderedPages<NativeNumberingPieceRow, NativeNumberingPieceLess>;
    using PointPages = persistent_ordered_detail::OrderedPages<NativeNumberingPointRow, NativeNumberingPointLess>;
    using RemotePages = persistent_ordered_detail::OrderedPages<NativeNumberingRemoteRow, NativeNumberingRemoteLess>;
    using BoundaryPages =
        persistent_ordered_detail::OrderedPages<NativeNumberingBoundaryRow, NativeNumberingBoundaryLess>;
    explicit NativePublicationNumbering(const geometry_detail::NativeConstructionAdmission& a)
        : pieces(a.allocator<NativeNumberingPieceRoute>()), points(a.allocator<NativeNumberingPointRoute>()),
          loops(a.allocator<std::uint64_t>())
    {
    }
    void visit_current_pieces(const AdaptiveSubfaceSnapshot&, const NativeNumberingPieceEmit&) const;
    [[nodiscard]] std::span<const NativeNumberingPieceRoute> local_pieces() const noexcept { return pieces; }
    [[nodiscard]] std::span<const NativeNumberingPointRoute> local_points() const noexcept { return points; }
    [[nodiscard]] std::span<const std::uint64_t> node_routes() const noexcept { return loops; }
    [[nodiscard]] std::uint64_t global_face_count() const noexcept { return global_faces; }
    [[nodiscard]] std::uint64_t global_node_count() const noexcept { return global_nodes; }

private:
    friend class NativeSparseAssembler;
    friend class NativePublicationOutputGraph;
    friend class NativePublicationExchange;
    struct LocalIncidenceRoute
    {
        const CanonicalSubfaceView* source = nullptr;
        std::size_t slot = 0;
    };
    void index_local_incidence(std::span<std::size_t>) const;
    void select_local_incidence(const NativeCellIncidenceProof&, std::span<const std::size_t>,
        geometry_detail::NativeConstructorVector<LocalIncidenceRoute>&) const;
    void visit_local_incidence(std::span<const LocalIncidenceRoute>, const NativeIncidenceEmit&) const;
    PiecePages::Ptr piece_root;
    PointPages::Ptr point_root;
    RemotePages::Ptr remote_root;
    BoundaryPages::Ptr boundary_root;
    std::shared_ptr<const NativeSparseFragments> fragments;
    geometry_detail::NativeConstructorVector<NativeNumberingPieceRoute> pieces;
    geometry_detail::NativeConstructorVector<NativeNumberingPointRoute> points;
    geometry_detail::NativeConstructorVector<std::uint64_t> loops;
    std::uint64_t global_faces = 0, global_nodes = 0;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
