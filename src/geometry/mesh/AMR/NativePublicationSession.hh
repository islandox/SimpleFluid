/** @file NativePublicationSession.hh
 * @brief Private staged producer for detached source and completed incidence facts.
 */
#pragma once
#include "geometry/PLIC/details/NativeWeakOwnerObservation.hh"
#include "geometry/mesh/AMR/AdaptiveSubfaceSnapshot.hh"
#include "geometry/mesh/AMR/NativePublicationProof.hh"
#include "geometry/mesh/AMR/NativePublicationViewData.hh"
#include "geometry/mesh/NativeFragmentAccess.hh"
#include <algorithm>
#include <mutex>
namespace SimpleFluid::Meshes::AMR
{
class DistributedAdaptiveHierarchy;
namespace adaptive_detail
{
class NativeSourceCoverage;
class NativeSourceSubsetCoverage;
class NativeSourceCoverageTransaction;

struct NativeSubfaceOccurrence
{
    std::shared_ptr<const void> control;
    std::size_t slot = 0;
    bool side = false;
};
using NativeSourceFaceRoute = NativeKernelSourceFace;
/** Source equivalence and metrics; no expanded procedural shell is retained. */
class SIMPLEFLUID_LOCAL NativeCellSourceProof final
{
    class ConstructionKey
    {
        friend class NativePublicationSession;
        ConstructionKey() = default;

    public:
        ConstructionKey(const ConstructionKey&) = default;
    };

public:
    explicit NativeCellSourceProof(ConstructionKey) {}
    [[nodiscard]] const NativeSourceBinding& binding() const noexcept { return *binding_; }
    [[nodiscard]] const AdaptiveLeaf& source() const noexcept { return binding_->source(); }
    [[nodiscard]] const auto& fragment() const noexcept { return fragment_; }
    [[nodiscard]] const AdaptiveCellMetrics& metrics() const noexcept { return metrics_; }
    [[nodiscard]] std::uint64_t native_cell() const noexcept { return native_cell_; }
    [[nodiscard]] std::span<const NativeSourceFaceRoute> faces() const noexcept { return faces_; }
    [[nodiscard]] NativeSourceTicket ticket() const noexcept { return ticket_; }

private:
    friend class NativePublicationSession;
    friend class NativePublicationExchange;
    std::shared_ptr<const NativeSourceBinding> binding_;
    std::shared_ptr<const NativeFragmentRecord> fragment_;
    AdaptiveCellMetrics metrics_;
    std::uint64_t native_cell_ = 0;
    geometry_detail::NativeConstructorStorageOwner metadata_;
    std::vector<NativeSourceFaceRoute> faces_;
    std::weak_ptr<const NativeKernelAuthority> authority_;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> authority_allocation_;
    GeometryConstructionControls controls_;
    NativeSourceTicket ticket_;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation_;
};
struct NativeIncidenceTicket
{
    int issuer = -1;
    std::uint64_t transaction = 0, ordinal = 0;
    auto operator<=>(const NativeIncidenceTicket&) const = default;
};
struct NativeIncidenceNumericPiece
{
    std::uint64_t face_gid = 0, first_gid = 0;
    std::uint64_t second_gid = std::numeric_limits<std::uint64_t>::max();
    std::span<const std::uint64_t> first_nodes, second_nodes;
};
using NativeIncidenceEmit = CartesianBorrowedCallback<const CanonicalSubfaceView&, const NativeIncidenceNumericPiece&>;
struct NativeRemoteFaceFacts
{
    std::vector<PLIC::PointIdentity<ArithmeticMode::Ordinary>> points;
    MeshUtils::Vec3 centroid{}, area_vector{};
    std::optional<std::uint64_t> native_face;
    bool native_is_second = false, whole = false;
};
class SIMPLEFLUID_LOCAL NativeRemoteIncidenceRecord final
{
    class ConstructionKey
    {
        friend class NativePublicationExchange;
        ConstructionKey() = default;

    public:
        ConstructionKey(const ConstructionKey&) = default;
    };

public:
    explicit NativeRemoteIncidenceRecord(ConstructionKey) {}

private:
    friend class NativePublicationExchange;
    friend class NativeCellIncidenceProof;
    friend class NativePublicationSession;
    geometry_detail::NativeConstructorStorageOwner metadata_;
    std::vector<AdaptiveTile> tiles_;
    std::vector<NativeRemoteFaceFacts> facts_;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation_;
};
/** Exact occurrences include the shared piece page control, slot, side,
 * source proof control, images and label bytes carried by those page owners. */
class SIMPLEFLUID_LOCAL NativeCellIncidenceProof final
{
    class ConstructionKey
    {
        friend class NativePublicationSession;
        friend class NativePublicationExchange;
        ConstructionKey() = default;

    public:
        ConstructionKey(const ConstructionKey&) = default;
    };

public:
    explicit NativeCellIncidenceProof(ConstructionKey) {}
    [[nodiscard]] const NativeCellSourceProof& source() const noexcept { return *source_; }
    [[nodiscard]] std::span<const CanonicalSubfaceView> pieces() const noexcept { return pieces_; }
    [[nodiscard]] const std::shared_ptr<const NativeRemoteIncidenceRecord>& remote_record() const noexcept
    {
        return remote_;
    }
    [[nodiscard]] std::span<const AdaptiveTile> remote_tiles() const noexcept
    {
        return remote_ ? std::span<const AdaptiveTile>(remote_->tiles_) : std::span<const AdaptiveTile>{};
    }
    [[nodiscard]] NativeIncidenceTicket ticket() const noexcept { return ticket_; }
    [[nodiscard]] std::span<const PLIC::PointIdentity<ArithmeticMode::Ordinary>> remote_points(std::size_t index) const
    {
        return remote_facts(index).points;
    }
    [[nodiscard]] std::uint64_t remote_native_face(std::size_t index) const
    {
        return remote_facts(index).native_face.value();
    }
    [[nodiscard]] bool remote_cut_incidence(std::size_t index) const { return !remote_facts(index).whole; }
    [[nodiscard]] int remote_native_sign(std::size_t index) const
    {
        const auto& facts = remote_facts(index);
        const auto found = std::find_if(source_->faces().begin(), source_->faces().end(),
            [&](const auto& face) { return face.native_face == facts.native_face; });
        if (found == source_->faces().end())
            throw std::invalid_argument("Remote native face is outside its source proof.");
        return facts.native_is_second ? -found->outward_sign : found->outward_sign;
    }
    [[nodiscard]] const NativeRemoteFaceFacts& remote_facts(std::size_t index) const
    {
        if (!remote_)
            throw std::invalid_argument("Local incidence has no remote face facts.");
        return remote_->facts_.at(index);
    }

private:
    friend class NativePublicationSession;
    friend class NativePublicationExchange;
    std::shared_ptr<const NativeCellSourceProof> source_;
    std::shared_ptr<const NativeRemoteIncidenceRecord> remote_;
    NativeIncidenceTicket ticket_;
    geometry_detail::NativeConstructorStorageOwner metadata_;
    std::vector<CanonicalSubfaceView> pieces_;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation_;
};
struct NativePublicationIndexRow
{
    std::shared_ptr<const NativeCellSourceProof> source;
    std::shared_ptr<const NativeCellIncidenceProof> incidence;
    [[nodiscard]] const AdaptiveCellKey& key() const noexcept { return source->source().key; }
};
struct NativePublicationIndexLess
{
    bool operator()(const NativePublicationIndexRow& a, const NativePublicationIndexRow& b) const
    {
        return a.key() < b.key();
    }
};
class NativePublicationNumbering;
struct NativePublicationFragmentRow
{
    std::shared_ptr<const NativeFragmentDescriptor> descriptor;
    std::shared_ptr<const NativeFragmentRecord> fragment;
    GeometryConstructionControls controls;
    std::size_t references = 0;
};
struct NativePublicationFragmentLess
{
    bool operator()(const NativePublicationFragmentRow& a, const NativePublicationFragmentRow& b) const
    {
        if (a.descriptor.get() != b.descriptor.get())
            return std::less<const void*>{}(a.descriptor.get(), b.descriptor.get());
        return a.descriptor.owner_before(b.descriptor);
    }
};
struct SIMPLEFLUID_LOCAL NativePublicationIndexes
{
    using Pages = persistent_ordered_detail::OrderedPages<NativePublicationIndexRow, NativePublicationIndexLess>;
    Pages::Ptr sources;
    using Fragments =
        persistent_ordered_detail::OrderedPages<NativePublicationFragmentRow, NativePublicationFragmentLess>;
    Fragments::Ptr fragments;
    NativePublicationKeyCatalog owned_keys;
    NativeRemoteKeyCatalog remote;
    std::shared_ptr<const NativePublicationNumbering> numbering;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation;
    [[nodiscard]] const NativePublicationIndexRow* find(const AdaptiveCellKey&) const;
};
class NativePublicationReceipt;
class NativePublicationNumbering;
/** Source-controller cache only. No publication root captures this object. */
struct SIMPLEFLUID_LOCAL NativePublicationCache
{
    mutable std::mutex mutex;
    std::shared_ptr<const NativePublicationIndexes> indexes;
    std::shared_ptr<const NativePublicationReceipt> receipt;
    std::uint64_t revision = 0;
    const NativePublicationOperation* commit_token = nullptr;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation;
};
struct NativePublicationSourceObservations
{
    PLIC::details::NativeWeakOwnerObservation layout, identity, topology, ownership, authority, subfaces;
    void project(PLIC::details::GeometryOwnerGraphEditor&, bool retain) const;
};
struct SIMPLEFLUID_LOCAL NativePublicationInputs
{
    std::shared_ptr<const AdaptiveLeafLayout> layout;
    std::shared_ptr<const AdaptiveProviderKernel> kernel;
    std::shared_ptr<const AdaptiveSubfaceSnapshot> subfaces;
    std::shared_ptr<NativePublicationCache> cache;
    HierarchyObservation observation;
    Teuchos::RCP<const Teuchos::Comm<int>> comm;
    std::uint64_t cell_begin = 0, global_cells = 0;
    std::size_t byte_budget = 0;
    std::shared_ptr<const NativeSourceCoverage> source_coverage;
    NativePublicationSourceObservations fixed_observations;
    std::shared_ptr<const void> ownership_owner;
};
struct NativeDirectoryLookup;
class NativeSourceCoverage;
class NativeSourceSubsetCoverage;
class NativeSourceCoverageTransaction;
class NativeSparseAccess;
class SIMPLEFLUID_LOCAL NativeSparseSeal final
{
public:
    NativeSparseSeal(const NativeSparseSeal&) = default;
    [[nodiscard]] const NativeCellSourceProof& source() const noexcept { return *source_; }
    [[nodiscard]] const NativeCellIncidenceProof& incidence() const noexcept { return *incidence_; }

private:
    friend class NativePublicationSession;
    friend class NativeSparseAccess;
    NativeSparseSeal(
        std::shared_ptr<const NativeCellSourceProof> source, std::shared_ptr<const NativeCellIncidenceProof> incidence)
        : source_(std::move(source)), incidence_(std::move(incidence))
    {
    }
    std::shared_ptr<const NativeCellSourceProof> source_;
    std::shared_ptr<const NativeCellIncidenceProof> incidence_;
};
/** Only this class may construct the private sparse assembler's seal. B obtains
 * borrowed current source and incidence rows, never an owning draft/key map. */
class SIMPLEFLUID_LOCAL NativePublicationSession final
{
public:
    /** Standalone/component compatibility constructor. Production integration
     * passes its already admitted operation domain through the second overload. */
    explicit NativePublicationSession(NativePublicationInputs);
    NativePublicationSession(NativePublicationInputs, geometry_detail::NativeConstructionAdmission,
        PLIC::details::NativeGeometryAccounting* = nullptr, NativeSourceCoverageTransaction* = nullptr);
    [[nodiscard]] static std::shared_ptr<NativePublicationCache> make_cache(
        const geometry_detail::NativeConstructionAdmission&);
    [[nodiscard]] static std::shared_ptr<NativePublicationCache> copy_cache(
        const std::shared_ptr<NativePublicationCache>&, const geometry_detail::NativeConstructionAdmission&);
    static void project_receipt(const std::shared_ptr<const NativePublicationReceipt>&,
        PLIC::details::GeometryOwnerGraphEditor&, bool retain,
        PLIC::details::NativeGeometryAccounting* = nullptr);
    [[nodiscard]] static NativePublicationInputs capture(const DistributedAdaptiveHierarchy&);
    [[nodiscard]] static NativeDirectoryLookup lookup_cells(const DistributedAdaptiveHierarchy&,
        std::span<const AdaptiveCellKey* const>, const geometry_detail::NativeConstructionAdmission&);
    [[nodiscard]] const NativePublicationInputs& inputs() const noexcept { return inputs_; }
    [[nodiscard]] const geometry_detail::NativeConstructionAdmission& admission() const noexcept { return admission_; }
    [[nodiscard]] PLIC::details::NativeGeometryAccounting* geometry_accounting() const noexcept
    { return accounting_; }
    [[nodiscard]] std::shared_ptr<const NativeSourceSubsetCoverage> retain_source_subset(
        const PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>&);
    [[nodiscard]] bool qualified() const noexcept;
    [[nodiscard]] static NativeSubfaceOccurrence occurrence(const CanonicalSubfaceView&);
    static void project_indexes(const std::shared_ptr<const NativePublicationIndexes>&,
        PLIC::details::GeometryOwnerGraphEditor&, bool retain,
        PLIC::details::NativeGeometryAccounting* = nullptr);
    static void project_remote_physical(const std::shared_ptr<const NativeRemoteIncidenceRecord>&,
        PLIC::details::GeometryOwnerGraphEditor&, bool retain,
        PLIC::details::NativeGeometryAccounting* = nullptr);
    static void project_source(const std::shared_ptr<const NativeCellSourceProof>&,
        PLIC::details::GeometryOwnerGraphEditor&, bool retain, bool prepaid = false, PLIC::details::NativeGeometryAccounting* = nullptr);
    [[nodiscard]] const std::shared_ptr<const NativePublicationNumbering>& numbering() const noexcept
    {
        return indexes_->numbering;
    }
    void stage_numbering(std::shared_ptr<const NativePublicationNumbering>);
    void stage_remote_membership(std::span<const std::shared_ptr<const NativeCellSourceProof>>);
    [[nodiscard]] NativeSparseSeal seal(const std::shared_ptr<const NativeCellSourceProof>&,
        const std::shared_ptr<const NativeCellIncidenceProof>&) const;
    void recheck(const DistributedAdaptiveHierarchy&) const;
    [[nodiscard]] std::shared_ptr<const NativeCellSourceProof> source(const AdaptiveLeaf&);
    [[nodiscard]] std::shared_ptr<const NativeCellIncidenceProof> incidence(
        const std::shared_ptr<const NativeCellSourceProof>&, std::span<const CanonicalSubfaceView>);
    /** Invalidation consumes hierarchy's already complete edit delta. */
    static std::shared_ptr<NativePublicationCache> edited(const std::shared_ptr<NativePublicationCache>&,
        std::span<const AdaptiveLeafPages::EditView>, const geometry_detail::NativeConstructionAdmission&);
    [[nodiscard]] const std::shared_ptr<const NativePublicationIndexes>& indexes() const noexcept { return indexes_; }
    /** B calls this after building all output arrays, native providers, actual
     * map and partition. Stage can throw; commit after collective agreement is
     * a no-throw transfer into the source cache. */
    void stage_result(std::shared_ptr<NativePublicationViewData>, Teuchos::RCP<const DefaultTpetraTypes::map_type>,
        std::uint64_t collective_identity);
    void commit() noexcept;
    [[nodiscard]] AdaptiveNativePublicationView view() const;
    [[nodiscard]] static bool receipt_matches(const NativePublicationInputs&,
        const std::shared_ptr<const CompositePartition<>>&, const Teuchos::RCP<const DefaultTpetraTypes::map_type>&);

private:
    friend class NativePublicationOperation;
    friend class NativePublicationExchange;
    friend class NativeSparseAccess;
    void commit_locked() noexcept;
    [[nodiscard]] std::shared_ptr<const NativeSourceSubsetCoverage> retain_source_subset(
        const PLIC::details::OwnerManifestPair&, const PLIC::details::NativeOwnerReceipts&);
    std::shared_ptr<const NativeCellSourceProof> source_impl(const AdaptiveLeaf&, std::optional<NativeSourceTicket>);
    void compact_remote(const NativeCellSourceProof&, NativeRemoteIncidenceRecord&) const;
    void verify_incidence(
        const NativeCellSourceProof&, std::span<const PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>>) const;
    void replace(NativePublicationIndexRow);
    static bool same_occurrence(const CanonicalSubfaceView&, const CanonicalSubfaceView&) noexcept;
    NativePublicationInputs inputs_;
    geometry_detail::NativeConstructionAdmission admission_;
    PLIC::details::NativeGeometryAccounting* accounting_ = nullptr;
    NativeSourceCoverageTransaction* coverage_transaction_ = nullptr;
    std::shared_ptr<const NativePublicationIndexes> indexes_;
    std::shared_ptr<const NativePublicationReceipt> receipt_;
    std::shared_ptr<NativePublicationViewData> result_;
    std::uint64_t cache_revision_ = 0, issuance_ = 0, next_ordinal_ = 0;
};
} // namespace adaptive_detail
} // namespace SimpleFluid::Meshes::AMR
