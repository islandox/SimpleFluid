/** @file GeometryOwnerGraph.hh
 * @brief Private first/last physical dependency editing for immutable owner graphs.
 */
#pragma once
#include "geometry/PLIC/CellGeometryView.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/GeometryOwnerManifest.hh"
#include "geometry/PLIC/details/NativePaidExtent.hh"
#include <array>
#include <exception>
#include <optional>
#include <span>
#include <vector>
namespace SimpleFluid::geometry_detail
{
struct NativeAllocationState;
class NativeAllocationEnvelope;
} // namespace SimpleFluid::geometry_detail
namespace SimpleFluid::Meshes::AMR::adaptive_detail { class NativePublicationOwnerGraph; }
namespace SimpleFluid::PLIC::allocation_detail { enum class Kind; }
namespace SimpleFluid::PLIC::details
{
struct GeometryOwnerGraphWork
{
    std::size_t dependency_entries = 0, edge_visits = 0;
    std::size_t first_transitions = 0, last_transitions = 0;
    geometry_demand::PageWork index, lookup;
};
/** Borrowed mutation context. Only the returned manifest retains physical owners.
 * Add replacement roots before retiring accepted roots to prune shared frontiers.
 * The caller publishes the completed manifest after every throwing edit succeeds.
 * Accepted values must be complete physical dependency graphs emitted by this
 * closed typed protocol, with exact incoming multiplicities. A flat allocation
 * census/export or arbitrary external-owner admission does not certify closure.
 */
class GeometryOwnerGraphEditor
{
public:
    GeometryOwnerGraphEditor(geometry_demand::GeometryOwnerManifest, std::uint8_t channel, std::size_t& used,
        std::size_t limit, GeometryAllocationLedger* = nullptr, GeometryOwnerGraphWork* = nullptr,
        std::shared_ptr<const geometry_detail::NativeAllocationState> native_administration = {},
        const NativeOwnerReceipts* previous_paid = nullptr);
    ~GeometryOwnerGraphEditor();
    /** Borrow a paired graph and scalar only while both editors live. Index
     * owner aliases keep exact physical cross-channel bytes without a census. */
    void pair_with(GeometryOwnerGraphEditor&, std::size_t& cross_payload_bytes);

    GeometryOwnerGraphEditor(const GeometryOwnerGraphEditor&) = delete;
    GeometryOwnerGraphEditor& operator=(const GeometryOwnerGraphEditor&) = delete;
    GeometryOwnerGraphEditor(GeometryOwnerGraphEditor&&) = delete;
    GeometryOwnerGraphEditor& operator=(GeometryOwnerGraphEditor&&) = delete;

    [[nodiscard]] static std::uint8_t external_kind() noexcept;
    [[nodiscard]] static std::uint8_t identity_payload_kind() noexcept;
    [[nodiscard]] static std::size_t edit_byte_bound(
        const geometry_demand::GeometryOwnerManifest&, std::size_t incoming_edges, std::size_t changed_keys,
        std::size_t typed_keys = 0);
    /** One actual shared pointer edge. The result gates child traversal. */
    bool external(const std::shared_ptr<const void>&, std::size_t bytes, bool retain, bool prepaid = false);
    /** Admit the one physical shared pointer held by a manifest record.
     * Stored payload reference multiplicity describes its graph, not this edge. */
    bool owner(const geometry_demand::OwnerRecord&, bool retain, bool prepaid = false);
    [[nodiscard]] std::size_t references(const geometry_demand::OwnerRecord&) const;
    template<ArithmeticMode M> void geometry(const ConvexGeometryPtr<M>&, bool retain);
    template<ArithmeticMode M>
    void catalog(const std::shared_ptr<const CellGeometryCatalog<M>>&, bool retain, bool metadata_prepaid = false);
    /** Construction receipt only, without an incoming graph edge or child
     * closure assertion. The later catalog walk still descends on graph first. */
    template<ArithmeticMode M>
    void prepay_catalog_page(const std::shared_ptr<const typename CellGeometryCatalog<M>::ExplicitPage>&);
    void cartesian(const std::shared_ptr<const CartesianCellSource>&, bool retain);
    void planar(const PlanarFacePtr<ArithmeticMode::Ordinary>&, bool retain);
    void identity(const PlaneSupportIdentity<ArithmeticMode::Ordinary>&, bool retain);
    void point_identity(const PointIdentity<ArithmeticMode::Ordinary>&, bool retain);
    void displacement_identity(const DisplacementIdentity<ArithmeticMode::Ordinary>&, bool retain);
    void native_fixed(
        const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>&, bool retain, bool prepaid = false);
    bool weak_observation(const NativeWeakOwnerObservation&, bool retain);
    void native_state(
        const std::shared_ptr<const geometry_detail::NativeAllocationState>&, bool retain, bool prepaid = false);
    void native_state_edges(
        const std::shared_ptr<const geometry_detail::NativeAllocationState>&, std::size_t references, bool retain);
    void record_paid_extent(const NativePaidExtentCharge&);
    void financial_pin(const NativePaidExtentCharge&, bool retain);
    void native_storage(const geometry_detail::NativeConstructorStorageOwner&);
    /** The finished graph's receipt root. */
    [[nodiscard]] const NativeOwnerReceipts& paid_receipts() const;
    [[nodiscard]] geometry_demand::GeometryOwnerManifest finish();

private:
    friend class NativeGeometryAccounting;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativeContactFundingAccess;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativePublicationOwnerGraph;
    /** Follow the first/last typed edge when plain physical pins may also
     * hold this allocation without visiting its children. */
    bool external_with_children(
        const std::shared_ptr<const void>&, std::size_t bytes, bool retain, bool prepaid = false);
    /** Aggregate roots can import flat component records and also visit
     * direct semantic owners of the same physical controls. */
    void distinguish_geometry_references() noexcept { impl_->separate_direct_edges = true; }

    friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativeSourceOwnerStore;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativePublicationOperation;
    void forget_paid_extent(const NativePaidExtentCharge&);
    void record_profile_charge(std::uint8_t kind, const std::shared_ptr<const void>& owner, std::size_t bytes,
        const std::shared_ptr<const void>& arena, const std::shared_ptr<const void>& immutable_lease,
        std::size_t charge_snapshot, std::uint64_t revision,
        NativePaidExtentCharge::FinancialProjector, NativePaidExtentCharge::FinancialInspector,
        NativePaidExtentCharge::FinancialValidator, const NativeWeakOwnerObservation& physical = {},
        const NativeWeakOwnerObservation& financial = {});
    void record_subset_charge(std::uint8_t kind, const std::shared_ptr<const void>& owner, std::size_t bytes,
        const std::shared_ptr<const void>& arena, const std::shared_ptr<const void>& detached_subset_lease,
        std::size_t charge_snapshot, std::uint64_t revision,
        NativePaidExtentCharge::FinancialProjector, NativePaidExtentCharge::FinancialInspector,
        NativePaidExtentCharge::FinancialValidator, bool retain = true,
        const NativeWeakOwnerObservation& physical = {}, const NativeWeakOwnerObservation& financial = {});
    // Bounded dependency traversal scratch belongs to the caller's stack.
    struct PendingOwner
    {
        std::uint8_t kind = 0;
        std::shared_ptr<const void> owner;
    };
    std::array<PendingOwner, geometry_demand::leaf_capacity> pending_{};
    struct Impl
    {
        geometry_demand::GeometryOwnerManifest result;
        std::uint8_t channel;
        std::size_t& used;
        std::size_t limit;
        GeometryAllocationLedger* ledger;
        GeometryOwnerGraphWork* work;
        std::shared_ptr<const geometry_detail::NativeAllocationState> native_administration;
        NativeOwnerReceipts paid;

        std::span<PendingOwner> inline_pending;
        geometry_demand::OwnerEdits edits{channel, native_administration};
        std::size_t pending_count = 0;
        Impl* partner = nullptr;
        std::size_t* cross_payload_bytes = nullptr;

        geometry_detail::NativeAdministrationVector<PendingOwner> pending;
        bool separate_direct_edges = false;
        bool adding = true, failed = false, finished = false, credited_predecessor = false;
        std::size_t financial_projection_depth = 0;
        std::size_t last_counter = 0;
        int construction_exceptions = std::uncaught_exceptions();
        struct Call
        {
            Impl& editor;
            int exceptions = std::uncaught_exceptions();
            explicit Call(Impl&);
            ~Call();
        };
        geometry_demand::Admission admission();
        std::optional<geometry_demand::ReceiptEdits> paid_edits;
        std::array<const NativeReceiptFact*, 8> recent_fixed_receipts{};
        bool paid_reference_edits = false;
        void edit_paid(const NativePaidExtentCharge&, NativeOwnerReceipts::Action = NativeOwnerReceipts::Action::Record);
        template<class Kind> std::size_t references(Kind, const std::shared_ptr<const void>&, std::size_t bytes) const;
        [[nodiscard]] geometry_demand::OwnerEdits::Edit& stage_owner(
            std::uint8_t kind, const std::shared_ptr<const void>&, std::size_t bytes);
        bool apply_edge(geometry_demand::OwnerEdits::Edit&, bool retain, bool prepaid, std::size_t references);

        template<class Kind>
        bool edge(Kind, const std::shared_ptr<const void>&, std::size_t bytes, bool retain, bool prepaid = false,
            std::size_t references = 1);
        struct TypedTransition { bool owner, children; };
        TypedTransition typed_edge(allocation_detail::Kind, const std::shared_ptr<const void>&, std::size_t bytes,
            bool retain, bool prepaid, std::size_t references = 1);
        bool direct_edge(allocation_detail::Kind, const std::shared_ptr<const void>&, std::size_t bytes,
            bool retain, bool prepaid = false);
        template<class Kind, class T> void push(Kind, const std::shared_ptr<const T>&, std::size_t&, std::size_t);
        [[nodiscard]] static bool source_base_children(bool transition) noexcept { return transition; }
        void native_typed_receipt(allocation_detail::Kind, const std::shared_ptr<const void>&,
            const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>&,
            const geometry_detail::NativeConstructorStorageOwner&, std::size_t&, std::size_t);
        void native_object_children(const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>&,
            const geometry_detail::NativeConstructorStorageOwner&, std::size_t&, std::size_t);
        bool native_fixed(const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>&, bool, bool);
        bool weak_observation(const NativeWeakOwnerObservation&, bool retain);
        bool native_state(const std::shared_ptr<const geometry_detail::NativeAllocationState>&, bool, bool,
            std::size_t references = 1);
        template<GeometryIdentityKind Kind, ArithmeticMode Mode>
        void retain_identity(const GeometryIdentity<Kind, Mode>&, std::size_t&, std::size_t);
        void retain_authorities(const std::vector<PlaneAuthority>&, std::size_t&, std::size_t);
        template<ArithmeticMode Mode> void drain(bool retain);
    } state_;
    Impl* impl_ = &state_;
};
} // namespace SimpleFluid::PLIC::details
