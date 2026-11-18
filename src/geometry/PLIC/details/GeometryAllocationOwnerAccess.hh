/** @file GeometryAllocationOwnerAccess.hh
 * @brief Internal scoped census of admitted geometry allocation owners.
 */
#pragma once
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/details/GeometryIndexControls.hh"

#include <cstdint>
#include <functional>

namespace SimpleFluid::geometry_detail
{
class NativeAllocationEnvelope;
class NativeConstructorStorage;
class NativeConstructionAdmission;
struct NativeAllocationState;
} // namespace SimpleFluid::geometry_detail
namespace SimpleFluid::PLIC::details
{
class NativePaidOwnerAuthority;
class NativeGeometryAccounting;
class GeometryOwnerGraphEditor;
class NativePaidExtentCharge;
class NativeWeakOwnerObservation;
/** Restricts a fresh ledger's bookkeeping independently of its payload cap.
 * The existing retain calls still check complete payload size and ownership.
 * Borrowed identities remain valid while the ledger pins the allocations. */
struct GeometryAllocationWork
{
    std::size_t pushes = 0, admissions = 0, pending_visits = 0;
    std::size_t import_calls = 0, import_rows = 0, export_calls = 0, export_rows = 0, manifest_visit_rows = 0;
    std::size_t metadata_credit_queries = 0, metadata_credit_seed_rows = 0, metadata_credit_created = 0;
    std::size_t paired_payload_credits = 0, import_fallbacks = 0, import_delta_rows = 0;
    std::size_t import_delta_pages = 0, certified_import_hits = 0;
    std::size_t manifest_lookup_pages = 0, index_control_lookup_pages = 0;
    std::size_t control_table_enumeration_rows = 0;
    std::size_t reservation_path_pages = 0;
};
class GeometryAllocationOwnerAccess
{
public:
    enum class Work
    {
        Push,
        Admit,
        PendingVisit,
        Import,
        ImportRow,
        Export,
        ExportRow,
        ManifestVisitRow,
        MetadataCreditQuery,
        MetadataCreditSeedRow,
        MetadataCreditCreated,
        PairedPayloadCredit,
        ImportFallback,
        ImportDeltaRow,
        ImportDeltaPage,
        CertifiedImportHit,
        ManifestLookupPage,
        IndexControlLookupPage,
        ControlTableEnumerationRow,
        ReservationPathPage
    };
    /** Thread-local actual event counts; separate from logical retained owner counts. */
    [[nodiscard]] static GeometryAllocationWork work_statistics() noexcept;
    static void reset_work_statistics() noexcept;
    static void observe_work(Work, std::size_t count = 1) noexcept;
    /** A failed owner-graph mutation cannot resume admission through its ledger. */
    static void poison(GeometryAllocationLedger&) noexcept;
    /** Retain observed own-index metadata through the full import checks. */
    static void retain_manifest_metadata(GeometryAllocationLedger&, std::uint8_t channel,
        const std::shared_ptr<const void>&, std::size_t payload, std::size_t admitted, std::size_t& used,
        std::size_t limit);
    /** Explicit diagnostic census, never called by ordinary cache queries. */
    [[nodiscard]] static bool imported_metadata_seeded(const GeometryAllocationLedger&) noexcept;
    static void visit_metadata_credits(
        const GeometryAllocationLedger&, const std::function<void(const std::shared_ptr<const void>&,
                                             std::size_t payload, std::size_t admitted, bool imported_bookkeeping)>&);

    static void admit_graph_owner(GeometryAllocationLedger&, std::uint8_t channel, std::uint8_t kind,
        const std::shared_ptr<const void>&, std::size_t bytes, std::size_t& used, std::size_t limit,
        bool prepaid = false);
    static void retain_weak_observation(GeometryAllocationLedger&, std::uint8_t channel,
        const NativeWeakOwnerObservation&, std::size_t& used, std::size_t limit);
    [[nodiscard]] static NativePaidExtentCharge paid_charge(const GeometryAllocationLedger&, std::uint8_t channel,
        std::uint8_t kind, const std::shared_ptr<const void>&, std::size_t bytes);
    [[nodiscard]] static std::shared_ptr<const geometry_detail::NativeAllocationState> native_state(
        const GeometryAllocationLedger&) noexcept;
    static void require_native_paid(const GeometryAllocationLedger&, std::uint8_t channel, std::uint8_t kind,
        const std::shared_ptr<const void>&, std::size_t);
    static void begin_native_bookkeeping(GeometryAllocationLedger&, const geometry_detail::NativeConstructionAdmission&,
        std::size_t& separate_logical_cursor, std::size_t logical_limit);
    static void begin(GeometryAllocationLedger&, std::size_t& cumulative, std::size_t bookkeeping_limit);
    /** Capture a fully prepaid payload graph under the caller's finite budget.
     * Own-index exclusive/control credits are prepaid as well. Every newly
     * constructed ledger map, credit-registry and scratch allocation is charged;
     * exclusive payload totals and strong owner pins remain complete. Imported
     * manifests must also have prepaid their shared index metadata. */
    static void begin_prepaid_capture(GeometryAllocationLedger&, std::size_t& cumulative, std::size_t byte_limit);
    static void visit(const GeometryAllocationLedger&,
        const std::function<void(std::uint8_t kind, const void* owner, std::size_t bytes)>&);
    /** Record an immutable owner whose allocation was already charged in the
     * same cumulative operation. Only ledger bookkeeping is charged here; a
     * later normal retain of the same owner then deduplicates instead of
     * charging the identical persistent bytes twice. */
    static void admit_prepaid(GeometryAllocationLedger&, const std::shared_ptr<const void>& owner,
        std::size_t exclusive_bytes, std::size_t& used, std::size_t limit);

private:
    friend class GeometryOwnerGraphEditor;
    /** Owner-created callback for index storage whose constructor already
     * reserved its allowance against the same operation cursor. */
    static void retain_created_metadata(GeometryAllocationLedger&, std::uint8_t channel,
        const std::shared_ptr<const void>&, std::size_t payload, std::size_t admitted, std::size_t& used,
        std::size_t limit);
    static void register_native_metadata(
        GeometryAllocationLedger&, std::uint8_t channel, const std::shared_ptr<const void>&, std::size_t);
    static void authorize_native_fixed(
        GeometryAllocationLedger&, const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>&);
    static void authorize_native_storage(GeometryAllocationLedger&,
        const std::shared_ptr<geometry_detail::NativeConstructorStorage>&,
        const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>&);
    static void authorize_native_semantic(GeometryAllocationLedger&, std::uint8_t, std::uint8_t,
        const std::shared_ptr<const void>&, const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>&);
    static void authorize_native_identity(GeometryAllocationLedger&, std::uint8_t, const GeometryIdentityBackingPtr&);
    static void authorize_native_state(
        GeometryAllocationLedger&, const std::shared_ptr<const geometry_detail::NativeAllocationState>&);
    friend class NativeGeometryAccounting;
    static void begin_native_capture(GeometryAllocationLedger&, const geometry_detail::NativeConstructionAdmission&,
        NativePaidOwnerAuthority&, std::size_t& cursor, std::size_t limit, std::uint8_t channel = 0);
    friend class GeometryOwnerUnion;
    friend class GeometryOwnerAdmissionPair;
    static GeometryAllocationLedger* paired_peer(const GeometryAllocationLedger&) noexcept;
    static void retain_verified_metadata(GeometryAllocationLedger&, std::uint8_t, const std::shared_ptr<const void>&,
        std::size_t, std::size_t, std::size_t&, std::size_t);
    enum class MetadataRole { Created, Verified, Retained };
    static void retain_metadata_impl(GeometryAllocationLedger&, std::uint8_t, const std::shared_ptr<const void>&,
        std::size_t, std::size_t, std::size_t&, std::size_t, MetadataRole);
    friend class SimpleFluid::PLIC::GeometryAllocationLedger;
    friend class GeometryOwnerImportAccess;
    static void seed_imported_metadata(GeometryAllocationLedger&, std::size_t& used, std::size_t limit);
};
/** Operation-local peer credit. This borrowed link is removed on every exit;
 * it never survives in a snapshot/root or keeps another ledger/controller alive. */
class GeometryOwnerAdmissionPair
{
public:
    GeometryOwnerAdmissionPair(GeometryAllocationLedger&, GeometryAllocationLedger&);
    ~GeometryOwnerAdmissionPair();
    GeometryOwnerAdmissionPair(const GeometryOwnerAdmissionPair&) = delete;
    GeometryOwnerAdmissionPair& operator=(const GeometryOwnerAdmissionPair&) = delete;

private:
    friend class GeometryAllocationOwnerAccess;
    static thread_local GeometryOwnerAdmissionPair* active_;
    GeometryAllocationLedger* first_;
    GeometryAllocationLedger* second_;
    GeometryOwnerAdmissionPair* previous_;
};
} // namespace SimpleFluid::PLIC::details
