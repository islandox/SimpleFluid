/** Private immutable physical owner union and sealed root-store bindings. */
#pragma once
#include "geometry/PLIC/details/GeometryIndexControls.hh"
#include "geometry/PLIC/details/GeometryOwnerGraph.hh"
#include <array>
#include <cstdint>
#include <optional>
#include <span>
namespace SimpleFluid::Meshes::AMR
{
struct AdaptiveGeometrySnapshotCache;
namespace adaptive_detail
{
class NativeGeometryRootStore;
class NativeSourceOwnerStore;
class GeometrySnapshotAccess;
} // namespace adaptive_detail
} // namespace SimpleFluid::Meshes::AMR
namespace SimpleFluid::PLIC::details
{
struct OwnerManifestPair
{
    std::array<geometry_demand::GeometryOwnerManifest, 2> owners;
    [[nodiscard]] std::size_t payload_bytes() const;
    [[nodiscard]] std::size_t logical_payload_bytes() const;
    [[nodiscard]] std::size_t index_admission_bytes() const;
    [[nodiscard]] std::size_t overlap_bytes() const;

private:
    friend class GeometryOwnerUnion;
    friend class NativeGeometryAccounting;
    friend class GeometryOwnerRootBinding;
    friend class GeometryOwnerImportAccess;
    void require_seal() const;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::GeometrySnapshotAccess;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativeSourceOwnerStore;
    static OwnerManifestPair sealed(std::array<geometry_demand::GeometryOwnerManifest, 2>, std::size_t overlap);
    std::array<std::weak_ptr<const void>, 2> sealed_roots_;
    std::size_t overlap_ = 0;
    bool sealed_ = false;
};
/** Only authenticated owning stores can issue a binding. Payload manifests
 * exclude their root header and their own indexes. Neither payload may retain
 * another binding's administrative manifest value as a body allocation. */
class GeometryOwnerRootBinding
{
public:
    GeometryOwnerRootBinding() = default;
    [[nodiscard]] explicit operator bool() const noexcept { return bool(header_); }
    [[nodiscard]] const OwnerManifestPair& payload() const noexcept { return payload_; }
    [[nodiscard]] const std::shared_ptr<const void>& header() const noexcept { return header_; }
    [[nodiscard]] const NativeOwnerReceipts* paid_receipts() const noexcept { return paid_; }
    [[nodiscard]] const NativePaidExtentGroup* source_group() const noexcept { return group_; }

private:
    friend struct SimpleFluid::Meshes::AMR::AdaptiveGeometrySnapshotCache;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativeGeometryRootStore;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativeSourceOwnerStore;
    friend class GeometryOwnerUnion;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::GeometrySnapshotAccess;
    GeometryOwnerRootBinding(std::shared_ptr<const void> header, std::size_t header_bytes, OwnerManifestPair payload,
        const NativeOwnerReceipts* paid = nullptr, const NativePaidExtentGroup* source_group = nullptr);
    std::shared_ptr<const void> header_;
    std::size_t header_bytes_ = 0;
    OwnerManifestPair payload_;
    // Borrowed immutable fields owned by the actual strong header above.
    const NativeOwnerReceipts* paid_ = nullptr;
    const NativePaidExtentGroup* group_ = nullptr;
};
struct GeometryOwnerUnionWork
{
    std::array<GeometryOwnerGraphWork, 2> channels;
    geometry_demand::PageWork controls;
    std::size_t imported_rows = 0;
    geometry_demand::PageWork native_administration;
    std::size_t native_domain_batches = 0, native_domain_edges = 0;
};
/** Maintained exact native allocator-domain closure for persistent indexes.
 * Reusable by source/contact owner stores. It retains only detached scalar
 * domains and its own immutable index; it never pins the source/body pair. */
class GeometryNativeAdministration
{
public:
    GeometryNativeAdministration() = default;
    [[nodiscard]] std::size_t resident_bytes() const noexcept;
    [[nodiscard]] std::size_t native_paid_bytes(const std::shared_ptr<const void>& actual_arena) const;
    [[nodiscard]] std::size_t direct_h_paid_bytes(const std::shared_ptr<const void>& actual_arena) const;
    [[nodiscard]] std::size_t publication_h_paid_bytes(const std::shared_ptr<const void>& actual_arena) const;
    [[nodiscard]] bool admits(const std::shared_ptr<const void>&, std::size_t bytes) const;
    [[nodiscard]] std::size_t uncovered_owner_bytes(
        const std::shared_ptr<const void>&, std::size_t bytes, const OwnerManifestPair& body) const;
    void visit(geometry_demand::MetadataSink) const;
    [[nodiscard]] GeometryNativeAdministration updated(const OwnerManifestPair& previous_body,
        const OwnerManifestPair& body, const GeometryIndexControls& previous_controls,
        const GeometryIndexControls& controls, std::size_t& used, std::size_t limit,
        const std::shared_ptr<const geometry_detail::NativeAllocationState>&, GeometryOwnerUnionWork&,
        const NativeOwnerReceipts* previous_receipts = nullptr, const NativeOwnerReceipts* receipts = nullptr,
        const NativeOwnerReceipts* previous_bounded_receipts = nullptr,
        const NativeOwnerReceipts* bounded_receipts = nullptr, std::shared_ptr<const void> source_group_header = {},
        const NativePaidExtentGroup* source_group = nullptr,
        geometry_demand::ReceiptEdits* body_presence = nullptr) const;

private:
    friend class GeometryOwnerUnion;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::GeometrySnapshotAccess;
    geometry_demand::GeometryOwnerManifest owners;
    std::shared_ptr<const geometry_detail::NativeAllocationState> current;
    std::size_t current_references = 0, body_credit = 0, current_paid_bytes = 0, current_publication_paid_bytes = 0;
    NativeOwnerReceipts domain_paid, paid;
    NativePaidOriginTotals origins, direct_origins, publication_origins, group_overlap;
    std::shared_ptr<const void> source_group_header;
    const NativePaidExtentGroup* source_group = nullptr;
};
/** A union holds current owner/index values only. Whole-union imports are
 * operation-local; persistent outputs import only an authenticated binding. */
class GeometryOwnerUnion
{
public:
    GeometryOwnerUnion() = default;
    [[nodiscard]] const OwnerManifestPair& manifests() const noexcept { return owners_; }
    [[nodiscard]] const GeometryIndexControls& index_controls() const noexcept { return controls_; }
    [[nodiscard]] std::size_t resident_bytes() const noexcept { return resident_bytes_; }
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
    [[nodiscard]] std::size_t stored_index_control_bytes() const noexcept { return controls_.control_bytes(); }
    void visit_administration(geometry_demand::MetadataSink) const;
    [[nodiscard]] bool admits_native_administration(const std::shared_ptr<const void>&, std::size_t bytes) const;
    [[nodiscard]] std::size_t native_administration_bytes() const noexcept;
    [[nodiscard]] std::size_t native_paid_bytes(const std::shared_ptr<const void>& actual_arena) const;
    [[nodiscard]] std::size_t direct_h_paid_bytes(const std::shared_ptr<const void>& actual_arena) const;
    /** Intersection with actual native publication mirror requests in this arena. */
    [[nodiscard]] std::size_t publication_h_paid_bytes(const std::shared_ptr<const void>& actual_arena) const;
    [[nodiscard]] std::size_t native_group_covered_bytes(const std::shared_ptr<const void>& actual_arena) const;
    /** Construction-only envelope for an already credited Ordinary root edit.
     * Excludes physical incoming payload, which the caller reserves separately.
     * changed_keys bounds all touched identities, including cancelling deltas;
     * new_input_owners includes any new binding header and input index owners.
     * anticipated_additions covers rows added before a future retirement.
     * Engage the typed allowance only for a native metadata edit.
     * unrelated populated-ledger fallback enumeration is outside this contract. */
    [[nodiscard]] std::size_t administration_byte_bound(std::size_t edges, std::size_t changed_keys,
        std::size_t input_index_controls, std::size_t new_input_owners, std::size_t anticipated_additions = 0,
        std::optional<std::size_t> anticipated_typed_additions = {}) const;
    [[nodiscard]] GeometryOwnerUnion staged(const GeometryOwnerRootBinding* retire,
        const GeometryOwnerRootBinding* retain, GeometryAllocationLedger&, GeometryAllocationLedger&, std::size_t& used,
        std::size_t limit, GeometryOwnerUnionWork* = nullptr) const;
    struct Replacement
    {
        const GeometryOwnerRootBinding* retire;
        const GeometryOwnerRootBinding* retain;
    };
    /** Apply one complete replacement to the physical and financial owner indexes. */
    [[nodiscard]] GeometryOwnerUnion staged(std::span<const Replacement>, GeometryAllocationLedger&,
        GeometryAllocationLedger&, std::size_t& used, std::size_t limit, GeometryOwnerUnionWork* = nullptr) const;
    void import(GeometryAllocationLedger&, GeometryAllocationLedger&, std::size_t& used, std::size_t limit) const;
    void import_subset(const GeometryOwnerRootBinding&, GeometryAllocationLedger&, GeometryAllocationLedger&,
        std::size_t& used, std::size_t limit) const;
    void certify_subset(const GeometryOwnerRootBinding&, GeometryAllocationLedger&, GeometryAllocationLedger&,
        std::size_t& used, std::size_t limit) const;
    void certify(const GeometryOwnerRootBinding& root, GeometryAllocationLedger& ordinary,
        GeometryAllocationLedger& bounded, std::size_t& used, std::size_t limit) const
    {
        certify_subset(root, ordinary, bounded, used, limit);
    }

private:
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::GeometrySnapshotAccess;
    struct SourceRoot
    {
        geometry_demand::GeometryOwnerManifest root;
        std::size_t references = 0;
    };
    // Six snapshot bindings, one separately typed native binding, one source
    // input composite and two unpublished replacements, each with two channels.
    static constexpr std::size_t source_capacity = 2 * (6 + 1 + 1 + 2);
    std::array<SourceRoot, source_capacity> sources_;
    void edit_sources(const GeometryOwnerRootBinding&, bool retain);
    void validate(const GeometryOwnerRootBinding&) const;
    GeometryNativeAdministration native_administration_;
    OwnerManifestPair owners_;
    GeometryIndexControls controls_;
    std::array<NativeOwnerReceipts, 2> paid_body_;
    std::shared_ptr<const void> source_group_header_;
    const NativePaidExtentGroup* source_group_ = nullptr;
    std::size_t resident_bytes_ = 0;
    std::uint64_t generation_ = 0;
};
} // namespace SimpleFluid::PLIC::details
