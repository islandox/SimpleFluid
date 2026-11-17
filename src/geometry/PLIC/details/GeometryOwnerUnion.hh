/** Private immutable physical owner union and sealed root-store bindings. */
#pragma once
#include "geometry/PLIC/details/GeometryIndexControls.hh"
#include "geometry/PLIC/details/GeometryOwnerGraph.hh"
#include <array>
#include <cstdint>
namespace SimpleFluid::Meshes::AMR
{
struct AdaptiveGeometrySnapshotCache;
namespace adaptive_detail
{
class NativeGeometryRootStore;
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
    friend class GeometryOwnerRootBinding;
    friend class GeometryOwnerImportAccess;
    void require_seal() const;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::GeometrySnapshotAccess;
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

private:
    friend struct SimpleFluid::Meshes::AMR::AdaptiveGeometrySnapshotCache;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativeGeometryRootStore;
    friend class GeometryOwnerUnion;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::GeometrySnapshotAccess;
    GeometryOwnerRootBinding(std::shared_ptr<const void> header, std::size_t header_bytes, OwnerManifestPair payload);
    std::shared_ptr<const void> header_;
    std::size_t header_bytes_ = 0;
    OwnerManifestPair payload_;
};
struct GeometryOwnerUnionWork
{
    std::array<GeometryOwnerGraphWork, 2> channels;
    geometry_demand::PageWork controls;
    std::size_t imported_rows = 0;
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
    /** Construction-only envelope for an already credited Ordinary root edit.
     * Excludes physical incoming payload, which the caller reserves separately.
     * changed_keys bounds all touched identities, including cancelling deltas;
     * new_input_owners includes any new binding header and input index owners.
     * anticipated_additions covers rows added before a future retirement;
     * unrelated populated-ledger fallback enumeration is outside this contract. */
    [[nodiscard]] std::size_t administration_byte_bound(std::size_t edges, std::size_t changed_keys,
        std::size_t input_index_controls, std::size_t new_input_owners, std::size_t anticipated_additions = 0) const;
    [[nodiscard]] GeometryOwnerUnion staged(const GeometryOwnerRootBinding* retire,
        const GeometryOwnerRootBinding* retain, GeometryAllocationLedger&, GeometryAllocationLedger&, std::size_t& used,
        std::size_t limit, GeometryOwnerUnionWork* = nullptr) const;
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
    OwnerManifestPair owners_;
    GeometryIndexControls controls_;
    std::size_t resident_bytes_ = 0;
    std::uint64_t generation_ = 0;
};
} // namespace SimpleFluid::PLIC::details
