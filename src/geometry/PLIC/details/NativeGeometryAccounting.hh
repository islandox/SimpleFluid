/** Closed bridge between monotone geometry work and retiring native admission. */
#pragma once
#include "geometry/PLIC/details/GeometryAllocationOwnerAccess.hh"
#include "geometry/PLIC/details/GeometryOwnerUnion.hh"
#include "geometry/details/NativeConstructorStorage.hh"
#include "geometry/details/NativeSchemaOwner.hh"
#include <map>
namespace SimpleFluid::geometry_detail
{
class NativeTpetraMapAdmission;
class NativeMeshHandleAccess;
} // namespace SimpleFluid::geometry_detail
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class NativePublicationOperation;
class NativeSourceOwnerStore;
class NativePublicationOutputGraph;
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
namespace SimpleFluid::PLIC::details
{
/** Borrowed exact accepted union plus disjoint residual source/contact extent.
 * Only the owning source/operation store can issue this certificate. The issuer
 * must retain and freeze the union and its source pins for the operation. */
class SealedSourcePrefix
{
public:
    SealedSourcePrefix(const SealedSourcePrefix&) = delete;
    SealedSourcePrefix(SealedSourcePrefix&&) noexcept = default;
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
    friend class NativeGeometryAccounting;
    friend class NativePaidOwnerAuthority;
    friend class SimpleFluid::geometry_detail::NativeTpetraMapAdmission;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativePublicationOperation;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativeSourceOwnerStore;
    SealedSourcePrefix(const GeometryOwnerUnion& current, std::size_t residual);
    void require_current() const;
    [[nodiscard]] bool admits_library_context(const std::shared_ptr<const void>&, std::size_t) const;
    void include_library_context_owner(const std::shared_ptr<const void>&, std::size_t, bool funded_by_source);
    struct LibraryOwner
    {
        std::shared_ptr<const void> owner;
        std::size_t bytes = 0;
    };
    const GeometryOwnerUnion* current_;
    std::uint64_t generation_;
    std::array<std::weak_ptr<const void>, 2> roots_;
    std::size_t bytes_;
    std::array<LibraryOwner, 16> library_owners_{};
    std::size_t library_owner_count_ = 0;
};
/** Operation-local proof registry. Rows are weak: this registry cannot pin
 * source controllers or prolong selected native payloads. Every record is
 * backed by exact native allocation receipts, never an asserted prepaid flag. */
class NativePaidOwnerAuthority
{
public:
    [[nodiscard]] bool admits(
        std::uint8_t channel, std::uint8_t kind, const std::shared_ptr<const void>&, std::size_t bytes) const;

private:
    friend class NativeGeometryAccounting;
    friend class GeometryAllocationOwnerAccess;
    friend struct SimpleFluid::PLIC::GeometryAllocationLedger::Impl;
    [[nodiscard]] bool admits_root(const geometry_demand::GeometryOwnerManifest&) const;
    void fixed(const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>&);
    void state(const std::shared_ptr<const geometry_detail::NativeAllocationState>&);
    void storage(const std::shared_ptr<geometry_detail::NativeConstructorStorage>&,
        const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>&);
    void semantic(std::uint8_t, std::uint8_t, const std::shared_ptr<const void>&,
        const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>&);
    void identity(std::uint8_t channel, const GeometryIdentityBackingPtr&);
    struct Key
    {
        std::uint8_t channel, kind;
        const void* address;
        bool operator<(const Key&) const noexcept;
    };
    struct Receipt
    {
        std::weak_ptr<const void> owner;
        std::size_t bytes;
        std::weak_ptr<const geometry_detail::NativeRetainedExtent> profile;
        bool profiled = false, concrete = false;
        NativePaidExtentCharge charge;
        std::weak_ptr<const geometry_detail::NativeConstructorStorage> storage;
        std::size_t storage_revision = 0, storage_bytes = 0;
    };
    NativePaidOwnerAuthority(geometry_detail::NativeConstructionAdmission, const SealedSourcePrefix&);
    void record(
        std::uint8_t, std::uint8_t, const std::shared_ptr<const void>&, std::size_t, NativePaidExtentCharge = {});
    void metadata(std::uint8_t, const std::shared_ptr<const void>&, std::size_t);
    geometry_detail::NativeConstructionAdmission admission_;
    const SealedSourcePrefix& prefix_;
    // A moved-out mutable ledger can reject after this borrowed operation ends.
    std::shared_ptr<const void> lifetime_;
    geometry_detail::NativeConstructorMap<Key, Receipt> receipts_;
};
/** Exact buffer proof emitted only by the immutable typed output visitor.
 * That visitor verifies the private vector's allocator, capacity and data
 * before construction; this token carries its target/control binding. */
class NativeActualBufferProof
{
private:
    friend class NativeGeometryAccounting;
    friend class NativePaidExtentCharge;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativePublicationOutputGraph;
    friend class SimpleFluid::geometry_detail::NativeTpetraMapAdmission;
    NativeActualBufferProof(std::shared_ptr<const void> owner,
        std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> envelope,
        std::shared_ptr<const geometry_detail::NativeAllocationState> state,
        geometry_detail::NativeAllocationExtent extent)
        : owner_(std::move(owner)), envelope_(std::move(envelope)), state_(std::move(state)), extent_(extent)
    {
    }
    std::shared_ptr<const void> owner_;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> envelope_;
    std::shared_ptr<const geometry_detail::NativeAllocationState> state_;
    geometry_detail::NativeAllocationExtent extent_;
};
class NativeGeometryAccounting
{
public:
    // An empty-prefix operation carries no source admission authority.
    explicit NativeGeometryAccounting(geometry_detail::NativeConstructionAdmission);
    NativeGeometryAccounting(geometry_detail::NativeConstructionAdmission, const SealedSourcePrefix&);
    ~NativeGeometryAccounting();
    NativeGeometryAccounting(const NativeGeometryAccounting&) = delete;
    NativeGeometryAccounting& operator=(const NativeGeometryAccounting&) = delete;
    [[nodiscard]] const geometry_detail::NativeConstructionAdmission& admission() const noexcept { return admission_; }
    [[nodiscard]] std::size_t& cursor() noexcept { return cursor_; }
    [[nodiscard]] std::size_t limit() const noexcept { return admission_.counters().limit; }
    [[nodiscard]] GeometryAllocationLedger& ordinary() noexcept { return ordinary_; }
    [[nodiscard]] GeometryAllocationLedger& bounded() noexcept { return bounded_; }
    void import_current(const GeometryOwnerUnion&);
    [[nodiscard]] OwnerManifestPair finish_pair(
        GeometryOwnerGraphEditor&, GeometryOwnerGraphEditor&, std::size_t overlap);
    // Exact receipt authorization. These functions validate active State,
    // actual target/shared controls and immutable receipt size before recording.
    void authorize_fixed(const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>&);
    void authorize_state(const std::shared_ptr<const geometry_detail::NativeAllocationState>&);
    void authorize_storage(const std::shared_ptr<geometry_detail::NativeConstructorStorage>&,
        const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>&);
    void authorize_schema(const std::shared_ptr<const geometry_detail::NativeSchemaOwner>&);
    void authorize_buffer(const NativeActualBufferProof&);
    void authorize_buffer(const std::shared_ptr<const void>& alias, std::size_t bytes,
        const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& target,
        const geometry_detail::NativeConstructorStorageOwner&);
    void authorize_semantic(std::uint8_t channel, std::uint8_t kind, const std::shared_ptr<const void>&,
        const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>&);

private:
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativePublicationOperation;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativeSourceOwnerStore;
    // The issuer has verified the actual library owner/resource and attached
    // this retained profile token to that owner's real allocation lifetime.
    friend class SimpleFluid::geometry_detail::NativeTpetraMapAdmission;
    friend class SimpleFluid::geometry_detail::NativeMeshHandleAccess;
    void authorize_profile_owner(std::uint8_t channel, const std::shared_ptr<const void>&,
        const std::shared_ptr<const geometry_detail::NativeRetainedExtent>&,
        const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>&);
    [[nodiscard]] static const SealedSourcePrefix& empty_prefix();
    geometry_detail::NativeConstructionAdmission admission_;
    std::size_t cursor_ = 0;
    const SealedSourcePrefix& prefix_;
    NativePaidOwnerAuthority paid_;
    GeometryAllocationLedger ordinary_, bounded_;
    GeometryOwnerAdmissionPair pair_;
};
} // namespace SimpleFluid::PLIC::details
