/** @file GeometryIdentityBacking.hh
 * @brief Private native payload ownership, independent of arithmetic providers.
 */
#pragma once
#include "geometry/PLIC/GeometryIdentity.hh"
#include "geometry/PLIC/details/GeometryIdentityCanonicalWriter.hh"
#include "geometry/details/NativeConstructorStorage.hh"

namespace SimpleFluid::PLIC::details
{
enum class GeometryIdentityNativeKind : std::uint8_t
{
    OrdinarySupport,
    ExactTuple3,
    ExactTuple4,
    ExactTuple6
};
class GeometryIdentityNativePayload
{
public:
    virtual ~GeometryIdentityNativePayload() = default;
    [[nodiscard]] virtual GeometryIdentityNativeKind native_kind() const noexcept = 0;
    /** This concrete payload's sizeof plus exclusively owned dynamic buffers
     * and limbs. Inline members are already included in sizeof. */
    [[nodiscard]] virtual std::size_t storage_bytes() const noexcept = 0;
    /** Prefix and suffix surround the separately shared parent expression. */
    virtual void emit_prefix(const GeometryIdentityCanonicalWriter&, std::size_t parent_bytes) const = 0;
    virtual void emit_suffix(const GeometryIdentityCanonicalWriter&) const {}
    [[nodiscard]] virtual bool encoded_fields_equal(const GeometryIdentityNativePayload&) const noexcept = 0;
    /** Sequential local emission. State is caller-owned and initially zero;
     * one call emits at most output.size() bytes, returning zero at the end. */
    [[nodiscard]] virtual std::size_t next_canonical_chunk(
        bool suffix, std::array<std::size_t, 4>& state, std::size_t parent_bytes, std::span<char> output) const = 0;
};
class NativeIdentityOwnerAccess;
struct NativeGeometryIdentityBacking;
struct GeometryIdentityBacking
{
    ~GeometryIdentityBacking();
    GeometryIdentityBackingPtr parent;
    std::unique_ptr<const GeometryIdentityNativePayload> native;
    std::size_t canonical_size = 0, prefix_size = 0, suffix_size = 0, depth = 0;
    std::uint64_t cached_hash = 14695981039346656037ULL;
    [[nodiscard]] std::size_t allocation_bytes() const noexcept;

private:
    friend class NativeIdentityOwnerAccess;
    friend struct NativeGeometryIdentityBacking;
    friend void dispose_identity_backing(const GeometryIdentityBacking*) noexcept;
    // The legacy deleter writes the next link only after the strong count
    // reaches zero. Only the admitted derived constructor sets a live
    // self-marker. The admitted control never invokes the legacy deleter.
    GeometryIdentityBacking* disposal_next = nullptr;
};
/** The admitted allocation carries its own resources in the same backing
 * allocation. Legacy split owners have the original compact base layout. */
struct NativeGeometryIdentityBacking final : GeometryIdentityBacking
{
    class ConstructionKey
    {
        friend GeometryIdentityBackingPtr make_geometry_identity_backing_admitted(
            std::unique_ptr<const GeometryIdentityNativePayload>&&, geometry_detail::NativeConstructorStorageOwner&&,
            const geometry_detail::NativeConstructionAdmission&, GeometryIdentityBackingPtr);
        ConstructionKey() = default;

    public:
        ConstructionKey(const ConstructionKey&) = default;
    };
    explicit NativeGeometryIdentityBacking(ConstructionKey) noexcept { disposal_next = this; }
    ~NativeGeometryIdentityBacking() { native.reset(); }
    [[nodiscard]] const auto& storage() const noexcept { return admitted_storage; }
    [[nodiscard]] const auto& envelope() const noexcept { return allocation_envelope; }

private:
    friend GeometryIdentityBackingPtr make_geometry_identity_backing_admitted(
        std::unique_ptr<const GeometryIdentityNativePayload>&&, geometry_detail::NativeConstructorStorageOwner&&,
        const geometry_detail::NativeConstructionAdmission&, GeometryIdentityBackingPtr);
    geometry_detail::NativeConstructorStorageOwner admitted_storage;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation_envelope;
};
/** Exact qualified allocation requests for one legacy split backing/control.
 * Includes the real custom-deleter control type, with no pointer allowance. */
[[nodiscard]] std::size_t geometry_identity_backing_allocation_bound();
[[nodiscard]] GeometryIdentityBackingPtr make_geometry_identity_backing(
    std::unique_ptr<const GeometryIdentityNativePayload> native, GeometryIdentityBackingPtr parent = {},
    std::size_t scratch_budget = std::numeric_limits<std::size_t>::max());
/** Original depth-one native supports only. The payload and its concrete
 * buffers were built under storage; no caller/session/controller is retained. */
[[nodiscard]] GeometryIdentityBackingPtr make_geometry_identity_backing_admitted(
    std::unique_ptr<const GeometryIdentityNativePayload>&&, geometry_detail::NativeConstructorStorageOwner&&,
    const geometry_detail::NativeConstructionAdmission&, GeometryIdentityBackingPtr parent = {});
/** Charge scratch to an already active numerical operation, when present.
 * This provider-neutral hook never starts an operation or supplies a cap. */
SIMPLEFLUID_LOCAL void charge_geometry_identity_scratch(std::size_t count, std::size_t size);
/** Iterative canonical emission; pointer scratch is O(parent depth). The
 * supplied scratch allowance belongs to this operation, with no default cap. */
void emit_geometry_identity_canonical(const GeometryIdentityBackingPtr&, const GeometryIdentityCanonicalWriter&,
    std::size_t scratch_budget = std::numeric_limits<std::size_t>::max());
/** Sequential canonical bytes without a serialized-key cache. Ancestry is
 * pinned and pointer scratch is charged to an existing active operation. */
class SIMPLEFLUID_LOCAL GeometryIdentityCanonicalCursor
{
public:
    explicit GeometryIdentityCanonicalCursor(
        const GeometryIdentityBackingPtr&, std::size_t scratch_budget = std::numeric_limits<std::size_t>::max());
    GeometryIdentityCanonicalCursor(const GeometryIdentityCanonicalCursor&) = delete;
    GeometryIdentityCanonicalCursor& operator=(const GeometryIdentityCanonicalCursor&) = delete;
    [[nodiscard]] std::size_t remaining() const noexcept { return remaining_; }
    /** Consume up to eight bytes, packed little endian and zero padded. */
    [[nodiscard]] std::uint64_t next_word();

private:
    void refill();
    GeometryIdentityBackingPtr owner_;
    std::vector<const GeometryIdentityBacking*> ancestry_;
    std::array<std::size_t, 4> state_{};
    std::array<char, std::max<std::size_t>(64, ordinary_identity_scalar_bytes)> buffer_{};
    std::size_t prefix_index_ = 0, suffix_index_ = 0, position_ = 0, available_ = 0, remaining_ = 0;
    bool suffix_ = false;
};
[[nodiscard]] const GeometryIdentityNativePayload& geometry_identity_native(const GeometryIdentityBackingPtr&);
/** New untrusted codec body import dispatches into mode-selected providers. */
[[nodiscard]] GeometryIdentityBackingPtr import_exact_identity_backing(
    GeometryIdentityKind, std::span<const char> checked_key, std::size_t byte_budget);
[[nodiscard]] std::size_t exact_identity_import_storage_bound(GeometryIdentityKind, std::span<const char> checked_key);
[[nodiscard]] std::strong_ordering compare_exact_identity_native(
    GeometryIdentityKind, const GeometryIdentityBackingPtr&, const GeometryIdentityBackingPtr&);
} // namespace SimpleFluid::PLIC::details
