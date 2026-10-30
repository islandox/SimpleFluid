/** @file GeometryIdentityBacking.hh
 * @brief Private native payload ownership, independent of arithmetic providers.
 */
#pragma once
#include "geometry/PLIC/GeometryIdentity.hh"

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
};
struct GeometryIdentityBacking
{
    std::vector<char> key;
    GeometryIdentityBackingPtr parent;
    std::unique_ptr<const GeometryIdentityNativePayload> native;
};
[[nodiscard]] GeometryIdentityBackingPtr make_geometry_identity_backing(std::vector<char> key,
    std::unique_ptr<const GeometryIdentityNativePayload> native, GeometryIdentityBackingPtr parent = {});
[[nodiscard]] const GeometryIdentityNativePayload& geometry_identity_native(const GeometryIdentityBackingPtr&);
/** New untrusted codec body import dispatches into mode-selected providers. */
[[nodiscard]] GeometryIdentityBackingPtr import_exact_identity_backing(
    GeometryIdentityKind, std::vector<char> checked_key, std::size_t byte_budget);
[[nodiscard]] std::size_t exact_identity_import_storage_bound(GeometryIdentityKind, std::span<const char> checked_key);
[[nodiscard]] std::strong_ordering compare_exact_identity_native(
    GeometryIdentityKind, const GeometryIdentityBackingPtr&, const GeometryIdentityBackingPtr&);
} // namespace SimpleFluid::PLIC::details
