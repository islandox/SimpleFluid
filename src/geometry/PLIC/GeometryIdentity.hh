/** @file GeometryIdentity.hh
 * @brief Mode- and kind-specific opaque geometry identities.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "utils/ArithmeticMode.hh"
#include <compare>
#include <cstddef>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimpleFluid::PLIC
{
enum class GeometryIdentityKind
{
    Point,
    Line,
    PlaneSupport,
    Displacement
};
namespace details
{
class GeometryIdentityAccess;
struct GeometryIdentityBacking;
using GeometryIdentityBackingPtr = std::shared_ptr<const GeometryIdentityBacking>;
[[nodiscard]] SIMPLEFLUID_LOCAL GeometryIdentityBackingPtr make_geometry_identity_backing(
    std::vector<char>, GeometryIdentityBackingPtr parent = {});
[[nodiscard]] SIMPLEFLUID_LOCAL const std::vector<char>& geometry_identity_key(
    const GeometryIdentityBackingPtr&) noexcept;
[[nodiscard]] SIMPLEFLUID_LOCAL std::size_t geometry_identity_backing_bytes() noexcept;
[[nodiscard]] SIMPLEFLUID_LOCAL std::size_t geometry_identity_payload_bytes(const GeometryIdentityBackingPtr&) noexcept;
[[nodiscard]] SIMPLEFLUID_LOCAL GeometryIdentityBackingPtr geometry_identity_parent(
    const GeometryIdentityBackingPtr&) noexcept;
} // namespace details

/** One immutable payload owner. The pin preserves its address throughout an
 * admission scope. Bytes include the backing struct and owned buffer capacity,
 * excluding allocator/control-block overhead and separately shared ancestors.
 */
struct GeometryIdentityAllocation
{
    std::shared_ptr<const void> owner;
    std::size_t payload_bytes;
};

/** Identity equality and ordering preserve the provider's canonical key.
 * Bytes are private; packet import belongs to GeometryIdentityCodec.hh. */
template<GeometryIdentityKind Kind, ArithmeticMode Mode>
    requires(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::Exact)
class GeometryIdentity
{
public:
    static constexpr auto kind = Kind;
    static constexpr auto mode = Mode;
    static constexpr bool shared_storage = Mode == ArithmeticMode::Exact || Kind == GeometryIdentityKind::PlaneSupport;
    GeometryIdentity()
        requires(Kind == GeometryIdentityKind::Displacement)
    = default;
    /** Own retained payload only. Visit shared ancestry to count its owners
     * separately; the inline identity object belongs to its enclosing owner. */
    [[nodiscard]] std::size_t payload_bytes() const noexcept
    {
        if constexpr (shared_storage)
            return details::geometry_identity_payload_bytes(storage_);
        else
            return storage_.capacity();
    }
    [[nodiscard]] const void* storage_identity() const noexcept
    {
        if constexpr (shared_storage)
            return storage_.get();
        else
            return nullptr;
    }
    [[nodiscard]] std::shared_ptr<const void> storage_owner() const noexcept
    {
        if constexpr (shared_storage)
            return storage_;
        else
            return {};
    }
    /** Zero displacement has an empty canonical body and no backing owner. */
    [[nodiscard]] bool is_zero() const noexcept
        requires(Kind == GeometryIdentityKind::Displacement)
    {
        if constexpr (shared_storage)
            return !storage_;
        else
            return storage_.empty();
    }
    [[nodiscard]] bool operator==(const GeometryIdentity& other) const noexcept
    {
        if constexpr (shared_storage)
            if (storage_ == other.storage_)
                return true;
        return key() == other.key();
    }
    [[nodiscard]] std::strong_ordering operator<=>(const GeometryIdentity& other) const noexcept
    {
        if constexpr (shared_storage)
            if (storage_ == other.storage_)
                return std::strong_ordering::equal;
        return key() <=> other.key();
    }

private:
    friend class details::GeometryIdentityAccess;
    template<class Identity> friend Identity decode_identity(std::span<const char>, std::size_t);
    template<GeometryIdentityKind K, ArithmeticMode M>
        requires(M == ArithmeticMode::Ordinary || M == ArithmeticMode::Exact)
    friend std::vector<char> canonical_key(const GeometryIdentity<K, M>&);
    template<GeometryIdentityKind K, ArithmeticMode M>
        requires(M == ArithmeticMode::Ordinary || M == ArithmeticMode::Exact)
    friend std::size_t canonical_key_size(const GeometryIdentity<K, M>&) noexcept;
    template<GeometryIdentityKind K, ArithmeticMode M, class Visitor>
        requires(M == ArithmeticMode::Ordinary || M == ArithmeticMode::Exact)
    friend void visit_identity_allocations(const GeometryIdentity<K, M>&, Visitor&&);
    explicit GeometryIdentity(std::vector<char> key)
    {
        if constexpr (shared_storage)
        {
            if constexpr (Kind == GeometryIdentityKind::Displacement)
                if (key.empty())
                    return;
            storage_ = details::make_geometry_identity_backing(std::move(key));
        }
        else
            storage_ = std::move(key);
    }
    GeometryIdentity(std::vector<char> key, const GeometryIdentity& parent)
        requires(shared_storage)
        : storage_(details::make_geometry_identity_backing(std::move(key), parent.storage_))
    {
    }
    [[nodiscard]] const std::vector<char>& key() const noexcept
    {
        if constexpr (shared_storage)
            return details::geometry_identity_key(storage_);
        else
            return storage_;
    }
    std::conditional_t<shared_storage, details::GeometryIdentityBackingPtr, std::vector<char>> storage_;
};
template<ArithmeticMode Mode> using PointIdentity = GeometryIdentity<GeometryIdentityKind::Point, Mode>;
template<ArithmeticMode Mode> using LineIdentity = GeometryIdentity<GeometryIdentityKind::Line, Mode>;
template<ArithmeticMode Mode> using PlaneSupportIdentity = GeometryIdentity<GeometryIdentityKind::PlaneSupport, Mode>;
template<ArithmeticMode Mode> using DisplacementIdentity = GeometryIdentity<GeometryIdentityKind::Displacement, Mode>;

/** Canonical key length without copying or exposing its bytes. */
template<GeometryIdentityKind Kind, ArithmeticMode Mode>
    requires(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::Exact)
[[nodiscard]] std::size_t canonical_key_size(const GeometryIdentity<Kind, Mode>& identity) noexcept
{
    return identity.key().size();
}

/** Explicit canonical map/key material; independent of the packet codec. */
template<GeometryIdentityKind Kind, ArithmeticMode Mode>
    requires(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::Exact)
[[nodiscard]] std::vector<char> canonical_key(const GeometryIdentity<Kind, Mode>& identity)
{
    return identity.key();
}

/** Visit the shared root and its retained support ancestry without allocating.
 * Return false from the visitor to stop at an already admitted owner. Ordinary
 * inline forms have no shared owner; their buffers belong to their container.
 */
template<GeometryIdentityKind Kind, ArithmeticMode Mode, class Visitor>
    requires(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::Exact)
void visit_identity_allocations(const GeometryIdentity<Kind, Mode>& identity, Visitor&& visitor)
{
    if constexpr (GeometryIdentity<Kind, Mode>::shared_storage)
        for (auto owner = identity.storage_; owner; owner = details::geometry_identity_parent(owner))
            if (!visitor(GeometryIdentityAllocation{owner, details::geometry_identity_payload_bytes(owner)}))
                break;
}
} // namespace SimpleFluid::PLIC
