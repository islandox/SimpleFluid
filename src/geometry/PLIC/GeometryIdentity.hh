/** @file GeometryIdentity.hh
 * @brief Mode- and kind-specific opaque geometry identities.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/PLIC/GeometryCodecTags.hh"
#include "utils/ArithmeticMode.hh"
#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimpleFluid::PLIC
{
enum class GeometryIdentityKind : std::uint8_t
{
    Point,
    Line,
    PlaneSupport,
    Displacement
};
namespace details
{
class GeometryIdentityAccess;
using OrdinaryIdentityPoint = std::array<long double, 3>;
struct OrdinaryIdentityLine
{
    std::uint64_t axis = 0;
    OrdinaryIdentityPoint direction{}, anchor{};
};
inline constexpr std::size_t ordinary_identity_scalar_bytes =
    16 + 8 * ((std::numeric_limits<long double>::digits + 31) / 32);
struct OrdinaryIdentityKey
{
    std::array<char, 9 + 6 * ordinary_identity_scalar_bytes> bytes{};
    std::size_t size = 0;
};
[[nodiscard]] SIMPLEFLUID_LOCAL OrdinaryIdentityPoint normalize_identity_point(OrdinaryIdentityPoint);
[[nodiscard]] SIMPLEFLUID_LOCAL OrdinaryIdentityLine normalize_identity_line(OrdinaryIdentityLine);
[[nodiscard]] SIMPLEFLUID_LOCAL OrdinaryIdentityPoint import_ordinary_identity_point(
    GeometryIdentityKind, std::span<const char>);
[[nodiscard]] SIMPLEFLUID_LOCAL OrdinaryIdentityLine import_ordinary_identity_line(std::span<const char>);
[[nodiscard]] SIMPLEFLUID_LOCAL OrdinaryIdentityKey ordinary_identity_key(
    GeometryIdentityKind, const OrdinaryIdentityPoint&) noexcept;
[[nodiscard]] SIMPLEFLUID_LOCAL OrdinaryIdentityKey ordinary_identity_key(const OrdinaryIdentityLine&) noexcept;
[[nodiscard]] SIMPLEFLUID_LOCAL std::size_t ordinary_identity_key_size(
    GeometryIdentityKind, const OrdinaryIdentityPoint&) noexcept;
[[nodiscard]] SIMPLEFLUID_LOCAL std::size_t ordinary_identity_key_size(const OrdinaryIdentityLine&) noexcept;

struct GeometryIdentityBacking;
using GeometryIdentityBackingPtr = std::shared_ptr<const GeometryIdentityBacking>;
[[nodiscard]] SIMPLEFLUID_LOCAL std::strong_ordering compare_geometry_identity_native(
    GeometryIdentityKind, ArithmeticMode, const GeometryIdentityBackingPtr&, const GeometryIdentityBackingPtr&);
[[nodiscard]] SIMPLEFLUID_LOCAL GeometryIdentityBackingPtr import_geometry_identity_backing(
    GeometryIdentityKind, ArithmeticMode, std::vector<char> checked_key, std::size_t byte_budget);

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
            return 0;
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
            return storage_ == details::OrdinaryIdentityPoint{};
    }
    [[nodiscard]] bool operator==(const GeometryIdentity& other) const noexcept
    {
        if constexpr (shared_storage)
        {
            if (storage_ == other.storage_)
                return true;
            return key() == other.key();
        }
        else if constexpr (Kind == GeometryIdentityKind::Line)
            return storage_.axis == other.storage_.axis && storage_.direction == other.storage_.direction &&
                   storage_.anchor == other.storage_.anchor;
        else
            return storage_ == other.storage_;
    }
    /** Native numeric/provenance order. Ordinary supports use the stable
     * variant order Axis, SourcePolygon, InputPlane, NormalizedPlane,
     * Translation, RetainedExact, ExtrusionDAG. Fields are lexicographic;
     * translations compare parent expression before shift, without flattening.
     * Frozen key bytes remain the serialization/equality/hash contract. */
    [[nodiscard]] std::strong_ordering operator<=>(const GeometryIdentity& other) const noexcept(!shared_storage)
    {
        if constexpr (shared_storage)
        {
            if (storage_ == other.storage_)
                return std::strong_ordering::equal;
            return details::compare_geometry_identity_native(Kind, Mode, storage_, other.storage_);
        }
        else
        {
            const auto points =
                [](const details::OrdinaryIdentityPoint& first, const details::OrdinaryIdentityPoint& second) noexcept
            {
                for (std::size_t i = 0; i < 3; ++i)
                {
                    if (first[i] < second[i])
                        return std::strong_ordering::less;
                    if (second[i] < first[i])
                        return std::strong_ordering::greater;
                }
                return std::strong_ordering::equal;
            };
            if constexpr (Kind == GeometryIdentityKind::Line)
            {
                if (storage_.axis != other.storage_.axis)
                    return storage_.axis <=> other.storage_.axis;
                const auto direction = points(storage_.direction, other.storage_.direction);
                return direction == 0 ? points(storage_.anchor, other.storage_.anchor) : direction;
            }
            else
                return points(storage_, other.storage_);
        }
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
    template<GeometryIdentityKind K, ArithmeticMode M>
        requires(M == ArithmeticMode::Ordinary || M == ArithmeticMode::Exact)
    friend std::uint64_t canonical_key_hash(const GeometryIdentity<K, M>&) noexcept;

    explicit GeometryIdentity(details::GeometryIdentityBackingPtr storage)
        requires(shared_storage)
        : storage_(std::move(storage))
    {
    }
    explicit GeometryIdentity(std::vector<char> key, std::size_t byte_budget = 64 * 1024 * 1024)
    {
        if constexpr (shared_storage)
            storage_ = details::import_geometry_identity_backing(Kind, Mode, std::move(key), byte_budget);
        else if constexpr (Kind == GeometryIdentityKind::Line)
            storage_ = details::import_ordinary_identity_line(key);
        else
            storage_ = details::import_ordinary_identity_point(Kind, key);
    }
    explicit GeometryIdentity(details::OrdinaryIdentityPoint point)
        requires(Mode == ArithmeticMode::Ordinary &&
                 (Kind == GeometryIdentityKind::Point || Kind == GeometryIdentityKind::Displacement))
        : storage_(std::move(point))
    {
    }
    explicit GeometryIdentity(details::OrdinaryIdentityLine line)
        requires(Mode == ArithmeticMode::Ordinary && Kind == GeometryIdentityKind::Line)
        : storage_(std::move(line))
    {
    }
    [[nodiscard]] const std::vector<char>& key() const noexcept
        requires(shared_storage)
    {
        return details::geometry_identity_key(storage_);
    }
    [[nodiscard]] details::OrdinaryIdentityKey inline_key() const noexcept
        requires(!shared_storage)
    {
        if constexpr (Kind == GeometryIdentityKind::Line)
            return details::ordinary_identity_key(storage_);
        else
            return details::ordinary_identity_key(Kind, storage_);
    }
    [[nodiscard]] std::size_t inline_key_size() const noexcept
        requires(!shared_storage)
    {
        if constexpr (Kind == GeometryIdentityKind::Line)
            return details::ordinary_identity_key_size(storage_);
        else
            return details::ordinary_identity_key_size(Kind, storage_);
    }
    using InlineStorage = std::conditional_t<Kind == GeometryIdentityKind::Line, details::OrdinaryIdentityLine,
        details::OrdinaryIdentityPoint>;
    std::conditional_t<shared_storage, details::GeometryIdentityBackingPtr, InlineStorage> storage_{};
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
    if constexpr (GeometryIdentity<Kind, Mode>::shared_storage)
        return identity.key().size();
    else
        return identity.inline_key_size();
}

/** Explicit canonical map/key material; independent of the packet codec. */
template<GeometryIdentityKind Kind, ArithmeticMode Mode>
    requires(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::Exact)
[[nodiscard]] std::vector<char> canonical_key(const GeometryIdentity<Kind, Mode>& identity)
{
    if constexpr (GeometryIdentity<Kind, Mode>::shared_storage)
        return identity.key();
    else
    {
        const auto key = identity.inline_key();
        return {key.bytes.begin(), key.bytes.begin() + key.size};
    }
}

/** Explicit compatibility adapter for the retired plain-char key-byte order.
 * Key export may allocate. Computational callers should use native ordering. */
struct [[deprecated("Legacy canonical byte ordering is retired; use native typed identity ordering.")]]
LegacyCanonicalKeyLess
{
    template<GeometryIdentityKind Kind, ArithmeticMode Mode>
    [[nodiscard]] bool operator()(
        const GeometryIdentity<Kind, Mode>& first, const GeometryIdentity<Kind, Mode>& second) const
    {
        return canonical_key(first) < canonical_key(second);
    }
};

/** Stable FNV-1a over unsigned canonical bytes, without materializing a key. */
template<GeometryIdentityKind Kind, ArithmeticMode Mode>
    requires(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::Exact)
[[nodiscard]] std::uint64_t canonical_key_hash(const GeometryIdentity<Kind, Mode>& identity) noexcept
{
    std::uint64_t hash = 14695981039346656037ULL;
    const auto consume = [&](char byte)
    {
        hash ^= static_cast<unsigned char>(byte);
        hash *= 1099511628211ULL;
    };
    if constexpr (GeometryIdentity<Kind, Mode>::shared_storage)
        for (const auto byte : identity.key())
            consume(byte);
    else
    {
        const auto key = identity.inline_key();
        for (std::size_t i = 0; i < key.size; ++i)
            consume(key.bytes[i]);
    }
    return hash;
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
