/** @file GeometryIdentity.hh
 * @brief Mode- and kind-specific opaque geometry identities.
 */
#pragma once
#include "utils/ArithmeticMode.hh"
#include <compare>
#include <cstddef>
#include <span>
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
}

/** Identity equality and ordering preserve the provider's canonical key.
 * Bytes are private; packet import belongs to GeometryIdentityCodec.hh. */
template<GeometryIdentityKind Kind, ArithmeticMode Mode>
    requires(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::Exact)
class GeometryIdentity
{
public:
    static constexpr auto kind = Kind;
    static constexpr auto mode = Mode;
    GeometryIdentity()
        requires(Kind == GeometryIdentityKind::Displacement)
    = default;
    [[nodiscard]] std::size_t payload_bytes() const noexcept { return key_.capacity(); }
    /** Stage 1 recognizes the empty legacy zero marker; tagged zero packets
     * retain their original key and therefore report false until codec migration. */
    [[nodiscard]] bool is_zero() const noexcept
        requires(Kind == GeometryIdentityKind::Displacement)
    {
        return key_.empty();
    }
    auto operator<=>(const GeometryIdentity&) const = default;

private:
    friend class details::GeometryIdentityAccess;
    template<class Identity> friend Identity decode_identity(std::span<const char>);
    template<GeometryIdentityKind K, ArithmeticMode M>
        requires(M == ArithmeticMode::Ordinary || M == ArithmeticMode::Exact)
    friend std::vector<char> canonical_key(const GeometryIdentity<K, M>&);
    template<GeometryIdentityKind K, ArithmeticMode M>
        requires(M == ArithmeticMode::Ordinary || M == ArithmeticMode::Exact)
    friend std::size_t canonical_key_size(const GeometryIdentity<K, M>&) noexcept;
    explicit GeometryIdentity(std::vector<char> key) : key_(std::move(key)) {}
    std::vector<char> key_;
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
    return identity.key_.size();
}

/** Explicit canonical map/key material; independent of the packet codec. */
template<GeometryIdentityKind Kind, ArithmeticMode Mode>
    requires(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::Exact)
[[nodiscard]] std::vector<char> canonical_key(const GeometryIdentity<Kind, Mode>& identity)
{
    return identity.key_;
}
} // namespace SimpleFluid::PLIC
