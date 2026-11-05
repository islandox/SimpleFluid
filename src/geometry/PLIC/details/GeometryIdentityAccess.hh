/** @file GeometryIdentityAccess.hh
 * @brief Private trusted issuance seam for geometry construction providers.
 */
#pragma once
#include "geometry/PLIC/GeometryIdentity.hh"
#include "geometry/PLIC/details/GeometryIdentityBacking.hh"
#include "geometry/PLIC/details/OrdinaryIdentitySupport.hh"
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
namespace SimpleFluid::PLIC::details
{
/** Immutable canonical scalar copied from an already-issued ordinary point.
 * There is no arbitrary scalar constructor: composing these copies preserves
 * the existing finite/canonical-zero admission without normalizing again. */
class CanonicalPointComponent
{
public:
    CanonicalPointComponent(const CanonicalPointComponent&) = default;
    CanonicalPointComponent& operator=(const CanonicalPointComponent&) = default;
    [[nodiscard]] long double value() const noexcept { return value_; }

private:
    explicit CanonicalPointComponent(long double value) noexcept : value_(value) {}
    long double value_;
    friend class GeometryIdentityAccess;
};
class GeometryIdentityAccess
{
public:
    template<class Identity>
    [[nodiscard]] static Identity share(
        std::unique_ptr<const GeometryIdentityNativePayload> native, GeometryIdentityBackingPtr parent = {})
        requires(Identity::shared_storage)
    {
        return Identity(make_geometry_identity_backing(std::move(native), std::move(parent)));
    }
    template<GeometryIdentityKind Kind, ArithmeticMode Mode>
    [[nodiscard]] static const GeometryIdentityBackingPtr& backing(const GeometryIdentity<Kind, Mode>& identity)
        requires(GeometryIdentity<Kind, Mode>::shared_storage)
    {
        return identity.storage_;
    }
    [[nodiscard]] static PointIdentity<ArithmeticMode::Ordinary> point(OrdinaryIdentityPoint coordinates)
    {
        return PointIdentity<ArithmeticMode::Ordinary>(normalize_identity_point(std::move(coordinates)));
    }
    [[nodiscard]] static DisplacementIdentity<ArithmeticMode::Ordinary> displacement(OrdinaryIdentityPoint coordinates)
    {
        return DisplacementIdentity<ArithmeticMode::Ordinary>(normalize_identity_point(std::move(coordinates)));
    }
    [[nodiscard]] static LineIdentity<ArithmeticMode::Ordinary> line(OrdinaryIdentityLine value)
    {
        return LineIdentity<ArithmeticMode::Ordinary>(normalize_identity_line(std::move(value)));
    }
    [[nodiscard]] static const OrdinaryIdentityPoint& point(const PointIdentity<ArithmeticMode::Ordinary>& identity)
    {
        return identity.storage_;
    }
    [[nodiscard]] static CanonicalPointComponent point_component(
        const PointIdentity<ArithmeticMode::Ordinary>& identity, std::size_t axis)
    {
        if (axis >= 3)
            throw std::out_of_range("Invalid canonical point component axis.");
        return CanonicalPointComponent(identity.storage_[axis]);
    }
    /** Lossless value assembly only. Source dictionaries must separately prove
     * the complete assembled identity equals their certified original point. */
    [[nodiscard]] static PointIdentity<ArithmeticMode::Ordinary> point_components(
        const std::array<CanonicalPointComponent, 3>& components) noexcept
    {
        return PointIdentity<ArithmeticMode::Ordinary>(
            OrdinaryIdentityPoint{components[0].value_, components[1].value_, components[2].value_});
    }
    [[nodiscard]] static const OrdinaryIdentityPoint& displacement(
        const DisplacementIdentity<ArithmeticMode::Ordinary>& identity)
    {
        return identity.storage_;
    }
    [[nodiscard]] static const OrdinaryIdentityLine& line(const LineIdentity<ArithmeticMode::Ordinary>& identity)
    {
        return identity.storage_;
    }
    [[nodiscard]] static PlaneSupportIdentity<ArithmeticMode::Ordinary> support_axis(std::uint64_t, long double);
    [[nodiscard]] static PlaneSupportIdentity<ArithmeticMode::Ordinary> support_source(
        std::vector<OrdinaryIdentityPoint>);
    [[nodiscard]] static PlaneSupportIdentity<ArithmeticMode::Ordinary> support_input(
        OrdinaryIdentityPoint normal, OrdinaryIdentityPoint anchor, long double offset);
    [[nodiscard]] static PlaneSupportIdentity<ArithmeticMode::Ordinary> support_plane(
        std::uint64_t axis, OrdinaryIdentityPoint normal, long double location);
    [[nodiscard]] static PlaneSupportIdentity<ArithmeticMode::Ordinary> translated_support(
        const PlaneSupportIdentity<ArithmeticMode::Ordinary>&, OrdinaryIdentityPoint shift);
    [[nodiscard]] static PlaneSupportIdentity<ArithmeticMode::Ordinary> retained_exact_support(
        const PlaneSupportIdentity<ArithmeticMode::Exact>&);
    [[nodiscard]] static PlaneSupportIdentity<ArithmeticMode::Ordinary> support_dag(
        std::vector<OrdinarySupportRecipe>, std::size_t first, std::size_t second);
    /** Checked canonical-key bodies are admitted only by untrusted record decoders. */
    template<class Identity> [[nodiscard]] static Identity decode_key(std::span<const char> key, std::size_t budget)
    {
        return Identity(key, budget);
    }
    /** Explicit streaming compatibility boundary. The callback must consume its
     * borrowed span before returning. This operation may allocate depth scratch. */
    template<GeometryIdentityKind Kind, ArithmeticMode Mode>
    static void emit_canonical(
        const GeometryIdentity<Kind, Mode>& identity, void* context, void (*append)(void*, std::span<const char>))
    {
        if constexpr (GeometryIdentity<Kind, Mode>::shared_storage)
            emit_geometry_identity_canonical(identity.storage_, {context, append});
        else
        {
            const auto key = identity.inline_key();
            append(context, {key.bytes.data(), key.size});
        }
    }
    template<GeometryIdentityKind Kind, ArithmeticMode Mode>
    [[nodiscard]] static GeometryIdentityCanonicalCursor canonical_cursor(const GeometryIdentity<Kind, Mode>& identity)
        requires(GeometryIdentity<Kind, Mode>::shared_storage)
    {
        return GeometryIdentityCanonicalCursor(identity.storage_);
    }
    /** Packed little-endian canonical bytes, zero padded at the tail. Ancestry
     * is revisited to locate a range; the query never allocates scratch. */
    template<GeometryIdentityKind Kind, ArithmeticMode Mode>
    [[nodiscard]] static std::uint64_t canonical_word_at(
        const GeometryIdentity<Kind, Mode>& identity, std::size_t byte_offset) noexcept
    {
        if constexpr (GeometryIdentity<Kind, Mode>::shared_storage)
            return geometry_identity_key_word(identity.storage_, byte_offset);
        else
        {
            const auto key = identity.inline_key();
            std::uint64_t word = 0;
            for (unsigned i = 0; i < 8 && byte_offset < key.size; ++i, ++byte_offset)
                word |= std::uint64_t(static_cast<unsigned char>(key.bytes[byte_offset])) << (8 * i);
            return word;
        }
    }
};
} // namespace SimpleFluid::PLIC::details
