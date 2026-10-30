/** @file GeometryIdentityAccess.hh
 * @brief Private trusted issuance seam for geometry construction providers.
 */
#pragma once
#include "geometry/PLIC/GeometryIdentity.hh"
#include "geometry/PLIC/details/GeometryIdentityBacking.hh"
#include "geometry/PLIC/details/OrdinaryIdentitySupport.hh"
#include <span>
namespace SimpleFluid::PLIC::details
{
class GeometryIdentityAccess
{
public:
    template<class Identity>
    [[nodiscard]] static Identity share(std::vector<char> key,
        std::unique_ptr<const GeometryIdentityNativePayload> native, GeometryIdentityBackingPtr parent = {})
        requires(Identity::shared_storage)
    {
        return Identity(make_geometry_identity_backing(std::move(key), std::move(native), std::move(parent)));
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
    template<class Identity> [[nodiscard]] static Identity decode_key(std::vector<char> key, std::size_t budget)
    {
        return Identity(std::move(key), budget);
    }
    template<GeometryIdentityKind Kind, ArithmeticMode Mode>
    [[nodiscard]] static std::span<const char> key(const GeometryIdentity<Kind, Mode>& identity)
        requires(GeometryIdentity<Kind, Mode>::shared_storage)
    {
        return identity.key();
    }
};
} // namespace SimpleFluid::PLIC::details
