/** @file GeometryIdentityAccess.hh
 * @brief Private trusted issuance seam for geometry construction providers.
 */
#pragma once
#include "geometry/PLIC/GeometryIdentity.hh"
#include <span>
namespace SimpleFluid::PLIC::details
{
class GeometryIdentityAccess
{
public:
    template<class Identity> [[nodiscard]] static Identity make(std::vector<char> key)
    {
        return Identity(std::move(key));
    }
    template<class Identity>
    [[nodiscard]] static Identity make(std::vector<char> key, const Identity& parent)
        requires(Identity::mode == ArithmeticMode::Ordinary && Identity::kind == GeometryIdentityKind::PlaneSupport)
    {
        return Identity(std::move(key), parent);
    }
    template<GeometryIdentityKind Kind, ArithmeticMode Mode>
    [[nodiscard]] static std::span<const char> key(const GeometryIdentity<Kind, Mode>& identity)
    {
        return identity.key();
    }
};
} // namespace SimpleFluid::PLIC::details
