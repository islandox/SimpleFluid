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
    template<GeometryIdentityKind Kind, ArithmeticMode Mode>
    [[nodiscard]] static std::span<const char> key(const GeometryIdentity<Kind, Mode>& identity)
    {
        return identity.key_;
    }
};
} // namespace SimpleFluid::PLIC::details
