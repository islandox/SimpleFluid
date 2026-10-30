/** @file GeometryIdentityCodec.hh
 * @brief Checked legacy-wire import and explicit identity export/promotion.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/PLIC/GeometryIdentity.hh"
#include <span>
namespace SimpleFluid::PLIC
{
namespace details
{
[[nodiscard]] SIMPLEFLUID_LOCAL std::vector<char> checked_identity_key(
    GeometryIdentityKind, ArithmeticMode, std::span<const char>);
}
/** Wire admission size; Stage 1 preserves the existing key bytes. */
template<GeometryIdentityKind Kind, ArithmeticMode Mode>
[[nodiscard]] std::size_t encoded_identity_size(const GeometryIdentity<Kind, Mode>& identity) noexcept
{
    return canonical_key_size(identity);
}
template<GeometryIdentityKind Kind, ArithmeticMode Mode>
[[nodiscard]] std::vector<char> encode_identity(const GeometryIdentity<Kind, Mode>& identity)
{
    return canonical_key(identity);
}
template<class Identity> [[nodiscard]] Identity decode_identity(std::span<const char> packet)
{
    return Identity(details::checked_identity_key(Identity::kind, Identity::mode, packet));
}
[[nodiscard]] SIMPLEFLUID_LOCAL PointIdentity<ArithmeticMode::Exact> promote_identity(
    const PointIdentity<ArithmeticMode::Ordinary>&);
[[nodiscard]] SIMPLEFLUID_LOCAL DisplacementIdentity<ArithmeticMode::Exact> promote_identity(
    const DisplacementIdentity<ArithmeticMode::Ordinary>&);
} // namespace SimpleFluid::PLIC
