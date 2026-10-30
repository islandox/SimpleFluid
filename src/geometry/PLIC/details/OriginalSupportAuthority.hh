/** @file OriginalSupportAuthority.hh
 * @brief Sealed native attachments for original immutable source authorities.
 */
#pragma once
#include "geometry/PLIC/ConvexGeometry.hh"
namespace SimpleFluid::PLIC::details
{
class OriginalSupportAuthorityAccess
{
public:
    [[nodiscard]] static PlaneAuthority issue(const PlaneSupportIdentity<ArithmeticMode::Ordinary>&);
    [[nodiscard]] static PlaneAuthority issue(const PlaneSupportIdentity<ArithmeticMode::Exact>&);
    /** Ownership accounting never validates mutable public key words. */
    [[nodiscard]] static const PlaneSupportIdentity<ArithmeticMode::Ordinary>* retained(const PlaneAuthority&) noexcept;
    /** Semantic access validates every frozen-key word including padding. */
    [[nodiscard]] static const PlaneSupportIdentity<ArithmeticMode::Ordinary>* get(const PlaneAuthority&);
    static void validate_boundary(const PlaneAuthority&);
    static void merge(PlaneAuthority& destination, const PlaneAuthority& source);
};
} // namespace SimpleFluid::PLIC::details
