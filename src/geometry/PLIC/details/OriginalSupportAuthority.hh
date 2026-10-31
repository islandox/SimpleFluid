/** @file OriginalSupportAuthority.hh
 * @brief Sealed native attachments for original immutable source authorities.
 */
#pragma once
#include "geometry/PLIC/ConvexGeometry.hh"
#include <set>
namespace SimpleFluid::PLIC::details
{
struct PlaneAuthorityLess
{
    using is_transparent = void;
    bool operator()(const PlaneAuthority& a, const PlaneAuthority& b) const { return a < b; }
    bool operator()(const PlaneAuthority& a, std::span<const std::uint64_t> b) const { return a.compare_words(b) < 0; }
    bool operator()(std::span<const std::uint64_t> a, const PlaneAuthority& b) const { return b.compare_words(a) > 0; }
};
using PlaneAuthorityRegistry = std::set<PlaneAuthority, PlaneAuthorityLess>;
class OriginalSupportAuthorityAccess
{
public:
    [[nodiscard]] static PlaneAuthority issue(const PlaneSupportIdentity<ArithmeticMode::Ordinary>&);
    [[nodiscard]] static PlaneAuthority issue(const PlaneSupportIdentity<ArithmeticMode::Exact>&);
    /** Copies of an issued authority share this immutable native owner. */
    [[nodiscard]] static const PlaneSupportIdentity<ArithmeticMode::Ordinary>* retained(const PlaneAuthority&) noexcept;
    /** Only trusted issuance or authenticated imports can attach native support. */
    [[nodiscard]] static const PlaneSupportIdentity<ArithmeticMode::Ordinary>* get(const PlaneAuthority&);
    static void validate_boundary(const PlaneAuthority&);
    static void merge(PlaneAuthority& destination, const PlaneAuthority& source);
    /** Untrusted words are matched including header length and final padding;
     * success reuses a supplied authority, never reconstructs one from words. */
    [[nodiscard]] static PlaneAuthority authenticate(std::span<const std::uint64_t>, const PlaneAuthorityRegistry&);
};
} // namespace SimpleFluid::PLIC::details
