/** @file NativePlanarCodecAccess.hh
 * @brief Direct Ordinary face wire emission under the caller's live domain.
 */
#pragma once
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/details/NativeConstructorStorage.hh"
namespace SimpleFluid::PLIC::details
{
class SIMPLEFLUID_LOCAL NativePlanarCodecAccess final
{
public:
    [[nodiscard]] static std::vector<char> canonical(
        const PlanarFace<ArithmeticMode::Ordinary>&, const geometry_detail::NativeConstructorStorageOwner&);
    [[nodiscard]] static std::size_t size(const PlanarFace<ArithmeticMode::Ordinary>&, bool incidence,
        const geometry_detail::NativeConstructionAdmission&);
    static void append(std::vector<char>&, const PlanarFace<ArithmeticMode::Ordinary>&, bool incidence,
        const geometry_detail::NativeConstructionAdmission&);
    /** Produces the existing full packet, including its unchanged numeric profile.
     * Retain destination storage until the returned byte buffer has died. */
    [[nodiscard]] static std::vector<char> encode(const PlanarFace<ArithmeticMode::Ordinary>&, bool incidence,
        const geometry_detail::NativeConstructorStorageOwner&);
};
} // namespace SimpleFluid::PLIC::details
