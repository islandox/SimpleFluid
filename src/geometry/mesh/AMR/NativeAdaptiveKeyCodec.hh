/** @file NativeAdaptiveKeyCodec.hh
 * @brief Private receipt-backed canonical ancestry codec; wire bytes grant no authority.
 */
#pragma once
#include "geometry/details/NativeConstructorStorage.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class SIMPLEFLUID_LOCAL NativeAdaptiveKeyCodec final
{
public:
    /** Append the canonical length and key directly into a pre-reserved caller packet. */
    static void write(Writer&, const AdaptiveCellKey&);
    static void write_payload(Writer&, const AdaptiveCellKey&);
    [[nodiscard]] static AdaptiveBytes encode(
        const AdaptiveCellKey&, const geometry_detail::NativeConstructorStorageOwner&);
    /** The caller keeps storage before the returned value, and retires it only after the value dies. */
    [[nodiscard]] static AdaptiveCellKey copy(
        const AdaptiveCellKey&, const geometry_detail::NativeConstructorStorageOwner&);
    [[nodiscard]] static AdaptiveCellKey decode(
        std::span<const char>, const geometry_detail::NativeConstructorStorageOwner&);
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
