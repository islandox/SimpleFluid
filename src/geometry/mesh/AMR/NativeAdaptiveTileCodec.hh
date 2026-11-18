/** Private admitted full tile codec; metadata never issues incidence authority. */
#pragma once
#include "geometry/mesh/AMR/NativeAdaptiveKeyCodec.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class SIMPLEFLUID_LOCAL NativeAdaptiveTileCodec final
{
public:
    [[nodiscard]] static std::size_t size(const AdaptiveTile&, const geometry_detail::NativeConstructionAdmission&);
    static void write(Writer&, const AdaptiveTile&, const geometry_detail::NativeConstructionAdmission&);
    static void write_payload(Writer&, const AdaptiveTile&, const geometry_detail::NativeConstructionAdmission&);
    /** Direct vector/string/key storage is owned by storage; shared sequence and
     * geometry controls retain their own independent exact allocation receipts. */
    [[nodiscard]] static AdaptiveTile decode(
        std::span<const char>, std::size_t byte_budget, const geometry_detail::NativeConstructorStorageOwner& storage);
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
