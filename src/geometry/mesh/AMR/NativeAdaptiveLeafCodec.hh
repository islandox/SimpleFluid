/** Private shared-domain leaf wire emission; no source proof is issued. */
#pragma once
#include "geometry/mesh/AMR/NativeAdaptiveKeyCodec.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct NativeDecodedLeaf
{
    geometry_detail::NativeConstructorStorageOwner storage;
    AdaptiveLeaf value;
};
class SIMPLEFLUID_LOCAL NativeAdaptiveLeafCodec final
{
public:
    [[nodiscard]] static std::size_t size(const AdaptiveLeaf&, const geometry_detail::NativeConstructionAdmission&);
    /** Append one length-framed leaf directly to an exactly preadmitted packet. */
    static void write(Writer&, const AdaptiveLeaf&, const geometry_detail::NativeConstructionAdmission&);
    static void write_payload(Writer&, const AdaptiveLeaf&, const geometry_detail::NativeConstructionAdmission&);
    [[nodiscard]] static NativeDecodedLeaf decode(std::span<const char>,
        const geometry_detail::NativeConstructionAdmission&, const AdaptiveLeaf* retained_parts = nullptr);
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
