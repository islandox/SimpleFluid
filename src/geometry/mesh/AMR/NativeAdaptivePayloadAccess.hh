/** Typed allocation projection only; decoded payloads carry no source authority. */
#pragma once
#include "geometry/details/NativeObjectOwnerAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class SIMPLEFLUID_LOCAL NativeAdaptivePayloadAccess
{
public:
    static bool bytes(
        const std::shared_ptr<const AdaptiveBytes>& value, const geometry_detail::NativeConstructorOwnerSink& sink)
    {
        using Access = geometry_detail::NativeObjectOwnerAccess<AdaptiveBytes>;
        if (!value || !Access::admitted(value))
            return false;
        if (sink.semantic(
                value, geometry_detail::NativeConstructorOwnerKind::ConstructionValues, Access::envelope(value), true))
            Access::visit_children(value, sink, true);
        return true;
    }
    static bool factor(const std::shared_ptr<const AdaptiveFactorRecipe>& value,
        const geometry_detail::NativeConstructorOwnerSink& sink)
    {
        using Access = geometry_detail::NativeObjectOwnerAccess<AdaptiveFactorRecipe>;
        if (!value || !Access::admitted(value))
            return false;
        if (!sink.semantic(value, geometry_detail::NativeConstructorOwnerKind::NativeMesh, Access::envelope(value), true))
            return true;
        Access::visit_children(value, sink, true);
        for (const auto& part : {value->root, value->base, value->axis})
            if (part && !bytes(part, sink))
                sink.edge(part, geometry_detail::NativeConstructorOwnerKind::ConstructionValues,
                    sizeof(*part) + part->capacity());
        return true;
    }
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
