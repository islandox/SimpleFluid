#pragma once
#include "geometry/mesh/NativeFragmentRecipe.hh"
#include "geometry/details/NativeObjectOwnerAccess.hh"
namespace SimpleFluid::Meshes
{
class SIMPLEFLUID_LOCAL NativeDescriptorCodecAccess final
{
public:
    static void append(std::vector<char>&, const NativeFragmentDescriptor&);
    [[nodiscard]] static std::shared_ptr<const NativeFragmentDescriptor> decode(std::span<const char>,
        const geometry_detail::NativeConstructionAdmission&,
        const std::shared_ptr<const NativeFragmentDescriptor>& retained_parts = {});
};
} // namespace SimpleFluid::Meshes
