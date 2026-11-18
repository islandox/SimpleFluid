/** Private shared-domain sparse ancestry lookup; wire bytes grant no authority. */
#pragma once
#include "geometry/details/NativeConstructorStorage.hh"
#include "geometry/mesh/AMR/DistributedAncestryDirectory.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct NativeDirectoryLookup
{
    geometry_detail::NativeConstructorStorageOwner storage;
    geometry_detail::NativeConstructorVector<AdaptiveOwner> owners;
    explicit NativeDirectoryLookup(const geometry_detail::NativeConstructionAdmission& admission)
        : storage(geometry_detail::native_constructor_storage(admission)), owners(admission.allocator<AdaptiveOwner>())
    {
    }
    [[nodiscard]] std::size_t size() const noexcept { return owners.size(); }
    [[nodiscard]] const AdaptiveOwner& at(std::size_t index) const { return owners.at(index); }
    [[nodiscard]] const AdaptiveOwner& operator[](std::size_t index) const noexcept { return owners[index]; }
};
class SIMPLEFLUID_LOCAL NativeDirectoryAccess final
{
public:
    [[nodiscard]] static NativeDirectoryLookup lookup(const DistributedAncestryDirectory&,
        std::span<const AdaptiveBytes>, const geometry_detail::NativeConstructionAdmission&, bool require_all = true);
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
