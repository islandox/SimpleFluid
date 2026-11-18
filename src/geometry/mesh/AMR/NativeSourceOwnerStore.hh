/** Closed current-source ownership and disjoint source accounting. */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class SIMPLEFLUID_LOCAL NativeSourceOwnerStore
{
public:
    [[nodiscard]] static geometry_detail::NativeAllocationMirror mirror(const std::shared_ptr<Budget>&);
    [[nodiscard]] static std::size_t publication_mirror_bytes(const std::shared_ptr<Budget>&);

private:
    [[nodiscard]] static geometry_detail::NativeAllocationMirror make_mirror(
        const std::shared_ptr<Budget>&, bool publication);
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
