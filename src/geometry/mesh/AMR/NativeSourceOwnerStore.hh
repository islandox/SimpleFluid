/** Closed current-source ownership and disjoint source accounting. */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/PLIC/details/GeometryOwnerUnion.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
using NativeContactSideIndexes = std::array<PLIC::details::geometry_demand::GeometryOwnerManifest, 4>;
class SIMPLEFLUID_LOCAL NativeSourceOwnerStore
{
public:
    [[nodiscard]] static PLIC::details::OwnerManifestPair single(
        PLIC::details::geometry_demand::GeometryOwnerManifest);
    [[nodiscard]] static geometry_detail::NativeAllocationMirror mirror(const std::shared_ptr<Budget>&);
    [[nodiscard]] static std::size_t publication_mirror_bytes(const std::shared_ptr<Budget>&);

private:
    [[nodiscard]] static geometry_detail::NativeAllocationMirror make_mirror(
        const std::shared_ptr<Budget>&, bool publication);
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
