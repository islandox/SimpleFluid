/** @file TestMeshes.hh
 * @brief Named custom meshes shared with their explicit implementation providers.
 */
#pragma once
#include "FVM/TransportSystem.hh"
#include "geometry/MeshHandle.hh"
#include <cstdint>

namespace SimpleFluid::test::transport_providers
{
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;

/** Same providers, IDs and maps as Handle, with generic cached-row dispatch. */
struct ReferenceHandle : Handle
{
    using Handle::Handle;
    bool supports_region_execution() const noexcept { return false; }
};

/** Minimal matrix-map fixture; changing epoch does not change row slots. */
struct EpochMesh
{
    const Handle& mesh;
    std::uint64_t epoch = 0;
    auto owned_cell_map() const { return mesh.owned_cell_map(); }
    auto overlap_cell_map() const { return mesh.overlap_cell_map(); }
    auto num_owned_cells() const { return mesh.num_owned_cells(); }
    auto geometry_epoch() const { return epoch; }
};
} // namespace SimpleFluid::test::transport_providers

namespace SimpleFluid
{
extern template class MeshTraversePlan<test::transport_providers::ReferenceHandle>;
} // namespace SimpleFluid
namespace SimpleFluid::FVM
{
extern template class TransportGeometryCache<test::transport_providers::ReferenceHandle>;
} // namespace SimpleFluid::FVM
