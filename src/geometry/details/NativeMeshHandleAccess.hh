/** @file NativeMeshHandleAccess.hh
 * @brief Local preallocated handle initialization around actual admitted maps.
 */
#pragma once
#include "geometry/MeshHandle.hh"
#include "parallel/details/NativePartitionAccess.hh"
namespace SimpleFluid::geometry_detail
{
class SIMPLEFLUID_LOCAL NativeMeshHandleAccess final
{
public:
    [[nodiscard]] static std::shared_ptr<const MeshHandle<>> make(
        const NativePartitionAccess::Result&, const NativeConstructionAdmission&);
    [[nodiscard]] static std::size_t library_upper(const std::shared_ptr<const MeshHandle<>>&);
    [[nodiscard]] static std::size_t scalar_bytes(const std::shared_ptr<const MeshHandle<>>&);
    static void visit(const std::shared_ptr<const MeshHandle<>>&, const NativeConstructorOwnerSink&,
        PLIC::details::GeometryOwnerGraphEditor&, bool, PLIC::details::NativeGeometryAccounting*);
};
} // namespace SimpleFluid::geometry_detail
