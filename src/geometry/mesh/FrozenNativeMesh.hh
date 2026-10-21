/** @file FrozenNativeMesh.hh
 * @brief Opaque accepted-copy native factory input, distinct from exact construction.
 */
#pragma once
#include <memory>
namespace SimpleFluid::Meshes::AMR
{
class FrozenNativeAccess;
class FrozenNativeModel;
} // namespace SimpleFluid::Meshes::AMR
namespace SimpleFluid::Meshes
{

class MultiRegionMesh;
/** Only the accepted-source exporter/model can construct this input. Callers
 * cannot manufacture a copy witness by supplying unvalidated native records. */
class FrozenNativeMeshInput
{
public:
    struct Data;
    FrozenNativeMeshInput(FrozenNativeMeshInput&&) noexcept = default;
    FrozenNativeMeshInput(const FrozenNativeMeshInput&) = default;

private:
    friend class ::SimpleFluid::Meshes::AMR::FrozenNativeAccess;
    friend class ::SimpleFluid::Meshes::AMR::FrozenNativeModel;
    friend class MultiRegionMesh;
    explicit FrozenNativeMeshInput(std::shared_ptr<const Data> data) : data_(std::move(data)) {}
    std::shared_ptr<const Data> data_;
};
} // namespace SimpleFluid::Meshes
