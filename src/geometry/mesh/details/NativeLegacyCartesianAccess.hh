#pragma once
#include "geometry/details/NativeConstructorOwnerSink.hh"
#include "geometry/mesh/MeshRegion.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include <span>
#include <string_view>
namespace SimpleFluid::Meshes::details
{
struct NativeLegacyCartesianRegion
{
    geometry_detail::NativeConstructorStorageOwner metadata;
    NativeIsoRegion<OrthogonalCartesian3D> region;
};
/** The legacy native object remains the metric authority. No RectilinearGeometry
 * replacement or coordinate-only reconstruction is used for owner-local detach. */
class SIMPLEFLUID_LOCAL NativeLegacyCartesianAccess
{
public:
    using Native = OrthogonalCartesian3D;
    using Provider = NativeRegionProvider<Native>;
    using Region = NativeIsoRegion<Native>;
    using Admission = geometry_detail::NativeConstructionAdmission;
    using Sink = geometry_detail::NativeConstructorOwnerSink;
    using Axes = std::array<std::span<const real_t>, 3>;
    using Names = std::array<std::string_view, 6>;
    static constexpr Names default_names{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
    static std::shared_ptr<const Native> make(const Axes&, const Names&, Vec3D<bool>, const Admission&);
    static std::shared_ptr<const Native> detached_copy(const Native&, const Admission&);
    static std::shared_ptr<const Native> detached_copy(const Native&, const OrthoMeshTopo&, const Admission&);
    /** Full existing native-layout tag1 payload: tag/name/layout/three axes/six names.
     * All scalars are memcpy-loaded, so packet alignment is never assumed. */
    static NativeLegacyCartesianRegion decode(std::span<const char>, const Admission&, std::string_view expected_name = {});
    static std::shared_ptr<const Provider> wrap(std::shared_ptr<const Native>, const Admission&);
    static NativeLegacyCartesianRegion make_region(
        std::string_view, const Axes&, const Names&, Vec3D<bool>, const Admission&);
    static NativeLegacyCartesianRegion detach_region(const Region&, const Admission&);
    static void visit(const std::shared_ptr<const Native>&, const Sink&);
    static void visit_provider(const std::shared_ptr<const Provider>&, const Sink&);
    static void visit_region(const Region&, const Sink&);

private:
    static OrthoMeshTopo copy_topology(const OrthoMeshTopo&, const Admission&);
    static std::shared_ptr<const Native> publish(Native::AdmittedStorage&&, const Admission&);
};
} // namespace SimpleFluid::Meshes::details
