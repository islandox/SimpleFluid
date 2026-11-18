/** Exact native planar ownership, separate from geometric/source authority. */
#pragma once
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/details/NativeConstructorStorage.hh"
namespace SimpleFluid::PLIC::details
{
class SIMPLEFLUID_LOCAL NativePlanarOwnerAccess final
{
public:
    using Face = PlanarFace<ArithmeticMode::Ordinary>;
    using Pointer = PlanarFacePtr<ArithmeticMode::Ordinary>;
    [[nodiscard]] static bool admitted(const Pointer&);
    [[nodiscard]] static const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& wrapper_envelope(
        const Pointer&);
    [[nodiscard]] static std::shared_ptr<const void> payload_owner(const Pointer&);
    [[nodiscard]] static const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& payload_envelope(
        const Pointer&);
    [[nodiscard]] static const geometry_detail::NativeConstructorStorageOwner& storage(const Pointer&);
    [[nodiscard]] static Pointer retained(std::span<const Face::WidePoint>, Face::WidePoint normal,
        PlaneSupportIdentity<ArithmeticMode::Ordinary>, const geometry_detail::NativeConstructionAdmission&);
    [[nodiscard]] static PlaneSupportIdentity<ArithmeticMode::Ordinary> source_support(
        std::span<const Face::WidePoint>, const geometry_detail::NativeConstructionAdmission&);
    [[nodiscard]] static Pointer cell_face(const ConvexCellGeometry<ArithmeticMode::Ordinary>&, std::size_t,
        const geometry_detail::NativeConstructionAdmission&);
    [[nodiscard]] static Pointer make(
        std::span<const GeometryPoint>, const geometry_detail::NativeConstructionAdmission&);
    [[nodiscard]] static Pointer make_wide(
        std::span<const Face::WidePoint>, const geometry_detail::NativeConstructionAdmission&);
    [[nodiscard]] static Pointer translated(const Face&, const DisplacementIdentity<ArithmeticMode::Ordinary>&,
        const geometry_detail::NativeConstructionAdmission&);
    [[nodiscard]] static Pointer decode(
        std::span<const char>, std::size_t, const geometry_detail::NativeConstructionAdmission&);
    [[nodiscard]] static Pointer oriented_like(
        const Pointer&, GeometryPoint, const geometry_detail::NativeConstructionAdmission&);
    [[nodiscard]] static Pointer intersection(
        const Face&, const Face&, const geometry_detail::NativeConstructionAdmission&);
    [[nodiscard]] static geometry_detail::NativeAdministrationVector<Pointer> subtract(
        const Face&, std::span<const Pointer>, const geometry_detail::NativeConstructionAdmission&);
    static void certify_partition(
        const Face&, std::span<const Pointer>, const geometry_detail::NativeConstructionAdmission&);
    [[nodiscard]] static Pointer reversed(const Face&, const geometry_detail::NativeConstructionAdmission&);
};
} // namespace SimpleFluid::PLIC::details
