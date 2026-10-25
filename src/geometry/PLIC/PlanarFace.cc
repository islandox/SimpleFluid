#include "geometry/PLIC/details/GeometryKernels.hh"

#include "geometry/PLIC/details/bounded/PlanarFace.tcc"
#include "geometry/PLIC/details/ordinary/PlanarFace.tcc"

namespace SimpleFluid::PLIC
{
template<ArithmeticMode Mode>
PlanarFaceOverlap<Mode> planar_face_overlap(std::span<const Point<Mode>> first, std::span<const Point<Mode>> second,
    const PlaneAuthority& support, GeometryLimits limits)
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);
    limits.arithmetic_mode = Mode;
    details::Operation scope(limits);
    if (first.size() < 3 || second.size() < 3 || first.size() > limits.maximum_vertices ||
        second.size() > limits.maximum_vertices || support.key.empty())
        details::reject(Mode, "planar overlap requires admitted nondegenerate source loops and a support authority.");
    details::active_work->charge(first.size(), sizeof(DynamicPoint));
    details::active_work->charge(second.size(), sizeof(DynamicPoint));
    std::vector<DynamicPoint> a, b;
    a.reserve(first.size());
    b.reserve(second.size());
    for (const auto& point : first)
        a.push_back(point.dynamic());
    for (const auto& point : second)
        b.push_back(point.dynamic());
    const auto source = [&]
    {
        if constexpr (Mode == ArithmeticMode::Ordinary)
            return details::ordinary_kernel::dynamic_planar_face_overlap<Mode>(a, b, support, limits);
        else
            return details::certification_kernel::dynamic_planar_face_overlap<Mode>(a, b, support, limits);
    }();
    const auto value = [](const DynamicGeometryValue& input) -> GeometryValue<Mode>
    {
        details::require_mode(input.arithmetic_mode, Mode);
        if constexpr (Mode == ArithmeticMode::Ordinary)
            return {input.value};
        else
            return {input.value, input.absolute_error};
    };
    PlanarFaceOverlap<Mode> result;
    result.first_projected_area = value(source.first_projected_area);
    result.second_projected_area = value(source.second_projected_area);
    result.overlap_projected_area = value(source.overlap_projected_area);
    result.overlap_over_first = value(source.overlap_over_first);
    result.overlap_over_second = value(source.overlap_over_second);
    details::active_work->charge(source.overlap_vertices.size(), sizeof(Point<Mode>));
    result.overlap_vertices.reserve(source.overlap_vertices.size());
    for (const auto& point : source.overlap_vertices)
        result.overlap_vertices.emplace_back(point);
    result.orientation = source.orientation;
    result.dropped_axis = source.dropped_axis;
    return result;
}
template PlanarFaceOverlap<ArithmeticMode::Ordinary> planar_face_overlap(
    std::span<const Point<ArithmeticMode::Ordinary>>, std::span<const Point<ArithmeticMode::Ordinary>>,
    const PlaneAuthority&, GeometryLimits);
template PlanarFaceOverlap<ArithmeticMode::BoundedCertification> planar_face_overlap(
    std::span<const Point<ArithmeticMode::BoundedCertification>>,
    std::span<const Point<ArithmeticMode::BoundedCertification>>, const PlaneAuthority&, GeometryLimits);
} // namespace SimpleFluid::PLIC
