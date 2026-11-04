/** @file OrdinaryIntegration.hh
 * @brief Shared compensated ordinary shell integration for stored and fixed-size geometry.
 */
#pragma once
#include "geometry/PLIC/details/GeometryKernels.hh"
#include "geometry/PLIC/details/ordinary/Predicates.hh"
namespace SimpleFluid::PLIC::details::ordinary_kernel
{
inline ordinary_predicates::Point predicate_point(const Point& point)
{
    return {point[0].value, point[1].value, point[2].value};
}
inline Scalar measured_volume(const ordinary_predicates::Result& determinant, int exponent)
{
    (void) ordinary_predicates::require_sign(determinant);
    const auto result = scaled<kernel_mode>(Scalar(determinant.value) / Scalar(6), exponent);
    if (result.value == 0 && determinant.value != 0)
    {
        ++statistics.rejections;
        throw GeometryError(
            GeometryErrorCode::UnresolvedArithmetic, "Ordinary volume scaling would lose a nonzero measure.");
    }
    return result;
}

template<ArithmeticMode Mode, class Geometry> Integrals integrate_ordinary(const Geometry& poly, bool moments)
{
    static_assert(Mode == kernel_mode);
    if (poly.vertices.size() == 4 && poly.faces.size() == 4 &&
        std::all_of(poly.faces.begin(), poly.faces.end(), [](const auto& face) { return face.size() == 3; }))
    {
        Integrals result;
        if (!moments)
        {
            const auto& anchor = poly.vertices.front().coordinates;
            for (const auto& face : poly.faces)
                if (std::find(face.begin(), face.end(), 0) == face.end())
                {
                    const auto determinant = ordinary_predicates::orientation(
                        {predicate_point(anchor), predicate_point(poly.vertices[face[0]].coordinates),
                            predicate_point(poly.vertices[face[1]].coordinates),
                            predicate_point(poly.vertices[face[2]].coordinates)});
                    result.volume = measured_volume(determinant, 3 * poly.frame.exponent);
                    break;
                }
        }
        else
        {
            Point center;
            for (const auto& vertex : poly.vertices)
                center = center + vertex.coordinates;
            result.volume = poly.volume;
            result.first = scaled<Mode>(center * Scalar(.25L), poly.frame.exponent) * poly.volume;
        }
        return result;
    }
    Point pivot;
    for (const auto& vertex : poly.vertices)
        pivot = pivot + vertex.coordinates;
    pivot = pivot * (Scalar(1) / Scalar(static_cast<long double>(poly.vertices.size())));
    ordinary_predicates::ResultAccumulator volume;
    std::array<Accumulator<Mode>, 3> first;
    for (const auto& face : poly.faces)
        for (std::size_t i = 1; i + 1 < face.size(); ++i)
        {
            const auto& a = poly.vertices[face[0]].coordinates;
            const auto& b = poly.vertices[face[i]].coordinates;
            const auto& c = poly.vertices[face[i + 1]].coordinates;
            const auto determinant = ordinary_predicates::orientation(
                {predicate_point(pivot), predicate_point(a), predicate_point(b), predicate_point(c)});
            volume.add(determinant);
            if (moments)
            {
                const auto weighted = (pivot + a + b + c) * Scalar(determinant.value);
                for (std::size_t axis = 0; axis < 3; ++axis)
                    first[axis].add(weighted[axis]);
            }
        }
    Integrals result;
    if (moments)
        result.volume = poly.volume;
    else
    {
        // Individual signed cones can be unresolved near a contact. Retain
        // every contribution and its evaluation bound; qualify the total.
        const auto determinant = volume.result();
        result.volume = measured_volume(determinant, 3 * poly.frame.exponent);
    }
    if (moments)
        for (std::size_t axis = 0; axis < 3; ++axis)
            result.first[axis] = scaled<Mode>(first[axis].result() / Scalar(24), 4 * poly.frame.exponent);
    return result;
}

template<ArithmeticMode Mode, class Geometry> Integrals integrate(const Geometry& poly, bool moments)
{
    static_assert(Mode == kernel_mode);
    return integrate_ordinary<Mode>(poly, moments);
}

} // namespace SimpleFluid::PLIC::details::ordinary_kernel
