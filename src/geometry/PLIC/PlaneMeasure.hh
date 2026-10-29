/** @file PlaneMeasure.hh
 * @brief Ordinary volume and face-area queries without closed cut-cell construction.
 */
#pragma once
#include "geometry/PLIC/ConvexGeometry.hh"
#include <span>

namespace SimpleFluid::PLIC
{
/** Numerical quality only. Error estimates are roundoff indicators, never
 * enclosures or certificates of input geometry or construction history. */
enum class PlaneMeasureStatus
{
    Resolved,
    RoundoffLimited
};
struct PlaneVolumeMeasure
{
    long double volume = 0, section_area = 0, numerical_error_estimate = 0;
    PlaneMeasureStatus status = PlaneMeasureStatus::Resolved;
};
struct PlaneAreaMeasure
{
    long double area = 0, numerical_error_estimate = 0;
    PlaneMeasureStatus status = PlaneMeasureStatus::Resolved;
};
/** A strict side for every original face vertex and offset endpoint. */
enum class WholeFacePlaneSide
{
    UnknownOrContact,
    StrictNegative,
    StrictPositive
};
/** Prepared immutable original boundary. Queries clip individual original face
 * loops, integrate them with a reference on the cut plane, and discard them.
 * No cap, closed clipped polyhedron, support identities, or moments are built.
 * Inputs and outputs have the Ordinary arithmetic contract exclusively. */
class OrdinaryPlaneMeasure
{
public:
    explicit OrdinaryPlaneMeasure(const ConvexCellGeometry<ArithmeticMode::Ordinary>&);
    [[nodiscard]] PlaneVolumeMeasure below(const GeometryHalfspace&, bool section_area = false) const;
    /** Evaluate original Ordinary vertices against both finite ordered offsets.
     * Contact or an unresolved sign yields UnknownOrContact. Malformed inputs
     * reject; this witness does not certify construction provenance. */
    [[nodiscard]] static WholeFacePlaneSide strict_face_side(
        std::span<const Point<ArithmeticMode::Ordinary>>, const GeometryHalfspace&, std::array<long double, 2> offsets);

private:
    GeometryPoint origin_{};
    long double scale_ = 0;
    std::vector<std::array<long double, 3>> vertices_;
    std::vector<std::vector<std::size_t>> faces_;
};
/** Original planar polygon coordinates are relative to origin. The loop may
 * have either orientation. Computes only the covered area of the halfspace;
 * intended for translating-plane face-flux quadrature. */
[[nodiscard]] PlaneAreaMeasure ordinary_plane_face_area(
    std::span<const std::array<long double, 3>> vertices, const GeometryHalfspace&, GeometryPoint origin = {});
} // namespace SimpleFluid::PLIC
