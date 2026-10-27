/** @file GeometryBackendTypes.hh
 * @brief Arithmetic-policy declarations and shared geometry resource limits.
 */
#pragma once
#include "utils/ArithmeticMode.hh"
#include "geometry/swept_volume/SweptVolumeDeclarations.hh"
#include <cstddef>
#include <memory>
#include <utility>

namespace SimpleFluid::PLIC
{
template<ArithmeticMode Mode> class ConvexCellGeometry;
template<ArithmeticMode Mode> class PlanarFace;
template<ArithmeticMode Mode> class EdgePointIndex;
template<ArithmeticMode Mode> class GeometryMeasure;
template<ArithmeticMode Mode> class PolyhedralComplex;
template<ArithmeticMode Mode> class ExtrudedConstruction;
template<ArithmeticMode Mode> struct PhaseMeasures;
template<ArithmeticMode Mode> struct PolyhedralComplexCell;
template<ArithmeticMode Mode> struct BackendStatistics;
template<ArithmeticMode Mode> using PlanarFacePtr = std::shared_ptr<const PlanarFace<Mode>>;
} // namespace SimpleFluid::PLIC
