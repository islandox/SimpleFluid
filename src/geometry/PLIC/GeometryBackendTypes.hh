/** @file GeometryBackendTypes.hh
 * @brief Arithmetic-policy declarations and shared geometry resource limits.
 */
#pragma once
#include "utils/ArithmeticMode.hh"
#include <cstddef>
#include <memory>
#include <utility>

namespace SimpleFluid::PLIC
{
template<ArithmeticMode Mode> class PlanarFace;
template<ArithmeticMode Mode> class EdgePointIndex;
template<ArithmeticMode Mode> class GeometryMeasure;
template<ArithmeticMode Mode> class ConvexFragment;
template<ArithmeticMode Mode> class NodeTrajectory;
template<ArithmeticMode Mode> class SweptFace;
template<ArithmeticMode Mode> class DonorExport;
template<ArithmeticMode Mode> class TransportPartition;
template<ArithmeticMode Mode> class PolyhedralComplex;
template<ArithmeticMode Mode> class ExtrudedConstruction;
template<ArithmeticMode Mode> struct SignedSweepSimplex;
template<ArithmeticMode Mode> struct SweepDonor;
template<ArithmeticMode Mode> struct SweepIntersection;
template<ArithmeticMode Mode> struct PhaseMeasures;
template<ArithmeticMode Mode> struct OrientedFaceSweep;
template<ArithmeticMode Mode> struct SweepRecipient;
template<ArithmeticMode Mode> struct TransportAtom;
template<ArithmeticMode Mode> struct PolyhedralComplexCell;
template<ArithmeticMode Mode> struct BackendStatistics;
template<ArithmeticMode Mode> using PlanarFacePtr = std::shared_ptr<const PlanarFace<Mode>>;
template<ArithmeticMode Mode> using ConvexFragmentPtr = std::shared_ptr<const ConvexFragment<Mode>>;
struct SweepLimits
{
    std::size_t maximum_simplices = 4096, maximum_atoms = 65536;
    std::size_t maximum_pair_tests = 262144;
    std::size_t byte_budget = 64 * 1024 * 1024;
};
struct TrajectoryLimits
{
    std::size_t maximum_constraints = 256, maximum_operations = 65536;
    std::size_t maximum_integer_bits = 1024 * 1024; ///< Operand cap for the explicit Exact specialization.
    std::size_t byte_budget = 64 * 1024 * 1024;
};
} // namespace SimpleFluid::PLIC
