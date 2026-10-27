/** @file SweptVolumeDeclarations.hh
 * @brief Swept-volume templates and construction resource limits.
 */
#pragma once
#include "utils/ArithmeticMode.hh"
#include <cstddef>
#include <memory>

namespace SimpleFluid::geometry::swept_volume
{
template<ArithmeticMode Mode> class ConvexFragment;
template<ArithmeticMode Mode> class NodeTrajectory;
template<ArithmeticMode Mode> class SweptFace;
template<ArithmeticMode Mode> class SweepAccess;
template<ArithmeticMode Mode> class DonorExport;
template<ArithmeticMode Mode> class TransportPartition;
template<ArithmeticMode Mode> struct SignedSweepSimplex;
template<ArithmeticMode Mode> struct SweepDonor;
template<ArithmeticMode Mode> struct SweepIntersection;
template<ArithmeticMode Mode> struct OrientedFaceSweep;
template<ArithmeticMode Mode> struct SweepRecipient;
template<ArithmeticMode Mode> struct TransportAtom;
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
} // namespace SimpleFluid::geometry::swept_volume
