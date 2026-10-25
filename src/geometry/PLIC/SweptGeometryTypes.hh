/** @file SweptGeometryTypes.hh
 * @brief Mode-selected geometry measures and signed transport constructions.
 */
#pragma once
#include "geometry/PLIC/PlanarGeometry.hh"
namespace SimpleFluid::PLIC
{
template<ArithmeticMode Mode> struct SignedSweepSimplex
{
    int sign = 1;
    ConvexFragmentPtr<Mode> geometry;
};
template<ArithmeticMode Mode> struct SweepDonor
{
    std::uint64_t id = 0;
    ConvexGeometryPtr<Mode> geometry;
};
template<ArithmeticMode Mode> struct SweepIntersection
{
    std::uint64_t donor = 0;
    int sign = 1;
    ConvexFragmentPtr<Mode> geometry;
};
template<ArithmeticMode Mode> struct OrientedFaceSweep
{
    std::shared_ptr<const SweptFace<Mode>> face;
    int outward_sign = 1;
};
template<ArithmeticMode Mode> struct SweepRecipient
{
    std::uint64_t id = 0;
    ConvexGeometryPtr<Mode> geometry;
    std::vector<OrientedFaceSweep<Mode>> outward_faces;
};
template<ArithmeticMode Mode> struct TransportAtom
{
    std::uint64_t donor = 0, recipient = 0;
    ConvexFragmentPtr<Mode> geometry;
};
} // namespace SimpleFluid::PLIC
