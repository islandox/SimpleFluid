/** @file GeneralVoFFlowSupport.hh
 * @brief Private coordinate operations shared by finite and exact projection.
 */
#pragma once
#include "solvers/VoFSolver.hh"
#include "geometry/MeshHandle.hh"
namespace SimpleFluid::detail::flow_support
{
using Vector = VoF::Vector;
inline double dot(const Vector& a, const Vector& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline Vector convert(const MeshUtils::Vec3& a)
{
    return {a.x, a.y, a.z};
}
} // namespace SimpleFluid::detail::flow_support
