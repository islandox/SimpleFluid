/** @file GeometryStatisticsSnapshot.hh
 * @brief Image-local geometry instrumentation identities and POD snapshots.
 */
#pragma once
#include "geometry/BoundedConvexGeometry.hh"
#include "geometry/ExactGeometryInstrumentation.hh"
namespace SimpleFluid::Meshes
{
/** Each static Mesh archive image may own separate thread-local counters.
 * Aggregate exact and bounded domains independently, counting a domain pointer
 * once. This identifies the linked images queried by a caller, not every exact
 * library in a process. Bounded counters are always active; exact sampling is
 * opt-in and must be enabled in each observed image. */
struct GeometryStatisticsSnapshot
{
    const void* exact_domain = nullptr;
    const void* bounded_domain = nullptr;
    ExactGeometryStatistics exact;
    BoundedGeometryStatistics bounded;
};
}
