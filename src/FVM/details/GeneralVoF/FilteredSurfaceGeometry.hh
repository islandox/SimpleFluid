/** @file FilteredSurfaceGeometry.hh
 * @brief Pure finite scalar-tetrahedron geometry and canonical facet identities.
 */
#pragma once
#include "FVM/GeneralVoFSurface.hh"
#include <array>
#include <optional>
#include <span>
#include <utility>

namespace SimpleFluid::VoF::general_vof_detail::filtered_surface_kernel
{
using Reason = GeneralVoFSurfaceResolutionReason;
using Wide = std::array<long double, 3>;
using Point = PLIC::GeometryPoint;
// Separate identity namespaces prevent node/face/cell center collisions.
struct NodeId
{
    unsigned kind = 0;
    std::uint64_t id = 0;
    auto operator<=>(const NodeId&) const = default;
};
using CutId = std::pair<NodeId, NodeId>;
struct Vertex
{
    NodeId id;
    Wide p;
    long double q = 0;
};
struct Tetra
{
    std::array<Vertex, 4> v;
    std::size_t cell = 0;
    std::uint64_t ordinal = 0;
};
struct Cut
{
    CutId id;
    Wide p;
};

struct FullFacetRecord
{
    std::size_t patch = 0;
    int orientation = 0;
    unsigned incidence = 0;
};
using FullFacetMap = std::map<std::array<NodeId, 3>, FullFacetRecord>;
[[noreturn]] SIMPLEFLUID_FVM_LOCAL void reject(Reason, const char*);
SIMPLEFLUID_FVM_LOCAL Wide wide(const Point&);
SIMPLEFLUID_FVM_LOCAL Wide sub(const Wide&, const Wide&);
SIMPLEFLUID_FVM_LOCAL Wide cross(const Wide&, const Wide&);
SIMPLEFLUID_FVM_LOCAL long double dot(const Wide&, const Wide&);
SIMPLEFLUID_FVM_LOCAL long double norm(const Wide&);
SIMPLEFLUID_FVM_LOCAL long double volume(const Wide&, const Wide&, const Wide&, const Wide&);
SIMPLEFLUID_FVM_LOCAL Cut intersection(const Vertex&, const Vertex&, long double);
SIMPLEFLUID_FVM_LOCAL long double clipped_volume(const Tetra&, long double);
SIMPLEFLUID_FVM_LOCAL std::vector<Cut> section(const Tetra&, long double, Wide&);
SIMPLEFLUID_FVM_LOCAL std::optional<std::size_t> match_full_facet(
    FullFacetMap&, const std::vector<CutId>&, std::size_t);
SIMPLEFLUID_FVM_LOCAL long double admit_published_volume(
    std::span<const GeneralVoFSurfacePatch>, Point origin, long double target, long double allowance);
} // namespace SimpleFluid::VoF::general_vof_detail::filtered_surface_kernel
