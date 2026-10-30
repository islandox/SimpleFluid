/** @file ConvexCellTypes.hh
 * @brief Immutable certified convex shells and actual-cell clipping.
 */
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "geometry/PLIC/CertifiedGeometryProperties.hh"
#include "geometry/PLIC/GeometryBackendTypes.hh"
#include "utils/ArithmeticMode.hh"

namespace SimpleFluid::Meshes
{
class MultiRegionMesh;
class ExtrudedTopology;
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::PLIC
{
using Meshes::ExtrudedTopology;
using Meshes::MultiRegionMesh;

/** Retain n dot (x-anchor) <= offset. All inputs must be finite; the nonzero
 * normal need not be normalized. A represented cell-local anchor plus scalar
 * offset preserves cuts too small to encode as a translated point on a plane.
 */
struct GeometryHalfspace
{
    GeometryPoint normal{}, anchor{};
    double offset = 0;
};

template<ArithmeticMode Mode> struct Section;
template<ArithmeticMode Mode> using GeometrySection = Section<Mode>;

/** Native represented coordinates and certified outward loops only. This is
 * not an export format for exact rational vertices created by clipping. */
struct NativeConvexShell
{
    std::vector<GeometryPoint> vertices;
    std::vector<std::vector<std::size_t>> outward_faces;
};

/** Enclosed coordinates exported from an authenticated source shell. Values
 * are relative to origin, preserving small cells at translated coordinates.
 * This is numerical input with uncertainty, never a lossless exact certificate. */
template<ArithmeticMode Mode> struct NumericalConvexShell
{
    GeometryPoint origin{};
    std::vector<std::array<CertifiedGeometryValue<Mode>, 3>> relative_vertices;
    std::vector<std::vector<std::size_t>> outward_faces;
    /** Collision-free opaque source support identities. The numerical importer
     * may use equality for authenticated incidence, never as FP coefficients. */
    std::vector<PlaneSupportIdentity<Mode>> face_support_identities;
};

template<ArithmeticMode Mode> class ConvexCellGeometry;
/** Frozen canonical source/cell key material; independent of wire envelopes. */
[[nodiscard]] std::vector<char> canonical_cell_key(const ConvexCellGeometry<ArithmeticMode::Ordinary>&);
[[nodiscard]] std::vector<char> canonical_cell_key(const ConvexCellGeometry<ArithmeticMode::Exact>&);

template<ArithmeticMode Mode> using ConvexGeometryPtr = std::shared_ptr<const ConvexCellGeometry<Mode>>;

struct ConvexFaceOrigin
{
    enum class Kind
    {
        ParentFace,
        CutPlane
    };
    Kind kind = Kind::ParentFace;
    std::size_t index = 0;
};

template<ArithmeticMode Mode> struct ConvexOctantPiece
{
    unsigned octant = 0;
    ConvexGeometryPtr<Mode> geometry;
    std::vector<ConvexFaceOrigin> face_origins;
};

template<ArithmeticMode Mode> struct ClipResult;
template<ArithmeticMode Mode> using ConvexClipResult = ClipResult<Mode>;

} // namespace SimpleFluid::PLIC
