/** @file ConvexCellGeometry.hh
 * @brief Immutable certified convex shells and actual-cell clipping.
 */
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace SimpleFluid::Meshes
{
class MultiRegionMesh;
using GeometryPoint = std::array<double, 3>;

/** Finite value and rigorous absolute conversion bound in the indicated SI
 * units. Nonzero first moments below the output range use an outward interval
 * containing zero and retain exact_nonzero. Overflow rejects; volume must
 * remain strictly positive and representable. Exact internal moments survive
 * every supported conversion unchanged. */
struct CertifiedGeometryValue
{
    long double value = 0, absolute_error = 0;
    /** Exact value is nonzero even when an outward moment interval straddles 0. */
    bool exact_nonzero = false;
    /** Reject overflow or disappearance of a nonzero represented measure. */
    [[nodiscard]] double as_double() const;
    /** Bound including the additional checked binary64 narrowing, same units. */
    [[nodiscard]] long double binary64_absolute_error() const;
};

struct GeometryMoments
{
    CertifiedGeometryValue volume; ///< m^3
    GeometryPoint origin{};
    std::array<CertifiedGeometryValue, 3> first_moment; ///< integral (x-origin) dV, m^4
};

/** Retain n dot (x-anchor) <= offset. All inputs must be finite; the nonzero
 * normal need not be normalized. A represented cell-local anchor plus scalar
 * offset preserves cuts too small to encode as a translated point on a plane.
 */
struct GeometryHalfspace
{
    GeometryPoint normal{}, anchor{};
    double offset = 0;
};

struct GeometrySection
{
    /** Display vertices only (unordered for supporting contact); exact geometry
     * remains retained internally for every subsequent operation. */
    std::vector<std::array<long double, 3>> vertices;
    CertifiedGeometryValue area; ///< m^2; face tangency can have positive area
};

/** Native represented coordinates and certified outward loops only. This is
 * not an export format for exact rational vertices created by clipping. */
struct NativeConvexShell
{
    std::vector<GeometryPoint> vertices;
    std::vector<std::vector<std::size_t>> outward_faces;
};

class ConvexCellGeometry;
using ConvexGeometryPtr = std::shared_ptr<const ConvexCellGeometry>;
struct ConvexClipResult
{
    ConvexGeometryPtr inside, outside; ///< null only when proven zero volume
    GeometrySection section;
    /** Areas of each input shell face remaining on the corresponding side. */
    std::vector<CertifiedGeometryValue> inside_apertures, outside_apertures;
};

/** Exact predicates/constructions on represented binary64 input, hidden in .cc.
 * Face loops must be simple, planar, outward, convex and form a closed manifold.
 * Inconclusive or unrepresentable output rejects; positive pieces never vanish.
 */
class ConvexCellGeometry
{
public:
    struct Impl;
    [[nodiscard]] static ConvexGeometryPtr make(
        std::span<const GeometryPoint> vertices, const std::vector<std::vector<std::size_t>>& outward_faces);
    [[nodiscard]] static ConvexGeometryPtr box(GeometryPoint lower, GeometryPoint upper);
    /** CCW planar XY polygon crossed with an ordered positive Z interval. */
    [[nodiscard]] static ConvexGeometryPtr extruded_polygon(
        std::span<const std::array<double, 2>> polygon, double lower_z, double upper_z);
    /** Adapt resident native convex cell loops, propagating all exact collinear
     * edge vertices. Translated periodic joins currently reject explicitly. */
    [[nodiscard]] static ConvexGeometryPtr from_native_cell(const MultiRegionMesh& mesh, std::uint64_t cell);
    /** Original binary64 vertices, exact outward orientation and propagated
     * incident edge vertices; suitable for sparse native-cell communication. */
    [[nodiscard]] static NativeConvexShell native_cell_shell(const MultiRegionMesh& mesh, std::uint64_t cell);
    /** Certify native atomic face loops after exact edge-vertex propagation;
     * equal coordinates merge exactly, never within a spatial tolerance. */
    [[nodiscard]] static ConvexGeometryPtr from_native_shell(
        std::span<const GeometryPoint> vertices, const std::vector<std::vector<std::size_t>>& outward_faces);
    [[nodiscard]] GeometryMoments moments() const;
    /** Binary64 value/interval output, also exercising platforms whose native
     * long double has binary64 range. First-moment overflow rejects even when
     * volume itself is representable; first-moment underflow uses an enclosure.
     * Exact internal moments remain retained. */
    [[nodiscard]] GeometryMoments moments_binary64() const;
    [[nodiscard]] std::size_t num_faces() const noexcept;
    [[nodiscard]] ConvexClipResult clip(const GeometryHalfspace& halfspace) const;
    [[nodiscard]] ConvexGeometryPtr intersect(const ConvexCellGeometry& other) const;
    /** Exact coverage/disjointness/containment against the actual represented shell. */
    void certify_partition(std::span<const ConvexGeometryPtr> pieces) const;
    /** Exact geometry equality, independent of shell ordering or construction order. */
    [[nodiscard]] bool equivalent(const ConvexCellGeometry& other) const;

private:
    explicit ConvexCellGeometry(std::shared_ptr<const Impl> implementation);
    std::shared_ptr<const Impl> impl_;
};
} // namespace SimpleFluid::Meshes
