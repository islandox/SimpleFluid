#pragma once
#include "geometry/PLIC/ConvexCellTypes.hh"
#include "geometry/PLIC/GeometryPacketCodec.hh"
namespace SimpleFluid::PLIC
{
template<> struct Section<ArithmeticMode::Exact>
{
    /** Display vertices only (unordered for supporting contact); exact geometry
     * remains retained internally for every subsequent operation. */
    std::vector<std::array<long double, 3>> vertices;
    CertifiedGeometryValue<ArithmeticMode::Exact> area; ///< m^2; face tangency can have positive area
};
template<> struct ClipResult<ArithmeticMode::Exact>
{
    ConvexGeometryPtr<ArithmeticMode::Exact> inside, outside; ///< null only when proven zero volume
    GeometrySection<ArithmeticMode::Exact> section;
    /** Areas of each input shell face remaining on the corresponding side. */
    std::vector<CertifiedGeometryValue<ArithmeticMode::Exact>> inside_apertures, outside_apertures;
};

/** Exact predicates/constructions on represented binary64 input, hidden in .cc.
 * Face loops must be simple, planar, outward, convex and form a closed manifold.
 * Inconclusive or unrepresentable output rejects; positive pieces never vanish.
 */
template<> class ConvexCellGeometry<ArithmeticMode::Exact>
{
public:
    static constexpr ArithmeticMode mode = ArithmeticMode::Exact;
    [[nodiscard]] constexpr ArithmeticMode arithmetic_mode() const noexcept { return mode; }
    struct Impl;
    [[nodiscard]] static ConvexGeometryPtr<ArithmeticMode::Exact> make(
        std::span<const GeometryPoint> vertices, const std::vector<std::vector<std::size_t>>& outward_faces);
    [[nodiscard]] static ConvexGeometryPtr<ArithmeticMode::Exact> from_geometry(
        const ConvexCellGeometry<ArithmeticMode::Ordinary>&);
    [[nodiscard]] static ConvexGeometryPtr<ArithmeticMode::Exact> box(GeometryPoint lower, GeometryPoint upper);
    /** CCW planar XY polygon crossed with an ordered positive Z interval. */
    [[nodiscard]] static ConvexGeometryPtr<ArithmeticMode::Exact> extruded_polygon(
        std::span<const std::array<double, 2>> polygon, double lower_z, double upper_z);
    /** Adapt resident native convex cell loops, propagating all exact collinear
     * edge vertices. Translated periodic joins currently reject explicitly. */
    [[nodiscard]] static ConvexGeometryPtr<ArithmeticMode::Exact> from_native_cell(
        const MultiRegionMesh& mesh, std::uint64_t cell);
    /** Exact shell faces in native cell_faces order, including canonical atomic
     * tiles. Use this when assigning native face ancestry to a new root. */
    [[nodiscard]] static ConvexGeometryPtr<ArithmeticMode::Exact> from_native_cell_atomic(
        const MultiRegionMesh& mesh, std::uint64_t cell);
    /** Original binary64 vertices, exact outward orientation and propagated
     * incident edge vertices; suitable for sparse native-cell communication. */
    [[nodiscard]] static NativeConvexShell native_cell_shell(const MultiRegionMesh& mesh, std::uint64_t cell);
    /** Certify native atomic face loops after exact edge-vertex propagation;
     * equal coordinates merge exactly, never within a spatial tolerance. */
    [[nodiscard]] static ConvexGeometryPtr<ArithmeticMode::Exact> from_native_shell(
        std::span<const GeometryPoint> vertices, const std::vector<std::vector<std::size_t>>& outward_faces);
    /** Export source coordinates once for bounded numerical geometry. This
     * conversion uses the source authority to enclose rounding; it computes no
     * exact volume, moment, intersection or finite transport ledger. */
    [[nodiscard]] NumericalConvexShell<ArithmeticMode::Exact> numerical_shell() const;
    /** Exact oriented volume with the same checked publication as moments().volume. */
    [[nodiscard]] CertifiedGeometryValue<ArithmeticMode::Exact> volume() const;
    [[nodiscard]] GeometryMoments<ArithmeticMode::Exact> moments() const;
    [[nodiscard]] GeometryPoint centroid() const;
    [[nodiscard]] std::vector<char> encode_certificate(std::size_t byte_budget = geometry_packet_default_budget) const;
    /** Capacity-based allocation estimate including exact arrays/limbs; excludes
     * allocator headers and shared control blocks, and does not measure RSS. */
    [[nodiscard]] std::size_t storage_bytes() const;
    [[nodiscard]] static ConvexGeometryPtr<ArithmeticMode::Exact> decode_certificate(
        std::span<const char> packet, std::size_t byte_budget = geometry_packet_default_budget);
    /** Binary64 value/interval output, also exercising platforms whose native
     * long double has binary64 range. First-moment overflow rejects even when
     * volume itself is representable; first-moment underflow uses an enclosure.
     * Exact internal moments remain retained. */
    [[nodiscard]] GeometryMoments<ArithmeticMode::Exact> moments_binary64() const;
    [[nodiscard]] std::shared_ptr<const PlanarFace<ArithmeticMode::Exact>> face_geometry(std::size_t face) const;
    [[nodiscard]] std::size_t num_faces() const noexcept;
    /** Three exact AABB midpoint cuts, retaining every positive intersection.
     * Quality compares exact child volume to its largest exact AABB span cubed.
     * Any unsupported positive sliver rejects the entire split transaction. */
    [[nodiscard]] std::vector<ConvexOctantPiece<ArithmeticMode::Exact>> split_aabb_midpoints(
        double minimum_scaled_volume = 0) const;
    [[nodiscard]] bool quality_above(double minimum_scaled_volume) const;
    [[nodiscard]] std::array<CertifiedGeometryValue<ArithmeticMode::Exact>, 3> axis_spans() const;
    [[nodiscard]] bool axis_span_exceeds(const ConvexCellGeometry&, std::size_t axis, double ratio) const;
    [[nodiscard]] ConvexClipResult<ArithmeticMode::Exact> clip(const GeometryHalfspace& halfspace) const;
    [[nodiscard]] ConvexGeometryPtr<ArithmeticMode::Exact> intersect(const ConvexCellGeometry& other) const;
    /** Exact coverage/disjointness/containment against the actual represented shell. */
    void certify_partition(std::span<const ConvexGeometryPtr<ArithmeticMode::Exact>> pieces) const;
    /** Exact geometry equality, independent of shell ordering or construction order. */
    [[nodiscard]] bool equivalent(const ConvexCellGeometry& other) const;

private:
    friend std::vector<char> canonical_cell_key(const ConvexCellGeometry<ArithmeticMode::Exact>&);
    friend class ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>;
    friend class ExtrudedConstruction<ArithmeticMode::Exact>;
    friend class PolyhedralComplex<ArithmeticMode::Exact>;
    friend class PlanarFace<ArithmeticMode::Exact>;
    explicit ConvexCellGeometry(std::shared_ptr<const Impl> implementation);
    std::shared_ptr<const Impl> impl_;
};
} // namespace SimpleFluid::PLIC
