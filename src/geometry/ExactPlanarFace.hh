/** @file ExactPlanarFace.hh
 * @brief Authoritative immutable planar face certificates for nested native joins.
 */
#pragma once
#include "geometry/ConvexCellGeometry.hh"
#include "geometry/MeshUtils.hh"
namespace SimpleFluid::Meshes
{
class ExactPlanarFace;
using ExactPlanarFacePtr = std::shared_ptr<const ExactPlanarFace>;
class ExactPlanarFace
{
public:
    struct Impl;
    [[nodiscard]] static ExactPlanarFacePtr make(std::span<const GeometryPoint> vertices);
    [[nodiscard]] static ExactPlanarFacePtr from_native_face(const MultiRegionMesh&, std::uint64_t face);
    [[nodiscard]] CertifiedGeometryValue area() const;
    [[nodiscard]] MeshUtils::Vec3 centroid() const;
    [[nodiscard]] MeshUtils::Vec3 area_vector() const;
    [[nodiscard]] ExactPlanarFacePtr reversed() const;
    [[nodiscard]] ExactPlanarFacePtr translated(GeometryPoint displacement) const;
    [[nodiscard]] ExactPlanarFacePtr oriented_like(GeometryPoint area_vector) const;
    [[nodiscard]] bool equivalent(const ExactPlanarFace&) const;
    [[nodiscard]] bool cell_bounds_overlap(const ConvexCellGeometry&) const;
    [[nodiscard]] bool contains(const ExactPlanarFace&) const;
    [[nodiscard]] bool oppositely_oriented(const ExactPlanarFace&) const;
    [[nodiscard]] static ConvexGeometryPtr cell_shell(std::span<const ExactPlanarFacePtr> outward_faces);
    /** Opposite winding is accepted; every tile must be whole and contained.
     * Exact coplanarity, disjointness, area and first moments are mandatory. */
    void certify_partition(std::span<const ExactPlanarFacePtr> tiles) const;
    [[nodiscard]] std::vector<char> encode() const;
    [[nodiscard]] static ExactPlanarFacePtr decode(std::span<const char>);

private:
    friend class ExactExtrudedConstruction;
    friend class ExactPolyhedralComplex;
    explicit ExactPlanarFace(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<const Impl> impl_;
};
} // namespace SimpleFluid::Meshes
