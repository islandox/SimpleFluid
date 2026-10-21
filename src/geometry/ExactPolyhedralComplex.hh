/** @file ExactPolyhedralComplex.hh
 * @brief Canonical exact atomic faces for convex adaptive cell complexes.
 */
#pragma once
#include "geometry/AdaptiveHierarchy.hh"
#include "geometry/ConvexCellGeometry.hh"
#include <string>

namespace SimpleFluid::Meshes
{
struct PolyhedralSurfaceKey
{
    enum class Kind : std::uint64_t
    {
        Root,
        Cut
    };
    Kind kind = Kind::Root;
    std::uint64_t region = 0;
    ConvexCellAncestry creator;
    std::uint64_t ordinal = 0;
    auto operator<=>(const PolyhedralSurfaceKey&) const = default;
};
struct PolyhedralSurfaceUse
{
    PolyhedralSurfaceKey key;
    int orientation = 1;
    bool exterior = true;
    /** Empty is a valid unnamed physical exterior; interior uses stay empty. */
    std::string boundary;
};
struct PolyhedralComplexCell
{
    AdaptiveCellKey key;
    std::uint64_t gid = 0;
    ConvexGeometryPtr geometry;
    std::vector<PolyhedralSurfaceUse> surfaces;
};

/** A native explicit topology whose geometric constructions stay exact.
 * Surface ancestry, rather than coordinate proximity, selects joins. Both
 * incident cut patterns contribute to their canonical atomic overlay.
 * Sparse packets retain full selected cells and only metric adjacent cells.
 */
class ExactPolyhedralComplex
{
public:
    struct Impl;
    [[nodiscard]] static std::shared_ptr<const ExactPolyhedralComplex> make(std::vector<PolyhedralComplexCell> cells);
    [[nodiscard]] std::uint64_t global_cells() const noexcept;
    [[nodiscard]] std::uint64_t global_faces() const noexcept;
    [[nodiscard]] std::uint64_t global_nodes() const noexcept;
    [[nodiscard]] const std::vector<std::uint64_t>& cell_ids() const noexcept;
    [[nodiscard]] const std::vector<std::uint64_t>& resident_cell_ids() const noexcept;
    [[nodiscard]] const std::vector<std::uint64_t>& face_ids() const noexcept;
    [[nodiscard]] const std::vector<std::uint64_t>& node_ids() const noexcept;
    [[nodiscard]] bool has_full_cell(std::uint64_t) const noexcept;
    [[nodiscard]] const std::vector<std::uint64_t>& cell_faces(std::uint64_t) const;
    [[nodiscard]] const std::vector<std::uint64_t>& cell_nodes(std::uint64_t) const;
    [[nodiscard]] const std::vector<std::uint64_t>& face_nodes(std::uint64_t) const;
    [[nodiscard]] std::uint64_t owner_cell(std::uint64_t) const;
    [[nodiscard]] std::uint64_t neighbor_cell(std::uint64_t) const;
    [[nodiscard]] const AdaptiveCellKey& cell_key(std::uint64_t) const;
    [[nodiscard]] const PolyhedralSurfaceUse& face_surface(std::uint64_t) const;
    [[nodiscard]] std::vector<char> face_identity(std::uint64_t) const;
    [[nodiscard]] MeshUtils::Vec3 node_coordinates(std::uint64_t) const;
    [[nodiscard]] double cell_volume(std::uint64_t) const;
    [[nodiscard]] MeshUtils::Vec3 cell_centroid(std::uint64_t) const;
    [[nodiscard]] double face_area(std::uint64_t) const;
    [[nodiscard]] MeshUtils::Vec3 face_centroid(std::uint64_t) const;
    [[nodiscard]] MeshUtils::Vec3 face_area_vector(std::uint64_t) const;
    [[nodiscard]] ConvexGeometryPtr cell_geometry(std::uint64_t) const;
    [[nodiscard]] ConvexGeometryPtr atomic_cell_geometry(std::uint64_t) const;
    [[nodiscard]] std::vector<char> encode() const;
    /** Empty explicit selection creates an empty resident shard. */
    [[nodiscard]] std::vector<char> encode(std::span<const std::uint64_t> visible_cells) const;
    [[nodiscard]] static std::shared_ptr<const ExactPolyhedralComplex> decode(std::span<const char> packet);
    /** Capacity-based allocation estimate, including deduplicated exact cell
     * certificates and bigint limbs; excludes allocator headers/shared control
     * blocks. Sparse residency counts and wire bytes are separate evidence. */
    [[nodiscard]] std::size_t storage_bytes() const;

private:
    explicit ExactPolyhedralComplex(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<const Impl> impl_;
};
} // namespace SimpleFluid::Meshes
