#pragma once
namespace SimpleFluid::PLIC
{
/** A native explicit topology whose geometric constructions use ordinary arithmetic.
 * Surface ancestry, rather than coordinate proximity, selects joins. Both
 * incident cut patterns contribute to their canonical atomic overlay.
 * Sparse packets retain full selected cells and only metric adjacent cells.
 */
template<> class PolyhedralComplex<ArithmeticMode::Ordinary>
{
public:
    struct Impl;
    [[nodiscard]] static std::shared_ptr<const PolyhedralComplex<ArithmeticMode::Ordinary>> make(
        std::vector<PolyhedralComplexCell<ArithmeticMode::Ordinary>> cells);
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
    [[nodiscard]] const ::SimpleFluid::Meshes::AMR::AdaptiveCellKey& cell_key(std::uint64_t) const;
    [[nodiscard]] const PolyhedralSurfaceUse& face_surface(std::uint64_t) const;
    [[nodiscard]] std::vector<char> face_identity(std::uint64_t) const;
    [[nodiscard]] MeshUtils::Vec3 node_coordinates(std::uint64_t) const;
    [[nodiscard]] double cell_volume(std::uint64_t) const;
    [[nodiscard]] MeshUtils::Vec3 cell_centroid(std::uint64_t) const;
    [[nodiscard]] double face_area(std::uint64_t) const;
    [[nodiscard]] MeshUtils::Vec3 face_centroid(std::uint64_t) const;
    [[nodiscard]] MeshUtils::Vec3 face_area_vector(std::uint64_t) const;
    [[nodiscard]] ConvexGeometryPtr<ArithmeticMode::Ordinary> cell_geometry(std::uint64_t) const;
    [[nodiscard]] ConvexGeometryPtr<ArithmeticMode::Ordinary> atomic_cell_geometry(std::uint64_t) const;
    [[nodiscard]] PlanarFacePtr<ArithmeticMode::Ordinary> face_geometry(std::uint64_t) const;
    [[nodiscard]] std::vector<char> encode() const;
    /** Empty explicit selection creates an empty resident shard. */
    [[nodiscard]] std::vector<char> encode(std::span<const std::uint64_t> visible_cells) const;
    [[nodiscard]] static std::shared_ptr<const PolyhedralComplex<ArithmeticMode::Ordinary>> decode(
        std::span<const char> packet);
    /** Capacity-based allocation estimate, including deduplicated finite cell
     * certificates and coordinate storage; excludes allocator headers/shared control
     * blocks. Sparse residency counts and wire bytes are separate evidence. */
    [[nodiscard]] std::size_t storage_bytes() const;

private:
    explicit PolyhedralComplex(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<const Impl> impl_;
};
} // namespace SimpleFluid::PLIC
