/** @file SparseNativeMesh.hh
 * @brief Owner-local native dispatch routes with procedural interior geometry.
 */
#pragma once
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/details/NativeConstructionAdmission.hh"
#include "geometry/details/NativeSchemaOwner.hh"
#include "geometry/mesh/MultiRegionMesh.hh"
#include <algorithm>
#include <map>
#include <span>
#include <variant>
namespace SimpleFluid::Meshes
{
/** Native recipe fragments are local to this shard. Full cells expose topology;
 * metric-only incident neighbors expose only their validated volume/centroid. */
struct SparseNativeCell
{
    std::uint64_t gid = 0;
    int owner = -1;
    std::size_t region = 0;
    std::uint64_t native_cell = 0;
    bool full = false;
    double volume = 0;
    PLIC::GeometryPoint centroid{};
};
struct SparseNativeFace
{
    std::uint64_t gid = 0, first = 0, second = MultiRegionMesh::invalid_cell_id();
    RegionFace native;
    std::optional<RegionFace> second_native;
    PLIC::GeometryPoint second_image{};          ///< Checked neighbor-to-first displacement for numeric FVM distances.
    PLIC::DisplacementIdentity<ArithmeticMode::Ordinary>
        second_image_identity;                   ///< Finite neighbor-to-first displacement certificate.
    std::vector<std::uint64_t> second_node_gids; ///< First loop order, in the second physical image.
    bool native_is_second = false;               ///< Primary native address represents the second incident image.
    bool cut_incidence = true; ///< Derived by the factory; a physical image override alone is not a cut.
    int boundary = MultiRegionMesh::invalid_boundary_id;
    /** Retained for cut incidence or a periodic first image with only the
     * second native provider resident. Ordinary uncut interiors remain native. */
    PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> override_geometry;
    std::vector<std::uint64_t> override_nodes;
};
struct SparseNativeNode
{
    std::uint64_t gid = 0;
    std::optional<std::pair<std::size_t, std::uint64_t>> native;
    PLIC::GeometryPoint introduced_coordinate{};
};
struct SparseNativeMeshInput
{
    std::vector<MultiRegionMesh::Region> fragments;
    std::vector<SparseNativeCell> cells;
    std::vector<SparseNativeFace> faces;
    std::vector<SparseNativeNode> nodes;
    /** Transient authority point proofs. Discarded after exact route validation;
     * procedural native node geometry is never expanded in retained storage. */
    std::map<std::uint64_t, PLIC::PointIdentity<ArithmeticMode::Ordinary>> node_certificates;
    /** Every native node used by local full cells has an authenticated route. */
    std::map<std::pair<std::size_t, std::uint64_t>, std::uint64_t> native_nodes;
    std::uint64_t global_cells = 0, global_faces = 0, global_nodes = 0;
    std::map<int, std::string> boundaries;
    std::vector<char> root_schema; ///< Compact original-root/kernel provenance, never active leaves.
    std::size_t byte_budget = 64 * 1024 * 1024;
};
/** Borrowed execution row. Rich geometry and images belong to the immutable
 * current incidence record; current algebraic loops are scalar output spans. */
struct SparseNativeFaceView
{
    std::uint64_t gid = 0, first = 0, second = MultiRegionMesh::invalid_cell_id();
    RegionFace native;
    std::optional<RegionFace> second_native;
    PLIC::GeometryPoint second_image{};
    PLIC::DisplacementIdentity<ArithmeticMode::Ordinary> second_image_identity;
    std::span<const std::uint64_t> second_node_gids;
    bool native_is_second = false, cut_incidence = true;
    int boundary = MultiRegionMesh::invalid_boundary_id;
    PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> override_geometry;
    std::span<const std::uint64_t> override_nodes;
};
struct NativeSparsePhysicalFace
{
    PLIC::DisplacementIdentity<ArithmeticMode::Ordinary> second_image;
    PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> override_geometry;
    bool cut_incidence = true;
};
struct NativeSparseFaceRoute
{
    std::uint64_t gid = 0, first = 0, second = MultiRegionMesh::invalid_cell_id();
    RegionFace native;
    std::optional<RegionFace> second_native;
    std::size_t loop_begin = 0, loop_size = 0, second_loop_begin = 0, second_loop_size = 0;
    const NativeSparsePhysicalFace* physical = nullptr;
    bool native_is_second = false;
    int boundary = MultiRegionMesh::invalid_boundary_id;
};
struct NativeSparseOrdinal
{
    std::size_t region = 0;
    std::uint64_t ordinal = 0;
    auto operator<=>(const NativeSparseOrdinal&) const = default;
};
struct NativeSparseAliasRoute
{
    NativeSparseOrdinal first;
    std::uint64_t second = 0;
    auto operator<=>(const NativeSparseAliasRoute&) const = default;
};
struct NativeSparseNodeRoute
{
    std::uint64_t gid = 0;
    std::optional<NativeSparseOrdinal> native;
    PLIC::GeometryPoint introduced_coordinate{};
};
struct NativeSparseBoundaryRoute
{
    int first = 0;
    const std::string* second = nullptr;
};
struct NativeSparseLogicalRoute
{
    RegionFace native;
    std::size_t begin = 0, size = 0;
};
struct NativeSparseSignRoute
{
    RegionFace native;
    std::uint64_t gid = 0;
    int sign = 1;
};
/** No key, label, shared control, point identity or physical loop is copied into
 * these current-numbering buffers. Their allocator owns only detached State. */
struct NativeSparseNumericIndex
{
    using Admission = geometry_detail::NativeConstructionAdmission;
    explicit NativeSparseNumericIndex(const Admission& a)
        : cells(a.allocator<SparseNativeCell>()), faces(a.allocator<NativeSparseFaceRoute>()),
          nodes(a.allocator<NativeSparseNodeRoute>()), native_cells(a.allocator<Alias>()),
          native_nodes(a.allocator<Alias>()), logical_faces(a.allocator<NativeSparseLogicalRoute>()),
          flux_signs(a.allocator<NativeSparseSignRoute>()), loops(a.allocator<std::uint64_t>()),
          boundaries(a.allocator<Boundary>())
    {
    }
    using Alias = NativeSparseAliasRoute;
    using Boundary = NativeSparseBoundaryRoute;
    geometry_detail::NativeConstructorVector<SparseNativeCell> cells;
    geometry_detail::NativeConstructorVector<NativeSparseFaceRoute> faces;
    geometry_detail::NativeConstructorVector<NativeSparseNodeRoute> nodes;
    geometry_detail::NativeConstructorVector<Alias> native_cells, native_nodes;
    geometry_detail::NativeConstructorVector<NativeSparseLogicalRoute> logical_faces;
    geometry_detail::NativeConstructorVector<NativeSparseSignRoute> flux_signs;
    geometry_detail::NativeConstructorVector<std::uint64_t> loops;
    geometry_detail::NativeConstructorVector<Boundary> boundaries;
    /** Authenticated current subset only; never a source controller or global union. */
    std::shared_ptr<const void> physical_owners;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation;
    std::size_t storage_bytes() const noexcept;
};
static_assert(std::is_trivially_copyable_v<SparseNativeCell>);
static_assert(std::is_trivially_copyable_v<NativeSparseFaceRoute>);
static_assert(std::is_trivially_copyable_v<NativeSparseNodeRoute>);
static_assert(std::is_trivially_copyable_v<NativeSparseAliasRoute>);
static_assert(std::is_trivially_copyable_v<NativeSparseLogicalRoute>);
static_assert(std::is_trivially_copyable_v<NativeSparseSignRoute>);
static_assert(std::is_trivially_copyable_v<NativeSparseBoundaryRoute>);
struct NativeSparseProof
{
};
struct FrozenSourceCopyProof {std::shared_ptr<const void> accepted_source;};
/** Private implementation detail exposed here for compiled sparse dispatch.
 * Maps hold routes, never expanded procedural face geometry or cell-face CSR. */
struct SparseNativeIndex
{
    std::variant<NativeSparseProof, FrozenSourceCopyProof> provenance;
    std::map<std::uint64_t, SparseNativeCell> cells;
    std::map<std::uint64_t, SparseNativeFace> faces;
    std::map<std::uint64_t, SparseNativeNode> nodes;
    std::map<std::pair<std::size_t, std::uint64_t>, std::uint64_t> native_cells, native_nodes;
    std::map<RegionFace, std::vector<std::uint64_t>> logical_faces;
    std::map<int, std::string> boundaries;
    std::map<std::pair<RegionFace, std::uint64_t>, int> flux_signs;
    std::string common_signature;
    std::shared_ptr<const geometry_detail::NativeSchemaOwner> schema_owner;
    [[nodiscard]] std::string configuration_signature(std::uint64_t, std::uint64_t, std::uint64_t) const;
    std::shared_ptr<const NativeSparseNumericIndex> numeric;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation;
    [[nodiscard]] const SparseNativeCell& cell(std::uint64_t) const;
    [[nodiscard]] SparseNativeFaceView face(std::uint64_t) const;
    [[nodiscard]] SparseNativeNode node(std::uint64_t) const;
    [[nodiscard]] std::uint64_t native_cell(std::pair<std::size_t, std::uint64_t>) const;
    [[nodiscard]] std::uint64_t native_node(std::pair<std::size_t, std::uint64_t>) const;
    [[nodiscard]] std::span<const std::uint64_t> logical(RegionFace) const;
    [[nodiscard]] int sign(RegionFace, std::uint64_t) const;
    [[nodiscard]] const std::string& boundary(int) const;
    [[nodiscard]] std::vector<int> boundary_ids() const;
    [[nodiscard]] std::size_t boundary_count() const noexcept;
    [[nodiscard]] std::size_t face_count() const noexcept;
    [[nodiscard]] std::uint64_t face_gid(std::size_t) const;
    std::size_t storage_bytes() const;
};
} // namespace SimpleFluid::Meshes
