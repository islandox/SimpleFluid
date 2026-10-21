/** @file SparseNativeMesh.hh
 * @brief Owner-local native dispatch routes with procedural interior geometry.
 */
#pragma once
#include "geometry/ConvexCellGeometry.hh"
#include "geometry/mesh/MultiRegionMesh.hh"
#include <map>
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
    GeometryPoint centroid{};
};
struct SparseNativeFace
{
    std::uint64_t gid = 0, first = 0, second = MultiRegionMesh::invalid_cell_id();
    RegionFace native;
    std::optional<RegionFace> second_native;
    GeometryPoint second_image{};                ///< Checked neighbor-to-first displacement for numeric FVM distances.
    std::vector<char> exact_second_image;        ///< Exact neighbor-to-first displacement certificate.
    std::vector<std::uint64_t> second_node_gids; ///< First loop order, in the second physical image.
    bool native_is_second = false;               ///< Primary native address represents the second incident image.
    bool cut_incidence = true; ///< Derived by the factory; a physical image override alone is not a cut.
    int boundary = MultiRegionMesh::invalid_boundary_id;
    /** Retained for cut incidence or a periodic first image with only the
     * second native provider resident. Ordinary uncut interiors remain native. */
    ExactPlanarFacePtr override_geometry;
    std::vector<std::uint64_t> override_nodes;
};
struct SparseNativeNode
{
    std::uint64_t gid = 0;
    std::optional<std::pair<std::size_t, std::uint64_t>> native;
    GeometryPoint introduced_coordinate{};
};
struct SparseNativeMeshInput
{
    std::vector<MultiRegionMesh::Region> fragments;
    std::vector<SparseNativeCell> cells;
    std::vector<SparseNativeFace> faces;
    std::vector<SparseNativeNode> nodes;
    /** Transient authority point proofs. Discarded after exact route validation;
     * procedural native node geometry is never expanded in retained storage. */
    std::map<std::uint64_t, std::vector<char>> node_certificates;
    /** Every native node used by local full cells has an authenticated route. */
    std::map<std::pair<std::size_t, std::uint64_t>, std::uint64_t> native_nodes;
    std::uint64_t global_cells = 0, global_faces = 0, global_nodes = 0;
    std::map<int, std::string> boundaries;
    std::vector<char> root_schema; ///< Compact original-root/kernel provenance, never active leaves.
    std::size_t byte_budget = 64 * 1024 * 1024;
};
struct ExactSparseNativeProof {};
struct FrozenSourceCopyProof {std::shared_ptr<const void> accepted_source;};
/** Private implementation detail exposed here for compiled sparse dispatch.
 * Maps hold routes, never expanded procedural face geometry or cell-face CSR. */
struct SparseNativeIndex
{
    std::variant<ExactSparseNativeProof,FrozenSourceCopyProof> provenance;
    std::map<std::uint64_t, SparseNativeCell> cells;
    std::map<std::uint64_t, SparseNativeFace> faces;
    std::map<std::uint64_t, SparseNativeNode> nodes;
    std::map<std::pair<std::size_t, std::uint64_t>, std::uint64_t> native_cells, native_nodes;
    std::map<RegionFace, std::vector<std::uint64_t>> logical_faces;
    std::map<int, std::string> boundaries;
    std::map<std::pair<RegionFace, std::uint64_t>, int> flux_signs;
    std::string common_signature;
    std::size_t storage_bytes() const;
};
} // namespace SimpleFluid::Meshes
