/**
 * @file CartesianAMR.hh
 * @brief Immutable dyadic Cartesian block hierarchies with native FVM interfaces.
 */
#pragma once

#include "geometry/mesh/MultiRegionMesh.hh"

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace SimpleFluid::Meshes
{

/** @brief Cartesian AMR domain; block cell counts are positive even integers. */
struct AMRGrid
{
    std::array<std::size_t, 3> root_blocks{1, 1, 1};
    std::array<std::size_t, 3> block_cells{4, 4, 4};
    std::array<double, 3> origin{0.0, 0.0, 0.0};
    std::array<double, 3> extent{1.0, 1.0, 1.0};
    std::array<bool, 3> periodic{false, false, false};
    unsigned max_level = 2;
};

/**
 * @brief Persistent spatial block key, independent of partition and leaf ordinal.
 *
 * At level L, index[a] ranges over root_blocks[a]*2^L blocks. Children have
 * indices 2*index plus binary offsets; the parent halves each index.
 */
struct AMRBlock
{
    unsigned level = 0;
    std::array<std::size_t, 3> index{};
    auto operator<=>(const AMRBlock&) const = default;
};

/** @brief Closed Cartesian coordinate box, in metres. */
struct AMRBox
{
    std::array<double, 3> lower{};
    std::array<double, 3> upper{};
};

/** @brief Positive-volume intersection with a globally indexed leaf cell. */
struct AMROverlap
{
    std::uint64_t cell = 0;
    AMRBox box;
};

/**
 * @brief Collectively constructed immutable balanced Cartesian leaf hierarchy.
 *
 * Leaves are ordered by AMRBlock. Cell GIDs are leaf ordinal followed by
 * x-fastest local cell ordinal. Only compact hierarchy/interface metadata is
 * replicated; MeshHandle on geometry() constructs native distributed maps.
 * Neighboring leaves, including edges/corners and periodic images, differ by
 * at most one level. Periodic face partners have equal levels.
 *
 * Construction and adapt() are collective on the default Tpetra communicator.
 * Invalid or rank-divergent descriptions reject collectively before publishing
 * a native mesh. The old hierarchy remains valid after successful adaptation.
 */
class CartesianAMR
{
public:
    explicit CartesianAMR(AMRGrid config);

    [[nodiscard]] const AMRGrid& config() const noexcept;
    [[nodiscard]] const std::vector<AMRBlock>& blocks() const noexcept;
    [[nodiscard]] const SP<const MultiRegionMesh>& geometry() const noexcept;
    [[nodiscard]] std::uint64_t topology_generation() const noexcept;
    [[nodiscard]] std::uint64_t partition_generation() const noexcept;

    [[nodiscard]] AMRBox block_box(std::size_t leaf) const;
    [[nodiscard]] std::array<double, 3> block_spacing(std::size_t leaf) const;
    [[nodiscard]] AMRBox cell_box(std::uint64_t cell) const;
    [[nodiscard]] std::size_t cell_block(std::uint64_t cell) const;

    /**
     * @brief Intersections for a finite domain-contained query; empty boxes yield no cells.
     * Boundary overshoot within 16 floating-point roundoff units at domain scale
     * is clamped to the exact domain boundary; larger excursions are rejected.
     */
    [[nodiscard]] std::vector<AMROverlap> overlaps(const AMRBox& query) const;

    /**
     * @brief Return a candidate hierarchy after requested changes and balance closure.
     * @param refine Leaf ordinals in this hierarchy, each below max_level.
     * @param coarsen Parent keys whose complete eight leaf children will merge.
     * @return Immutable candidate; unchanged requests share the native geometry.
     *
     * Conflicting or incomplete requests reject collectively. Closure refines
     * coarse neighbors as required and can undo coarsening that violates balance.
     * Topology and partition generations advance only when the leaf set changes.
     */
    [[nodiscard]] std::shared_ptr<const CartesianAMR> adapt(
        const std::vector<std::size_t>& refine, const std::vector<AMRBlock>& coarsen = {}) const;

private:
    struct LeafDirectory;
    CartesianAMR(AMRGrid config, std::vector<AMRBlock> blocks, std::uint64_t generation);
    [[nodiscard]] double coordinate(std::size_t axis, std::uint64_t finest_cell_index) const;
    [[nodiscard]] AMRBox key_box(const AMRBlock& block) const;
    void build_geometry();

    AMRGrid config_;
    std::vector<AMRBlock> blocks_;
    SP<const MultiRegionMesh> geometry_;
    std::shared_ptr<const LeafDirectory> leaf_directory_;
    std::uint64_t generation_ = 0;
};

} // namespace SimpleFluid::Meshes
