/**
 * @file MeshReorderingFactory.hh
 * @brief Stable cell-local reordering for compact selected subdomains.
 */

#pragma once

#include "SimpleFluidExport.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/mesh/SemiStructMeshTopo.hh"

#include <compare>
#include <cstddef>
#include <functional>
#include <span>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Build mesh handles whose selected cells occupy compact local prefixes.
 *
 * The factory preserves the parent mesh's communicator, ownership, face order,
 * geometry IDs, and Tpetra global IDs. Only the rank-local cell ordering is
 * changed: selected owned cells precede other owned cells, and selected ghost
 * cells precede other ghosts. The returned range certificate lets a subdomain
 * translate cell ordinals arithmetically without retaining dense forward and
 * reverse indexers.
 *
 * The factory consumes a uniquely owned mutable handle and reorders that
 * handle in place, so no second parent-sized MeshHandle or local/global
 * indexer is retained. If every rank is already selected-first, no mutation is
 * performed. Construct mesh-backed fields and caches only after reordering.
 *
 * @tparam Pack Tpetra scalar, ordinal, communicator, and map types.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes> class SIMPLEFLUID_PUBLIC_TYPE MeshReorderingFactory
{
public:
    using mesh_type = MeshHandle<Pack>;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using Vec3 = typename mesh_type::Vec3;

    /** @brief Pure local predicate evaluated for parent-owned cells. */
    using cell_selector_type = std::function<bool(global_ordinal_type, const Vec3&)>;

    /** @brief Half-open range of cell local ordinals. */
    struct LocalCellRange
    {
        size_t begin = 0;
        size_t end = 0;

        constexpr size_t size() const noexcept { return end - begin; }
        constexpr bool empty() const noexcept { return begin == end; }
        constexpr bool contains(size_t local_id) const noexcept { return local_id >= begin && local_id < end; }

        constexpr auto operator<=>(const LocalCellRange&) const = default;
    };

    /**
     * @brief Reordered mesh plus its selected owned and ghost prefix sizes.
     *
     * Selected owned cells occupy `[0, selected_owned_cells)`. Selected ghost
     * cells occupy `[mesh->num_owned_cells(), mesh->num_owned_cells() +
     * selected_ghost_cells)`.
     */
    struct SelectedCellLayout
    {
        SP<const mesh_type> mesh;
        size_t selected_owned_cells = 0;
        size_t selected_ghost_cells = 0;

        LocalCellRange selected_owned_range() const noexcept { return {0, selected_owned_cells}; }

        LocalCellRange selected_ghost_range() const noexcept
        {
            const auto begin = mesh ? mesh->num_owned_cells() : 0;
            return {begin, begin + selected_ghost_cells};
        }

        size_t selected_local_cells() const noexcept { return selected_owned_cells + selected_ghost_cells; }
    };

    /**
     * @brief Place selected cells first within the owned and ghost groups.
     *
     * Selection is evaluated only for authoritative owned cells and imported
     * to the overlap map. Callback failure is agreed collectively before any
     * rank proceeds to another collective operation.
     *
     * Native decomposed orthogonal and XY/Z handles preserve their source
     * geometry lookup when composing a local permutation. Existing algebraic GIDs stay attached to
     * physical cells, so owned maps may cease to be contiguous after this
     * explicit selection; initial partition-time contiguous numbering is not
     * repeated. Face and node identities/order remain unchanged.
     *
     * @param parent Source mesh handle. Ownership is consumed; when any rank
     *        needs a permutation, the handle must be uniquely owned on every
     *        rank so existing fields cannot silently retain the old ordering.
     * @param selector Pure local selection predicate receiving stable geometry
     *        ID and centroid. It must not perform communication.
     * @return Reordered mesh and selected-range certificate.
     * @throws std::invalid_argument If the parent or selector is null, if an
     *         overlap cell has no owner in the communicator, or if a required
     *         reordering does not have unique mesh-handle ownership.
     * @throws std::runtime_error On ranks where another rank's callback fails.
     */
    static SelectedCellLayout selected_cells_first(SP<mesh_type>&& parent, cell_selector_type selector);

    /**
     * @brief Group XY cells by partition, then order each induced graph by deterministic reverse Cuthill-McKee.
     *
     * This pure base-topology operation returns new-to-original XY ordinals.
     * Partition labels are ordered numerically; traversal seeds and neighbors
     * use induced degree then original ordinal as their deterministic tie break.
     * Disconnected components are reversed separately in their seed order.
     * Geometry, canonical global IDs and input topology are never modified.
     * Callers apply this permutation to local ordering before fields/caches exist.
     *
     * @param topology Immutable polygonal XY adjacency; Z repetitions are ignored.
     * @param xy_owners One partition label per original XY cell; labels may be sparse.
     * @return A bijection of [0, number of XY cells), grouped by partition label.
     * @throws std::invalid_argument If owner count differs from XY cell count or
     *         a label is the unknown-owner sentinel size_t::max().
     */
    static std::vector<std::size_t> partitioned_xy_order(
        const Meshes::SemiStructMeshTopo& topology, std::span<const std::size_t> xy_owners);
};

} // namespace SimpleFluid
