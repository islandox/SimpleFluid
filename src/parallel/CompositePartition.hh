/** @file CompositePartition.hh
 *  @brief Topology-preserving composite partition plans and local results.
 */
#pragma once

#include "dataclass/TpetraTypes.hh"
#include "geometry/mesh/MultiRegionMesh.hh"
#include "parallel/MPI_interface.hh"
#include <span>
#include <variant>

namespace SimpleFluid
{
template<TpetraTypePack Pack> class MeshPartitioner;

/** @brief Partition policy changes ownership, never a region's topology family. */
struct CompositePartitionOptions
{
    enum class Policy { TopologyAware, CellGraph, ContiguousCanonical };
    Policy policy = Policy::TopologyAware;
    size_t ghost_layers = 1;
    /** Maximum cells in each indivisible structured/product graph unit. */
    size_t target_cells_per_fragment = 64;
    double imbalance_tolerance = 1.1;
    /** Reject an infeasible bound or a result above the requested tolerance. */
    bool require_balance = false;
};

struct CompositeBalanceReport
{
    uint64_t total_cells = 0, max_owned_cells = 0, largest_unit = 0;
    double measured_imbalance = 1, minimum_imbalance_bound = 1;
    bool tolerance_satisfied = true;
};

/** @brief A complete source is held by one rank; other ranks may pass nullptr. */
class CompositeMeshSource
{
public:
    explicit CompositeMeshSource(std::shared_ptr<const Meshes::MultiRegionMesh> mesh,
                             int source_rank = 0)
        : d_mesh(std::move(mesh)), d_source_rank(source_rank) {}
    const std::shared_ptr<const Meshes::MultiRegionMesh>& mesh() const noexcept { return d_mesh; }
    int source_rank() const noexcept { return d_source_rank; }
private:
    std::shared_ptr<const Meshes::MultiRegionMesh> d_mesh;
    int d_source_rank;
};
using CompositeSource = CompositeMeshSource;

/** @brief An owned native region selection. IDs are independent of rank and field order. */
struct CompositeRegionFragment
{
    struct OrthogonalBox
    {
        std::array<size_t, 3> begin{}, extent{};
    };
    struct ExtrudedProduct
    {
        std::vector<uint64_t> base_cells;
        size_t layer_begin = 0, layer_count = 0;
    };
    /** Native ordinal selection; also used for implicit regions by CellGraph. */
    struct ExplicitCells { std::vector<uint64_t> cells; };
    size_t region = 0;
    Meshes::RegionLayout::Family family = Meshes::RegionLayout::Family::Explicit;
    std::variant<OrthogonalBox, ExtrudedProduct, ExplicitCells> selection{OrthogonalBox{}};
};

/**
 * @brief Immutable rank-local ownership and halo plan.
 *
 * Canonical IDs remain those of the source catalog. Owned cells precede ghost
 * cells in cell_ids(); cell_owner_ranks() uses this local order. A plan pins its
 * source geometry revision and communicator and cannot be applied to a different
 * source. No global cell-to-rank array is retained on the destination ranks.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class CompositePartitionPlan
{
public:
    using ID = uint64_t;
    using comm_type = typename Pack::comm_type;
    const std::vector<ID>& cell_ids() const noexcept { return d_cells; }
    std::span<const ID> owned_cells() const noexcept { return {d_cells.data(), d_owned_cells}; }
    std::span<const ID> ghost_cells() const noexcept { return std::span<const ID>(d_cells).subspan(d_owned_cells); }
    const std::vector<int>& cell_owner_ranks() const noexcept { return d_cell_owners; }
    const std::vector<ID>& face_ids() const noexcept { return d_faces; }
    std::span<const ID> owned_faces() const noexcept { return {d_faces.data(), d_owned_faces}; }
    std::span<const ID> overlap_faces() const noexcept { return std::span<const ID>(d_faces).subspan(d_owned_faces); }
    const std::vector<ID>& node_ids() const noexcept { return d_nodes; }
    const std::vector<CompositeRegionFragment>& fragments() const noexcept { return d_fragments; }
    const Teuchos::RCP<const comm_type>& communicator() const noexcept { return d_comm; }
    const CompositePartitionOptions& options() const noexcept { return d_options; }
    const CompositeBalanceReport& balance() const noexcept { return d_balance; }
    ID global_cells() const noexcept { return d_global_cells; }
    ID global_faces() const noexcept { return d_global_faces; }
    ID global_nodes() const noexcept { return d_global_nodes; }
    int source_rank() const noexcept { return d_source_rank; }
private:
    friend class MeshPartitioner<Pack>;
    CompositePartitionPlan() = default;
    std::vector<ID> d_cells, d_faces, d_nodes;
    std::vector<int> d_cell_owners;
    std::vector<CompositeRegionFragment> d_fragments;
    size_t d_owned_cells = 0, d_owned_faces = 0;
    ID d_global_cells = 0, d_global_faces = 0, d_global_nodes = 0;
    int d_source_rank = 0;
    CompositePartitionOptions d_options;
    CompositeBalanceReport d_balance;
    Teuchos::RCP<const comm_type> d_comm;
    // The source rank retains immutable per-destination plans for packet creation.
    std::shared_ptr<const std::vector<std::vector<ID>>> d_destination_cells;
    std::weak_ptr<const Meshes::MultiRegionMesh> d_source;
    uint64_t d_source_epoch = 0;
};

/** @brief Rank-local native geometry plus a completed ownership plan. */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class CompositePartition
{
public:
    const std::shared_ptr<Meshes::MultiRegionMesh>& geometry() const noexcept { return d_geometry; }
    const CompositePartitionPlan<Pack>& plan() const noexcept { return d_plan; }
private:
    friend class MeshPartitioner<Pack>;
    CompositePartition(std::shared_ptr<Meshes::MultiRegionMesh> geometry,
                       CompositePartitionPlan<Pack> plan)
        : d_geometry(std::move(geometry)), d_plan(std::move(plan)) {}
    std::shared_ptr<Meshes::MultiRegionMesh> d_geometry;
    CompositePartitionPlan<Pack> d_plan;
};
} // namespace SimpleFluid
