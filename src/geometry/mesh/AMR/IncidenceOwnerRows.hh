/** @file IncidenceOwnerRows.hh
 * @brief Candidate-local physical tile contribution accounting.
 */
#pragma once
#include "CartesianIncidenceSource.hh"
#include "ContactPayloadCensus.hh"
#include "DistributedAdaptiveIncidence.hh"
#include "PooledCellPoints.hh"
#include "PooledTile.hh"
#include "TileBodyPages.hh"
#include <memory>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Accepted count snapshots and cell-index pages are immutable. Only touched
 * cells acquire controllers bound to this candidate's pools. The arena and
 * pools must outlive the controller; keeper-backed snapshots retain their
 * arenas independently. Borrowed-arena forks copy their counts instead, and
 * returned row handles must be destroyed before their borrowed arena.
 * Begin a delta before changing contributions and finish after both old and new
 * contributions are complete. Slots remain exact pool handles, not wire keys. */
class IncidenceOwnerRows
{
public:
    using Slot = std::size_t;
    struct Patch
    {
        IncidenceRowHandle row;
        const AdaptiveLeaf* leaf;
    };
    struct PersistenceWork
    {
        std::size_t copied_cells = 0, copied_pages = 0, copied_bytes = 0, legacy_fork_visits = 0;
    };
    IncidenceOwnerRows(Budget&, IdentityBudget&, IncidencePools&, const AdaptiveProviderKernel&, int rank,
        std::shared_ptr<Budget> keeper = {});
    IncidenceOwnerRows(
        const IncidenceOwnerRows&, Budget&, IdentityBudget&, IncidencePools&, std::shared_ptr<Budget> keeper = {});
    ~IncidenceOwnerRows();
    IncidenceOwnerRows(const IncidenceOwnerRows&) = delete;
    IncidenceOwnerRows& operator=(const IncidenceOwnerRows&) = delete;
    void insert(Slot cell, std::uint64_t generation = 0);
    void erase(Slot cell);
    void begin_delta();
    void add(const TileBodyScalars&, TileBodyLoopView loop);
    void remove(const TileBodyScalars&, TileBodyLoopView loop);
    void add(const PooledTile& tile, std::span<const Slot> loop)
    {
        add(pooled_tile_scalars(tile), TileBodyLoopView(loop));
    }
    void remove(const PooledTile& tile, std::span<const Slot> loop)
    {
        remove(pooled_tile_scalars(tile), TileBodyLoopView(loop));
    }
    /** Returned vector capacity is charged to the caller; release that charge
     * after destroying its allocation. Row handles release their own charge. */
    [[nodiscard]] std::vector<Patch> finish(const OwnedLeafLookup&, bool keep_identity = true);
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
    /** Source-only mutable owner; shared roots/counts import their own union. */
    [[nodiscard]] std::size_t source_overlap_metadata_bytes() const noexcept;
    [[nodiscard]] std::size_t inherited_overlap_metadata_bytes() const noexcept;
    void finish_inherited_overlap() noexcept;
    [[nodiscard]] PersistenceWork persistence_work() const noexcept;
    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor, bool include_object = false) const
    {
        visit_payload(PayloadVisit(visitor), include_object);
    }

private:
    using PayloadVisit = CartesianBorrowedCallback<const ContactPayloadAllocation&>;
    struct Storage;
    void visit_payload(const PayloadVisit&, bool) const;
    std::shared_ptr<Budget> keeper_;
    Budget* budget_;
    IncidencePools* pools_;
    const AdaptiveProviderKernel* kernel_;
    int rank_;
    bool active_ = true;
    std::unique_ptr<Storage> storage_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
