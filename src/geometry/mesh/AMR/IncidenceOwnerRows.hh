/** @file IncidenceOwnerRows.hh
 * @brief Candidate-local physical tile contribution accounting.
 */
#pragma once
#include "DistributedAdaptiveIncidence.hh"
#include "PooledCellPoints.hh"
#include "PooledTile.hh"
#include <map>
#include <memory>
#include <set>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** All operations mutate only this candidate. The arena and pools must outlive
 * it. Without an arena keeper, it must also outlive every returned row.
 * With a keeper, row handles retain the arena independently. Start a delta before retiring or adding tiles;
 * finish only after all old contributions have been removed and all new ones
 * added. Slots are local typed pool handles, never wire or ordering keys. */
class IncidenceOwnerRows
{
public:
    using Slot = std::size_t;
    struct Patch
    {
        Slot cell;
        IncidenceRowHandle row;
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
    void add(const PooledTile&, std::span<const Slot> loop);
    void remove(const PooledTile&, std::span<const Slot> loop);
    /** Returned vector capacity is charged to the caller; release that charge
     * after destroying its allocation. Row handles release their own charge. */
    [[nodiscard]] std::vector<Patch> finish(const OwnedLeafLookup&);
    [[nodiscard]] std::size_t metadata_bytes() const noexcept
    {
        std::size_t bytes = charge_;
        for (const auto& [slot, entry] : cells_)
            bytes += entry.points->metadata_bytes();
        return bytes;
    }

private:
    struct Entry
    {
        std::unique_ptr<PooledCellPoints> points;
        std::uint64_t generation = 0;
    };
    using CellIndex = std::map<Slot, Entry>;
    using DirtySet = std::set<Slot>;
    static constexpr std::size_t cell_charge =
        sizeof(CellIndex::value_type) + 4 * sizeof(void*) + sizeof(PooledCellPoints);
    static constexpr std::size_t dirty_charge = sizeof(DirtySet::value_type) + 4 * sizeof(void*);
    void touch(Slot);
    std::shared_ptr<Budget> keeper_;
    Budget* budget_;
    IdentityBudget* identities_;
    IncidencePools* pools_;
    const AdaptiveProviderKernel* kernel_;
    int rank_;
    bool active_ = true;
    CellIndex cells_;
    DirtySet dirty_;
    std::size_t charge_ = 0;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
