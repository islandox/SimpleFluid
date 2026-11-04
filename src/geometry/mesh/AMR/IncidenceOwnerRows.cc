#include "IncidenceOwnerRows.hh"
#include "TilePointContributions.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIdentityBudget.hh"
#include <limits>
#include <optional>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
IncidenceOwnerRows::IncidenceOwnerRows(Budget& b, IdentityBudget& i, IncidencePools& p, const AdaptiveProviderKernel& k,
    int rank, std::shared_ptr<Budget> keeper)
    : keeper_(std::move(keeper)), budget_(&b), identities_(&i), pools_(&p), kernel_(&k), rank_(rank)
{
    if (keeper_ && keeper_.get() != &b)
        throw std::invalid_argument("Incidence owner budget keeper mismatch.");
    if (rank < 0)
        throw std::invalid_argument("Invalid incidence owner rank.");
}
IncidenceOwnerRows::IncidenceOwnerRows(
    const IncidenceOwnerRows& other, Budget& b, IdentityBudget& i, IncidencePools& p, std::shared_ptr<Budget> keeper)
    : IncidenceOwnerRows(b, i, p, *other.kernel_, other.rank_, std::move(keeper))
{
    try
    {
        for (const auto& [slot, entry] : other.cells_)
        {
            budget_->add(cell_charge);
            charge_ += cell_charge;
            cells_.emplace(slot, Entry{std::make_unique<PooledCellPoints>(*entry.points, b, p), entry.generation});
        }
    }
    catch (...)
    {
        cells_.clear();
        budget_->used -= charge_;
        charge_ = 0;
        throw;
    }
}
IncidenceOwnerRows::~IncidenceOwnerRows()
{
    cells_.clear();
    dirty_.clear();
    budget_->used -= charge_;
}
void IncidenceOwnerRows::insert(Slot cell, std::uint64_t generation)
{
    if (!active_)
        throw std::invalid_argument("Incidence delta is not active.");
    (void) pools_->cells.get(cell);
    if (cells_.contains(cell))
        throw std::invalid_argument("Duplicate incidence owner cell.");
    budget_->add(cell_charge);
    charge_ += cell_charge;
    cells_.emplace(cell, Entry{std::make_unique<PooledCellPoints>(*budget_, *pools_), generation});
    touch(cell);
}
void IncidenceOwnerRows::erase(Slot cell)
{
    if (!active_)
        throw std::invalid_argument("Incidence delta is not active.");
    if (dirty_.erase(cell))
    {
        budget_->used -= dirty_charge;
        charge_ -= dirty_charge;
    }
    if (cells_.erase(cell) != 1)
        throw std::invalid_argument("Unknown incidence owner cell.");
    charge_ -= cell_charge;
    budget_->used -= cell_charge;
}
void IncidenceOwnerRows::begin_delta()
{
    if (active_ && !dirty_.empty())
        throw std::invalid_argument("Unfinished incidence delta.");
    active_ = true;
    const auto bytes = dirty_.size() * dirty_charge;
    dirty_.clear();
    budget_->used -= bytes;
    charge_ -= bytes;
}
void IncidenceOwnerRows::touch(Slot cell)
{
    if (!active_)
        throw std::invalid_argument("Incidence delta is not active.");
    auto& entry = cells_.at(cell);
    if (dirty_.contains(cell))
        return;
    budget_->add(dirty_charge);
    charge_ += dirty_charge;
    dirty_.insert(cell);
    entry.points->begin_delta();
}
void IncidenceOwnerRows::add(const PooledTile& tile, std::span<const Slot> loop)
{
    visit_tile_physical_points(tile, loop, rank_, *pools_,
        [&](Slot cell, Slot point)
        {
            touch(cell);
            cells_.at(cell).points->add(point);
        });
}
void IncidenceOwnerRows::remove(const PooledTile& tile, std::span<const Slot> loop)
{
    visit_tile_physical_points(tile, loop, rank_, *pools_,
        [&](Slot cell, Slot point)
        {
            touch(cell);
            cells_.at(cell).points->remove(point);
        });
}
std::vector<IncidenceOwnerRows::Patch> IncidenceOwnerRows::finish(const OwnedLeafLookup& lookup)
{
    if (!active_)
        throw std::invalid_argument("Incidence delta is not active.");
    std::vector<Patch> result;
    struct ResultCredit
    {
        Budget& budget;
        std::vector<Patch>& rows;
        std::size_t bytes = 0;
        bool transferred = false;
        ~ResultCredit()
        {
            if (!transferred)
            {
                std::vector<Patch>().swap(rows);
                budget.used -= bytes;
            }
        }
        void grow()
        {
            if (rows.size() != rows.capacity())
                return;
            const auto old = rows.capacity();
            if (old > budget.limit / sizeof(Patch) / 2)
                throw std::invalid_argument("Incidence row-patch capacity overflows.");
            const auto next = old ? old * 2 : 1;
            const auto allocation = next * sizeof(Patch);
            budget.add(allocation);
            try
            {
                rows.reserve(next);
            }
            catch (...)
            {
                budget.used -= allocation;
                throw;
            }
            budget.used -= bytes;
            bytes = allocation;
            if (rows.capacity() != next)
                throw std::invalid_argument("Incidence row-patch capacity differs from admission.");
        }
    } result_credit{*budget_, result};
    std::optional<ContactPointBatch> batch;
    for (const auto slot : dirty_)
    {
        auto& entry = cells_.at(slot);
        if (!entry.points->changed())
        {
            entry.points->finish_delta();
            continue;
        }
        if (entry.generation == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Incidence generation exhausted.");
        const auto& leaf = lookup(pools_->cells.get(slot));
        if (leaf.key != pools_->cells.get(slot))
            throw std::invalid_argument("Incorrect incidence owner lookup.");
        if (keeper_ && !batch)
            batch.emplace(pools_->points);
        auto row = batch ? entry.points->row(leaf, *kernel_, *batch) : entry.points->row(leaf, *kernel_);
        // Page credits and the frozen directory release their own admission.
        // Only the row-exclusive buffers transfer to the row's deleter.
        const auto row_body_charge = key_capacity(row.key) + row.domain.capacity() + row.points.storage_bytes();
        struct RowCredit
        {
            Budget& budget;
            ContactIncidenceRow& row;
            std::size_t bytes;
            bool transferred = false;
            ~RowCredit()
            {
                if (!transferred)
                {
                    row = {};
                    budget.used -= bytes;
                }
            }
        } row_credit{*budget_, row, row_body_charge};
        row.incidence_generation = entry.generation + 1;
        // Destroy the row body before returning its charge; the arena must
        // outlive every published handle. Identity-ledger pins stay charged.
        constexpr auto owner_charge = sizeof(ContactIncidenceRow) + 128;
        budget_->add(owner_charge);
        row_credit.bytes += owner_charge;
        auto* const budget = budget_;
        auto destroy = [budget, keeper = keeper_, charge = row_body_charge + owner_charge](
                           const ContactIncidenceRow* value)
        {
            (void) keeper;
            delete value;
            budget->used -= charge;
        };
        std::unique_ptr<const ContactIncidenceRow, decltype(destroy)> pending(
            new ContactIncidenceRow(std::move(row)), std::move(destroy));
        row_credit.transferred = true;
        IncidenceRowHandle owned(std::move(pending));
        result_credit.grow();
        result.push_back({slot, std::move(owned)});
        ++entry.generation;
        entry.points->finish_delta();
    }
    if (batch)
        batch->freeze();
    const auto dirty_bytes = dirty_.size() * dirty_charge;
    dirty_.clear();
    charge_ -= dirty_bytes;
    budget_->used -= dirty_bytes;
    active_ = false;
    result_credit.transferred = true;
    return result;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
