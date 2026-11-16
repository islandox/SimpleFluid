#include "IncidenceOwnerRows.hh"
#include "PersistentOrderedPages.hh"
#include "TilePointContributions.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIdentityBudget.hh"
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct IncidenceOwnerRows::Storage
{
    using Snapshot = PooledCellPoints::SnapshotPtr;
    struct Row
    {
        Slot cell = 0;
        Snapshot points;
        std::uint64_t generation = 0;
    };
    struct Less
    {
        bool operator()(const Row& a, const Row& b) const noexcept { return a.cell < b.cell; }
    };
    // Each immutable count snapshot belongs to exactly one cell in a root.
    struct Weights
    {
        static constexpr bool enabled = true;
        struct Summary
        {
            std::size_t bytes = 0, same_arena_bytes = 0;
        };
        static void add_row(Summary& total, const Row& row, const Budget* arena)
        {
            if (!row.points)
                return;
            const auto bytes = row.points->metadata_bytes();
            total.bytes = persistent_ordered_detail::checked_sum(total.bytes, bytes);
            if (row.points->origin() == arena)
                total.same_arena_bytes = persistent_ordered_detail::checked_sum(total.same_arena_bytes, bytes);
        }
        static void add_child(Summary& total, const Summary& child, bool same_arena)
        {
            total.bytes = persistent_ordered_detail::checked_sum(total.bytes, child.bytes);
            if (same_arena)
                total.same_arena_bytes =
                    persistent_ordered_detail::checked_sum(total.same_arena_bytes, child.same_arena_bytes);
        }
    };
    using Tree =
        persistent_ordered_detail::OrderedPages<Row, Less, false, persistent_ordered_detail::OrderedPageWork, Weights>;
    using Root = Tree::Ptr;
    struct Working
    {
        std::unique_ptr<PooledCellPoints> points;
        // The current index pins this snapshot until this cell is published or erased.
        const PooledCellPoints::Snapshot* original;
        std::uint64_t generation = 0;
    };
    using WorkingMap = std::map<Slot, Working>;
    static constexpr auto working_bytes = sizeof(WorkingMap::value_type) + 4 * sizeof(void*) + sizeof(PooledCellPoints);

    ContactGroupAllocation header, imported, working_charge;
    Root root;
    WorkingMap dirty;
    IncidencePools* pools;
    bool has_keeper, keep_overlap = false;
    PersistenceWork work{};

    Storage(ContactGroupAllocation charge, bool kept, IncidencePools& values)
        : header(std::move(charge)), imported(header.keeper()), working_charge(header.keeper()), pools(&values),
          has_keeper(kept)
    {
    }
    [[nodiscard]] Budget& arena() const noexcept { return header.arena(); }
    [[nodiscard]] std::shared_ptr<Budget> keeper() const noexcept
    {
        return has_keeper ? header.keeper() : std::shared_ptr<Budget>{};
    }
    [[nodiscard]] const Row* find(Slot cell) const noexcept
    {
        const auto* row = Tree::lower_bound(root, [&](const Row& value) { return value.cell < cell; });
        return row && row->cell == cell ? row : nullptr;
    }
    [[nodiscard]] const Row& at(Slot cell) const
    {
        if (const auto* row = find(cell))
            return *row;
        throw std::invalid_argument("Unknown incidence owner cell.");
    }
    [[nodiscard]] static std::size_t total(const Root& value) noexcept
    {
        return value ? value->subtree_bytes + value->weights.bytes : 0;
    }
    [[nodiscard]] std::size_t inherited(const Root& value) const noexcept
    {
        if (!value)
            return 0;
        const auto bytes = total(value);
        return value->charge.arena.get() == &arena() ? bytes - value->same_arena_bytes - value->weights.same_arena_bytes
                                                     : bytes;
    }
    [[nodiscard]] std::size_t overlap_bytes() const noexcept { return imported.bytes() - inherited(root); }
    void finish_overlap() noexcept
    {
        imported.release(overlap_bytes());
        keep_overlap = false;
    }
    void publish(Root next, const persistent_ordered_detail::OrderedPageWork& changed)
    {
        const auto foreign = inherited(next);
        if (foreign > imported.bytes())
            throw std::logic_error("Owner-row edit introduced an unadmitted inherited body.");
        root = std::move(next);
        if (!keep_overlap)
            imported.release(imported.bytes() - foreign);
        work.copied_pages += changed.copied_pages;
        work.copied_bytes += changed.copied_bytes;
    }
    void set(Row replacement, bool expected)
    {
        persistent_ordered_detail::OrderedPageWork changed;
        auto next = Tree(header.keeper(), &changed).edit(root, replacement, &replacement, expected);
        publish(std::move(next), changed);
    }
    void erase(Slot cell)
    {
        const Row key{cell, {}, 0};
        persistent_ordered_detail::OrderedPageWork changed;
        auto next = Tree(header.keeper(), &changed).edit(root, key, nullptr, true);
        publish(std::move(next), changed);
    }
    void fork_from(const Storage& source)
    {
        if (root || !dirty.empty() || !source.dirty.empty())
            throw std::invalid_argument("Owner-row fork requires completed source counts and empty destination.");
        const bool distinct = &arena() != &source.arena();
        if (distinct)
        {
            imported.add(total(source.root));
            keep_overlap = true;
        }
        if (has_keeper && source.has_keeper && distinct)
        {
            root = source.root;
            return;
        }
        // Compatibility for borrowed arenas: source headers and counts cannot
        // escape the source budget's scope. Only this path copies all rows.
        (void) Tree::visit_from(
            source.root, [](const Row&) { return false; },
            [&](const Row& row)
            {
                if (!row.points)
                    throw std::invalid_argument("Cannot fork an unfinished cell count row.");
                PooledCellPoints copy(*row.points, arena(), *pools);
                auto snapshot = copy.freeze(keeper());
                set(Row{row.cell, std::move(snapshot), row.generation}, false);
                ++work.legacy_fork_visits;
                return true;
            });
    }
    void stage(Slot cell, const Row* original, std::uint64_t generation)
    {
        ContactGroupAllocation staged(header.keeper());
        staged.add(working_bytes);
        auto points = original ? std::make_unique<PooledCellPoints>(*original->points, arena(), *pools)
                               : std::make_unique<PooledCellPoints>(arena(), *pools);
        points->begin_delta();
        dirty.emplace(cell, Working{std::move(points), original ? original->points.get() : nullptr, generation});
        working_charge.transfer_from(staged, working_bytes);
        if (original)
            ++work.copied_cells;
    }
    void insert(Slot cell, std::uint64_t generation)
    {
        if (dirty.contains(cell) || find(cell))
            throw std::invalid_argument("Duplicate incidence owner cell.");
        stage(cell, nullptr, generation);
    }
    void remove_cell(Slot cell)
    {
        const auto current = dirty.find(cell);
        if (current == dirty.end() || find(cell))
            erase(cell);
        if (current != dirty.end())
        {
            dirty.erase(current);
            working_charge.release(working_bytes);
        }
    }
    PooledCellPoints& touch(Slot cell)
    {
        auto current = dirty.find(cell);
        if (current == dirty.end())
        {
            const auto& row = at(cell);
            stage(cell, &row, row.generation);
            current = dirty.find(cell);
        }
        return *current->second.points;
    }
    void publish_counts(Slot cell, Working& current)
    {
        const bool unchanged = current.original && current.points->matches(*current.original);
        current.points->finish_delta();
        if (unchanged)
            return; // An exact no-op finish neither allocates nor looks up geometry.
        auto snapshot = current.points->freeze(keeper());
        const bool existing = current.original != nullptr;
        set(Row{cell, std::move(snapshot), current.generation}, existing);
        current.original = nullptr;
    }
    void clear_work() noexcept
    {
        const auto bytes = dirty.size() * working_bytes;
        dirty.clear();
        working_charge.release(bytes);
    }
    [[nodiscard]] std::size_t metadata_bytes() const noexcept
    {
        auto bytes = header.bytes() + total(root) + overlap_bytes() + working_charge.bytes();
        for (const auto& [cell, current] : dirty)
            bytes += current.points->metadata_bytes();
        return bytes;
    }
    void visit_payload(const PayloadVisit& visitor) const
    {
        contact_payload_object(visitor, this, GeometryContactPayloadRole::OwnerRowsObject, sizeof(*this));
        Tree::visit_pages(root,
            [&](const Root& page)
            {
                contact_payload_object(
                    visitor, page.get(), GeometryContactPayloadRole::OwnerRowsCellNodes, sizeof(Tree::Node), true);
                contact_payload_buffer(visitor, page.get(), GeometryContactPayloadRole::OwnerRowsCellNodes, page->rows);
                contact_payload_buffer(
                    visitor, page.get(), GeometryContactPayloadRole::OwnerRowsCellNodes, page->children);
                for (const auto& row : page->rows)
                    if (row.points)
                        row.points->visit_borrowed_payload(visitor);
            });
        for (const auto& value : dirty)
        {
            contact_payload_node_value(visitor, this, GeometryContactPayloadRole::OwnerRowsDirtyNodes, value);
            value.second.points->visit_borrowed_payload(visitor, true);
        }
    }
};

IncidenceOwnerRows::IncidenceOwnerRows(Budget& b, IdentityBudget&, IncidencePools& p, const AdaptiveProviderKernel& k,
    int rank, std::shared_ptr<Budget> keeper)
    : keeper_(std::move(keeper)), budget_(&b), pools_(&p), kernel_(&k), rank_(rank)
{
    if (keeper_ && keeper_.get() != &b)
        throw std::invalid_argument("Incidence owner budget keeper mismatch.");
    if (rank < 0)
        throw std::invalid_argument("Invalid incidence owner rank.");
    ContactGroupAllocation charge(keeper_ ? keeper_ : std::shared_ptr<Budget>(std::shared_ptr<Budget>{}, &b));
    charge.add(sizeof(Storage));
    storage_ = std::make_unique<Storage>(std::move(charge), bool(keeper_), p);
}
IncidenceOwnerRows::IncidenceOwnerRows(
    const IncidenceOwnerRows& other, Budget& b, IdentityBudget& i, IncidencePools& p, std::shared_ptr<Budget> keeper)
    : IncidenceOwnerRows(b, i, p, *other.kernel_, other.rank_, std::move(keeper))
{
    storage_->fork_from(*other.storage_);
}
IncidenceOwnerRows::~IncidenceOwnerRows() = default;
void IncidenceOwnerRows::insert(Slot cell, std::uint64_t generation)
{
    if (!active_)
        throw std::invalid_argument("Incidence delta is not active.");
    (void) pools_->cells.get(cell);
    storage_->insert(cell, generation);
}
void IncidenceOwnerRows::erase(Slot cell)
{
    if (!active_)
        throw std::invalid_argument("Incidence delta is not active.");
    storage_->remove_cell(cell);
}
void IncidenceOwnerRows::begin_delta()
{
    if (active_ && !storage_->dirty.empty())
        throw std::invalid_argument("Unfinished incidence delta.");
    active_ = true;
}
void IncidenceOwnerRows::add(const TileBodyScalars& tile, TileBodyLoopView loop)
{
    visit_tile_physical_points(tile, loop, rank_, *pools_,
        [&](Slot cell, Slot point)
        {
            if (!active_)
                throw std::invalid_argument("Incidence delta is not active.");
            storage_->touch(cell).add(point);
        });
}
void IncidenceOwnerRows::remove(const TileBodyScalars& tile, TileBodyLoopView loop)
{
    visit_tile_physical_points(tile, loop, rank_, *pools_,
        [&](Slot cell, Slot point)
        {
            if (!active_)
                throw std::invalid_argument("Incidence delta is not active.");
            storage_->touch(cell).remove(point);
        });
}
std::vector<IncidenceOwnerRows::Patch> IncidenceOwnerRows::finish(const OwnedLeafLookup& lookup, bool keep_identity)
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
    for (auto& [slot, entry] : storage_->dirty)
    {
        if (!entry.points->changed())
        {
            storage_->publish_counts(slot, entry);
            continue;
        }
        if (entry.generation == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Incidence generation exhausted.");
        const auto& leaf = lookup(pools_->cells.get(slot));
        if (leaf.key != pools_->cells.get(slot))
            throw std::invalid_argument("Incorrect incidence owner lookup.");
        if (keeper_ && !batch)
            batch.emplace(pools_->points);
        auto row = batch ? entry.points->row(leaf, *kernel_, *batch, keep_identity) : entry.points->row(leaf, *kernel_);
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
        result.push_back({std::move(owned), &leaf});
        ++entry.generation;
        storage_->publish_counts(slot, entry);
    }
    if (batch)
        batch->freeze();
    storage_->clear_work();
    active_ = false;
    result_credit.transferred = true;
    return result;
}

std::size_t IncidenceOwnerRows::metadata_bytes() const noexcept
{
    return storage_->metadata_bytes();
}
std::size_t IncidenceOwnerRows::source_overlap_metadata_bytes() const noexcept
{
    // Fork imports the shared tree and snapshots, but owns a distinct controller.
    return storage_->header.bytes();
}
std::size_t IncidenceOwnerRows::inherited_overlap_metadata_bytes() const noexcept
{
    return storage_->overlap_bytes();
}
void IncidenceOwnerRows::finish_inherited_overlap() noexcept
{
    storage_->finish_overlap();
}
IncidenceOwnerRows::PersistenceWork IncidenceOwnerRows::persistence_work() const noexcept
{
    return storage_->work;
}
void IncidenceOwnerRows::visit_payload(const PayloadVisit& visitor, bool include_object) const
{
    if (include_object)
        contact_payload_object(visitor, this, GeometryContactPayloadRole::OwnerRowsObject, sizeof(*this));
    storage_->visit_payload(visitor);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
