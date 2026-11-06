/** @file PersistentSlotValues.hh
 * @brief Present-or-absent integral values on immutable ordered metadata pages.
 */
#pragma once
#include "ContactGroupAllocation.hh"
#include "ContactPayloadCensus.hh"
#include "PersistentOrderedPages.hh"
#include <type_traits>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Zero is an ordinary stored value. A fork imports its shared subtree credit
 * once, and keeps replaced inherited pages admitted until finish_overlap().
 * Each candidate arena must be fresh with respect to its source lineage. */
template<class Value> class PersistentSlotValues
{
    static_assert(std::is_integral_v<Value> && std::is_unsigned_v<Value>);
    struct Row
    {
        std::size_t slot;
        Value value;
    };
    struct Less
    {
        bool operator()(const Row& a, const Row& b) const noexcept { return a.slot < b.slot; }
    };
    using Index = persistent_ordered_detail::OrderedPages<Row, Less>;
    using Root = typename Index::Ptr;

public:
    explicit PersistentSlotValues(std::shared_ptr<Budget> arena) : imported_(std::move(arena)), has_keeper_(true) {}
    PersistentSlotValues(Budget& budget, std::shared_ptr<Budget> keeper = {})
        : imported_(keeper ? keeper : std::shared_ptr<Budget>(std::shared_ptr<Budget>{}, &budget)),
          has_keeper_(bool(keeper))
    {
        if (imported_.keeper().get() != &budget)
            throw std::invalid_argument("Numeric metadata keeper does not own its budget.");
    }
    PersistentSlotValues(const PersistentSlotValues& source, std::shared_ptr<Budget> arena)
        : imported_(std::move(arena)), has_keeper_(true), keep_overlap_(true)
    {
        if (&imported_.arena() == &source.imported_.arena())
            throw std::invalid_argument("Numeric metadata fork requires an independent candidate arena.");
        imported_.add(source.metadata_bytes());
        if (source.has_keeper_)
            root_ = source.root_;
        else
            source.visit([&](std::size_t slot, Value value) { set(slot, value); });
    }
    PersistentSlotValues(const PersistentSlotValues& source, Budget& budget, std::shared_ptr<Budget> keeper = {})
        : PersistentSlotValues(budget, std::move(keeper))
    {
        keep_overlap_ = true;
        imported_.add(source.metadata_bytes());
        if (has_keeper_ && source.has_keeper_ && &imported_.arena() != &source.imported_.arena())
            root_ = source.root_;
        else
            // Borrowed budgets cannot escape their source scope. Copy values
            // for this legacy path while retaining its construction overlap.
            source.visit([&](std::size_t slot, Value value) { set(slot, value); });
    }
    PersistentSlotValues(const PersistentSlotValues&) = delete;
    PersistentSlotValues& operator=(const PersistentSlotValues&) = delete;
    PersistentSlotValues(PersistentSlotValues&&) noexcept = default;
    [[nodiscard]] std::size_t size() const noexcept { return root_ ? root_->size : 0; }
    [[nodiscard]] std::size_t metadata_bytes() const noexcept { return root_ ? root_->subtree_bytes : 0; }
    [[nodiscard]] const Value* find(std::size_t slot) const noexcept
    {
        const auto* row = Index::lower_bound(root_, [&](const Row& value) { return value.slot < slot; });
        return row && row->slot == slot ? &row->value : nullptr;
    }
    [[nodiscard]] bool contains(std::size_t slot) const noexcept { return find(slot) != nullptr; }
    [[nodiscard]] Value at(std::size_t slot) const
    {
        if (const auto* value = find(slot))
            return *value;
        throw std::out_of_range("Numeric metadata slot is absent.");
    }
    void set(std::size_t slot, Value value)
    {
        const auto* old = find(slot);
        if (old && *old == value)
        {
            work_ = {};
            return;
        }
        persistent_ordered_detail::OrderedPageWork work;
        const Row replacement{slot, value};
        auto next = Index(imported_.keeper(), &work).edit(root_, replacement, &replacement, old != nullptr);
        const auto inherited = inherited_bytes(next);
        if (inherited > imported_.bytes())
            throw std::logic_error("Numeric metadata edit introduced unadmitted foreign pages.");
        root_ = std::move(next);
        if (!keep_overlap_)
            imported_.release(imported_.bytes() - inherited);
        work_ = work;
    }
    [[nodiscard]] std::size_t overlap_bytes() const noexcept { return imported_.bytes() - inherited_bytes(root_); }
    void finish_overlap() noexcept
    {
        imported_.release(overlap_bytes());
        keep_overlap_ = false;
    }
    [[nodiscard]] const persistent_ordered_detail::OrderedPageWork& last_edit_work() const noexcept { return work_; }
    template<class Visit> void visit(Visit&& visitor) const
    {
        (void) Index::visit_from(
            root_, [](const Row&) { return false; },
            [&](const Row& row)
            {
                visitor(row.slot, row.value);
                return true;
            });
    }
    template<class Visit> void visit_borrowed_payload(Visit&& visitor, ContactPayloadRole role) const
    {
        Index::visit_pages(root_,
            [&](const Root& node)
            {
                contact_payload_object(visitor, node.get(), role, sizeof(typename Index::Node), true);
                contact_payload_buffer(visitor, node.get(), role, node->rows);
                contact_payload_buffer(visitor, node.get(), role, node->children);
            });
    }

private:
    [[nodiscard]] std::size_t inherited_bytes(const Root& root) const noexcept
    {
        if (!root)
            return 0;
        return root->charge.arena.get() == imported_.keeper().get() ? root->subtree_bytes - root->same_arena_bytes
                                                                    : root->subtree_bytes;
    }
    ContactGroupAllocation imported_;
    Root root_;
    persistent_ordered_detail::OrderedPageWork work_;
    bool has_keeper_ = false;
    bool keep_overlap_ = false;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
