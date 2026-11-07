/** Immutable exact owner-cover facts, independent of exchange context. */
#pragma once
#include "AdaptiveProviderKernel.hh"
#include "ContactPayloadCensus.hh"
#include "ContactPhysicalResidency.hh"
#include "PersistentOrderedPages.hh"
#include "geometry/PLIC/ScalarPacket.hh"
#include <algorithm>
#include <array>
#include <cmath>
#include <compare>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class ContactRankCoverReceiptBuilder;
class ContactRankCoverReceipts : public std::enable_shared_from_this<ContactRankCoverReceipts>
{
    struct Domain
    {
        ContactGroupAllocation admission;
        AdaptiveBytes bytes;
        Domain(ContactGroupAllocation memory, std::span<const char> input)
            : admission(std::move(memory)), bytes(input.begin(), input.end())
        {
            if (bytes.capacity() != input.size())
                throw std::invalid_argument("Rank-cover domain capacity differs from admission.");
        }
    };
    using Packets = std::array<std::uint64_t, 6 * PLIC::ScalarPacketWords{}.size()>;
    struct Selector
    {
        int owner;
        std::span<const char> domain;
        Packets packets;
    };
    struct Row
    {
        int owner = -1;
        std::shared_ptr<const Domain> domain;
        Packets packets;
        AdaptiveContactBounds bounds;
        std::size_t occurrences = 1;
        [[nodiscard]] Selector selector() const noexcept { return {owner, domain->bytes, packets}; }
    };
    static bool before(const Selector& a, const Selector& b)
    {
        if (a.owner != b.owner)
            return a.owner < b.owner;
        const auto compare =
            std::lexicographical_compare_three_way(a.domain.begin(), a.domain.end(), b.domain.begin(), b.domain.end());
        return compare < 0 || (compare == 0 && a.packets < b.packets);
    }
    struct Less
    {
        bool operator()(const Row& a, const Row& b) const { return before(a.selector(), b.selector()); }
    };
    using Tree = persistent_ordered_detail::OrderedPages<Row, Less>;
    static Packets packets(const AdaptiveContactBounds& bounds)
    {
        Packets result;
        std::size_t at = 0;
        for (const auto* axis : {&bounds.lower, &bounds.upper})
            for (auto value : *axis)
                for (auto word : PLIC::scalar_packet(value))
                    result[at++] = word;
        return result;
    }
    static std::pair<const Row*, std::size_t> find(const Tree::Ptr& root, const Selector& key)
    {
        const auto* node = root.get();
        std::size_t ordinal = 0;
        while (node && !node->leaf())
        {
            const Tree::Node* next = nullptr;
            for (const auto& child : node->children)
            {
                if (!before(child->maximum->selector(), key))
                {
                    next = child.get();
                    break;
                }
                ordinal += child->size;
            }
            node = next;
        }
        if (!node)
            return {nullptr, ordinal};
        const auto row = std::lower_bound(node->rows.begin(), node->rows.end(), key,
            [](const Row& value, const Selector& selector) { return before(value.selector(), selector); });
        ordinal += std::size_t(row - node->rows.begin());
        return row == node->rows.end() || before(key, row->selector())
                   ? std::pair<const Row*, std::size_t>{nullptr, ordinal}
                   : std::pair<const Row*, std::size_t>{&*row, ordinal};
    }

public:
    struct Work
    {
        std::size_t received_covers = 0, reconciled_selectors = 0, transient_counter_bytes = 0;
        std::size_t copied_pages = 0, copied_page_bytes = 0, copied_domain_bytes = 0;
    };
    [[nodiscard]] Work work() const noexcept { return work_; }
    [[nodiscard]] std::size_t distinct_cover_count() const noexcept { return root_ ? root_->size : 0; }
    [[nodiscard]] std::size_t page_height() const noexcept
    {
        std::size_t height = 0;
        const auto* page = root_.get();
        while (page)
        {
            ++height;
            page = page->children.empty() ? nullptr : page->children.front().get();
        }
        return height;
    }
    [[nodiscard]] bool contains(int owner, std::span<const char> domain, const std::array<long double, 3>& point) const
    {
        bool result = false;
        Tree::visit_from(
            root_,
            [&](const Row& row)
            {
                const auto compare = std::lexicographical_compare_three_way(
                    row.domain->bytes.begin(), row.domain->bytes.end(), domain.begin(), domain.end());
                return row.owner < owner || (row.owner == owner && compare < 0);
            },
            [&](const Row& row)
            {
                if (row.owner != owner || !std::ranges::equal(row.domain->bytes, domain))
                    return false;
                bool inside = true;
                for (unsigned a = 0; a != 3; ++a)
                    inside = inside && point[a] >= row.bounds.lower[a] && point[a] <= row.bounds.upper[a];
                result = result || inside;
                return !result;
            });
        return result;
    }
    template<class Visitor> void visit(Visitor&& visitor) const
    {
        Tree::visit_from(
            root_, [](const Row&) { return false; },
            [&](const Row& row)
            {
                visitor(row.owner, std::span<const char>(row.domain->bytes), row.bounds, row.occurrences);
                return true;
            });
    }
    void visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>& visitor) const
    {
        contact_payload_object(visitor, this, ContactPayloadRole::RankCoverSnapshot, sizeof(*this), true);
        Tree::visit_pages(root_,
            [&](const auto& node)
            {
                contact_payload_object(visitor, node.get(), ContactPayloadRole::RankCoverPage, sizeof(*node), true);
                contact_payload_buffer(visitor, node.get(), ContactPayloadRole::RankCoverPage, node->rows);
                contact_payload_buffer(visitor, node.get(), ContactPayloadRole::RankCoverPage, node->children);
                for (const auto& row : node->rows)
                {
                    contact_payload_object(
                        visitor, row.domain.get(), ContactPayloadRole::RankCoverDomain, sizeof(Domain), true);
                    contact_payload_buffer(
                        visitor, row.domain.get(), ContactPayloadRole::RankCoverDomain, row.domain->bytes);
                }
            });
    }
    struct BorrowedLeafPage
    {
        std::weak_ptr<const void> control;
        const void* object;
        const void* rows;
        std::size_t object_bytes, row_bytes, first_ordinal, count;
    };
    /** Diagnostic only. The caller pins this immutable snapshot throughout
     * observation. Weak controls do not acquire a strong page pin; no mutable
     * page or construction authority escapes the callback. */
    void visit_borrowed_leaf_pages(const std::function<void(const BorrowedLeafPage&)>& visit) const
    {
        std::size_t first = 0;
        const auto descend = [&](auto&& self, const Tree::Ptr& node) -> void
        {
            if (!node)
                return;
            if (node->leaf())
            {
                visit(BorrowedLeafPage{std::weak_ptr<const void>(node), node.get(), node->rows.data(), sizeof(*node),
                    node->rows.capacity() * sizeof(Row), first, node->rows.size()});
                first += node->rows.size();
            }
            else
                for (const auto& child : node->children)
                    self(self, child);
        };
        descend(descend, root_);
    }
    /** First/last ownership of actual pages stops at an unchanged subtree.
     * The caller pins this snapshot while its usage entries are live. */
    std::size_t observe(ContactPhysicalResidency& residency, bool added) const
    {
        using Role = ContactPhysicalResidency::Role;
        std::size_t visited = 0;
        const auto enter = [&](const auto& owner, std::size_t bytes, const Budget* origin)
        {
            ++visited;
            const auto change = added ? residency.retain(Role::RankCovers, owner, bytes, origin)
                                      : residency.release(Role::RankCovers, owner);
            return added ? change.inserted() : change.retired();
        };
        if (!enter(shared_from_this(), admission_.bytes(), &admission_.arena()))
            return visited;
        const auto descend = [&](auto&& self, const Tree::Ptr& node) -> void
        {
            if (!node || !enter(node, node->charge.bytes, node->charge.arena.get()))
                return;
            for (const auto& child : node->children)
                self(self, child);
            for (const auto& row : node->rows)
                (void) enter(row.domain, row.domain->admission.bytes(), &row.domain->admission.arena());
        };
        descend(descend, root_);
        return visited;
    }

private:
    friend class ContactRankCoverReceiptBuilder;
    ContactGroupAllocation admission_;
    Tree::Ptr root_;
    Work work_;
    explicit ContactRankCoverReceipts(ContactGroupAllocation memory) : admission_(std::move(memory)) {}
};

/** Synchronous wire receipt reconciliation. Numeric counts are validation
 * scratch; no domain or complete geometric-cover array is cloned. */
class ContactRankCoverReceiptBuilder
{
    using Snapshot = ContactRankCoverReceipts;
    using Row = Snapshot::Row;
    ContactGroupAllocation scratch_;
    std::shared_ptr<const Snapshot> previous_;
    std::shared_ptr<Snapshot> result_;
    persistent_ordered_detail::OrderedPageWork work_;
    Snapshot::Tree tree_;
    std::vector<std::size_t> seen_;

public:
    ContactRankCoverReceiptBuilder(std::shared_ptr<Budget> arena, std::shared_ptr<const Snapshot> previous = {})
        : scratch_(arena), previous_(std::move(previous)), tree_(arena, &work_)
    {
        ContactGroupAllocation memory(arena);
        memory.add(sizeof(Snapshot) + 128);
        result_ = std::shared_ptr<Snapshot>(new Snapshot(std::move(memory)));
        if (previous_)
        {
            result_->root_ = previous_->root_;
            scratch_.reserve(seen_, previous_->distinct_cover_count());
            seen_.resize(previous_->distinct_cover_count());
            result_->work_.transient_counter_bytes = seen_.capacity() * sizeof(std::size_t);
        }
    }
    void accept(int owner, std::span<const char> domain, const AdaptiveContactBounds& bounds)
    {
        if (owner < 0 || domain.empty())
            throw std::invalid_argument("Invalid owner-cover fact.");
        for (unsigned a = 0; a != 3; ++a)
            if (!std::isfinite(bounds.lower[a]) || !std::isfinite(bounds.upper[a]) || bounds.upper[a] < bounds.lower[a])
                throw std::invalid_argument("Invalid owner-cover bounds.");
        ++result_->work_.received_covers;
        const Snapshot::Selector selector{owner, domain, Snapshot::packets(bounds)};
        if (previous_)
            if (const auto [row, ordinal] = Snapshot::find(previous_->root_, selector); row)
            {
                if (seen_[ordinal] == std::numeric_limits<std::size_t>::max())
                    throw std::invalid_argument("Owner-cover occurrence count overflows.");
                ++seen_[ordinal];
                return;
            }
        if (const auto [row, ordinal] = Snapshot::find(result_->root_, selector); row)
        {
            auto next = *row;
            if (next.occurrences == std::numeric_limits<std::size_t>::max())
                throw std::invalid_argument("Owner-cover occurrence count overflows.");
            ++next.occurrences;
            result_->root_ = tree_.edit(result_->root_, *row, &next, true);
            return;
        }
        std::shared_ptr<const Snapshot::Domain> domain_owner;
        const auto reuse = Snapshot::Tree::lower_bound(result_->root_,
            [&](const Row& row)
            {
                return row.owner < owner ||
                       (row.owner == owner && std::lexicographical_compare(row.domain->bytes.begin(),
                                                  row.domain->bytes.end(), domain.begin(), domain.end()));
            });
        if (reuse && reuse->owner == owner && std::ranges::equal(reuse->domain->bytes, domain))
            domain_owner = reuse->domain;
        else
        {
            ContactGroupAllocation memory(scratch_.keeper());
            memory.add(sizeof(Snapshot::Domain) + 128 + domain.size());
            result_->work_.copied_domain_bytes += memory.bytes();
            domain_owner = std::make_shared<Snapshot::Domain>(std::move(memory), domain);
        }
        const Row row{owner, std::move(domain_owner), selector.packets, bounds, 1};
        result_->root_ = tree_.edit(result_->root_, row, &row, false);
    }
    [[nodiscard]] std::shared_ptr<const Snapshot> finish()
    {
        if (previous_)
        {
            std::size_t ordinal = 0;
            Snapshot::Tree::visit_from(
                previous_->root_, [](const Row&) { return false; },
                [&](const Row& row)
                {
                    ++result_->work_.reconciled_selectors;
                    const auto occurrences = seen_[ordinal++];
                    if (!occurrences)
                        result_->root_ = tree_.edit(result_->root_, row, nullptr, true);
                    else if (occurrences != row.occurrences)
                    {
                        auto next = row;
                        next.occurrences = occurrences;
                        result_->root_ = tree_.edit(result_->root_, row, &next, true);
                    }
                    return true;
                });
        }
        result_->work_.copied_pages = work_.copied_pages;
        result_->work_.copied_page_bytes = work_.copied_bytes;
        previous_.reset();
        return std::exchange(result_, {});
    }
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
