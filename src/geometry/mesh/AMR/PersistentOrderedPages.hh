/** @file PersistentOrderedPages.hh
 * @brief Immutable ordered metadata pages with exact subtree allocation totals.
 */
#pragma once
#include "geometry/details/NativeConstructionAdmission.hh"
#include "geometry/details/NativeConstructorOwnerSink.hh"
#include "geometry/mesh/AMR/AdaptivePipelineAudit.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/NativeContactFunding.hh"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail::persistent_ordered_detail
{
inline constexpr auto control_bytes = 4 * sizeof(void*);

inline std::size_t checked_sum(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::invalid_argument("Ordered metadata admission size overflows.");
    return a + b;
}

inline std::size_t checked_product(std::size_t a, std::size_t b)
{
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("Ordered metadata admission size overflows.");
    return a * b;
}

struct NativePageAccounting
{
    geometry_detail::NativeConstructorStorageOwner storage;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation, node;
    explicit NativePageAccounting(const geometry_detail::NativeConstructionAdmission& admission)
        : storage(geometry_detail::native_constructor_storage(admission)) {}
    [[nodiscard]] std::size_t bytes() const
    {
        auto value = checked_sum(allocation->fixed_allocation_bytes(), storage.envelope->fixed_allocation_bytes());
        value = checked_sum(value, storage.owner->exclusive_bytes());
        return node ? checked_sum(value, node->fixed_allocation_bytes()) : value;
    }
};
struct PageCharge
{
    std::shared_ptr<Budget> arena;
    std::shared_ptr<NativePageAccounting> native;
    std::size_t bytes = 0, allocator_bytes = 0;

    explicit PageCharge(
        std::shared_ptr<Budget> owner, std::shared_ptr<const geometry_detail::NativeAllocationState> domain = {})
        : arena(std::move(owner))
    {
        if (domain)
        {
            auto tracked = geometry_detail::NativeAllocationEnvelopeAccess::make<NativePageAccounting>(
                geometry_detail::NativeConstructionAdmission::from_state(domain),
                geometry_detail::NativeConstructionAdmission::from_state(domain));
            tracked.owner->allocation = std::move(tracked.envelope);
            native = std::move(tracked.owner);
            bytes = allocator_bytes = native->bytes();
        }
    }
    ~PageCharge()
    {
        const auto held = bytes - allocator_bytes;
        if (native)
        {
            if (held) std::terminate();
        }
        else
            arena->used -= held;
    }
    PageCharge(const PageCharge&) = delete;
    PageCharge& operator=(const PageCharge&) = delete;

    void add(std::size_t value)
    {
        const auto next = checked_sum(bytes, value);
        if (native)
            throw std::invalid_argument("Native ordered rich row buffers require actual constructor receipts.");
        else
            arena->add(value);
        bytes = next;
    }

    void release(std::size_t value) noexcept
    {
        if (value > bytes - allocator_bytes)
            std::terminate();
        bytes -= value;
        if (native)
            std::terminate();
        else
            arena->used -= value;
    }
    template<class T> void append(std::vector<T>& values)
    {
        if (native)
        {
            if (values.size() < values.capacity()) return;
            const auto previous = values.capacity();
            if (previous > arena->limit / sizeof(T) / 2)
                throw std::invalid_argument("Native ordered vector capacity exceeds its active cap.");
            const auto next = previous ? 2 * previous : 1;
            std::vector<T> replacement;
            native->storage.owner->reserve(replacement, next);
            replacement.insert(replacement.end(), std::make_move_iterator(values.begin()),
                std::make_move_iterator(values.end()));
            values.swap(replacement);
            if (previous) native->storage.owner->reset_buffer(replacement);
            bytes = allocator_bytes = native->bytes();
            return;
        }
        const auto before = arena->used;
        try
        {
            arena->append_capacity(values);
        }
        catch (...)
        {
            bytes += arena->used - before;
            throw;
        }
        bytes += arena->used - before;
    }
};

struct OrderedPageWork
{
    std::size_t copied_pages = 0, copied_bytes = 0, retired_foreign_index_bytes = 0;
};

inline constexpr std::size_t leaf_capacity = 64, branch_capacity = 32;

struct NoRowWeights
{
    struct Summary {};
    static constexpr bool enabled = false;
};

struct NoRowBuffers
{
    static constexpr bool enabled = false;
};

/** Optional weights summarize row-exclusive immutable payloads. Implementations
 * must allocate nothing in add_row/add_child, check aggregate overflow, and
 * distinguish bodies created in the current node arena from inherited bodies.
 * Shared payloads appearing in several rows require a separate owner union. */
template<class Row, class Less, bool FixedLeafCapacity = false, class Work = OrderedPageWork,
    class Weights = NoRowWeights, std::size_t LeafCapacity = leaf_capacity, class Buffers = NoRowBuffers>
class OrderedPages
{
    static_assert(LeafCapacity >= 2);

public:
    struct Node;
    using Ptr = std::shared_ptr<const Node>;
    struct Node
    {
        PageCharge charge;
        std::vector<Row> rows;
        std::vector<Ptr> children;
        const Row* maximum = nullptr;
        std::size_t size = 0, subtree_bytes = 0, same_arena_bytes = 0;
        [[no_unique_address]] typename Weights::Summary weights{};
        explicit Node(const std::shared_ptr<Budget>& arena,
            std::shared_ptr<const geometry_detail::NativeAllocationState> native = {})
            : charge(arena, std::move(native))
        {
        }
        ~Node()
        {
            if (charge.native)
            {
                charge.native->storage.owner->reset_buffer(rows);
                charge.native->storage.owner->reset_buffer(children);
            }
        }
        [[nodiscard]] bool leaf() const noexcept { return children.empty(); }
        [[nodiscard]] std::size_t occupancy() const noexcept { return leaf() ? rows.size() : children.size(); }

    private:
        friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativeContactFundingAccess;
        [[nodiscard]] NativeContactOwnerFunding native_funding(const Ptr& owner, std::size_t bytes) const
        {
            if (!charge.native) return {};
            const auto& actual = charge.native;
            if (owner.get() != this || !actual->node || !actual->node->matches(owner) || bytes != charge.bytes)
                throw std::invalid_argument("Native contact index has a foreign actual page or extent.");
            NativeContactOwnerFunding result;
            result.original_bytes = bytes;
            result.index_accounting = actual;
            result.index_owner = actual->node;
            result.index_extent = actual->allocation;
            result.index_storage = actual->storage;
            return result;
        }
    };

    explicit OrderedPages(std::shared_ptr<Budget> arena, Work* work = nullptr, Less less = {},
        std::shared_ptr<const geometry_detail::NativeAllocationState> native = {})
        : arena_(std::move(arena)), native_(std::move(native)), work_(work), less_(std::move(less))
    {
        if (native_)
        {
            const auto same_arena = [&](const std::shared_ptr<const void>& owner)
            {
                return owner && owner.get() == arena_.get() && !owner.owner_before(arena_) &&
                    !arena_.owner_before(owner);
            };
            const auto& mirror = native_->mirror;
            if (!same_arena(native_->counter_owner) &&
                (!same_arena(mirror.owner) || !mirror.owner_proof.matches(mirror.owner)))
                throw std::invalid_argument("Native ordered pages have a foreign scalar admission domain.");
        }
        if (native_ && !PLIC::geometry_encoding_capacity_profile_supported())
            throw std::invalid_argument("Native ordered page vector capacity profile is not qualified.");
    }

    [[nodiscard]] Ptr build(std::span<const Row> sorted) const
    {
        if (sorted.empty())
            return {};
        for (std::size_t slot = 1; slot < sorted.size(); ++slot)
            if (!less_(sorted[slot - 1], sorted[slot]))
                throw std::invalid_argument("Ordered metadata index contains duplicate selectors.");
        PageCharge scratch(arena_, native_);
        std::vector<Ptr> pages;
        const auto groups = ceiling(sorted.size(), LeafCapacity);
        reserve(pages, groups, scratch);
        std::size_t offset = 0;
        for (std::size_t group = 0; group < groups; ++group)
        {
            const auto count = group_size(sorted.size(), groups, group);
            pages.push_back(leaf(sorted.subspan(offset, count)));
            offset += count;
        }
        while (pages.size() > 1)
        {
            std::vector<Ptr> parents;
            const auto count = ceiling(pages.size(), branch_capacity);
            reserve(parents, count, scratch);
            offset = 0;
            for (std::size_t group = 0; group < count; ++group)
            {
                const auto width = group_size(pages.size(), count, group);
                parents.push_back(branch(std::span(pages).subspan(offset, width)));
                offset += width;
            }
            const auto old = pages.capacity() * sizeof(Ptr);
            pages.swap(parents);
            if (scratch.native)
            {
                scratch.native->storage.owner->reset_buffer(parents);
                scratch.bytes = scratch.allocator_bytes = scratch.native->bytes();
            }
            else
            {
                std::vector<Ptr>().swap(parents);
                scratch.release(old);
            }
        }
        return std::move(pages.front());
    }

    [[nodiscard]] Ptr edit(const Ptr& old, const Row& key, const Row* replacement, bool expected) const
    {
        if (!old)
        {
            if (expected || !replacement)
                throw std::invalid_argument("Ordered metadata edit lacks its accepted selector.");
            return leaf(std::span(replacement, 1));
        }
        auto result = rewrite(*old, key, replacement, expected);
        if (result.count == 0)
            return {};
        if (result.count == 2)
            return branch(std::span(result.pages));
        auto root = std::move(result.pages[0]);
        while (root->children.size() == 1)
        {
            retire(*root);
            auto child = root->children.front();
            root = std::move(child);
        }
        return root;
    }

    template<class Before> [[nodiscard]] static const Row* lower_bound(const Ptr& root, const Before& before)
    {
        auto node = root.get();
        while (node && !node->leaf())
        {
            const auto child = std::partition_point(node->children.begin(), node->children.end(),
                [&](const Ptr& value) { return before(*value->maximum); });
            if (child == node->children.end())
                return nullptr;
            node = child->get();
        }
        if (!node)
            return nullptr;
        const auto row = std::partition_point(node->rows.begin(), node->rows.end(), before);
        return row == node->rows.end() ? nullptr : &*row;
    }

    template<class Before, class Visit>
    static bool visit_from(const Ptr& root, const Before& before, const Visit& visit)
    {
        if (!root || before(*root->maximum))
            return true;
        if (root->leaf())
        {
            const auto begin = std::partition_point(root->rows.begin(), root->rows.end(), before);
            return std::all_of(begin, root->rows.end(), visit);
        }
        for (const auto& child : root->children)
            if (!visit_from(child, before, visit))
                return false;
        return true;
    }

    /** Exact native Node and receipt vertices; leaves child/row fields to the
     * owning typed visitor after this first/last semantic transition. */
    static bool visit_native_node(const Ptr& node, const geometry_detail::NativeConstructorOwnerSink& sink)
    {
        if (!node || !node->charge.native)
            throw std::invalid_argument("Qualified native metadata node lacks actual allocation receipts.");
        const auto& native = node->charge.native;
        if (!sink.semantic(node, geometry_detail::NativeConstructorOwnerKind::Topology, native->node)) return false;
        if (sink.semantic(native, geometry_detail::NativeConstructorOwnerKind::Storage, native->allocation))
        {
            sink.fixed(native->allocation);
            sink.fixed(native->node);
            sink.storage(native->storage);
        }
        return true;
    }
    template<class Visit> static void visit_pages(const Ptr& root, const Visit& visit)
    {
        if (!root)
            return;
        visit(root);
        for (const auto& child : root->children)
            visit_pages(child, visit);
    }

private:
    struct Replacement
    {
        std::array<Ptr, 2> pages;
        std::size_t count;
    };
    static std::size_t ceiling(std::size_t count, std::size_t width) { return count / width + (count % width != 0); }
    static std::size_t group_size(std::size_t count, std::size_t groups, std::size_t group)
    {
        return count / groups + (group < count % groups);
    }
    template<class Value> static void reserve(std::vector<Value>& values, std::size_t count, PageCharge& charge)
    {
        if (charge.native)
        {
            charge.native->storage.owner->reserve(values, count);
            charge.bytes = charge.allocator_bytes = charge.native->bytes();
            return;
        }
        charge.add(checked_product(count, sizeof(Value)));
        values.reserve(count);
        if (values.capacity() != count)
            throw std::invalid_argument("Ordered metadata page capacity differs from admission.");
    }

    void retire(const Node& node) const
    {
        if (work_ && node.charge.arena != arena_)
            work_->retired_foreign_index_bytes = checked_sum(work_->retired_foreign_index_bytes, node.charge.bytes);
    }
    void record(const Node& node) const
    {
        record_pipeline_work(PipelineWork::OrderedPageCreated);
        record_pipeline_work(PipelineWork::OrderedPageAdmittedBytes, node.charge.bytes);
        if (work_)
        {
            ++work_->copied_pages;
            work_->copied_bytes = checked_sum(work_->copied_bytes, node.charge.bytes);
        }
    }
    [[nodiscard]] std::shared_ptr<Node> allocate() const
    {
        if (native_)
        {
            const auto admission = geometry_detail::NativeConstructionAdmission::from_state(native_);
            auto tracked = geometry_detail::NativeAllocationEnvelopeAccess::make<Node>(admission, arena_);
            auto node = std::move(tracked.owner);
            PageCharge charge(arena_, native_);
            charge.native->node = std::move(tracked.envelope);
            node->charge.native = std::move(charge.native);
            node->charge.bytes = node->charge.allocator_bytes = node->charge.native->bytes();
            charge.bytes = charge.allocator_bytes = 0;
            return node;
        }
        PageCharge initial(arena_);
        initial.add(sizeof(Node) + control_bytes);
        auto node = std::make_shared<Node>(arena_);
        node->charge.bytes = std::exchange(initial.bytes, 0);
        return node;
    }
    [[nodiscard]] Ptr leaf(std::span<const Row> values) const
    {
        auto node = allocate();
        reserve(node->rows, FixedLeafCapacity ? LeafCapacity : values.size(), node->charge);
        [[maybe_unused]] std::size_t copied_buffers = 0;
        if constexpr (Buffers::enabled)
        {
            for (const auto& row : values)
                copied_buffers = checked_sum(copied_buffers, Buffers::bytes(row));
            node->charge.add(copied_buffers);
        }
        node->rows.assign(values.begin(), values.end());
        if constexpr (Buffers::enabled)
        {
            std::size_t actual = 0;
            for (const auto& row : node->rows)
                actual = checked_sum(actual, Buffers::bytes(row));
            if (actual > copied_buffers)
                throw std::invalid_argument("Ordered row buffer copy exceeds its qualified admission.");
            node->charge.release(copied_buffers - actual);
        }
        node->maximum = &node->rows.back();
        node->size = values.size();
        node->subtree_bytes = node->charge.bytes;
        node->same_arena_bytes = node->charge.bytes;
        if constexpr (Weights::enabled)
            for (const auto& row : node->rows)
                Weights::add_row(node->weights, row, arena_.get());
        record(*node);
        return node;
    }
    [[nodiscard]] Ptr branch(std::span<const Ptr> values) const
    {
        auto node = allocate();
        reserve(node->children, branch_capacity, node->charge);
        node->children.assign(values.begin(), values.end());
        node->subtree_bytes = node->charge.bytes;
        node->same_arena_bytes = node->charge.bytes;
        for (const auto& child : values)
        {
            node->size = checked_sum(node->size, child->size);
            node->subtree_bytes = checked_sum(node->subtree_bytes, child->subtree_bytes);
            if (child->charge.arena == arena_)
                node->same_arena_bytes = checked_sum(node->same_arena_bytes, child->same_arena_bytes);
            if constexpr (Weights::enabled)
                Weights::add_child(node->weights, child->weights, child->charge.arena == arena_);
        }
        node->maximum = values.back()->maximum;
        record(*node);
        return node;
    }
    [[nodiscard]] Replacement pack_leaves(std::span<const Row> rows) const
    {
        if (rows.empty())
            return {{}, 0};
        if (rows.size() <= LeafCapacity)
            return {{leaf(rows), {}}, 1};
        const auto middle = rows.size() / 2;
        return {{leaf(rows.first(middle)), leaf(rows.subspan(middle))}, 2};
    }
    [[nodiscard]] Replacement pack_branches(std::span<const Ptr> children) const
    {
        if (children.empty())
            return {{}, 0};
        if (children.size() <= branch_capacity)
            return {{branch(children), {}}, 1};
        const auto middle = children.size() / 2;
        return {{branch(children.first(middle)), branch(children.subspan(middle))}, 2};
    }
    std::span<const Ptr> rebalance(std::span<Ptr> children) const
    {
        std::size_t index = 0;
        while (index < children.size() && children.size() > 1)
        {
            const auto& child = *children[index];
            const auto minimum = child.leaf() ? LeafCapacity / 2 : branch_capacity / 2;
            if (child.occupancy() >= minimum)
            {
                ++index;
                continue;
            }
            const auto left = index == 0 ? index : index - 1;
            const auto replacement = redistribute(*children[left], *children[left + 1]);
            children[left] = replacement.pages[0];
            if (replacement.count == 1)
            {
                std::move(children.begin() + left + 2, children.end(), children.begin() + left + 1);
                children = children.first(children.size() - 1);
            }
            else
                children[left + 1] = replacement.pages[1];
            index = left;
        }
        return children;
    }
    [[nodiscard]] Replacement redistribute(const Node& first, const Node& second) const
    {
        retire(first);
        retire(second);
        if (first.leaf())
        {
            PageCharge scratch(arena_, native_);
            std::array<Row, 2 * LeafCapacity> rows;
            [[maybe_unused]] std::size_t admitted_buffers = 0;
            if constexpr (Buffers::enabled)
            {
                for (const auto& row : first.rows)
                    admitted_buffers = checked_sum(admitted_buffers, Buffers::bytes(row));
                for (const auto& row : second.rows)
                    admitted_buffers = checked_sum(admitted_buffers, Buffers::bytes(row));
                scratch.add(admitted_buffers);
            }
            auto end = std::copy(first.rows.begin(), first.rows.end(), rows.begin());
            end = std::copy(second.rows.begin(), second.rows.end(), end);
            if constexpr (Buffers::enabled)
            {
                std::size_t actual_buffers = 0;
                for (auto row = rows.begin(); row != end; ++row)
                    actual_buffers = checked_sum(actual_buffers, Buffers::bytes(*row));
                if (actual_buffers > admitted_buffers)
                    throw std::invalid_argument("Ordered row buffer copy exceeds its qualified admission.");
                scratch.release(admitted_buffers - actual_buffers);
            }
            return pack_leaves(std::span(rows.begin(), end));
        }
        std::array<Ptr, 2 * branch_capacity> children;
        auto end = std::copy(first.children.begin(), first.children.end(), children.begin());
        end = std::copy(second.children.begin(), second.children.end(), end);
        return pack_branches(rebalance(std::span(children.begin(), end)));
    }
    [[nodiscard]] Replacement rewrite(const Node& node, const Row& key, const Row* replacement, bool expected) const
    {
        if (node.leaf())
        {
            const auto at = std::lower_bound(node.rows.begin(), node.rows.end(), key, less_);
            const bool present = at != node.rows.end() && !less_(key, *at);
            if (present != expected)
                throw std::invalid_argument("Ordered metadata edit changed its accepted selector.");
            retire(node);
            PageCharge scratch(arena_, native_);
            std::array<Row, LeafCapacity + 1> rows;
            [[maybe_unused]] std::size_t admitted_buffers = 0;
            if constexpr (Buffers::enabled)
            {
                for (auto row = node.rows.begin(); row != at; ++row)
                    admitted_buffers = checked_sum(admitted_buffers, Buffers::bytes(*row));
                if (replacement)
                    admitted_buffers = checked_sum(admitted_buffers, Buffers::bytes(*replacement));
                const auto rest = present ? std::next(at) : at;
                for (auto row = rest; row != node.rows.end(); ++row)
                    admitted_buffers = checked_sum(admitted_buffers, Buffers::bytes(*row));
                scratch.add(admitted_buffers);
            }
            auto end = std::copy(node.rows.begin(), at, rows.begin());
            if (replacement)
                *end++ = *replacement;
            end = std::copy(present ? std::next(at) : at, node.rows.end(), end);
            if constexpr (Buffers::enabled)
            {
                std::size_t actual_buffers = 0;
                for (auto row = rows.begin(); row != end; ++row)
                    actual_buffers = checked_sum(actual_buffers, Buffers::bytes(*row));
                if (actual_buffers > admitted_buffers)
                    throw std::invalid_argument("Ordered row buffer copy exceeds its qualified admission.");
                scratch.release(admitted_buffers - actual_buffers);
            }
            return pack_leaves(std::span(rows.begin(), end));
        }
        auto at = std::lower_bound(node.children.begin(), node.children.end(), key,
            [&](const Ptr& value, const Row& target) { return less_(*value->maximum, target); });
        if (at == node.children.end())
            --at;
        const auto edited = rewrite(**at, key, replacement, expected);
        retire(node);
        std::array<Ptr, branch_capacity + 1> children;
        auto end = std::copy(node.children.begin(), at, children.begin());
        end = std::copy_n(edited.pages.begin(), edited.count, end);
        end = std::copy(std::next(at), node.children.end(), end);
        return pack_branches(rebalance(std::span(children.begin(), end)));
    }
    std::shared_ptr<Budget> arena_;
    std::shared_ptr<const geometry_detail::NativeAllocationState> native_;
    Work* work_;
    [[no_unique_address]] Less less_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail::persistent_ordered_detail
