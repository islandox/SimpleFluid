/** @file PersistentAppendSlots.hh
 * @brief Dense append-only ordinals in immutable metadata pages.
 */
#pragma once
#include "geometry/mesh/AMR/PersistentOrderedPages.hh"

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Stable insertion ordinals backed by a 32-way tree. Each append copies the
 * last path. Payload bodies are owned independently of the slot metadata. */
template<class T> class PersistentAppendSlots
{
    static constexpr auto bits = 5u;
    static constexpr auto width = std::size_t{1} << bits;
    struct Node;

public:
    using Root = std::shared_ptr<const Node>;

    explicit PersistentAppendSlots(std::shared_ptr<Budget> arena) : arena_(std::move(arena))
    {
        if (!arena_)
            throw std::invalid_argument("Persistent slots require an allocation arena.");
    }

    [[nodiscard]] static std::size_t size(const Root& root) noexcept { return root ? root->size : 0; }
    [[nodiscard]] static std::size_t storage_bytes(const Root& root) noexcept { return root ? root->bytes : 0; }
    [[nodiscard]] std::size_t inherited_bytes(const Root& root) const noexcept
    {
        return storage_bytes(root) - (root && root->charge.arena == arena_ ? root->local_bytes : 0);
    }

    [[nodiscard]] Root append(const Root& root, const T& value) const
    {
        if (!root)
            return single(0, value);
        if (root->size == std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("Persistent slot count overflows.");
        if (root->size == capacity(root->depth))
        {
            const std::array children{root, single(root->depth, value)};
            return branch(root->depth + 1, children);
        }
        return append_to(*root, value);
    }

    [[nodiscard]] static const T& at(const Root& root, std::size_t slot)
    {
        if (slot >= size(root))
            throw std::out_of_range("Persistent slot ordinal is not present.");
        auto node = root.get();
        while (node->depth)
        {
            const auto shift = bits * node->depth;
            const auto child = (slot >> shift) & (width - 1);
            node = node->children[child].get();
        }
        return node->values[slot & (width - 1)];
    }

    /** Find the first entry for which the monotonic predicate is false. */
    template<class Before> [[nodiscard]] static const T* lower_bound(const Root& root, const Before& before)
    {
        auto node = root.get();
        while (node && node->depth)
        {
            const auto found = std::partition_point(
                node->children.begin(), node->children.end(), [&](const Root& child) { return before(*child->last); });
            if (found == node->children.end())
                return nullptr;
            node = found->get();
        }
        if (!node)
            return nullptr;
        const auto found = std::partition_point(node->values.begin(), node->values.end(), before);
        return found == node->values.end() ? nullptr : std::addressof(*found);
    }

    /** Allocation identity census, for diagnostics or explicit verification. */
    template<class Visit> static void visit_pages(const Root& root, const Visit& visit)
    {
        if (!root)
            return;
        visit(root.get(), root->charge.bytes);
        for (const auto& child : root->children)
            visit_pages(child, visit);
    }

private:
    using Charge = persistent_ordered_detail::PageCharge;

    struct Node
    {
        Charge charge;
        std::vector<T> values;
        std::vector<Root> children;
        const T* last = nullptr;
        std::size_t size = 0, bytes = 0, local_bytes = 0;
        unsigned depth;

        Node(std::shared_ptr<Budget> arena, unsigned depth_) : charge(std::move(arena)), depth(depth_) {}
    };

    static std::size_t capacity(unsigned depth) noexcept
    {
        const auto exponent = bits * (depth + 1);
        return exponent >= std::numeric_limits<std::size_t>::digits ? std::numeric_limits<std::size_t>::max()
                                                                    : std::size_t{1} << exponent;
    }

    [[nodiscard]] std::shared_ptr<Node> allocate(unsigned depth) const
    {
        Charge metadata(arena_);
        metadata.add(sizeof(Node) + persistent_ordered_detail::control_bytes);
        auto result = std::make_shared<Node>(arena_, depth);
        result->charge.bytes = std::exchange(metadata.bytes, 0);
        return result;
    }

    template<class Value> static void reserve(std::vector<Value>& values, std::size_t count, Charge& charge)
    {
        charge.add(persistent_ordered_detail::checked_product(count, sizeof(Value)));
        values.reserve(count);
        if (values.capacity() != count)
            throw std::invalid_argument("Persistent slot capacity differs from admission.");
    }

    [[nodiscard]] Root leaf(std::span<const T> previous, const T& value) const
    {
        auto next = allocate(0);
        reserve(next->values, previous.size() + 1, next->charge);
        next->values.assign(previous.begin(), previous.end());
        next->values.push_back(value);
        next->last = std::addressof(next->values.back());
        next->size = next->values.size();
        next->bytes = next->local_bytes = next->charge.bytes;
        return next;
    }

    [[nodiscard]] Root branch(unsigned depth, std::span<const Root> children) const
    {
        auto next = allocate(depth);
        reserve(next->children, children.size(), next->charge);
        next->children.assign(children.begin(), children.end());
        next->last = children.back()->last;
        next->bytes = next->local_bytes = next->charge.bytes;
        for (const auto& child : children)
        {
            next->size = persistent_ordered_detail::checked_sum(next->size, child->size);
            next->bytes = persistent_ordered_detail::checked_sum(next->bytes, child->bytes);
            if (child->charge.arena == arena_)
                next->local_bytes = persistent_ordered_detail::checked_sum(next->local_bytes, child->local_bytes);
        }
        return next;
    }

    [[nodiscard]] Root single(unsigned depth, const T& value) const
    {
        if (!depth)
            return leaf({}, value);
        const auto child = single(depth - 1, value);
        return branch(depth, std::span(&child, 1));
    }

    [[nodiscard]] Root append_to(const Node& root, const T& value) const
    {
        if (!root.depth)
            return leaf(root.values, value);
        std::array<Root, width> children;
        auto end = std::copy(root.children.begin(), root.children.end(), children.begin());
        const auto& last = root.children.back();
        if (last->size == capacity(last->depth))
            *end++ = single(last->depth, value);
        else
            *std::prev(end) = append_to(*last, value);
        return branch(root.depth, std::span(children.begin(), end));
    }

    std::shared_ptr<Budget> arena_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
