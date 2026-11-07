/** @file GeometryDemandPages.tcc
 * @brief Compiled-provider implementation for 64-row/32-child geometry pages.
 */
#pragma once
#include "geometry/PLIC/details/GeometryDemandPages.hh"
#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimpleFluid::PLIC::details::geometry_demand
{
inline constexpr std::size_t control_bytes = 128, leaf_capacity = 64, branch_capacity = 32;
inline std::size_t checked_sum(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Geometry demand size overflows.");
    return a + b;
}
inline void Admission::charge(std::size_t count, std::size_t width) const
{
    if (!retain_bytes || (width && count > std::numeric_limits<std::size_t>::max() / width))
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Geometry demand admission is invalid.");
    retain_bytes(context, count * width);
}
template<class Row, class Less> class OrderedPages
{
public:
    struct Node;
    using Ptr = std::shared_ptr<const Node>;
    static_assert(std::is_nothrow_copy_constructible_v<Row> && std::is_nothrow_copy_assignable_v<Row>);
    struct Node
    {
        std::size_t bytes = 0, payload_bytes = 0;
        std::vector<Row> rows;
        std::vector<Ptr> children;
        const Row* maximum = nullptr;
        std::size_t size = 0, subtree_bytes = 0, subtree_payload_bytes = 0, subtree_owner_bytes = 0, logical_size = 0,
                    subtree_owned_admission_bytes = 0;
        [[nodiscard]] bool leaf() const noexcept { return children.empty(); }
        [[nodiscard]] std::size_t occupancy() const noexcept { return leaf() ? rows.size() : children.size(); }
    };

    explicit OrderedPages(Admission admission, PageWork* work = nullptr) : admission_(admission), work_(work) {}

    [[nodiscard]] Ptr build(std::span<const Row> sorted) const
    {
        if (sorted.empty())
            return {};
        for (std::size_t slot = 1; slot < sorted.size(); ++slot)
            if (!Less{}(sorted[slot - 1], sorted[slot]))
                throw GeometryError(
                    GeometryErrorCode::InvalidInput, "Ordered metadata index contains duplicate selectors.");
        std::vector<Ptr> pages;
        const auto groups = ceiling(sorted.size(), leaf_capacity);
        reserve(pages, groups);
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
            reserve(parents, count);
            offset = 0;
            for (std::size_t group = 0; group < count; ++group)
            {
                const auto width = group_size(pages.size(), count, group);
                parents.push_back(branch(std::span(pages).subspan(offset, width)));
                offset += width;
            }
            pages.swap(parents);
            std::vector<Ptr>().swap(parents);
        }
        return std::move(pages.front());
    }

    [[nodiscard]] Ptr edit(const Ptr& old, const Row& key, const Row* replacement, bool expected) const
    {
        if (replacement && (Less{}(key, *replacement) || Less{}(*replacement, key)))
            throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry demand replacement changes its key.");
        if (!old)
        {
            if (expected || !replacement)
                throw GeometryError(
                    GeometryErrorCode::InvalidInput, "Ordered metadata edit lacks its accepted selector.");
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

    [[nodiscard]] static const Row& at(const Ptr& root, std::size_t slot, PageWork* work = nullptr)
    {
        if (!root || slot >= root->size)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid geometry demand rank.");
        auto node = root.get();
        while (!node->leaf())
        {
            if (work)
                ++work->visited_pages;
            auto child = node->children.begin();
            while (slot >= (*child)->size)
            {
                slot -= (*child)->size;
                ++child;
            }
            node = child->get();
        }
        if (work)
            ++work->visited_pages;
        return node->rows[slot];
    }
    struct LogicalRow
    {
        const Row* row;
        std::size_t offset;
    };
    [[nodiscard]] static LogicalRow logical_at(const Ptr& root, std::size_t slot, PageWork* work = nullptr)
    {
        if (!root || slot >= root->logical_size)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid logical geometry demand rank.");
        auto node = root.get();
        while (!node->leaf())
        {
            if (work)
                ++work->visited_pages;
            auto child = node->children.begin();
            while (slot >= (*child)->logical_size)
            {
                slot -= (*child)->logical_size;
                ++child;
            }
            node = child->get();
        }
        if (work)
            ++work->visited_pages;
        for (const auto& row : node->rows)
        {
            const auto count = [&]
            {
                if constexpr (requires { row.logical_size(); })
                    return row.logical_size();
                else
                    return std::size_t{1};
            }();
            if (slot < count)
                return {&row, slot};
            slot -= count;
        }
        throw GeometryError(GeometryErrorCode::InvalidTopology, "Geometry demand page lost its logical row.");
    }
    template<class Before>
    [[nodiscard]] static std::size_t logical_rank(const Ptr& root, const Before& before, PageWork* work = nullptr)
    {
        std::size_t result = 0;
        auto node = root.get();
        while (node)
        {
            if (work)
                ++work->visited_pages;
            if (node->leaf())
            {
                for (const auto& row : node->rows)
                {
                    if (!before(row))
                        return result;
                    if constexpr (requires { row.logical_size(); })
                        result = checked_sum(result, row.logical_size());
                    else
                        result = checked_sum(result, 1);
                }
                return result;
            }
            auto child = node->children.begin();
            while (child != node->children.end() && before(*(*child)->maximum))
            {
                result = checked_sum(result, (*child)->logical_size);
                ++child;
            }
            if (child == node->children.end())
                return result;
            node = child->get();
        }
        return result;
    }

    template<class Before>
    [[nodiscard]] static std::size_t rank(const Ptr& root, const Before& before, PageWork* work = nullptr)
    {
        std::size_t result = 0;
        auto node = root.get();
        while (node)
        {
            if (work)
                ++work->visited_pages;
            if (node->leaf())
                return checked_sum(
                    result, std::partition_point(node->rows.begin(), node->rows.end(), before) - node->rows.begin());
            auto child = node->children.begin();
            while (child != node->children.end() && before(*(*child)->maximum))
            {
                result = checked_sum(result, (*child)->size);
                ++child;
            }
            if (child == node->children.end())
                return result;
            node = child->get();
        }
        return result;
    }

    template<class Before>
    [[nodiscard]] static const Row* lower_bound(const Ptr& root, const Before& before, PageWork* work = nullptr)
    {
        auto node = root.get();
        while (node && !node->leaf())
        {
            if (work)
                ++work->visited_pages;
            const auto child = std::partition_point(node->children.begin(), node->children.end(),
                [&](const Ptr& value) { return before(*value->maximum); });
            if (child == node->children.end())
                return nullptr;
            node = child->get();
        }
        if (!node)
            return nullptr;
        if (work)
            ++work->visited_pages;
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
    template<class Value> void reserve(std::vector<Value>& values, std::size_t count) const
    {
        admission_.charge(count, sizeof(Value));
        values.reserve(count);
        if (values.capacity() != count)
            throw GeometryError(GeometryErrorCode::ResourceLimit, "Geometry demand page capacity is unqualified.");
    }
    void retire(const Node&) const
    {
        if (work_)
            ++work_->retired_page_visits;
    }
    void record(const Node& node) const
    {
        if (work_)
        {
            ++work_->copied_pages;
            work_->copied_bytes = checked_sum(work_->copied_bytes, node.bytes);
            work_->copied_payload_bytes = checked_sum(work_->copied_payload_bytes, node.payload_bytes);
        }
    }
    [[nodiscard]] std::shared_ptr<Node> allocate() const
    {
        admission_.charge(1, sizeof(Node) + control_bytes);
        auto node = std::make_shared<Node>();
        node->bytes = sizeof(Node) + control_bytes;
        node->payload_bytes = sizeof(Node);
        return node;
    }
    [[nodiscard]] Ptr leaf(std::span<const Row> values) const
    {
        auto node = allocate();
        reserve(node->rows, values.size());
        node->bytes = checked_sum(node->bytes, values.size() * sizeof(Row));
        node->rows.assign(values.begin(), values.end());
        node->maximum = &node->rows.back();
        node->size = values.size();
        for (const auto& row : node->rows)
        {
            if constexpr (requires { row.logical_size(); })
                node->logical_size = checked_sum(node->logical_size, row.logical_size());
            else
                node->logical_size = checked_sum(node->logical_size, 1);
            if constexpr (requires { row.exclusive_admission_bytes(); })
                node->subtree_owned_admission_bytes =
                    checked_sum(node->subtree_owned_admission_bytes, row.exclusive_admission_bytes());
        }
        if constexpr (requires(const Row& row) { row.exclusive_payload_bytes(); })
            for (const auto& row : node->rows)
                node->subtree_owner_bytes = checked_sum(node->subtree_owner_bytes, row.exclusive_payload_bytes());
        node->payload_bytes = node->bytes - control_bytes;
        node->subtree_bytes = node->bytes;
        node->subtree_payload_bytes = node->payload_bytes;
        record(*node);
        return node;
    }
    [[nodiscard]] Ptr branch(std::span<const Ptr> values) const
    {
        auto node = allocate();
        reserve(node->children, branch_capacity);
        node->bytes = checked_sum(node->bytes, branch_capacity * sizeof(Ptr));
        node->children.assign(values.begin(), values.end());
        node->payload_bytes = node->bytes - control_bytes;
        node->subtree_bytes = node->bytes;
        node->subtree_payload_bytes = node->payload_bytes;
        for (const auto& child : values)
        {
            node->size = checked_sum(node->size, child->size);
            node->logical_size = checked_sum(node->logical_size, child->logical_size);
            node->subtree_owned_admission_bytes =
                checked_sum(node->subtree_owned_admission_bytes, child->subtree_owned_admission_bytes);
            node->subtree_bytes = checked_sum(node->subtree_bytes, child->subtree_bytes);
            node->subtree_payload_bytes = checked_sum(node->subtree_payload_bytes, child->subtree_payload_bytes);
            node->subtree_owner_bytes = checked_sum(node->subtree_owner_bytes, child->subtree_owner_bytes);
        }
        node->maximum = values.back()->maximum;
        record(*node);
        return node;
    }
    [[nodiscard]] Replacement pack_leaves(std::span<const Row> rows) const
    {
        if (rows.empty())
            return {{}, 0};
        if (rows.size() <= leaf_capacity)
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
            const auto minimum = child.leaf() ? leaf_capacity / 2 : branch_capacity / 2;
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
            std::array<Row, 2 * leaf_capacity> rows;
            auto end = std::copy(first.rows.begin(), first.rows.end(), rows.begin());
            end = std::copy(second.rows.begin(), second.rows.end(), end);
            return pack_leaves(std::span(rows.begin(), end));
        }
        std::array<Ptr, 2 * branch_capacity> children;
        auto end = std::copy(first.children.begin(), first.children.end(), children.begin());
        end = std::copy(second.children.begin(), second.children.end(), end);
        return pack_branches(rebalance(std::span(children.begin(), end)));
    }
    [[nodiscard]] Replacement rewrite(const Node& node, const Row& key, const Row* replacement, bool expected) const
    {
        if (work_)
            ++work_->visited_pages;
        if (node.leaf())
        {
            const auto at = std::lower_bound(node.rows.begin(), node.rows.end(), key, Less{});
            const bool present = at != node.rows.end() && !Less{}(key, *at);
            if (present != expected)
                throw GeometryError(
                    GeometryErrorCode::InvalidInput, "Ordered metadata edit changed its accepted selector.");
            retire(node);
            std::array<Row, leaf_capacity + 1> rows;
            auto end = std::copy(node.rows.begin(), at, rows.begin());
            if (replacement)
                *end++ = *replacement;
            end = std::copy(present ? std::next(at) : at, node.rows.end(), end);
            return pack_leaves(std::span(rows.begin(), end));
        }
        auto at = std::lower_bound(node.children.begin(), node.children.end(), key,
            [](const Ptr& value, const Row& target) { return Less{}(*value->maximum, target); });
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
    Admission admission_;
    PageWork* work_;
};
} // namespace SimpleFluid::PLIC::details::geometry_demand
