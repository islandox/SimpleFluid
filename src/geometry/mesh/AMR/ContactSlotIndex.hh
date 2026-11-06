/** @file ContactSlotIndex.hh
 * @brief Persistent lookup through bounded numeric slot buckets.
 */
#pragma once
#include "ContactSlotPages.hh"
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Numeric locators in bounded hash buckets. A lookup uses its caller's
 * complete value equality, including genuine 64-bit hash collisions. */
class ContactSlotIndex
{
    static constexpr std::size_t bits = 4, width = std::size_t(1) << bits;
    static constexpr auto depth_limit = std::numeric_limits<std::uint64_t>::digits / bits;
    using Slot = std::size_t;
    struct Node
    {
        std::array<Slot, width> entries{};
        Slot next = 0;
        bool branch = false;
    };

public:
    explicit ContactSlotIndex(std::shared_ptr<Budget> arena) : pages_(std::move(arena)) {}
    ContactSlotIndex(const ContactSlotIndex& source, std::shared_ptr<Budget> arena)
        : pages_(source.pages_, std::move(arena)), root_(source.root_), free_(source.free_)
    {}

    template<class Matches> [[nodiscard]] std::optional<Slot> find(std::uint64_t digest, Matches&& matches) const
    {
        auto current = root_;
        std::size_t depth = 0;
        while (current && node(current).branch)
            current = node(current).entries[digit(digest, depth++)];
        for (; current; current = node(current).next)
            for (const auto entry : node(current).entries)
                if (entry && matches(entry - 1))
                    return entry - 1;
        return {};
    }

    template<class DigestOf> void insert(Slot id, const DigestOf& digest_of)
    {
        if (id == std::numeric_limits<Slot>::max())
            throw std::invalid_argument("Contact record index extent overflows.");
        if (!root_)
            root_ = make_node();
        insert(root_, 0, id + 1, digest_of(id), digest_of);
    }

    void erase(std::uint64_t digest, Slot id)
    {
        auto current = root_;
        std::size_t depth = 0;
        std::array<Slot, depth_limit> parents{};
        while (current && node(current).branch)
        {
            parents[depth] = current;
            current = node(current).entries[digit(digest, depth++)];
        }
        auto previous = Slot(0);
        for (; current; previous = current, current = node(current).next)
            for (std::size_t i = 0; i < width; ++i)
            {
                const auto entry = node(current).entries[i];
                if (!entry || entry - 1 != id)
                    continue;
                edit(current).entries[i] = 0;
                if (std::ranges::any_of(node(current).entries, [](auto value) { return value != 0; }))
                    return;
                erase_empty(current, previous, std::span(parents).first(depth), digest);
                return;
            }
        throw std::logic_error("Contact record index lacks its active slot.");
    }

    [[nodiscard]] std::size_t storage_bytes() const noexcept { return pages_.storage_bytes(); }
    [[nodiscard]] std::size_t overlap_bytes() const noexcept { return pages_.overlap_bytes(); }
    void finish_overlap() noexcept { pages_.finish_overlap(); }
    [[nodiscard]] std::size_t copied_pages() const noexcept { return pages_.copied_pages(); }
    [[nodiscard]] std::size_t split_records() const noexcept { return split_records_; }
    template<class Visit> void visit_pages(Visit&& visit) const { pages_.visit_pages(visit); }

private:
    [[nodiscard]] const Node& node(Slot id) const { return pages_[id - 1]; }
    [[nodiscard]] Node& edit(Slot id) { return pages_.edit(id - 1); }
    [[nodiscard]] Slot make_node()
    {
        if (free_)
        {
            const auto result = free_;
            free_ = node(result).next;
            edit(result) = {};
            return result;
        }
        const auto index = pages_.size();
        pages_.edit(index) = {};
        return index + 1;
    }
    void free_node(Slot id)
    {
        edit(id) = {};
        edit(id).next = free_;
        free_ = id;
    }
    static std::size_t digit(std::uint64_t digest, std::size_t depth) noexcept
    {
        return std::size_t(digest >> (depth * bits)) & (width - 1);
    }

    template<class DigestOf>
    void insert(Slot current, std::size_t depth, Slot entry, std::uint64_t digest, const DigestOf& digest_of)
    {
        if (node(current).branch)
        {
            const auto slot = digit(digest, depth);
            auto child = node(current).entries[slot];
            if (!child)
            {
                child = make_node();
                edit(current).entries[slot] = child;
            }
            insert(child, depth + 1, entry, digest, digest_of);
            return;
        }
        const auto& before = node(current);
        if (const auto vacancy = std::ranges::find(before.entries, 0); vacancy != before.entries.end())
        {
            const auto slot = std::size_t(vacancy - before.entries.begin());
            edit(current).entries[slot] = entry;
            return;
        }
        if (depth == depth_limit)
        {
            auto next = before.next;
            if (!next)
            {
                next = make_node();
                edit(current).next = next;
            }
            // Handle adversarial full-digest collisions without recursion
            // proportional to the number of colliding keys.
            while (std::ranges::none_of(node(next).entries, [](auto value) { return value == 0; }))
            {
                auto following = node(next).next;
                if (!following)
                {
                    following = make_node();
                    edit(next).next = following;
                }
                next = following;
            }
            insert(next, depth, entry, digest, digest_of);
            return;
        }
        const auto old = before.entries;
        edit(current) = Node{{}, 0, true};
        split_records_ += old.size();
        for (const auto value : old)
            insert(current, depth, value, digest_of(value - 1), digest_of);
        insert(current, depth, entry, digest, digest_of);
    }

    void erase_empty(Slot current, Slot previous, std::span<const Slot> parents, std::uint64_t digest)
    {
        if (previous)
        {
            edit(previous).next = node(current).next;
            free_node(current);
            return;
        }
        if (const auto following = node(current).next)
        {
            const auto replacement = node(following);
            edit(current) = replacement;
            free_node(following);
            return;
        }
        free_node(current);
        for (auto depth = parents.size(); depth; --depth)
        {
            const auto parent = parents[depth - 1];
            edit(parent).entries[digit(digest, depth - 1)] = 0;
            if (std::ranges::any_of(node(parent).entries, [](auto value) { return value != 0; }))
                return;
            free_node(parent);
        }
        root_ = 0;
    }

    ContactSlotPages<Node> pages_;
    Slot root_ = 0, free_ = 0;
    std::size_t split_records_ = 0;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
