/** @file ContactSlotPages.hh
 * @brief Copy-on-write radix pages for contact metadata slots.
 */
#pragma once
#include "ContactGroupAllocation.hh"
#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** A private editable selection and its immutable shared roots. A fork
 * admits all physically shared pages in one charge before touching a slot.
 * Copying a path transfers its replaced page charges to the editing arena.
 * No locator refers to a previous table or contact controller. */
template<class T> class ContactSlotPages
{
    static constexpr std::size_t bits = 4, width = std::size_t(1) << bits, mask = width - 1;
    struct Node;
    using Pointer = std::shared_ptr<Node>;
    struct Node
    {
        ContactGroupAllocation admission;
        std::size_t subtree_bytes = 0;
        bool branch;
        Node(ContactGroupAllocation charge, bool branch_)
            : admission(std::move(charge)), subtree_bytes(admission.bytes()), branch(branch_) {}
    };
    struct Branch : Node
    {
        std::array<Pointer, width> children{};
        explicit Branch(ContactGroupAllocation charge) : Node(std::move(charge), true) {}
    };
    struct Leaf : Node
    {
        std::array<T, width> values{};
        explicit Leaf(ContactGroupAllocation charge) : Node(std::move(charge), false) {}
    };

public:
    explicit ContactSlotPages(std::shared_ptr<Budget> arena) : imported_(std::move(arena)) {}
    ContactSlotPages(const ContactSlotPages& source, std::shared_ptr<Budget> arena)
        : imported_(std::move(arena)), depth_(source.depth_), extent_(source.extent_), keep_overlap_(true)
    {
        if (source.root_)
            imported_.add(source.root_->subtree_bytes);
        root_ = source.root_;
    }
    ContactSlotPages(ContactSlotPages&&) noexcept = default;
    ContactSlotPages(const ContactSlotPages&) = delete;

    [[nodiscard]] std::size_t size() const noexcept { return extent_; }
    [[nodiscard]] std::size_t storage_bytes() const noexcept { return root_ ? root_->subtree_bytes : 0; }
    [[nodiscard]] std::size_t copied_pages() const noexcept { return copied_pages_; }
    [[nodiscard]] std::size_t allocated_pages() const noexcept { return allocated_pages_; }

    [[nodiscard]] std::size_t overlap_bytes() const noexcept { return replaced_imports_; }
    void finish_overlap() noexcept
    {
        imported_.release(replaced_imports_);
        replaced_imports_ = 0;
        keep_overlap_ = false;
    }
    [[nodiscard]] const T* find(std::size_t slot) const noexcept
    {
        if (slot >= extent_)
            return nullptr;
        auto* node = root_.get();
        for (auto depth = depth_; node && depth; --depth)
            node = static_cast<const Branch&>(*node).children[digit(slot, depth)].get();
        return node ? &static_cast<const Leaf&>(*node).values[digit(slot, 0)] : nullptr;
    }
    [[nodiscard]] const T& operator[](std::size_t slot) const
    {
        const auto* result = find(slot);
        if (!result)
            throw std::out_of_range("Unknown contact metadata slot.");
        return *result;
    }
    [[nodiscard]] T& edit(std::size_t slot)
    {
        if (slot == std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("Contact metadata slot overflows.");
        while (depth_ < (std::numeric_limits<std::size_t>::digits - 1) / bits &&
               (slot >> (bits * (depth_ + 1))))
        {
            auto parent = allocate<Branch>();
            parent->children[0] = std::move(root_);
            parent->subtree_bytes += parent->children[0] ? parent->children[0]->subtree_bytes : 0;
            root_ = std::move(parent);
            ++depth_;
        }
        auto& value = edit(root_, depth_, slot);
        extent_ = std::max(extent_, slot + 1);
        return value;
    }
    template<class Visit> void visit_pages(Visit&& visit) const { visit_pages(root_.get(), visit); }

private:
    static std::size_t digit(std::size_t slot, std::size_t depth) noexcept
    {
        return (slot >> (depth * bits)) & mask;
    }
    template<class Page> std::shared_ptr<Page> allocate()
    {
        ContactGroupAllocation charge(imported_.keeper());
        charge.add(sizeof(Page) + 128);
        auto result = geometry_detail::domain_shared<geometry_detail::AllocationDomain::ContactHalo, Page>(
            std::move(charge));
        ++allocated_pages_;
        return result;
    }
    template<class Page> Page& writable(Pointer& current)
    {
        if (current && current.use_count() == 1 &&
            &current->admission.arena() == &imported_.arena())
            return static_cast<Page&>(*current);
        auto next = allocate<Page>();
        if (current)
        {
            if constexpr (std::is_same_v<Page, Leaf>)
                next->values = static_cast<const Leaf&>(*current).values;
            else
                next->children = static_cast<const Branch&>(*current).children;
            next->subtree_bytes += current->subtree_bytes - current->admission.bytes();
            if (&current->admission.arena() != &imported_.arena())
            {
                if (keep_overlap_)
                    replaced_imports_ += current->admission.bytes();
                else
                    imported_.release(current->admission.bytes());
            }
            ++copied_pages_;
        }
        auto* result = next.get();
        current = std::move(next);
        return *result;
    }
    T& edit(Pointer& current, std::size_t depth, std::size_t slot)
    {
        if (!depth)
            return writable<Leaf>(current).values[digit(slot, 0)];
        auto& branch = writable<Branch>(current);
        auto& child = branch.children[digit(slot, depth)];
        const auto before = child ? child->subtree_bytes : 0;
        try
        {
            auto& result = edit(child, depth - 1, slot);
            branch.subtree_bytes = branch.subtree_bytes - before + child->subtree_bytes;
            return result;
        }
        catch (...)
        {
            branch.subtree_bytes = branch.subtree_bytes - before + (child ? child->subtree_bytes : 0);
            throw;
        }
    }
    template<class Visit> static void visit_pages(const Node* current, Visit& visit)
    {
        if (!current)
            return;
        visit(current, current->branch, current->admission.bytes());
        if (current->branch)
            for (const auto& child : static_cast<const Branch&>(*current).children)
                visit_pages(child.get(), visit);
    }

    ContactGroupAllocation imported_;
    Pointer root_;
    std::size_t depth_ = 0, extent_ = 0, copied_pages_ = 0, allocated_pages_ = 0, replaced_imports_ = 0;
    bool keep_overlap_ = false;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
