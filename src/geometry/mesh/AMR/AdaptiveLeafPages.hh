/** @file AdaptiveLeafPages.hh
 * @brief Ordered persistent pages for immutable owner-local AMR records.
 */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"

#include <array>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct Budget;
struct AdaptiveLeafPageBuilder;

/** A persistent, sized range of canonical leaves in complete typed-key order.
 * Each record stores its key once. Interior bounds borrow immutable child keys.
 * Keep this value or a copy alive while using its iterators or references. */
class AdaptiveLeafPages
{
    struct Node;
    using NodePtr = std::shared_ptr<const Node>;

public:
    static constexpr std::size_t leaf_capacity = 64, branch_capacity = 32;
    struct Storage
    {
        std::size_t nodes = 0, leaf_pages = 0, leaf_rows = 0, leaf_capacity = 0;
        std::size_t maximum_leaf_rows = 0, maximum_fanout = 0, retained_bytes = 0;
    };
    struct Edit
    {
        AdaptiveCellKey key;
        std::optional<std::uint64_t> expected_revision; ///< Missing only for a new key.
        std::optional<AdaptiveLeaf> leaf;               ///< Missing only for a retirement.
    };
    struct EditView
    {
        const AdaptiveCellKey& key;
        std::optional<std::uint64_t> expected_revision;
        const AdaptiveLeaf* leaf;
    };
    struct EditStats
    {
        std::size_t new_nodes = 0, new_leaf_pages = 0, copied_rows = 0;
    };

    using LeafPageVisitor = std::function<void(std::span<const AdaptiveLeaf>)>;
    struct PageChanges
    {
        LeafPageVisitor retire, insert;
    };

    class const_iterator
    {
    public:
        using value_type = AdaptiveLeaf;
        using difference_type = std::ptrdiff_t;
        using pointer = const AdaptiveLeaf*;
        using reference = const AdaptiveLeaf&;
        using iterator_category = std::forward_iterator_tag;

        const_iterator() = default;
        [[nodiscard]] reference operator*() const noexcept;
        [[nodiscard]] pointer operator->() const noexcept { return &**this; }
        const_iterator& operator++() noexcept;
        const_iterator operator++(int) noexcept
        {
            auto previous = *this;
            ++*this;
            return previous;
        }
        [[nodiscard]] bool operator==(const const_iterator& other) const noexcept
        {
            return leaf_ == other.leaf_ && row_ == other.row_;
        }

    private:
        friend class AdaptiveLeafPages;
        explicit const_iterator(const Node*) noexcept;
        void descend(const Node*) noexcept;
        struct Branch
        {
            const Node* node;
            std::size_t next_child;
        };
        // At least 16 children per non-root interior page bounds the depth by
        // the addressable leaf count, including the root and one leaf level.
        std::array<Branch, std::numeric_limits<std::size_t>::digits / 4 + 2> path_{};
        const Node* leaf_ = nullptr;
        std::size_t depth_ = 0, row_ = 0;
    };

    AdaptiveLeafPages() = default;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept { return !root_; }
    [[nodiscard]] const_iterator begin() const noexcept { return const_iterator(root_.get()); }
    [[nodiscard]] const_iterator end() const noexcept { return {}; }
    [[nodiscard]] const AdaptiveLeaf* find(const AdaptiveCellKey&) const;
    [[nodiscard]] bool contains(const AdaptiveCellKey& key) const { return find(key) != nullptr; }
    [[nodiscard]] const AdaptiveLeaf& at(const AdaptiveCellKey&) const;
    [[nodiscard]] std::optional<std::size_t> ordinal_of(const AdaptiveCellKey&) const;
    [[nodiscard]] const AdaptiveLeaf& at_ordinal(std::size_t) const;

    /** Build from strictly increasing typed keys. The arena charges exclusive
     * page storage, copied record buffers and simultaneous builder scratch.
     * Shared recipe and certificate bodies belong to their separate owners. */
    [[nodiscard]] static AdaptiveLeafPages build(std::span<const AdaptiveLeaf>, std::shared_ptr<Budget> arena);
    /** Conservative capacity charge for buffers owned directly by this row.
     * The inline record and shared provider and certificate owners are separate. */
    [[nodiscard]] static std::size_t buffer_bytes(const AdaptiveLeaf&);
    /** Consume rows in strictly increasing order. Transfer the supplied total
     * buffer charge to their pages; the arena also admits new slots, headers
     * and scratch. On failure the callback may have consumed a prefix. */
    [[nodiscard]] static AdaptiveLeafPages build_moving(std::size_t count,
        const std::function<AdaptiveLeaf&&()>& next, std::shared_ptr<Budget> arena, std::size_t moved_buffer_bytes);
    /** Apply strictly ordered unique edits to a private root. A rejected edit
     * leaves the source and its allocation charges intact. */
    [[nodiscard]] static AdaptiveLeafPages edit(const AdaptiveLeafPages&, std::span<const Edit>,
        std::shared_ptr<Budget> arena, EditStats* = nullptr);
    /** Apply a borrowed edit selection. Keep every referenced row and key
     * alive and quiescent until the private root has been constructed.
     * Page-change callbacks require a fresh, empty arena so physical source
     * and newly allocated pages have disjoint admission owners. */
    [[nodiscard]] static AdaptiveLeafPages edit_borrowed(const AdaptiveLeafPages&, std::span<const EditView>,
        std::shared_ptr<Budget> arena, EditStats* = nullptr, const PageChanges* = nullptr);
    void visit_leaf_pages(const LeafPageVisitor&) const;
    /** Visit each page owner while this value is pinned. The charge covers the
     * page and exclusive buffers, and excludes shared provider bodies. */
    void visit_pages(const std::function<void(const void*, std::size_t)>&) const;
    [[nodiscard]] Storage storage() const;
    [[nodiscard]] std::size_t retained_bytes() const noexcept;
    [[nodiscard]] const void* page_identity(const AdaptiveCellKey&) const;

private:
    friend struct AdaptiveLeafPageBuilder;
    explicit AdaptiveLeafPages(NodePtr root) : root_(std::move(root)) {}
    [[nodiscard]] const Node* find_page(const AdaptiveCellKey&, std::size_t& prefix) const;
    NodePtr root_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
