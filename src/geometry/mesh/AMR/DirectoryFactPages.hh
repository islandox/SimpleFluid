/** @file DirectoryFactPages.hh
 * @brief Immutable sharded ancestry facts and separate numeric address output.
 */
#pragma once
#include "geometry/PLIC/GeometryPacketCodec.hh"
#include "geometry/mesh/AMR/ContactGroupAllocation.hh"
#include "geometry/mesh/AMR/PersistentOrderedPages.hh"

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct DirectoryRegistrationView
{
    std::span<const char> key;
    int owner;
    std::uint64_t revision, address;
    bool topology_revision = false;
};
struct DirectoryFactView
{
    std::span<const char> key;
    int owner;
    std::uint64_t revision, address;
};

/** Fact buffers belong to their immutable leaf page. A candidate copies at
 * most the affected 64-row pages and branch paths. Numeric addresses are
 * values in shard-key order; they never determine an owner's cell ordinal. */
class DirectoryFactPages
{
    struct Fact
    {
        AdaptiveBytes key;
        int owner;
        bool topology_revision;
        std::uint64_t literal_revision;
    };
    struct Less
    {
        bool operator()(const Fact& a, const Fact& b) const noexcept { return a.key < b.key; }
    };
    struct Buffers
    {
        static constexpr bool enabled = true;
        static std::size_t bytes(const Fact& row) noexcept { return row.key.capacity(); }
    };
    using Tree = persistent_ordered_detail::OrderedPages<Fact, Less, false, persistent_ordered_detail::OrderedPageWork,
        persistent_ordered_detail::NoRowWeights, persistent_ordered_detail::leaf_capacity, Buffers>;
    using Root = Tree::Ptr;
    using Charge = persistent_ordered_detail::PageCharge;

public:
    struct Work
    {
        std::size_t created_pages = 0, created_page_bytes = 0, fact_edits = 0;
    };

    /** completed_private_edits is diagnostic progress on a discarded private
     * candidate, including on rejection. It is never a publication receipt. */
    [[nodiscard]] static DirectoryFactPages reconcile(const DirectoryFactPages* accepted,
        std::span<const DirectoryRegistrationView> rows, std::shared_ptr<Budget> arena, std::uint64_t topology,
        std::size_t* completed_private_edits = nullptr)
    {
        if (!PLIC::geometry_encoding_capacity_profile_supported() ||
            (accepted && accepted->imported_.keeper() == arena))
            throw std::invalid_argument("Directory facts require a qualified profile and a fresh arena.");
        if (completed_private_edits)
            *completed_private_edits = 0;
        DirectoryFactPages candidate(std::move(arena), topology);
        if (accepted)
        {
            candidate.imported_.add(accepted->fact_bytes());
            candidate.root_ = accepted->root_;
        }
        const auto before = [](std::span<const char> a, std::span<const char> b)
        { return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()); };
        std::span<const char> previous;
        for (std::size_t i = 0; i < rows.size(); ++i)
        {
            if (rows[i].key.empty() || rows[i].owner < 0 || (i && !before(previous, rows[i].key)) ||
                (rows[i].topology_revision && rows[i].revision != topology))
                throw std::invalid_argument("Invalid or duplicate validated directory facts.");
            previous = rows[i].key;
        }

        const Tree tree(candidate.imported_.keeper(), &candidate.work_pages_);
        if (!accepted)
        {
            Charge scratch(candidate.imported_.keeper());
            std::vector<Fact> staged;
            scratch.add(persistent_ordered_detail::checked_product(rows.size(), sizeof(Fact)));
            staged.reserve(rows.size());
            if (staged.capacity() != rows.size())
                throw std::invalid_argument("Directory bootstrap fact capacity differs from admission.");
            for (const auto& row : rows)
            {
                scratch.add(row.key.size());
                auto value = fact(row);
                if (value.key.capacity() != row.key.size())
                    throw std::invalid_argument("Directory bootstrap key capacity differs from admission.");
                staged.push_back(std::move(value));
            }
            candidate.root_ = tree.build(staged);
            candidate.work_.fact_edits = rows.size();
            if (completed_private_edits)
                *completed_private_edits = rows.size();
        }
        else
        {
            std::size_t old = 0;
            for (const auto& row : rows)
            {
                while (old < accepted->size() && before(accepted->fact_at(old).key, row.key))
                {
                    candidate.root_ = tree.edit(candidate.root_, accepted->fact_at(old++), nullptr, true);
                    ++candidate.work_.fact_edits;
                    if (completed_private_edits)
                        *completed_private_edits = candidate.work_.fact_edits;
                }
                const auto* prior = old < accepted->size() && std::ranges::equal(accepted->fact_at(old).key, row.key)
                                        ? &accepted->fact_at(old)
                                        : nullptr;
                if (prior)
                    ++old;
                const auto revision = row.topology_revision ? 0 : row.revision;
                if (prior && prior->owner == row.owner && prior->topology_revision == row.topology_revision &&
                    prior->literal_revision == revision)
                    continue;
                Charge scratch(candidate.imported_.keeper());
                scratch.add(row.key.size());
                auto replacement = fact(row);
                if (replacement.key.capacity() != row.key.size())
                    throw std::invalid_argument("Directory candidate key capacity differs from admission.");
                candidate.root_ = tree.edit(candidate.root_, replacement, &replacement, prior != nullptr);
                ++candidate.work_.fact_edits;
                if (completed_private_edits)
                    *completed_private_edits = candidate.work_.fact_edits;
            }
            while (old < accepted->size())
            {
                candidate.root_ = tree.edit(candidate.root_, accepted->fact_at(old++), nullptr, true);
                ++candidate.work_.fact_edits;
                if (completed_private_edits)
                    *completed_private_edits = candidate.work_.fact_edits;
            }
        }
        if (candidate.size() != rows.size())
            throw std::logic_error("Directory numeric projection count differs from its exact facts.");
        candidate.uniform_address_ = rows.empty() ? std::numeric_limits<std::uint64_t>::max() : rows.front().address;
        const bool uniform = std::all_of(
            rows.begin(), rows.end(), [&](const auto& row) { return row.address == candidate.uniform_address_; });
        if (!uniform)
        {
            candidate.numeric_.add(persistent_ordered_detail::checked_product(rows.size(), sizeof(std::uint64_t)));
            candidate.addresses_.reserve(rows.size());
            if (candidate.addresses_.capacity() != rows.size())
                throw std::invalid_argument("Directory numeric projection capacity differs from admission.");
            for (const auto& row : rows)
                candidate.addresses_.push_back(row.address);
        }
        candidate.work_.created_pages = candidate.work_pages_.copied_pages;
        candidate.work_.created_page_bytes = candidate.work_pages_.copied_bytes;
        return candidate;
    }

    DirectoryFactPages(DirectoryFactPages&&) noexcept = default;
    DirectoryFactPages(const DirectoryFactPages&) = delete;
    DirectoryFactPages& operator=(const DirectoryFactPages&) = delete;

    [[nodiscard]] std::size_t size() const noexcept { return root_ ? root_->size : 0; }
    [[nodiscard]] std::size_t fact_bytes() const noexcept { return root_ ? root_->subtree_bytes : 0; }
    [[nodiscard]] std::size_t numeric_bytes() const noexcept { return numeric_.bytes(); }
    [[nodiscard]] std::size_t storage_bytes() const noexcept { return fact_bytes() + numeric_bytes(); }
    [[nodiscard]] std::size_t shared_fact_bytes() const noexcept
    {
        return root_ ? root_->subtree_bytes - (root_->charge.arena == imported_.keeper() ? root_->same_arena_bytes : 0)
                     : 0;
    }
    [[nodiscard]] std::size_t overlap_bytes() const noexcept { return imported_.bytes() - shared_fact_bytes(); }
    void finish_overlap() noexcept { imported_.release(overlap_bytes()); }
    [[nodiscard]] const Work& work() const noexcept { return work_; }
    [[nodiscard]] std::shared_ptr<const void> root_owner() const noexcept { return root_; }

    [[nodiscard]] std::optional<DirectoryFactView> find(std::span<const char> key) const noexcept
    {
        const auto* node = root_.get();
        std::size_t ordinal = 0;
        const auto precedes = [&](const Fact& row)
        { return std::lexicographical_compare(row.key.begin(), row.key.end(), key.begin(), key.end()); };
        while (node && !node->leaf())
        {
            auto child = node->children.begin();
            while (child != node->children.end() && precedes(*(*child)->maximum))
                ordinal += (*child++)->size;
            node = child == node->children.end() ? nullptr : child->get();
        }
        if (!node)
            return {};
        const auto found = std::partition_point(node->rows.begin(), node->rows.end(), precedes);
        if (found == node->rows.end() || !std::ranges::equal(found->key, key))
            return {};
        ordinal += found - node->rows.begin();
        return view(*found, ordinal);
    }

    /** Explicit full diagnostics. Mutation and admission use cached totals. */
    template<class Visit> void visit_storage(const Visit& visit) const
    {
        Tree::visit_pages(root_,
            [&](const Root& page)
            {
                visit(page.get(), sizeof(typename Tree::Node), page->charge.bytes, false);
                if (page->rows.capacity())
                    visit(page->rows.data(), page->rows.capacity() * sizeof(Fact), 0, false);
                if (page->children.capacity())
                    visit(page->children.data(), page->children.capacity() * sizeof(Root), 0, false);
                for (const auto& row : page->rows)
                    if (row.key.capacity())
                        visit(row.key.data(), row.key.capacity(), 0, false);
            });
        if (addresses_.capacity())
            visit(addresses_.data(), numeric_bytes(), numeric_bytes(), true);
    }
    [[nodiscard]] const void* page_identity(std::span<const char> key) const noexcept
    {
        const auto* page = root_.get();
        while (page && !page->leaf())
        {
            const auto child = std::partition_point(page->children.begin(), page->children.end(),
                [&](const auto& value)
                {
                    const auto& maximum = value->maximum->key;
                    return std::lexicographical_compare(maximum.begin(), maximum.end(), key.begin(), key.end());
                });
            page = child == page->children.end() ? nullptr : child->get();
        }
        return find(key) ? page : nullptr;
    }

private:
    explicit DirectoryFactPages(std::shared_ptr<Budget> arena, std::uint64_t topology)
        : imported_(arena), numeric_(std::move(arena)), topology_(topology)
    {
    }
    static Fact fact(const DirectoryRegistrationView& row)
    {
        return {AdaptiveBytes(row.key.begin(), row.key.end()), row.owner, row.topology_revision,
            row.topology_revision ? 0 : row.revision};
    }
    [[nodiscard]] const Fact& fact_at(std::size_t ordinal) const noexcept
    {
        auto* node = root_.get();
        while (!node->leaf())
        {
            auto child = node->children.begin();
            while (ordinal >= (*child)->size)
            {
                ordinal -= (*child)->size;
                ++child;
            }
            node = child->get();
        }
        return node->rows[ordinal];
    }
    [[nodiscard]] DirectoryFactView view(const Fact& row, std::size_t ordinal) const noexcept
    {
        return {row.key, row.owner, row.topology_revision ? topology_ : row.literal_revision,
            addresses_.empty() ? uniform_address_ : addresses_[ordinal]};
    }

    ContactGroupAllocation imported_, numeric_;
    Root root_;
    std::vector<std::uint64_t> addresses_;
    std::uint64_t topology_, uniform_address_ = std::numeric_limits<std::uint64_t>::max();
    persistent_ordered_detail::OrderedPageWork work_pages_;
    Work work_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
