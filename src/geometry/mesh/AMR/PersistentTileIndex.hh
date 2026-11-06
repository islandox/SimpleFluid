/** @file PersistentTileIndex.hh
 * @brief Copy-on-write support/emission metadata for active incidence tiles.
 */
#pragma once
#include "geometry/mesh/AMR/ContactPayloadCensus.hh"
#include "geometry/mesh/AMR/PersistentOrderedPages.hh"

#include <cstdint>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Persistent support/emission order, independent of physical tile body
 * ownership. Forks share the complete root. Entry payload owners travel with
 * their rows, and all edits allocate only affected ordered pages.
 *
 * New page capacity is charged to the supplied arena. The enclosing
 * incidence controller carries the inherited page charge using
 * inherited_metadata_bytes(). Each fork receives a fresh arena for this
 * index lineage. A raw Budget must outlive all its pages. */
template<class Entry> class PersistentTileIndex
{
public:
    using Key = std::pair<std::size_t, std::uint64_t>;

private:
    struct Row
    {
        Key key{};
        Entry value{};
    };
    struct Less
    {
        bool operator()(const Row& left, const Row& right) const noexcept { return left.key < right.key; }
    };
    using Tree = persistent_ordered_detail::OrderedPages<Row, Less>;
    using Root = typename Tree::Ptr;
    using PageCharge = persistent_ordered_detail::PageCharge;

public:
    explicit PersistentTileIndex(Budget& budget, std::shared_ptr<Budget> keeper = {})
        : arena_(keeper ? std::move(keeper) : std::shared_ptr<Budget>(std::shared_ptr<Budget>{}, &budget))
    {
        if (arena_.get() != &budget)
            throw std::invalid_argument("Tile index arena does not own its budget.");
    }

    PersistentTileIndex(const PersistentTileIndex&) = delete;
    PersistentTileIndex& operator=(const PersistentTileIndex&) = delete;

    void fork_from(const PersistentTileIndex& source)
    {
        if (root_ || batch_live_ || source.batch_live_)
            throw std::logic_error("Tile metadata fork needs a quiescent source and empty destination.");
        root_ = source.root_;
    }

    [[nodiscard]] std::size_t size() const noexcept { return root_ ? root_->size : 0; }
    [[nodiscard]] std::size_t metadata_bytes() const noexcept { return root_ ? root_->subtree_bytes : 0; }
    [[nodiscard]] std::size_t inherited_metadata_bytes() const noexcept
    {
        if (!root_ || root_->charge.arena != arena_)
            return metadata_bytes();
        return root_->subtree_bytes - root_->same_arena_bytes;
    }
    [[nodiscard]] Key key_at(std::size_t ordinal) const { return row_at(ordinal).key; }
    [[nodiscard]] const Entry& value_at(std::size_t ordinal) const { return row_at(ordinal).value; }

    [[nodiscard]] std::size_t lower_bound(Key key) const noexcept
    {
        const auto* node = root_.get();
        std::size_t prefix = 0;
        while (node && !node->leaf())
        {
            auto child = node->children.begin();
            while (child != node->children.end() && (*child)->maximum->key < key)
            {
                prefix += (*child)->size;
                ++child;
            }
            node = child == node->children.end() ? nullptr : child->get();
        }
        if (!node)
            return prefix;
        const auto row = std::lower_bound(node->rows.begin(), node->rows.end(), key,
            [](const Row& value, const Key& target) { return value.key < target; });
        return prefix + std::size_t(row - node->rows.begin());
    }

    [[nodiscard]] const Entry* find(Key key) const noexcept
    {
        const auto* row = lower_row(root_, key);
        return row && row->key == key ? &row->value : nullptr;
    }

    [[nodiscard]] const Entry& at(Key key) const
    {
        if (const auto* value = find(key))
            return *value;
        throw std::out_of_range("Missing active tile metadata.");
    }

    /** Edit a private copy. The callback must replace any mutable payload
     * owner before changing its contents. A failed update preserves the root. */
    template<class Edit> void update(Key key, Edit&& edit)
    {
        if (batch_live_)
            throw std::logic_error("Tile metadata update overlaps a staged batch.");
        Row replacement{key, at(key)};
        edit(replacement.value);
        root_ = Tree(arena_).edit(root_, replacement, &replacement, true);
    }

    template<class Visitor> void visit_borrowed_payload(Visitor&& visit, bool include_object = false) const
    {
        if (include_object)
            contact_payload_object(visit, this, ContactPayloadRole::TileIndexObject, sizeof(*this));
        Tree::visit_pages(root_,
            [&](const Root& page)
            {
                contact_payload_object(visit, page.get(), ContactPayloadRole::TileIndexPageObject, page->charge.bytes);
            });
    }

    class Batch
    {
    public:
        explicit Batch(PersistentTileIndex& owner) : owner_(owner), scratch_(owner.arena_)
        {
            if (owner_.batch_live_)
                throw std::logic_error("Tile metadata batch already active.");
            owner_.batch_live_ = true;
        }

        ~Batch()
        {
            if (!published_)
                owner_.batch_live_ = false;
        }
        Batch(const Batch&) = delete;
        Batch& operator=(const Batch&) = delete;

        void replace_support(std::size_t support)
        {
            ensure_open();
            scratch_.append(retired_supports_);
            retired_supports_.push_back(support);
        }

        void insert(Key key, Entry value)
        {
            ensure_open();
            scratch_.append(inserted_);
            inserted_.push_back({key, std::move(value)});
        }

        void publish(std::size_t tile_limit = std::numeric_limits<std::size_t>::max())
        {
            ensure_open();
            std::sort(retired_supports_.begin(), retired_supports_.end());
            retired_supports_.erase(
                std::unique(retired_supports_.begin(), retired_supports_.end()), retired_supports_.end());
            std::sort(inserted_.begin(), inserted_.end(), Less{});
            if (std::adjacent_find(inserted_.begin(), inserted_.end(),
                    [](const Row& a, const Row& b) { return a.key == b.key; }) != inserted_.end())
                throw std::invalid_argument("Duplicate staged tile metadata key.");

            auto root = owner_.root_;
            const Tree tree(owner_.arena_);
            if (!root)
            {
                if (inserted_.size() > tile_limit)
                    throw std::invalid_argument("Active tile count exceeds its limit.");
                root = tree.build(inserted_);
            }
            else
            {
                for (const auto support : retired_supports_)
                    while (const auto* row = lower_row(root, {support, 0}))
                    {
                        if (row->key.first != support)
                            break;
                        const Row key{row->key, {}};
                        root = tree.edit(root, key, nullptr, true);
                    }
                for (const auto& row : inserted_)
                    root = tree.edit(root, row, &row, false);
            }
            if (root && root->size > tile_limit)
                throw std::invalid_argument("Active tile count exceeds its limit.");
            owner_.root_ = std::move(root);
            inserted_.clear();
            owner_.batch_live_ = false;
            published_ = true;
        }

    private:
        void ensure_open() const
        {
            if (published_)
                throw std::logic_error("Tile metadata batch is already published.");
        }

        PersistentTileIndex& owner_;
        PageCharge scratch_;
        std::vector<std::size_t> retired_supports_;
        std::vector<Row> inserted_;
        bool published_ = false;
    };

private:
    [[nodiscard]] static const Row* lower_row(const Root& root, Key key) noexcept
    {
        return Tree::lower_bound(root, [&](const Row& value) { return value.key < key; });
    }

    [[nodiscard]] const Row& row_at(std::size_t ordinal) const
    {
        if (ordinal >= size())
            throw std::out_of_range("Tile metadata ordinal is outside the publication.");
        const auto* node = root_.get();
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

    std::shared_ptr<Budget> arena_;
    Root root_;
    bool batch_live_ = false;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
