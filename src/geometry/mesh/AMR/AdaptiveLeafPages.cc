/** @file AdaptiveLeafPages.cc
 * @brief Immutable ordered page construction, traversal and sparse edits.
 */
#include "geometry/mesh/AMR/AdaptiveLeafPages.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
class PageCharge
{
public:
    explicit PageCharge(std::shared_ptr<Budget> arena) : arena_(std::move(arena)) {}
    PageCharge(PageCharge&& other) noexcept
        : arena_(std::move(other.arena_)), bytes_(std::exchange(other.bytes_, 0))
    {
    }
    ~PageCharge()
    {
        if (arena_)
            arena_->used -= bytes_;
    }
    void add(std::size_t bytes)
    {
        arena_->add(bytes);
        bytes_ += bytes;
    }
    void transfer_to(PageCharge& destination, std::size_t bytes)
    {
        if (arena_ != destination.arena_ || bytes > bytes_)
            throw std::invalid_argument("Moved adaptive row buffers exceed page precharge.");
        bytes_ -= bytes;
        destination.bytes_ += bytes;
    }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] const Budget* arena() const noexcept { return arena_.get(); }

private:
    std::shared_ptr<Budget> arena_;
    std::size_t bytes_ = 0;
};

template<class T> struct ScratchAllocator
{
    using value_type = T;
    Budget* arena;

    explicit ScratchAllocator(Budget& budget) noexcept : arena(&budget) {}
    template<class U> ScratchAllocator(const ScratchAllocator<U>& other) noexcept : arena(other.arena) {}
    T* allocate(std::size_t count)
    {
        if (count > arena->limit / sizeof(T))
            throw std::invalid_argument("Adaptive leaf page scratch exceeds budget.");
        const auto bytes = count * sizeof(T);
        arena->add(bytes);
        try
        {
            return std::allocator<T>{}.allocate(count);
        }
        catch (...)
        {
            arena->used -= bytes;
            throw;
        }
    }
    void deallocate(T* value, std::size_t count) noexcept
    {
        std::allocator<T>{}.deallocate(value, count);
        arena->used -= count * sizeof(T);
    }
    template<class U> bool operator==(const ScratchAllocator<U>& other) const noexcept { return arena == other.arena; }
};

template<class T> using ScratchVector = std::vector<T, ScratchAllocator<T>>;

} // namespace

std::size_t AdaptiveLeafPages::buffer_bytes(const AdaptiveLeaf& leaf)
{
    Budget charge{std::numeric_limits<std::size_t>::max()};
    std::visit(
        [&](const auto& ancestry)
        {
            if constexpr (requires { ancestry.path; })
                charge.add(ancestry.path.capacity() * sizeof(std::uint64_t));
            else if constexpr (requires { ancestry.base.path; })
            {
                charge.add(ancestry.base.path.capacity() * sizeof(std::uint64_t));
                charge.add(ancestry.axial.path.capacity() * sizeof(std::uint64_t));
            }
        },
        leaf.key.ancestry);
    charge.add(leaf.domain.capacity());
    charge.add(leaf.parent_family.capacity());
    charge.add(leaf.surfaces.capacity() * sizeof(AdaptiveSurface));
    for (const auto& surface : leaf.surfaces)
    {
        charge.add(surface.key.capacity());
        charge.add(surface.boundary.capacity());
        charge.add(surface.authority_image.payload_bytes());
    }
    return charge.used;
}

namespace
{
std::size_t groups_for(std::size_t count, std::size_t capacity) noexcept
{
    return count / capacity + (count % capacity != 0);
}

std::size_t group_size(std::size_t count, std::size_t groups, std::size_t group) noexcept
{
    return count / groups + (group < count % groups);
}
} // namespace

struct AdaptiveLeafPages::Node
{
    explicit Node(PageCharge initial) : charge(std::move(initial)) {}
    [[nodiscard]] bool terminal() const noexcept { return children.empty(); }
    [[nodiscard]] std::size_t occupancy() const noexcept { return terminal() ? rows.size() : children.size(); }
    [[nodiscard]] auto lower_bound(const AdaptiveCellKey& key) const
    {
        return std::lower_bound(rows.begin(), rows.end(), key,
            [](const AdaptiveLeaf& row, const AdaptiveCellKey& value) { return row.key < value; });
    }

    PageCharge charge; // Release only after records and child owners.
    std::vector<AdaptiveLeaf> rows;
    std::vector<NodePtr> children;
    const AdaptiveCellKey* maximum = nullptr;
    std::size_t size = 0, subtree_bytes = 0;
};

struct AdaptiveLeafPageBuilder
{
    using Node = AdaptiveLeafPages::Node;
    using NodePtr = AdaptiveLeafPages::NodePtr;
    using Nodes = ScratchVector<NodePtr>;
    using Rows = ScratchVector<const AdaptiveLeaf*>;
    using Stats = AdaptiveLeafPages::EditStats;

    static constexpr auto leaf_capacity = AdaptiveLeafPages::leaf_capacity;
    static constexpr auto branch_capacity = AdaptiveLeafPages::branch_capacity;

    std::shared_ptr<Budget> arena;
    Stats stats;
    const AdaptiveLeafPages::PageChanges* changes = nullptr;

    void retire(const Node& page) const
    {
        if (changes && page.charge.arena() != arena.get())
            changes->retire(page.rows);
    }

    void visit_new(const Node& page) const
    {
        if (page.charge.arena() != arena.get())
            return;
        if (page.terminal())
            changes->insert(page.rows);
        else
            for (const auto& child : page.children)
                visit_new(*child);
    }

    [[nodiscard]] Nodes nodes() const { return Nodes(ScratchAllocator<NodePtr>(*arena)); }
    [[nodiscard]] Rows rows() const { return Rows(ScratchAllocator<const AdaptiveLeaf*>(*arena)); }

    [[nodiscard]] std::shared_ptr<Node> allocate()
    {
        PageCharge charge(arena);
        charge.add(sizeof(Node) + 4 * sizeof(void*));
        auto node = std::make_shared<Node>(std::move(charge));
        ++stats.new_nodes;
        return node;
    }

    template<class Next> [[nodiscard]] NodePtr leaf(std::size_t count, Next& next,
        const AdaptiveCellKey*& previous, PageCharge* moved_buffers)
    {
        auto node = allocate();
        node->charge.add(count * sizeof(AdaptiveLeaf));
        node->rows.reserve(count);
        if (node->rows.capacity() != count)
            throw std::invalid_argument("Adaptive leaf page capacity differs from admission.");
        for (std::size_t i = 0; i < count; ++i)
        {
            auto&& row = next();
            if (previous && !(*previous < row.key))
                throw std::invalid_argument("Adaptive leaf page input is not strictly ordered.");
            const auto bytes = AdaptiveLeafPages::buffer_bytes(row);
            if constexpr (std::is_lvalue_reference_v<decltype(row)>)
                node->charge.add(bytes);
            else
                moved_buffers->transfer_to(node->charge, bytes);
            // With the supported STL profiles a vector or string copy fits
            // within the source capacity, and moving transfers its allocation.
            node->rows.push_back(std::forward<decltype(row)>(row));
            const auto& stored = node->rows.back();
            if (AdaptiveLeafPages::buffer_bytes(stored) > bytes)
                throw std::invalid_argument("Adaptive leaf row capacity exceeds preadmission.");
            previous = &stored.key;
        }
        node->maximum = &node->rows.back().key;
        node->size = count;
        node->subtree_bytes = node->charge.bytes();
        ++stats.new_leaf_pages;
        stats.copied_rows += count;
        return node;
    }

    [[nodiscard]] NodePtr branch(std::span<const NodePtr> children)
    {
        auto node = allocate();
        node->charge.add(children.size() * sizeof(NodePtr));
        node->children.reserve(children.size());
        if (node->children.capacity() != children.size())
            throw std::invalid_argument("Adaptive leaf branch capacity differs from admission.");
        node->subtree_bytes = node->charge.bytes();
        for (const auto& child : children)
        {
            if (child->size > std::numeric_limits<std::size_t>::max() - node->size ||
                child->subtree_bytes > std::numeric_limits<std::size_t>::max() - node->subtree_bytes)
                throw std::overflow_error("Adaptive leaf page totals overflow.");
            node->size += child->size;
            node->subtree_bytes += child->subtree_bytes;
            node->children.push_back(child);
        }
        node->maximum = node->children.back()->maximum;
        return node;
    }

    template<class Next>
    [[nodiscard]] Nodes pack_leaves(std::size_t count, Next next, PageCharge* moved_buffers = nullptr)
    {
        const auto groups = groups_for(count, leaf_capacity);
        auto result = nodes();
        result.reserve(groups);
        const AdaptiveCellKey* previous = nullptr;
        for (std::size_t group = 0; group < groups; ++group)
            result.push_back(leaf(group_size(count, groups, group), next, previous, moved_buffers));
        return result;
    }

    [[nodiscard]] Nodes pack_branches(const Nodes& children)
    {
        const auto groups = groups_for(children.size(), branch_capacity);
        auto result = nodes();
        result.reserve(groups);
        std::size_t offset = 0;
        for (std::size_t group = 0; group < groups; ++group)
        {
            const auto count = group_size(children.size(), groups, group);
            result.push_back(branch(std::span<const NodePtr>(children).subspan(offset, count)));
            offset += count;
        }
        return result;
    }

    [[nodiscard]] NodePtr finish(Nodes level)
    {
        while (level.size() > 1)
            level = pack_branches(level);
        if (level.empty())
            return {};
        auto root = std::move(level.front());
        while (root->children.size() == 1)
        {
            auto child = root->children.front();
            root = std::move(child);
        }
        return root;
    }

    [[nodiscard]] Nodes pack_rows(const Rows& selected)
    {
        auto next = selected.begin();
        return pack_leaves(selected.size(), [&]() -> const AdaptiveLeaf& { return **next++; });
    }

    [[nodiscard]] Nodes redistribute(const Node& left, const Node& right)
    {
        if (left.terminal())
        {
            retire(left);
            retire(right);
            auto selected = rows();
            selected.reserve(left.rows.size() + right.rows.size());
            for (const auto* page : {&left, &right})
                for (const auto& row : page->rows)
                    selected.push_back(&row);
            return pack_rows(selected);
        }
        auto children = nodes();
        children.reserve(left.children.size() + right.children.size());
        children.insert(children.end(), left.children.begin(), left.children.end());
        children.insert(children.end(), right.children.begin(), right.children.end());
        rebalance(children);
        return pack_branches(children);
    }

    void rebalance(Nodes& children)
    {
        std::size_t index = 0;
        while (index < children.size() && children.size() > 1)
        {
            const auto& child = *children[index];
            const auto minimum = (child.terminal() ? leaf_capacity : branch_capacity) / 2;
            if (child.occupancy() >= minimum)
            {
                ++index;
                continue;
            }
            const auto left = index == 0 ? index : index - 1;
            auto replacement = redistribute(*children[left], *children[left + 1]);
            children[left] = std::move(replacement.front());
            if (replacement.size() == 1)
                children.erase(children.begin() + left + 1);
            else
                children[left + 1] = std::move(replacement.back());
            index = left;
        }
    }

    template<class Change>
    [[nodiscard]] Nodes edit_rows(std::span<const AdaptiveLeaf> original, std::span<const Change> edits)
    {
        auto selected = rows();
        auto count = original.size();
        for (const auto& edit : edits)
        {
            if (!edit.expected_revision)
                ++count;
            else if (!edit.leaf)
                --count;
        }
        selected.reserve(count);
        auto previous = original.begin();
        for (const auto& edit : edits)
        {
            while (previous != original.end() && previous->key < edit.key)
                selected.push_back(&*previous++);
            if (edit.expected_revision)
                ++previous;
            if (edit.leaf)
                selected.push_back(&*edit.leaf);
        }
        while (previous != original.end())
            selected.push_back(&*previous++);
        return pack_rows(selected);
    }

    template<class Change> [[nodiscard]] Nodes edit_node(const Node& source, std::span<const Change> edits)
    {
        if (source.terminal())
        {
            retire(source);
            return edit_rows(source.rows, edits);
        }
        auto children = nodes();
        children.reserve(source.children.size());
        for (const auto& child : source.children)
        {
            const auto boundary = child == source.children.back()
                ? edits.end()
                : std::upper_bound(edits.begin(), edits.end(), *child->maximum,
                      [](const AdaptiveCellKey& key, const Change& edit) { return key < edit.key; });
            const auto count = std::size_t(boundary - edits.begin());
            if (count == 0)
                children.push_back(child);
            else
            {
                auto replacement = edit_node(*child, edits.first(count));
                children.insert(children.end(), std::make_move_iterator(replacement.begin()),
                    std::make_move_iterator(replacement.end()));
            }
            edits = edits.subspan(count);
        }
        rebalance(children);
        return pack_branches(children);
    }

    template<class Change>
    [[nodiscard]] static AdaptiveLeafPages edit(const AdaptiveLeafPages& source, std::span<const Change> edits,
        std::shared_ptr<Budget> arena, Stats* stats, const AdaptiveLeafPages::PageChanges* changes = nullptr)
    {
        if (!arena || (changes && arena->used))
            throw std::invalid_argument("Adaptive leaf page changes require a fresh budget arena.");
        for (std::size_t index = 0; index < edits.size(); ++index)
        {
            const auto& edit = edits[index];
            const auto* existing = source.find(edit.key);
            if ((index && !(edits[index - 1].key < edit.key)) ||
                (edit.leaf && edit.leaf->key != edit.key) ||
                (edit.expected_revision ? !existing || existing->revision != *edit.expected_revision
                                        : existing || !edit.leaf))
                throw std::invalid_argument("Unordered, stale or invalid adaptive leaf page edit.");
        }
        if (edits.empty())
        {
            if (stats)
                *stats = {};
            return source;
        }
        AdaptiveLeafPageBuilder builder{std::move(arena), {}, changes};
        auto pages = source.root_ ? builder.edit_node(*source.root_, edits) : builder.edit_rows({}, edits);
        AdaptiveLeafPages result(builder.finish(std::move(pages)));
        if (changes && result.root_)
            builder.visit_new(*result.root_);
        if (stats)
            *stats = builder.stats;
        return result;
    }
};

AdaptiveLeafPages::const_iterator::const_iterator(const Node* root) noexcept
{
    if (root)
        descend(root);
}

void AdaptiveLeafPages::const_iterator::descend(const Node* node) noexcept
{
    while (!node->terminal())
    {
        path_[depth_++] = {node, 1};
        node = node->children.front().get();
    }
    leaf_ = node;
    row_ = 0;
}

AdaptiveLeafPages::const_iterator::reference AdaptiveLeafPages::const_iterator::operator*() const noexcept
{
    return leaf_->rows[row_];
}

AdaptiveLeafPages::const_iterator& AdaptiveLeafPages::const_iterator::operator++() noexcept
{
    if (++row_ < leaf_->rows.size())
        return *this;
    while (depth_)
    {
        auto& branch = path_[depth_ - 1];
        if (branch.next_child < branch.node->children.size())
        {
            descend(branch.node->children[branch.next_child++].get());
            return *this;
        }
        --depth_;
    }
    leaf_ = nullptr;
    row_ = 0;
    return *this;
}

std::size_t AdaptiveLeafPages::size() const noexcept
{
    return root_ ? root_->size : 0;
}

const AdaptiveLeafPages::Node* AdaptiveLeafPages::find_page(const AdaptiveCellKey& key, std::size_t& prefix) const
{
    const auto* node = root_.get();
    while (node && !node->terminal())
    {
        auto child = node->children.begin();
        while (child != node->children.end() && *(*child)->maximum < key)
        {
            prefix += (*child)->size;
            ++child;
        }
        node = child == node->children.end() ? nullptr : child->get();
    }
    return node;
}

const AdaptiveLeaf* AdaptiveLeafPages::find(const AdaptiveCellKey& key) const
{
    std::size_t unused = 0;
    const auto* node = find_page(key, unused);
    if (!node)
        return nullptr;
    const auto row = node->lower_bound(key);
    return row != node->rows.end() && row->key == key ? &*row : nullptr;
}

const AdaptiveLeaf& AdaptiveLeafPages::at(const AdaptiveCellKey& key) const
{
    const auto* row = find(key);
    if (!row)
        throw std::out_of_range("Adaptive leaf key is not locally owned.");
    return *row;
}

std::optional<std::size_t> AdaptiveLeafPages::ordinal_of(const AdaptiveCellKey& key) const
{
    std::size_t prefix = 0;
    const auto* node = find_page(key, prefix);
    if (!node)
        return {};
    const auto row = node->lower_bound(key);
    if (row == node->rows.end() || row->key != key)
        return {};
    return prefix + std::size_t(row - node->rows.begin());
}

const AdaptiveLeaf& AdaptiveLeafPages::at_ordinal(std::size_t ordinal) const
{
    if (ordinal >= size())
        throw std::out_of_range("Adaptive leaf ordinal is not locally owned.");
    const auto* node = root_.get();
    while (!node->terminal())
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

AdaptiveLeafPages AdaptiveLeafPages::build(std::span<const AdaptiveLeaf> leaves, std::shared_ptr<Budget> arena)
{
    if (!arena)
        throw std::invalid_argument("Adaptive leaf pages require a budget arena.");
    for (std::size_t index = 1; index < leaves.size(); ++index)
        if (!(leaves[index - 1].key < leaves[index].key))
            throw std::invalid_argument("Adaptive leaf page input is not strictly ordered.");
    AdaptiveLeafPageBuilder builder{std::move(arena), {}};
    auto row = leaves.begin();
    return AdaptiveLeafPages(builder.finish(
        builder.pack_leaves(leaves.size(), [&]() -> const AdaptiveLeaf& { return *row++; })));
}

AdaptiveLeafPages AdaptiveLeafPages::build_moving(std::size_t count, const std::function<AdaptiveLeaf&&()>& next,
    std::shared_ptr<Budget> arena, std::size_t moved_buffer_bytes)
{
    if (!arena)
        throw std::invalid_argument("Adaptive leaf pages require a budget arena.");
    PageCharge moved_buffers(arena);
    moved_buffers.add(moved_buffer_bytes);
    AdaptiveLeafPageBuilder builder{std::move(arena), {}};
    auto pages = builder.pack_leaves(count, next, &moved_buffers);
    if (moved_buffers.bytes() != 0)
        throw std::invalid_argument("Moved adaptive row buffers differ from page precharge.");
    return AdaptiveLeafPages(builder.finish(std::move(pages)));
}

AdaptiveLeafPages AdaptiveLeafPages::edit(const AdaptiveLeafPages& source, std::span<const Edit> edits,
    std::shared_ptr<Budget> arena, EditStats* stats)
{
    return AdaptiveLeafPageBuilder::edit(source, edits, std::move(arena), stats);
}

AdaptiveLeafPages AdaptiveLeafPages::edit_borrowed(const AdaptiveLeafPages& source, std::span<const EditView> edits,
    std::shared_ptr<Budget> arena, EditStats* stats, const PageChanges* changes)
{
    return AdaptiveLeafPageBuilder::edit(source, edits, std::move(arena), stats, changes);
}

void AdaptiveLeafPages::visit_leaf_pages(const LeafPageVisitor& visitor) const
{
    const auto visit = [&](const auto& self, const Node& page) -> void
    {
        if (page.terminal())
            visitor(page.rows);
        else
            for (const auto& child : page.children)
                self(self, *child);
    };
    if (root_)
        visit(visit, *root_);
}

void AdaptiveLeafPages::visit_pages(const std::function<void(const void*, std::size_t)>& visitor) const
{
    const auto visit = [&](const auto& self, const Node& node) -> void
    {
        visitor(&node, node.charge.bytes());
        for (const auto& child : node.children)
            self(self, *child);
    };
    if (root_)
        visit(visit, *root_);
}

AdaptiveLeafPages::Storage AdaptiveLeafPages::storage() const
{
    Storage result;
    visit_pages(
        [&](const void* owner, std::size_t bytes)
        {
            const auto& page = *static_cast<const Node*>(owner);
            ++result.nodes;
            result.retained_bytes += bytes;
            if (page.terminal())
            {
                ++result.leaf_pages;
                result.leaf_rows += page.rows.size();
                result.leaf_capacity += page.rows.capacity();
                result.maximum_leaf_rows = std::max(result.maximum_leaf_rows, page.rows.size());
            }
            else
                result.maximum_fanout = std::max(result.maximum_fanout, page.children.size());
        });
    return result;
}

std::size_t AdaptiveLeafPages::retained_bytes() const noexcept
{
    return root_ ? root_->subtree_bytes : 0;
}

const void* AdaptiveLeafPages::page_identity(const AdaptiveCellKey& key) const
{
    std::size_t unused = 0;
    const auto* node = find_page(key, unused);
    if (!node)
        return nullptr;
    const auto row = node->lower_bound(key);
    return row != node->rows.end() && row->key == key ? node : nullptr;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
