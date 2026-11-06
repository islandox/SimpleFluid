/** @file ContactOwnerUsage.cc
 * @brief Counted AVL paths with cached physical and inherited metadata bytes.
 */
#include "ContactOwnerUsage.hh"
#include <algorithm>
#include <atomic>
#include <limits>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
std::size_t sum(std::size_t first, std::size_t second)
{
    if (second > std::numeric_limits<std::size_t>::max() - first)
        throw std::overflow_error("Contact owner usage overflows.");
    return first + second;
}
bool less(ContactOwnerUsage::Key first, ContactOwnerUsage::Key second) noexcept
{
    if (first.role != second.role)
        return first.role < second.role;
    return std::less<const void*>{}(first.allocation, second.allocation);
}
bool has_owner(const std::shared_ptr<const void>& owner) noexcept
{
    const std::shared_ptr<const void> empty;
    return owner.owner_before(empty) || empty.owner_before(owner);
}
bool same_control(const std::weak_ptr<const void>& expected, const std::shared_ptr<const void>& owner) noexcept
{
    return !expected.owner_before(owner) && !owner.owner_before(expected);
}
std::uint64_t next_editor()
{
    // A scalar construction identity avoids retaining a former editor and also
    // distinguishes independent indexes allocated from the same budget arena.
    static std::atomic<std::uint64_t> sequence{1};
    auto value = sequence.load(std::memory_order_relaxed);
    for (;;)
    {
        if (value == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Contact owner editor identity exhausted.");
        if (sequence.compare_exchange_weak(value, value + 1, std::memory_order_relaxed))
            return value;
    }
}
} // namespace

struct ContactOwnerUsage::Node
{
    ContactGroupAllocation admission;
    Key key;
    Entry entry;
    Pointer left, right;
    std::uint64_t editor;
    std::size_t payload, storage, inherited = 0, count = 1;
    unsigned height = 1;

    Node(ContactGroupAllocation charge, Key key_, Entry entry_, Pointer left_, Pointer right_, std::uint64_t editor_)
        : admission(std::move(charge)), key(key_), entry(std::move(entry_)), left(std::move(left_)),
          right(std::move(right_)), editor(editor_), payload(entry.bytes), storage(admission.bytes())
    {
        for (const auto* child : {left.get(), right.get()})
            if (child)
            {
                payload = sum(payload, child->payload);
                storage = sum(storage, child->storage);
                inherited = sum(inherited, child->editor == editor ? child->inherited : child->storage);
                count = sum(count, child->count);
                height = std::max(height, child->height + 1);
            }
    }
};

struct ContactOwnerUsageEditor
{
    using Node = ContactOwnerUsage::Node;
    using Pointer = ContactOwnerUsage::Pointer;
    using Key = ContactOwnerUsage::Key;
    using Entry = ContactOwnerUsage::Entry;
    const std::shared_ptr<Budget>& arena;
    std::uint64_t editor;
    ContactOwnerUsage::Work work{};

    [[nodiscard]] const Node* find(const Pointer& root, Key key)
    {
        auto* node = root.get();
        while (node)
        {
            ++work.visited_nodes;
            if (less(key, node->key))
                node = node->left.get();
            else if (less(node->key, key))
                node = node->right.get();
            else
                return node;
        }
        return nullptr;
    }
    [[nodiscard]] static unsigned height(const Pointer& node) noexcept { return node ? node->height : 0; }
    [[nodiscard]] Pointer make(Key key, Entry entry, const Pointer& left, const Pointer& right)
    {
        ContactGroupAllocation charge(arena);
        charge.add(sizeof(Node) + 128);
        auto result = std::make_shared<const Node>(std::move(charge), key, std::move(entry), left, right, editor);
        ++work.created_nodes;
        return result;
    }
    [[nodiscard]] Pointer make(const Node& node, const Pointer& left, const Pointer& right)
    {
        return make(node.key, node.entry, left, right);
    }
    [[nodiscard]] Pointer balance(Key key, Entry entry, const Pointer& left, const Pointer& right)
    {
        if (height(left) > height(right) + 1)
        {
            if (height(left->right) > height(left->left))
            {
                const auto& pivot = left->right;
                return make(*pivot, make(*left, left->left, pivot->left), make(key, entry, pivot->right, right));
            }
            return make(*left, left->left, make(key, entry, left->right, right));
        }
        if (height(right) > height(left) + 1)
        {
            if (height(right->left) > height(right->right))
            {
                const auto& pivot = right->left;
                return make(*pivot, make(key, entry, left, pivot->left), make(*right, pivot->right, right->right));
            }
            return make(*right, make(key, entry, left, right->left), right->right);
        }
        return make(key, entry, left, right);
    }
    [[nodiscard]] Pointer rewrite(const Pointer& node, Key key, Entry entry)
    {
        if (!node)
        {
            if (!entry.references)
                throw std::logic_error("Missing contact owner during retirement.");
            return make(key, entry, {}, {});
        }
        ++work.visited_nodes;
        if (less(key, node->key))
            return balance(node->key, node->entry, rewrite(node->left, key, entry), node->right);
        if (less(node->key, key))
            return balance(node->key, node->entry, node->left, rewrite(node->right, key, entry));
        if (entry.references)
            return make(key, entry, node->left, node->right);
        if (!node->left || !node->right)
            return node->left ? node->left : node->right;
        auto* successor = node->right.get();
        while (successor->left)
        {
            ++work.visited_nodes;
            successor = successor->left.get();
        }
        return balance(successor->key, successor->entry, node->left, rewrite(node->right, successor->key, Entry{}));
    }
};

ContactOwnerUsage::ContactOwnerUsage(std::shared_ptr<Budget> arena)
    : imported_(std::move(arena)), editor_(next_editor())
{
}
ContactOwnerUsage::ContactOwnerUsage(const ContactOwnerUsage& source, std::shared_ptr<Budget> arena)
    : imported_(std::move(arena)), editor_(next_editor()), keep_overlap_(true)
{
    if (&imported_.arena() == &source.imported_.arena())
        throw std::invalid_argument("Contact owner fork requires an independent candidate arena.");
    imported_.add(source.storage_bytes());
    root_ = source.root_;
}
ContactOwnerUsage::~ContactOwnerUsage() = default;

ContactOwnerUsage::Transition ContactOwnerUsage::retain(Key key, std::size_t bytes, const Budget* origin)
{
    return retain_impl(key, bytes, origin, nullptr);
}
ContactOwnerUsage::Transition ContactOwnerUsage::retain(
    Key key, std::size_t bytes, const Budget* origin, const std::shared_ptr<const void>& owner)
{
    return retain_impl(key, bytes, origin, &owner);
}
ContactOwnerUsage::Transition ContactOwnerUsage::retain_impl(
    Key key, std::size_t bytes, const Budget* origin, const std::shared_ptr<const void>* owner)
{
    if (!key.allocation)
        throw std::invalid_argument("Contact owner allocation identity is null.");
    if (owner && !has_owner(*owner))
        throw std::invalid_argument("Contact owner control is empty.");
    ContactOwnerUsageEditor edit{imported_.keeper(), editor_};
    const auto* old = edit.find(root_, key);
    if (old && (old->entry.bytes != bytes || old->entry.origin != origin))
        throw std::invalid_argument("Contact owner capacity or origin differs for the same allocation.");
    if (old && (old->entry.has_control != bool(owner) || (owner && !same_control(old->entry.control, *owner))))
        throw std::invalid_argument("Contact owner control differs for the same allocation.");
    const auto before = old ? old->entry.references : 0;
    const auto after = sum(before, 1);
    Entry entry{after, bytes, origin, {}, false};
    if (owner)
    {
        entry.control = *owner;
        entry.has_control = true;
    }
    auto next = edit.rewrite(root_, key, entry);
    publish(std::move(next), edit.work);
    return {before, after, bytes, origin};
}
ContactOwnerUsage::Transition ContactOwnerUsage::release(Key key)
{
    return release_impl(key, nullptr);
}
ContactOwnerUsage::Transition ContactOwnerUsage::release(Key key, const std::shared_ptr<const void>& owner)
{
    return release_impl(key, &owner);
}
ContactOwnerUsage::Transition ContactOwnerUsage::release_impl(Key key, const std::shared_ptr<const void>* owner)
{
    ContactOwnerUsageEditor edit{imported_.keeper(), editor_};
    const auto* old = edit.find(root_, key);
    if (!old)
        throw std::invalid_argument("Cannot retire an unreferenced contact owner.");
    if (old->entry.has_control != bool(owner) ||
        (owner && (!has_owner(*owner) || !same_control(old->entry.control, *owner))))
        throw std::invalid_argument("Cannot retire a foreign contact owner control.");
    const auto before = old->entry;
    auto entry = before;
    --entry.references;
    auto next = edit.rewrite(root_, key, entry);
    publish(std::move(next), edit.work);
    return {before.references, entry.references, before.bytes, before.origin};
}
bool ContactOwnerUsage::authentic(Key key, const std::shared_ptr<const void>& owner) const noexcept
{
    if (!has_owner(owner))
        return false;
    const auto entry = find(key);
    return entry && entry->has_control && same_control(entry->control, owner);
}
std::optional<ContactOwnerUsage::Entry> ContactOwnerUsage::find(Key key) const noexcept
{
    const auto* node = root_.get();
    while (node)
    {
        if (less(key, node->key))
            node = node->left.get();
        else if (less(node->key, key))
            node = node->right.get();
        else
            return node->entry;
    }
    return {};
}
std::size_t ContactOwnerUsage::payload_bytes() const noexcept
{
    return root_ ? root_->payload : 0;
}
std::size_t ContactOwnerUsage::storage_bytes() const noexcept
{
    return root_ ? root_->storage : 0;
}
std::size_t ContactOwnerUsage::distinct_owners() const noexcept
{
    return root_ ? root_->count : 0;
}
std::size_t ContactOwnerUsage::inherited_bytes(const Pointer& node) const noexcept
{
    return !node ? 0 : (node->editor == editor_ ? node->inherited : node->storage);
}
std::size_t ContactOwnerUsage::overlap_bytes() const noexcept
{
    return imported_.bytes() - inherited_bytes(root_);
}
void ContactOwnerUsage::publish(Pointer next, Work work)
{
    const auto inherited = inherited_bytes(next);
    if (inherited > imported_.bytes())
        throw std::logic_error("Contact owner edit introduced unadmitted inherited metadata.");
    root_ = std::move(next);
    if (!keep_overlap_)
        imported_.release(imported_.bytes() - inherited);
    work_ = work;
}
void ContactOwnerUsage::finish_overlap() noexcept
{
    imported_.release(overlap_bytes());
    keep_overlap_ = false;
}
void ContactOwnerUsage::visit_entries(const std::function<void(Key, Entry)>& visitor) const
{
    const auto visit = [&](const auto& self, const Node* node) -> void
    {
        if (!node)
            return;
        self(self, node->left.get());
        visitor(node->key, node->entry);
        self(self, node->right.get());
    };
    visit(visit, root_.get());
}
void ContactOwnerUsage::visit_pages(const std::function<void(Page)>& visitor) const
{
    const auto visit = [&](const auto& self, const Node* node) -> void
    {
        if (!node)
            return;
        visitor(Page{node, &node->admission.arena(), node->admission.bytes(), node->editor != editor_, sizeof(Node)});
        self(self, node->left.get());
        self(self, node->right.get());
    };
    visit(visit, root_.get());
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
