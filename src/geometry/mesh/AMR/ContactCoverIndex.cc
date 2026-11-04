/** @file ContactCoverIndex.cc
 * @brief Flat balanced bounds tree borrowing normalized point groups.
 */
#include "geometry/mesh/AMR/ContactCoverIndex.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
constexpr std::size_t absent = std::numeric_limits<std::size_t>::max();
class MetadataEnvironment
{
public:
    MetadataEnvironment()
    {
        if (std::fegetenv(&saved_) != 0)
            throw std::runtime_error("Cannot capture contact-cover floating-point environment.");
        std::fenv_t held;
        if (std::feholdexcept(&held) != 0)
        {
            (void) std::fesetenv(&saved_);
            throw std::runtime_error("Cannot isolate contact-cover floating-point exceptions.");
        }
    }
    ~MetadataEnvironment() { (void) std::fesetenv(&saved_); }
    MetadataEnvironment(const MetadataEnvironment&) = delete;
    MetadataEnvironment& operator=(const MetadataEnvironment&) = delete;

private:
    std::fenv_t saved_{};
};
std::size_t sum(std::size_t a, std::size_t b)
{
    if (b > absent - a)
        throw std::invalid_argument("Contact-cover capacity overflows.");
    return a + b;
}
std::size_t product(std::size_t a, std::size_t b)
{
    if (b && a > absent / b)
        throw std::invalid_argument("Contact-cover capacity overflows.");
    return a * b;
}
void validate(const std::array<long double, 3>* point, const AdaptiveBytes* domain)
{
    if (!point || !domain || domain->empty())
        throw std::invalid_argument("Contact-cover point needs coordinates and a nonempty domain.");
    for (const auto value : *point)
        if (!std::isfinite(value))
            throw std::invalid_argument("Contact-cover coordinates must be finite.");
}
AdaptiveContactBounds united(const AdaptiveContactBounds& a, const AdaptiveContactBounds& b) noexcept
{
    AdaptiveContactBounds result;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        result.lower[axis] = std::min(a.lower[axis], b.lower[axis]);
        result.upper[axis] = std::max(a.upper[axis], b.upper[axis]);
    }
    return result;
}
long double extent(const AdaptiveContactBounds& bounds) noexcept
{
    long double result = 0;
    for (std::size_t axis = 0; axis < 3; ++axis)
        result += bounds.upper[axis] - bounds.lower[axis];
    return result;
}
/** Includes construction scratch and returned buffers until ownership moves
 * to the caller. Failed admission never refunds cumulative admission. */
class Scratch
{
public:
    explicit Scratch(Budget& budget) : budget_(budget) {}
    ~Scratch() { budget_.used -= live_; }
    void add(std::size_t bytes)
    {
        budget_.add(bytes);
        live_ += bytes;
    }
    void retire(std::size_t bytes) noexcept
    {
        budget_.used -= bytes;
        live_ -= bytes;
    }
    void transfer(std::size_t bytes) noexcept { live_ -= bytes; }
    template<class T> void reserve(std::vector<T>& values, std::size_t capacity)
    {
        if (capacity <= values.capacity())
            return;
        const auto previous = product(values.capacity(), sizeof(T));
        add(product(capacity, sizeof(T)));
        values.reserve(capacity);
        if (values.capacity() != capacity)
            throw std::invalid_argument("Contact-cover vector capacity differs from admission.");
        retire(previous);
    }

private:
    Budget& budget_;
    std::size_t live_ = 0;
};
} // namespace

struct ContactCoverIndex::Impl
{
    // Child tags use the low bit: even for a leaf, odd for a branch. Parent
    // tags always name a branch. Free slots reuse their parent as a free link.
    struct Leaf
    {
        const std::array<long double, 3>* point = nullptr;
        const AdaptiveBytes* domain = nullptr;
        std::size_t parent = absent;
    };
    struct Branch
    {
        AdaptiveContactBounds bounds;
        std::size_t parent = absent, left = absent, right = absent;
        std::size_t homogeneous = absent, count = 0, height = 0;
    };
    Budget& budget;
    int owner;
    std::unique_ptr<Leaf[]> leaves;
    std::unique_ptr<Branch[]> branches;
    std::size_t capacity = 0, leaf_slots = 0, branch_slots = 0;
    std::size_t free_leaf = absent, free_branch = absent, root = absent, count = 0;
    static std::size_t base_bytes() noexcept { return sizeof(ContactCoverIndex) + sizeof(Impl); }

    Impl(Budget& budget_, int owner_) : budget(budget_), owner(owner_) {}
    ~Impl()
    {
        const auto bytes = storage_bytes();
        leaves.reset();
        branches.reset();
        budget.used -= bytes;
    }
    static bool is_leaf(std::size_t tag) noexcept { return !(tag & 1); }
    static std::size_t leaf_tag(std::size_t slot) noexcept { return slot * 2; }
    static std::size_t branch_tag(std::size_t slot) noexcept { return slot * 2 + 1; }
    Leaf& leaf(std::size_t tag) noexcept { return leaves[tag / 2]; }
    const Leaf& leaf(std::size_t tag) const noexcept { return leaves[tag / 2]; }
    Branch& branch(std::size_t tag) noexcept { return branches[tag / 2]; }
    const Branch& branch(std::size_t tag) const noexcept { return branches[tag / 2]; }
    std::size_t& parent(std::size_t tag) noexcept { return is_leaf(tag) ? leaf(tag).parent : branch(tag).parent; }
    std::size_t height(std::size_t tag) const noexcept { return is_leaf(tag) ? 1 : branch(tag).height; }
    std::size_t points(std::size_t tag) const noexcept { return is_leaf(tag) ? 1 : branch(tag).count; }
    std::size_t homogeneous(std::size_t tag) const noexcept { return is_leaf(tag) ? tag / 2 : branch(tag).homogeneous; }
    AdaptiveContactBounds bounds(std::size_t tag) const noexcept
    {
        if (!is_leaf(tag))
            return branch(tag).bounds;
        return {*leaf(tag).point, *leaf(tag).point};
    }
    std::size_t storage_bytes() const noexcept
    {
        return base_bytes() + capacity * sizeof(Leaf) + (capacity ? capacity - 1 : 0) * sizeof(Branch);
    }
    void reserve(std::size_t requested)
    {
        if (requested <= capacity)
            return;
        // This also guarantees that no encoded child can equal absent.
        if (requested > absent / 2)
            throw std::invalid_argument("Contact-cover handle capacity overflows.");
        const auto leaf_bytes = product(requested, sizeof(Leaf));
        const auto branch_bytes = product(requested - 1, sizeof(Branch));
        const auto next_bytes = sum(leaf_bytes, branch_bytes);
        (void) sum(base_bytes(), next_bytes);
        budget.add(next_bytes);
        try
        {
            auto next_leaves = std::make_unique<Leaf[]>(requested);
            auto next_branches = requested > 1 ? std::make_unique<Branch[]>(requested - 1) : nullptr;
            if (leaf_slots)
                std::copy_n(leaves.get(), leaf_slots, next_leaves.get());
            if (branch_slots)
                std::copy_n(branches.get(), branch_slots, next_branches.get());
            const auto previous = storage_bytes() - base_bytes();
            leaves.swap(next_leaves);
            branches.swap(next_branches);
            capacity = requested;
            // Old arrays must die before their charge is retired.
            next_leaves.reset();
            next_branches.reset();
            budget.used -= previous;
        }
        catch (...)
        {
            budget.used -= next_bytes;
            throw;
        }
    }
    void update(std::size_t tag) noexcept
    {
        auto& node = branch(tag);
        node.bounds = united(bounds(node.left), bounds(node.right));
        node.count = points(node.left) + points(node.right);
        node.height = 1 + std::max(height(node.left), height(node.right));
        const auto a = homogeneous(node.left), b = homogeneous(node.right);
        node.homogeneous = a != absent && b != absent && *leaves[a].domain == *leaves[b].domain ? a : absent;
    }
    void replace_link(std::size_t parent_tag, std::size_t old_tag, std::size_t new_tag) noexcept
    {
        if (parent_tag == absent)
            root = new_tag;
        else
        {
            auto& node = branch(parent_tag);
            (node.left == old_tag ? node.left : node.right) = new_tag;
        }
        parent(new_tag) = parent_tag;
    }
    std::size_t rotate_left(std::size_t tag) noexcept
    {
        auto& old = branch(tag);
        const auto next_tag = old.right;
        auto& next = branch(next_tag);
        replace_link(old.parent, tag, next_tag);
        old.right = next.left;
        parent(old.right) = tag;
        next.left = tag;
        old.parent = next_tag;
        update(tag);
        update(next_tag);
        return next_tag;
    }
    std::size_t rotate_right(std::size_t tag) noexcept
    {
        auto& old = branch(tag);
        const auto next_tag = old.left;
        auto& next = branch(next_tag);
        replace_link(old.parent, tag, next_tag);
        old.left = next.right;
        parent(old.left) = tag;
        next.right = tag;
        old.parent = next_tag;
        update(tag);
        update(next_tag);
        return next_tag;
    }
    void balance(std::size_t tag) noexcept
    {
        while (tag != absent)
        {
            update(tag);
            const auto& node = branch(tag);
            const auto left_height = height(node.left), right_height = height(node.right);
            if (right_height > left_height + 1)
            {
                const auto child = node.right;
                if (height(branch(child).left) > height(branch(child).right))
                    rotate_right(child);
                tag = rotate_left(tag);
            }
            else if (left_height > right_height + 1)
            {
                const auto child = node.left;
                if (height(branch(child).right) > height(branch(child).left))
                    rotate_left(child);
                tag = rotate_right(tag);
            }
            tag = branch(tag).parent;
        }
    }
    Leaf& checked_leaf(std::size_t handle)
    {
        if (handle >= leaf_slots || !leaves[handle].point)
            throw std::invalid_argument("Contact-cover handle is not live.");
        return leaves[handle];
    }
};

ContactCoverIndex::ContactCoverIndex(Budget& budget, int owner)
{
    if (owner < 0)
        throw std::invalid_argument("Contact-cover owner must be nonnegative.");
    budget.add(Impl::base_bytes());
    try
    {
        impl_ = std::make_unique<Impl>(budget, owner);
    }
    catch (...)
    {
        budget.used -= Impl::base_bytes();
        throw;
    }
}
ContactCoverIndex::~ContactCoverIndex() = default;
ContactCoverIndex::ContactCoverIndex(const ContactCoverIndex& source, Budget& budget)
    : ContactCoverIndex(budget, source.impl_->owner)
{
    auto& target = *impl_;
    const auto& original = *source.impl_;
    target.reserve(original.capacity);
    if (original.leaf_slots)
        std::copy_n(original.leaves.get(), original.leaf_slots, target.leaves.get());
    if (original.branch_slots)
        std::copy_n(original.branches.get(), original.branch_slots, target.branches.get());
    target.leaf_slots = original.leaf_slots;
    target.branch_slots = original.branch_slots;
    target.free_leaf = original.free_leaf;
    target.free_branch = original.free_branch;
    target.root = original.root;
    target.count = original.count;
}
void ContactCoverIndex::reserve(std::size_t maximum_points)
{
    impl_->reserve(maximum_points);
}
ContactCoverIndex::Handle ContactCoverIndex::insert(
    const std::array<long double, 3>* point, const AdaptiveBytes* domain)
{
    const MetadataEnvironment environment;
    validate(point, domain);
    auto& tree = *impl_;
    // Both needed slots exist before any membership or topology changes.
    if (tree.free_leaf == absent && tree.leaf_slots == tree.capacity)
        tree.reserve(tree.capacity ? product(tree.capacity, 2) : 1);
    const auto slot = tree.free_leaf != absent ? tree.free_leaf : tree.leaf_slots++;
    if (tree.free_leaf != absent)
        tree.free_leaf = tree.leaves[slot].parent;
    auto& leaf = tree.leaves[slot];
    leaf = {point, domain, absent};
    const auto tag = Impl::leaf_tag(slot);
    if (tree.root == absent)
        tree.root = tag;
    else
    {
        const auto branch_slot = tree.free_branch != absent ? tree.free_branch : tree.branch_slots++;
        if (tree.free_branch != absent)
            tree.free_branch = tree.branches[branch_slot].parent;
        auto sibling = tree.root;
        const AdaptiveContactBounds point_bounds{*point, *point};
        while (!Impl::is_leaf(sibling))
        {
            const auto& node = tree.branch(sibling);
            const auto a = extent(united(tree.bounds(node.left), point_bounds));
            const auto b = extent(united(tree.bounds(node.right), point_bounds));
            sibling = a < b || (a == b && tree.height(node.left) <= tree.height(node.right)) ? node.left : node.right;
        }
        const auto branch_tag = Impl::branch_tag(branch_slot);
        auto& branch = tree.branches[branch_slot];
        branch = {};
        branch.left = sibling;
        branch.right = tag;
        tree.replace_link(tree.parent(sibling), sibling, branch_tag);
        tree.parent(sibling) = branch_tag;
        leaf.parent = branch_tag;
        tree.balance(branch_tag);
    }
    ++tree.count;
    return slot;
}
void ContactCoverIndex::erase(Handle handle)
{
    const MetadataEnvironment environment;
    auto& tree = *impl_;
    auto& leaf = tree.checked_leaf(handle);
    const auto tag = Impl::leaf_tag(handle);
    const auto parent_tag = leaf.parent;
    if (parent_tag == absent)
        tree.root = absent;
    else
    {
        auto& parent = tree.branch(parent_tag);
        const auto sibling = parent.left == tag ? parent.right : parent.left;
        const auto grandparent = parent.parent;
        tree.replace_link(grandparent, parent_tag, sibling);
        parent.parent = tree.free_branch;
        tree.free_branch = parent_tag / 2;
        tree.balance(grandparent);
    }
    leaf.point = nullptr;
    leaf.domain = nullptr;
    leaf.parent = tree.free_leaf;
    tree.free_leaf = handle;
    --tree.count;
}
void ContactCoverIndex::rebind(Handle handle, const std::array<long double, 3>* point, const AdaptiveBytes* domain)
{
    const MetadataEnvironment environment;
    validate(point, domain);
    auto& leaf = impl_->checked_leaf(handle);
    leaf.point = point;
    leaf.domain = domain;
}
std::optional<AdaptiveContactBounds> ContactCoverIndex::aggregate_bounds() const noexcept
{
    if (impl_->root == absent)
        return {};
    return impl_->bounds(impl_->root);
}
std::vector<ClosedContactCover> ContactCoverIndex::ownership_frontier_covers(std::size_t target_fragments)
{
    const MetadataEnvironment environment;
    if (!target_fragments)
        throw std::invalid_argument("Contact-cover frontier needs a positive target.");
    auto& tree = *impl_;
    Scratch scratch(tree.budget);
    std::vector<std::size_t> frontier;
    std::vector<ClosedContactCover> output;
    auto append = [&](std::size_t tag)
    {
        if (frontier.size() == frontier.capacity())
            scratch.reserve(frontier, frontier.capacity() ? product(frontier.capacity(), 2) : 1);
        frontier.push_back(tag);
    };
    auto mandatory = [&](auto&& self, std::size_t tag) -> void
    {
        if (tag == absent)
            return;
        if (tree.homogeneous(tag) != absent)
        {
            append(tag);
            return;
        }
        self(self, tree.branch(tag).left);
        self(self, tree.branch(tag).right);
    };
    mandatory(mandatory, tree.root);
    auto active = frontier.size();
    for (std::size_t cursor = 0; active < target_fragments && cursor < frontier.size(); ++cursor)
    {
        const auto tag = frontier[cursor];
        if (tag == absent || Impl::is_leaf(tag))
            continue;
        frontier[cursor] = absent;
        append(tree.branch(tag).left);
        append(tree.branch(tag).right);
        ++active;
    }
    scratch.reserve(output, active);
    std::size_t payload = 0;
    for (const auto tag : frontier)
    {
        if (tag == absent)
            continue;
        const auto& domain = *tree.leaves[tree.homogeneous(tag)].domain;
        scratch.add(domain.size());
        ClosedContactCover cover{domain, tree.bounds(tag), tree.owner, tree.points(tag)};
        if (cover.domain.capacity() != domain.size())
            throw std::invalid_argument("Contact-cover domain capacity differs from admission.");
        payload = sum(payload, cover.domain.capacity());
        output.push_back(std::move(cover));
    }
    scratch.transfer(sum(product(output.capacity(), sizeof(ClosedContactCover)), payload));
    return output;
}
std::size_t ContactCoverIndex::storage_bytes() const noexcept
{
    return impl_->storage_bytes();
}
std::size_t ContactCoverIndex::point_count() const noexcept
{
    return impl_->count;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
