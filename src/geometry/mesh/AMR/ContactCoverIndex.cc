/** @file ContactCoverIndex.cc
 * @brief Flat balanced bounds tree borrowing normalized point groups.
 */
#include "ContactCoverIndex.hh"
#include "ContactPointFacts.hh"
#include "ContactPointGroups.hh"
#include "ContactSlotPages.hh"
#include "DistributedAdaptiveWire.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
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
        const PointFactDirectory* directory = nullptr;
        union
        {
            ContactPointRef reference{};
            ContactGroupId group;
        };
        const AdaptiveBytes* domain = nullptr;
        std::size_t parent = absent;
    };
    struct Branch
    {
        AdaptiveContactBounds bounds;
        std::size_t parent = absent, left = absent, right = absent;
        std::size_t homogeneous = absent, count = 0, height = 0, mandatory_frontier = 0;
    };
    Budget& budget;
    int owner;
    std::unique_ptr<Leaf[]> leaves;
    std::unique_ptr<Branch[]> branches;
    std::optional<ContactSlotPages<Leaf>> leaf_pages;
    std::optional<ContactSlotPages<Branch>> branch_pages;
    GroupResolver groups{};
    std::size_t capacity = 0, leaf_slots = 0, branch_slots = 0;
    std::size_t free_leaf = absent, free_branch = absent, root = absent, count = 0;
    static std::size_t base_bytes() noexcept { return sizeof(ContactCoverIndex) + sizeof(Impl); }

    Impl(Budget& budget_, int owner_) : budget(budget_), owner(owner_) {}
    Impl(std::shared_ptr<Budget> arena, int owner_, GroupResolver resolver)
        : budget(*arena), owner(owner_), leaf_pages(std::in_place, arena), branch_pages(std::in_place, arena),
          groups(resolver) {}
    Impl(const Impl& source, std::shared_ptr<Budget> arena, GroupResolver resolver)
        : budget(*arena), owner(source.owner), leaf_pages(std::in_place, *source.leaf_pages, arena),
          branch_pages(std::in_place, *source.branch_pages, arena), groups(resolver), capacity(source.capacity),
          leaf_slots(source.leaf_slots), branch_slots(source.branch_slots), free_leaf(source.free_leaf),
          free_branch(source.free_branch), root(source.root), count(source.count) {}
    ~Impl()
    {
        const auto bytes = private_metadata_bytes();
        leaves.reset();
        branches.reset();
        budget.used -= bytes;
    }
    static bool is_leaf(std::size_t tag) noexcept { return !(tag & 1); }
    static std::size_t leaf_tag(std::size_t slot) noexcept { return slot * 2; }
    static std::size_t branch_tag(std::size_t slot) noexcept { return slot * 2 + 1; }
    Leaf& edit_leaf(std::size_t tag)
    {
        return leaf_pages ? leaf_pages->edit(tag / 2) : leaves[tag / 2];
    }
    const Leaf& leaf(std::size_t tag) const
    {
        return leaf_pages ? (*leaf_pages)[tag / 2] : leaves[tag / 2];
    }
    Branch& edit_branch(std::size_t tag)
    {
        return branch_pages ? branch_pages->edit(tag / 2) : branches[tag / 2];
    }
    const Branch& branch(std::size_t tag) const
    {
        return branch_pages ? (*branch_pages)[tag / 2] : branches[tag / 2];
    }
    std::size_t& edit_parent(std::size_t tag)
    {
        return is_leaf(tag) ? edit_leaf(tag).parent : edit_branch(tag).parent;
    }
    std::size_t parent(std::size_t tag) const
    {
        return is_leaf(tag) ? leaf(tag).parent : branch(tag).parent;
    }
    [[nodiscard]] GroupObservation observe(const Leaf& value) const
    {
        if (groups.get)
            return groups.get(groups.state, value.group);
        if (!value.directory)
            return {*value.point, value.domain};
        return {PLIC::details::GeometryIdentityAccess::point(value.directory->get(value.reference)), value.domain};
    }
    [[nodiscard]] const AdaptiveBytes& domain(std::size_t slot) const
    {
        const auto& value = leaf(leaf_tag(slot));
        return groups.get ? *groups.get(groups.state, value.group).domain : *value.domain;
    }
    std::size_t height(std::size_t tag) const noexcept { return is_leaf(tag) ? 1 : branch(tag).height; }
    std::size_t points(std::size_t tag) const noexcept { return is_leaf(tag) ? 1 : branch(tag).count; }
    std::size_t homogeneous(std::size_t tag) const noexcept { return is_leaf(tag) ? tag / 2 : branch(tag).homogeneous; }
    std::size_t frontier_size(std::size_t tag) const noexcept
    {
        return is_leaf(tag) ? 1 : branch(tag).mandatory_frontier;
    }
    AdaptiveContactBounds bounds(std::size_t tag) const noexcept
    {
        if (!is_leaf(tag))
            return branch(tag).bounds;
        const auto point = observe(leaf(tag)).coordinates;
        return {point, point};
    }
    std::size_t private_metadata_bytes() const noexcept
    {
        return leaf_pages ? base_bytes() :
            base_bytes() + capacity * sizeof(Leaf) + (capacity ? capacity - 1 : 0) * sizeof(Branch);
    }
    std::size_t storage_bytes() const noexcept
    {
        return private_metadata_bytes() + (leaf_pages ? leaf_pages->storage_bytes() + branch_pages->storage_bytes() : 0);
    }
    void reserve(std::size_t requested)
    {
        if (requested <= capacity)
            return;
        // This also guarantees that no encoded child can equal absent.
        if (requested > absent / 2)
            throw std::invalid_argument("Contact-cover handle capacity overflows.");
        if (leaf_pages)
        {
            capacity = requested;
            return;
        }
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
    void update(std::size_t tag)
    {
        auto& node = edit_branch(tag);
        node.bounds = united(bounds(node.left), bounds(node.right));
        node.count = points(node.left) + points(node.right);
        node.height = 1 + std::max(height(node.left), height(node.right));
        const auto a = homogeneous(node.left), b = homogeneous(node.right);
        node.homogeneous = a != absent && b != absent && domain(a) == domain(b) ? a : absent;
        node.mandatory_frontier =
            node.homogeneous != absent ? 1 : sum(frontier_size(node.left), frontier_size(node.right));
    }
    void replace_link(std::size_t parent_tag, std::size_t old_tag, std::size_t new_tag)
    {
        if (parent_tag == absent)
            root = new_tag;
        else
        {
            auto& node = edit_branch(parent_tag);
            (node.left == old_tag ? node.left : node.right) = new_tag;
        }
        edit_parent(new_tag) = parent_tag;
    }
    std::size_t rotate_left(std::size_t tag)
    {
        auto& old = edit_branch(tag);
        const auto next_tag = old.right;
        auto& next = edit_branch(next_tag);
        replace_link(old.parent, tag, next_tag);
        old.right = next.left;
        edit_parent(old.right) = tag;
        next.left = tag;
        old.parent = next_tag;
        update(tag);
        update(next_tag);
        return next_tag;
    }
    std::size_t rotate_right(std::size_t tag)
    {
        auto& old = edit_branch(tag);
        const auto next_tag = old.left;
        auto& next = edit_branch(next_tag);
        replace_link(old.parent, tag, next_tag);
        old.left = next.right;
        edit_parent(old.left) = tag;
        next.right = tag;
        old.parent = next_tag;
        update(tag);
        update(next_tag);
        return next_tag;
    }
    void balance(std::size_t tag)
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
    Handle insert(Leaf binding);
    Leaf& checked_leaf(std::size_t handle)
    {
        if (handle >= leaf_slots)
            throw std::invalid_argument("Contact-cover handle is not live.");
        const auto tag = leaf_tag(handle);
        const auto& value = leaf(tag);
        if (groups.get ? !value.group.generation : (!value.point && !value.directory))
            throw std::invalid_argument("Contact-cover handle is not live.");
        return edit_leaf(tag);
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
ContactCoverIndex::ContactCoverIndex(std::shared_ptr<Budget> arena, int owner, GroupResolver groups)
{
    if (!arena || owner < 0 || !groups.state || !groups.get)
        throw std::invalid_argument("Contact-cover group source is invalid.");
    Scratch charge(*arena);
    charge.add(Impl::base_bytes());
    impl_ = std::make_unique<Impl>(arena, owner, groups);
    charge.transfer(Impl::base_bytes());
}
ContactCoverIndex::ContactCoverIndex(
    const ContactCoverIndex& source, std::shared_ptr<Budget> arena, GroupResolver groups)
{
    if (!arena || !source.impl_->groups.get || !groups.state || !groups.get)
        throw std::invalid_argument("Contact-cover group fork is invalid.");
    Scratch charge(*arena);
    charge.add(Impl::base_bytes());
    impl_ = std::make_unique<Impl>(*source.impl_, arena, groups);
    charge.transfer(Impl::base_bytes());
}
ContactCoverIndex::~ContactCoverIndex() = default;
ContactCoverIndex::ContactCoverIndex(const ContactCoverIndex& source, Budget& budget)
    : ContactCoverIndex(budget, source.impl_->owner)
{
    auto& target = *impl_;
    const auto& original = *source.impl_;
    if (original.groups.get)
        throw std::invalid_argument("Group cover fork requires the candidate resolver.");
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
    return insert_binding(point, nullptr, {}, domain);
}
ContactCoverIndex::Handle ContactCoverIndex::insert(
    const PointFactDirectory& directory, ContactPointRef reference, const AdaptiveBytes* domain)
{
    return insert_binding(nullptr, &directory, reference, domain);
}
ContactCoverIndex::Handle ContactCoverIndex::insert_binding(const std::array<long double, 3>* point,
    const PointFactDirectory* directory, ContactPointRef reference, const AdaptiveBytes* domain)
{
    if (impl_->groups.get || (!point && !directory) || !domain)
        throw std::invalid_argument("Invalid borrowed contact-cover binding.");
    Impl::Leaf binding;
    binding.point = point;
    binding.directory = directory;
    binding.reference = reference;
    binding.domain = domain;
    return impl_->insert(binding);
}
ContactCoverIndex::Handle ContactCoverIndex::insert(ContactGroupId id)
{
    if (!impl_->groups.get || !id.generation)
        throw std::invalid_argument("Invalid contact-cover group binding.");
    Impl::Leaf binding;
    binding.group = id;
    return impl_->insert(binding);
}
ContactCoverIndex::Handle ContactCoverIndex::Impl::insert(Leaf binding)
{
    const MetadataEnvironment environment;
    const auto observation = observe(binding);
    validate(&observation.coordinates, observation.domain);
    // Both needed slots exist before any membership or topology changes.
    if (free_leaf == absent && leaf_slots == capacity)
        reserve(capacity ? product(capacity, 2) : 1);
    const auto slot = free_leaf != absent ? free_leaf : leaf_slots++;
    if (free_leaf != absent)
        free_leaf = leaf(leaf_tag(slot)).parent;
    auto& leaf = edit_leaf(leaf_tag(slot));
    leaf = binding;
    const auto tag = Impl::leaf_tag(slot);
    if (root == absent)
        root = tag;
    else
    {
        const auto branch_slot = free_branch != absent ? free_branch : branch_slots++;
        if (free_branch != absent)
            free_branch = branch(branch_tag(branch_slot)).parent;
        auto sibling = root;
        const auto& value = observation.coordinates;
        const AdaptiveContactBounds point_bounds{value, value};
        while (!Impl::is_leaf(sibling))
        {
            const auto& node = branch(sibling);
            const auto a = extent(united(bounds(node.left), point_bounds));
            const auto b = extent(united(bounds(node.right), point_bounds));
            sibling = a < b || (a == b && height(node.left) <= height(node.right)) ? node.left : node.right;
        }
        const auto joining = branch_tag(branch_slot);
        auto& branch = edit_branch(joining);
        branch = {};
        branch.left = sibling;
        branch.right = tag;
        replace_link(parent(sibling), sibling, joining);
        edit_parent(sibling) = joining;
        leaf.parent = joining;
        balance(joining);
    }
    ++count;
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
        auto& parent = tree.edit_branch(parent_tag);
        const auto sibling = parent.left == tag ? parent.right : parent.left;
        const auto grandparent = parent.parent;
        tree.replace_link(grandparent, parent_tag, sibling);
        parent.parent = tree.free_branch;
        tree.free_branch = parent_tag / 2;
        tree.balance(grandparent);
    }
    leaf.point = nullptr;
    leaf.directory = nullptr;
    if (tree.groups.get)
        leaf.group = {};
    else
        leaf.reference = {};
    leaf.domain = nullptr;
    leaf.parent = tree.free_leaf;
    tree.free_leaf = handle;
    --tree.count;
}
void ContactCoverIndex::rebind(Handle handle, const std::array<long double, 3>* point, const AdaptiveBytes* domain)
{
    const MetadataEnvironment environment;
    if (impl_->groups.get)
        throw std::invalid_argument("Group cover entries retain logical bindings.");
    validate(point, domain);
    auto& leaf = impl_->checked_leaf(handle);
    leaf.point = point;
    leaf.directory = nullptr;
    leaf.reference = {};
    leaf.domain = domain;
}
void ContactCoverIndex::rebind(
    Handle handle, const PointFactDirectory& directory, ContactPointRef reference, const AdaptiveBytes* domain)
{
    const MetadataEnvironment environment;
    if (impl_->groups.get)
        throw std::invalid_argument("Group cover entries retain logical bindings.");
    const auto value = directory.get(reference);
    validate(&PLIC::details::GeometryIdentityAccess::point(value), domain);
    auto& leaf = impl_->checked_leaf(handle);
    leaf.point = nullptr;
    leaf.directory = &directory;
    leaf.reference = reference;
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
        const auto& domain = tree.domain(tree.homogeneous(tag));
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
bool ContactCoverIndex::visit_ownership_frontier(
    std::size_t target, std::size_t& cursor, const OwnershipFrontierVisitor& visit) const
{
    const MetadataEnvironment environment;
    if (!target)
        throw std::invalid_argument("Contact-cover frontier needs a positive target.");
    const auto& tree = *impl_;
    if (tree.root == absent)
        return true;
    const auto emit = [&](std::size_t tag)
    {
        const BorrowedOwnershipCover cover{
            tree.domain(tree.homogeneous(tag)), tree.bounds(tag), tree.owner, tree.points(tag)};
        if (!visit(cover))
            return false;
        ++cursor;
        return true;
    };
    if (tree.frontier_size(tree.root) >= target)
    {
        // The mandatory frontier itself reaches the target. Its exact DFS
        // order is already encoded by the persistent tree's children.
        std::size_t skip = cursor;
        const auto walk = [&](auto&& self, std::size_t tag) -> bool
        {
            const auto size = tree.frontier_size(tag);
            if (skip >= size)
            {
                skip -= size;
                return true;
            }
            if (tree.homogeneous(tag) != absent)
                return emit(tag);
            const auto& branch = tree.branch(tag);
            return self(self, branch.left) && self(self, branch.right);
        };
        return walk(walk, tree.root);
    }
    // A small mandatory frontier uses the original soft-split queue order.
    // Scratch is bounded by the requested target, independently of point N.
    Scratch scratch(impl_->budget);
    std::vector<std::size_t> frontier;
    const auto append = [&](std::size_t tag)
    {
        if (frontier.size() == frontier.capacity())
            scratch.reserve(frontier, frontier.capacity() ? product(frontier.capacity(), 2) : 1);
        frontier.push_back(tag);
    };
    const auto mandatory = [&](auto&& self, std::size_t tag) -> void
    {
        if (tree.homogeneous(tag) != absent)
            append(tag);
        else
        {
            self(self, tree.branch(tag).left);
            self(self, tree.branch(tag).right);
        }
    };
    mandatory(mandatory, tree.root);
    auto active = frontier.size();
    for (std::size_t at = 0; active < target && at < frontier.size(); ++at)
    {
        const auto tag = frontier[at];
        if (tag == absent || Impl::is_leaf(tag))
            continue;
        frontier[at] = absent;
        append(tree.branch(tag).left);
        append(tree.branch(tag).right);
        ++active;
    }
    std::size_t ordinal = 0;
    for (const auto tag : frontier)
        if (tag != absent && ordinal++ >= cursor && !emit(tag))
            return false;
    return true;
}
std::size_t ContactCoverIndex::storage_bytes() const noexcept
{
    return impl_->storage_bytes();
}
std::size_t ContactCoverIndex::private_metadata_bytes() const noexcept { return impl_->private_metadata_bytes(); }
std::size_t ContactCoverIndex::overlap_bytes() const noexcept
{
    return impl_->leaf_pages ? impl_->leaf_pages->overlap_bytes() + impl_->branch_pages->overlap_bytes() : 0;
}
std::size_t ContactCoverIndex::copied_pages() const noexcept
{
    return impl_->leaf_pages ? impl_->leaf_pages->copied_pages() + impl_->branch_pages->copied_pages() : 0;
}
void ContactCoverIndex::finish_overlap() noexcept
{
    if (impl_->leaf_pages)
    {
        impl_->leaf_pages->finish_overlap();
        impl_->branch_pages->finish_overlap();
    }
}
void ContactCoverIndex::visit_borrowed_pages(const std::function<void(const void*, bool, std::size_t)>& visit) const
{
    if (impl_->leaf_pages)
    {
        impl_->leaf_pages->visit_pages(visit);
        impl_->branch_pages->visit_pages(visit);
    }
}
std::size_t ContactCoverIndex::point_count() const noexcept
{
    return impl_->count;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
