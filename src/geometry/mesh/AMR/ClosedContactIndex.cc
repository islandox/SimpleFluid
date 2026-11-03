/** @file ClosedContactIndex.cc
 * @brief Incremental owner-local closed AABB tree with checked admission.
 */
#include "geometry/mesh/AMR/ClosedContactIndex.hh"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
bool closed_contact_overlap(const AdaptiveContactBounds& a, const AdaptiveContactBounds& b) noexcept
{
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (a.upper[axis] < b.lower[axis] || b.upper[axis] < a.lower[axis])
            return false;
    return true;
}
bool closed_contact_overlap(const AdaptiveContactImage& a, const AdaptiveContactImage& b) noexcept
{
    if (a.logical && b.logical && a.logical->exact_embedding && b.logical->exact_embedding &&
        a.logical->frame == b.logical->frame && a.logical->period == b.logical->period)
        for (std::size_t axis = 0; axis < 3; ++axis)
            if (a.logical->upper[axis] < b.logical->lower[axis] || b.logical->upper[axis] < a.logical->lower[axis])
                return false;
    return closed_contact_overlap(a.bounds, b.bounds);
}
namespace
{
/** Metadata heuristics may overflow for perfectly valid finite bounds. Keep
 * their rounding and exception state local, including on admission failure. */
class MetadataEnvironment
{
public:
    MetadataEnvironment()
    {
        if (std::fegetenv(&saved_) != 0)
            throw std::runtime_error("Cannot capture closed-contact metadata floating-point environment.");
        std::fenv_t held;
        if (std::feholdexcept(&held) != 0)
        {
            (void) std::fesetenv(&saved_);
            throw std::runtime_error("Cannot isolate closed-contact metadata floating-point exceptions.");
        }
    }
    ~MetadataEnvironment() { (void) std::fesetenv(&saved_); }
    MetadataEnvironment(const MetadataEnvironment&) = delete;
    MetadataEnvironment& operator=(const MetadataEnvironment&) = delete;

private:
    std::fenv_t saved_{};
};
void admit_scratch(std::size_t bytes, std::size_t limit)
{
    if (bytes > limit)
        throw std::invalid_argument("Closed-contact index operation scratch cap exhausted.");
}
std::size_t add(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::invalid_argument("Closed-contact index capacity overflows.");
    return a + b;
}
std::size_t product(std::size_t a, std::size_t b)
{
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("Closed-contact index capacity overflows.");
    return a * b;
}
void validate(const AdaptiveContactBounds& bounds)
{
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (!std::isfinite(bounds.lower[axis]) || !std::isfinite(bounds.upper[axis]) ||
            bounds.upper[axis] < bounds.lower[axis])
            throw std::invalid_argument("Invalid closed-contact coordinate range.");
}
void validate(const AdaptiveContactImage& image)
{
    validate(image.bounds);
    if (image.logical)
    {
        if (image.logical->frame.empty())
            throw std::invalid_argument("Closed-contact logical frame is empty.");
        for (std::size_t axis = 0; axis < 3; ++axis)
            if (!image.logical->period[axis] || image.logical->upper[axis] < image.logical->lower[axis] ||
                image.logical->upper[axis] > image.logical->period[axis])
                throw std::invalid_argument("Invalid closed-contact logical range.");
    }
}
AdaptiveContactBounds united(const AdaptiveContactBounds& a, const AdaptiveContactBounds& b) noexcept
{
    AdaptiveContactBounds bounds;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        bounds.lower[axis] = std::min(a.lower[axis], b.lower[axis]);
        bounds.upper[axis] = std::max(a.upper[axis], b.upper[axis]);
    }
    return bounds;
}
long double extent(const AdaptiveContactBounds& bounds) noexcept
{
    // A sum avoids products overflowing for large finite coordinates. Infinite
    // sums merely tie the insertion heuristic; they never affect culling.
    long double value = 0;
    for (std::size_t axis = 0; axis < 3; ++axis)
        value += bounds.upper[axis] - bounds.lower[axis];
    return value;
}
std::size_t key_storage(const AdaptiveCellKey& key)
{
    return std::visit(
        [](const auto& ancestry) -> std::size_t
        {
            using Ancestry = std::decay_t<decltype(ancestry)>;
            if constexpr (std::is_same_v<Ancestry, ConvexCellAncestry>)
                return product(ancestry.path.capacity(), sizeof(std::uint64_t));
            else if constexpr (std::is_same_v<Ancestry, ProductCellAncestry>)
                return product(
                    add(ancestry.base.path.capacity(), ancestry.axial.path.capacity()), sizeof(std::uint64_t));
            else
                return 0;
        },
        key.ancestry);
}
} // namespace

struct ClosedContactIndex::Impl
{
    struct Node;
    struct State
    {
        ClosedContactEntry entry;
        std::vector<Node*> leaves;
        std::size_t bytes = 0;
    };
    struct Node
    {
        Node* parent = nullptr;
        std::unique_ptr<Node> left, right;
        AdaptiveContactBounds bounds;
        State* state = nullptr;
        std::size_t image = 0, count = 1;
        unsigned height = 1;
        const State* homogeneous = nullptr;
    };
    struct Prepared
    {
        std::unique_ptr<State> state;
        std::vector<std::unique_ptr<Node>> leaves, branches;
    };
    std::map<AdaptiveCellKey, std::unique_ptr<State>> records;
    std::unique_ptr<Node> root;
    std::size_t limit;
    ClosedContactIndexStatistics stats;

    explicit Impl(std::size_t budget) : limit(budget)
    {
        if (!budget)
            throw std::invalid_argument("Closed-contact index needs a finite byte budget.");
        admit(sizeof(Impl));
        stats.retained_bytes = sizeof(Impl);
    }
    void admit(std::size_t bytes, std::size_t scratch = 0)
    {
        const auto live = add(stats.retained_bytes, scratch);
        if (live > limit || bytes > limit - live)
            throw std::invalid_argument("Closed-contact index live byte budget exhausted.");
        stats.admitted_bytes = add(stats.admitted_bytes, bytes);
        stats.peak_bytes = std::max(stats.peak_bytes, live + bytes);
    }
    static unsigned height(const std::unique_ptr<Node>& node) noexcept { return node ? node->height : 0; }
    static void update(Node& node) noexcept
    {
        if (node.state)
        {
            node.bounds = node.state->entry.images[node.image].bounds;
            node.count = node.height = 1;
            node.homogeneous = node.state;
            return;
        }
        node.bounds = united(node.left->bounds, node.right->bounds);
        node.count = node.left->count + node.right->count;
        node.height = 1 + std::max(node.left->height, node.right->height);
        const auto* a = node.left->homogeneous;
        const auto* b = node.right->homogeneous;
        node.homogeneous =
            a && b && a->entry.owner == b->entry.owner && a->entry.domain == b->entry.domain ? a : nullptr;
    }
    std::unique_ptr<Node>& link(Node* node) noexcept
    {
        if (!node->parent)
            return root;
        return node->parent->left.get() == node ? node->parent->left : node->parent->right;
    }
    Node* rotate_left(Node* node) noexcept
    {
        auto& target = link(node);
        auto old = std::move(target);
        auto next = std::move(old->right);
        auto* parent = old->parent;
        old->right = std::move(next->left);
        old->right->parent = old.get();
        next->left = std::move(old);
        next->left->parent = next.get();
        next->parent = parent;
        update(*next->left);
        update(*next);
        target = std::move(next);
        return target.get();
    }
    Node* rotate_right(Node* node) noexcept
    {
        auto& target = link(node);
        auto old = std::move(target);
        auto next = std::move(old->left);
        auto* parent = old->parent;
        old->left = std::move(next->right);
        old->left->parent = old.get();
        next->right = std::move(old);
        next->right->parent = next.get();
        next->parent = parent;
        update(*next->right);
        update(*next);
        target = std::move(next);
        return target.get();
    }
    void balance(Node* node) noexcept
    {
        while (node)
        {
            update(*node);
            if (!node->state)
            {
                const int skew = int(height(node->right)) - int(height(node->left));
                if (skew > 1)
                {
                    if (height(node->right->left) > height(node->right->right))
                        rotate_right(node->right.get());
                    node = rotate_left(node);
                }
                else if (skew < -1)
                {
                    if (height(node->left->right) > height(node->left->left))
                        rotate_left(node->left.get());
                    node = rotate_right(node);
                }
            }
            node = node->parent;
        }
    }
    Prepared prepare(ClosedContactEntry entry)
    {
        if (entry.domain.empty() || entry.owner < 0 || entry.images.empty())
            throw std::invalid_argument("Closed-contact entry needs domain, owner and complete images.");
        std::size_t bytes = sizeof(State) + sizeof(decltype(records)::value_type) + 4 * sizeof(void*);
        bytes = add(bytes, product(2, key_storage(entry.key)));
        bytes = add(bytes, entry.domain.capacity());
        bytes = add(bytes, product(entry.images.capacity(), sizeof(AdaptiveContactImage)));
        bytes = add(bytes, product(entry.images.size(), sizeof(Node*) + 2 * sizeof(Node)));
        for (const auto& image : entry.images)
        {
            validate(image);
            bytes = add(bytes, image.physical_to_contact.payload_bytes());
            if (image.logical)
            {
                bytes = add(bytes, image.logical->frame.capacity());
            }
        }
        // Stage every potentially allocating operation before changing tree
        // membership. Failed admission leaves the old index queryable.
        admit(add(bytes, product(entry.images.size(), 2 * sizeof(std::unique_ptr<Node>))));
        Prepared pending;
        pending.state = std::make_unique<State>();
        pending.state->entry = std::move(entry);
        pending.state->bytes = bytes;
        const auto count = pending.state->entry.images.size();
        pending.state->leaves.reserve(count);
        pending.leaves.reserve(count);
        pending.branches.reserve(count);
        for (std::size_t image = 0; image < count; ++image)
        {
            auto node = std::make_unique<Node>();
            node->state = pending.state.get();
            node->image = image;
            update(*node);
            pending.state->leaves.push_back(node.get());
            pending.leaves.push_back(std::move(node));
            pending.branches.push_back(std::make_unique<Node>());
        }
        return pending;
    }
    void attach(Prepared& pending) noexcept
    {
        for (std::size_t image = 0; image < pending.leaves.size(); ++image)
        {
            auto leaf = std::move(pending.leaves[image]);
            if (!root)
            {
                root = std::move(leaf);
                continue;
            }
            auto* sibling = root.get();
            while (!sibling->state)
            {
                const auto a = extent(united(sibling->left->bounds, leaf->bounds));
                const auto b = extent(united(sibling->right->bounds, leaf->bounds));
                sibling = a < b || (a == b && sibling->left->height <= sibling->right->height)
                              ? sibling->left.get()
                              : sibling->right.get();
            }
            auto& target = link(sibling);
            auto branch = std::move(pending.branches[image]);
            branch->parent = sibling->parent;
            branch->left = std::move(target);
            branch->right = std::move(leaf);
            branch->left->parent = branch.get();
            branch->right->parent = branch.get();
            target = std::move(branch);
            balance(target.get());
        }
    }
    void detach(State& state) noexcept
    {
        for (auto* leaf : state.leaves)
        {
            auto* parent = leaf->parent;
            if (!parent)
            {
                root.reset();
                continue;
            }
            auto& target = link(parent);
            auto old = std::move(target);
            auto sibling = old->left.get() == leaf ? std::move(old->right) : std::move(old->left);
            sibling->parent = old->parent;
            target = std::move(sibling);
            balance(target->parent);
        }
    }
    template<class T>
    void append(std::vector<T>& output, T value, std::size_t payload = 0,
        std::size_t scratch_cap = std::numeric_limits<std::size_t>::max())
    {
        if (output.size() == output.capacity())
        {
            const auto capacity = output.capacity() ? product(output.capacity(), 2) : 1;
            const auto previous = add(product(output.capacity(), sizeof(T)), payload);
            const auto next = product(capacity, sizeof(T));
            admit_scratch(add(previous, next), scratch_cap);
            admit(next, previous);
            output.reserve(capacity);
        }
        output.push_back(std::move(value));
    }
};

ClosedContactIndex::ClosedContactIndex(std::size_t budget) : impl_(std::make_unique<Impl>(budget)) {}
ClosedContactIndex::~ClosedContactIndex() = default;
ClosedContactIndex::ClosedContactIndex(ClosedContactIndex&&) noexcept = default;
ClosedContactIndex& ClosedContactIndex::operator=(ClosedContactIndex&&) noexcept = default;
void ClosedContactIndex::insert(ClosedContactEntry entry)
{
    const MetadataEnvironment environment;
    if (impl_->records.contains(entry.key))
        throw std::invalid_argument("Duplicate closed-contact cell key.");
    auto pending = impl_->prepare(std::move(entry));
    auto key = pending.state->entry.key;
    auto [position, inserted] = impl_->records.emplace(std::move(key), std::move(pending.state));
    (void) inserted;
    impl_->stats.retained_bytes += position->second->bytes;
    ++impl_->stats.entries;
    impl_->stats.images += position->second->entry.images.size();
    impl_->attach(pending);
}
void ClosedContactIndex::erase(const AdaptiveCellKey& key)
{
    const MetadataEnvironment environment;
    const auto found = impl_->records.find(key);
    if (found == impl_->records.end())
        return;
    impl_->detach(*found->second);
    impl_->stats.retained_bytes -= found->second->bytes;
    --impl_->stats.entries;
    impl_->stats.images -= found->second->entry.images.size();
    impl_->records.erase(found);
}
void ClosedContactIndex::replace(ClosedContactEntry entry)
{
    const MetadataEnvironment environment;
    const auto found = impl_->records.find(entry.key);
    if (found == impl_->records.end())
        throw std::invalid_argument("Replacing an absent closed-contact cell key.");
    if (entry.revision < found->second->entry.revision)
        throw std::invalid_argument("Replacing a closed-contact entry with a stale revision.");
    auto pending = impl_->prepare(std::move(entry));
    impl_->detach(*found->second);
    impl_->stats.retained_bytes -= found->second->bytes;
    impl_->stats.images -= found->second->entry.images.size();
    found->second = std::move(pending.state);
    impl_->stats.retained_bytes += found->second->bytes;
    impl_->stats.images += found->second->entry.images.size();
    impl_->attach(pending);
}
const ClosedContactEntry* ClosedContactIndex::find(const AdaptiveCellKey& key) const
{
    const auto found = impl_->records.find(key);
    return found == impl_->records.end() ? nullptr : &found->second->entry;
}
std::vector<const ClosedContactEntry*> ClosedContactIndex::query(
    std::span<const char> domain, std::span<const AdaptiveContactImage> images, std::size_t scratch_cap)
{
    const MetadataEnvironment environment;
    ++impl_->stats.queries;
    std::vector<const ClosedContactEntry*> output;
    for (const auto& image : images)
    {
        validate(image);
        auto visit = [&](auto&& self, const Impl::Node* node) -> void
        {
            if (!node)
                return;
            ++impl_->stats.visited_nodes;
            if (!closed_contact_overlap(image.bounds, node->bounds))
                return;
            if (node->state)
            {
                ++impl_->stats.scanned_entries;
                const auto& entry = node->state->entry;
                if (std::ranges::equal(domain, entry.domain) &&
                    closed_contact_overlap(image, entry.images[node->image]))
                    impl_->append(output, &entry, 0, scratch_cap);
                return;
            }
            self(self, node->left.get());
            self(self, node->right.get());
        };
        visit(visit, impl_->root.get());
    }
    std::sort(output.begin(), output.end(), [](const auto* a, const auto* b) { return a->key < b->key; });
    output.erase(std::unique(output.begin(), output.end()), output.end());
    return output;
}
std::vector<ClosedContactCover> ClosedContactIndex::ownership_covers(
    std::size_t maximum_fragment_images, std::size_t scratch_cap)
{
    const MetadataEnvironment environment;
    if (!maximum_fragment_images)
        throw std::invalid_argument("Closed-contact covers need a positive fragment limit.");
    std::vector<ClosedContactCover> output;
    std::size_t output_payload = 0;
    auto visit = [&](auto&& self, const Impl::Node* node) -> void
    {
        if (!node)
            return;
        if (node->homogeneous && node->count <= maximum_fragment_images)
        {
            const auto& entry = node->homogeneous->entry;
            const auto previous = add(output_payload, product(output.capacity(), sizeof(ClosedContactCover)));
            admit_scratch(add(previous, entry.domain.size()), scratch_cap);
            impl_->admit(entry.domain.size(), previous);
            output_payload = add(output_payload, entry.domain.size());
            impl_->append(
                output, ClosedContactCover{entry.domain, node->bounds, entry.owner, node->count}, output_payload,
                scratch_cap);
            return;
        }
        self(self, node->left.get());
        self(self, node->right.get());
    };
    visit(visit, impl_->root.get());
    return output;
}
const ClosedContactIndexStatistics& ClosedContactIndex::statistics() const noexcept { return impl_->stats; }
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
