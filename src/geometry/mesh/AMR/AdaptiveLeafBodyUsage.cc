/** @file AdaptiveLeafBodyUsage.cc
 * @brief Page-local owner census and persistent reference count updates.
 */
#include "geometry/mesh/AMR/AdaptiveLeafBodyUsage.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/PLIC/details/GeometryAllocationOwnerAccess.hh"
#include "geometry/mesh/AMR/NativeSourceOwnerStore.hh"
#include "geometry/details/NativeAllocationGraph.hh"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
constexpr auto maximum_bytes = std::numeric_limits<std::size_t>::max();

std::size_t sum(std::size_t a, std::size_t b)
{
    if (b > maximum_bytes - a)
        throw std::overflow_error("Adaptive leaf body accounting overflows.");
    return a + b;
}

struct Scratch
{
    Budget& arena;
    std::size_t bytes = 0;
    ~Scratch() { arena.used -= bytes; }
    void add(std::size_t n)
    {
        arena.add(n);
        bytes += n;
    }
};

struct Charge
{
    std::shared_ptr<Budget> arena;
    std::size_t bytes;

    Charge(std::shared_ptr<Budget> budget, std::size_t size) : arena(std::move(budget)), bytes(size)
    {
        arena->add(bytes);
    }
    Charge(Charge&& other) noexcept : arena(std::move(other.arena)), bytes(std::exchange(other.bytes, 0)) {}
    ~Charge()
    {
        if (arena)
            arena->used -= bytes;
    }
};

} // namespace

struct AdaptiveLeafBodyUsage::Node
{
    using Key = AdaptiveLeafBodyDelta::Key;

    Charge charge;
    Key key;
    using References = std::array<std::size_t, 2>;
    std::size_t payload, payload_sum, leaf_payload_sum, storage_sum;
    References references;
    std::weak_ptr<const void> control;
    NodePtr left, right;
    int height;

    Node(Charge admission, Key owner, std::size_t bytes, References counts, std::weak_ptr<const void> identity,
        NodePtr before, NodePtr after)
        : charge(std::move(admission)), key(owner), payload(bytes), payload_sum(bytes),
          leaf_payload_sum(counts[0] ? bytes : 0), storage_sum(charge.bytes), references(counts), control(std::move(identity)),
          left(std::move(before)), right(std::move(after)), height(1)
    {
        for (const auto* child : {left.get(), right.get()})
            if (child)
            {
                payload_sum = sum(payload_sum, child->payload_sum);
                leaf_payload_sum = sum(leaf_payload_sum, child->leaf_payload_sum);
                storage_sum = sum(storage_sum, child->storage_sum);
                height = std::max(height, child->height + 1);
            }
    }
};

void AdaptiveLeafBodyUsage::visit_storage(const std::function<void(const void*, std::size_t)>& visitor) const
{
    if (membership_)
    {
        membership_->visit_allocations([&](const auto& extent) { visitor(extent.address, extent.bytes); });
        const auto& state = membership_->state();
        for (const auto& extent : {state->allocation, state->counter_allocations.object,
                 state->counter_allocations.control})
            visitor(extent.address, extent.bytes);
    }
    const auto visit = [&](const auto& self, const NodePtr& node) -> void
    {
        if (!node)
            return;
        visitor(node.get(), sizeof(Node));
        self(self, node->left);
        self(self, node->right);
    };
    visit(visit, root_);
}

struct AdaptiveLeafBodyBuilder
{
    using Node = AdaptiveLeafBodyUsage::Node;
    using Ptr = AdaptiveLeafBodyUsage::NodePtr;
    using Key = AdaptiveLeafBodyDelta::Key;
    using References = Node::References;

    const std::shared_ptr<Budget>& arena;

    [[nodiscard]] Ptr make(Key key, std::size_t bytes, References counts, const std::weak_ptr<const void>& control,
        const Ptr& left, const Ptr& right) const
    {
        Charge charge(arena, sizeof(Node) + 4 * sizeof(void*));
        return std::make_shared<const Node>(std::move(charge), key, bytes, counts, control, left, right);
    }

    [[nodiscard]] Ptr make(const Node& value, const Ptr& left, const Ptr& right) const
    {
        return make(value.key, value.payload, value.references, value.control, left, right);
    }

    [[nodiscard]] static int height(const Ptr& node) noexcept { return node ? node->height : 0; }

    [[nodiscard]] Ptr balance(Key key, std::size_t bytes, References counts, const std::weak_ptr<const void>& control,
        const Ptr& left, const Ptr& right) const
    {
        if (height(left) > height(right) + 1)
        {
            if (height(left->right) > height(left->left))
            {
                const auto& pivot = left->right;
                return make(*pivot, make(*left, left->left, pivot->left),
                    make(key, bytes, counts, control, pivot->right, right));
            }
            return make(*left, left->left, make(key, bytes, counts, control, left->right, right));
        }
        if (height(right) > height(left) + 1)
        {
            if (height(right->left) > height(right->right))
            {
                const auto& pivot = right->left;
                return make(*pivot, make(key, bytes, counts, control, left, pivot->left),
                    make(*right, pivot->right, right->right));
            }
            return make(*right, make(key, bytes, counts, control, left, right->left), right->right);
        }
        return make(key, bytes, counts, control, left, right);
    }

    [[nodiscard]] Ptr set(const Ptr& node, Key key, std::size_t bytes, References counts, const std::weak_ptr<const void>& control) const
    {
        if (!node)
            return make(key, bytes, counts, control, {}, {});
        if (key < node->key)
            return balance(node->key, node->payload, node->references, node->control,
                set(node->left, key, bytes, counts, control), node->right);
        if (node->key < key)
            return balance(node->key, node->payload, node->references, node->control,
                node->left, set(node->right, key, bytes, counts, control));
        if (counts[0] || counts[1])
            return make(key, bytes, counts, control, node->left, node->right);
        if (!node->left || !node->right)
            return node->left ? node->left : node->right;
        const auto* successor = node->right.get();
        while (successor->left)
            successor = successor->left.get();
        return balance(successor->key, successor->payload, successor->references, successor->control, node->left,
            set(node->right, successor->key, successor->payload, {}, successor->control));
    }

    [[nodiscard]] static const Node* find(const Ptr& root, Key key) noexcept
    {
        const auto* node = root.get();
        while (node)
        {
            if (key < node->key)
                node = node->left.get();
            else if (node->key < key)
                node = node->right.get();
            else
                return node;
        }
        return nullptr;
    }

    template<class Iterator> [[nodiscard]] Ptr build(std::size_t size, Iterator& next, std::size_t role) const
    {
        if (!size)
            return {};
        auto left = build(size / 2, next, role);
        const auto& [key, usage] = *next++;
        if (usage.retired || !usage.inserted)
            throw std::logic_error("Invalid initial adaptive leaf body count.");
        auto right = build(size - size / 2 - 1, next, role);
        References counts{};
        counts[role] = usage.inserted;
        return make(key, usage.payload, counts, usage.control, left, right);
    }
};

std::size_t AdaptiveLeafBodyUsage::payload_bytes() const noexcept
{
    return root_ ? root_->payload_sum : 0;
}

std::size_t AdaptiveLeafBodyUsage::leaf_payload_bytes() const noexcept
{
    return root_ ? root_->leaf_payload_sum : 0;
}

std::size_t AdaptiveLeafBodyUsage::storage_bytes() const noexcept
{
    std::size_t membership_bytes = 0;
    if (membership_)
    {
        const auto& state = membership_->state();
        membership_bytes = sum(membership_->fixed_allocation_bytes(),
            sum(state->allocation_admission_bytes, geometry_detail::NativeAllocationGraphAccess::validate_state(state)));
    }
    return sum(root_ ? root_->storage_sum : 0, sum(membership_bytes, sum(owners_.index_admission_bytes(),
        sum(paid_.metadata_admission_bytes(), administration_.resident_bytes()))));
}

std::size_t AdaptiveLeafBodyUsage::projected_bytes() const
{
    return sum(payload_bytes(), storage_bytes());
}
std::optional<AdaptiveLeafBodyRole> AdaptiveLeafBodyUsage::physical_role(std::uint8_t kind,
    const std::shared_ptr<const void>& owner, std::size_t bytes) const
{
    if (!owner || !owners_.owners[0].admits(0, kind, owner, bytes)) return {};
    const auto* row = AdaptiveLeafBodyBuilder::find(root_, {kind, owner.get()});
    if (!row || row->payload != bytes) return {};
    return row->references[0] ? AdaptiveLeafBodyRole::Leaf : AdaptiveLeafBodyRole::Family;
}
bool AdaptiveLeafBodyUsage::same_membership(const AdaptiveLeafBodyUsage& other) const noexcept
{
    return root_ == other.root_;
}
void AdaptiveLeafBodyUsage::project_administration(PLIC::details::GeometryOwnerGraphEditor& graph, bool retain) const
{
    if (membership_) graph.native_fixed(membership_, retain);
    const auto visit = [&](const auto& self, const NodePtr& node) -> void
    {
        if (!node || !graph.external(node, sizeof(Node) + 4 * sizeof(void*), retain)) return;
        graph.external(node->charge.arena, sizeof(Budget) + 4 * sizeof(void*), retain);
        self(self, node->left);
        self(self, node->right);
    };
    visit(visit, root_);
}
void AdaptiveLeafBodyUsage::visit_role_delta(const AdaptiveLeafBodyUsage& previous, AdaptiveLeafBodyRole role,
    PLIC::details::NativePaidExtentGroup::MemberSink sink) const
{
    const auto selected = [&](const Node* node)
    {
        return node && (role == AdaptiveLeafBodyRole::Leaf ? node->references[0] != 0 :
            node->references[0] == 0 && node->references[1] != 0);
    };
    const auto visit = [&](const auto& self, const NodePtr& node, const NodePtr& other, bool retain) -> void
    {
        if (!node) return;
        const auto* old = AdaptiveLeafBodyBuilder::find(other, node->key);
        if (old == node.get()) return;
        if (selected(node.get()) && !selected(old))
        {
            const auto owner = node->control.lock();
            if (!owner || owner.get() != node->key.address)
                throw std::logic_error("Source body group lost its canonical actual owner control.");
            sink.visit(sink.context, node->key.kind, owner, node->payload, retain);
        }
        self(self, node->left, other, retain);
        self(self, node->right, other, retain);
    };
    visit(visit, root_, previous.root_, true);
    visit(visit, previous.root_, root_, false);
}
bool AdaptiveLeafBodyDelta::Key::operator<(const Key& other) const noexcept
{
    if (kind != other.kind)
        return kind < other.kind;
    return std::less<const void*>{}(address, other.address);
}

AdaptiveLeafBodyDelta::AdaptiveLeafBodyDelta(std::shared_ptr<Budget> arena) : arena_(std::move(arena))
{
    if (!arena_)
        throw std::invalid_argument("Adaptive leaf body changes require a budget arena.");
}

AdaptiveLeafBodyDelta::~AdaptiveLeafBodyDelta()
{
    changes_.clear();
    decltype(projections_){}.swap(projections_);
    arena_->used -= charged_;
}

void AdaptiveLeafBodyDelta::record(std::span<const AdaptiveLeaf> rows, bool inserted)
{
    if (rows.empty())
        return;
    Scratch scratch{*arena_};
    PLIC::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    PLIC::details::GeometryAllocationOwnerAccess::begin(ledger, used, arena_->limit - arena_->used);
    PLIC::details::GeometryOwnerGraphEditor graph({}, 0, used, maximum_bytes, &ledger);
    for (const auto& row : rows)
        NativeSourceOwnerStore::leaf(row, graph, true);
    const auto capture = graph.finish();
    if (projections_.size() == projections_.capacity())
    {
        const auto old = projections_.capacity();
        const auto next = old ? sum(old, old) : 1;
        const auto bytes = next * sizeof(projections_[0]);
        arena_->add(bytes);
        try { projections_.reserve(next); }
        catch (...) { arena_->used -= bytes; throw; }
        if (projections_.capacity() != next)
            throw std::logic_error("Adaptive source projection capacity differs from admission.");
        charged_ += bytes - old * sizeof(projections_[0]);
        arena_->used -= old * sizeof(projections_[0]);
    }
    projections_.emplace_back(rows, inserted);
    scratch.add(sum(ledger.bookkeeping_bytes(), capture.metadata_admission_bytes()));
    capture.visit([&](const PLIC::details::geometry_demand::OwnerRecord& owner)
        {
            const auto bytes = owner.bytes;
            const Key key{owner.key.kind, owner.owner.get()};
            auto found = changes_.lower_bound(key);
            if (found == changes_.end() || key < found->first)
            {
                constexpr auto entry = sizeof(decltype(changes_)::value_type) + 4 * sizeof(void*);
                arena_->add(entry);
                charged_ += entry;
                found = changes_.emplace_hint(found, key, Counts{bytes, 0, 0, owner.owner});
            }
            if (found->second.payload != bytes || found->second.control.owner_before(owner.owner) ||
                owner.owner.owner_before(found->second.control))
                throw std::logic_error("Adaptive leaf body control/capacity differs for the same physical owner.");
            auto& count = inserted ? found->second.inserted : found->second.retired;
            count = sum(count, 1);
        });
}

AdaptiveLeafBodyDelta::Result AdaptiveLeafBodyDelta::apply(const AdaptiveLeafBodyUsage& source,
    AdaptiveLeafBodyRole role, const AdaptiveLeafBodyUsage* accepted) const
{
    const auto prefix = arena_->used;
    const geometry_detail::NativeConstructionAdmission native(
        geometry_detail::NativeConstructionCounterAccess::create<Budget>(arena_->limit, prefix,
            NativeSourceOwnerStore::mirror(arena_)));
    struct PrefixRelease
    {
        const geometry_detail::NativeConstructionAdmission& admission;
        std::size_t bytes;
        ~PrefixRelease() { admission.release_prefix(bytes); }
    } prefix_release{native, prefix};
    std::size_t graph_used = 0, overlap = source.owners_.overlap_bytes();
    PLIC::details::GeometryOwnerGraphEditor ordinary(source.owners_.owners[0], 0, graph_used,
        arena_->limit, nullptr, nullptr, native.state(), &source.paid_);
    PLIC::details::GeometryOwnerGraphEditor bounded(source.owners_.owners[1], 1, graph_used,
        arena_->limit, nullptr, nullptr, native.state());
    ordinary.pair_with(bounded, overlap);
    // Retain replacements before retiring old edges; exact shared controls
    // prune unchanged descriptor/Origin/provider subgraphs on page replacement.
    for (const bool adding : {true, false})
        for (const auto& [rows, inserted] : projections_)
            if (inserted == adding)
                for (const auto& row : rows)
                    NativeSourceOwnerStore::leaf(row, ordinary, adding);
    auto owners = NativeSourceOwnerStore::seal_default(ordinary, bounded, overlap);
    auto paid = ordinary.paid_receipts();
    PLIC::details::GeometryOwnerUnionWork graph_work;
    auto administration = source.administration_.updated(source.owners_, owners, {}, {}, graph_used,
        arena_->limit, native.state(), graph_work, &source.paid_, &paid);
    const AdaptiveLeafBodyBuilder builder{arena_};
    const auto slot = static_cast<std::size_t>(role);
    const auto make_membership = [&]
    {
        using Observed = geometry_detail::NativeAllocationEnvelopeAccess;
        if (source.membership_)
        {
            const auto& state = source.membership_->state();
            const auto& origin = state->mirror.owner;
            if (origin.get() == arena_.get() && !origin.owner_before(arena_) && !arena_.owner_before(origin))
                return Observed::make<unsigned char>(geometry_detail::NativeConstructionAdmission::from_state(state), 0)
                    .envelope;
        }
        // This fixed-control domain has no flattened body-administration
        // entries. Its exact live closure is always visited through the seal.
        const auto occupied = arena_->used;
        const geometry_detail::NativeConstructionAdmission admission(
            geometry_detail::NativeConstructionCounterAccess::create<Budget>(arena_->limit, occupied,
                NativeSourceOwnerStore::mirror(arena_)));
        const PrefixRelease release{admission, occupied};
        return Observed::make<unsigned char>(admission, 0).envelope;
    };
    const auto finish = [&](AdaptiveLeafBodyUsage& usage)
    {
        usage.owners_ = std::move(owners);
        usage.paid_ = std::move(paid);
        usage.administration_ = std::move(administration);
        if (usage.root_ == source.root_)
            usage.membership_ = source.membership_;
        else if (usage.root_)
            usage.membership_ = make_membership();
    };
    if (!source.root_ && !accepted)
    {
        auto next = changes_.begin();
        AdaptiveLeafBodyUsage usage(builder.build(changes_.size(), next, slot));
        finish(usage);
        const auto bytes = usage.payload_bytes();
        return {std::move(usage), bytes, 0};
    }
    auto root = source.root_;
    std::size_t added = 0;
    for (const auto& [key, delta] : changes_)
    {
        const auto* previous = builder.find(source.root_, key);
        auto counts = previous ? previous->references : AdaptiveLeafBodyBuilder::References{};
        const auto old = counts[slot];
        if (delta.retired > old || (previous && previous->payload != delta.payload))
            throw std::logic_error("Adaptive body retirement does not match the source owners.");
        counts[slot] = sum(old - delta.retired, delta.inserted);
        if (counts[slot] == old)
            continue;
        if (!previous && (!accepted || !builder.find(accepted->root_, key)))
            added = sum(added, delta.payload);
        root = builder.set(root, key, delta.payload, counts, delta.control);
    }
    AdaptiveLeafBodyUsage usage(std::move(root));
    finish(usage);
    const auto reused = usage.payload_bytes() - added;
    return {std::move(usage), added, reused};
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
