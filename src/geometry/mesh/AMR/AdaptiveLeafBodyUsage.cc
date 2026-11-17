/** @file AdaptiveLeafBodyUsage.cc
 * @brief Page-local owner census and persistent reference count updates.
 */
#include "geometry/mesh/AMR/AdaptiveLeafBodyUsage.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/PLIC/details/GeometryAllocationOwnerAccess.hh"

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

void retain_body(PLIC::GeometryAllocationLedger& ledger, const AdaptiveLeaf& row, std::size_t& used)
{
    const auto part = [&](const auto& payload)
    {
        if (payload)
            ledger.retain_external_owner(payload, sum(sizeof(AdaptiveBytes), payload->capacity()), used, maximum_bytes);
    };
    if (row.native_recipe)
    {
        const auto& recipe = *row.native_recipe;
        const auto bytes = sum(sum(sizeof(NativeFragmentDescriptor), recipe.view.capacity()),
            recipe.parts.capacity() * sizeof(decltype(recipe.parts)::value_type));
        ledger.retain_external_owner(row.native_recipe, bytes, used, maximum_bytes);
        for (const auto& payload : recipe.parts)
            part(payload);
    }
    part(row.join_recipe);
    if (row.factor_recipe)
    {
        ledger.retain_external_owner(row.factor_recipe, sizeof(AdaptiveFactorRecipe), used, maximum_bytes);
        part(row.factor_recipe->root);
        part(row.factor_recipe->base);
        part(row.factor_recipe->axis);
    }
    if (row.geometry)
        ledger.retain(row.geometry, used, maximum_bytes);
    for (const auto& face : row.surfaces)
    {
        if (face.geometry)
            ledger.retain_planar_face(face.geometry, used, maximum_bytes);
        if (face.native_whole_geometry)
            ledger.retain_planar_face(face.native_whole_geometry, used, maximum_bytes);
    }
}
} // namespace

struct AdaptiveLeafBodyUsage::Node
{
    using Key = AdaptiveLeafBodyDelta::Key;

    Charge charge;
    Key key;
    using References = std::array<std::size_t, 2>;
    std::size_t payload, payload_sum, leaf_payload_sum, storage_sum;
    References references;
    NodePtr left, right;
    int height;

    Node(Charge admission, Key owner, std::size_t bytes, References counts, NodePtr before, NodePtr after)
        : charge(std::move(admission)), key(owner), payload(bytes), payload_sum(bytes),
          leaf_payload_sum(counts[0] ? bytes : 0), storage_sum(charge.bytes), references(counts),
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

    [[nodiscard]] Ptr make(Key key, std::size_t bytes, References counts, const Ptr& left, const Ptr& right) const
    {
        Charge charge(arena, sizeof(Node) + 4 * sizeof(void*));
        return std::make_shared<const Node>(std::move(charge), key, bytes, counts, left, right);
    }

    [[nodiscard]] Ptr make(const Node& value, const Ptr& left, const Ptr& right) const
    {
        return make(value.key, value.payload, value.references, left, right);
    }

    [[nodiscard]] static int height(const Ptr& node) noexcept { return node ? node->height : 0; }

    [[nodiscard]] Ptr balance(Key key, std::size_t bytes, References counts, const Ptr& left, const Ptr& right) const
    {
        if (height(left) > height(right) + 1)
        {
            if (height(left->right) > height(left->left))
            {
                const auto& pivot = left->right;
                return make(*pivot, make(*left, left->left, pivot->left),
                    make(key, bytes, counts, pivot->right, right));
            }
            return make(*left, left->left, make(key, bytes, counts, left->right, right));
        }
        if (height(right) > height(left) + 1)
        {
            if (height(right->left) > height(right->right))
            {
                const auto& pivot = right->left;
                return make(*pivot, make(key, bytes, counts, left, pivot->left),
                    make(*right, pivot->right, right->right));
            }
            return make(*right, make(key, bytes, counts, left, right->left), right->right);
        }
        return make(key, bytes, counts, left, right);
    }

    [[nodiscard]] Ptr set(const Ptr& node, Key key, std::size_t bytes, References counts) const
    {
        if (!node)
            return make(key, bytes, counts, {}, {});
        if (key < node->key)
            return balance(node->key, node->payload, node->references, set(node->left, key, bytes, counts), node->right);
        if (node->key < key)
            return balance(node->key, node->payload, node->references, node->left, set(node->right, key, bytes, counts));
        if (counts[0] || counts[1])
            return make(key, bytes, counts, node->left, node->right);
        if (!node->left || !node->right)
            return node->left ? node->left : node->right;
        const auto* successor = node->right.get();
        while (successor->left)
            successor = successor->left.get();
        return balance(successor->key, successor->payload, successor->references, node->left,
            set(node->right, successor->key, successor->payload, {}));
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
        return make(key, usage.payload, counts, left, right);
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
    return root_ ? root_->storage_sum : 0;
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
    for (const auto& row : rows)
        retain_body(ledger, row, used);
    scratch.add(ledger.bookkeeping_bytes());
    PLIC::details::GeometryAllocationOwnerAccess::visit(ledger,
        [&](std::uint8_t kind, const void* address, std::size_t bytes)
        {
            const Key key{kind, address};
            auto found = changes_.lower_bound(key);
            if (found == changes_.end() || key < found->first)
            {
                constexpr auto entry = sizeof(decltype(changes_)::value_type) + 4 * sizeof(void*);
                arena_->add(entry);
                charged_ += entry;
                found = changes_.emplace_hint(found, key, Counts{bytes});
            }
            if (found->second.payload != bytes)
                throw std::logic_error("Adaptive leaf body capacity differs for the same physical owner.");
            auto& count = inserted ? found->second.inserted : found->second.retired;
            count = sum(count, 1);
        });
}

AdaptiveLeafBodyDelta::Result AdaptiveLeafBodyDelta::apply(const AdaptiveLeafBodyUsage& source,
    AdaptiveLeafBodyRole role, const AdaptiveLeafBodyUsage* accepted) const
{
    const AdaptiveLeafBodyBuilder builder{arena_};
    const auto slot = static_cast<std::size_t>(role);
    if (!source.root_ && !accepted)
    {
        auto next = changes_.begin();
        AdaptiveLeafBodyUsage usage(builder.build(changes_.size(), next, slot));
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
        root = builder.set(root, key, delta.payload, counts);
    }
    AdaptiveLeafBodyUsage usage(std::move(root));
    const auto reused = usage.payload_bytes() - added;
    return {std::move(usage), added, reused};
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
