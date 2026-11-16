/** @file CartesianSourceInventory.hh
 * @brief Persistent Cartesian ancestry selectors and exact streamed inventory identities.
 */
#pragma once
#include "NativeContactFunding.hh"
#include "geometry/mesh/AMR/CartesianAuthorityCredits.hh"
#include "geometry/mesh/AMR/CartesianIncidenceAuthority.hh"
#include "geometry/mesh/AMR/PersistentOrderedPages.hh"
#include <array>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
enum class CartesianDependencyVisit
{
    Flat,
    Physical,
    Observed
};

/** Sized borrowed view of a persistent ordered root. Rows remain valid while
 * its inventory (or enclosing batch) is pinned. No range owns a page. */
template<class Row, class Node> class CartesianInventoryRows
{
public:
    class Iterator
    {
    public:
        using value_type = Row;
        using reference = const Row&;
        using pointer = const Row*;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;
        Iterator() = default;
        reference operator*() const { return CartesianInventoryRows(root_)[index_]; }
        pointer operator->() const { return &**this; }
        Iterator& operator++()
        {
            ++index_;
            return *this;
        }
        Iterator operator++(int)
        {
            auto before = *this;
            ++*this;
            return before;
        }
        bool operator==(const Iterator&) const = default;

    private:
        friend class CartesianInventoryRows;
        Iterator(const Node* root, std::size_t index) : root_(root), index_(index) {}
        const Node* root_ = nullptr;
        std::size_t index_ = 0;
    };
    explicit CartesianInventoryRows(const Node* root) noexcept : root_(root) {}
    [[nodiscard]] std::size_t size() const noexcept { return root_ ? root_->size : 0; }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] Iterator begin() const noexcept { return {root_, 0}; }
    [[nodiscard]] Iterator end() const noexcept { return {root_, size()}; }
    [[nodiscard]] const Row& operator[](std::size_t index) const
    {
        if (index >= size())
            throw std::out_of_range("Cartesian inventory row index is out of bounds.");
        auto* node = root_;
        while (!node->leaf())
        {
            for (const auto& child : node->children)
            {
                if (index < child->size)
                {
                    node = child.get();
                    break;
                }
                index -= child->size;
            }
        }
        return node->rows[index];
    }
    [[nodiscard]] const Row& front() const { return (*this)[0]; }
    [[nodiscard]] const Row& back() const { return (*this)[size() - 1]; }

private:
    const Node* root_;
};

/** Owner-local typed ancestry. The ordered batch/range graph is authoritative;
 * the independent spatial selector index only prunes exact candidate tests.
 * Neither structure owns a predecessor inventory or a mutable editor. */
class CartesianSourceInventory : public std::enable_shared_from_this<CartesianSourceInventory>
{
    struct RangeLess
    {
        bool operator()(const CartesianIncidenceRange&, const CartesianIncidenceRange&) const noexcept;
    };
    using RangeTree = persistent_ordered_detail::OrderedPages<CartesianIncidenceRange, RangeLess>;

public:
    struct CapturedSource
    {
        std::shared_ptr<const CartesianIncidenceSource> source;
        std::fenv_t environment{};
    };
    struct Retirement
    {
        AdaptiveCellKey key;
        std::uint64_t revision = 0;
    };
    class Batch
    {
    public:
        using Ranges = CartesianInventoryRows<CartesianIncidenceRange, RangeTree::Node>;
        [[nodiscard]] const std::shared_ptr<const CartesianIncidenceSource>& source() const noexcept { return source_; }
        [[nodiscard]] const std::fenv_t& evaluation_state() const noexcept { return environment_; }
        [[nodiscard]] Ranges ranges() const noexcept { return Ranges(ranges_.get()); }
        [[nodiscard]] std::size_t cell_count() const noexcept { return cells_; }
        [[nodiscard]] bool contains(const CartesianCellAncestry&) const noexcept;
        [[nodiscard]] std::size_t selector_slot() const noexcept { return slot_; }

    private:
        std::shared_ptr<const CartesianIncidenceSource> source_;
        std::fenv_t environment_{};
        RangeTree::Ptr ranges_;
        std::size_t cells_ = 0, slot_ = 0;
        friend class CartesianSourceInventory;
    };

private:
    struct BatchLess
    {
        bool operator()(const Batch& first, const Batch& second) const noexcept
        {
            return first.selector_slot() < second.selector_slot();
        }
    };
    struct BatchWeights
    {
        static constexpr bool enabled = true;
        struct Summary
        {
            std::size_t cells = 0, maximum_domain = 0, identity_bytes = 0;
        };
        static void add_row(Summary&, const Batch&, const Budget*);
        static void add_child(Summary&, const Summary&, bool);
    };
    using BatchTree = persistent_ordered_detail::OrderedPages<Batch, BatchLess, false,
        persistent_ordered_detail::OrderedPageWork, BatchWeights>;

    /** Endpoint index divided by 2^level. Comparisons never shift a large
     * coordinate upward or approximate an ancestry interval physically. */
    struct Endpoint
    {
        std::uint64_t index = 0;
        unsigned level = 0;
        [[nodiscard]] bool operator<(const Endpoint&) const noexcept;
    };
    struct Selector
    {
        std::uint64_t region = 0;
        std::size_t batch = 0;
        CartesianIncidenceRange range;
    };
    struct SelectorLess
    {
        bool operator()(const Selector&, const Selector&) const noexcept;
    };
    struct SelectorWeights
    {
        static constexpr bool enabled = true;
        struct Summary
        {
            std::uint64_t first_region = 0, last_region = 0;
            std::array<Endpoint, 3> lower{}, upper{};
            bool present = false;
        };
        static void add_row(Summary&, const Selector&, const Budget*);
        static void add_child(Summary&, const Summary&, bool);
    };
    using SelectorTree = persistent_ordered_detail::OrderedPages<Selector, SelectorLess, false,
        persistent_ordered_detail::OrderedPageWork, SelectorWeights>;

public:
    using Batches = CartesianInventoryRows<Batch, BatchTree::Node>;
    /** Only real trusted capsules and exact captured controls enter this graph.
     * Its complete canonical wire order follows source insertion order. */
    [[nodiscard]] static std::shared_ptr<const CartesianSourceInventory> make(ContactHaloContext, int owner,
        std::uint64_t version, std::span<const CapturedSource>, Budget&, std::shared_ptr<Budget> keeper,
        std::shared_ptr<CartesianAuthorityArena> arena);
    [[nodiscard]] std::shared_ptr<const CartesianSourceInventory> derive(ContactHaloContext next, std::uint64_t version,
        std::span<const Retirement>, std::span<const CapturedSource>, Budget&, std::shared_ptr<Budget> keeper,
        std::shared_ptr<CartesianAuthorityArena> arena) const;
    ~CartesianSourceInventory();
    CartesianSourceInventory(const CartesianSourceInventory&) = delete;
    CartesianSourceInventory& operator=(const CartesianSourceInventory&) = delete;
    [[nodiscard]] Batches batches() const noexcept { return Batches(batches_.get()); }
    [[nodiscard]] const Batch* lookup(const AdaptiveCellKey&) const noexcept;
    [[nodiscard]] bool contains(const AdaptiveCellKey& key) const noexcept { return lookup(key) != nullptr; }
    [[nodiscard]] std::uint64_t revision(const AdaptiveCellKey&) const;
    [[nodiscard]] std::size_t cell_count() const noexcept { return batches_ ? batches_->weights.cells : 0; }
    [[nodiscard]] int owner() const noexcept { return owner_; }
    [[nodiscard]] std::uint64_t version() const noexcept { return version_; }
    [[nodiscard]] const ContactHaloContext& context() const noexcept { return context_; }
    [[nodiscard]] std::size_t maximum_domain_bytes() const noexcept
    {
        return batches_ ? batches_->weights.maximum_domain : 0;
    }
    [[nodiscard]] std::size_t identity_size() const noexcept
    {
        return 96 + (batches_ ? batches_->weights.identity_bytes : 0);
    }
    /** Caller admits identity_size() bytes in its writer before emission. */
    void emit_identity(Writer&) const;
    class EncodedIdentity
    {
    public:
        [[nodiscard]] const AdaptiveBytes& bytes() const noexcept { return writer_.data; }

    private:
        friend class CartesianSourceInventory;
        EncodedIdentity(Budget&, const CartesianSourceInventory&);
        EncodedIdentity(const EncodedIdentity&) = delete;
        EncodedIdentity& operator=(const EncodedIdentity&) = delete;
        cartesian_detail::Allocation memory_;
        Writer writer_;
    };
    [[nodiscard]] EncodedIdentity encode_identity(Budget&) const;

    [[nodiscard]] std::size_t metadata_bytes() const noexcept { return charge_; }
    [[nodiscard]] const Budget* allocation_budget() const noexcept { return budget_; }
    [[nodiscard]] bool has_arena_keeper() const noexcept { return bool(keeper_); }
    [[nodiscard]] bool has_observed_dependencies() const noexcept { return observed_dependencies_; }
    [[nodiscard]] bool dependencies_admitted_in(const CartesianAuthorityArena&) const noexcept;
    void retain_dependencies(cartesian_detail::CreditConstruction&,
        std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>>&, cartesian_detail::Allocation&) const;
    struct DependencyWork
    {
        std::size_t allocated_pages = 0, allocated_bytes = 0, inherited_admission_visits = 0;
        std::size_t rebuilt_batches = 0, selector_nodes = 0, selector_rows = 0, selector_tests = 0;
    };
    [[nodiscard]] DependencyWork dependency_work() const noexcept { return work_; }

    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor) const
    {
        contact_payload_object(
            visitor, this, GeometryContactPayloadRole::CartesianInventoryObject, sizeof(*this), true);
        contact_payload_buffer(visitor, this, GeometryContactPayloadRole::CartesianInventoryLeases, leases_);
        const auto visit_page = [&](const auto& node, ContactPayloadRole object, ContactPayloadRole rows)
        {
            contact_payload_object(visitor, node.get(), object, sizeof(*node), true);
            contact_payload_buffer(visitor, node.get(), rows, node->rows);
            contact_payload_buffer(
                visitor, node.get(), GeometryContactPayloadRole::CartesianInventoryDependencyChildren, node->children);
        };
        BatchTree::visit_pages(batches_,
            [&](const auto& node)
            {
                visit_page(node, GeometryContactPayloadRole::CartesianInventoryDependencyNode,
                    GeometryContactPayloadRole::CartesianInventoryBatches);
            });
        for (const auto& batch : batches())
        {
            batch.source_->visit_borrowed_payload(visitor);
            RangeTree::visit_pages(batch.ranges_,
                [&](const auto& node)
                {
                    visit_page(node, GeometryContactPayloadRole::CartesianInventoryDependencyNode,
                        GeometryContactPayloadRole::CartesianInventoryRanges);
                });
        }
        SelectorTree::visit_pages(selectors_,
            [&](const auto& node)
            {
                visit_page(node, GeometryContactPayloadRole::CartesianInventoryDependencyNode,
                    GeometryContactPayloadRole::CartesianInventoryDependencyRows);
            });
        for (const auto& lease : leases_)
            lease->visit_borrowed_payload(visitor);
    }
    template<class Visitor> void visit_dependencies(Visitor&& visitor) const
    {
        for (const auto& batch : batches())
            visitor(batch.source_, batch.source_->allocation_budget(), batch.source_->metadata_bytes(),
                batch.source_->has_arena_keeper());
        for (const auto& lease : leases_)
            visitor(lease, lease->budget, lease->metadata, bool(lease->keeper));
    }
    template<class Visitor> void visit_physical_dependencies(Visitor&& visitor) const
    {
        visit_dependencies(visitor);
        const auto visit = [&](const auto& node)
        { visitor(node, node->charge.arena.get(), node->charge.bytes, bool(keeper_)); };
        BatchTree::visit_pages(batches_, visit);
        for (const auto& batch : batches())
            RangeTree::visit_pages(batch.ranges_, visit);
        SelectorTree::visit_pages(selectors_, visit);
    }
    template<class Visitor> void visit_observed_dependencies(Visitor&& visitor) const
    {
        if (!keeper_)
        {
            visit_dependencies(visitor);
            return;
        }
        const auto visit = [&](const auto& node)
        { return visitor(node, node->charge.arena.get(), node->charge.bytes, true); };
        const auto descend = [&](auto&& self, const auto& node) -> void
        {
            if (node && visit(node))
                for (const auto& child : node->children)
                    self(self, child);
        };
        const auto visit_batches = [&](auto&& self, const BatchTree::Ptr& node) -> void
        {
            if (!node || !visit(node))
                return;
            if (!node->leaf())
            {
                for (const auto& child : node->children)
                    self(self, child);
                return;
            }
            for (const auto& batch : node->rows)
            {
                visitor(batch.source_, batch.source_->allocation_budget(), batch.source_->metadata_bytes(),
                    batch.source_->has_arena_keeper());
                descend(descend, batch.ranges_);
            }
        };
        visit_batches(visit_batches, batches_);
        descend(descend, selectors_);
        for (const auto& lease : leases_)
            visitor(lease, lease->budget, lease->metadata, bool(lease->keeper));
    }

private:
    friend class OwnedSupportMembership;
    struct Builder;
    [[nodiscard]] bool is_added(const Batch& batch) const noexcept { return batch.slot_ >= first_added_slot_; }
    [[nodiscard]] bool derived_from(const std::shared_ptr<const CartesianSourceInventory>&) const noexcept;
    friend class NativeContactFundingAccess;
    static constexpr std::size_t source_control_bytes = 128;
    [[nodiscard]] NativeContactOwnerFunding native_funding(
        const std::shared_ptr<const CartesianSourceInventory>&, std::size_t) const;
    CartesianSourceInventory() = default;
    static std::shared_ptr<const CartesianSourceInventory> construct(const CartesianSourceInventory*,
        ContactHaloContext, int, std::uint64_t, std::span<const Retirement>, std::span<const CapturedSource>, Budget&,
        std::shared_ptr<Budget>, std::shared_ptr<CartesianAuthorityArena>);
    [[nodiscard]] const Selector* select(const AdaptiveCellKey&, DependencyWork* work = nullptr) const noexcept;
    [[nodiscard]] const Batch* batch_at(std::size_t slot) const noexcept;
    ContactHaloContext context_;
    int owner_ = -1;
    std::uint64_t version_ = 0;
    std::size_t first_added_slot_ = 0, next_slot_ = 0;
    bool observed_dependencies_ = false, shared_dependencies_ = false;
    BatchTree::Ptr batches_;
    SelectorTree::Ptr selectors_;
    DependencyWork work_;
    std::weak_ptr<const CartesianSourceInventory> predecessor_;
    PLIC::details::NativeWeakOwnerObservation predecessor_control_;
    std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>> leases_;
    Budget* budget_ = nullptr;
    std::shared_ptr<Budget> keeper_;
    std::size_t charge_ = 0;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
