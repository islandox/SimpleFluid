/** @file AdaptiveFamilyLayout.hh
 * @brief Immutable ordered owner-local saved families.
 */
#pragma once
#include "geometry/PLIC/GeometryPacketCodec.hh"
#include "geometry/mesh/AMR/AdaptiveLeafPages.hh"
#include "geometry/mesh/AMR/ContactGroupAllocation.hh"
#include "geometry/mesh/AMR/PersistentOrderedPages.hh"

#include <tuple>

namespace SimpleFluid::Meshes::AMR
{
/** A borrowed family record. The enclosing layout owns its arrays and parent.
 * Product directory revisions follow the layout epoch; parent revisions are
 * part of the saved provider witness. */
struct AdaptiveFamilyView
{
    const AdaptiveBytes& key;
    const AdaptiveLeaf& parent;
    const std::vector<AdaptiveCellKey>& children;
    const AdaptiveBytes& anchor;
    AdaptiveSplit split;
    std::uint64_t revision;

    /** Explicit owning copy for one requested capsule. Admit its copy first
     * when using a bounded operation. */
    [[nodiscard]] AdaptiveFamily materialize() const { return {key, parent, children, anchor, split, revision}; }
};

namespace adaptive_detail
{
class AdaptiveFamilyStorage;
}

/** Borrowed ordered queries over an immutable, co-published family and anchor
 * index. Keep this layout alive while using an iterator, view or reference. */
class AdaptiveFamilyLayout
{
    friend class adaptive_detail::AdaptiveFamilyStorage;
    using Budget = adaptive_detail::Budget;
    using Charge = adaptive_detail::persistent_ordered_detail::PageCharge;
    struct Record
    {
        Charge charge;
        const AdaptiveFamily value;
        Record(const std::shared_ptr<Budget>& arena, const AdaptiveFamily& family) : charge(arena), value(family) {}
    };
    using Row = std::shared_ptr<const Record>;

    struct FamilyLess
    {
        bool operator()(const Row& a, const Row& b) const noexcept { return a->value.key < b->value.key; }
    };
    struct AnchorLess
    {
        bool operator()(const Row& a, const Row& b) const noexcept
        {
            return a->value.anchor == b->value.anchor ? a->value.key < b->value.key : a->value.anchor < b->value.anchor;
        }
    };
    struct Weights
    {
        static constexpr bool enabled = true;
        struct Summary
        {
            std::size_t total = 0, local = 0;
        };
        static void add_row(Summary& sum, const Row& row, const Budget* arena)
        {
            using adaptive_detail::persistent_ordered_detail::checked_sum;
            sum.total = checked_sum(sum.total, row->charge.bytes);
            if (row->charge.arena.get() == arena)
                sum.local = checked_sum(sum.local, row->charge.bytes);
        }
        static void add_child(Summary& sum, const Summary& child, bool same_arena)
        {
            using adaptive_detail::persistent_ordered_detail::checked_sum;
            sum.total = checked_sum(sum.total, child.total);
            if (same_arena)
                sum.local = checked_sum(sum.local, child.local);
        }
    };
    using Primary = adaptive_detail::persistent_ordered_detail::OrderedPages<Row, FamilyLess, false,
        adaptive_detail::persistent_ordered_detail::OrderedPageWork, Weights>;
    using Secondary = adaptive_detail::persistent_ordered_detail::OrderedPages<Row, AnchorLess>;
    using PrimaryNode = typename Primary::Node;

    [[nodiscard]] AdaptiveFamilyView view(const AdaptiveFamily& row) const noexcept
    {
        return {row.key, row.parent, row.children, row.anchor, row.split,
            row.parent.provider == AdaptiveProviderKind::Product ? topology_ : row.revision};
    }
    [[nodiscard]] static std::pair<const PrimaryNode*, std::size_t> locate(
        const PrimaryNode* page, std::size_t ordinal) noexcept
    {
        while (page && !page->leaf())
        {
            auto child = page->children.begin();
            while (ordinal >= (*child)->size)
            {
                ordinal -= (*child)->size;
                ++child;
            }
            page = child->get();
        }
        return {page, ordinal};
    }
    [[nodiscard]] const Row* record(const AdaptiveBytes& key) const noexcept
    {
        const auto* found = Primary::lower_bound(primary_, [&](const Row& row) { return row->value.key < key; });
        return found && (*found)->value.key == key ? found : nullptr;
    }

public:
    class const_iterator
    {
    public:
        using value_type = AdaptiveFamilyView;
        using reference = value_type;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;
        using iterator_concept = std::forward_iterator_tag;

        const_iterator() = default;
        [[nodiscard]] value_type operator*() const { return layout_->view(page_->rows[offset_]->value); }
        const_iterator& operator++()
        {
            ++ordinal_;
            if (ordinal_ == layout_->size())
            {
                page_ = nullptr;
                offset_ = 0;
            }
            else if (++offset_ == page_->rows.size())
                std::tie(page_, offset_) = locate(layout_->primary_.get(), ordinal_);
            return *this;
        }
        const_iterator operator++(int)
        {
            auto previous = *this;
            ++*this;
            return previous;
        }
        bool operator==(const const_iterator&) const = default;

    private:
        friend class AdaptiveFamilyLayout;
        const_iterator(const AdaptiveFamilyLayout* layout, std::size_t ordinal) : layout_(layout), ordinal_(ordinal)
        {
            if (ordinal < layout->size())
                std::tie(page_, offset_) = locate(layout->primary_.get(), ordinal);
        }

        const AdaptiveFamilyLayout* layout_ = nullptr;
        const PrimaryNode* page_ = nullptr;
        std::size_t offset_ = 0, ordinal_ = 0;
    };

    AdaptiveFamilyLayout(AdaptiveFamilyLayout&&) noexcept = default;
    AdaptiveFamilyLayout(const AdaptiveFamilyLayout&) = delete;
    AdaptiveFamilyLayout& operator=(const AdaptiveFamilyLayout&) = delete;

    [[nodiscard]] const_iterator begin() const noexcept { return {this, 0}; }
    [[nodiscard]] const_iterator end() const noexcept { return {this, size()}; }
    [[nodiscard]] std::size_t size() const noexcept { return primary_ ? primary_->size : 0; }
    [[nodiscard]] bool empty() const noexcept { return !primary_; }
    [[nodiscard]] bool contains(const AdaptiveBytes& key) const noexcept { return record(key) != nullptr; }
    [[nodiscard]] std::optional<AdaptiveFamilyView> find(const AdaptiveBytes& key) const noexcept
    {
        if (const auto* found = record(key))
            return view((*found)->value);
        return {};
    }
    [[nodiscard]] AdaptiveFamilyView at(const AdaptiveBytes& key) const
    {
        if (const auto found = find(key))
            return *found;
        throw std::out_of_range("Saved adaptive family is absent.");
    }
    [[nodiscard]] AdaptiveFamilyView at_ordinal(std::size_t ordinal) const
    {
        if (ordinal >= size())
            throw std::out_of_range("Saved adaptive family ordinal is absent.");
        const auto [page, offset] = locate(primary_.get(), ordinal);
        return view(page->rows[offset]->value);
    }
    template<class Visit> void visit_anchored(const AdaptiveBytes& anchor, Visit&& visit) const
    {
        (void) Secondary::visit_from(
            anchors_, [&](const Row& row) { return row->value.anchor < anchor; },
            [&](const Row& row)
            {
                if (row->value.anchor != anchor)
                    return false;
                visit(view(row->value));
                return true;
            });
    }

private:
    explicit AdaptiveFamilyLayout(std::shared_ptr<Budget> arena, std::uint64_t topology)
        : inherited_(std::move(arena)), topology_(topology)
    {
    }
    // The inherited allowance outlives both roots.
    adaptive_detail::ContactGroupAllocation inherited_;
    typename Primary::Ptr primary_;
    typename Secondary::Ptr anchors_;
    std::uint64_t topology_;
};

namespace adaptive_detail
{
/** Build and edit private roots. The family-key and canonical encoded-anchor
 * selectors share the exact same record. The primary accounts each record
 * once; its parent bodies belong to the hierarchy's AdaptiveLeafBodyUsage.
 * Provider proof/anchor-owner validation precedes these storage operations. */
class AdaptiveFamilyStorage
{
    using Layout = AdaptiveFamilyLayout;
    using Row = Layout::Row;
    using Primary = Layout::Primary;
    using Secondary = Layout::Secondary;
    using Charge = Layout::Charge;
    using Record = Layout::Record;

public:
    struct Edit
    {
        const AdaptiveBytes& key;
        std::optional<std::uint64_t> expected_revision;
        const AdaptiveFamily* replacement = nullptr;
    };
    struct Work
    {
        std::size_t copied_pages = 0, copied_page_bytes = 0, records = 0, record_bytes = 0;
    };

    [[nodiscard]] static Layout build(
        std::span<const AdaptiveFamily> families, std::shared_ptr<Budget> arena, std::uint64_t topology)
    {
        auto row = families.begin();
        return build_from(
            families.size(), [&]() -> const AdaptiveFamily& { return *row++; }, std::move(arena), topology);
    }

    template<class Next>
    [[nodiscard]] static Layout build_from(
        std::size_t count, Next&& family, std::shared_ptr<Budget> arena, std::uint64_t topology)
    {
        Layout next(std::move(arena), topology);
        Charge scratch(next.inherited_.keeper());
        std::vector<Row> rows;
        scratch.add(persistent_ordered_detail::checked_product(count, sizeof(Row)));
        rows.reserve(count);
        if (rows.capacity() != count)
            throw std::invalid_argument("Family bootstrap capacity differs from admission.");
        for (std::size_t i = 0; i < count; ++i)
            rows.push_back(make_record(family(), next.inherited_.keeper()));
        std::sort(rows.begin(), rows.end(), Layout::FamilyLess{});
        next.primary_ = Primary(next.inherited_.keeper()).build(rows);
        std::sort(rows.begin(), rows.end(), Layout::AnchorLess{});
        next.anchors_ = Secondary(next.inherited_.keeper()).build(rows);
        return next;
    }

    /** Edits must be unique and in family-key order. The fresh arena first
     * admits the accepted row/index union, including paths retired by the
     * edit. Use an arena distinct from all source-lineage arenas. Finish
     * overlap after the enclosing operation accepts. The parent
     * callback sees every changed row once after both private roots succeed.
     * It may update private AdaptiveLeafBodyDelta scratch. Discard that scratch
     * if this operation throws. Neither root is published before return. */
    template<class Observe>
    [[nodiscard]] static Layout edited(const Layout& source, std::span<const Edit> edits, std::shared_ptr<Budget> arena,
        std::uint64_t topology, Work& work, Observe&& observe)
    {
        using persistent_ordered_detail::checked_sum;
        if (arena.get() == &source.inherited_.arena() || topology < source.topology_)
            throw std::invalid_argument("Family edits require a fresh arena and nondecreasing topology.");
        Layout next(std::move(arena), topology);
        next.inherited_.add(storage_bytes(source));
        next.primary_ = source.primary_;
        next.anchors_ = source.anchors_;
        persistent_ordered_detail::OrderedPageWork primary_work, anchor_work;
        const Primary primary(next.inherited_.keeper(), &primary_work);
        const Secondary secondary(next.inherited_.keeper(), &anchor_work);
        Work staged;
        const AdaptiveBytes* previous = nullptr;
        for (const auto& edit : edits)
        {
            if (previous && !(*previous < edit.key))
                throw std::invalid_argument("Saved family edits are not strictly ordered.");
            previous = &edit.key;
            const auto* found = source.record(edit.key);
            if (bool(found) != edit.expected_revision.has_value() || (!found && !edit.replacement) ||
                (found && source.view((*found)->value).revision != *edit.expected_revision) ||
                (edit.replacement && edit.replacement->key != edit.key))
                throw std::invalid_argument("Saved family edit does not match its accepted record.");
            Row replacement;
            if (edit.replacement)
            {
                replacement = make_record(*edit.replacement, next.inherited_.keeper());
                ++staged.records;
                staged.record_bytes = checked_sum(staged.record_bytes, replacement->charge.bytes);
            }
            const auto& selector = found ? *found : replacement;
            next.primary_ =
                primary.edit(next.primary_, selector, replacement ? &replacement : nullptr, found != nullptr);
            if (found && replacement && (*found)->value.anchor == replacement->value.anchor)
                next.anchors_ = secondary.edit(next.anchors_, *found, &replacement, true);
            else
            {
                if (found)
                    next.anchors_ = secondary.edit(next.anchors_, *found, nullptr, true);
                if (replacement)
                    next.anchors_ = secondary.edit(next.anchors_, replacement, &replacement, false);
            }
        }
        if ((next.anchors_ ? next.anchors_->size : 0) != next.size())
            throw std::logic_error("Saved family selectors differ in size.");
        staged.copied_pages = checked_sum(primary_work.copied_pages, anchor_work.copied_pages);
        staged.copied_page_bytes = checked_sum(primary_work.copied_bytes, anchor_work.copied_bytes);
        for (const auto& edit : edits)
        {
            const auto* before = source.record(edit.key);
            const auto* after = next.record(edit.key);
            observe(before ? &(*before)->value.parent : nullptr, after ? &(*after)->value.parent : nullptr);
        }
        work = staged;
        return next;
    }

    /** Deep buffers copied by materialize(). A retaining container admits
     * its object slots separately. */
    [[nodiscard]] static std::size_t copy_buffer_bytes(const AdaptiveFamilyView& family) { return buffers(family); }

    [[nodiscard]] static std::size_t storage_bytes(const Layout& layout) noexcept
    {
        return (layout.primary_ ? layout.primary_->subtree_bytes + layout.primary_->weights.total : 0) +
               (layout.anchors_ ? layout.anchors_->subtree_bytes : 0);
    }
    [[nodiscard]] static std::size_t inherited_bytes(const Layout& layout) noexcept
    {
        auto bytes = storage_bytes(layout);
        const auto& arena = layout.inherited_.keeper();
        if (layout.primary_ && layout.primary_->charge.arena == arena)
            bytes -= layout.primary_->same_arena_bytes + layout.primary_->weights.local;
        if (layout.anchors_ && layout.anchors_->charge.arena == arena)
            bytes -= layout.anchors_->same_arena_bytes;
        return bytes;
    }
    [[nodiscard]] static std::size_t overlap_bytes(const Layout& layout) noexcept
    {
        return layout.inherited_.bytes() - inherited_bytes(layout);
    }
    static void finish_overlap(Layout& layout) noexcept { layout.inherited_.release(overlap_bytes(layout)); }

    /** Diagnostic physical union, including the admission allowance for each
     * allocated node/control. Parent bodies are a separate census. */
    template<class Visit> static void visit_storage(const Layout& layout, const Visit& visit)
    {
        Primary::visit_pages(layout.primary_,
            [&](const auto& page)
            {
                visit(page.get(), page->charge.bytes);
                for (const auto& row : page->rows)
                    visit(row.get(), row->charge.bytes);
            });
        Secondary::visit_pages(layout.anchors_, [&](const auto& page) { visit(page.get(), page->charge.bytes); });
    }

private:
    template<class Family> static std::size_t buffers(const Family& family)
    {
        using persistent_ordered_detail::checked_product;
        Budget bytes{std::numeric_limits<std::size_t>::max()};
        bytes.add(family.key.capacity());
        bytes.add(family.anchor.capacity());
        bytes.add(checked_product(family.children.capacity(), sizeof(AdaptiveCellKey)));
        for (const auto& child : family.children)
            std::visit(
                [&](const auto& ancestry)
                {
                    if constexpr (requires { ancestry.path; })
                        bytes.add(checked_product(ancestry.path.capacity(), sizeof(std::uint64_t)));
                    else if constexpr (requires { ancestry.base.path; })
                    {
                        bytes.add(checked_product(ancestry.base.path.capacity(), sizeof(std::uint64_t)));
                        bytes.add(checked_product(ancestry.axial.path.capacity(), sizeof(std::uint64_t)));
                    }
                },
                child.ancestry);
        bytes.add(AdaptiveLeafPages::buffer_bytes(family.parent));
        return bytes.used;
    }
    [[nodiscard]] static Row make_record(const AdaptiveFamily& family, const std::shared_ptr<Budget>& arena)
    {
        using namespace persistent_ordered_detail;
        if (family.key.empty() || family.anchor.empty() || family.children.empty() ||
            !PLIC::geometry_encoding_capacity_profile_supported())
            throw std::invalid_argument("Incomplete saved family or unqualified capacity profile.");
        constexpr auto header = sizeof(Record) + control_bytes;
        Charge admission(arena);
        admission.add(checked_sum(header, buffers(family)));
        auto record = std::make_shared<Record>(arena, family);
        const auto actual = checked_sum(header, buffers(record->value));
        if (actual > admission.bytes)
            throw std::invalid_argument("Saved family row capacity exceeds preadmission.");
        record->charge.bytes = actual;
        admission.bytes -= actual;
        return record;
    }
};
} // namespace adaptive_detail
} // namespace SimpleFluid::Meshes::AMR
