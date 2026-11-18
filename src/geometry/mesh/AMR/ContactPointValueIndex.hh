/** @file ContactPointValueIndex.hh
 * @brief Persistent ordinal selectors over independently owned immutable point values.
 */
#pragma once
#include "CartesianIncidenceBudget.hh"
#include "ContactPayloadCensus.hh"
#include "IdentityPoolHash.hh"
#include "PersistentOrderedPages.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include <limits>
#include <optional>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** A root cannot read values by itself. Every operation borrows the actual
 * immutable data bundle, which must pin all referenced ordinals. No page
 * retains that reader, its controller, or any historical snapshot. */
class ContactPointValueIndex
{
public:
    using Point = PLIC::PointIdentity<ArithmeticMode::Ordinary>;
    using Component = PLIC::details::CanonicalPointComponent;
    enum class Routing
    {
        Ordered,
        Canonical,
        Custom
    };
    struct Probe
    {
        std::size_t scalar_comparisons = 0, point_comparisons = 0, identity_checks = 0, directory_comparisons = 0;
    };
    struct Reader
    {
        const void* context;
        Point (*point)(const void*, std::size_t, Probe*);
        const Component& (*component)(const void*, unsigned, std::size_t, Probe*);
    };
    struct Work
    {
        persistent_ordered_detail::OrderedPageWork pages;
        Probe probes;
    };

    explicit ContactPointValueIndex(Routing routing = Routing::Ordered) : routing_(routing) {}

    [[nodiscard]] static bool equal(const Point& a, const Point& b) noexcept
    {
        using Access = PLIC::details::GeometryIdentityAccess;
        const auto bytes = PLIC::canonical_key_size(a);
        if (bytes != PLIC::canonical_key_size(b))
            return false;
        for (std::size_t offset = 0; offset < bytes; offset += 8)
            if (Access::canonical_word_at(a, offset) != Access::canonical_word_at(b, offset))
                return false;
        return true;
    }
    [[nodiscard]] static bool less(const Point& a, const Point& b) noexcept
    {
        if (a < b)
            return true;
        if (b < a)
            return false;
        // Numeric ties still require the complete canonical identity.
        using Access = PLIC::details::GeometryIdentityAccess;
        const auto a_bytes = PLIC::canonical_key_size(a), b_bytes = PLIC::canonical_key_size(b);
        for (std::size_t offset = 0; offset < std::min(a_bytes, b_bytes); ++offset)
        {
            const auto first = static_cast<unsigned char>(Access::canonical_word_at(a, offset));
            const auto second = static_cast<unsigned char>(Access::canonical_word_at(b, offset));
            if (first != second)
                return first < second;
        }
        return a_bytes < b_bytes;
    }

    [[nodiscard]] std::optional<std::size_t> find_component(
        const Reader& read, unsigned axis, long double value, Probe* work = nullptr) const
    {
        const auto search = [&]<class Index>(const typename Index::Ptr& root) -> std::optional<std::size_t>
        {
            const auto* row = Index::lower_bound(root,
                [&](auto slot)
                {
                    if (work)
                        ++work->scalar_comparisons;
                    return read.component(read.context, axis, slot, work).value() < value;
                });
            if (!row)
                return {};
            if (work)
                ++work->scalar_comparisons;
            return read.component(read.context, axis, *row, work).value() == value ? std::optional<std::size_t>(*row)
                                                                                   : std::nullopt;
        };
        if (const auto found = search.template operator()<Scalars32>(scalars32_.at(axis)))
            return found;
        return search.template operator()<Scalars64>(scalars64_.at(axis));
    }

    [[nodiscard]] std::optional<std::size_t> find(
        const Reader& read, const Point& point, std::uint64_t route, Probe* work = nullptr) const
    {
        if (routing_ == Routing::Canonical && IdentityPoolHasher<Point>{}(point) != route)
            return {};
        const auto before = [&](std::size_t slot)
        {
            if (work)
                ++work->point_comparisons;
            return less(read.point(read.context, slot, work), point);
        };
        std::optional<std::size_t> slot;
        if (routing_ == Routing::Custom)
        {
            const auto* row = CustomPoints::lower_bound(custom_, [&](const Routed& row) { return before(row.slot); });
            if (row && row->route == route)
                slot = row->slot;
        }
        else if (const auto* row = Points32::lower_bound(points32_, before))
            slot = *row;
        if (slot)
        {
            if (work)
                ++work->identity_checks;
            if (equal(read.point(read.context, *slot, work), point))
                return slot;
        }
        slot.reset();
        if (routing_ != Routing::Custom)
            if (const auto* row = Points64::lower_bound(points64_, before))
                slot = *row;
        if (!slot)
            return {};
        if (work)
            ++work->identity_checks;
        return equal(read.point(read.context, *slot, work), point) ? slot : std::nullopt;
    }

    /** All input component/selector/exception vectors must already be final.
     * Only private roots are changed; an exception leaves published roots intact. */
    void append(const Reader& read, std::array<std::size_t, 3> first_components,
        std::array<std::size_t, 3> component_counts, std::size_t first_point, std::size_t point_count,
        std::span<const std::uint64_t> routes, const std::shared_ptr<Budget>& arena, Work& work,
        const std::shared_ptr<const geometry_detail::NativeAllocationState>& native = {})
    {
        if (!arena || !read.context || !read.point || !read.component || first_point > point_count)
            throw std::invalid_argument("Unbound point index append.");
        for (unsigned axis = 0; axis < 3; ++axis)
            if (first_components[axis] > component_counts[axis])
                throw std::invalid_argument("Point component index append reverses its ordinals.");
        auto scalars32 = scalars32_;
        auto scalars64 = scalars64_;
        auto points32 = points32_;
        auto points64 = points64_;
        auto custom = custom_;
        cartesian_detail::Allocation scratch(*arena);
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            const ScalarOrder less{read, axis, &work.probes};
            const auto split = std::min(component_counts[axis], narrow_limit);
            reject_cross_root_duplicates<Scalars64>(
                scalars64[axis], std::min(first_components[axis], split), split, less);
            scalars32[axis] = append_range<Scalars32, std::uint32_t>(
                scalars32[axis], std::min(first_components[axis], split), split, arena, less, true, work, native);
            reject_cross_root_duplicates<Scalars32>(
                scalars32[axis], std::max(first_components[axis], split), component_counts[axis], less);
            scalars64[axis] = append_range<Scalars64, std::uint64_t>(scalars64[axis],
                std::max(first_components[axis], split), component_counts[axis], arena, less, true, work, native);
        }
        if (routing_ != Routing::Ordered && routes.size() != point_count - first_point)
            throw std::invalid_argument("Point index routing does not match its appended values.");
        if (routing_ == Routing::Canonical)
            for (auto slot = first_point; slot < point_count; ++slot)
                if (IdentityPoolHasher<Point>{}(read.point(read.context, slot, &work.probes)) !=
                    routes[slot - first_point])
                    throw std::invalid_argument("Point index input does not match its canonical routing policy.");
        if (routing_ == Routing::Custom)
        {
            std::vector<Routed> rows;
            scratch.reserve(rows, routes.size());
            for (auto slot = first_point; slot < point_count; ++slot)
                rows.push_back({slot, routes[slot - first_point]});
            const RoutedOrder less{read, &work.probes};
            std::sort(rows.begin(), rows.end(), less);
            const CustomPoints index(arena, &work.pages, less, native);
            if (!custom)
                custom = index.build(rows);
            else
                for (const auto& row : rows)
                    custom = index.edit(custom, row, &row, false);
        }
        else
        {
            const PointOrder less{read, &work.probes};
            const auto split = std::min(point_count, narrow_limit);
            reject_cross_root_duplicates<Points64>(points64, std::min(first_point, split), split, less);
            points32 = append_range<Points32, std::uint32_t>(
                points32, std::min(first_point, split), split, arena, less, true, work, native);
            reject_cross_root_duplicates<Points32>(points32, std::max(first_point, split), point_count, less);
            points64 = append_range<Points64, std::uint64_t>(
                points64, std::max(first_point, split), point_count, arena, less, true, work, native);
        }
        scalars32_ = std::move(scalars32);
        scalars64_ = std::move(scalars64);
        points32_ = std::move(points32);
        points64_ = std::move(points64);
        custom_ = std::move(custom);
    }

    [[nodiscard]] std::size_t metadata_bytes() const noexcept
    {
        auto bytes = storage(points32_) + storage(points64_) + storage(custom_);
        for (unsigned axis = 0; axis < 3; ++axis)
            bytes += storage(scalars32_[axis]) + storage(scalars64_[axis]);
        return bytes;
    }
    template<bool Prune, class Visitor> void visit_dependencies(Visitor&& visitor) const
    {
        for (const auto& root : scalars32_)
            visit_nodes<Prune>(root, visitor);
        for (const auto& root : scalars64_)
            visit_nodes<Prune>(root, visitor);
        visit_nodes<Prune>(points32_, visitor);
        visit_nodes<Prune>(points64_, visitor);
        visit_nodes<Prune>(custom_, visitor);
    }
    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor) const
    {
        const auto visit = [&](const auto& root, ContactPayloadRole role)
        {
            const auto descend = [&](auto&& self, const auto& page) -> void
            {
                if (!page)
                    return;
                contact_payload_object(visitor, page.get(), role, sizeof(*page), true);
                contact_payload_buffer(visitor, page.get(), role, page->rows);
                contact_payload_buffer(visitor, page.get(), role, page->children);
                for (const auto& child : page->children)
                    self(self, child);
            };
            descend(descend, root);
        };
        for (const auto& root : scalars32_)
            visit(root, ContactPayloadRole::SourcePointScalarIndex);
        for (const auto& root : scalars64_)
            visit(root, ContactPayloadRole::SourcePointScalarIndex);
        visit(points32_, ContactPayloadRole::SourcePointIdentityIndex);
        visit(points64_, ContactPayloadRole::SourcePointIdentityIndex);
        visit(custom_, ContactPayloadRole::SourcePointIdentityIndex);
    }

private:
    struct ScalarOrder
    {
        Reader read{};
        unsigned axis = 0;
        Probe* work = nullptr;
        bool operator()(std::size_t a, std::size_t b) const
        {
            if (work)
                ++work->scalar_comparisons;
            return read.component(read.context, axis, a, work).value() <
                   read.component(read.context, axis, b, work).value();
        }
    };
    struct PointOrder
    {
        Reader read{};
        Probe* work = nullptr;
        bool operator()(std::size_t a, std::size_t b) const
        {
            if (work)
                ++work->point_comparisons;
            return less(read.point(read.context, a, work), read.point(read.context, b, work));
        }
    };
    struct Routed
    {
        std::size_t slot;
        std::uint64_t route;
    };
    struct RoutedOrder
    {
        Reader read{};
        Probe* work = nullptr;
        bool operator()(const Routed& a, const Routed& b) const { return PointOrder{read, work}(a.slot, b.slot); }
    };
    // Ordinal rows are small. Pack more into each leaf to amortize the
    // fixed page and control storage across long append histories.
    template<class Ordinal, class Less>
    using OrdinalPages = persistent_ordered_detail::OrderedPages<Ordinal, Less, false,
        persistent_ordered_detail::OrderedPageWork, persistent_ordered_detail::NoRowWeights, 128>;
    using Scalars32 = OrdinalPages<std::uint32_t, ScalarOrder>;
    using Scalars64 = OrdinalPages<std::uint64_t, ScalarOrder>;
    using Points32 = OrdinalPages<std::uint32_t, PointOrder>;
    using Points64 = OrdinalPages<std::uint64_t, PointOrder>;
    using CustomPoints = persistent_ordered_detail::OrderedPages<Routed, RoutedOrder>;

    template<class Root> static std::size_t storage(const Root& root) noexcept
    {
        return root ? root->subtree_bytes : 0;
    }
    template<bool Prune, class Root, class Visitor> static void visit_nodes(const Root& page, Visitor&& visit)
    {
        if (!page)
            return;
        const std::shared_ptr<Budget> empty;
        const bool keeper = page->charge.arena.owner_before(empty) || empty.owner_before(page->charge.arena);
        if constexpr (Prune)
        {
            if (!visit(page, page->charge.arena.get(), page->charge.bytes, keeper))
                return;
        }
        else
            visit(page, page->charge.arena.get(), page->charge.bytes, keeper);
        for (const auto& child : page->children)
            visit_nodes<Prune>(child, visit);
    }
    static constexpr std::size_t narrow_limit = std::numeric_limits<std::uint32_t>::max();
    template<class Index, class Less>
    static void reject_cross_root_duplicates(
        const typename Index::Ptr& root, std::size_t first, std::size_t count, const Less& less)
    {
        if (!root)
            return;
        for (auto slot = first; slot < count; ++slot)
        {
            const auto* found = Index::lower_bound(root, [&](auto old) { return less(old, slot); });
            if (found && !less(slot, *found))
                throw std::invalid_argument("Point index width partitions repeat an accepted value.");
        }
    }
    template<class Index, class UInt, class Less>
    static typename Index::Ptr append_range(const typename Index::Ptr& root, std::size_t first, std::size_t count,
        const std::shared_ptr<Budget>& arena, const Less& less, bool sort, Work& work,
        const std::shared_ptr<const geometry_detail::NativeAllocationState>& native)
    {
        if (first == count)
            return root;
        cartesian_detail::Allocation scratch(*arena);
        std::vector<UInt> slots;
        scratch.reserve(slots, count - first);
        for (auto slot = first; slot < count; ++slot)
            slots.push_back(UInt(slot));
        if (sort)
            std::sort(slots.begin(), slots.end(), less);
        const Index index(arena, &work.pages, less, native);
        if (!root)
            return index.build(slots);
        auto next = root;
        for (const auto slot : slots)
            next = index.edit(next, slot, &slot, false);
        return next;
    }
    Routing routing_;
    std::array<Scalars32::Ptr, 3> scalars32_;
    std::array<Scalars64::Ptr, 3> scalars64_;
    Points32::Ptr points32_;
    Points64::Ptr points64_;
    CustomPoints::Ptr custom_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
