/** @file GeometryDemandPages.tcc
 * @brief Compiled-provider implementation for 64-row/32-child geometry pages.
 */
#pragma once
#include "geometry/PLIC/details/GeometryDemandPages.hh"
#include "geometry/details/NativeAllocationGraph.hh"
#include "geometry/details/AllocationDomain.hh"
#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimpleFluid::PLIC::details::geometry_demand
{
inline std::size_t checked_sum(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Geometry demand size overflows.");
    return a + b;
}
inline void Admission::charge(std::size_t count, std::size_t width) const
{
    if (!retain_bytes || (width && count > std::numeric_limits<std::size_t>::max() / width))
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Geometry demand admission is invalid.");
    retain_bytes(context, count * width);
}
inline ReceiptEdits::ReceiptEdits(Admission admission)
    : admission_(std::move(admission)), overflow_(admission_.allocator<NativeOwnerReceipts::Edit>())
{
}
inline ReceiptEdits::~ReceiptEdits()
{
    clear();
}
template<class Value>
const Value& ReceiptEdits::Charges<Value>::append(const Value& value, const Admission& admission)
{
    Value* stored;
    if (size < local.size())
        stored = &local[size];
    else
    {
        constexpr auto block_size = std::tuple_size_v<decltype(Block::values)>;
        const auto index = (size - local.size()) % block_size;
        if (!index)
        {
            admission.charge(1, sizeof(Block));
            auto allocate = admission.allocator<Block>();
            auto* next = allocate.allocate(1);
            std::construct_at(next);
            observe_receipt_block(sizeof(Block), !std::is_same_v<Value, NativeReceiptFact>);
            next->previous = tail;
            tail = next;
        }
        stored = &tail->values[index];
    }
    ++size;
    return *stored = value;
}
template<class Value>
void ReceiptEdits::Charges<Value>::clear(const Admission& admission) noexcept
{
    for (auto& value : std::span(local).first(std::min(size, local.size())))
        value = {};
    size = 0;
    auto allocate = admission.allocator<Block>();
    while (tail)
    {
        auto* released = tail;
        tail = tail->previous;
        std::destroy_at(released);
        allocate.deallocate(released, 1);
    }
}
inline void ReceiptEdits::append(const NativePaidExtentCharge& charge, NativeOwnerReceipts::Action action)
{
    const NativePaidExtentRef ref(&charge);
    if (ref.has_native_witness())
        append_borrowed(NativePaidExtentRef(witnesses_.append(ref.native_witness(), admission_)), action);
    else if (const auto* proof = ref.proof())
        append_borrowed(&proofs_.append(*proof, admission_), action);
    else
        append_borrowed(NativePaidExtentRef(facts_.append(ref.fact(), admission_)), action);
}
inline void ReceiptEdits::append_borrowed(NativePaidExtentRef charge, NativeOwnerReceipts::Action action)
{
    if (overflow_.empty() && size_ < inline_.size())
    {
        inline_[size_++] = {charge, action};
        return;
    }
    if (overflow_.size() == overflow_.capacity())
    {
        const auto current = std::max(inline_.size(), overflow_.capacity());
        const auto next = checked_sum(current, current);
        admission_.charge(next, sizeof(NativeOwnerReceipts::Edit));
        overflow_.reserve(next);
        if (overflow_.capacity() != next)
            throw GeometryError(GeometryErrorCode::ResourceLimit, "Paid receipt edit capacity is unqualified.");
    }
    if (size_)
    {
        overflow_.insert(overflow_.end(), inline_.begin(), inline_.begin() + size_);
        size_ = 0;
    }
    overflow_.push_back({charge, action});
}
inline std::span<NativeOwnerReceipts::Edit> ReceiptEdits::rows() noexcept
{
    if (!overflow_.empty()) return overflow_;
    return std::span(inline_).first(size_);
}
inline void ReceiptEdits::retain_prefix(std::size_t count) noexcept
{
    if (count > rows().size())
        std::terminate();
    if (overflow_.empty())
        size_ = count;
    else
        overflow_.resize(count);
}
inline void ReceiptEdits::clear() noexcept
{
    overflow_.clear();
    size_ = 0;
    facts_.clear(admission_);
    witnesses_.clear(admission_);
    proofs_.clear(admission_);
}
inline void visit_native_state(
    const std::shared_ptr<const geometry_detail::NativeAllocationState>& state, MetadataSink sink)
{
    if (sink.project_native_state(state))
        return;
    const auto counter_bytes = geometry_detail::NativeAllocationGraphAccess::validate_state(state);
    sink.paid(NativePaidExtentCharge::state(state));
    if (!sink.native_owner(state, state->allocation_admission_bytes))
        return;
    sink.paid(NativePaidExtentCharge::counter(state));
    const bool first_counter = sink.native_owner(state->counter_owner, counter_bytes);
    const bool last_counter = sink.native_owner(state->counter_owner, counter_bytes);
    if (!state->mirror.owner)
        return;
    sink.paid(NativePaidExtentCharge::mirror(state));
    for (unsigned edge = 0; edge != (first_counter || last_counter ? 4u : 2u); ++edge)
        sink.native_owner(state->mirror.owner, state->mirror.owner_proof.bytes());
}
namespace native_page_detail
{
/** Captures actual rebound shared-control size into its admitted heap target.
 * Deallocation never dereferences the target after its destruction. */
template<class T, class Owner, geometry_detail::AllocationDomain Domain> class SIMPLEFLUID_LOCAL ControlAllocator
{
    template<class, class, geometry_detail::AllocationDomain> friend class ControlAllocator;

public:
    using value_type = T;
    template<class U> struct rebind { using other = ControlAllocator<U, Owner, Domain>; };
    ControlAllocator(PageAllocationAllocator<T, Domain> allocator, Owner* owner)
        : allocator_(std::move(allocator)), owner_(owner)
    {
    }
    template<class U>
    ControlAllocator(const ControlAllocator<U, Owner, Domain>& other) noexcept
        : allocator_(other.allocator_), owner_(other.owner_)
    {
    }
    T* allocate(std::size_t count)
    {
        auto* value = allocator_.allocate(count);
        const auto bytes = count * sizeof(T);
        std::memcpy(static_cast<void*>(owner_), &bytes, sizeof(bytes));
        return value;
    }
    void deallocate(T* address, std::size_t count) noexcept { allocator_.deallocate(address, count); }
    template<class U> bool operator==(const ControlAllocator<U, Owner, Domain>& other) const noexcept
    {
        return allocator_ == other.allocator_ && owner_ == other.owner_;
    }

private:
    PageAllocationAllocator<T, Domain> allocator_;
    Owner* owner_;
};
static_assert(sizeof(ControlAllocator<std::byte, std::byte, geometry_detail::AllocationDomain::OwnerUnion>) ==
              sizeof(ControlAllocator<std::byte, std::byte, geometry_detail::AllocationDomain::GeometrySnapshot>));
} // namespace native_page_detail
template<class T, geometry_detail::AllocationDomain Domain, class... Args>
std::shared_ptr<T> Admission::shared(Args&&... args) const
{
    if (!native)
        return std::allocate_shared<T>(
            geometry_detail::FixedDomainAllocator<std::remove_cv_t<T>, Domain>{},
            std::forward<Args>(args)...);
#if !defined(__cpp_rtti)
    throw GeometryError(GeometryErrorCode::InvalidInput, "Native metadata requires authenticated shared-control RTTI.");
#endif
    static_assert(sizeof(T) >= sizeof(std::size_t));
    auto memory = PageAllocationAllocator<T, Domain>(geometry_detail::NativeConstructionAllocator<T>(native));
    auto* value = memory.allocate(1);
    NativeMetadataDelete deleter{native, value, 0, false,
        [](const void* raw, std::size_t control) noexcept
        {
            const auto& object = *static_cast<const T*>(raw);
            if constexpr (requires { object.bytes; })
                return object.bytes;
            else
                return sizeof(T) + control;
        },
        [](void* raw, const std::shared_ptr<const geometry_detail::NativeAllocationState>& state,
            bool constructed) noexcept
        {
            auto* object = static_cast<T*>(raw);
            std::size_t buffers = 0;
            if (constructed)
            {
                if constexpr (requires { object->native_buffer_bytes(); })
                    buffers = object->native_buffer_bytes();
                std::destroy_at(object);
            }
            if (buffers)
                state->retire(buffers);
            geometry_detail::NativeConstructionAllocator<T>(state).deallocate_for(Domain, object, 1);
        }};
    // Capture into the unconstructed heap target, never a retained stack slot.
    // The control allocator never reads this target during deallocation.
    std::shared_ptr<T> owner(value, std::move(deleter), native_page_detail::ControlAllocator<T, T, Domain>(memory, value));
    auto* proof = std::get_deleter<NativeMetadataDelete>(owner);
    std::memcpy(&proof->control_bytes, static_cast<const void*>(value), sizeof(proof->control_bytes));
    std::construct_at(value, std::forward<Args>(args)...);
    proof->constructed = true;
    return owner;
}
template<class Row, class Less, bool CompactBranches, class Witnesses, geometry_detail::AllocationDomain Domain>
class OrderedPages
{
public:
    struct Node;
    using Ptr = std::shared_ptr<const Node>;
    template<class T> using Buffer = std::vector<T, PageBufferAllocator<T, Domain>>;
    static_assert(std::is_nothrow_copy_constructible_v<Row> && std::is_nothrow_copy_assignable_v<Row>);
    struct Node
    {
        std::size_t bytes = 0, payload_bytes = 0;
        std::vector<Row> rows;
        std::vector<Ptr> children;
        [[no_unique_address]] Witnesses witnesses;
        [[nodiscard]] std::size_t native_buffer_bytes() const noexcept
        {
            const auto index = rows.capacity() * sizeof(Row) + children.capacity() * sizeof(Ptr);
            if constexpr (std::is_same_v<Witnesses, NoLeafWitnesses>)
                return index;
            else
                return index + witnesses.buffer_bytes();
        }
        const Row* maximum = nullptr;
        std::size_t size = 0, subtree_bytes = 0, subtree_payload_bytes = 0, subtree_owner_bytes = 0, logical_size = 0,
                    subtree_owned_admission_bytes = 0;
        [[nodiscard]] bool has_native_administration() const noexcept
        {
            if constexpr (native_flag_in_owned_admission<Row>)
                return subtree_owned_admission_bytes & 1;
            else
                return true;
        }
        [[nodiscard]] std::size_t owned_admission_bytes() const noexcept
        {
            if constexpr (native_flag_in_owned_admission<Row>)
                return subtree_owned_admission_bytes & ~std::size_t{1};
            else
                return subtree_owned_admission_bytes;
        }
        [[nodiscard]] bool leaf() const noexcept { return children.empty(); }
        [[nodiscard]] std::size_t occupancy() const noexcept { return leaf() ? rows.size() : children.size(); }
    };

    explicit OrderedPages(Admission admission, PageWork* work = nullptr) : admission_(admission), work_(work) {}

    [[nodiscard]] Ptr build(std::span<const Row> sorted) const { return build_from(sorted); }

    /** Source rows may borrow the payload until the new leaves are filled. */
    template<class Source> [[nodiscard]] Ptr build_from(std::span<const Source> sorted) const
    {
        if (sorted.empty())
            return {};
        for (std::size_t slot = 1; slot < sorted.size(); ++slot)
            if (!Less{}(sorted[slot - 1], sorted[slot]))
                throw GeometryError(
                    GeometryErrorCode::InvalidInput, "Ordered metadata index contains duplicate selectors.");
        Buffer<Ptr> pages(PageBufferAllocator<Ptr, Domain>(admission_.native));
        const auto groups = ceiling(sorted.size(), leaf_capacity);
        reserve(pages, groups);
        std::size_t offset = 0;
        for (std::size_t group = 0; group < groups; ++group)
        {
            const auto count = group_size(sorted.size(), groups, group);
            pages.push_back(leaf(sorted.subspan(offset, count)));
            offset += count;
        }
        while (pages.size() > 1)
        {
            Buffer<Ptr> parents(PageBufferAllocator<Ptr, Domain>(admission_.native));
            const auto count = ceiling(pages.size(), branch_capacity);
            reserve(parents, count);
            offset = 0;
            for (std::size_t group = 0; group < count; ++group)
            {
                const auto width = group_size(pages.size(), count, group);
                parents.push_back(branch(std::span(pages).subspan(offset, width)));
                offset += width;
            }
            pages.swap(parents);
            Buffer<Ptr>().swap(parents);
        }
        return std::move(pages.front());
    }

    [[nodiscard]] Ptr edit(const Ptr& old, const Row& key, const Row* replacement, bool expected) const
    {
        if (replacement && (Less{}(key, *replacement) || Less{}(*replacement, key)))
            throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry demand replacement changes its key.");
        if (!old)
        {
            if (expected || !replacement)
                throw GeometryError(
                    GeometryErrorCode::InvalidInput, "Ordered metadata edit lacks its accepted selector.");
            return leaf(std::span(replacement, 1));
        }
        auto result = rewrite(*old, key, replacement, expected);
        if (result.count == 0)
            return {};
        if (result.count == 2)
            return branch(std::span(result.pages));
        auto root = std::move(result.pages[0]);
        while (root->children.size() == 1)
        {
            retire(*root);
            auto child = root->children.front();
            root = std::move(child);
        }
        return root;
    }

    /** Apply sorted final rows in one path-copy edit. A false keep result
     * removes an existing row. Unchanged subtrees preserve exact controls. */
    template<class Keep> [[nodiscard]] Ptr patched(const Ptr& old, std::span<const Row> changes, const Keep& keep) const
    {
        return patched_from(old, changes, keep);
    }

    template<class Source, class Keep>
    [[nodiscard]] Ptr patched_from(const Ptr& old, std::span<const Source> changes, const Keep& keep) const
    {
        return patched_range(old, changes.begin(), changes.end(), keep);
    }

    /** A forward range may borrow records directly from ordered edit pages. */
    template<std::forward_iterator Iterator, class Keep>
    [[nodiscard]] Ptr patched_range(const Ptr& old, Iterator first, Iterator last, const Keep& keep) const
    {
        if (first == last)
            return old;
        for (auto previous = first, at = std::next(first); at != last; previous = at++)
            if (!Less{}(*previous, *at))
                throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry batch keys are not strictly ordered.");
        auto roots = patch_rewrite(old, first, last, keep, true);
        while (roots.size() > 1)
            roots = pack_many_branches(roots);
        if (roots.empty())
            return {};
        auto root = roots.front();
        while (root->children.size() == 1)
        {
            retire(*root);
            root = root->children.front();
        }
        return root;
    }

    [[nodiscard]] static const Row& at(const Ptr& root, std::size_t slot, PageWork* work = nullptr)
    {
        if (!root || slot >= root->size)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid geometry demand rank.");
        auto node = root.get();
        while (!node->leaf())
        {
            if (work)
                ++work->visited_pages;
            auto child = node->children.begin();
            while (slot >= (*child)->size)
            {
                slot -= (*child)->size;
                ++child;
            }
            node = child->get();
        }
        if (work)
            ++work->visited_pages;
        return node->rows[slot];
    }
    struct LogicalRow
    {
        const Row* row;
        std::size_t offset;
    };
    [[nodiscard]] static LogicalRow logical_at(const Ptr& root, std::size_t slot, PageWork* work = nullptr)
    {
        if (!root || slot >= root->logical_size)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid logical geometry demand rank.");
        auto node = root.get();
        while (!node->leaf())
        {
            if (work)
                ++work->visited_pages;
            auto child = node->children.begin();
            while (slot >= (*child)->logical_size)
            {
                slot -= (*child)->logical_size;
                ++child;
            }
            node = child->get();
        }
        if (work)
            ++work->visited_pages;
        for (const auto& row : node->rows)
        {
            const auto count = [&]
            {
                if constexpr (requires { row.logical_size(); })
                    return row.logical_size();
                else
                    return std::size_t{1};
            }();
            if (slot < count)
                return {&row, slot};
            slot -= count;
        }
        throw GeometryError(GeometryErrorCode::InvalidTopology, "Geometry demand page lost its logical row.");
    }
    template<class Before>
    [[nodiscard]] static std::size_t logical_rank(const Ptr& root, const Before& before, PageWork* work = nullptr)
    {
        std::size_t result = 0;
        auto node = root.get();
        while (node)
        {
            if (work)
                ++work->visited_pages;
            if (node->leaf())
            {
                for (const auto& row : node->rows)
                {
                    if (!before(row))
                        return result;
                    if constexpr (requires { row.logical_size(); })
                        result = checked_sum(result, row.logical_size());
                    else
                        result = checked_sum(result, 1);
                }
                return result;
            }
            auto child = node->children.begin();
            while (child != node->children.end() && before(*(*child)->maximum))
            {
                result = checked_sum(result, (*child)->logical_size);
                ++child;
            }
            if (child == node->children.end())
                return result;
            node = child->get();
        }
        return result;
    }

    template<class Before>
    [[nodiscard]] static std::size_t rank(const Ptr& root, const Before& before, PageWork* work = nullptr)
    {
        std::size_t result = 0;
        auto node = root.get();
        while (node)
        {
            if (work)
                ++work->visited_pages;
            if (node->leaf())
                return checked_sum(
                    result, std::partition_point(node->rows.begin(), node->rows.end(), before) - node->rows.begin());
            auto child = node->children.begin();
            while (child != node->children.end() && before(*(*child)->maximum))
            {
                result = checked_sum(result, (*child)->size);
                ++child;
            }
            if (child == node->children.end())
                return result;
            node = child->get();
        }
        return result;
    }

    template<class Before>
    [[nodiscard]] static const Row* lower_bound(const Ptr& root, const Before& before, PageWork* work = nullptr)
    {
        auto node = root.get();
        while (node && !node->leaf())
        {
            if (work)
                ++work->visited_pages;
            const auto child = std::partition_point(node->children.begin(), node->children.end(),
                [&](const Ptr& value) { return before(*value->maximum); });
            if (child == node->children.end())
                return nullptr;
            node = child->get();
        }
        if (!node)
            return nullptr;
        if (work)
            ++work->visited_pages;
        const auto row = std::partition_point(node->rows.begin(), node->rows.end(), before);
        return row == node->rows.end() ? nullptr : &*row;
    }

    template<class Before, class Visit>
    static bool visit_from(const Ptr& root, const Before& before, const Visit& visit)
    {
        if (!root || before(*root->maximum))
            return true;
        if (root->leaf())
        {
            const auto begin = std::partition_point(root->rows.begin(), root->rows.end(), before);
            return std::all_of(begin, root->rows.end(), visit);
        }
        for (const auto& child : root->children)
            if (!visit_from(child, before, visit))
                return false;
        return true;
    }

    template<class RowVisit>
    static void visit_dependencies(const Ptr& root, MetadataSink metadata, const RowVisit& row_visit)
    {
        if (!root || !metadata(root, root->payload_bytes, root->bytes))
            return;
        const auto state = metadata_state(root);
        if (state)
        {
            // Two vector allocators plus object deleter and shared-control
            // allocator each hold one State, even for empty vector buffers.
            for (unsigned edge = 0; edge != 2; ++edge)
                visit_native_state(state, metadata);
        }
        if (root->leaf())
            for (const auto& row : root->rows)
                row_visit(row);
        else
            for (const auto& child : root->children)
                visit_dependencies(child, metadata, row_visit);
    }

    /** Prune exact shared page controls before descending either frontier.
     * Native ownership flags also skip every wholly default subtree. */
    static void visit_native_administration_delta(
        const Ptr& previous, const Ptr& next, NativeAdministrationSink sink, PageWork* work = nullptr)
    {
        visit_native_administration_delta(previous, next, sink, work, false, [](const Row&, bool) {});
    }
    template<class RowVisit>
    static void visit_native_administration_delta(const Ptr& previous, const Ptr& next,
        NativeAdministrationSink sink, PageWork* work, bool include_default, const RowVisit& row_visit)
    {
        const auto shared = [&](const Ptr& candidate, const Ptr& other)
        {
            auto node = other;
            while (node)
            {
                if (work)
                    ++work->visited_pages;
                if (node.get() == candidate.get() && !node.owner_before(candidate) && !candidate.owner_before(node))
                    return true;
                if (node->leaf() || Less{}(*node->maximum, *candidate->maximum))
                    return false;
                const auto at = std::lower_bound(node->children.begin(), node->children.end(), *candidate->maximum,
                    [](const Ptr& child, const Row& key) { return Less{}(*child->maximum, key); });
                if (at == node->children.end())
                    return false;
                node = *at;
            }
            return false;
        };
        const auto visit = [&](const auto& self, const Ptr& candidate, const Ptr& other, bool retain) -> void
        {
            if (!candidate || (!include_default && !candidate->has_native_administration()) || shared(candidate, other))
                return;
            if (work)
                ++work->visited_pages;
            sink.owner(candidate, retain);
            sink(metadata_state(candidate), 2, retain);
            if (candidate->leaf())
                for (const auto& row : candidate->rows) row_visit(row, retain);
            for (const auto& child : candidate->children)
                self(self, child, other, retain);
        };
        visit(visit, next, previous, true);
        visit(visit, previous, next, false);
    }

    template<class RowVisit>
    static void visit_borrowed_dependencies(
        const Ptr& root, BorrowedMetadataSink metadata, const RowVisit& row_visit, const void* parent = nullptr)
    {
        if (!root || !metadata(root.get(), parent, root->payload_bytes))
            return;
        if (root->leaf())
            for (const auto& row : root->rows)
                row_visit(row, root.get());
        else
            for (const auto& child : root->children)
                visit_borrowed_dependencies(child, metadata, row_visit, root.get());
    }

    template<class Visit> static void visit_pages(const Ptr& root, const Visit& visit)
    {
        if (!root)
            return;
        visit(root);
        for (const auto& child : root->children)
            visit_pages(child, visit);
    }

private:
    struct Replacement
    {
        std::array<Ptr, 2> pages;
        std::size_t count;
    };
    static std::size_t ceiling(std::size_t count, std::size_t width) { return count / width + (count % width != 0); }
    static std::size_t group_size(std::size_t count, std::size_t groups, std::size_t group)
    {
        return count / groups + (group < count % groups);
    }
    template<class Value, class Allocator>
    void reserve(std::vector<Value, Allocator>& values, std::size_t count) const
    {
        admission_.charge(count, sizeof(Value));
        values.reserve(count);
        if (values.capacity() != count)
            throw GeometryError(GeometryErrorCode::ResourceLimit, "Geometry demand page capacity is unqualified.");
    }
    template<class Value> void reserve(std::vector<Value>& values, std::size_t count) const
    {
        admission_.charge(count, sizeof(Value));
        const auto bytes = count * sizeof(Value);
        if (admission_.native)
        {
            geometry_detail::NativeConstructorStorage::require_profile();
            admission_.native->add(bytes);
        }
        try
        {
#if SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT
            if (!values.capacity())
            {
                // Empty concrete buffers invoke only the upstream allocator.
                const geometry_detail::AllocationDomainScope domain(Domain);
                values.reserve(count);
            }
            else
#endif
            {
                values.reserve(count);
            }
        }
        catch (...)
        {
            if (admission_.native)
                admission_.native->retire(bytes);
            throw;
        }
        if (values.capacity() != count)
        {
            std::vector<Value>().swap(values);
            if (admission_.native)
                admission_.native->retire(bytes);
            throw GeometryError(GeometryErrorCode::ResourceLimit, "Geometry native concrete capacity is unqualified.");
        }
    }
    void retire(const Node&) const
    {
        if (work_)
            ++work_->retired_page_visits;
    }
    void record(const std::shared_ptr<Node>& pointer) const
    {
        const auto& node = *pointer;
        if (work_)
        {
            ++work_->copied_pages;
            work_->copied_bytes = checked_sum(work_->copied_bytes, node.bytes);
            work_->copied_payload_bytes = checked_sum(work_->copied_payload_bytes, node.payload_bytes);
        }
        admission_.created(pointer, node.payload_bytes, node.bytes);
    }
    [[nodiscard]] std::shared_ptr<Node> allocate() const
    {
        if (admission_.native)
            geometry_detail::NativeConstructorStorage::require_profile();
        admission_.charge(1, sizeof(Node) + control_bytes);
        auto node = admission_.shared<Node, Domain>();
        node->bytes = sizeof(Node) + metadata_control_bytes(node);
        node->payload_bytes = sizeof(Node);
        if constexpr (native_flag_in_owned_admission<Row>)
            node->subtree_owned_admission_bytes = bool(admission_.native);
        return node;
    }
    template<class Range> [[nodiscard]] Ptr leaf(const Range& values) const
    {
        auto node = allocate();
        reserve(node->rows, values.size());
        if constexpr (std::is_same_v<Witnesses, NoLeafWitnesses>)
            for (const auto& value : values)
                node->rows.emplace_back(value);
        else
            node->witnesses.build(node->rows, values, [&](auto& storage, std::size_t count) { reserve(storage, count); });
        node->bytes = checked_sum(node->bytes, node->native_buffer_bytes());
        node->maximum = &node->rows.back();
        node->size = values.size();
        for (const auto& row : node->rows)
        {
            if constexpr (requires { row.logical_size(); })
                node->logical_size = checked_sum(node->logical_size, row.logical_size());
            else
                node->logical_size = checked_sum(node->logical_size, 1);
            if constexpr (requires { row.exclusive_admission_bytes(); })
            {
                if constexpr (native_flag_in_owned_admission<Row>)
                    if (row.exclusive_admission_bytes() & 1)
                        throw GeometryError(
                            GeometryErrorCode::InvalidInput, "Packed native-control row has an odd control extent.");
                node->subtree_owned_admission_bytes =
                    checked_sum(node->subtree_owned_admission_bytes, row.exclusive_admission_bytes());
            }
        }
        if constexpr (requires(const Row& row) { row.exclusive_payload_bytes(); })
            for (const auto& row : node->rows)
                node->subtree_owner_bytes = checked_sum(node->subtree_owner_bytes, row.exclusive_payload_bytes());
        node->payload_bytes = node->bytes - (admission_.native ? 0 : control_bytes);
        node->subtree_bytes = node->bytes;
        node->subtree_payload_bytes = node->payload_bytes;
        record(node);
        return node;
    }
    [[nodiscard]] Ptr branch(std::span<const Ptr> values) const
    {
        auto node = allocate();
        const auto capacity = CompactBranches ? values.size() : branch_capacity;
        reserve(node->children, capacity);
        node->bytes = checked_sum(node->bytes, capacity * sizeof(Ptr));
        node->children.assign(values.begin(), values.end());
        node->payload_bytes = node->bytes - (admission_.native ? 0 : control_bytes);
        node->subtree_bytes = node->bytes;
        node->subtree_payload_bytes = node->payload_bytes;
        for (const auto& child : values)
        {
            if constexpr (native_flag_in_owned_admission<Row>)
                node->subtree_owned_admission_bytes |= child->subtree_owned_admission_bytes & 1;
            node->size = checked_sum(node->size, child->size);
            node->logical_size = checked_sum(node->logical_size, child->logical_size);
            node->subtree_owned_admission_bytes =
                checked_sum(node->subtree_owned_admission_bytes, child->owned_admission_bytes());
            node->subtree_bytes = checked_sum(node->subtree_bytes, child->subtree_bytes);
            node->subtree_payload_bytes = checked_sum(node->subtree_payload_bytes, child->subtree_payload_bytes);
            node->subtree_owner_bytes = checked_sum(node->subtree_owner_bytes, child->subtree_owner_bytes);
        }
        node->maximum = values.back()->maximum;
        record(node);
        return node;
    }
    [[nodiscard]] Replacement pack_leaves(std::span<const Row> rows) const
    {
        if (rows.empty())
            return {{}, 0};
        if (rows.size() <= leaf_capacity)
            return {{leaf(rows), {}}, 1};
        const auto middle = rows.size() / 2;
        return {{leaf(rows.first(middle)), leaf(rows.subspan(middle))}, 2};
    }
    [[nodiscard]] Replacement pack_branches(std::span<const Ptr> children) const
    {
        if (children.empty())
            return {{}, 0};
        if (children.size() <= branch_capacity)
            return {{branch(children), {}}, 1};
        const auto middle = children.size() / 2;
        return {{branch(children.first(middle)), branch(children.subspan(middle))}, 2};
    }
    std::span<const Ptr> rebalance(std::span<Ptr> children) const
    {
        std::size_t index = 0;
        while (index < children.size() && children.size() > 1)
        {
            const auto& child = *children[index];
            const auto minimum = child.leaf() ? leaf_capacity / 2 : branch_capacity / 2;
            if (child.occupancy() >= minimum)
            {
                ++index;
                continue;
            }
            const auto left = index == 0 ? index : index - 1;
            const auto replacement = redistribute(*children[left], *children[left + 1]);
            children[left] = replacement.pages[0];
            if (replacement.count == 1)
            {
                std::move(children.begin() + left + 2, children.end(), children.begin() + left + 1);
                children = children.first(children.size() - 1);
            }
            else
                children[left + 1] = replacement.pages[1];
            index = left;
        }
        return children;
    }
    [[nodiscard]] Replacement redistribute(const Node& first, const Node& second) const
    {
        retire(first);
        retire(second);
        if (first.leaf())
        {
            std::array<Row, 2 * leaf_capacity> rows;
            auto end = std::copy(first.rows.begin(), first.rows.end(), rows.begin());
            end = std::copy(second.rows.begin(), second.rows.end(), end);
            return pack_leaves(std::span(rows.begin(), end));
        }
        std::array<Ptr, 2 * branch_capacity> children;
        auto end = std::copy(first.children.begin(), first.children.end(), children.begin());
        end = std::copy(second.children.begin(), second.children.end(), end);
        return pack_branches(rebalance(std::span(children.begin(), end)));
    }
    [[nodiscard]] Replacement rewrite(const Node& node, const Row& key, const Row* replacement, bool expected) const
    {
        if (work_)
            ++work_->visited_pages;
        if (node.leaf())
        {
            const auto at = std::lower_bound(node.rows.begin(), node.rows.end(), key, Less{});
            const bool present = at != node.rows.end() && !Less{}(key, *at);
            if (present != expected)
                throw GeometryError(
                    GeometryErrorCode::InvalidInput, "Ordered metadata edit changed its accepted selector.");
            retire(node);
            std::array<Row, leaf_capacity + 1> rows;
            auto end = std::copy(node.rows.begin(), at, rows.begin());
            if (replacement)
                *end++ = *replacement;
            end = std::copy(present ? std::next(at) : at, node.rows.end(), end);
            return pack_leaves(std::span(rows.begin(), end));
        }
        auto at = std::lower_bound(node.children.begin(), node.children.end(), key,
            [](const Ptr& value, const Row& target) { return Less{}(*value->maximum, target); });
        if (at == node.children.end())
            --at;
        const auto edited = rewrite(**at, key, replacement, expected);
        retire(node);
        std::array<Ptr, branch_capacity + 1> children;
        auto end = std::copy(node.children.begin(), at, children.begin());
        end = std::copy_n(edited.pages.begin(), edited.count, end);
        end = std::copy(std::next(at), node.children.end(), end);
        return pack_branches(rebalance(std::span(children.begin(), end)));
    }
    Buffer<Ptr> pack_many_branches(std::span<const Ptr> children) const
    {
        Buffer<Ptr> result(PageBufferAllocator<Ptr, Domain>(admission_.native));
        if (children.empty())
            return result;
        const auto groups = ceiling(children.size(), branch_capacity);
        reserve(result, groups);
        std::size_t offset = 0;
        for (std::size_t group = 0; group < groups; ++group)
        {
            const auto count = group_size(children.size(), groups, group);
            result.push_back(branch(children.subspan(offset, count)));
            offset += count;
        }
        return result;
    }
    using PageBuffer = Buffer<Ptr>;

    template<class Iterator, class Keep>
    PageBuffer patch_leaf(const Ptr& old, Iterator first, Iterator last, const Keep& keep) const
    {
        using Source = std::iter_value_t<Iterator>;
        using Slot = std::conditional_t<stack_safe_default<Source>, Source, std::optional<Source>>;
        // Once a full leaf has a half-leaf successor, it can be sealed.
        // Keep the final tail available so it can share rows with its predecessor.
        std::array<Slot, leaf_capacity + leaf_capacity / 2> pending{};
        std::size_t size = 0;
        PageBuffer result(PageBufferAllocator<Ptr, Domain>(admission_.native));
        const auto read = [](const Slot& slot) -> const Source&
        {
            if constexpr (stack_safe_default<Source>)
                return slot;
            else
                return *slot;
        };
        const auto emit = [&](std::size_t count)
        {
            if (result.size() == result.capacity())
            {
                const auto capacity = result.capacity();
                reserve(result, capacity ? checked_sum(capacity, capacity) : 1);
            }
            result.push_back(leaf(std::span(pending).first(count) | std::views::transform(read)));
            std::move(pending.begin() + count, pending.begin() + size, pending.begin());
            size -= count;
        };
        const auto append = [&](const auto& value)
        {
            pending[size++] = Source(value);
            if (size == pending.size())
                emit(leaf_capacity);
        };
        const auto accepted = old ? std::span<const Row>(old->rows) : std::span<const Row>{};
        auto left = accepted.begin();
        for (; first != last; ++first)
        {
            while (left != accepted.end() && Less{}(*left, *first))
                append(*left++);
            const bool present = left != accepted.end() && !Less{}(*first, *left);
            if (present)
                ++left;
            if (keep(*first))
                append(*first);
            else if (!present)
                throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry batch removes an absent row.");
        }
        while (left != accepted.end())
            append(*left++);
        if (size > leaf_capacity)
            emit(size / 2);
        if (size)
            emit(size);
        if (old)
            retire(*old);
        return result;
    }

    template<class Iterator, class Keep>
    PageBuffer patch_rewrite(const Ptr& old, Iterator first, Iterator last, const Keep& keep, bool root = false) const
    {
        if (work_)
            ++work_->visited_pages;
        if (!old || old->leaf())
            return patch_leaf(old, first, last, keep);

        std::array<PageBuffer, branch_capacity> replacements;
        std::array<bool, branch_capacity> changed{};
        std::size_t count = 0;
        for (std::size_t index = 0; index < old->children.size(); ++index)
        {
            const auto& child = old->children[index];
            auto end = first;
            if (index + 1 == old->children.size())
                end = last;
            else
                while (end != last && !Less{}(*child->maximum, *end))
                    ++end;
            if (first == end)
                ++count;
            else
            {
                changed[index] = true;
                replacements[index] = patch_rewrite(child, first, end, keep);
                count = checked_sum(count, replacements[index].size());
            }
            first = end;
        }

        // Size branch scratch from the actual surviving and split pages.
        // An edit concentrated in one child can contribute thousands of rows
        // but only the resulting page pointers enter the parent.
        std::array<Ptr, branch_capacity> local;
        PageBuffer overflow(PageBufferAllocator<Ptr, Domain>(admission_.native));
        std::span<Ptr> children;
        if (count <= local.size())
            children = std::span(local).first(count);
        else
        {
            reserve(overflow, count);
            overflow.resize(count);
            children = overflow;
        }
        auto out = children.begin();
        for (std::size_t index = 0; index < old->children.size(); ++index)
            if (changed[index])
            {
                auto& pages = replacements[index];
                out = std::move(pages.begin(), pages.end(), out);
                PageBuffer().swap(pages);
            }
            else
                *out++ = old->children[index];

        retire(*old);
        const auto balanced = rebalance(children);
        if (root && balanced.size() == 1)
        {
            // Only the root may lose a level.
            PageBuffer result(PageBufferAllocator<Ptr, Domain>(admission_.native));
            reserve(result, 1);
            result.push_back(std::move(children.front()));
            return result;
        }
        return pack_many_branches(balanced);
    }
    Admission admission_;
    PageWork* work_;
};
} // namespace SimpleFluid::PLIC::details::geometry_demand
